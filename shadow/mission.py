#!/usr/bin/env python3
"""Controlador central da missão: percurso → sala de resgate → percurso.

``mission.py`` permanece vivo durante toda a rodada e é a única autoridade
que decide se o robô está no segue-linha, na entrada, no resgate ou se está
recuperando a conexão. ``main.py`` e ``resgate.py`` continuam úteis para testes
isolados, mas não iniciam um ao outro durante a missão completa.

Modelo de propriedade do hardware
---------------------------------
Segue-linha e resgate disputam a mesma serial, o mesmo Arduino e os mesmos
motores; por isso NUNCA coexistem. A missão alterna entre duas configurações
mutuamente exclusivas:

* **percurso** — dois processos filhos (visão da câmera 1 + controle da
  serial), exatamente como ``main.py``;
* **resgate** — a função principal de ``resgate.py`` roda dentro deste
  processo, depois que os filhos do percurso morreram, e assume a câmera 0,
  a serial e a trava.

Entre as duas existe o handoff, cuja ORDEM é o contrato de segurança da
missão. Essa ordem está declarada em ``controle/missao.py``
(``HANDOFF_TO_RESCUE`` / ``HANDOFF_TO_LINE``) e é testada em
``tests/test_missao.py``. Aqui ela é apenas *executada*.

Onde cada passo realmente acontece
----------------------------------
Alguns passos do handoff só podem ocorrer dentro de um processo filho, porque
é ele quem possui o recurso:

* ``stop_motors`` e ``led_off`` acontecem em ``controle/ciclo.py``, no momento
  em que a faixa prata é confirmada — é o último instante em que a serial do
  segue-linha ainda existe;
* ``close_line_camera`` acontece no ``finally`` do processo de visão;
* ``release_serial`` acontece no ``finally`` do processo de controle.

Nesses casos o passo aqui é uma **verificação**: o supervisor confirma que o
filho realmente terminou (e portanto que o recurso foi liberado) e se recusa a
seguir em frente caso contrário. Nunca se abre a câmera seguinte com o
processo anterior ainda vivo.
"""

import argparse
from enum import Enum
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiprocessing import Process, shared_memory  # noqa: E402

import config  # noqa: E402
from controle.missao import (  # noqa: E402
    HANDOFF_TO_LINE,
    HANDOFF_TO_RESCUE,
    HandoffExecutor,
)


CHILD_JOIN_TIMEOUT_S = 6.0
#: Códigos devolvidos pela rotina de resgate ao controlador central.
RESCUE_EXIT_OK = 0
RESCUE_EXIT_ARDUINO_DESCONECTADO = 5
RESCUE_RETURN_COMPLETED = "completed"
RESCUE_RETURN_STOPPED = "stopped"


class EstadoMissao(str, Enum):
    """Estados de alto nível; somente ``mission.py`` pode alterá-los."""

    INICIALIZANDO = "INICIALIZANDO"
    SEGUE_LINHA = "SEGUE_LINHA"
    ENTRADA_RESGATE = "ENTRADA_RESGATE"
    RESGATE = "RESGATE"
    FINALIZANDO_RESGATE = "FINALIZANDO_RESGATE"
    RECONECTANDO = "RECONECTANDO"
    ENCERRADO = "ENCERRADO"


TRANSICOES_PERMITIDAS = {
    EstadoMissao.INICIALIZANDO: {
        EstadoMissao.SEGUE_LINHA,
        EstadoMissao.RECONECTANDO,
        EstadoMissao.ENCERRADO,
    },
    EstadoMissao.SEGUE_LINHA: {
        EstadoMissao.ENTRADA_RESGATE,
        EstadoMissao.RECONECTANDO,
        EstadoMissao.ENCERRADO,
    },
    EstadoMissao.ENTRADA_RESGATE: {
        EstadoMissao.RESGATE,
        EstadoMissao.RECONECTANDO,
        EstadoMissao.ENCERRADO,
    },
    EstadoMissao.RESGATE: {
        EstadoMissao.FINALIZANDO_RESGATE,
        EstadoMissao.RECONECTANDO,
        EstadoMissao.ENCERRADO,
    },
    EstadoMissao.FINALIZANDO_RESGATE: {
        EstadoMissao.SEGUE_LINHA,
        EstadoMissao.RECONECTANDO,
        EstadoMissao.ENCERRADO,
    },
    EstadoMissao.RECONECTANDO: {
        EstadoMissao.SEGUE_LINHA,
        EstadoMissao.RECONECTANDO,
        EstadoMissao.ENCERRADO,
    },
    EstadoMissao.ENCERRADO: set(),
}


def mudar_estado(estado_atual, novo_estado, motivo=""):
    """Valida e registra toda troca de modo da missão."""
    if novo_estado == estado_atual:
        return estado_atual
    if novo_estado not in TRANSICOES_PERMITIDAS[estado_atual]:
        raise RuntimeError(
            f"transicao de missao proibida: {estado_atual.value} -> "
            f"{novo_estado.value}")
    sufixo = f" ({motivo})" if motivo else ""
    print(f"[MISSION] Estado: {novo_estado.value}{sufixo}")
    return novo_estado


def rescue_return_action(returncode):
    """Define quando e seguro devolver o robô ao percurso.

    Somente o término normal da saída preta confirmada devolve o robô ao
    percurso *na mesma posição*. Qualquer outro código encerra a tentativa de
    resgate e faz o supervisor criar uma nova sessão de segue-linha, que fica
    parada até a serial do Arduino estar disponível. Assim, uma placa
    desligada jamais encerra o ``mission.py`` nem mantém o robô preso no
    resgate.
    """
    if int(returncode) == RESCUE_EXIT_OK:
        return RESCUE_RETURN_COMPLETED
    return RESCUE_RETURN_STOPPED


def iniciar_visao(debug):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from visao.processamento import vision_loop
    # Um erro ao reabrir a camera depois do resgate ocorria somente no filho e
    # deixava o controle esperando ``vision_ready`` sem diagnostico. Publique
    # o erro para o supervisor encerrar esta tentativa e iniciar outra.
    try:
        print("[visao] iniciando camera de linha")
        vision_loop(debug)
    except Exception as err:  # noqa: BLE001
        from shared.dados_compartilhados import status
        status.value = f"Falha camera de linha: {err}"
        print(f"[visao] FALHA ao iniciar camera de linha: {err}")
        raise


def iniciar_controle():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from controle.ciclo import control_loop
    control_loop()


def iniciar_vigia_yolo(camera_index):
    """Procura vítimas pela câmera frontal, sem jamais comandar motores."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from shared.dados_compartilhados import (
        rescue_yolo_confirmed,
        status,
        terminate,
    )
    from visao.vigia_vitimas_missao import vigiar_vitimas
    vigiar_vitimas(camera_index, terminate, rescue_yolo_confirmed, status)


def _tecla_fecha_debug(tecla):
    return (tecla & 0xFF) in (ord("q"), 27)


def iniciar_debug_linha():
    """Mostra o frame compartilhado sem inicializar o Qt no supervisor."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import cv2
    import numpy as np
    from shared.dados_compartilhados import terminate

    shm = shared_memory.SharedMemory(name=config.DEBUG_SHM_NAME)
    frame = np.ndarray(
        (config.camera_y, config.camera_x, 3),
        dtype=np.uint8,
        buffer=shm.buf,
    )
    try:
        while not terminate.value:
            cv2.imshow(
                "Shadow2026 - camera de linha",
                frame.copy(),
            )
            if _tecla_fecha_debug(cv2.waitKey(30)):
                terminate.value = True
                break
    finally:
        cv2.destroyAllWindows()
        shm.close()


class MissionSystem:
    """Liga os passos do handoff às operações reais de processo e trava."""

    def __init__(self, shared, motor_lock, args):
        self.shared = shared
        self.motor_lock = motor_lock
        self.args = args
        self.children = []
        # Visão de linha e controle/serial são a fase de percurso. O vigia
        # YOLO e a janela debug são auxiliares: não podem bloquear a volta à
        # linha se demorarem a reabrir depois de um resgate.
        self.line_children = []
        self.resgate_ativo = False
        self.argumentos_resgate = None
        self.rescue_returncode = None
        # O vigia frontal so pode autorizar uma entrada por execucao da missao.
        # Depois do primeiro handoff, as retomadas ficam em segue-linha puro.
        self.yolo_rescue_consumed = False
        self._lock_held = True

    def _definir_compartilhado(self, nome, valor):
        campo = getattr(self.shared, nome, None)
        if campo is not None:
            campo.value = valor

    def _limpar_deteccoes_percurso(self):
        """Remove frames, votos e decisões que não valem na nova fase."""
        valores = {
            "vision_ready": False,
            "line_detected": False,
            "line_ahead": False,
            "line_angle": 0,
            "line_angle_y": -1,
            "line_size": 0.0,
            "last_bottom_point": config.camera_x / 2,
            "last_bottom_point_y": 0,
            "line_status": "line_detected",
            "turn_dir": "straight",
            "green_turn_target": 0,
            "preferencia_linha_esquerda": False,
            "red_detected": False,
            "red_candidate": False,
            "green_candidate": False,
            "gap_angle": 0.0,
            "gap_center_x": -180.0,
            "gap_center_y": -1.0,
            "gap_end_width": -1.0,
            "black_average": 0.0,
            "rescue_yolo_confirmed": False,
        }
        for nome, valor in valores.items():
            self._definir_compartilhado(nome, valor)

    def _children_essenciais(self):
        """Retorna apenas visão de linha e controle que fazem o robô andar."""
        # O fallback preserva os testes e sistemas antigos que injetam apenas
        # ``children`` sem separar os processos auxiliares.
        return self.line_children or self.children

    # -- ciclo de vida do percurso ---------------------------------------
    def start_line_phase(self):
        """Sobe visão (câmera 1) e controle (serial)."""
        self.shared.terminate.value = False
        self._definir_compartilhado("vision_ready", False)
        self.shared.rescue_requested.value = False
        self._definir_compartilhado("rescue_yolo_confirmed", False)
        self.shared.red_finished.value = False
        self.shared.mission_mode.value = True
        self._definir_compartilhado("status", "Inicializando percurso")

        vision = Process(
            target=iniciar_visao, args=(self.args.debug,), name="shadow-visao")
        vision.start()
        time.sleep(.5)
        control = Process(target=iniciar_controle, name="shadow-controle")
        control.start()
        self.children = [vision, control]
        self.line_children = [vision, control]
        if self.args.debug:
            debug = Process(
                target=iniciar_debug_linha,
                name="shadow-debug-linha",
            )
            debug.start()
            self.children.append(debug)
        print("[missão] percurso ativo: câmera 1 e serial com os filhos")

    def _start_vigia_yolo(self):
        """Abre a câmera frontal só após o primeiro frame da linha.

        O libcamera ainda pode estar alocando buffers nos primeiros frames
        após o resgate. Abrir as duas câmeras nessa janela já deixou a visão
        de linha sem frame e o controle venceu seu timeout de 10 s.
        """
        import config_resgate
        if not (
            config_resgate.MISSION_YOLO_RESCUE_ENABLED
            and not self.yolo_rescue_consumed
            and hasattr(self.args, "rescue_camera_index")
        ):
            return
        if any(child.name == "shadow-vigia-yolo" for child in self.children):
            return
        vigilante = Process(
            target=iniciar_vigia_yolo,
            args=(self.args.rescue_camera_index,),
            name="shadow-vigia-yolo",
        )
        vigilante.start()
        self.children.append(vigilante)
        print("[missão] segue-linha pronto; vigia YOLO da câmera frontal ativo")

    def consume_yolo_rescue_request(self):
        """Desabilita o vigia para as retomadas depois do handoff YOLO."""
        if self.yolo_rescue_consumed:
            return False
        confirmado = getattr(self.shared, "rescue_yolo_confirmed", None)
        if confirmado is None or not confirmado.value:
            return False
        self.yolo_rescue_consumed = True
        confirmado.value = False
        print("[missão] primeira vítima YOLO consumida; retomadas serão só segue-linha")
        return True

    def wait_line_ready(self):
        """Espera a câmera e a serial ficarem prontas sem travar para sempre."""
        prazo = time.monotonic() + (
            config.SERIAL_HANDSHAKE_TIMEOUT
            + config.VISION_READY_TIMEOUT
            + 2.0
        )
        while time.monotonic() < prazo:
            # A confirmação YOLO pode chegar no primeiro frame. Nesse caso o
            # controle para os motores e encerra de propósito para entregar a
            # serial ao resgate; isso não é falha de inicialização.
            if self.shared.rescue_requested.value:
                return True
            if not all(
                child.is_alive() for child in self._children_essenciais()
            ):
                return False
            texto = str(self.shared.status.value).lower()
            if "falha camera de linha" in texto:
                return False
            if "pronto" in texto or "seguindo linha" in texto:
                self._start_vigia_yolo()
                return True
            time.sleep(.05)
        raise RuntimeError(
            "camera/Arduino nao ficaram prontos dentro do prazo de inicio")

    def wait_line_phase(self):
        """Bloqueia até o handoff, o fim da prova ou a morte de um filho.

        Devolve ``"rescue"``, ``"finished"`` ou ``"child_died"``.
        """
        last_status = ""
        while True:
            if self.shared.rescue_requested.value:
                return "rescue"
            if self.shared.red_finished.value:
                return "finished"
            if self.shared.terminate.value:
                return "quit"
            if not all(
                child.is_alive() for child in self._children_essenciais()
            ):
                # Um filho caiu: pode ter sido exceção, Ctrl-C ou o próprio
                # fim normal do controle. Quem decide é o flag já lido acima.
                return "child_died"
            if self.shared.status.value != last_status:
                last_status = self.shared.status.value
                print(f"[status] {last_status}")
            time.sleep(.05)

    # -- passos do handoff para o resgate --------------------------------
    def stop_motors(self):
        """PARAR já foi enviado por quem tinha a serial; aqui é verificação.

        No caminho de ida, ``controle/ciclo.py`` envia PARAR ao terminar a
        entrada. No caminho de volta, ``resgate.py`` envia PARAR no
        ``finally``. O supervisor nunca escreve na serial: fazer isso criaria
        um segundo dono, que é exatamente o que a missão precisa evitar.
        """
        return True

    def terminate_line_children(self):
        self.shared.terminate.value = True

    def join_line_children(self):
        deadline = time.monotonic() + CHILD_JOIN_TIMEOUT_S
        for child in self.children:
            child.join(timeout=max(0.0, deadline - time.monotonic()))
        for child in self.children:
            if child.is_alive():
                print(f"[missão] forçando término de {child.name}")
                child.terminate()
                child.join(timeout=1.0)

    def assert_line_children_dead(self):
        vivos = [child.name for child in self.children if child.is_alive()]
        if vivos:
            raise RuntimeError(
                f"processos do segue-linha ainda vivos: {vivos}; "
                "o resgate não pode começar")
        self.children = []
        self.line_children = []

    def close_line_camera(self):
        """A câmera 1 é fechada no ``finally`` do processo de visão."""
        self.assert_line_children_dead()

    def release_serial(self):
        """A serial é fechada no ``finally`` de quem a abriu."""
        if self.children:
            self.assert_line_children_dead()

    def release_motor_lock(self):
        if self._lock_held:
            self.motor_lock.release()
            self._lock_held = False

    def acquire_rescue_motor_lock(self):
        """A rotina de resgate adquire a própria trava.

        O supervisor apenas garante que a dele já foi liberada — caso
        contrário o resgate falharia ao iniciar, com a mensagem correta.
        """
        if self._lock_held:
            raise RuntimeError(
                "a trava dos motores ainda pertence ao supervisor")

    def open_rescue_serial(self):
        """``resgate.py`` abre a própria serial ao iniciar."""
        return True

    def open_rescue_camera(self):
        """``resgate.py`` abre a câmera 0 depois da serial."""
        return True

    def start_rescue(self):
        """Prepara a chamada direta; nenhuma outra aplicação é iniciada."""
        self.argumentos_resgate = argparse.Namespace(
            camera_index=self.args.rescue_camera_index,
            target="any",
            drive=True,
            debug=self.args.debug,
            video=None,
            sem_vitimas=False,
            sem_marcadores=False,
            gerenciado_pela_missao=True,
        )
        self.resgate_ativo = True
        self.rescue_returncode = None
        print("[missão] iniciando a rotina de resgate dentro de mission.py")

    def wait_rescue(self):
        if not self.resgate_ativo or self.argumentos_resgate is None:
            raise RuntimeError("o resgate não foi iniciado")
        from resgate import executar_resgate

        try:
            self.rescue_returncode = executar_resgate(
                self.argumentos_resgate)
        finally:
            self.resgate_ativo = False
            self.argumentos_resgate = None
        return self.rescue_returncode

    # -- passos do handoff de volta ao percurso ---------------------------
    def close_rescue_camera(self):
        """A câmera 0 é fechada no ``finally`` do ``resgate.py``."""
        if self.resgate_ativo:
            raise RuntimeError(
                "a rotina de resgate ainda esta ativa; "
                "a câmera de linha não pode abrir")

    def acquire_line_motor_lock(self):
        if not self._lock_held:
            self.motor_lock.acquire()
            self._lock_held = True

    def open_line_camera(self):
        """A câmera 1 reabre quando o processo de visão sobe."""
        return True

    def open_line_serial(self):
        """A serial reabre quando o processo de controle sobe."""
        return True

    def _preparar_retomada_linha(self):
        """Apaga decisoes antigas antes do primeiro frame pos-resgate.

        A terceira linha ja foi confirmada pela rotina de resgate, ainda com
        a camera inferior aberta. Aqui apenas zeramos memorias antigas antes
        de iniciar o segue-linha normal.
        """
        self._limpar_deteccoes_percurso()
        self.shared.line_crop.value = config.LINE_CROP_NORMAL
        self.shared.min_line_size.value = config.MIN_LINE_SIZE_DEFAULT

    def reacquire_line(self):
        self._preparar_retomada_linha()
        self.start_line_phase()

    def _preparar_nova_tentativa(self):
        """Volta todos os dados ao inicio da missao, antes da faixa prata."""
        self._limpar_deteccoes_percurso()
        self.shared.line_crop.value = config.LINE_CROP_INITIAL
        self.shared.min_line_size.value = config.MIN_LINE_SIZE_DEFAULT
        self.shared.entry_armed.value = True
        self.shared.entry_silver_detected.value = False
        self.shared.entry_silver_confirmed.value = False
        self.shared.entry_silver_votes.value = 0
        self.shared.entry_silver_reason.value = ""
        self.shared.entry_silver_state.value = 0
        self.shared.rescue_requested.value = False
        self._definir_compartilhado("rescue_yolo_confirmed", False)
        self.shared.red_finished.value = False
        self.shared.status.value = "Reiniciando missao - aguardando Arduino"

    def _encerrar_resgate_para_recuperacao(self):
        # A chamada de resgate e sincrona. Quando a execucao chega aqui, o
        # ``finally`` dela ja fechou camera, serial, workers e trava.
        self.resgate_ativo = False
        self.argumentos_resgate = None

    def reiniciar_missao_do_percurso(self, motivo):
        """Fecha a tentativa atual e sobe outra, sempre antes da prata."""
        print(f"[missao] recuperacao: {motivo}")
        self.shared.terminate.value = True
        self._encerrar_resgate_para_recuperacao()
        if self.children:
            self.join_line_children()
        self.children = []
        self.line_children = []
        if not self._lock_held:
            self.motor_lock.acquire()
            self._lock_held = True
        self._preparar_nova_tentativa()
        print("[missao] recursos limpos; tentando iniciar o percurso")
        espera_camera = max(
            config.MISSION_RECOVERY_DELAY_S,
            config.MISSION_CAMERA_HANDOFF_SETTLE_S,
        )
        print(f"[missao] aguardando {espera_camera:.1f}s para liberar camera")
        time.sleep(espera_camera)
        self.start_line_phase()

    @staticmethod
    def _portas_arduino_presentes():
        """Retorna apenas portas que podem pertencer ao Arduino do robô.

        Esta consulta não abre a serial nem envia comandos. Ela serve para o
        supervisor distinguir uma falha do resgate de um ciclo físico de
        desligar/ligar a placa.
        """
        try:
            from serial.tools import list_ports
        except ImportError:
            return set()
        prefixos = tuple(config.SERIAL_PORT_PREFIXES)
        return {
            porta.device
            for porta in list_ports.comports()
            if porta.device.startswith(prefixos)
        }

    def aguardar_ciclo_do_arduino(self, motivo):
        """Rearma o percurso somente após um desligamento físico sustentado.

        Uma reconexão USB breve pode ser cabo, ruído ou reset por queda de
        tensão. Ela não autoriza devolver os motores ao segue-linha. Para
        sinalizar que corrigiu e reposicionou o robô, o operador desliga a
        alimentação do Arduino pelo intervalo mínimo configurado e o religa.
        """
        print(
            "[missao] resgate interrompido; segue-linha bloqueado. "
            f"{motivo}. Desligue o Arduino por pelo menos "
            f"{config.MISSION_ARDUINO_DESLIGAMENTO_MINIMO_S:.0f} s e "
            "religue-o para reiniciar.")
        desligado_desde = None
        while True:
            time.sleep(config.MISSION_RECOVERY_DELAY_S)
            agora = time.monotonic()
            portas = self._portas_arduino_presentes()
            if not portas:
                if desligado_desde is None:
                    desligado_desde = agora
                    print("[missao] Arduino desligado; aguardando religacao")
                continue
            if desligado_desde is None:
                continue
            duracao_desligado = agora - desligado_desde
            if duracao_desligado < config.MISSION_ARDUINO_DESLIGAMENTO_MINIMO_S:
                print(
                    "[missao] reconexao curta do Arduino "
                    f"({duracao_desligado:.1f} s); segue-linha continua "
                    "bloqueado. Faca um desligamento completo.")
                desligado_desde = None
                continue
            print("[missao] Arduino religado; reiniciando pelo percurso")
            return

    # -- encerramento ----------------------------------------------------
    def shutdown(self):
        self.shared.terminate.value = True
        self.resgate_ativo = False
        self.argumentos_resgate = None
        if self.children:
            self.join_line_children()
        if self._lock_held:
            self.motor_lock.release()
            self._lock_held = False


def _criar_memoria_debug():
    try:
        return shared_memory.SharedMemory(
            name=config.DEBUG_SHM_NAME, create=True,
            size=config.DEBUG_SHM_SIZE)
    except FileExistsError:
        stale = shared_memory.SharedMemory(name=config.DEBUG_SHM_NAME)
        stale.close()
        stale.unlink()
        return shared_memory.SharedMemory(
            name=config.DEBUG_SHM_NAME, create=True,
            size=config.DEBUG_SHM_SIZE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Missão completa do Shadow2026: percurso, resgate e volta")
    parser.add_argument(
        "--debug", action="store_true",
        help="repassa --debug para as duas fases")
    parser.add_argument(
        "--rescue-camera-index", type=int, default=None,
        help="índice da câmera de resgate (padrão: config_resgate)")
    return parser.parse_args()


def main():
    """Executa tentativas de missao ate Ctrl-C ou desligamento da Raspberry."""
    args = parse_args()
    if args.rescue_camera_index is None:
        import config_resgate
        args.rescue_camera_index = config_resgate.RESCUE_CAMERA_INDEX

    from controle.trava_motores import MotorLockError, MotorOwnerLock
    motor_lock = MotorOwnerLock("missao")
    while True:
        try:
            motor_lock.acquire()
            break
        except MotorLockError as err:
            # Uma trava remanescente ou outro processo ainda em desligamento
            # nao pode encerrar o supervisor. Esperar e tentar de novo tambem
            # preserva o dono atual: nunca removemos a trava de outro processo.
            print(f"[missao] aguardando a trava dos motores: {err}")
            time.sleep(config.MISSION_RECOVERY_DELAY_S)

    import shared.dados_compartilhados as shared

    shm = _criar_memoria_debug() if args.debug else None
    system = MissionSystem(shared, motor_lock, args)
    codigo = 0
    tentativas = 1
    estado_atual = EstadoMissao.INICIALIZANDO
    primeira_partida = True

    def iniciar_percurso_ate_pronto(motivo):
        """Tenta recuperar câmera/serial sem deixar ``mission.py`` morrer."""
        nonlocal estado_atual, primeira_partida, tentativas
        while True:
            try:
                print(f"[missao] iniciando tentativa {tentativas}")
                if primeira_partida:
                    primeira_partida = False
                    system.start_line_phase()
                else:
                    system.reiniciar_missao_do_percurso(motivo)
                if not system.wait_line_ready():
                    raise RuntimeError(
                        "um processo terminou durante a inicializacao")
                estado_atual = mudar_estado(
                    estado_atual,
                    EstadoMissao.SEGUE_LINHA,
                    "camera e Arduino prontos",
                )
                return
            except KeyboardInterrupt:
                raise
            except Exception as erro_recuperacao:  # noqa: BLE001
                print(
                    "[missao] ainda aguardando recursos para reiniciar: "
                    f"{erro_recuperacao}")
                if estado_atual != EstadoMissao.RECONECTANDO:
                    estado_atual = mudar_estado(
                        estado_atual,
                        EstadoMissao.RECONECTANDO,
                        str(erro_recuperacao),
                    )
                tentativas += 1
                time.sleep(config.MISSION_RECOVERY_DELAY_S)

    try:
        iniciar_percurso_ate_pronto("inicializacao")
        while True:
            try:
                resultado = system.wait_line_phase()

                if resultado == "finished":
                    # A faixa vermelha encerra somente a *tentativa* atual.
                    # O supervisor deve continuar vivo para iniciar o proximo
                    # percurso, tal como faz depois de um quit do debug.
                    motivo = "faixa vermelha final alcancada"
                    print(f"[missao] {motivo}; reiniciando automaticamente")
                    tentativas += 1
                    estado_atual = mudar_estado(
                        estado_atual,
                        EstadoMissao.RECONECTANDO,
                        motivo,
                    )
                    iniciar_percurso_ate_pronto(motivo)
                    continue

                if resultado == "quit":
                    # No modo debug, q/Esc fecha a janela e sinaliza
                    # ``terminate``. Isso nao pode encerrar a missao da
                    # Raspberry: a tentativa e simplesmente refeita.
                    motivo = "sinal de parada da fase de percurso"
                    print(f"[missao] {motivo}; reiniciando")
                    tentativas += 1
                    estado_atual = mudar_estado(
                        estado_atual,
                        EstadoMissao.RECONECTANDO,
                        motivo,
                    )
                    iniciar_percurso_ate_pronto(motivo)
                    continue

                if resultado == "child_died":
                    if not shared.rescue_requested.value:
                        raise RuntimeError(
                            "um processo do percurso terminou; "
                            "reiniciando antes da faixa prata")

                # A primeira confirmacao do vigia e consumida nesta execucao:
                # mesmo que o resgate falhe, a volta nao reabre o YOLO.
                system.consume_yolo_rescue_request()
                estado_atual = mudar_estado(
                    estado_atual,
                    EstadoMissao.ENTRADA_RESGATE,
                    "entrada do resgate confirmada",
                )
                print("[missao] faixa PRATA confirmada; executando o handoff")
                HandoffExecutor(system, HANDOFF_TO_RESCUE).run()
                estado_atual = mudar_estado(
                    estado_atual,
                    EstadoMissao.RESGATE,
                    "segue-linha encerrado e motores entregues ao resgate",
                )

                returncode = system.wait_rescue()
                rescue_action = rescue_return_action(returncode)
                if rescue_action == RESCUE_RETURN_STOPPED:
                    motivo = (
                        f"resgate terminou com codigo {returncode}; "
                        "voltando ao segue-linha e aguardando Arduino")
                    print(
                        "[missao] resgate interrompido; reiniciando o "
                        "segue-linha. Os motores ficarao parados ate o "
                        "Arduino reconectar.")
                    estado_atual = mudar_estado(
                        estado_atual,
                        EstadoMissao.RECONECTANDO,
                        motivo,
                    )
                    tentativas += 1
                    if returncode == RESCUE_EXIT_ARDUINO_DESCONECTADO:
                        # O resgate so devolve este codigo depois de observar
                        # a queda real da serial e liberar seus recursos.
                        # Nao espere uma SEGUNDA observacao da porta USB: se
                        # o Arduino for desligado/religado rapido, a porta ja
                        # pode ter voltado e essa espera ficaria infinita.
                        # A sessao nova faz o handshake e repete ate o Uno
                        # responder, sempre com os motores parados no boot.
                        print(
                            "[missao] Arduino caiu no resgate; "
                            "reiniciando imediatamente pelo segue-linha")
                    iniciar_percurso_ate_pronto(motivo)
                    continue

                estado_atual = mudar_estado(
                    estado_atual,
                    EstadoMissao.FINALIZANDO_RESGATE,
                    "deposito vermelho e saida concluidos",
                )
                print("[missao] resgate concluido; voltando ao percurso")

                HandoffExecutor(system, HANDOFF_TO_LINE).run()
                if not system.wait_line_ready():
                    raise RuntimeError(
                        "segue-linha terminou durante a retomada")
                estado_atual = mudar_estado(
                    estado_atual,
                    EstadoMissao.SEGUE_LINHA,
                    "deposito concluido e linha retomada",
                )

            except KeyboardInterrupt:
                raise
            except Exception as err:               # noqa: BLE001
                print(f"[missao] tentativa interrompida: {err}")
                tentativas += 1
                estado_atual = mudar_estado(
                    estado_atual,
                    EstadoMissao.RECONECTANDO,
                    str(err),
                )
                iniciar_percurso_ate_pronto(str(err))

    except KeyboardInterrupt:
        print("\n[missao] Ctrl-C - encerrando...")
        codigo = 130
    finally:
        system.shutdown()
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        if estado_atual != EstadoMissao.ENCERRADO:
            estado_atual = mudar_estado(
                estado_atual, EstadoMissao.ENCERRADO)
        print(f"[missao] estado final: {estado_atual.value}")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
