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
  serial), com o LED aceso, exatamente como ``main.py``;
* **resgate** — a função principal de ``resgate.py`` roda dentro deste
  processo, depois que os filhos do percurso morreram, e assume a câmera 0,
  a serial e a trava com o LED apagado.

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
RESCUE_RETURN_RESTART_AFTER_ARDUINO = "restart_after_arduino"
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
    """Define a unica situacao em que uma falha pode rearmar o percurso."""
    if int(returncode) == RESCUE_EXIT_OK:
        return RESCUE_RETURN_COMPLETED
    if int(returncode) == RESCUE_EXIT_ARDUINO_DESCONECTADO:
        return RESCUE_RETURN_RESTART_AFTER_ARDUINO
    return RESCUE_RETURN_STOPPED


def iniciar_visao(debug):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from visao.processamento import vision_loop
    vision_loop(debug)


def iniciar_controle():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from controle.ciclo import control_loop
    control_loop()


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
        self.resgate_ativo = False
        self.argumentos_resgate = None
        self.rescue_returncode = None
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
        }
        for nome, valor in valores.items():
            self._definir_compartilhado(nome, valor)

    # -- ciclo de vida do percurso ---------------------------------------
    def start_line_phase(self):
        """Sobe visão (câmera 1) e controle (serial + LED aceso)."""
        self.shared.terminate.value = False
        self._definir_compartilhado("vision_ready", False)
        self.shared.rescue_requested.value = False
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
        if self.args.debug:
            debug = Process(
                target=iniciar_debug_linha,
                name="shadow-debug-linha",
            )
            debug.start()
            self.children.append(debug)
        print("[missão] percurso ativo: câmera 1 e serial com os filhos")

    def wait_line_ready(self):
        """Espera a câmera e a serial ficarem prontas sem travar para sempre."""
        prazo = time.monotonic() + (
            config.SERIAL_HANDSHAKE_TIMEOUT
            + config.VISION_READY_TIMEOUT
            + 2.0
        )
        while time.monotonic() < prazo:
            if not all(child.is_alive() for child in self.children):
                return False
            texto = str(self.shared.status.value).lower()
            if "pronto" in texto or "seguindo linha" in texto:
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
            if not all(child.is_alive() for child in self.children):
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

    def led_off(self):
        """LED APAGADO enviado por ``ciclo.py`` antes de liberar a serial."""
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

    def assert_led_off(self):
        """``resgate.py`` reafirma LED APAGADO na serial nova."""
        return True

    def open_rescue_camera(self):
        """``resgate.py`` abre a câmera 0 depois da serial e do LED."""
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

    def led_on(self):
        """``ciclo.py`` envia LED ACESO ao assumir a serial."""
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
        if not self._lock_held:
            self.motor_lock.acquire()
            self._lock_held = True
        self._preparar_nova_tentativa()
        print("[missao] recursos limpos; tentando iniciar o percurso")
        time.sleep(config.MISSION_RECOVERY_DELAY_S)
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
        """Não rearma o percurso até a placa ter sumido e reaparecido.

        Depois que o resgate começou, um retorno inesperado não autoriza
        devolver os motores ao segue-linha: a posição, as vítimas na garra e
        a etapa física já não podem ser presumidas. O operador sinaliza que
        reposicionou o robô fazendo o ciclo físico do Arduino.
        """
        viu_desconectado = not self._portas_arduino_presentes()
        print(
            "[missao] resgate interrompido; segue-linha bloqueado. "
            f"{motivo}. Desligue e religue o Arduino para reiniciar.")
        while True:
            time.sleep(config.MISSION_RECOVERY_DELAY_S)
            portas = self._portas_arduino_presentes()
            if not portas:
                viu_desconectado = True
            elif viu_desconectado:
                print("[missao] Arduino reconectado; reiniciando pelo percurso")
                return

    def aguardar_arduino_reconectado(self):
        """Espera a reconexão depois que o próprio resgate detectou a queda."""
        print(
            "[missao] Arduino desconectado no resgate; aguardando "
            "reconexao para reiniciar pelo percurso")
        while not self._portas_arduino_presentes():
            time.sleep(config.MISSION_RECOVERY_DELAY_S)

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
                if rescue_action == RESCUE_RETURN_RESTART_AFTER_ARDUINO:
                    system.aguardar_arduino_reconectado()
                    motivo = "Arduino reconectado depois de cair no resgate"
                    estado_atual = mudar_estado(
                        estado_atual,
                        EstadoMissao.RECONECTANDO,
                        motivo,
                    )
                    tentativas += 1
                    iniciar_percurso_ate_pronto(motivo)
                    continue
                if rescue_action == RESCUE_RETURN_STOPPED:
                    # Um erro comum permanece em RESGATE. Somente o ciclo
                    # físico do Arduino autoriza limpar a tentativa.
                    system.aguardar_ciclo_do_arduino(
                        f"resgate terminou com codigo {returncode}")
                    motivo = (
                        f"resgate terminou com codigo {returncode}; "
                        "Arduino foi reiniciado")
                    estado_atual = mudar_estado(
                        estado_atual,
                        EstadoMissao.RECONECTANDO,
                        motivo,
                    )
                    tentativas += 1
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
                if estado_atual in (
                    EstadoMissao.ENTRADA_RESGATE,
                    EstadoMissao.RESGATE,
                ):
                    # Depois da entrada confirmada nenhum erro comum pode
                    # devolver os motores ao segue-linha.
                    system.aguardar_ciclo_do_arduino(str(err))
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
