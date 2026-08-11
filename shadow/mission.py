#!/usr/bin/env python3
"""Supervisor da missão completa: percurso → sala de resgate → percurso.

Este programa NÃO substitui ``main.py`` nem ``resgate.py``. Ele os coordena.
Os dois continuam funcionando isolados, exatamente como antes, e continuam
sendo a forma recomendada de depurar cada metade separadamente.

Modelo de propriedade do hardware
---------------------------------
Segue-linha e resgate disputam a mesma serial, o mesmo Arduino e os mesmos
motores; por isso NUNCA coexistem. A missão alterna entre duas configurações
mutuamente exclusivas:

* **percurso** — dois processos filhos (visão da câmera 1 + controle da
  serial), com o LED aceso, exatamente como ``main.py``;
* **resgate** — um subprocesso ``resgate.py --drive`` que é dono da câmera 0,
  da serial e da trava, com o LED apagado.

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
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiprocessing import Process, shared_memory  # noqa: E402

import config  # noqa: E402
from controle.missao import (  # noqa: E402
    HANDOFF_TO_LINE,
    HANDOFF_TO_RESCUE,
    HandoffError,
    HandoffExecutor,
    MissionCoordinator,
    MissionState,
    VALID_POLICIES,
    POLICY_NEAREST_VALID,
)


SHADOW_ROOT = Path(__file__).resolve().parent
CHILD_JOIN_TIMEOUT_S = 6.0
#: Códigos de saída do subprocesso de resgate, lidos pelo supervisor.
RESCUE_EXIT_OK = 0
RESCUE_EXIT_INCOMPLETE = 3


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
        self.rescue_process = None
        self.rescue_returncode = None
        self._lock_held = True

    # -- ciclo de vida do percurso ---------------------------------------
    def start_line_phase(self):
        """Sobe visão (câmera 1) e controle (serial + LED aceso)."""
        self.shared.terminate.value = False
        self.shared.rescue_requested.value = False
        self.shared.red_finished.value = False
        self.shared.mission_mode.value = True

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
        """O subprocesso de resgate adquire a própria trava.

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
        comando = [
            sys.executable,
            str(SHADOW_ROOT / "resgate.py"),
            "--drive",
            "--camera-index", str(self.args.rescue_camera_index),
            "--policy", self.args.policy,
            "--gerenciado-pela-missao",
        ]
        if self.args.debug:
            comando.append("--debug")
        print(f"[missão] iniciando o resgate: {' '.join(comando)}")
        self.rescue_process = subprocess.Popen(comando, cwd=str(SHADOW_ROOT))

    def wait_rescue(self):
        if self.rescue_process is None:
            raise RuntimeError("o resgate não foi iniciado")
        self.rescue_returncode = self.rescue_process.wait()
        self.rescue_process = None
        return self.rescue_returncode

    # -- passos do handoff de volta ao percurso ---------------------------
    def close_rescue_camera(self):
        """A câmera 0 é fechada no ``finally`` do ``resgate.py``."""
        if self.rescue_process is not None:
            raise RuntimeError(
                "o processo de resgate ainda está vivo; "
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

        A leitura dedicada entrega qualquer ramificacao visivel ao segue-linha
        e so pede o pivo dianteiro quando nao encontra ramo. Zerar qualquer
        memoria verde/lateral impede uma decisao anterior de contaminar essa
        retomada.
        """
        self.shared.vision_ready.value = False
        self.shared.line_detected.value = False
        self.shared.line_ahead.value = False
        self.shared.line_angle.value = 0
        self.shared.line_angle_y.value = -1
        self.shared.line_size.value = 0.0
        self.shared.last_bottom_point.value = config.camera_x / 2
        self.shared.last_bottom_point_y.value = 0
        self.shared.line_status.value = "line_detected"
        self.shared.turn_dir.value = "straight"
        self.shared.green_turn_target.value = 0
        self.shared.preferencia_linha_esquerda.value = False
        self.shared.line_crop.value = config.LINE_CROP_NORMAL
        self.shared.min_line_size.value = config.MIN_LINE_SIZE_DEFAULT
        self.shared.exit_line_search_pending.value = True

    def reacquire_line(self):
        self._preparar_retomada_linha()
        self.start_line_phase()

    def _preparar_nova_tentativa(self):
        """Volta todos os dados ao inicio da missao, antes da faixa prata."""
        self.shared.vision_ready.value = False
        self.shared.line_detected.value = False
        self.shared.line_ahead.value = False
        self.shared.line_angle.value = 0
        self.shared.line_angle_y.value = -1
        self.shared.line_size.value = 0.0
        self.shared.last_bottom_point.value = config.camera_x / 2
        self.shared.last_bottom_point_y.value = 0
        self.shared.line_status.value = "line_detected"
        self.shared.turn_dir.value = "straight"
        self.shared.green_turn_target.value = 0
        self.shared.preferencia_linha_esquerda.value = False
        self.shared.line_crop.value = config.LINE_CROP_INITIAL
        self.shared.min_line_size.value = config.MIN_LINE_SIZE_DEFAULT
        self.shared.entry_armed.value = True
        self.shared.entry_silver_detected.value = False
        self.shared.entry_silver_confirmed.value = False
        self.shared.entry_silver_votes.value = 0
        self.shared.entry_silver_reason.value = ""
        self.shared.rescue_requested.value = False
        self.shared.red_finished.value = False
        self.shared.exit_line_search_pending.value = False
        self.shared.status.value = "Reiniciando missao - aguardando Arduino"

    def _encerrar_resgate_para_recuperacao(self):
        if self.rescue_process is None:
            return
        self.rescue_process.terminate()
        try:
            self.rescue_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.rescue_process.kill()
            self.rescue_process.wait(timeout=1)
        self.rescue_process = None

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
        print(
            "[missao] reposicione o robo antes da faixa prata e religue o "
            "Arduino; o percurso sera iniciado automaticamente")
        time.sleep(config.MISSION_RECOVERY_DELAY_S)
        self.start_line_phase()

    def pausar_apos_final(self):
        """Mantem o supervisor vivo apos a faixa vermelha, sem movimento."""
        self.shared.terminate.value = True
        if self.children:
            self.join_line_children()
        self.children = []
        self.shared.mission_mode.value = False
        self.shared.status.value = "Missao concluida - Ctrl-C para encerrar"

    # -- encerramento ----------------------------------------------------
    def shutdown(self):
        self.shared.terminate.value = True
        if self.rescue_process is not None:
            self.rescue_process.terminate()
            try:
                self.rescue_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rescue_process.kill()
            self.rescue_process = None
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
        "--policy", choices=VALID_POLICIES, default=POLICY_NEAREST_VALID,
        help=(
            "ordem de resgate das vítimas; 'silver_first' prioriza as duas "
            "vivas (vantagem no mundial), 'nearest_valid' é o padrão OBR"))
    parser.add_argument(
        "--rescue-camera-index", type=int, default=None,
        help="índice da câmera de resgate (padrão: config_resgate)")
    return parser.parse_args()


def _main_antigo():
    args = parse_args()
    if args.rescue_camera_index is None:
        import config_resgate
        args.rescue_camera_index = config_resgate.RESCUE_CAMERA_INDEX

    from controle.trava_motores import MotorLockError, MotorOwnerLock
    motor_lock = MotorOwnerLock("missao")
    try:
        motor_lock.acquire()
    except MotorLockError as err:
        print(f"[missão] ERRO: {err}")
        return 1

    import shared.dados_compartilhados as shared

    shm = _criar_memoria_debug() if args.debug else None
    coordinator = MissionCoordinator(policy=args.policy)
    system = MissionSystem(shared, motor_lock, args)
    codigo = 0

    try:
        system.start_line_phase()
        while True:
            resultado = system.wait_line_phase()

            if resultado == "quit":
                print("[missão] debug encerrado pelo usuário.")
                coordinator.abort("janela de debug encerrada")
                break

            if resultado == "finished":
                coordinator.on_red_finish()
                print("[missão] faixa vermelha final alcançada.")
                break

            if resultado == "child_died":
                if shared.rescue_requested.value:
                    resultado = "rescue"
                else:
                    print(
                        "[missão] um processo do percurso terminou sem pedir "
                        "o resgate; encerrando por segurança.")
                    coordinator.abort("processo do percurso encerrou")
                    codigo = 2
                    break

            # Handoff percurso → resgate, na ordem testada.
            coordinator.on_entry_candidate()
            coordinator.on_entry_confirmed()
            coordinator.on_zone_entered()
            print("[missão] faixa PRATA confirmada; executando o handoff")
            HandoffExecutor(system, HANDOFF_TO_RESCUE).run()
            coordinator.on_rescue_started()

            returncode = system.wait_rescue()
            if returncode == RESCUE_EXIT_OK:
                print("[missão] resgate concluído; voltando ao percurso")
            else:
                print(
                    f"[missão] o resgate terminou com código {returncode}; "
                    "faixa preta não confirmada; permanecendo parado")
                coordinator.abort(
                    f"saída do resgate não confirmada: código {returncode}")
                codigo = returncode or RESCUE_EXIT_INCOMPLETE
                break

            # Handoff resgate → percurso, na ordem testada.
            HandoffExecutor(system, HANDOFF_TO_LINE).run()
            coordinator.state = MissionState.FOLLOW_LINE

    except HandoffError as err:
        print(f"[missão] ERRO no handoff: {err}")
        coordinator.abort(str(err))
        codigo = 2
    except KeyboardInterrupt:
        print("\n[missão] Ctrl-C — encerrando…")
        coordinator.abort("Ctrl-C")
        codigo = 130
    except Exception as err:                      # noqa: BLE001
        print(f"[missão] ERRO inesperado: {err}")
        coordinator.abort(str(err))
        codigo = 2
    finally:
        system.shutdown()
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        inventory = coordinator.inventory
        print(
            f"[missão] estado final: {coordinator.state} | "
            f"prata {inventory.silver_deposited}/2, "
            f"preta {inventory.black_deposited}/1")
    return codigo


def main():
    """Executa tentativas de missao ate Ctrl-C ou desligamento da Raspberry."""
    args = parse_args()
    if args.rescue_camera_index is None:
        import config_resgate
        args.rescue_camera_index = config_resgate.RESCUE_CAMERA_INDEX

    from controle.trava_motores import MotorLockError, MotorOwnerLock
    motor_lock = MotorOwnerLock("missao")
    try:
        motor_lock.acquire()
    except MotorLockError as err:
        print(f"[missao] ERRO: {err}")
        return 1

    import shared.dados_compartilhados as shared

    shm = _criar_memoria_debug() if args.debug else None
    coordinator = MissionCoordinator(policy=args.policy)
    system = MissionSystem(shared, motor_lock, args)
    codigo = 0
    tentativas = 1

    try:
        print(f"[missao] iniciando tentativa {tentativas}")
        try:
            system.start_line_phase()
        except Exception as erro_inicio:           # noqa: BLE001
            print(f"[missao] nao foi possivel iniciar: {erro_inicio}")
            coordinator.abort(str(erro_inicio))
            tentativas += 1
            while True:
                try:
                    print(f"[missao] iniciando tentativa {tentativas}")
                    system.reiniciar_missao_do_percurso(str(erro_inicio))
                    coordinator = MissionCoordinator(policy=args.policy)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as erro_recuperacao:  # noqa: BLE001
                    print(
                        "[missao] ainda aguardando recursos para "
                        f"reiniciar: {erro_recuperacao}")
                    time.sleep(config.MISSION_RECOVERY_DELAY_S)
        while True:
            try:
                resultado = system.wait_line_phase()

                if resultado == "quit":
                    print("[missao] debug encerrado pelo usuario.")
                    coordinator.abort("janela de debug encerrada")
                    break

                if resultado == "finished":
                    coordinator.on_red_finish()
                    system.pausar_apos_final()
                    print(
                        "[missao] faixa vermelha final alcancada; programa "
                        "permanece ativo e parado. Use Ctrl-C para encerrar")
                    while True:
                        time.sleep(1.0)

                if resultado == "child_died":
                    if not shared.rescue_requested.value:
                        raise RuntimeError(
                            "um processo do percurso terminou; "
                            "reiniciando antes da faixa prata")

                coordinator.on_entry_candidate()
                coordinator.on_entry_confirmed()
                coordinator.on_zone_entered()
                print("[missao] faixa PRATA confirmada; executando o handoff")
                HandoffExecutor(system, HANDOFF_TO_RESCUE).run()
                coordinator.on_rescue_started()

                returncode = system.wait_rescue()
                if returncode != RESCUE_EXIT_OK:
                    raise RuntimeError(
                        f"resgate terminou com codigo {returncode}; "
                        "reiniciando antes da faixa prata")
                print("[missao] resgate concluido; voltando ao percurso")

                HandoffExecutor(system, HANDOFF_TO_LINE).run()
                coordinator.state = MissionState.FOLLOW_LINE

            except KeyboardInterrupt:
                raise
            except Exception as err:               # noqa: BLE001
                print(f"[missao] tentativa interrompida: {err}")
                coordinator.abort(str(err))
                tentativas += 1
                while True:
                    try:
                        print(f"[missao] iniciando tentativa {tentativas}")
                        system.reiniciar_missao_do_percurso(str(err))
                        coordinator = MissionCoordinator(policy=args.policy)
                        break
                    except KeyboardInterrupt:
                        raise
                    except Exception as erro_recuperacao:  # noqa: BLE001
                        print(
                            "[missao] ainda aguardando recursos para "
                            f"reiniciar: {erro_recuperacao}")
                        time.sleep(config.MISSION_RECOVERY_DELAY_S)

    except KeyboardInterrupt:
        print("\n[missao] Ctrl-C - encerrando...")
        coordinator.abort("Ctrl-C")
        codigo = 130
    finally:
        system.shutdown()
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        inventory = coordinator.inventory
        print(
            f"[missao] estado final: {coordinator.state} | "
            f"prata {inventory.silver_deposited}/2, "
            f"preta {inventory.black_deposited}/1")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
