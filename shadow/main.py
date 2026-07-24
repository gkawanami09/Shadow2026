#!/usr/bin/env python3
"""Inicia os processos de visão e controle do segue-linha."""

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiprocessing import Process, shared_memory  # noqa: E402

import config  # noqa: E402


def iniciar_visao(debug):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from visao.processamento import vision_loop
    vision_loop(debug)


def iniciar_controle():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from controle.ciclo import control_loop
    control_loop()


def _criar_memoria_debug():
    try:
        return shared_memory.SharedMemory(name=config.DEBUG_SHM_NAME, create=True,
                                          size=config.DEBUG_SHM_SIZE)
    except FileExistsError:
        # sobra de uma execucao anterior que caiu — recria limpa
        stale = shared_memory.SharedMemory(name=config.DEBUG_SHM_NAME)
        stale.close()
        stale.unlink()
        return shared_memory.SharedMemory(name=config.DEBUG_SHM_NAME, create=True,
                                          size=config.DEBUG_SHM_SIZE)


def main():
    parser = argparse.ArgumentParser(description="Segue-linha do Shadow2026")
    parser.add_argument("--debug", action="store_true",
                        help="mostra a janela com o frame anotado")
    parser.add_argument("--vision-only", action="store_true",
                        help="roda apenas o processo de visão (bring-up)")
    args = parser.parse_args()

    motor_lock = None
    if not args.vision_only:
        from controle.trava_motores import MotorLockError, MotorOwnerLock
        motor_lock = MotorOwnerLock("segue-linha")
        try:
            motor_lock.acquire()
        except MotorLockError as err:
            parser.error(str(err))

    # Importa os dados compartilhados depois de ler os argumentos.
    from shared.dados_compartilhados import status, terminate

    shm = _criar_memoria_debug() if args.debug else None

    vision_p = Process(target=iniciar_visao, args=(args.debug,),
                       name="shadow-visao")
    vision_p.start()

    control_p = None
    if not args.vision_only:
        time.sleep(.5)
        control_p = Process(target=iniciar_controle, name="shadow-controle")
        control_p.start()

    children = [p for p in (vision_p, control_p) if p is not None]

    last_status = ""
    try:
        if args.debug:
            import cv2
            import numpy as np
            frame = np.ndarray((config.camera_y, config.camera_x, 3),
                               dtype=np.uint8, buffer=shm.buf)
            while all(p.is_alive() for p in children):
                cv2.imshow("Shadow2026 - camera de linha", frame.copy())
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                if status.value != last_status:
                    print(f"[status] {status.value}")
                    last_status = status.value
            cv2.destroyAllWindows()
        else:
            while all(p.is_alive() for p in children):
                if status.value != last_status:
                    print(f"[status] {status.value}")
                    last_status = status.value
                time.sleep(.1)
    except KeyboardInterrupt:
        print("\nCtrl-C — encerrando…")
    finally:
        terminate.value = True
        deadline = time.monotonic() + 5
        for p in children:
            p.join(timeout=max(0, deadline - time.monotonic()))
        for p in children:
            if p.is_alive():
                print(f"[shutdown] forçando término de {p.name}")
                p.terminate()
                p.join(timeout=1)
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        if motor_lock is not None:
            motor_lock.release()
        print("Encerrado.")


if __name__ == "__main__":
    main()
