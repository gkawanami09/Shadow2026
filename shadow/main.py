#!/usr/bin/env python3
"""
main.py — Shadow2026 line follower entry point.
Ported from Overengineering² Reading Dossier, Section 7 (process model)
  Original source: robot_v.3/Python/main/main.py (lines 592-609 — process
  spawning; the CustomTkinter GUI, lines 40-590, was NOT ported)
Shadow2026 adaptations:
  - 4 workers + GUI → 2 processes: vision (Picamera2 + detection) and control
    (state machine + serial). Headless by default.
  - --debug: opens ONE OpenCV window with the annotated line-camera frame,
    transported via multiprocessing.shared_memory (created here, written by
    the vision process). 'q' in the window or Ctrl-C stops everything.
  - --vision-only: spawns only the vision process (Phase B bring-up).
  - Shutdown: terminate flag → children exit their loops → control's finally
    sends PARAR and closes the serial port. Children ignore SIGINT so the
    parent coordinates the shutdown order.

Uso:
    python3 shadow/main.py            # operacao normal (headless)
    python3 shadow/main.py --debug    # com janela anotada
    python3 shadow/main.py --vision-only --debug   # so visao (Fase B)
"""

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiprocessing import Process, shared_memory  # noqa: E402

import config  # noqa: E402


def vision_main(debug):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from vision.pipeline import vision_loop
    vision_loop(debug)


def control_main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from control.loop import control_loop
    control_loop()


def _create_debug_shm():
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
    parser = argparse.ArgumentParser(description="Shadow2026 line follower")
    parser.add_argument("--debug", action="store_true",
                        help="mostra a janela com o frame anotado")
    parser.add_argument("--vision-only", action="store_true",
                        help="roda apenas o processo de visão (bring-up)")
    args = parser.parse_args()

    # importa mp_manager APOS o parse (instancia o Manager)
    from shared.mp_manager import status, terminate

    shm = _create_debug_shm() if args.debug else None

    vision_p = Process(target=vision_main, args=(args.debug,), name="shadow-vision")
    vision_p.start()

    control_p = None
    if not args.vision_only:
        time.sleep(.5)  # partida escalonada, como no OE²
        control_p = Process(target=control_main, name="shadow-control")
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
        print("Encerrado.")


if __name__ == "__main__":
    main()
