#!/usr/bin/env python3
"""
tools/steer_test.py — dirige o robô pelo teclado, através de steer().
Rebuilt from robot_v.3/Python/test/steer_test.py (que falava GPIO/L298N);
esta versão usa a MESMA lei steer(angle, speed) de control/steer.py sobre o
link serial SPEC 01 — validando a Fase A com o robô na mão.

Teclas (terminal, sem janela):
    w  frente            s  ré
    a  pivot esquerda    d  pivot direita
    q  arco esquerda     e  arco direita
    espaço  parar        +/-  velocidade      x  sair
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.steer import init_steering, steer  # noqa: E402
from serial_link.arduino import Arduino  # noqa: E402

if sys.platform == "win32":
    import msvcrt

    def get_key():
        if msvcrt.kbhit():
            return msvcrt.getch().decode(errors="ignore").lower()
        return None
else:
    import select
    import termios
    import tty

    def get_key():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


ARC_ANGLE = 60  # arco suave para q/e


def main():
    print("Conectando ao Arduino…")
    arduino = Arduino()
    init_steering(arduino)

    speed = .5
    command = ("stop",)

    old_settings = None
    if sys.platform != "win32":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    print(__doc__)
    print(f"Velocidade: {speed:.1f} — dirigindo. RODAS SUSPENSAS primeiro!")

    try:
        while True:
            key = get_key()
            if key == "x" or key == "\x03":
                break
            elif key == "w":
                command = ("steer", 0)
            elif key == "s":
                command = ("steer", 200)
            elif key == "a":
                command = ("steer", -180)
            elif key == "d":
                command = ("steer", 180)
            elif key == "q":
                command = ("steer", -ARC_ANGLE)
            elif key == "e":
                command = ("steer", ARC_ANGLE)
            elif key == " ":
                command = ("stop",)
            elif key in ("+", "="):
                speed = min(1.0, speed + .1)
                print(f"\rVelocidade: {speed:.1f}   ", end="", flush=True)
            elif key == "-":
                speed = max(.1, speed - .1)
                print(f"\rVelocidade: {speed:.1f}   ", end="", flush=True)

            if command[0] == "stop":
                steer()
            else:
                steer(command[1], speed)

            arduino.refresh()
            time.sleep(.05)
    finally:
        steer()
        arduino.close()
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nParado. Até mais.")


if __name__ == "__main__":
    main()
