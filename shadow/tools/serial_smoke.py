#!/usr/bin/env python3
"""
tools/serial_smoke.py — teste de fumaça do link serial (Fase A).

Com as RODAS SUSPENSAS:
    python3 -m shadow.tools.serial_smoke            # ciclo: para → frente → para → ré → para
    python3 -m shadow.tools.serial_smoke --watchdog # testa o watchdog de 1 s do Uno
    python3 -m shadow.tools.serial_smoke --port /dev/ttyACM0

O ciclo usa velocidade 0.5 (PWM 60). No teste de watchdog o script manda
frente e SILENCIA por 3 s — os motores devem parar sozinhos em ~1 s.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serial_link.arduino import Arduino  # noqa: E402


def hold(arduino, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        arduino.refresh()
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description="Teste de fumaça do link serial SPEC 01")
    parser.add_argument("--port", default=None, help="porta serial (padrão: auto-detect)")
    parser.add_argument("--watchdog", action="store_true", help="testa o watchdog de 1 s")
    parser.add_argument("--pwm", type=int, default=60, help="PWM do teste (padrão 60 = speed 0.5)")
    args = parser.parse_args()

    pwm = max(0, min(args.pwm, 120))

    print("Conectando ao Arduino…")
    arduino = Arduino(port=args.port)
    arduino.ping()
    time.sleep(.2)

    try:
        if args.watchdog:
            print(f"WATCHDOG: mandando frente (LADO {pwm} {pwm}) e silenciando por 3 s.")
            print("Os 4 motores devem PARAR SOZINHOS em ~1 s. Observando…")
            arduino.lado(pwm, pwm)
            time.sleep(3)  # silêncio proposital — sem refresh!
            print("Fim do silêncio. Enviando PARAR.")
            arduino.parar()
            return

        print("PARAR (1 s)…")
        arduino.parar()
        hold(arduino, 1)

        print(f"FRENTE — LADO {pwm} {pwm} (2 s)… as 4 rodas devem girar para FRENTE")
        arduino.lado(pwm, pwm)
        hold(arduino, 2)

        print("PARAR (1 s)…")
        arduino.parar()
        hold(arduino, 1)

        print(f"RÉ — LADO -{pwm} -{pwm} (2 s)… as 4 rodas devem girar para TRÁS")
        arduino.lado(-pwm, -pwm)
        hold(arduino, 2)

        print("PARAR.")
        arduino.parar()
        time.sleep(.2)

        print("\nCiclo completo. Se alguma roda girou ao contrário, ajuste o")
        print("multiplicador DIRECAO_* dela em Shadow2026/arduino/motor_controller/config.h")
        print("e regrave o firmware (RUNBOOK.md, seção 5).")
    finally:
        arduino.close()


if __name__ == "__main__":
    main()
