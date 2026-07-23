#!/usr/bin/env python3
"""Terminal manual do Shadow para enviar comandos ao firmware do Arduino.

Uso:
    python3 -m shadow.tools.controle_serial
    python3 -m shadow.tools.controle_serial --port /dev/ttyACM0
    python3 -m shadow.tools.controle_serial --comando "SERVO FUTABA 20"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serial_link.arduino import Arduino  # noqa: E402


AJUDA = """
Digite exatamente o comando que deseja enviar, por exemplo:
  SERVO GARRA_ESQ 20     move 20 graus no sentido positivo
  SERVO GARRA_DIR -20    move 20 graus no sentido contrario
  SERVO CACAMBA 10
  SERVO FUTABA -5
  LED ACESO
  LED APAGADO
  ULTRASSOM
  PING
  STATUS
  PARAR

Tambem aceita os comandos existentes de motores. Digite "sair" para fechar.
""".strip()


def enviar(arduino, comando):
    resposta = arduino.comando_serial(comando)
    if resposta is None:
        print("Arduino nao respondeu.")
        return

    print(f"Arduino: {resposta}")
    if resposta.startswith("OK ULTRASSOM "):
        try:
            distancia_mm = int(resposta.split()[-1])
        except (ValueError, IndexError):
            return
        if distancia_mm < 0:
            print("Distancia: sem eco")
        else:
            print(f"Distancia: {distancia_mm} mm ({distancia_mm / 10:.1f} cm)")


def interativo(arduino):
    print(AJUDA)
    while True:
        try:
            comando = input("Shadow> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not comando:
            continue
        if comando.lower() in ("sair", "exit", "quit"):
            break
        if comando.lower() in ("ajuda", "help", "?"):
            print(AJUDA)
            continue
        enviar(arduino, comando)


def main():
    parser = argparse.ArgumentParser(description="Terminal serial manual do Shadow")
    parser.add_argument("--port", default=None, help="porta; padrao: deteccao automatica")
    parser.add_argument("--comando", help="envia um comando e encerra")
    args = parser.parse_args()

    print("Conectando ao Arduino...")
    arduino = Arduino(port=args.port)
    try:
        if args.comando:
            enviar(arduino, args.comando)
        else:
            interativo(arduino)
    finally:
        arduino.close()
        print("Conexao fechada; motores parados.")


if __name__ == "__main__":
    main()
