"""Terminal interativo para controlar o Arduino pela Raspberry Pi.

Exemplos:
    python3 raspberry/controle_serial.py
    python3 raspberry/controle_serial.py --porta /dev/ttyACM0
    python3 raspberry/controle_serial.py --comando "SERVO FUTABA 90"
"""

import argparse
import time

from utils import abrir_serial, enviar_comando


AJUDA = """
Comandos disponiveis:
  PING
  STATUS
  PARAR
  SERVO GARRA_ESQ <0..180>
  SERVO GARRA_DIR <0..180>
  SERVO CACAMBA <0..180>
  SERVO FUTABA <0..180>
  LED ACESO
  LED APAGADO
  ULTRASSOM
  MOTOR <FE|TE|FD|TD> <-255..255>
  LADO <esquerda> <direita>
  RODAS <FE> <TE> <FD> <TD>
  FRENTE <velocidade>
  TRAS <velocidade>
  GIRAR_ESQ <velocidade>
  GIRAR_DIR <velocidade>

Comandos deste terminal:
  ajuda    mostra esta lista
  sair     envia PARAR e fecha a conexao
""".strip()


def enviar(conexao, comando):
    """Envia uma linha ao Arduino e imprime sua resposta."""
    resposta = enviar_comando(conexao, comando, esperar_resposta=True)
    if resposta:
        print(f"Arduino: {resposta}")

        if resposta.startswith("OK ULTRASSOM "):
            try:
                distancia_mm = int(resposta.split()[-1])
                if distancia_mm >= 0:
                    print(f"Distancia: {distancia_mm} mm ({distancia_mm / 10:.1f} cm)")
                else:
                    print("Distancia: sem eco")
            except (ValueError, IndexError):
                pass
    else:
        print("Arduino nao respondeu.")


def executar_interativo(conexao):
    print(AJUDA)
    print("\nDigite um comando e pressione Enter.")

    while True:
        try:
            comando = input("Arduino> ").strip()
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

        enviar(conexao, comando)


def main():
    parser = argparse.ArgumentParser(
        description="Controla motores, servos, LED e ultrassonico pela Serial."
    )
    parser.add_argument(
        "--porta",
        default=None,
        help="Porta serial, por exemplo /dev/ttyACM0. O padrao detecta automaticamente.",
    )
    parser.add_argument(
        "--comando",
        help='Envia apenas um comando, por exemplo "SERVO FUTABA 90".',
    )
    args = parser.parse_args()

    conexao = abrir_serial(args.porta, baud_rate=115200, timeout=2.0)

    try:
        # Abrir a USB normalmente reinicia o Uno. Aguarda o boot e descarta o
        # banner para ele nao ser confundido com a resposta do primeiro comando.
        time.sleep(2.0)
        conexao.reset_input_buffer()

        if args.comando:
            enviar(conexao, args.comando)
        else:
            executar_interativo(conexao)
    finally:
        try:
            enviar(conexao, "PARAR")
        except RuntimeError:
            pass
        conexao.close()
        print("Conexao fechada; motores parados.")


if __name__ == "__main__":
    main()
