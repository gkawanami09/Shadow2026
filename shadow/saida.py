#!/usr/bin/env python3
"""Teste direto da manobra pos-vermelho, sem executar o resgate inteiro.

Uso na Raspberry::

    python3 shadow/saida.py

Este comando movimenta o robo. Nunca o execute enquanto ``mission.py``,
``main.py`` ou ``resgate.py`` estiverem usando os motores.
"""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from controle.direcao import init_steering, steer  # noqa: E402
from controle.saida_parede_resgate import (  # noqa: E402
    executar_alinhamento_parede,
)
from controle.trava_motores import MotorLockError, MotorOwnerLock  # noqa: E402


# Adiado para que ``python3 shadow/saida.py --help`` funcione ate em um
# computador de desenvolvimento sem pyserial. Na Raspberry, ``main`` importa
# a classe real antes de tocar na serial; os testes injetam uma classe falsa.
Arduino = None


EXIT_OK = 0
EXIT_FALHA_MANOBRA = 3
EXIT_FALHA_INICIALIZACAO = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Testa somente a saida pos-vermelho: avanco curto, giro de 90 "
            "graus e alinhamento na parede direita."))
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    trava = MotorOwnerLock("teste-saida")
    try:
        trava.acquire()
    except MotorLockError as erro:
        print(f"[saida] recusada: {erro}")
        return EXIT_FALHA_INICIALIZACAO

    arduino = None
    try:
        classe_arduino = Arduino
        if classe_arduino is None:
            from comunicacao_serial.arduino import Arduino as classe_arduino
        arduino = classe_arduino()
        init_steering(arduino)
        steer()
        # Uma desconexao deve encerrar este teste; nunca reconectar e repetir
        # uma manobra baseada numa postura antiga do robo.
        arduino.travar_sessao()
        print("[saida] iniciando como se o deposito vermelho tivesse terminado")
        resultado = executar_alinhamento_parede(arduino)
        if resultado == "alinhado_parede":
            print("[saida] alinhamento concluido; robo parado")
            return EXIT_OK
        print("[saida] alinhamento interrompido; robo parado")
        return EXIT_FALHA_MANOBRA
    except KeyboardInterrupt:
        print("\n[saida] interrompida; robo parado")
        return 130
    except (OSError, RuntimeError) as erro:
        print(f"[saida] falha: {erro}")
        return EXIT_FALHA_INICIALIZACAO
    finally:
        if arduino is not None:
            try:
                steer()
            except (OSError, RuntimeError):
                pass
            try:
                arduino.led("APAGADO")
            except (OSError, RuntimeError):
                pass
            try:
                arduino.close()
            except (OSError, RuntimeError):
                pass
        trava.release()


if __name__ == "__main__":
    sys.exit(main())
