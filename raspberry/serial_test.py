"""Teste simples da comunicacao serial e dos motores do robo."""

import argparse
import time

from config import BAUD_RATE, SERIAL_PORT, TIMEOUT_SERIAL, VELOCIDADE_TESTE_BAIXA
from utils import abrir_serial, enviar_comando


def enviar_e_mostrar(conexao, comando):
    """Envia um comando e mostra claramente a resposta do Arduino."""
    print(f"Enviando: {comando}")
    resposta = enviar_comando(conexao, comando)
    print(f"Resposta: {resposta or '(sem resposta)'}")
    return resposta


def enviar_ping(conexao):
    return enviar_e_mostrar(conexao, "PING")


def parar(conexao):
    return enviar_e_mostrar(conexao, "PARAR")


def controlar_motor(conexao, motor, velocidade):
    return enviar_e_mostrar(conexao, f"MOTOR {motor} {velocidade}")


def controlar_lados(conexao, velocidade_esquerda, velocidade_direita):
    return enviar_e_mostrar(conexao, f"LADO {velocidade_esquerda} {velocidade_direita}")


def controlar_rodas(conexao, fe, te, fd, td):
    return enviar_e_mostrar(conexao, f"RODAS {fe} {te} {fd} {td}")


def modo_interativo(conexao):
    print("Modo interativo. Digite comandos como 'MOTOR FE 60' ou 'PARAR'.")
    print("Digite 'sair' para encerrar. O robo sera parado ao sair.")
    while True:
        try:
            comando = input("> ").strip()
        except EOFError:
            comando = "sair"
        if comando.lower() == "sair":
            break
        if comando:
            enviar_e_mostrar(conexao, comando)


def teste_motores(conexao):
    print("ATENCAO: deixe o robo suspenso, com as rodas fora do chao.")
    input("Pressione ENTER para continuar ou CTRL+C para cancelar.")

    enviar_ping(conexao)
    for motor in ("FE", "TE", "FD", "TD"):
        controlar_motor(conexao, motor, VELOCIDADE_TESTE_BAIXA)
        time.sleep(0.7)
        parar(conexao)
        if motor != "TD":
            time.sleep(0.5)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Teste serial do robo OBR.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou 'auto'.")
    parser.add_argument("--comando", help="Envia um unico comando ao Arduino.")
    parser.add_argument("--interativo", action="store_true", help="Abre o modo interativo.")
    parser.add_argument("--teste-motores", action="store_true", help="Testa um motor por vez com o robo suspenso.")
    return parser.parse_args()


def main():
    argumentos = ler_argumentos()
    if sum(bool(opcao) for opcao in (argumentos.comando, argumentos.interativo, argumentos.teste_motores)) != 1:
        print("Escolha exatamente uma opcao: --comando, --interativo ou --teste-motores.")
        return 2

    porta = SERIAL_PORT if argumentos.porta == "auto" else argumentos.porta
    conexao = None
    try:
        conexao = abrir_serial(porta, BAUD_RATE, TIMEOUT_SERIAL)
        # A abertura da USB serial pode reiniciar o Arduino.
        time.sleep(2.0)
        conexao.reset_input_buffer()

        if argumentos.comando:
            enviar_e_mostrar(conexao, argumentos.comando)
        elif argumentos.interativo:
            modo_interativo(conexao)
        else:
            teste_motores(conexao)
    except (RuntimeError, ValueError) as erro:
        print("Erro ao abrir ou usar serial.")
        print("Verifique se o Arduino esta conectado e se a porta esta correta.")
        print(f"Detalhe: {erro}")
        return 1
    except KeyboardInterrupt:
        print("\nTeste cancelado pelo usuario.")
        return 130
    finally:
        if conexao and conexao.is_open:
            try:
                parar(conexao)
            except (RuntimeError, ValueError):
                pass
            conexao.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
