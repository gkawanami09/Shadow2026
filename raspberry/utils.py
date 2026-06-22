"""Funcoes simples para a comunicacao serial do robo."""

import serial
from serial.tools import list_ports


def detectar_porta_serial():
    """Retorna a porta mais provavel do Arduino, ou None se nao encontrar."""
    portas = list(list_ports.comports())
    if not portas:
        print("Nenhuma porta serial foi encontrada.")
        return None

    for porta in portas:
        nome = porta.device.lower()
        descricao = (porta.description or "").lower()
        if any(item in nome for item in ("ttyacm", "ttyusb", "com")) or "arduino" in descricao:
            print(f"Porta serial detectada: {porta.device}")
            return porta.device

    print("Portas seriais disponiveis:")
    for porta in portas:
        print(f"- {porta.device}: {porta.description}")
    return None


def abrir_serial(porta=None, baud_rate=115200, timeout=2.0):
    """Abre e retorna uma conexao serial com o Arduino."""
    porta_escolhida = porta or detectar_porta_serial()
    if not porta_escolhida:
        raise RuntimeError("Nao foi possivel detectar a porta serial.")

    try:
        conexao = serial.Serial(porta_escolhida, baud_rate, timeout=timeout)
    except serial.SerialException as erro:
        raise RuntimeError(f"Erro ao abrir serial: {erro}") from erro

    print(f"Porta serial usada: {conexao.port}")
    return conexao


def enviar_comando(conexao, comando, esperar_resposta=True):
    """Envia um comando terminado por nova linha e retorna a resposta."""
    comando = comando.strip()
    if not comando:
        raise ValueError("O comando nao pode estar vazio.")

    conexao.write((comando + "\n").encode("utf-8"))
    conexao.flush()
    if not esperar_resposta:
        return None

    resposta = conexao.readline().decode("utf-8", errors="replace").strip()
    return resposta


def limitar_velocidade(velocidade, limite=120):
    """Limita uma velocidade ao intervalo seguro definido."""
    return max(-limite, min(limite, int(velocidade)))
