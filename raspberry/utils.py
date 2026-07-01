"""Funcoes simples para a comunicacao serial do robo."""

import serial
import time
from serial.tools import list_ports


SERIAL_WRITE_TIMEOUT = 0.20
SERIAL_INTER_BYTE_TIMEOUT = 0.05



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
    """Abre e retorna uma conexao serial com o Arduino.

    O write_timeout evita que o loop do robo fique preso indefinidamente caso o
    Arduino reinicie/desconecte por ruido ou queda de tensao dos motores.
    """
    porta_escolhida = porta or detectar_porta_serial()
    if not porta_escolhida:
        raise RuntimeError("Nao foi possivel detectar a porta serial.")

    try:
        conexao = serial.Serial(
            porta_escolhida,
            baud_rate,
            timeout=timeout,
            write_timeout=SERIAL_WRITE_TIMEOUT,
            inter_byte_timeout=SERIAL_INTER_BYTE_TIMEOUT,
        )
    except serial.SerialException as erro:
        raise RuntimeError(f"Erro ao abrir serial: {erro}") from erro

    print(f"Porta serial usada: {conexao.port}")
    return conexao



def enviar_comando(conexao, comando, esperar_resposta=True):
    """Envia um comando terminado por nova linha e retorna a resposta.

    No loop de controle, normalmente ``esperar_resposta`` fica falso. Nesse caso
    nao fazemos flush a cada frame, porque flush pode bloquear se o Arduino
    resetar/desconectar enquanto os motores puxam corrente.
    """
    comando = comando.strip()
    if not comando:
        raise ValueError("O comando nao pode estar vazio.")

    try:
        conexao.write((comando + "\n").encode("utf-8"))
        if not esperar_resposta:
            return None
        conexao.flush()
        resposta = conexao.readline().decode("utf-8", errors="replace").strip()
        return resposta
    except serial.SerialTimeoutException as erro:
        raise RuntimeError(f"Timeout escrevendo na serial. Arduino pode ter resetado/desconectado: {erro}") from erro
    except serial.SerialException as erro:
        raise RuntimeError(f"Erro de comunicacao serial. Verifique Arduino, USB e alimentacao dos motores: {erro}") from erro



def enviar_comando_ler_respostas(conexao, comando, timeout=1.0):
    """Envia um comando e coleta todas as respostas recebidas no periodo."""
    conexao.write((comando.strip() + "\n").encode("utf-8"))
    conexao.flush()
    inicio, respostas = time.monotonic(), []
    while time.monotonic() - inicio < timeout:
        if conexao.in_waiting > 0:
            linha = conexao.readline().decode("utf-8", errors="ignore").strip()
            if linha:
                respostas.append(linha)
        else:
            time.sleep(0.01)
    return respostas



def limitar_velocidade(velocidade, limite=120):
    """Limita uma velocidade ao intervalo seguro definido."""
    return max(-limite, min(limite, int(velocidade)))
