"""Teste simples da camera CSI e captura de imagens para debug."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH, PASTA_CAPTURAS

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_CALIBRACAO = RAIZ_PROJETO / "calibration" / "camera.json"


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Teste da camera CSI do robo OBR.")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--salvar", action="store_true", help="Salva um unico frame.")
    modo.add_argument("--sequencia", type=int, metavar="QUANTIDADE", help="Salva varios frames.")
    modo.add_argument("--preview", action="store_true", help="Mostra frames em uma janela OpenCV.")
    parser.add_argument("--largura", type=int, default=CAMERA_WIDTH, help="Largura da imagem.")
    parser.add_argument("--altura", type=int, default=CAMERA_HEIGHT, help="Altura da imagem.")
    parser.add_argument("--intervalo", type=float, default=0.2, help="Intervalo entre frames da sequencia.")
    return parser.parse_args()


def criar_pasta_capturas():
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def iniciar_camera(largura, altura):
    if Picamera2 is None:
        raise RuntimeError(
            "biblioteca picamera2 nao encontrada. "
            "Instale/verifique a camera no Raspberry Pi OS antes de continuar."
        )

    camera = Picamera2()
    configuracao = camera.create_still_configuration(main={"size": (largura, altura)})
    camera.configure(configuracao)
    camera.start()
    # Aguarda exposicao e ganho automaticos estabilizarem antes da primeira foto.
    time.sleep(2.0)
    return camera


def capturar_frame_bgr(camera):
    """Captura um frame RGB da CSI e converte para BGR usado pelo OpenCV."""
    frame_rgb = camera.capture_array("main")
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def salvar_frame(frame, caminho):
    if not cv2.imwrite(str(caminho), frame):
        raise RuntimeError(f"Nao foi possivel salvar a imagem em: {caminho}")


def atualizar_calibracao(caminho_imagem):
    """Registra somente que a captura de teste foi concluida com sucesso."""
    try:
        dados = json.loads(ARQUIVO_CALIBRACAO.read_text(encoding="utf-8"))
        dados["camera_testada"] = True
        dados["ultima_imagem_teste"] = str(caminho_imagem)
        ARQUIVO_CALIBRACAO.write_text(
            json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError) as erro:
        print(f"Aviso: nao foi possivel atualizar camera.json: {erro}")


def salvar_imagem_unica(camera, pasta):
    print("Capturando frame...")
    inicio = time.monotonic()
    frame = capturar_frame_bgr(camera)
    tempo_captura = time.monotonic() - inicio
    caminho = pasta / f"frame_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    salvar_frame(frame, caminho)
    atualizar_calibracao(caminho)
    print(f"Imagem salva em: {caminho}")
    print(f"Tempo de captura: {tempo_captura:.3f} s")


def salvar_sequencia(camera, pasta, quantidade, intervalo):
    if quantidade <= 0:
        raise ValueError("A quantidade da sequencia deve ser maior que zero.")
    if intervalo < 0:
        raise ValueError("O intervalo nao pode ser negativo.")

    print(f"Capturando sequencia de {quantidade} frames...")
    inicio = time.monotonic()
    ultimo_caminho = None
    for indice in range(1, quantidade + 1):
        frame = capturar_frame_bgr(camera)
        caminho = pasta / f"frame_{indice:03d}.jpg"
        salvar_frame(frame, caminho)
        ultimo_caminho = caminho
        print(f"Imagem salva em: {caminho}")
        if indice < quantidade:
            time.sleep(intervalo)

    duracao = time.monotonic() - inicio
    fps = quantidade / duracao if duracao > 0 else 0
    if ultimo_caminho:
        atualizar_calibracao(ultimo_caminho)
    print(f"FPS estimado da sequencia: {fps:.1f}")


def mostrar_preview(camera):
    print("Preview iniciado. Pressione q para fechar.")
    print("Se estiver usando SSH sem interface grafica, o preview pode nao abrir.")
    try:
        while True:
            frame = capturar_frame_bgr(camera)
            cv2.imshow("Camera OBR - pressione q para sair", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except cv2.error as erro:
        raise RuntimeError(
            "Nao foi possivel abrir o preview. Em SSH sem interface grafica, use --salvar."
        ) from erro
    finally:
        cv2.destroyAllWindows()


def main():
    argumentos = ler_argumentos()
    if argumentos.largura <= 0 or argumentos.altura <= 0:
        print("Erro: largura e altura devem ser maiores que zero.")
        return 1

    print("Iniciando teste da camera...")
    print(f"Resolucao: {argumentos.largura}x{argumentos.altura}")
    print(f"FPS configurado de referencia: {CAMERA_FPS}")
    camera = None
    try:
        pasta = criar_pasta_capturas()
        camera = iniciar_camera(argumentos.largura, argumentos.altura)
        if argumentos.preview:
            mostrar_preview(camera)
        elif argumentos.sequencia is not None:
            salvar_sequencia(camera, pasta, argumentos.sequencia, argumentos.intervalo)
        else:
            salvar_imagem_unica(camera, pasta)
        print("Teste finalizado com sucesso.")
        return 0
    except (RuntimeError, ValueError) as erro:
        print("Erro ao abrir camera ou capturar imagem.")
        print(f"Detalhe: {erro}")
        print("Verifique a conexao da camera, o Raspberry Pi OS e se outro programa esta usando a camera.")
        return 1
    finally:
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
