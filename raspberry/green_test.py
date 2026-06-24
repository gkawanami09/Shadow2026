"""Executa o detector de verde em imagem salva ou em frames da camera CSI."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    GREEN_INTERVALO_DEBUG,
    GREEN_SALVAR_DEBUG_EVENTOS,
    PASTA_CAPTURAS,
)
from green_detector import criar_debug_verde, criar_mascara_linha_global, detectar_verde
from line_test import detectar_linha


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta marcacoes verdes sem controlar o robo.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--camera", action="store_true", help="Le frames da camera CSI.")
    origem.add_argument("--imagem", help="Caminho para uma imagem BGR salva.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva mascara e imagem anotada.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra o debug em uma janela OpenCV.")
    return parser.parse_args()


def imprimir_resultado(resultado, resultado_linha):
    print(
        f"linha_encontrada: {resultado_linha['encontrou_linha']} | "
        f"black_mask_pixels: {cv2.countNonZero(resultado['mascara_linha_global'])}"
    )
    print(
        f"verde final: {resultado['tipo_confirmado']} | "
        f"detectado: {resultado['tipo_detectado']} | "
        f"confirmados: {resultado['qtd_contornos_confirmados']}/{resultado['qtd_contornos_detectados']} | "
        f"conf: {resultado['confianca']:.2f} | "
        f"area_esq: {resultado['area_esquerda']:.0f} | "
        f"area_dir: {resultado['area_direita']:.0f} | "
        f"area_centro: {resultado['area_centro']:.0f} | "
        f"qtd: {resultado['qtd_contornos']} | obs: {resultado['observacao']}"
    )
    for indice, contorno in enumerate(resultado["contornos"][:5], start=1):
        print(
            f"contorno {indice} | lado: {contorno['lado']} | "
            f"area: {contorno['area']:.0f} | S: {contorno['mean_s']:.0f} | "
            f"G-R: {contorno['g_minus_r']:.0f} | "
            f"G-B: {contorno['g_minus_b']:.0f} | "
            f"ratio: {contorno['green_ratio']:.2f} | "
            f"conf: {contorno['confirmado']} | "
            f"motivo: {contorno['motivo_confirmacao']} | "
            f"black: {contorno['black_near_pixels']} | "
            f"zone: {contorno['area_in_confirm_zone_ratio']:.2f}"
        )


def salvar_debug(frame, resultado):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    caminho_debug = pasta / f"debug_verde_{timestamp}.jpg"
    caminho_mascara = pasta / f"debug_verde_mascara_{timestamp}.png"
    if not cv2.imwrite(str(caminho_debug), criar_debug_verde(frame, resultado)):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho_debug}")
    if not cv2.imwrite(str(caminho_mascara), resultado["mascara"]):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho_mascara}")
    print(f"Debug salvo: {caminho_debug}")
    return caminho_debug, caminho_mascara


def processar_frame(frame, x_referencia=None):
    resultado_linha = detectar_linha(frame)
    if x_referencia is None:
        x_referencia = resultado_linha["centro_imagem_x"]
    mascara_linha_global = criar_mascara_linha_global(resultado_linha)
    resultado = detectar_verde(frame, x_referencia, mascara_linha_global)
    imprimir_resultado(resultado, resultado_linha)
    return resultado


def executar_imagem(caminho, salvar, mostrar):
    frame = cv2.imread(str(caminho))
    if frame is None:
        raise RuntimeError("Nao foi possivel carregar a imagem. Verifique o caminho informado.")
    resultado = processar_frame(frame)
    debug = criar_debug_verde(frame, resultado)
    if salvar:
        salvar_debug(frame, resultado)
    if mostrar:
        cv2.imshow("Debug verde - pressione uma tecla", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def executar_camera(salvar, mostrar):
    camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
    ultimo_debug = 0.0
    ultimo_tipo_salvo = None
    print("Detector de verde iniciado. Pressione q na janela ou CTRL+C para sair.")
    try:
        while True:
            frame = capturar_frame_bgr(camera)
            resultado = processar_frame(frame)
            agora = time.monotonic()
            if (
                salvar
                and GREEN_SALVAR_DEBUG_EVENTOS
                and resultado["tipo"] != ultimo_tipo_salvo
                and agora - ultimo_debug >= GREEN_INTERVALO_DEBUG
            ):
                salvar_debug(frame, resultado)
                ultimo_debug = agora
                ultimo_tipo_salvo = resultado["tipo"]

            if mostrar:
                cv2.imshow("Debug verde - pressione q para sair", criar_debug_verde(frame, resultado))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.stop()
        if mostrar:
            cv2.destroyAllWindows()


def main():
    argumentos = ler_argumentos()
    if not argumentos.camera and not argumentos.imagem:
        print("Use --camera ou --imagem.")
        return 0
    try:
        if argumentos.camera:
            executar_camera(argumentos.salvar_debug, argumentos.mostrar)
        else:
            executar_imagem(argumentos.imagem, argumentos.salvar_debug, argumentos.mostrar)
        return 0
    except (RuntimeError, ValueError, cv2.error) as erro:
        print(f"Erro: {erro}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
