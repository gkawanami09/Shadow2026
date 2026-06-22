"""Deteccao basica da linha preta, sem controlar motores."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    AREA_MINIMA_LINHA,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    IMAGEM_TESTE_PADRAO,
    LIMIAR_PRETO,
    PASTA_CAPTURAS,
    ROI_X_FIM,
    ROI_X_INICIO,
    ROI_Y_FIM,
    ROI_Y_INICIO,
)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta uma linha preta em uma imagem.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--imagem", help="Caminho de uma imagem ja salva.")
    origem.add_argument("--camera", action="store_true", help="Captura uma imagem nova da camera CSI.")
    parser.add_argument("--limiar", type=int, default=LIMIAR_PRETO, help="Limiar de preto entre 0 e 255.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra a imagem de debug em uma janela.")
    parser.add_argument("--salvar-mascara", action="store_true", help="Salva tambem a mascara binaria.")
    return parser.parse_args()


def mostrar_instrucoes():
    print("Informe uma imagem ou use a camera.")
    print("Exemplo: python3 raspberry/line_test.py --imagem captures/frame.jpg")
    print("Exemplo: python3 raspberry/line_test.py --camera --salvar-mascara")


def criar_pasta_capturas():
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def carregar_imagem(caminho):
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise RuntimeError(f"Nao foi possivel carregar a imagem: {caminho}")
    print(f"Imagem carregada: {caminho}")
    return imagem


def calcular_roi(imagem):
    altura, largura = imagem.shape[:2]
    x_inicio = int(largura * ROI_X_INICIO)
    x_fim = int(largura * ROI_X_FIM)
    y_inicio = int(altura * ROI_Y_INICIO)
    y_fim = int(altura * ROI_Y_FIM)

    if x_inicio >= x_fim or y_inicio >= y_fim:
        raise RuntimeError("A ROI configurada e invalida. Revise os valores em config.py.")
    return imagem[y_inicio:y_fim, x_inicio:x_fim], x_inicio, x_fim, y_inicio, y_fim


def criar_mascara_linha(roi, limiar):
    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    desfocada = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, mascara = cv2.threshold(desfocada, limiar, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def encontrar_linha(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, 0.0

    maior_contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(maior_contorno)
    if area < AREA_MINIMA_LINHA:
        return None, area
    return maior_contorno, area


def desenhar_debug(imagem, contorno, area, x_inicio, x_fim, y_inicio, y_fim):
    debug = imagem.copy()
    altura, largura = imagem.shape[:2]
    centro_imagem_x = largura // 2
    cv2.rectangle(debug, (x_inicio, y_inicio), (x_fim, y_fim), (255, 255, 0), 2)
    cv2.line(debug, (centro_imagem_x, 0), (centro_imagem_x, altura), (0, 255, 255), 2)

    resultado = {"centro_imagem_x": centro_imagem_x, "area": area, "erro": None}
    if contorno is None:
        cv2.putText(debug, "Linha nao encontrada", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return debug, resultado

    momentos = cv2.moments(contorno)
    if momentos["m00"] == 0:
        return desenhar_debug(imagem, None, area, x_inicio, x_fim, y_inicio, y_fim)

    centro_roi_x = int(momentos["m10"] / momentos["m00"])
    centro_roi_y = int(momentos["m01"] / momentos["m00"])
    centro_linha_x = x_inicio + centro_roi_x
    centro_linha_y = y_inicio + centro_roi_y
    erro = centro_linha_x - centro_imagem_x
    contorno_completo = contorno + np.array([[[x_inicio, y_inicio]]])

    cv2.drawContours(debug, [contorno_completo], -1, (0, 255, 0), 2)
    cv2.circle(debug, (centro_linha_x, centro_linha_y), 7, (0, 0, 255), -1)
    cv2.putText(debug, f"Erro: {erro}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    resultado.update(
        {"centro_linha_x": centro_linha_x, "centro_linha_y": centro_linha_y, "erro": erro}
    )
    return debug, resultado


def salvar_imagem(caminho, imagem):
    if not cv2.imwrite(str(caminho), imagem):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho}")


def mostrar_debug(imagem):
    try:
        cv2.imshow("Debug da linha - pressione q para sair", imagem)
        while cv2.waitKey(20) & 0xFF != ord("q"):
            pass
    except cv2.error as erro:
        raise RuntimeError("Preview indisponivel. Em SSH sem interface grafica, abra a imagem salva.") from erro
    finally:
        cv2.destroyAllWindows()


def main():
    argumentos = ler_argumentos()
    caminho_imagem = argumentos.imagem or IMAGEM_TESTE_PADRAO
    if not argumentos.camera and not caminho_imagem:
        mostrar_instrucoes()
        return 0
    if not 0 <= argumentos.limiar <= 255:
        print("Erro: o limiar deve estar entre 0 e 255.")
        return 1

    camera = None
    try:
        if argumentos.camera:
            print("Capturando imagem nova da camera...")
            camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
            imagem = capturar_frame_bgr(camera)
        else:
            imagem = carregar_imagem(caminho_imagem)

        altura, largura = imagem.shape[:2]
        roi, x_inicio, x_fim, y_inicio, y_fim = calcular_roi(imagem)
        mascara = criar_mascara_linha(roi, argumentos.limiar)
        contorno, area = encontrar_linha(mascara)
        debug, resultado = desenhar_debug(imagem, contorno, area, x_inicio, x_fim, y_inicio, y_fim)

        pasta = criar_pasta_capturas()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_debug = pasta / f"debug_linha_{timestamp}.jpg"
        salvar_imagem(caminho_debug, debug)
        if argumentos.salvar_mascara:
            caminho_mascara = pasta / f"mascara_linha_{timestamp}.jpg"
            salvar_imagem(caminho_mascara, mascara)
            print(f"Mascara salva em: {caminho_mascara}")

        print(f"Resolucao: {largura}x{altura}")
        print(f"ROI: x={x_inicio} ate {x_fim}, y={y_inicio} ate {y_fim}")
        print(f"Centro da imagem: {resultado['centro_imagem_x']}")
        print(f"Area da linha: {area:.0f}")
        print(f"Debug salvo em: {caminho_debug}")
        if contorno is None:
            print("Linha nao encontrada.")
            print("Tente aumentar o limiar, ajustar a ROI ou melhorar a iluminacao.")
            return 2

        print(f"Centro da linha: {resultado['centro_linha_x']}")
        print(f"Erro: {resultado['erro']}")
        if argumentos.mostrar:
            mostrar_debug(debug)
        return 0
    except RuntimeError as erro:
        print(f"Erro: {erro}")
        return 1
    finally:
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
