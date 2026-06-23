"""Detecta linha preta por faixa BGR, sem enviar comandos ao Arduino."""

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
    DILATE_LINHA,
    DIVISAO_TOPO_BAIXO,
    ERODE_FINAL,
    ERODE_INICIAL,
    KERNEL_LINHA,
    PASTA_CAPTURAS,
    PRETO_MAX_BAIXO_BGR,
    PRETO_MAX_TOPO_BGR,
    PRETO_MIN_BGR,
    ROI_X_FIM,
    ROI_X_INICIO,
    ROI_Y_FIM,
    ROI_Y_INICIO,
)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta a linha preta por faixa BGR.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--imagem", help="Caminho da imagem salva.")
    origem.add_argument("--camera", action="store_true", help="Captura uma imagem nova da camera CSI.")
    parser.add_argument("--salvar-tudo", action="store_true", help="Salva todas as imagens de debug.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra o debug final em uma janela.")
    parser.add_argument("--preto-topo", type=int, help="Maximo B, G e R para o topo da ROI.")
    parser.add_argument("--preto-baixo", type=int, help="Maximo B, G e R para a parte baixa da ROI.")
    parser.add_argument("--area-minima", type=float, default=AREA_MINIMA_LINHA, help="Area minima do contorno.")
    return parser.parse_args()


def mostrar_instrucoes():
    print("Informe uma imagem ou use a camera.")
    print("Exemplo: python3 raspberry/line_test.py --imagem captures/frame.jpg --salvar-tudo")
    print("Exemplo: python3 raspberry/line_test.py --camera --preto-baixo 160")


def criar_pasta_capturas():
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def carregar_imagem(caminho):
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise RuntimeError("nao foi possivel carregar a imagem. Verifique o caminho informado.")
    print(f"Imagem carregada: {caminho}")
    return imagem


def calcular_roi(imagem):
    altura, largura = imagem.shape[:2]
    x_inicio = int(largura * ROI_X_INICIO)
    x_fim = int(largura * ROI_X_FIM)
    y_inicio = int(altura * ROI_Y_INICIO)
    y_fim = int(altura * ROI_Y_FIM)
    if x_inicio >= x_fim or y_inicio >= y_fim:
        raise RuntimeError("A ROI configurada e invalida. Revise config.py.")

    roi = imagem[y_inicio:y_fim, x_inicio:x_fim]
    y_divisao = int(roi.shape[0] * DIVISAO_TOPO_BAIXO)
    if y_divisao <= 0 or y_divisao >= roi.shape[0]:
        raise RuntimeError("A divisao entre topo e baixo e invalida. Revise config.py.")
    return roi, x_inicio, x_fim, y_inicio, y_fim, y_divisao


def criar_mascara_bgr(roi, y_divisao, maximo_topo, maximo_baixo):
    roi_topo = roi[:y_divisao, :]
    roi_baixo = roi[y_divisao:, :]
    preto_min = np.array(PRETO_MIN_BGR, dtype=np.uint8)
    preto_max_topo = np.array([maximo_topo] * 3, dtype=np.uint8)
    preto_max_baixo = np.array([maximo_baixo] * 3, dtype=np.uint8)
    mascara_topo = cv2.inRange(roi_topo, preto_min, preto_max_topo)
    mascara_baixo = cv2.inRange(roi_baixo, preto_min, preto_max_baixo)

    # Futuramente, quando detectarmos verde, podemos remover a mascara verde da
    # mascara preta para evitar confundir marcador verde com linha preta.
    return np.vstack([mascara_topo, mascara_baixo])


def limpar_mascara(mascara_bruta):
    kernel = np.ones((KERNEL_LINHA, KERNEL_LINHA), np.uint8)
    mascara_limpa = cv2.erode(mascara_bruta, kernel, iterations=ERODE_INICIAL)
    mascara_limpa = cv2.dilate(mascara_limpa, kernel, iterations=DILATE_LINHA)
    return cv2.erode(mascara_limpa, kernel, iterations=ERODE_FINAL)


def encontrar_linha(mascara_limpa, area_minima):
    contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_validos = [contorno for contorno in contornos if cv2.contourArea(contorno) >= area_minima]
    if not contornos_validos:
        return None, 0.0
    maior_contorno = max(contornos_validos, key=cv2.contourArea)
    return maior_contorno, cv2.contourArea(maior_contorno)


def criar_debug_original(imagem, x_inicio, x_fim, y_inicio, y_fim, y_divisao):
    debug = imagem.copy()
    cv2.rectangle(debug, (x_inicio, y_inicio), (x_fim, y_fim), (255, 255, 0), 2)
    y_linha = y_inicio + y_divisao
    cv2.line(debug, (x_inicio, y_linha), (x_fim, y_linha), (255, 0, 255), 2)
    return debug


def criar_debug_final(imagem, contorno, area, x_inicio, x_fim, y_inicio, y_fim, y_divisao):
    debug = criar_debug_original(imagem, x_inicio, x_fim, y_inicio, y_fim, y_divisao)
    altura, largura = imagem.shape[:2]
    centro_imagem_x = largura // 2
    cv2.line(debug, (centro_imagem_x, 0), (centro_imagem_x, altura), (0, 255, 255), 2)
    resultado = {"centro_imagem_x": centro_imagem_x, "erro": None, "area": area}

    if contorno is None:
        cv2.putText(debug, "Linha nao encontrada", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return debug, resultado

    momentos = cv2.moments(contorno)
    if momentos["m00"] == 0:
        return criar_debug_final(imagem, None, area, x_inicio, x_fim, y_inicio, y_fim, y_divisao)

    centro_linha_x = x_inicio + int(momentos["m10"] / momentos["m00"])
    centro_linha_y = y_inicio + int(momentos["m01"] / momentos["m00"])
    erro = centro_linha_x - centro_imagem_x
    contorno_completo = contorno + np.array([[[x_inicio, y_inicio]]])
    cv2.drawContours(debug, [contorno_completo], -1, (0, 255, 0), 2)
    cv2.circle(debug, (centro_linha_x, centro_linha_y), 7, (0, 0, 255), -1)
    cv2.putText(debug, f"Erro: {erro}  Area: {area:.0f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    resultado.update({"centro_linha_x": centro_linha_x, "centro_linha_y": centro_linha_y, "erro": erro})
    return debug, resultado


def salvar_imagem(caminho, imagem):
    if not cv2.imwrite(str(caminho), imagem):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho}")


def salvar_debugs(pasta, timestamp, imagem, roi, mascara_bruta, mascara_limpa, debug_original, debug_final, salvar_tudo):
    caminho_final = pasta / f"debug_linha_final_{timestamp}.jpg"
    salvar_imagem(caminho_final, debug_final)
    if salvar_tudo:
        salvar_imagem(pasta / f"debug_original_roi_{timestamp}.jpg", debug_original)
        salvar_imagem(pasta / f"debug_roi_{timestamp}.jpg", roi)
        salvar_imagem(pasta / f"debug_mascara_bruta_{timestamp}.jpg", mascara_bruta)
        salvar_imagem(pasta / f"debug_mascara_limpa_{timestamp}.jpg", mascara_limpa)
    return caminho_final


def mostrar_debug(imagem):
    try:
        cv2.imshow("Debug linha - pressione q para sair", imagem)
        while cv2.waitKey(20) & 0xFF != ord("q"):
            pass
    except cv2.error as erro:
        raise RuntimeError("Preview indisponivel. Em SSH sem interface grafica, abra o arquivo salvo.") from erro
    finally:
        cv2.destroyAllWindows()


def main():
    argumentos = ler_argumentos()
    if not argumentos.imagem and not argumentos.camera:
        mostrar_instrucoes()
        return 0
    if not 0 <= argumentos.area_minima:
        print("Erro: a area minima nao pode ser negativa.")
        return 1

    maximo_topo = argumentos.preto_topo if argumentos.preto_topo is not None else PRETO_MAX_TOPO_BGR[0]
    maximo_baixo = argumentos.preto_baixo if argumentos.preto_baixo is not None else PRETO_MAX_BAIXO_BGR[0]
    if not 0 <= maximo_topo <= 255 or not 0 <= maximo_baixo <= 255:
        print("Erro: os limites de preto devem estar entre 0 e 255.")
        return 1

    camera = None
    try:
        if argumentos.camera:
            print("Capturando imagem nova da camera...")
            camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
            imagem = capturar_frame_bgr(camera)
        else:
            imagem = carregar_imagem(argumentos.imagem)

        altura, largura = imagem.shape[:2]
        roi, x_inicio, x_fim, y_inicio, y_fim, y_divisao = calcular_roi(imagem)
        mascara_bruta = criar_mascara_bgr(roi, y_divisao, maximo_topo, maximo_baixo)
        mascara_limpa = limpar_mascara(mascara_bruta)
        contorno, area = encontrar_linha(mascara_limpa, argumentos.area_minima)
        debug_original = criar_debug_original(imagem, x_inicio, x_fim, y_inicio, y_fim, y_divisao)
        debug_final, resultado = criar_debug_final(imagem, contorno, area, x_inicio, x_fim, y_inicio, y_fim, y_divisao)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_final = salvar_debugs(criar_pasta_capturas(), timestamp, imagem, roi, mascara_bruta, mascara_limpa, debug_original, debug_final, argumentos.salvar_tudo)

        print(f"Resolucao: {largura}x{altura}")
        print(f"ROI: x={x_inicio} ate {x_fim}, y={y_inicio} ate {y_fim}")
        print(f"Divisao topo/baixo na ROI: {y_divisao} px")
        print(f"Preto topo max BGR: {[maximo_topo] * 3}")
        print(f"Preto baixo max BGR: {[maximo_baixo] * 3}")
        print(f"Area maior contorno: {area:.0f}")
        print(f"Centro imagem X: {resultado['centro_imagem_x']}")
        print(f"Debug salvo: {caminho_final}")
        if contorno is None:
            print("Status: LINHA_NAO_ENCONTRADA")
            print("Possiveis solucoes: aumente --preto-baixo, --preto-topo, diminua a area minima ou ajuste a ROI.")
            return 2

        print(f"Centro linha: x={resultado['centro_linha_x']}, y={resultado['centro_linha_y']}")
        print(f"Erro: {resultado['erro']}")
        print("Status: LINHA_ENCONTRADA")
        if argumentos.mostrar:
            mostrar_debug(debug_final)
        return 0
    except RuntimeError as erro:
        print(f"Erro: {erro}")
        return 1
    finally:
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
