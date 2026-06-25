"""Detecta linha preta por BGR e disponibiliza os dados para outros testes."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    AREA_MINIMA_LINHA, CAMERA_HEIGHT, CAMERA_WIDTH, DILATE_LINHA,
    DIVISAO_TOPO_BAIXO, ERODE_FINAL, ERODE_INICIAL, KERNEL_LINHA,
    PASTA_CAPTURAS, PRETO_MAX_BAIXO_BGR, PRETO_MAX_TOPO_BGR,
    PRETO_MIN_BGR, ROI_X_FIM, ROI_X_INICIO, ROI_Y_FIM, ROI_Y_INICIO,
)


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
    preto_min = np.array(PRETO_MIN_BGR, dtype=np.uint8)
    mascara_topo = cv2.inRange(roi[:y_divisao, :], preto_min, np.array([maximo_topo] * 3, dtype=np.uint8))
    mascara_baixo = cv2.inRange(roi[y_divisao:, :], preto_min, np.array([maximo_baixo] * 3, dtype=np.uint8))
    # Futuramente, a mascara verde pode ser removida daqui antes dos contornos.
    return np.vstack([mascara_topo, mascara_baixo])


def limpar_mascara(mascara_bruta):
    kernel = np.ones((KERNEL_LINHA, KERNEL_LINHA), np.uint8)
    mascara = cv2.erode(mascara_bruta, kernel, iterations=ERODE_INICIAL)
    mascara = cv2.dilate(mascara, kernel, iterations=DILATE_LINHA)
    return cv2.erode(mascara, kernel, iterations=ERODE_FINAL)


def detectar_linha(imagem, salvar_debug=False, prefixo_debug="linha", preto_topo=None, preto_baixo=None, area_minima=None):
    """Retorna dados da linha preta detectada em uma imagem BGR."""
    maximo_topo = PRETO_MAX_TOPO_BGR[0] if preto_topo is None else preto_topo
    maximo_baixo = PRETO_MAX_BAIXO_BGR[0] if preto_baixo is None else preto_baixo
    area_minima = AREA_MINIMA_LINHA if area_minima is None else area_minima
    if not 0 <= maximo_topo <= 255 or not 0 <= maximo_baixo <= 255:
        raise RuntimeError("Os limites de preto devem estar entre 0 e 255.")

    roi, x_inicio, x_fim, y_inicio, y_fim, y_divisao = calcular_roi(imagem)
    mascara_bruta = criar_mascara_bgr(roi, y_divisao, maximo_topo, maximo_baixo)
    mascara_limpa = limpar_mascara(mascara_bruta)
    contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = [contorno for contorno in contornos if cv2.contourArea(contorno) >= area_minima]
    contorno = max(validos, key=cv2.contourArea) if validos else None
    area = cv2.contourArea(contorno) if contorno is not None else 0.0
    altura, largura = imagem.shape[:2]
    resultado = {
        "encontrou_linha": False, "imagem_original": imagem, "roi": roi,
        "mascara_bruta": mascara_bruta, "mascara_limpa": mascara_limpa,
        "contorno_principal": contorno, "x_inicio_roi": x_inicio,
        "x_fim_roi": x_fim, "y_inicio_roi": y_inicio, "y_fim_roi": y_fim,
        "y_divisao_roi": y_divisao, "largura": largura, "altura": altura,
        "centro_imagem_x": largura // 2, "centro_linha_x": None,
        "centro_linha_y": None, "erro": None, "area": area,
        "preto_topo": maximo_topo, "preto_baixo": maximo_baixo,
    }
    if contorno is not None:
        momentos = cv2.moments(contorno)
        if momentos["m00"] != 0:
            resultado["encontrou_linha"] = True
            resultado["centro_linha_x"] = x_inicio + int(momentos["m10"] / momentos["m00"])
            resultado["centro_linha_y"] = y_inicio + int(momentos["m01"] / momentos["m00"])
            resultado["erro"] = resultado["centro_linha_x"] - resultado["centro_imagem_x"]
    if salvar_debug:
        salvar_debug_linha(resultado, prefixo_debug, salvar_tudo=True)
    return resultado


def criar_debug_linha(resultado):
    debug = resultado["imagem_original"].copy()
    x_inicio, x_fim = resultado["x_inicio_roi"], resultado["x_fim_roi"]
    y_inicio, y_fim = resultado["y_inicio_roi"], resultado["y_fim_roi"]
    cv2.rectangle(debug, (x_inicio, y_inicio), (x_fim, y_fim), (255, 255, 0), 2)
    cv2.line(debug, (x_inicio, y_inicio + resultado["y_divisao_roi"]), (x_fim, y_inicio + resultado["y_divisao_roi"]), (255, 0, 255), 2)
    centro_x = resultado["centro_imagem_x"]
    cv2.line(debug, (centro_x, 0), (centro_x, resultado["altura"]), (0, 255, 255), 2)
    if resultado["encontrou_linha"]:
        contorno = resultado["contorno_principal"] + np.array([[[x_inicio, y_inicio]]])
        cv2.drawContours(debug, [contorno], -1, (0, 255, 0), 2)
        ponto = (resultado["centro_linha_x"], resultado["centro_linha_y"])
        cv2.circle(debug, ponto, 7, (0, 0, 255), -1)
        texto = f"Erro: {resultado['erro']} Area: {resultado['area']:.0f}"
    else:
        texto = "Linha nao encontrada"
    cv2.putText(debug, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return debug


def salvar_imagem(caminho, imagem):
    if not cv2.imwrite(str(caminho), imagem):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho}")


def salvar_debug_linha(resultado, prefixo="linha", salvar_tudo=False):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = pasta / f"debug_{prefixo}_final_{timestamp}.jpg"
    salvar_imagem(final, criar_debug_linha(resultado))
    if salvar_tudo:
        salvar_imagem(pasta / f"debug_{prefixo}_roi_{timestamp}.jpg", resultado["roi"])
        salvar_imagem(pasta / f"debug_{prefixo}_mascara_bruta_{timestamp}.jpg", resultado["mascara_bruta"])
        salvar_imagem(pasta / f"debug_{prefixo}_mascara_limpa_{timestamp}.jpg", resultado["mascara_limpa"])
    return final


def carregar_imagem(caminho):
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise RuntimeError("nao foi possivel carregar a imagem. Verifique o caminho informado.")
    print(f"Imagem carregada: {caminho}")
    return imagem


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta a linha preta por faixa BGR.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--imagem", help="Caminho da imagem salva.")
    origem.add_argument("--camera", action="store_true", help="Captura uma imagem nova da camera CSI.")
    parser.add_argument("--salvar-tudo", action="store_true", help="Salva todas as imagens de debug.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra o debug final.")
    parser.add_argument("--preto-topo", type=int, help="Maximo B, G e R para o topo.")
    parser.add_argument("--preto-baixo", type=int, help="Maximo B, G e R para a parte baixa.")
    parser.add_argument("--area-minima", type=float, default=AREA_MINIMA_LINHA, help="Area minima do contorno.")
    return parser.parse_args()


def main():
    argumentos = ler_argumentos()
    if not argumentos.imagem and not argumentos.camera:
        print("Use --imagem captures/frame.jpg ou --camera.")
        return 0
    camera = None
    try:
        if argumentos.camera:
            camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
            imagem = capturar_frame_bgr(camera)
        else:
            imagem = carregar_imagem(argumentos.imagem)
        resultado = detectar_linha(imagem, preto_topo=argumentos.preto_topo, preto_baixo=argumentos.preto_baixo, area_minima=argumentos.area_minima)
        caminho_debug = salvar_debug_linha(resultado, "linha", argumentos.salvar_tudo)
        print(f"Resolucao: {resultado['largura']}x{resultado['altura']}")
        print(f"Area maior contorno: {resultado['area']:.0f}")
        print(f"Debug salvo: {caminho_debug}")
        if not resultado["encontrou_linha"]:
            print("Status: LINHA_NAO_ENCONTRADA")
            return 2
        print(f"Centro linha: x={resultado['centro_linha_x']}, y={resultado['centro_linha_y']}")
        print(f"Erro: {resultado['erro']}")
        print("Status: LINHA_ENCONTRADA")
        if argumentos.mostrar:
            cv2.imshow("Debug linha - pressione uma tecla", criar_debug_linha(resultado))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0
    except RuntimeError as erro:
        print(f"Erro: {erro}")
        return 1
    finally:
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
