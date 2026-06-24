"""Executa a analise visual de verde acionavel sem controlar o robo."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import CAMERA_HEIGHT, CAMERA_WIDTH, GREEN_INTERVALO_DEBUG, PASTA_CAPTURAS
from green_action import analisar_intersecao_preta, decidir_verde_acionavel
from green_detector import criar_debug_verde, criar_mascara_linha_global, detectar_verde
from line_test import detectar_linha


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Analisa verde acionavel sem mover o robo.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--camera", action="store_true")
    origem.add_argument("--imagem")
    parser.add_argument("--salvar-debug", action="store_true")
    parser.add_argument("--mostrar", action="store_true")
    return parser.parse_args()


def processar_frame(frame):
    linha = detectar_linha(frame)
    mascara_linha = criar_mascara_linha_global(linha)
    verde = detectar_verde(frame, linha["centro_imagem_x"], mascara_linha)
    intersecao = analisar_intersecao_preta(mascara_linha, linha["centro_imagem_x"])
    acao = decidir_verde_acionavel(verde, intersecao)
    print(
        f"linha_encontrada: {linha['encontrou_linha']} | black C/L/R: "
        f"{intersecao['black_center']}/{intersecao['black_left']}/{intersecao['black_right']} | "
        f"ratio C/L/R: {intersecao['black_ratio_center']:.3f}/"
        f"{intersecao['black_ratio_left']:.3f}/{intersecao['black_ratio_right']:.3f}"
    )
    print(
        f"verde confirmado: {verde['tipo_confirmado']} | detectado: {verde['tipo_detectado']} | "
        f"confirmados: {verde['qtd_contornos_confirmados']}/{verde['qtd_contornos_detectados']}"
    )
    print(f"intersecao: {acao['tipo_intersecao']} | verde_acionavel: {acao['verde_acionavel']} | acao_visual: {acao['acao_visual']} | motivo: {acao['motivo_acao']}")
    for indice, contorno in enumerate(verde["contornos_confirmados"][:5], start=1):
        print(
            f"contorno {indice} | lado: {contorno['lado']} | acionavel: {contorno.get('acionavel', False)} | "
            f"motivo: {contorno.get('motivo_acionavel', '')} | acima: {contorno.get('ratio_acima_intersecao', 0):.2f} | "
            f"possivel_depois: {contorno.get('possivel_verde_depois_intersecao', False)}"
        )
    return verde, acao


def criar_debug_acao(frame, resultado_verde, resultado_acao):
    debug = criar_debug_verde(frame, resultado_verde)
    cores = {"zona_esquerda": (96, 96, 96), "zona_centro": (160, 160, 160), "zona_direita": (96, 96, 96)}
    for nome, cor in cores.items():
        x1, y1, x2, y2 = resultado_acao["analise_intersecao"]["zonas"][nome]
        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), cor, 1)
    for contorno in resultado_verde["contornos_confirmados"]:
        x, y, w, h = contorno["bbox"]
        cor = (0, 255, 0) if contorno.get("acionavel") and contorno["lado"] == "ESQUERDA" else (255, 0, 0) if contorno.get("acionavel") else (0, 255, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), cor, 3)
    texto = [
        f"intersecao: {resultado_acao['tipo_intersecao']} | acao: {resultado_acao['acao_visual']}",
        f"acionavel: {resultado_acao['verde_acionavel']} | {resultado_acao['motivo_acao']}",
    ]
    for indice, linha in enumerate(texto):
        cv2.putText(debug, linha, (15, 155 + indice * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(debug, linha, (15, 155 + indice * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
    return debug


def salvar_debug(frame, verde, acao):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_verde_acao_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    if not cv2.imwrite(str(caminho), criar_debug_acao(frame, verde, acao)):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho}")
    print(f"Debug salvo: {caminho}")


def main():
    argumentos = ler_argumentos()
    if not argumentos.camera and not argumentos.imagem:
        print("Use --camera ou --imagem.")
        return 0
    camera = None
    try:
        if argumentos.imagem:
            frame = cv2.imread(argumentos.imagem)
            if frame is None:
                raise RuntimeError("Nao foi possivel carregar a imagem.")
            verde, acao = processar_frame(frame)
            if argumentos.salvar_debug:
                salvar_debug(frame, verde, acao)
            if argumentos.mostrar:
                cv2.imshow("Verde acionavel", criar_debug_acao(frame, verde, acao))
                cv2.waitKey(0)
        else:
            camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
            ultimo_estado, ultimo_debug = None, 0.0
            while True:
                frame = capturar_frame_bgr(camera)
                verde, acao = processar_frame(frame)
                estado = (acao["acao_visual"], acao["verde_acionavel"])
                agora = time.monotonic()
                if argumentos.salvar_debug and estado != ultimo_estado and agora - ultimo_debug >= GREEN_INTERVALO_DEBUG:
                    salvar_debug(frame, verde, acao)
                    ultimo_estado, ultimo_debug = estado, agora
                if argumentos.mostrar:
                    cv2.imshow("Verde acionavel - q para sair", criar_debug_acao(frame, verde, acao))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        return 0
    except (RuntimeError, ValueError, cv2.error) as erro:
        print(f"Erro: {erro}")
        return 1
    finally:
        if camera is not None:
            camera.stop()
        if argumentos.mostrar:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
