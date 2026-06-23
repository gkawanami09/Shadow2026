"""Calcula um comando de segue-linha, mas nunca aciona motores."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    CAMERA_HEIGHT, CAMERA_WIDTH, CORRECAO_MAXIMA, FAIXA_ALTA_FIM,
    FAIXA_ALTA_INICIO, FAIXA_BAIXA_FIM, FAIXA_BAIXA_INICIO,
    FAIXA_MEDIA_FIM, FAIXA_MEDIA_INICIO, KP_SEGUE_LINHA, PASTA_CAPTURAS,
    PESO_FAIXA_ALTA, PESO_FAIXA_BAIXA, PESO_FAIXA_MEDIA,
    VELOCIDADE_BASE_SEGUE_LINHA, VELOCIDADE_MAXIMA_SEGUE_LINHA,
    VELOCIDADE_MINIMA_SEGUE_LINHA,
)
from line_test import carregar_imagem, detectar_linha


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Sugere comando de segue-linha sem usar motores.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--imagem", help="Caminho da imagem salva.")
    origem.add_argument("--camera", action="store_true", help="Captura uma imagem nova da camera CSI.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva a imagem de debug.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra a imagem de debug.")
    parser.add_argument("--kp", type=float, default=KP_SEGUE_LINHA, help="Ganho proporcional.")
    parser.add_argument("--velocidade-base", type=int, default=VELOCIDADE_BASE_SEGUE_LINHA, help="Velocidade base sugerida.")
    return parser.parse_args()


def calcular_erro_faixa(mascara, x_inicio_roi, centro_imagem_x, inicio, fim):
    altura = mascara.shape[0]
    y_inicio = int(altura * inicio)
    y_fim = int(altura * fim)
    faixa = mascara[y_inicio:y_fim, :]
    _, xs = np.where(faixa > 0)
    quantidade = len(xs)
    resultado = {"encontrou": quantidade > 0, "centro_x": None, "erro": None, "quantidade_pixels": quantidade, "y_inicio": y_inicio, "y_fim": y_fim}
    if quantidade > 0:
        resultado["centro_x"] = x_inicio_roi + int(np.mean(xs))
        resultado["erro"] = resultado["centro_x"] - centro_imagem_x
    return resultado


def calcular_erro_final(faixas):
    pesos = {"baixa": PESO_FAIXA_BAIXA, "media": PESO_FAIXA_MEDIA, "alta": PESO_FAIXA_ALTA}
    encontradas = [nome for nome, faixa in faixas.items() if faixa["encontrou"]]
    if not encontradas:
        return None
    soma_pesos = sum(pesos[nome] for nome in encontradas)
    return sum(faixas[nome]["erro"] * pesos[nome] for nome in encontradas) / soma_pesos


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def calcular_comando(erro_final, kp, velocidade_base):
    correcao = limitar(kp * erro_final, -CORRECAO_MAXIMA, CORRECAO_MAXIMA)
    esquerda = round(limitar(velocidade_base + correcao, VELOCIDADE_MINIMA_SEGUE_LINHA, VELOCIDADE_MAXIMA_SEGUE_LINHA))
    direita = round(limitar(velocidade_base - correcao, VELOCIDADE_MINIMA_SEGUE_LINHA, VELOCIDADE_MAXIMA_SEGUE_LINHA))
    return correcao, esquerda, direita, f"LADO {esquerda} {direita}"


def criar_debug_follow(resultado_linha, faixas, erro_final, comando, velocidade_base, kp):
    debug = resultado_linha["imagem_original"].copy()
    x_inicio, x_fim = resultado_linha["x_inicio_roi"], resultado_linha["x_fim_roi"]
    y_inicio, y_fim = resultado_linha["y_inicio_roi"], resultado_linha["y_fim_roi"]
    cores = {"baixa": (0, 255, 0), "media": (255, 0, 0), "alta": (255, 0, 255)}
    cv2.rectangle(debug, (x_inicio, y_inicio), (x_fim, y_fim), (255, 255, 0), 2)
    for nome, faixa in faixas.items():
        y1 = y_inicio + faixa["y_inicio"]
        y2 = y_inicio + faixa["y_fim"]
        cv2.rectangle(debug, (x_inicio, y1), (x_fim, y2), cores[nome], 1)
        if faixa["encontrou"]:
            cv2.circle(debug, (faixa["centro_x"], (y1 + y2) // 2), 6, cores[nome], -1)
    centro_imagem = resultado_linha["centro_imagem_x"]
    cv2.line(debug, (centro_imagem, 0), (centro_imagem, resultado_linha["altura"]), (0, 255, 255), 2)
    linhas = [
        f"Baixa: {faixas['baixa']['erro']}", f"Media: {faixas['media']['erro']}",
        f"Alta: {faixas['alta']['erro']}", f"Base: {velocidade_base}  KP: {kp:.2f}",
        f"Erro final: {erro_final:.1f}",
        f"{comando} - SEM MOTOR",
    ]
    for indice, texto in enumerate(linhas):
        cv2.putText(debug, texto, (15, 28 + indice * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return debug


def salvar_debug(imagem):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_follow_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    if not cv2.imwrite(str(caminho), imagem):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho}")
    return caminho


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
            print("Imagem: camera")
        else:
            imagem = carregar_imagem(argumentos.imagem)
        resultado = detectar_linha(imagem)
        if not resultado["encontrou_linha"]:
            print("Status linha: LINHA_NAO_ENCONTRADA")
            print("Modo: SIMULACAO_SEM_MOTOR")
            print("Comando sugerido: PARAR")
            print("MOTORES NAO ACIONADOS")
            return 2
        mascara = resultado["mascara_limpa"]
        faixas = {
            "baixa": calcular_erro_faixa(mascara, resultado["x_inicio_roi"], resultado["centro_imagem_x"], FAIXA_BAIXA_INICIO, FAIXA_BAIXA_FIM),
            "media": calcular_erro_faixa(mascara, resultado["x_inicio_roi"], resultado["centro_imagem_x"], FAIXA_MEDIA_INICIO, FAIXA_MEDIA_FIM),
            "alta": calcular_erro_faixa(mascara, resultado["x_inicio_roi"], resultado["centro_imagem_x"], FAIXA_ALTA_INICIO, FAIXA_ALTA_FIM),
        }
        erro_final = calcular_erro_final(faixas)
        if erro_final is None:
            print("Status linha: LINHA_NAO_ENCONTRADA")
            print("Modo: SIMULACAO_SEM_MOTOR")
            print("Comando sugerido: PARAR")
            print("MOTORES NAO ACIONADOS")
            return 2
        correcao_bruta = argumentos.kp * erro_final
        correcao, esquerda, direita, comando = calcular_comando(erro_final, argumentos.kp, argumentos.velocidade_base)
        debug = criar_debug_follow(resultado, faixas, erro_final, comando, argumentos.velocidade_base, argumentos.kp)
        print("Status linha: LINHA_ENCONTRADA")
        print("Modo: SIMULACAO_SEM_MOTOR")
        for nome in ("baixa", "media", "alta"):
            print(f"{nome.capitalize()}: {faixas[nome]['erro']}")
        print(f"Erro final: {erro_final:.2f}")
        print(f"Velocidade base: {argumentos.velocidade_base}")
        print(f"KP: {argumentos.kp}")
        print(f"Correcao: {correcao_bruta:.2f}")
        print(f"Correcao limitada: {correcao:.2f}")
        print(f"Velocidade esquerda: {esquerda}")
        print(f"Velocidade direita: {direita}")
        print(f"Comando sugerido: {comando}")
        print("MOTORES NAO ACIONADOS")
        if argumentos.salvar_debug:
            print(f"Debug salvo em: {salvar_debug(debug)}")
        if argumentos.mostrar:
            cv2.imshow("Debug follow - pressione uma tecla", debug)
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
