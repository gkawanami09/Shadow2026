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
from config import (Y_TRAVA_BAIXA_INICIO, Y_TRAVA_BAIXA_FIM,
    MIN_PIXELS_LINHA_SEGURA_BAIXA, MIN_PIXELS_LINHA_FRACA_BAIXA,
    LIMIAR_ERRO_BAIXO_RISCO_PERDA)
from config import (
    NUM_FAIXAS_CAMINHO, MIN_PIXELS_FAIXA_CAMINHO, DISTANCIA_MAX_ENTRE_FAIXAS,
    Y_CONTROLE_PERTO_INICIO, Y_CONTROLE_PERTO_FIM, Y_LOOKAHEAD_INICIO,
    Y_LOOKAHEAD_FIM, MIN_PIXELS_LINHA_PERTO, KP_LATERAL, KP_DIRECAO,
    CORRECAO_MAXIMA_VETOR, VELOCIDADE_BASE_VETOR, VELOCIDADE_MINIMA_VETOR,
    VELOCIDADE_MAXIMA_VETOR, VELOCIDADE_BASE_CURVA_FORTE,
    LIMIAR_DIRECAO_CURVA_FORTE, REDUZIR_VELOCIDADE_EM_CURVA,
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


def extrair_caminho_linha(mascara_limpa, x_inicio_roi, y_inicio_roi, centro_imagem_x):
    """Reconstrói o caminho da linha, da parte baixa da ROI para o topo."""
    altura = mascara_limpa.shape[0]
    pontos, x_anterior = [], None
    for indice in range(NUM_FAIXAS_CAMINHO - 1, -1, -1):
        y1, y2 = int(altura * indice / NUM_FAIXAS_CAMINHO), int(altura * (indice + 1) / NUM_FAIXAS_CAMINHO)
        faixa = mascara_limpa[y1:y2, :]
        contornos, _ = cv2.findContours(faixa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        for contorno in contornos:
            pixels = int(cv2.contourArea(contorno))
            if pixels >= MIN_PIXELS_FAIXA_CAMINHO:
                momentos = cv2.moments(contorno)
                if momentos["m00"]:
                    candidatos.append((int(momentos["m10"] / momentos["m00"]), pixels))
        escolhido = min(candidatos, key=lambda item: abs(item[0] - x_anterior)) if candidatos and x_anterior is not None else (candidatos[0] if candidatos else None)
        ponto = {"x": None, "y": y_inicio_roi + (y1 + y2) // 2, "erro": None, "pixels": 0, "encontrou": False}
        if escolhido and (x_anterior is None or abs(escolhido[0] - x_anterior) <= DISTANCIA_MAX_ENTRE_FAIXAS):
            ponto.update({"x": x_inicio_roi + escolhido[0], "erro": x_inicio_roi + escolhido[0] - centro_imagem_x, "pixels": escolhido[1], "encontrou": True})
            x_anterior = escolhido[0]
        pontos.append(ponto)
    return pontos


def calcular_controle_vetor(pontos):
    perto = [p for p in pontos if p["encontrou"] and Y_CONTROLE_PERTO_INICIO <= (p["y"] - pontos[-1]["y"] + 1) / max(1, pontos[0]["y"] - pontos[-1]["y"] + 1) <= Y_CONTROLE_PERTO_FIM]
    validos = [p for p in pontos if p["encontrou"]]
    if not validos:
        return None
    ponto_perto = validos[0]
    ponto_lookahead = validos[min(len(validos) - 1, max(1, len(validos) // 2))]
    erro_lateral = sum(p["erro"] for p in (perto or [ponto_perto])) / len(perto or [ponto_perto])
    erro_direcao = (ponto_lookahead["x"] - ponto_perto["x"]) / max(ponto_perto["y"] - ponto_lookahead["y"], 1) * 100
    base = VELOCIDADE_BASE_CURVA_FORTE if REDUZIR_VELOCIDADE_EM_CURVA and abs(erro_direcao) >= LIMIAR_DIRECAO_CURVA_FORTE else VELOCIDADE_BASE_VETOR
    correcao = max(-CORRECAO_MAXIMA_VETOR, min(CORRECAO_MAXIMA_VETOR, KP_LATERAL * erro_lateral + KP_DIRECAO * erro_direcao))
    esq, dir = round(max(VELOCIDADE_MINIMA_VETOR, min(VELOCIDADE_MAXIMA_VETOR, base + correcao))), round(max(VELOCIDADE_MINIMA_VETOR, min(VELOCIDADE_MAXIMA_VETOR, base - correcao)))
    return {"erro_lateral": erro_lateral, "erro_direcao": erro_direcao, "correcao": correcao, "comando": f"LADO {esq} {dir}", "ponto_perto": ponto_perto, "ponto_lookahead": ponto_lookahead, "linha_baixa": bool(perto)}


def medir_linha_baixa(mascara_limpa, x_inicio_roi, centro_imagem_x):
    altura = mascara_limpa.shape[0]
    y1, y2 = int(altura * Y_TRAVA_BAIXA_INICIO), int(altura * Y_TRAVA_BAIXA_FIM)
    _, xs = np.where(mascara_limpa[y1:y2, :] > 0)
    pixels = len(xs)
    centro_x = x_inicio_roi + int(np.mean(xs)) if pixels else None
    erro = centro_x - centro_imagem_x if centro_x is not None else None
    segura = pixels >= MIN_PIXELS_LINHA_SEGURA_BAIXA
    fraca = MIN_PIXELS_LINHA_FRACA_BAIXA <= pixels < MIN_PIXELS_LINHA_SEGURA_BAIXA
    risco = not segura or (erro is not None and abs(erro) >= LIMIAR_ERRO_BAIXO_RISCO_PERDA)
    return {"encontrou": pixels > 0, "pixels": pixels, "centro_x": centro_x, "erro": erro, "segura": segura, "fraca": fraca, "risco_perda": risco}


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def calcular_comando(erro_final, kp, velocidade_base, correcao_maxima=CORRECAO_MAXIMA, velocidade_minima=VELOCIDADE_MINIMA_SEGUE_LINHA, velocidade_maxima=VELOCIDADE_MAXIMA_SEGUE_LINHA):
    """Calcula LADO com limites ajustaveis para normal ou curva fechada."""
    correcao = limitar(kp * erro_final, -correcao_maxima, correcao_maxima)
    esquerda = round(limitar(velocidade_base + correcao, velocidade_minima, velocidade_maxima))
    direita = round(limitar(velocidade_base - correcao, velocidade_minima, velocidade_maxima))
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
