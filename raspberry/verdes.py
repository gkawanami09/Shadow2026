"""Detector visual de marcadores verdes para debug da OBR.

Este arquivo nao move motores, nao abre Serial e nao altera o segue-linha.
Ele analisa apenas imagem/camera e retorna a decisao visual de verde.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_test import capturar_frame_bgr, iniciar_camera
from config import CAMERA_HEIGHT, CAMERA_WIDTH, PASTA_CAPTURAS
from line_test import detectar_linha


VERDE_H_MIN = 40
VERDE_H_MAX = 85
VERDE_S_MIN = 70
VERDE_V_MIN = 45
VERDE_G_MENOS_R_MIN = 20
VERDE_G_MENOS_B_MIN = 15

KERNEL_VERDE = 5
OPEN_VERDE = 1
CLOSE_VERDE = 2

VERDE_Y_MIN_REL = 0.30
VERDE_Y_MAX_REL = 0.95
VERDE_AREA_MIN_LONGE = 35
VERDE_AREA_MIN_PERTO = 120
VERDE_AREA_MAX_REL_QUADRO = 0.25
VERDE_AREA_ABSURDA_REL_QUADRO = 0.45
VERDE_ASPECTO_MIN = 0.45
VERDE_ASPECTO_MAX = 2.20
VERDE_FILL_RATIO_MIN = 0.25
VERDE_PROPORCAO_MIN = 1.12
VERDE_MARGEM_DEPOIS_MIN_PX = 6
VERDE_MARGEM_DEPOIS_MAX_PX = 16
VERDE_MARGEM_DEPOIS_MULT_ALTURA = 0.08
VERDE_FRAC_MIN_BBOX_ANTES = 0.45

PRETO_MARGEM_VERDE_MULT = 0.80
PRETO_MIN_PIXELS_ABS = 10
PRETO_MIN_PIXELS_REL = 0.10
PRETO_MIN_LADOS_AO_REDOR = 1

MARGEM_CENTRO_LINHA_MULT = 0.80
BUSCA_LINHA_DY = (0, 10, 20, -10, -20, 35, -35)

AREA_MIN_VERDE_ABSOLUTA = 20
LARGURA_MIN_VERDE = 4
ALTURA_MIN_VERDE = 4

MULT_LARGURA_CRUZAMENTO = 2.6
LARGURA_LINHA_FALLBACK = 30.0
INTERVALO_DEBUG_CAMERA = 0.60


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def encontrar_segmentos(linha_binaria):
    segmentos = []
    em_segmento = False
    inicio = 0
    for indice, valor in enumerate(linha_binaria):
        if valor and not em_segmento:
            inicio = indice
            em_segmento = True
        elif not valor and em_segmento:
            segmentos.append((inicio, indice - 1))
            em_segmento = False
    if em_segmento:
        segmentos.append((inicio, len(linha_binaria) - 1))
    return segmentos


def distancia_caixas(caixa_a, caixa_b):
    ax, ay, aw, ah = caixa_a
    bx, by, bw, bh = caixa_b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0)
    dy = max(by - (ay + ah), ay - (by + bh), 0)
    return (dx * dx + dy * dy) ** 0.5


def contar_pixels_regiao(mascara, x1, y1, x2, y2):
    altura, largura = mascara.shape[:2]
    x1 = int(limitar(x1, 0, largura - 1))
    x2 = int(limitar(x2, 1, largura))
    y1 = int(limitar(y1, 0, altura - 1))
    y2 = int(limitar(y2, 1, altura))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(cv2.countNonZero(mascara[y1:y2, x1:x2]))


def area_minima_verde_por_y(cy, altura):
    y_rel = cy / max(altura, 1)
    faixa = max(VERDE_Y_MAX_REL - VERDE_Y_MIN_REL, 1e-6)
    t = limitar((y_rel - VERDE_Y_MIN_REL) / faixa, 0.0, 1.0)
    return VERDE_AREA_MIN_LONGE + t * (VERDE_AREA_MIN_PERTO - VERDE_AREA_MIN_LONGE)


def roi_preto_tem_linha(mascara_linha, x1, y1, x2, y2):
    altura, largura = mascara_linha.shape[:2]
    x1 = int(limitar(x1, 0, largura - 1))
    x2 = int(limitar(x2, 1, largura))
    y1 = int(limitar(y1, 0, altura - 1))
    y2 = int(limitar(y2, 1, altura))
    if x2 <= x1 or y2 <= y1:
        return False, 0, (x1, y1, x2, y2)

    area_roi = max((x2 - x1) * (y2 - y1), 1)
    pixels = int(cv2.countNonZero(mascara_linha[y1:y2, x1:x2]))
    minimo = max(PRETO_MIN_PIXELS_ABS, int(area_roi * PRETO_MIN_PIXELS_REL))
    return pixels >= minimo, pixels, (x1, y1, x2, y2)


def criar_mascara_linha_global(resultado_linha):
    altura = resultado_linha["altura"]
    largura = resultado_linha["largura"]
    mascara_linha = np.zeros((altura, largura), dtype=np.uint8)
    x_inicio = resultado_linha["x_inicio_roi"]
    x_fim = resultado_linha["x_fim_roi"]
    y_inicio = resultado_linha["y_inicio_roi"]
    y_fim = resultado_linha["y_fim_roi"]
    mascara_linha[y_inicio:y_fim, x_inicio:x_fim] = resultado_linha["mascara_limpa"]
    return mascara_linha


def criar_mascara_verde(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    minimo = np.array([VERDE_H_MIN, VERDE_S_MIN, VERDE_V_MIN], dtype=np.uint8)
    maximo = np.array([VERDE_H_MAX, 255, 255], dtype=np.uint8)
    mascara_hsv = cv2.inRange(hsv, minimo, maximo)

    b, g, r = cv2.split(frame_bgr)
    mascara_dominancia = (
        (g.astype(np.int16) - r.astype(np.int16) >= VERDE_G_MENOS_R_MIN)
        & (g.astype(np.int16) - b.astype(np.int16) >= VERDE_G_MENOS_B_MIN)
    ).astype(np.uint8) * 255

    mascara = cv2.bitwise_and(mascara_hsv, mascara_dominancia)
    kernel = np.ones((KERNEL_VERDE, KERNEL_VERDE), np.uint8)
    if OPEN_VERDE > 0:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=OPEN_VERDE)
    if CLOSE_VERDE > 0:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_VERDE)
    return mascara


def remover_verde_da_mascara_linha(mascara_linha, mascara_verde):
    """Remove pixels verdes da mascara da linha para evitar falso preto no marcador."""
    kernel = np.ones((5, 5), np.uint8)
    mascara_verde_expandida = cv2.dilate(mascara_verde, kernel, iterations=1)
    mascara_verde_invertida = cv2.bitwise_not(mascara_verde_expandida)
    return cv2.bitwise_and(mascara_linha, mascara_verde_invertida)


def encontrar_candidatos_verdes(frame_bgr, mascara_verde):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    contornos, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        x, y, w, h = cv2.boundingRect(contorno)
        if area < AREA_MIN_VERDE_ABSOLUTA or w < LARGURA_MIN_VERDE or h < ALTURA_MIN_VERDE:
            continue

        regiao_mascara = mascara_verde[y:y + h, x:x + w]
        regiao_hsv = hsv[y:y + h, x:x + w]
        regiao_bgr = frame_bgr[y:y + h, x:x + w]
        pixels_verdes = regiao_mascara > 0
        if not np.any(pixels_verdes):
            continue

        media_hsv = cv2.mean(regiao_hsv, mask=regiao_mascara)
        media_bgr = cv2.mean(regiao_bgr, mask=regiao_mascara)
        fill_ratio = float(cv2.countNonZero(regiao_mascara)) / max(w * h, 1)
        mean_b, mean_g, mean_r = media_bgr[:3]
        candidato = {
            "bbox": (int(x), int(y), int(w), int(h)),
            "centro": (int(x + w / 2), int(y + h / 2)),
            "area": area,
            "fill_ratio": fill_ratio,
            "mean_h": float(media_hsv[0]),
            "mean_s": float(media_hsv[1]),
            "mean_v": float(media_hsv[2]),
            "mean_b": float(mean_b),
            "mean_g": float(mean_g),
            "mean_r": float(mean_r),
            "g_menos_r": float(mean_g - mean_r),
            "g_menos_b": float(mean_g - mean_b),
            "proporcao_verde": float(mean_g / max(mean_r, mean_b, 1.0)),
            "lado": "CENTRO",
            "valido": False,
            "motivo": "nao_validado",
            "confianca": 0.0,
            "falso_depois_cruzamento": False,
        }
        candidatos.append(candidato)
    return candidatos


def estimar_largura_linha(mascara_linha):
    altura, largura = mascara_linha.shape[:2]
    larguras_validas = []
    for y in np.linspace(int(altura * 0.55), int(altura * 0.90), 12).astype(int):
        y1 = limitar(y - 3, 0, altura - 1)
        y2 = limitar(y + 4, 1, altura)
        faixa = mascara_linha[y1:y2, :]
        linha = cv2.reduce(faixa, 0, cv2.REDUCE_MAX).flatten() > 0
        segmentos = encontrar_segmentos(linha)
        if not segmentos:
            continue
        centro = largura / 2
        segmento = min(segmentos, key=lambda seg: abs(((seg[0] + seg[1]) / 2) - centro))
        largura_segmento = segmento[1] - segmento[0] + 1
        if 4 <= largura_segmento <= largura * 0.35:
            larguras_validas.append(largura_segmento)
    if not larguras_validas:
        return LARGURA_LINHA_FALLBACK
    return float(np.median(larguras_validas))


def encontrar_segmento_linha_por_y(mascara_linha, y, x_alvo=None):
    altura, largura = mascara_linha.shape[:2]
    y = int(limitar(y, 0, altura - 1))
    y1 = limitar(y - 5, 0, altura - 1)
    y2 = limitar(y + 6, 1, altura)
    faixa = mascara_linha[y1:y2, :]
    linha = cv2.reduce(faixa, 0, cv2.REDUCE_MAX).flatten() > 0
    segmentos = encontrar_segmentos(linha)
    if not segmentos:
        return None
    if x_alvo is None:
        x_alvo = largura / 2
    segmento = min(segmentos, key=lambda seg: abs(((seg[0] + seg[1]) / 2) - x_alvo))
    centro = int((segmento[0] + segmento[1]) / 2)
    largura_segmento = int(segmento[1] - segmento[0] + 1)
    return centro, largura_segmento, int(y)


def encontrar_centro_linha_por_y(mascara_linha, y):
    segmento = encontrar_segmento_linha_por_y(mascara_linha, y)
    if segmento is None:
        return None
    return segmento[0]


def encontrar_referencia_linha_local(mascara_linha, cx, cy):
    for dy in BUSCA_LINHA_DY:
        segmento = encontrar_segmento_linha_por_y(mascara_linha, cy + dy, x_alvo=cx)
        if segmento is not None:
            x_linha, largura_linha, y_linha = segmento
            if 3 <= largura_linha <= mascara_linha.shape[1] * 0.45:
                return {
                    "x_linha": int(x_linha),
                    "y_linha": int(y_linha),
                    "largura_linha_px": float(largura_linha),
                }
    return None


def cruzamento_pode_ser_referencia(cruzamento, largura_imagem):
    if not cruzamento.get("detectado") or cruzamento.get("centro") is None:
        return False

    cx_cruz, _ = cruzamento["centro"]
    if cx_cruz < largura_imagem * 0.15 or cx_cruz > largura_imagem * 0.85:
        return False

    return bool(cruzamento.get("linha_baixo", False))


def obter_referencia_lado_verde(candidato, cruzamento, mascara_linha, largura_linha_px):
    _, cy = candidato["centro"]
    _, largura = mascara_linha.shape[:2]

    if cruzamento_pode_ser_referencia(cruzamento, largura):
        return {
            "x_linha": int(cruzamento["centro"][0]),
            "y_linha": int(cruzamento["centro"][1]),
            "largura_linha_px": float(cruzamento.get("largura_linha_px", largura_linha_px)),
            "origem": "cruzamento",
            "confianca_referencia": 1.0,
        }

    x_alvo_principal = largura / 2
    for dy in BUSCA_LINHA_DY:
        segmento = encontrar_segmento_linha_por_y(
            mascara_linha,
            cy + dy,
            x_alvo=x_alvo_principal,
        )
        if segmento is None:
            continue

        x_linha, largura_linha, y_linha = segmento
        if 3 <= largura_linha <= largura * 0.45:
            return {
                "x_linha": int(x_linha),
                "y_linha": int(y_linha),
                "largura_linha_px": float(largura_linha),
                "origem": "linha_local_centro",
                "confianca_referencia": 0.75,
            }

    return {
        "x_linha": int(largura / 2),
        "y_linha": int(cy),
        "largura_linha_px": float(largura_linha_px),
        "origem": "centro_imagem_fallback",
        "confianca_referencia": 0.35,
    }


def analisar_cruzamento(mascara_linha):
    altura, largura = mascara_linha.shape[:2]
    largura_linha_px = estimar_largura_linha(mascara_linha)
    melhor = None

    for y in np.linspace(int(altura * 0.25), int(altura * 0.82), 32).astype(int):
        x_linha = encontrar_centro_linha_por_y(mascara_linha, y + int(largura_linha_px * 1.5))
        if x_linha is None:
            x_linha = encontrar_centro_linha_por_y(mascara_linha, y)
        if x_linha is None:
            continue

        corredor = max(12, largura_linha_px * 1.4)
        altura_faixa = max(10, int(largura_linha_px * 0.5))
        alcance_lateral = max(corredor * 4.5, largura_linha_px * MULT_LARGURA_CRUZAMENTO)
        alcance_vertical = max(largura_linha_px * 4.0, 45)

        y1 = y - altura_faixa
        y2 = y + altura_faixa
        x_centro_1 = x_linha - corredor / 2
        x_centro_2 = x_linha + corredor / 2

        pixels_esquerda = contar_pixels_regiao(
            mascara_linha,
            x_linha - alcance_lateral,
            y1,
            x_centro_1,
            y2,
        )
        pixels_direita = contar_pixels_regiao(
            mascara_linha,
            x_centro_2,
            y1,
            x_linha + alcance_lateral,
            y2,
        )
        pixels_frente = contar_pixels_regiao(
            mascara_linha,
            x_linha - corredor / 2,
            y - alcance_vertical,
            x_linha + corredor / 2,
            y - largura_linha_px * 0.8,
        )
        pixels_baixo = contar_pixels_regiao(
            mascara_linha,
            x_linha - corredor / 2,
            y + largura_linha_px * 0.8,
            x_linha + corredor / 2,
            y + alcance_vertical,
        )

        min_lateral = max(20, largura_linha_px * 1.0)
        min_vertical = max(20, largura_linha_px * 1.0)
        ramo_esquerda = pixels_esquerda >= min_lateral
        ramo_direita = pixels_direita >= min_lateral
        ramo_frente = pixels_frente >= min_vertical
        linha_baixo = pixels_baixo >= min_vertical

        tem_lateral = ramo_esquerda or ramo_direita
        tem_vertical = linha_baixo and ramo_frente
        if not (tem_lateral and tem_vertical):
            continue

        soma_lateral = pixels_esquerda + pixels_direita
        confianca = limitar(
            0.35
            + 0.20 * (soma_lateral / max(largura_linha_px * 8.0, 1))
            + (0.15 if ramo_esquerda else 0.0)
            + (0.15 if ramo_direita else 0.0)
            + (0.15 if ramo_frente else 0.0)
            + (0.15 if linha_baixo else 0.0),
            0.0,
            1.0,
        )
        segmento_inicio = x_linha - alcance_lateral if ramo_esquerda else x_centro_1
        segmento_fim = x_linha + alcance_lateral if ramo_direita else x_centro_2
        candidato = {
            "detectado": True,
            "centro": (int(x_linha), int(y)),
            "y_cruzamento": int(y),
            "largura_linha_px": largura_linha_px,
            "ramo_esquerda": bool(ramo_esquerda),
            "ramo_direita": bool(ramo_direita),
            "ramo_frente": bool(ramo_frente),
            "linha_baixo": bool(linha_baixo),
            "confianca": confianca,
            "motivo": "cruzamento_lateral_detectado",
            "segmento": (
                int(limitar(segmento_inicio, 0, largura - 1)),
                int(limitar(segmento_fim, 0, largura - 1)),
            ),
        }
        if melhor is None or candidato["confianca"] > melhor["confianca"]:
            melhor = candidato

    if melhor is not None and melhor["confianca"] >= 0.45:
        return melhor
    return {
        "detectado": False,
        "centro": None,
        "y_cruzamento": None,
        "largura_linha_px": largura_linha_px,
        "ramo_esquerda": False,
        "ramo_direita": False,
        "ramo_frente": False,
        "confianca": 0.0,
        "motivo": "sem_cruzamento",
    }


def calcular_lado_verde(candidato, cruzamento, mascara_linha, largura_linha_px):
    cx, _ = candidato["centro"]
    referencia = obter_referencia_lado_verde(
        candidato,
        cruzamento,
        mascara_linha,
        largura_linha_px,
    )
    x_referencia = referencia["x_linha"]

    margem_centro = max(10, referencia["largura_linha_px"] * MARGEM_CENTRO_LINHA_MULT)
    if cx < x_referencia - margem_centro:
        return "ESQUERDA", int(x_referencia)
    if cx > x_referencia + margem_centro:
        return "DIREITA", int(x_referencia)
    return "CENTRO", int(x_referencia)


def calcular_lado_preliminar(candidato, cruzamento, mascara_linha, largura_linha_px):
    lado, x_referencia = calcular_lado_verde(candidato, cruzamento, mascara_linha, largura_linha_px)
    candidato = dict(candidato)
    candidato["lado_preliminar"] = lado
    candidato["x_referencia_preliminar"] = x_referencia
    return candidato


def unir_dois_candidatos(a, b):
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    area_total = max(a["area"] + b["area"], 1.0)
    unido = dict(a)
    unido["bbox"] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    unido["centro"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    unido["area"] = float(a["area"] + b["area"])
    unido["fill_ratio"] = float(unido["area"] / max((x2 - x1) * (y2 - y1), 1))
    for chave in ("mean_h", "mean_s", "mean_v", "mean_b", "mean_g", "mean_r", "g_menos_r", "g_menos_b", "proporcao_verde"):
        unido[chave] = float((a[chave] * a["area"] + b[chave] * b["area"]) / area_total)
    unido["lado"] = "CENTRO"
    unido["valido"] = False
    unido["motivo"] = "nao_validado"
    unido["confianca"] = 0.0
    unido["falso_depois_cruzamento"] = False
    if a.get("lado_preliminar") == b.get("lado_preliminar"):
        unido["lado_preliminar"] = a.get("lado_preliminar")
        unido["x_referencia_preliminar"] = a.get(
            "x_referencia_preliminar",
            b.get("x_referencia_preliminar"),
        )
    return unido


def pode_juntar_candidatos(a, b, largura_linha_px):
    lado_a = a.get("lado_preliminar", "CENTRO")
    lado_b = b.get("lado_preliminar", "CENTRO")
    if lado_a != lado_b:
        return False

    distancia = distancia_caixas(a["bbox"], b["bbox"])
    distancia_max = max(12, largura_linha_px * 1.2)
    if lado_a == "CENTRO":
        distancia_max = max(8, largura_linha_px * 0.6)

    unido = unir_dois_candidatos(a, b)
    _, _, largura_unida, altura_unida = unido["bbox"]
    uniao_plausivel = largura_unida <= largura_linha_px * 10 and altura_unida <= largura_linha_px * 10
    return distancia <= distancia_max and uniao_plausivel


def juntar_candidatos_verdes(candidatos, largura_linha_px):
    candidatos = [dict(candidato) for candidato in candidatos]
    mudou = True
    while mudou:
        mudou = False
        usados = [False] * len(candidatos)
        novos = []
        for i, candidato in enumerate(candidatos):
            if usados[i]:
                continue
            atual = candidato
            usados[i] = True
            for j in range(i + 1, len(candidatos)):
                if usados[j]:
                    continue
                outro = candidatos[j]
                if pode_juntar_candidatos(atual, outro, largura_linha_px):
                    atual = unir_dois_candidatos(atual, outro)
                    usados[j] = True
                    mudou = True
            novos.append(atual)
        candidatos = novos
    return candidatos


def analisar_preto_ao_redor_do_verde(candidato, mascara_linha):
    x, y, w, h = candidato["bbox"]
    margem = max(6, int(max(w, h) * PRETO_MARGEM_VERDE_MULT))

    preto_acima, pixels_acima, roi_acima = roi_preto_tem_linha(
        mascara_linha, x, y - margem, x + w, y
    )
    preto_abaixo, pixels_abaixo, roi_abaixo = roi_preto_tem_linha(
        mascara_linha, x, y + h, x + w, y + h + margem
    )
    preto_esquerda, pixels_esquerda, roi_esquerda = roi_preto_tem_linha(
        mascara_linha, x - margem, y, x, y + h
    )
    preto_direita, pixels_direita, roi_direita = roi_preto_tem_linha(
        mascara_linha, x + w, y, x + w + margem, y + h
    )

    return {
        "preto_acima": bool(preto_acima),
        "preto_abaixo": bool(preto_abaixo),
        "preto_esquerda": bool(preto_esquerda),
        "preto_direita": bool(preto_direita),
        "pixels_acima": int(pixels_acima),
        "pixels_abaixo": int(pixels_abaixo),
        "pixels_esquerda": int(pixels_esquerda),
        "pixels_direita": int(pixels_direita),
        "rois_preto": {
            "ACIMA": roi_acima,
            "ABAIXO": roi_abaixo,
            "ESQUERDA": roi_esquerda,
            "DIREITA": roi_direita,
        },
    }


def analisar_posicao_verde_em_relacao_cruzamento(candidato, cruzamento):
    if not cruzamento.get("detectado") or cruzamento.get("y_cruzamento") is None:
        return {
            "posicao": "SEM_CRUZAMENTO",
            "verde_antes_intersecao": True,
            "verde_depois_intersecao": False,
            "y_cruzamento": None,
            "dy_centro_cruzamento": None,
            "frac_bbox_antes": None,
        }

    _, y, _, h = candidato["bbox"]
    _, cy = candidato["centro"]
    y_cruzamento = int(cruzamento["y_cruzamento"])
    margem = int(
        limitar(
            h * VERDE_MARGEM_DEPOIS_MULT_ALTURA,
            VERDE_MARGEM_DEPOIS_MIN_PX,
            VERDE_MARGEM_DEPOIS_MAX_PX,
        )
    )

    # Na imagem, a regiao antes da intersecao fica abaixo do y do cruzamento.
    y_limite_antes = y_cruzamento + margem
    altura_antes = max(0, (y + h) - y_limite_antes)
    frac_bbox_antes = altura_antes / max(h, 1)
    dy_centro = cy - y_cruzamento

    if cy < y_cruzamento - margem and frac_bbox_antes < VERDE_FRAC_MIN_BBOX_ANTES:
        posicao = "DEPOIS"
        verde_antes = False
    elif cy >= y_cruzamento - margem:
        posicao = "ANTES"
        verde_antes = True
    elif frac_bbox_antes >= VERDE_FRAC_MIN_BBOX_ANTES:
        posicao = "SOBREPOSTO"
        verde_antes = True
    else:
        posicao = "DEPOIS"
        verde_antes = False

    return {
        "posicao": posicao,
        "verde_antes_intersecao": verde_antes,
        "verde_depois_intersecao": not verde_antes,
        "y_cruzamento": y_cruzamento,
        "dy_centro_cruzamento": float(dy_centro),
        "frac_bbox_antes": float(frac_bbox_antes),
    }


def validar_verde(candidato, cruzamento, mascara_linha, largura_linha_px):
    candidato = dict(candidato)
    x, y, w, h = candidato["bbox"]
    cx, cy = candidato["centro"]
    altura, largura = mascara_linha.shape[:2]
    candidato.update(analisar_preto_ao_redor_do_verde(candidato, mascara_linha))
    candidato["area_minima_y"] = float(area_minima_verde_por_y(cy, altura))
    candidato["x_referencia"] = None
    candidato["linha_referencia"] = None
    candidato["pixels_linha_proxima"] = 0
    candidato["falso_depois_cruzamento"] = False
    candidato["posicao_cruzamento"] = "NAO_ANALISADO"
    candidato["verde_antes_intersecao"] = False
    candidato["verde_depois_intersecao"] = False
    candidato["dy_centro_cruzamento"] = None
    candidato["frac_bbox_antes"] = None
    area_quadro = largura * altura
    area_rel_quadro = candidato["area"] / max(area_quadro, 1)
    candidato["area_rel_quadro"] = float(area_rel_quadro)
    verde_grande_mas_plausivel = area_rel_quadro > VERDE_AREA_MAX_REL_QUADRO
    candidato["verde_grande_mas_plausivel"] = bool(verde_grande_mas_plausivel)

    y_rel = cy / max(altura, 1)
    if y_rel < VERDE_Y_MIN_REL or y_rel > VERDE_Y_MAX_REL:
        candidato["motivo"] = "fora_roi_vertical"
        candidato["confianca"] = 0.0
        return candidato

    confianca = 0.0
    cor_boa = (
        VERDE_H_MIN <= candidato["mean_h"] <= VERDE_H_MAX
        and candidato["mean_s"] >= VERDE_S_MIN
        and candidato["mean_v"] >= VERDE_V_MIN
        and candidato["proporcao_verde"] >= VERDE_PROPORCAO_MIN
        and candidato["fill_ratio"] >= VERDE_FILL_RATIO_MIN
        and candidato["g_menos_r"] >= VERDE_G_MENOS_R_MIN
        and candidato["g_menos_b"] >= VERDE_G_MENOS_B_MIN
    )
    if cor_boa:
        confianca += 0.25
    else:
        candidato["motivo"] = "cor_hsv_invalida"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    if candidato["area"] < candidato["area_minima_y"]:
        candidato["motivo"] = "area_pequena_perspectiva"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    if area_rel_quadro > VERDE_AREA_ABSURDA_REL_QUADRO:
        candidato["motivo"] = "area_absurda"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    aspecto = w / max(h, 1)
    candidato["aspecto"] = float(aspecto)
    if aspecto < VERDE_ASPECTO_MIN or aspecto > VERDE_ASPECTO_MAX:
        candidato["motivo"] = "aspecto_invalido"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato
    confianca += 0.20

    posicao_cruzamento = analisar_posicao_verde_em_relacao_cruzamento(candidato, cruzamento)
    candidato["posicao_cruzamento"] = posicao_cruzamento["posicao"]
    candidato["verde_depois_intersecao"] = posicao_cruzamento["verde_depois_intersecao"]
    candidato["verde_antes_intersecao"] = posicao_cruzamento["verde_antes_intersecao"]
    candidato["dy_centro_cruzamento"] = posicao_cruzamento["dy_centro_cruzamento"]
    candidato["frac_bbox_antes"] = posicao_cruzamento["frac_bbox_antes"]
    if posicao_cruzamento["verde_depois_intersecao"]:
        candidato["valido"] = False
        candidato["falso_depois_cruzamento"] = True
        candidato["motivo"] = "verde_depois_intersecao_ignorado"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    margem = int(max(10, largura_linha_px * 1.5))
    x1 = int(limitar(x - margem, 0, largura - 1))
    y1 = int(limitar(y - margem, 0, altura - 1))
    x2 = int(limitar(x + w + margem, 1, largura))
    y2 = int(limitar(y + h + margem, 1, altura))
    pixels_linha_proxima = int(cv2.countNonZero(mascara_linha[y1:y2, x1:x2]))
    linha_proxima = pixels_linha_proxima >= max(12, largura_linha_px * 1.2)
    candidato["pixels_linha_proxima"] = pixels_linha_proxima
    if linha_proxima:
        confianca += 0.25
    else:
        candidato["motivo"] = "sem_linha_preta_proxima"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    preto_lados = sum(
        1
        for chave in ("preto_acima", "preto_abaixo", "preto_esquerda", "preto_direita")
        if candidato[chave]
    )
    if preto_lados < PRETO_MIN_LADOS_AO_REDOR:
        candidato["motivo"] = "sem_preto_ao_redor"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato
    confianca += 0.20

    sobre_linha = cv2.countNonZero(mascara_linha[y:y + h, x:x + w])
    if sobre_linha > max(8, candidato["area"] * 0.25):
        candidato["motivo"] = "verde_em_cima_da_linha"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    referencia_linha = obter_referencia_lado_verde(
        candidato,
        cruzamento,
        mascara_linha,
        largura_linha_px,
    )
    candidato["linha_referencia"] = referencia_linha
    candidato["x_referencia"] = referencia_linha["x_linha"]
    if referencia_linha["origem"] == "centro_imagem_fallback":
        candidato["motivo"] = "referencia_linha_fraca"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    x_linha = referencia_linha["x_linha"]
    largura_linha_local = referencia_linha["largura_linha_px"]

    margem_centro = max(10, largura_linha_local * MARGEM_CENTRO_LINHA_MULT)
    candidato["margem_centro_linha"] = float(margem_centro)
    if cx < x_linha - margem_centro:
        lado = "ESQUERDA"
    elif cx > x_linha + margem_centro:
        lado = "DIREITA"
    else:
        lado = "CENTRO"
    candidato["lado"] = lado

    if lado == "CENTRO":
        candidato["motivo"] = "verde_central_inseguro"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    coerente_esquerda = candidato["preto_acima"] or candidato["preto_direita"] or candidato["preto_abaixo"]
    coerente_direita = candidato["preto_acima"] or candidato["preto_esquerda"] or candidato["preto_abaixo"]
    if lado == "ESQUERDA" and not coerente_esquerda:
        candidato["motivo"] = "sem_preto_ao_redor"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato
    if lado == "DIREITA" and not coerente_direita:
        candidato["motivo"] = "sem_preto_ao_redor"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    if cruzamento_pode_ser_referencia(cruzamento, largura):
        confianca += 0.05

    confianca += 0.10
    if verde_grande_mas_plausivel:
        confianca -= 0.05
        candidato["observacao_tamanho"] = "verde_grande_mas_plausivel"
    candidato["valido"] = True
    candidato["motivo"] = "verde_esquerda_valido" if lado == "ESQUERDA" else "verde_direita_valido"
    candidato["confianca"] = limitar(confianca, 0.0, 1.0)
    return candidato


def decidir_verdes(candidatos, cruzamento):
    validos_antes = [
        candidato
        for candidato in candidatos
        if candidato.get("valido", False)
        and candidato.get("verde_antes_intersecao", False)
        and not candidato.get("verde_depois_intersecao", False)
        and not candidato.get("falso_depois_cruzamento", False)
    ]
    validos_sobrepostos = [
        candidato
        for candidato in candidatos
        if candidato.get("valido", False)
        and candidato.get("posicao_cruzamento") == "SOBREPOSTO"
        and not candidato.get("verde_depois_intersecao", False)
        and not candidato.get("falso_depois_cruzamento", False)
    ]
    validos_para_acao = validos_antes + [
        candidato
        for candidato in validos_sobrepostos
        if not any(candidato is valido_antes for valido_antes in validos_antes)
    ]
    falsos_depois = [
        candidato
        for candidato in candidatos
        if candidato.get("verde_depois_intersecao", False)
        or candidato.get("falso_depois_cruzamento", False)
    ]
    validos_esquerda = [c for c in validos_para_acao if c.get("lado") == "ESQUERDA"]
    validos_direita = [c for c in validos_para_acao if c.get("lado") == "DIREITA"]

    if validos_esquerda and validos_direita:
        confianca = min(max(max(c["confianca"] for c in validos_esquerda), max(c["confianca"] for c in validos_direita)), 1.0)
        return "RETORNO", "verde_duplo_valido", confianca, True, "VERDE_ANTES_INTERSECAO"
    if len(validos_esquerda) == 1 and not validos_direita:
        return (
            "ESQUERDA",
            "verde_esquerda_valido",
            validos_esquerda[0]["confianca"],
            True,
            "VERDE_ANTES_INTERSECAO",
        )
    if len(validos_direita) == 1 and not validos_esquerda:
        return (
            "DIREITA",
            "verde_direita_valido",
            validos_direita[0]["confianca"],
            True,
            "VERDE_ANTES_INTERSECAO",
        )
    if len(validos_esquerda) > 1 or len(validos_direita) > 1:
        return "INSEGURO", "verde_inseguro", 0.45, False, "INSEGURO"
    if falsos_depois:
        return (
            "NENHUM",
            "apenas_verde_depois_intersecao_ignorado",
            0.0,
            False,
            "VERDE_DEPOIS_INTERSECAO",
        )
    if cruzamento["detectado"]:
        return "NENHUM", "cruzamento_sem_verde_ignorado", 0.0, False, "SEM_VERDE_VALIDO"
    return "NENHUM", "sem_verde_valido", 0.0, False, "SEM_VERDE_VALIDO"


def analisar_verdes(frame_bgr):
    resultado_linha = detectar_linha(frame_bgr)
    mascara_linha_original = criar_mascara_linha_global(resultado_linha)
    mascara_verde = criar_mascara_verde(frame_bgr)
    mascara_linha = remover_verde_da_mascara_linha(mascara_linha_original, mascara_verde)
    cruzamento = analisar_cruzamento(mascara_linha)
    largura_linha_px = cruzamento["largura_linha_px"]
    candidatos = encontrar_candidatos_verdes(frame_bgr, mascara_verde)
    candidatos = [
        calcular_lado_preliminar(c, cruzamento, mascara_linha, largura_linha_px)
        for c in candidatos
    ]
    candidatos = juntar_candidatos_verdes(candidatos, largura_linha_px)
    verdes = [validar_verde(c, cruzamento, mascara_linha, largura_linha_px) for c in candidatos]
    decisao, motivo, confianca, acao_permitida, origem_decisao = decidir_verdes(
        verdes,
        cruzamento,
    )
    tem_verde_valido_antes = any(
        candidato.get("valido", False)
        and (
            candidato.get("verde_antes_intersecao", False)
            or candidato.get("posicao_cruzamento") == "SOBREPOSTO"
        )
        and not candidato.get("verde_depois_intersecao", False)
        and not candidato.get("falso_depois_cruzamento", False)
        for candidato in verdes
    )
    tem_verde_falso_depois = any(
        candidato.get("verde_depois_intersecao", False)
        or candidato.get("falso_depois_cruzamento", False)
        for candidato in verdes
    )
    return {
        "decisao": decisao,
        "motivo": motivo,
        "confianca": limitar(confianca, 0.0, 1.0),
        "acao_permitida": bool(acao_permitida),
        "origem_decisao": origem_decisao,
        "tem_verde_valido_antes": bool(tem_verde_valido_antes),
        "tem_verde_falso_depois": bool(tem_verde_falso_depois),
        "cruzamento": cruzamento,
        "verdes": verdes,
        "mascaras": {
            "linha": mascara_linha,
            "linha_original": mascara_linha_original,
            "verde": mascara_verde,
        },
    }


def formatar_log(resultado):
    cruzamento = resultado["cruzamento"]
    verdes = resultado["verdes"]
    validos_esquerda = sum(1 for verde in verdes if verde["valido"] and verde["lado"] == "ESQUERDA")
    validos_direita = sum(1 for verde in verdes if verde["valido"] and verde["lado"] == "DIREITA")
    falsos = sum(1 for verde in verdes if verde.get("falso_depois_cruzamento"))
    return (
        f"[VERDES] dec={resultado['decisao']} conf={resultado['confianca']:.2f} motivo={resultado['motivo']} "
        f"acao={int(resultado['acao_permitida'])} origem={resultado['origem_decisao']} "
        f"valido_antes={int(resultado['tem_verde_valido_antes'])} "
        f"falso_depois={int(resultado['tem_verde_falso_depois'])} | "
        f"cruz={int(cruzamento['detectado'])} E={int(cruzamento['ramo_esquerda'])} "
        f"D={int(cruzamento['ramo_direita'])} F={int(cruzamento['ramo_frente'])} | "
        f"verdes E/D={validos_esquerda}/{validos_direita} falsos={falsos} total={len(verdes)}"
    )


def imprimir_log_detalhado(resultado):
    print(
        f"resumo decisao={resultado['decisao']} motivo={resultado['motivo']} "
        f"confianca={resultado['confianca']:.2f} acao_permitida={resultado['acao_permitida']} "
        f"origem_decisao={resultado['origem_decisao']} "
        f"tem_verde_valido_antes={resultado['tem_verde_valido_antes']} "
        f"tem_verde_falso_depois={resultado['tem_verde_falso_depois']}"
    )
    for indice, verde in enumerate(resultado["verdes"], start=1):
        referencia = verde.get("linha_referencia") or {}
        print(
            f"verde#{indice} lado={verde['lado']} bbox={verde['bbox']} area={verde['area']:.0f} "
            f"valido={verde['valido']} motivo={verde['motivo']} conf={verde['confianca']:.2f} "
            f"preto_acima={verde.get('preto_acima', False)} preto_abaixo={verde.get('preto_abaixo', False)} "
            f"preto_esquerda={verde.get('preto_esquerda', False)} preto_direita={verde.get('preto_direita', False)} "
            f"pixels_acima={verde.get('pixels_acima', 0)} pixels_abaixo={verde.get('pixels_abaixo', 0)} "
            f"pixels_esquerda={verde.get('pixels_esquerda', 0)} pixels_direita={verde.get('pixels_direita', 0)} "
            f"origem_ref={referencia.get('origem', 'nenhuma')} x_ref={verde.get('x_referencia')} "
            f"area_rel={verde.get('area_rel_quadro', 0.0):.3f} "
            f"grande_plausivel={verde.get('verde_grande_mas_plausivel', False)} "
            f"observacao_tamanho={verde.get('observacao_tamanho', 'nenhuma')} "
            f"posicao_cruzamento={verde.get('posicao_cruzamento', 'NAO_ANALISADO')} "
            f"verde_depois_intersecao={verde.get('verde_depois_intersecao', False)} "
            f"dy_centro_cruzamento={verde.get('dy_centro_cruzamento')} "
            f"frac_bbox_antes={verde.get('frac_bbox_antes')}"
        )


def criar_debug_verdes(frame_bgr, resultado):
    debug = frame_bgr.copy()
    mascara_linha = resultado["mascaras"]["linha"]
    mascara_verde = resultado["mascaras"]["verde"]

    sobreposicao = debug.copy()
    sobreposicao[mascara_linha > 0] = (255, 255, 0)
    sobreposicao[mascara_verde > 0] = (0, 255, 0)
    debug = cv2.addWeighted(sobreposicao, 0.35, debug, 0.65, 0)

    cruzamento = resultado["cruzamento"]
    altura, largura = debug.shape[:2]
    if cruzamento["detectado"]:
        cx, cy = cruzamento["centro"]
        cv2.line(debug, (cx, 0), (cx, altura), (255, 0, 255), 2)
        cv2.line(debug, (0, cruzamento["y_cruzamento"]), (largura, cruzamento["y_cruzamento"]), (255, 255, 255), 2)
        if "segmento" in cruzamento:
            x1, x2 = cruzamento["segmento"]
            y = cruzamento["y_cruzamento"]
            cv2.line(debug, (x1, y), (x2, y), (0, 255, 255), 4)
        cv2.circle(debug, cruzamento["centro"], 7, (255, 0, 255), -1)
    else:
        cv2.line(debug, (largura // 2, 0), (largura // 2, altura), (80, 80, 80), 1)

    for verde in resultado["verdes"]:
        x, y, w, h = verde["bbox"]
        if verde.get("falso_depois_cruzamento"):
            cor = (0, 0, 255)
        elif verde["valido"] and verde["lado"] == "ESQUERDA":
            cor = (0, 255, 0)
        elif verde["valido"] and verde["lado"] == "DIREITA":
            cor = (255, 0, 0)
        elif verde["lado"] == "CENTRO":
            cor = (0, 255, 255)
        else:
            cor = (0, 165, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), cor, 2)
        cv2.circle(debug, verde["centro"], 4, cor, -1)
        for nome_roi, roi in verde.get("rois_preto", {}).items():
            rx1, ry1, rx2, ry2 = roi
            tem_preto = verde.get(f"preto_{nome_roi.lower()}", False)
            cor_roi = (255, 255, 0) if tem_preto else (90, 90, 90)
            cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), cor_roi, 1)
        referencia = verde.get("linha_referencia")
        if referencia is not None:
            x_linha = referencia["x_linha"]
            y_linha = referencia["y_linha"]
            cv2.line(debug, (x_linha, max(0, y_linha - 12)), (x_linha, min(altura - 1, y_linha + 12)), (255, 0, 255), 2)
            cv2.line(debug, (x_linha, y_linha), verde["centro"], (255, 0, 255), 1)
        if verde.get("verde_depois_intersecao"):
            texto = f"DEPOIS {verde['motivo']} {verde['confianca']:.2f}"
        else:
            texto = f"{verde['lado']} {verde['motivo']} {verde['confianca']:.2f}"
        cv2.putText(debug, texto, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(debug, texto, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)

    verdes_validos = sum(1 for verde in resultado["verdes"] if verde["valido"])
    linhas = [
        f"Decisao: {resultado['decisao']}  Motivo: {resultado['motivo']}  Conf: {resultado['confianca']:.2f}",
        f"Cruzamento: {int(cruzamento['detectado'])}  E/D/F: {int(cruzamento['ramo_esquerda'])}/{int(cruzamento['ramo_direita'])}/{int(cruzamento['ramo_frente'])}  Verdes validos: {verdes_validos}",
        f"Largura linha: {cruzamento['largura_linha_px']:.1f}px",
    ]
    for indice, texto in enumerate(linhas):
        y = 28 + indice * 24
        cv2.putText(debug, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(debug, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return debug


def salvar_debug_verdes(frame_bgr, resultado):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    agora = datetime.now()
    caminho = pasta / f"debug_verdes_{agora.strftime('%Y%m%d_%H%M%S')}_{agora.microsecond // 1000:03d}.jpg"
    debug = criar_debug_verdes(frame_bgr, resultado)
    if not cv2.imwrite(str(caminho), debug):
        raise RuntimeError(f"Nao foi possivel salvar debug: {caminho}")
    print(f"Debug salvo: {caminho}")
    return caminho


def executar_imagem(caminho, mostrar, salvar_debug, log_detalhe):
    frame = cv2.imread(str(caminho))
    if frame is None:
        print(f"Erro: nao foi possivel carregar imagem: {caminho}")
        return 1
    resultado = analisar_verdes(frame)
    print(formatar_log(resultado))
    if log_detalhe:
        imprimir_log_detalhado(resultado)
    if salvar_debug:
        salvar_debug_verdes(frame, resultado)
    if mostrar:
        cv2.imshow("verdes", criar_debug_verdes(frame, resultado))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def executar_camera(mostrar, salvar_debug, log_detalhe):
    camera = None
    ultima_decisao = None
    ultimo_debug = 0.0
    try:
        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
        while True:
            frame = capturar_frame_bgr(camera)
            resultado = analisar_verdes(frame)
            print(formatar_log(resultado))
            if log_detalhe:
                imprimir_log_detalhado(resultado)

            agora = time.monotonic()
            mudou_decisao = resultado["decisao"] != ultima_decisao
            if salvar_debug and (mudou_decisao or agora - ultimo_debug >= INTERVALO_DEBUG_CAMERA):
                salvar_debug_verdes(frame, resultado)
                ultimo_debug = agora
            ultima_decisao = resultado["decisao"]

            if mostrar:
                cv2.imshow("verdes", criar_debug_verdes(frame, resultado))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("CTRL+C recebido. Encerrando verdes.")
        return 130
    except RuntimeError as erro:
        print(f"Erro: {erro}")
        return 1
    finally:
        if camera is not None:
            camera.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta verdes por visao computacional para debug.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    origem.add_argument("--imagem", help="Caminho de uma imagem salva.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra a janela de debug.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva imagem de debug em captures.")
    parser.add_argument("--log-detalhe", action="store_true", help="Imprime dados dos candidatos verdes.")
    return parser.parse_args()


def main():
    argumentos = ler_argumentos()
    if not argumentos.camera and not argumentos.imagem:
        print("Use --camera ou --imagem caminho/para/imagem.jpg")
        return 0
    if argumentos.imagem:
        return executar_imagem(argumentos.imagem, argumentos.mostrar, argumentos.salvar_debug, argumentos.log_detalhe)
    return executar_camera(argumentos.mostrar, argumentos.salvar_debug, argumentos.log_detalhe)


if __name__ == "__main__":
    raise SystemExit(main())
