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


VERDE_H_MIN = 35
VERDE_H_MAX = 95
VERDE_S_MIN = 50
VERDE_V_MIN = 25

KERNEL_VERDE = 5
OPEN_VERDE = 1
CLOSE_VERDE = 2

AREA_MIN_VERDE_ABSOLUTA = 40
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
    mascara = cv2.inRange(hsv, minimo, maximo)
    kernel = np.ones((KERNEL_VERDE, KERNEL_VERDE), np.uint8)
    if OPEN_VERDE > 0:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=OPEN_VERDE)
    if CLOSE_VERDE > 0:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_VERDE)
    return mascara


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


def encontrar_centro_linha_por_y(mascara_linha, y):
    altura, largura = mascara_linha.shape[:2]
    y = int(limitar(y, 0, altura - 1))
    y1 = limitar(y - 5, 0, altura - 1)
    y2 = limitar(y + 6, 1, altura)
    faixa = mascara_linha[y1:y2, :]
    linha = cv2.reduce(faixa, 0, cv2.REDUCE_MAX).flatten() > 0
    segmentos = encontrar_segmentos(linha)
    if not segmentos:
        return None
    centro_imagem = largura / 2
    segmento = min(segmentos, key=lambda seg: abs(((seg[0] + seg[1]) / 2) - centro_imagem))
    return int((segmento[0] + segmento[1]) / 2)


def analisar_cruzamento(mascara_linha):
    altura, largura = mascara_linha.shape[:2]
    largura_linha_px = estimar_largura_linha(mascara_linha)
    melhor = None

    for y in np.linspace(int(altura * 0.25), int(altura * 0.82), 32).astype(int):
        y1 = limitar(y - 6, 0, altura - 1)
        y2 = limitar(y + 7, 1, altura)
        faixa = mascara_linha[y1:y2, :]
        linha = cv2.reduce(faixa, 0, cv2.REDUCE_MAX).flatten() > 0
        segmentos = encontrar_segmentos(linha)
        if not segmentos:
            continue
        segmentos = [seg for seg in segmentos if seg[1] - seg[0] + 1 >= largura_linha_px * 1.2]
        if not segmentos:
            continue
        segmento = max(segmentos, key=lambda seg: seg[1] - seg[0] + 1)
        largura_segmento = segmento[1] - segmento[0] + 1
        centro_x = int((segmento[0] + segmento[1]) / 2)
        if largura_segmento < largura_linha_px * MULT_LARGURA_CRUZAMENTO:
            continue

        corredor = max(12, largura_linha_px * 1.5)
        limite_esq = centro_x - corredor / 2
        limite_dir = centro_x + corredor / 2
        ramo_esquerda = segmento[0] < limite_esq - largura_linha_px * 0.7
        ramo_direita = segmento[1] > limite_dir + largura_linha_px * 0.7

        y_frente_1 = limitar(y - int(largura_linha_px * 3.5), 0, altura - 1)
        y_frente_2 = limitar(y - int(largura_linha_px * 1.2), 1, altura)
        x_frente_1 = int(limitar(centro_x - corredor, 0, largura - 1))
        x_frente_2 = int(limitar(centro_x + corredor, 1, largura))
        frente = mascara_linha[y_frente_1:y_frente_2, x_frente_1:x_frente_2]
        ramo_frente = cv2.countNonZero(frente) >= max(20, largura_linha_px * 1.8)

        if not (ramo_esquerda or ramo_direita):
            continue
        confianca = limitar(
            0.35 + 0.25 * (largura_segmento / max(largura_linha_px * 4.0, 1))
            + (0.15 if ramo_esquerda else 0.0)
            + (0.15 if ramo_direita else 0.0)
            + (0.10 if ramo_frente else 0.0),
            0.0,
            1.0,
        )
        candidato = {
            "detectado": True,
            "centro": (centro_x, int(y)),
            "y_cruzamento": int(y),
            "largura_linha_px": largura_linha_px,
            "ramo_esquerda": bool(ramo_esquerda),
            "ramo_direita": bool(ramo_direita),
            "ramo_frente": bool(ramo_frente),
            "confianca": confianca,
            "motivo": "cruzamento_detectado",
            "segmento": (int(segmento[0]), int(segmento[1])),
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
    cx, cy = candidato["centro"]
    if cruzamento["detectado"] and cruzamento["centro"] is not None:
        x_referencia = cruzamento["centro"][0]
    else:
        x_linha = encontrar_centro_linha_por_y(mascara_linha, cy)
        x_referencia = x_linha if x_linha is not None else mascara_linha.shape[1] // 2

    margem_centro = max(10, largura_linha_px * 0.8)
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


def validar_verde(candidato, cruzamento, mascara_linha, largura_linha_px):
    candidato = dict(candidato)
    x, y, w, h = candidato["bbox"]
    cx, cy = candidato["centro"]
    altura, largura = mascara_linha.shape[:2]
    lado, x_referencia = calcular_lado_verde(candidato, cruzamento, mascara_linha, largura_linha_px)
    candidato["lado"] = lado
    candidato["x_referencia"] = x_referencia

    confianca = 0.0
    cor_boa = (
        VERDE_H_MIN <= candidato["mean_h"] <= VERDE_H_MAX
        and candidato["mean_s"] >= VERDE_S_MIN
        and candidato["mean_v"] >= VERDE_V_MIN
        and candidato["proporcao_verde"] >= 1.02
    )
    if cor_boa:
        confianca += 0.25

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

    largura_min_rel = 0.7 * largura_linha_px
    largura_max_rel = 8.0 * largura_linha_px
    altura_min_rel = 0.5 * largura_linha_px
    altura_max_rel = 8.0 * largura_linha_px
    if w < largura_min_rel or h < altura_min_rel:
        candidato["motivo"] = "verde_muito_pequeno_relativo"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato
    if w > largura_max_rel or h > altura_max_rel:
        candidato["motivo"] = "verde_grande_demais_relativo"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato
    confianca += 0.20

    sobre_linha = cv2.countNonZero(mascara_linha[y:y + h, x:x + w])
    if sobre_linha > max(8, candidato["area"] * 0.25):
        candidato["motivo"] = "verde_em_cima_da_linha"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    if cruzamento["detectado"]:
        confianca += 0.20
        margem_y = max(12, largura_linha_px * 1.5)
        if cy < cruzamento["y_cruzamento"] - margem_y:
            candidato["falso_depois_cruzamento"] = True
            candidato["motivo"] = "falso_depois_cruzamento"
            candidato["confianca"] = limitar(confianca, 0.0, 1.0)
            return candidato

    if lado == "CENTRO":
        candidato["motivo"] = "verde_central_inseguro"
        candidato["confianca"] = limitar(confianca, 0.0, 1.0)
        return candidato

    confianca += 0.10
    candidato["valido"] = True
    candidato["motivo"] = "verde_esquerda_valido" if lado == "ESQUERDA" else "verde_direita_valido"
    candidato["confianca"] = limitar(confianca, 0.0, 1.0)
    return candidato


def decidir_verdes(candidatos, cruzamento):
    validos_esquerda = [c for c in candidatos if c["valido"] and c["lado"] == "ESQUERDA"]
    validos_direita = [c for c in candidatos if c["valido"] and c["lado"] == "DIREITA"]
    falsos = [c for c in candidatos if c.get("falso_depois_cruzamento")]

    if not cruzamento["detectado"]:
        return "NENHUM", "sem_cruzamento", 0.0

    if validos_esquerda and validos_direita:
        confianca = min(max(max(c["confianca"] for c in validos_esquerda), max(c["confianca"] for c in validos_direita)), 1.0)
        return "RETORNO", "verde_duplo_valido", confianca
    if len(validos_esquerda) == 1 and not validos_direita:
        return "ESQUERDA", "verde_esquerda_valido", validos_esquerda[0]["confianca"]
    if len(validos_direita) == 1 and not validos_esquerda:
        return "DIREITA", "verde_direita_valido", validos_direita[0]["confianca"]
    if len(validos_esquerda) > 1 or len(validos_direita) > 1:
        return "INSEGURO", "verde_inseguro", 0.45
    if falsos:
        return "RETO", "verde_falso_depois_cruzamento", max(c["confianca"] for c in falsos)
    return "RETO", "cruzamento_sem_verde", max(0.55, cruzamento["confianca"])


def analisar_verdes(frame_bgr):
    resultado_linha = detectar_linha(frame_bgr)
    mascara_linha = criar_mascara_linha_global(resultado_linha)
    mascara_verde = criar_mascara_verde(frame_bgr)
    cruzamento = analisar_cruzamento(mascara_linha)
    largura_linha_px = cruzamento["largura_linha_px"]
    candidatos = encontrar_candidatos_verdes(frame_bgr, mascara_verde)
    candidatos = [
        calcular_lado_preliminar(c, cruzamento, mascara_linha, largura_linha_px)
        for c in candidatos
    ]
    candidatos = juntar_candidatos_verdes(candidatos, largura_linha_px)
    verdes = [validar_verde(c, cruzamento, mascara_linha, largura_linha_px) for c in candidatos]
    decisao, motivo, confianca = decidir_verdes(verdes, cruzamento)
    return {
        "decisao": decisao,
        "motivo": motivo,
        "confianca": limitar(confianca, 0.0, 1.0),
        "cruzamento": cruzamento,
        "verdes": verdes,
        "mascaras": {
            "linha": mascara_linha,
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
        f"[VERDES] dec={resultado['decisao']} conf={resultado['confianca']:.2f} motivo={resultado['motivo']} | "
        f"cruz={int(cruzamento['detectado'])} E={int(cruzamento['ramo_esquerda'])} "
        f"D={int(cruzamento['ramo_direita'])} F={int(cruzamento['ramo_frente'])} | "
        f"verdes E/D={validos_esquerda}/{validos_direita} falsos={falsos} total={len(verdes)}"
    )


def imprimir_log_detalhado(resultado):
    for indice, verde in enumerate(resultado["verdes"], start=1):
        print(
            f"verde#{indice} lado={verde['lado']} bbox={verde['bbox']} area={verde['area']:.0f} "
            f"valido={verde['valido']} motivo={verde['motivo']} conf={verde['confianca']:.2f} "
            f"falso={verde.get('falso_depois_cruzamento', False)}"
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
