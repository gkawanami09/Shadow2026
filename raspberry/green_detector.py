"""Detector HSV reutilizavel para marcacoes verdes da pista."""

import cv2
import numpy as np

from config import (
    GREEN_CENTER_DEADBAND_REL,
    GREEN_DUPLO_MIN_AREA_RATIO,
    GREEN_H_MAX,
    GREEN_H_MIN,
    GREEN_MAX_AREA_REL,
    GREEN_MAX_ASPECT_RATIO,
    GREEN_MIN_AREA,
    GREEN_MEAN_G_MINUS_B_MIN,
    GREEN_MEAN_G_MINUS_R_MIN,
    GREEN_MEAN_S_MIN,
    GREEN_MIN_ASPECT_RATIO,
    GREEN_MIN_FILL_RATIO,
    GREEN_MIN_GREEN_RATIO,
    GREEN_MIN_HEIGHT,
    GREEN_MIN_WIDTH,
    GREEN_MORPH_CLOSE_ITER,
    GREEN_MORPH_KERNEL,
    GREEN_MORPH_OPEN_ITER,
    GREEN_ROI_Y_MAX_REL,
    GREEN_ROI_Y_MIN_REL,
    GREEN_S_MIN,
    GREEN_V_MIN,
)


def lado_por_x(x_centro, x_referencia, largura, deadband_rel):
    """Classifica uma coordenada X em relacao a uma referencia vertical."""
    margem = largura * deadband_rel
    if x_centro < x_referencia - margem:
        return "ESQUERDA"
    if x_centro > x_referencia + margem:
        return "DIREITA"
    return "CENTRO"


def criar_mascara_verde(frame_bgr):
    """Retorna a mascara HSV limpa e os limites verticais da ROI."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr deve ser uma imagem BGR com tres canais.")

    altura, largura = frame_bgr.shape[:2]
    y_inicio = int(altura * GREEN_ROI_Y_MIN_REL)
    y_fim = int(altura * GREEN_ROI_Y_MAX_REL)
    if not 0 <= y_inicio < y_fim <= altura:
        raise RuntimeError("A ROI vertical de verde e invalida. Revise config.py.")

    roi = frame_bgr[y_inicio:y_fim, :largura]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    limite_inferior = np.array([GREEN_H_MIN, GREEN_S_MIN, GREEN_V_MIN], dtype=np.uint8)
    limite_superior = np.array([GREEN_H_MAX, 255, 255], dtype=np.uint8)
    mascara = cv2.inRange(hsv, limite_inferior, limite_superior)

    kernel = np.ones((GREEN_MORPH_KERNEL, GREEN_MORPH_KERNEL), dtype=np.uint8)
    mascara_limpa = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel, iterations=GREEN_MORPH_OPEN_ITER
    )
    mascara_limpa = cv2.morphologyEx(
        mascara_limpa, cv2.MORPH_CLOSE, kernel, iterations=GREEN_MORPH_CLOSE_ITER
    )
    return {
        "mascara": mascara_limpa,
        "roi_bgr": roi,
        "roi_hsv": hsv,
        "y_inicio_roi": y_inicio,
        "y_fim_roi": y_fim,
    }


def extrair_contornos_verdes(
    mascara, roi_bgr, roi_hsv, y_inicio_roi, largura_frame, altura_frame, x_referencia=None
):
    """Extrai contornos verdes validos da mascara relativa a ROI."""
    if x_referencia is None:
        x_referencia = largura_frame // 2

    area_frame = largura_frame * altura_frame
    area_maxima = GREEN_MAX_AREA_REL * area_frame
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = []
    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        x, y, largura, altura = cv2.boundingRect(contorno)
        if area < GREEN_MIN_AREA or area > area_maxima:
            continue
        if largura < GREEN_MIN_WIDTH or altura < GREEN_MIN_HEIGHT:
            continue

        fill_ratio = area / (largura * altura)
        if fill_ratio < GREEN_MIN_FILL_RATIO:
            continue

        mascara_contorno = np.zeros_like(mascara)
        cv2.drawContours(mascara_contorno, [contorno], -1, 255, thickness=cv2.FILLED)
        mean_h, mean_s, mean_v, _ = cv2.mean(roi_hsv, mask=mascara_contorno)
        mean_b, mean_g, mean_r, _ = cv2.mean(roi_bgr, mask=mascara_contorno)
        g_minus_r = mean_g - mean_r
        g_minus_b = mean_g - mean_b
        green_ratio = mean_g / max(mean_r, mean_b, 1.0)
        aspect_ratio = largura / altura
        if (
            mean_s < GREEN_MEAN_S_MIN
            or g_minus_r < GREEN_MEAN_G_MINUS_R_MIN
            or g_minus_b < GREEN_MEAN_G_MINUS_B_MIN
            or green_ratio < GREEN_MIN_GREEN_RATIO
            or not GREEN_MIN_ASPECT_RATIO <= aspect_ratio <= GREEN_MAX_ASPECT_RATIO
        ):
            continue

        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        cx = int(momentos["m10"] / momentos["m00"])
        cy_global = y_inicio_roi + int(momentos["m01"] / momentos["m00"])
        y_global = y_inicio_roi + y
        validos.append(
            {
                "area": area,
                "bbox": (x, y_global, largura, altura),
                "centro": (cx, cy_global),
                "lado": lado_por_x(
                    cx, x_referencia, largura_frame, GREEN_CENTER_DEADBAND_REL
                ),
                "fill_ratio": fill_ratio,
                "mean_h": mean_h,
                "mean_s": mean_s,
                "mean_v": mean_v,
                "mean_b": mean_b,
                "mean_g": mean_g,
                "mean_r": mean_r,
                "g_minus_r": g_minus_r,
                "g_minus_b": g_minus_b,
                "green_ratio": green_ratio,
                "aspect_ratio": aspect_ratio,
            }
        )
    return sorted(validos, key=lambda item: item["area"], reverse=True)


def resumir_verde(contornos):
    """Resume os contornos em uma classificacao visual, sem tomar movimento."""
    areas = {"ESQUERDA": 0.0, "DIREITA": 0.0, "CENTRO": 0.0}
    for contorno in contornos:
        areas[contorno["lado"]] += contorno["area"]

    area_esq = areas["ESQUERDA"]
    area_dir = areas["DIREITA"]
    area_centro = areas["CENTRO"]
    tipo = "NENHUM"
    confianca = 0.0
    observacao = "sem_contornos"

    if area_esq > 0 and area_dir > 0:
        menor_area, maior_area = sorted((area_esq, area_dir))
        if menor_area / maior_area >= GREEN_DUPLO_MIN_AREA_RATIO:
            tipo = "DUPLO"
            confianca = min(1.0, (area_esq + area_dir) / 800.0)
            observacao = "ok"
        elif area_esq > area_dir:
            tipo = "ESQUERDA"
            confianca = min(1.0, area_esq / 500.0)
            observacao = "duplo_desbalanceado"
        else:
            tipo = "DIREITA"
            confianca = min(1.0, area_dir / 500.0)
            observacao = "duplo_desbalanceado"
    elif area_esq > 0:
        tipo = "ESQUERDA"
        confianca = min(1.0, area_esq / 500.0)
        observacao = "ok"
    elif area_dir > 0:
        tipo = "DIREITA"
        confianca = min(1.0, area_dir / 500.0)
        observacao = "ok"
    elif area_centro > 0:
        tipo = "AMBIGUO"
        confianca = 0.3
        observacao = "centro_deadband"

    return {
        "tipo": tipo,
        "area_esquerda": area_esq,
        "area_direita": area_dir,
        "area_centro": area_centro,
        "qtd_contornos": len(contornos),
        "confianca": confianca,
        "observacao": observacao,
    }


def detectar_verde(frame_bgr, x_referencia=None):
    """Detecta marcacoes verdes, retornando somente dados de percepcao."""
    altura, largura = frame_bgr.shape[:2]
    if x_referencia is None:
        x_referencia = largura // 2
    mascara_resultado = criar_mascara_verde(frame_bgr)
    contornos = extrair_contornos_verdes(
        mascara_resultado["mascara"],
        mascara_resultado["roi_bgr"],
        mascara_resultado["roi_hsv"],
        mascara_resultado["y_inicio_roi"],
        largura,
        altura,
        x_referencia,
    )
    resumo = resumir_verde(contornos)
    return {
        **resumo,
        "contornos": contornos,
        "mascara": mascara_resultado["mascara"],
        "y_inicio_roi": mascara_resultado["y_inicio_roi"],
        "y_fim_roi": mascara_resultado["y_fim_roi"],
        "x_referencia": x_referencia,
    }


def criar_debug_verde(frame_bgr, resultado_verde):
    """Desenha ROI, referencia, deadband e contornos aceitos no frame."""
    debug = frame_bgr.copy()
    altura, largura = debug.shape[:2]
    y_inicio = resultado_verde["y_inicio_roi"]
    y_fim = resultado_verde["y_fim_roi"]
    x_referencia = resultado_verde["x_referencia"]
    margem = int(largura * GREEN_CENTER_DEADBAND_REL)

    cv2.rectangle(debug, (0, y_inicio), (largura - 1, y_fim - 1), (255, 255, 0), 2)
    cv2.line(debug, (x_referencia, 0), (x_referencia, altura - 1), (0, 255, 255), 2)
    cv2.rectangle(
        debug,
        (max(0, x_referencia - margem), y_inicio),
        (min(largura - 1, x_referencia + margem), y_fim - 1),
        (0, 165, 255),
        1,
    )

    cores = {"ESQUERDA": (0, 255, 0), "DIREITA": (255, 0, 0), "CENTRO": (0, 165, 255)}
    rotulos = {"ESQUERDA": "E", "DIREITA": "D", "CENTRO": "C"}
    for contorno in resultado_verde["contornos"]:
        x, y, w, h = contorno["bbox"]
        lado = contorno["lado"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), cores[lado], 2)
        cv2.putText(debug, rotulos[lado], (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cores[lado], 2)
        metricas = (
            f"{rotulos[lado]} S:{contorno['mean_s']:.0f} "
            f"GR:{contorno['g_minus_r']:.0f} GB:{contorno['g_minus_b']:.0f}"
        )
        cv2.putText(debug, metricas, (x, min(altura - 8, y + h + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cores[lado], 1)

    linhas = [
        f"verde: {resultado_verde['tipo']} | conf: {resultado_verde['confianca']:.2f}",
        "area E/D/C: "
        f"{resultado_verde['area_esquerda']:.0f}/"
        f"{resultado_verde['area_direita']:.0f}/"
        f"{resultado_verde['area_centro']:.0f} | qtd: {resultado_verde['qtd_contornos']}",
        f"obs: {resultado_verde['observacao']}",
    ]
    for indice, texto in enumerate(linhas):
        cv2.putText(debug, texto, (15, 30 + indice * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(debug, texto, (15, 30 + indice * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
    return debug
