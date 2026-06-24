"""Detector HSV reutilizavel para marcacoes verdes da pista."""

import cv2
import numpy as np

from config import (
    GREEN_CENTER_DEADBAND_REL,
    GREEN_CONFIRM_MAX_DIST_BLACK_PX,
    GREEN_CONFIRM_MIN_AREA_IN_ZONE_RATIO,
    GREEN_CONFIRM_MIN_BLACK_NEAR_PIXELS,
    GREEN_CONFIRM_Y_MAX_REL,
    GREEN_CONFIRM_Y_MIN_REL,
    GREEN_DUPLO_MIN_AREA_RATIO,
    GREEN_EXPECTED_SLOT_HEIGHT_REL,
    GREEN_EXPECTED_SLOT_WIDTH_REL,
    GREEN_EXPECTED_SLOT_Y_MIN_REL,
    GREEN_H_MAX,
    GREEN_H_MIN,
    GREEN_LINE_MASK_DILATE_ITER,
    GREEN_LINE_MASK_DILATE_KERNEL,
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
    GREEN_REQUIRE_LINE_PROXIMITY,
    GREEN_S_MIN,
    GREEN_V_MIN,
)


def criar_mascara_linha_global(resultado_linha):
    """Reconstrói e dilata a máscara de linha na coordenada global do frame."""
    altura = resultado_linha["altura"]
    largura = resultado_linha["largura"]
    mascara_global = np.zeros((altura, largura), dtype=np.uint8)
    y_inicio = resultado_linha["y_inicio_roi"]
    y_fim = resultado_linha["y_fim_roi"]
    x_inicio = resultado_linha["x_inicio_roi"]
    x_fim = resultado_linha["x_fim_roi"]
    mascara_global[y_inicio:y_fim, x_inicio:x_fim] = resultado_linha["mascara_limpa"]
    kernel = np.ones(
        (GREEN_LINE_MASK_DILATE_KERNEL, GREEN_LINE_MASK_DILATE_KERNEL), dtype=np.uint8
    )
    return cv2.dilate(mascara_global, kernel, iterations=GREEN_LINE_MASK_DILATE_ITER)


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


def confirmar_contorno_por_linha(
    contorno_info, mascara_linha_global, altura_frame, largura_frame
):
    """Confirma se um contorno aprovado por cor esta associado a linha preta."""
    x, y, largura, altura = contorno_info["bbox"]
    y_zona_inicio = int(altura_frame * GREEN_CONFIRM_Y_MIN_REL)
    y_zona_fim = int(altura_frame * GREEN_CONFIRM_Y_MAX_REL)
    altura_intersecao = max(0, min(y + altura, y_zona_fim) - max(y, y_zona_inicio))
    area_in_confirm_zone_ratio = altura_intersecao / altura

    x_inicio = max(0, x - GREEN_CONFIRM_MAX_DIST_BLACK_PX)
    x_fim = min(largura_frame, x + largura + GREEN_CONFIRM_MAX_DIST_BLACK_PX)
    y_inicio = max(0, y - GREEN_CONFIRM_MAX_DIST_BLACK_PX)
    y_fim = min(altura_frame, y + altura + GREEN_CONFIRM_MAX_DIST_BLACK_PX)
    black_near_pixels = int(cv2.countNonZero(mascara_linha_global[y_inicio:y_fim, x_inicio:x_fim]))

    if area_in_confirm_zone_ratio < GREEN_CONFIRM_MIN_AREA_IN_ZONE_RATIO:
        return False, "fora_zona_confirmacao", black_near_pixels, area_in_confirm_zone_ratio
    if GREEN_REQUIRE_LINE_PROXIMITY and black_near_pixels < GREEN_CONFIRM_MIN_BLACK_NEAR_PIXELS:
        return False, "sem_linha_preta_proxima", black_near_pixels, area_in_confirm_zone_ratio
    return True, "confirmado_linha_proxima", black_near_pixels, area_in_confirm_zone_ratio


def detectar_verde(frame_bgr, x_referencia=None, mascara_linha_global=None):
    """Detecta verde por cor e o confirma geometricamente pela linha preta."""
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
    if mascara_linha_global is not None:
        if mascara_linha_global.shape[:2] != (altura, largura):
            raise ValueError("mascara_linha_global deve ter o mesmo tamanho do frame.")
        for contorno in contornos:
            confirmado, motivo, black_pixels, area_zona = confirmar_contorno_por_linha(
                contorno, mascara_linha_global, altura, largura
            )
            contorno.update(
                {
                    "confirmado": confirmado,
                    "motivo_confirmacao": motivo,
                    "black_near_pixels": black_pixels,
                    "area_in_confirm_zone_ratio": area_zona,
                }
            )
    else:
        for contorno in contornos:
            contorno.update(
                {
                    "confirmado": False,
                    "motivo_confirmacao": "sem_mascara_linha",
                    "black_near_pixels": 0,
                    "area_in_confirm_zone_ratio": 0.0,
                }
            )

    contornos_confirmados = [item for item in contornos if item["confirmado"]]
    resumo_detectado = resumir_verde(contornos)
    resumo_confirmado = resumir_verde(contornos_confirmados)
    if mascara_linha_global is None:
        observacao = "sem_mascara_linha"
    elif contornos_confirmados:
        observacao = resumo_confirmado["observacao"]
    elif contornos:
        motivos = {item["motivo_confirmacao"] for item in contornos}
        observacao = motivos.pop() if len(motivos) == 1 else "contornos_nao_confirmados"
    else:
        observacao = resumo_confirmado["observacao"]

    return {
        **resumo_confirmado,
        "tipo_detectado": resumo_detectado["tipo"],
        "tipo_confirmado": resumo_confirmado["tipo"],
        "tipo": resumo_confirmado["tipo"],
        "confirmado": resumo_confirmado["tipo"] != "NENHUM",
        "observacao": observacao,
        "contornos": contornos,
        "contornos_confirmados": contornos_confirmados,
        "qtd_contornos_detectados": len(contornos),
        "qtd_contornos_confirmados": len(contornos_confirmados),
        "mascara": mascara_resultado["mascara"],
        "mascara_linha_global": mascara_linha_global,
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

    mascara_linha = resultado_verde.get("mascara_linha_global")
    if mascara_linha is not None:
        overlay = debug.copy()
        overlay[mascara_linha > 0] = (255, 255, 0)
        debug = cv2.addWeighted(debug, 0.78, overlay, 0.22, 0)

    cv2.rectangle(debug, (0, y_inicio), (largura - 1, y_fim - 1), (255, 255, 0), 2)
    y_confirm_inicio = int(altura * GREEN_CONFIRM_Y_MIN_REL)
    y_confirm_fim = int(altura * GREEN_CONFIRM_Y_MAX_REL)
    cv2.rectangle(debug, (0, y_confirm_inicio), (largura - 1, y_confirm_fim), (255, 0, 255), 1)
    cv2.line(debug, (x_referencia, 0), (x_referencia, altura - 1), (0, 255, 255), 2)
    cv2.rectangle(
        debug,
        (max(0, x_referencia - margem), y_inicio),
        (min(largura - 1, x_referencia + margem), y_fim - 1),
        (0, 165, 255),
        1,
    )
    slot_largura = int(largura * GREEN_EXPECTED_SLOT_WIDTH_REL)
    slot_altura = int(altura * GREEN_EXPECTED_SLOT_HEIGHT_REL)
    slot_y = int(altura * GREEN_EXPECTED_SLOT_Y_MIN_REL)
    slot_y_fim = min(altura - 1, slot_y + slot_altura)
    cv2.rectangle(
        debug,
        (max(0, x_referencia - margem - slot_largura), slot_y),
        (max(0, x_referencia - margem), slot_y_fim),
        (128, 128, 128),
        1,
    )
    cv2.rectangle(
        debug,
        (min(largura - 1, x_referencia + margem), slot_y),
        (min(largura - 1, x_referencia + margem + slot_largura), slot_y_fim),
        (128, 128, 128),
        1,
    )

    cores = {"ESQUERDA": (0, 255, 0), "DIREITA": (255, 0, 0), "CENTRO": (0, 165, 255)}
    rotulos = {"ESQUERDA": "E", "DIREITA": "D", "CENTRO": "C"}
    for contorno in resultado_verde["contornos"]:
        x, y, w, h = contorno["bbox"]
        lado = contorno["lado"]
        cor = cores[lado] if contorno["confirmado"] else (0, 255, 255)
        rotulo = rotulos[lado] if contorno["confirmado"] else f"{rotulos[lado]}?"
        cv2.rectangle(debug, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(debug, rotulo, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        metricas = (
            f"{rotulos[lado]} S:{contorno['mean_s']:.0f} "
            f"GR:{contorno['g_minus_r']:.0f} GB:{contorno['g_minus_b']:.0f}"
        )
        cv2.putText(debug, metricas, (x, min(altura - 25, y + h + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
        motivo = "ok" if contorno["confirmado"] else contorno["motivo_confirmacao"]
        cv2.putText(debug, motivo, (x, min(altura - 8, y + h + 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

    linhas = [
        f"verde final: {resultado_verde['tipo_confirmado']} | detectado: {resultado_verde['tipo_detectado']}",
        f"confirmados: {resultado_verde['qtd_contornos_confirmados']}/{resultado_verde['qtd_contornos_detectados']} | conf: {resultado_verde['confianca']:.2f}",
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
