"""Logica visual para decidir se verde confirmado e acionavel na intersecao."""

import cv2

from config import (
    GREEN_ACTION_CENTER_ZONE_WIDTH_REL,
    GREEN_ACTION_DUPLO_MIN_BALANCE_RATIO,
    GREEN_ACTION_GREEN_MUST_BE_BELOW_INTERSECTION,
    GREEN_ACTION_INTERSECTION_BAND_HEIGHT_REL,
    GREEN_ACTION_MAX_GREEN_ABOVE_INTERSECTION_RATIO,
    GREEN_ACTION_MIN_BLACK_PIXELS_CENTER,
    GREEN_ACTION_MIN_BLACK_PIXELS_LEFT,
    GREEN_ACTION_MIN_BLACK_PIXELS_RIGHT,
    GREEN_ACTION_MIN_BLACK_RATIO_CENTER,
    GREEN_ACTION_MIN_BLACK_RATIO_LEFT,
    GREEN_ACTION_MIN_BLACK_RATIO_RIGHT,
    GREEN_ACTION_MIN_CONFIRMED_GREEN_AREA,
    GREEN_ACTION_SIDE_ZONE_WIDTH_REL,
    GREEN_ACTION_Y_NEAR_MAX_REL,
    GREEN_ACTION_Y_NEAR_MIN_REL,
)


def dividir_zonas_intersecao(largura, altura, x_referencia):
    """Define zonas laterais e central dentro da faixa baixa de intersecao."""
    y2 = int(altura * GREEN_ACTION_Y_NEAR_MAX_REL)
    y1 = max(
        int(altura * GREEN_ACTION_Y_NEAR_MIN_REL),
        y2 - int(altura * GREEN_ACTION_INTERSECTION_BAND_HEIGHT_REL),
    )
    # A largura configurada representa cada metade do corredor central. Assim,
    # a linha principal e sua dilatacao nao vazam para as zonas laterais.
    largura_centro = int(largura * GREEN_ACTION_CENTER_ZONE_WIDTH_REL) * 2
    largura_lado = int(largura * GREEN_ACTION_SIDE_ZONE_WIDTH_REL)
    centro_inicio = max(0, x_referencia - largura_centro // 2)
    centro_fim = min(largura, x_referencia + largura_centro // 2)
    esquerda_inicio = max(0, centro_inicio - largura_lado)
    direita_fim = min(largura, centro_fim + largura_lado)
    return {
        "zona_esquerda": (esquerda_inicio, y1, centro_inicio, y2),
        "zona_centro": (centro_inicio, y1, centro_fim, y2),
        "zona_direita": (centro_fim, y1, direita_fim, y2),
        "zona_intersecao": (0, y1, largura, y2),
    }


def contar_pixels_zona(mascara, zona):
    """Conta pixels nao-zero e preenchimento da zona informada."""
    x1, y1, x2, y2 = zona
    recorte = mascara[y1:y2, x1:x2]
    area_zona = recorte.size
    pixels = int(cv2.countNonZero(recorte))
    return {"pixels": pixels, "ratio": pixels / area_zona if area_zona else 0.0, "area_zona": area_zona}


def analisar_intersecao_preta(mascara_linha_global, x_referencia):
    """Classifica a estrutura preta atual sem interpretar movimento."""
    altura, largura = mascara_linha_global.shape[:2]
    zonas = dividir_zonas_intersecao(largura, altura, x_referencia)
    esquerda = contar_pixels_zona(mascara_linha_global, zonas["zona_esquerda"])
    centro = contar_pixels_zona(mascara_linha_global, zonas["zona_centro"])
    direita = contar_pixels_zona(mascara_linha_global, zonas["zona_direita"])
    left = esquerda["pixels"] >= GREEN_ACTION_MIN_BLACK_PIXELS_LEFT and esquerda["ratio"] >= GREEN_ACTION_MIN_BLACK_RATIO_LEFT
    center = centro["pixels"] >= GREEN_ACTION_MIN_BLACK_PIXELS_CENTER and centro["ratio"] >= GREEN_ACTION_MIN_BLACK_RATIO_CENTER
    right = direita["pixels"] >= GREEN_ACTION_MIN_BLACK_PIXELS_RIGHT and direita["ratio"] >= GREEN_ACTION_MIN_BLACK_RATIO_RIGHT
    if center and left and right:
        tipo = "CRUZ"
    elif center and left:
        tipo = "LATERAL_ESQ"
    elif center and right:
        tipo = "LATERAL_DIR"
    elif center:
        tipo = "RETA"
    elif left or right:
        tipo = "AMBIGUA"
    else:
        tipo = "NENHUMA"
    return {
        "intersecao_detectada": tipo in {"CRUZ", "LATERAL_ESQ", "LATERAL_DIR"},
        "tipo_intersecao": tipo,
        "black_center": centro["pixels"], "black_left": esquerda["pixels"], "black_right": direita["pixels"],
        "black_ratio_center": centro["ratio"], "black_ratio_left": esquerda["ratio"], "black_ratio_right": direita["ratio"],
        "center_presente": center, "left_presente": left, "right_presente": right, "zonas": zonas,
    }


def verde_esta_em_posicao_acionavel(contorno, analise_intersecao, altura_frame):
    """Avalia compatibilidade visual de um contorno verde confirmado."""
    x, y, w, h = contorno["bbox"]
    _, y_inter_1, _, _ = analise_intersecao["zonas"]["zona_intersecao"]
    altura_acima = max(0, min(y + h, y_inter_1) - y)
    ratio_acima = altura_acima / h if h else 0.0
    contorno["ratio_acima_intersecao"] = ratio_acima
    contorno["possivel_verde_depois_intersecao"] = (
        GREEN_ACTION_GREEN_MUST_BE_BELOW_INTERSECTION
        and ratio_acima > GREEN_ACTION_MAX_GREEN_ABOVE_INTERSECTION_RATIO
    )
    if not contorno.get("confirmado", False):
        return False, "verde_nao_confirmado"
    if contorno["area"] < GREEN_ACTION_MIN_CONFIRMED_GREEN_AREA:
        return False, "verde_area_pequena"
    if contorno.get("possivel_verde_depois_intersecao", False):
        return False, "verde_depois_intersecao"
    if contorno["lado"] == "CENTRO":
        return False, "verde_central_ambiguo"
    tipo = analise_intersecao["tipo_intersecao"]
    if not analise_intersecao["intersecao_detectada"]:
        return False, "sem_intersecao_atual"
    if (tipo == "LATERAL_ESQ" and contorno["lado"] != "ESQUERDA") or (tipo == "LATERAL_DIR" and contorno["lado"] != "DIREITA"):
        return False, "verde_lado_incompativel"
    return True, "verde_acionavel"


def decidir_verde_acionavel(resultado_verde, analise_intersecao):
    """Gera apenas intencao visual a partir da intersecao e verde confirmado."""
    confirmados = resultado_verde.get("contornos_confirmados", [])
    tipo_intersecao = analise_intersecao["tipo_intersecao"]
    for contorno in confirmados:
        acionavel, motivo = verde_esta_em_posicao_acionavel(contorno, analise_intersecao, 0)
        contorno["acionavel"] = acionavel
        contorno["motivo_acionavel"] = motivo

    base = {
        "intersecao_detectada": analise_intersecao["intersecao_detectada"],
        "tipo_intersecao": tipo_intersecao,
        "contornos_acionaveis": [],
        "qtd_contornos_acionaveis": 0,
        "area_acionavel_esquerda": 0.0,
        "area_acionavel_direita": 0.0,
        "area_acionavel_centro": 0.0,
        "analise_intersecao": analise_intersecao,
    }
    if tipo_intersecao in {"NENHUMA", "RETA"}:
        return {**base, "verde_acionavel": "NENHUM", "acao_visual": "SEGUIR_RETO", "motivo_acao": "sem_intersecao_atual"}
    if tipo_intersecao == "AMBIGUA":
        return {**base, "verde_acionavel": "NENHUM", "acao_visual": "SEGUIR_RETO", "motivo_acao": "intersecao_ambigua"}

    acionaveis = [item for item in confirmados if item["acionavel"]]
    areas = {"ESQUERDA": 0.0, "DIREITA": 0.0, "CENTRO": 0.0}
    for contorno in acionaveis:
        areas[contorno["lado"]] += contorno["area"]
    base.update({
        "contornos_acionaveis": acionaveis, "qtd_contornos_acionaveis": len(acionaveis),
        "area_acionavel_esquerda": areas["ESQUERDA"], "area_acionavel_direita": areas["DIREITA"], "area_acionavel_centro": areas["CENTRO"],
    })
    if not acionaveis:
        motivos_rejeicao = [item.get("motivo_acionavel") for item in confirmados]
        if "verde_depois_intersecao" in motivos_rejeicao:
            return {
                **base,
                "verde_acionavel": "NENHUM",
                "acao_visual": "SEGUIR_RETO",
                "motivo_acao": "verde_depois_intersecao",
            }
        somente_centro = confirmados and all(item["lado"] == "CENTRO" for item in confirmados)
        motivo = "verde_ambiguo" if somente_centro else "intersecao_sem_verde"
        verde = "AMBIGUO" if somente_centro else "NENHUM"
        return {**base, "verde_acionavel": verde, "acao_visual": "SEGUIR_RETO", "motivo_acao": motivo}
    if areas["ESQUERDA"] and areas["DIREITA"]:
        menor, maior = sorted((areas["ESQUERDA"], areas["DIREITA"]))
        if menor / maior >= GREEN_ACTION_DUPLO_MIN_BALANCE_RATIO:
            return {**base, "verde_acionavel": "DUPLO", "acao_visual": "PREPARAR_RETORNO", "motivo_acao": "verde_acionavel"}
        lado = "ESQUERDA" if areas["ESQUERDA"] > areas["DIREITA"] else "DIREITA"
        return {**base, "verde_acionavel": lado, "acao_visual": f"PREPARAR_{lado}", "motivo_acao": "duplo_desbalanceado"}
    lado = "ESQUERDA" if areas["ESQUERDA"] else "DIREITA"
    return {**base, "verde_acionavel": lado, "acao_visual": f"PREPARAR_{lado}", "motivo_acao": "verde_acionavel"}
