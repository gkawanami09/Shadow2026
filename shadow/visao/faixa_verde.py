"""Mede a faixa transversal usada para posicionar o giro verde."""

import cv2
import numpy as np

import config


def _preparar_faixas(mascara_preta):
    mascara = np.asarray(mascara_preta)
    if mascara.ndim != 2 or mascara.size == 0:
        return None
    _altura, largura = mascara.shape
    meio = largura / 2.
    tolerancia_centro = config.GREEN_BRANCH_CENTER_TOLERANCE_PX
    comprimento_minimo = config.GREEN_BRANCH_TRANSVERSE_MIN_RUN_PX
    kernel = np.ones((1, comprimento_minimo), dtype=np.uint8)
    horizontal = cv2.morphologyEx(
        mascara.astype(np.uint8, copy=False),
        cv2.MORPH_OPEN,
        kernel,
    )
    centro_inicio = max(int(meio - tolerancia_centro), 0)
    centro_fim = min(int(meio + tolerancia_centro) + 1, largura)
    toca_eixo = np.any(
        horizontal[:, centro_inicio:centro_fim] > 0,
        axis=1,
    )
    return horizontal, toca_eixo, meio, comprimento_minimo


def tem_ramo_lateral(mascara_preta):
    """Detecta um T/cruz ligado ao eixo, mesmo alcançando apenas um lado."""
    preparado = _preparar_faixas(mascara_preta)
    if preparado is None:
        return False
    horizontal, toca_eixo, meio, comprimento_minimo = preparado
    largura = horizontal.shape[1]
    limite_esquerdo = max(int(meio - comprimento_minimo), 1)
    limite_direito = min(
        int(meio + comprimento_minimo), largura - 1)
    alcanca_esquerda = np.any(
        horizontal[:, :limite_esquerdo] > 0, axis=1)
    alcanca_direita = np.any(
        horizontal[:, limite_direito:] > 0, axis=1)
    return bool(np.any(
        toca_eixo & (alcanca_esquerda | alcanca_direita)))


def altura_faixa_transversal(mascara_preta, direcao):
    """Retorna o centro Y da faixa horizontal ligada ao eixo, ou ``-1``.

    Uma curva em U pode possuir extremos distantes no mesmo Y, mas eles sao
    separados por piso branco. Por isso medimos trechos pretos *continuos* em
    cada linha da mascara, e nao apenas a largura total do contorno.
    """
    if direcao not in ("left", "right"):
        return -1.

    preparado = _preparar_faixas(mascara_preta)
    if preparado is None:
        return -1.
    horizontal, toca_eixo, meio, comprimento_minimo = preparado
    largura = horizontal.shape[1]
    if direcao == "left":
        limite_lado = max(int(meio - comprimento_minimo), 1)
        alcanca_lado = np.any(horizontal[:, :limite_lado] > 0, axis=1)
    else:
        limite_lado = min(int(meio + comprimento_minimo), largura - 1)
        alcanca_lado = np.any(horizontal[:, limite_lado:] > 0, axis=1)

    linhas_validas = np.flatnonzero(toca_eixo & alcanca_lado)
    if linhas_validas.size == 0:
        return -1.
    comprimentos = np.count_nonzero(horizontal, axis=1)

    grupos = []
    inicio = 0
    for indice in range(1, linhas_validas.size):
        if linhas_validas[indice] - linhas_validas[indice - 1] > 2:
            grupos.append((inicio, indice))
            inicio = indice
    grupos.append((inicio, linhas_validas.size))
    melhor_grupo = max(
        grupos,
        key=lambda intervalo: int(np.max(
            comprimentos[linhas_validas[slice(*intervalo)]])),
    )
    return float(np.median(linhas_validas[slice(*melhor_grupo)]))
