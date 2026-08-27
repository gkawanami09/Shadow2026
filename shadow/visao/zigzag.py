"""Reconhece trechos curtos que alternam de lado e retornam ao eixo.

O detector usa somente a geometria do contorno preto selecionado pelo
segue-linha. Ele nao conhece motores nem guarda tempo: a confirmacao entre
frames pertence ao controlador.
"""

import cv2
import numpy as np

import config


def _centros_por_faixa(contorno, bandas):
    """Rasteriza apenas o bounding box e devolve centros de baixo para cima."""
    if contorno is None or len(contorno) < 3:
        vazia = np.empty(0, dtype=np.float64)
        return vazia, vazia, 0

    x, y, largura, altura = cv2.boundingRect(contorno)
    if largura <= 0 or altura <= 0:
        vazia = np.empty(0, dtype=np.float64)
        return vazia, vazia, 0

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    deslocado = np.asarray(contorno, dtype=np.int32).copy()
    deslocado[:, 0, 0] -= x
    deslocado[:, 0, 1] -= y
    cv2.drawContours(mascara, [deslocado], -1, 255, thickness=cv2.FILLED)

    centros = []
    larguras = []
    limites = np.linspace(0, altura, int(bandas) + 1, dtype=np.int32)
    # A ordem baixo->cima representa perto->longe na camera voltada ao piso.
    for indice in range(int(bandas) - 1, -1, -1):
        y0, y1 = int(limites[indice]), int(limites[indice + 1])
        if y1 <= y0:
            continue
        _, xs = np.nonzero(mascara[y0:y1])
        if xs.size:
            centros.append(float(x + np.mean(xs)))
            larguras.append(float(np.ptp(xs) + 1))

    return (
        np.asarray(centros, dtype=np.float64),
        np.asarray(larguras, dtype=np.float64),
        altura,
    )


def _amplitudes_alternadas(centros, ruido):
    """Agrupa deltas consecutivos de mesmo sinal em deslocamentos laterais."""
    if centros.size < 2:
        return []

    deltas = np.diff(centros)
    movimentos = []
    sinal_atual = 0
    acumulado = 0.

    for delta in deltas:
        if abs(delta) <= ruido:
            continue
        sinal = 1 if delta > 0 else -1
        if sinal_atual == 0 or sinal == sinal_atual:
            sinal_atual = sinal
            acumulado += float(delta)
            continue
        movimentos.append(acumulado)
        sinal_atual = sinal
        acumulado = float(delta)

    if sinal_atual:
        movimentos.append(acumulado)
    return movimentos


def detectar_zigzag(contorno):
    """Retorna True apenas para uma alternancia ampla que volta ao corredor."""
    if not config.ZIGZAG_ENABLED:
        return False

    centros, larguras, altura = _centros_por_faixa(
        contorno, config.ZIGZAG_SAMPLE_BANDS)
    if altura < config.camera_y * config.ZIGZAG_MIN_VERTICAL_SPAN_RATIO:
        return False

    minimo_centros = int(np.ceil(
        config.ZIGZAG_SAMPLE_BANDS * config.ZIGZAG_MIN_BAND_COVERAGE))
    if centros.size < minimo_centros:
        return False
    if np.max(larguras) > config.ZIGZAG_MAX_BAND_WIDTH_PX:
        return False

    movimentos = _amplitudes_alternadas(
        centros, config.ZIGZAG_DIRECTION_NOISE_PX)
    relevantes = [
        movimento for movimento in movimentos
        if abs(movimento) >= config.ZIGZAG_MIN_RUN_PX
    ]
    if len(relevantes) < config.ZIGZAG_MIN_RUNS:
        return False

    # Remover movimentos pequenos nao pode transformar dois movimentos de
    # mesmo lado em uma falsa alternancia.
    if any(a * b >= 0 for a, b in zip(relevantes, relevantes[1:])):
        return False

    if np.ptp(centros) < config.ZIGZAG_MIN_TOTAL_SPAN_PX:
        return False

    # O atalho reto so e seguro quando o trecho distante retorna para perto
    # do mesmo corredor lateral em que o robo entrou no desenho.
    return bool(
        abs(float(centros[-1] - centros[0]))
        <= config.ZIGZAG_MAX_NET_OFFSET_PX
    )
