"""Extrai um ponto futuro da linha completa selecionada pela visao."""

from typing import NamedTuple

import cv2
import numpy as np

import config


class PontoFuturo(NamedTuple):
    x: float
    y: float
    valido: bool


def _intervalos(projecao):
    """Devolve intervalos inclusivos de pixels ativos em uma linha booleana."""
    indices = np.flatnonzero(projecao)
    if not indices.size:
        return []
    cortes = np.flatnonzero(np.diff(indices) > 1) + 1
    grupos = np.split(indices, cortes)
    return [(int(grupo[0]), int(grupo[-1])) for grupo in grupos]


def extrair_ponto_futuro(contorno, *, mascara_linha=None, origem_x=None):
    """Segue a faixa conectada de baixo para cima e escolhe um lookahead.

    Havendo mais de um ramo na mesma profundidade, continua pelo intervalo
    mais proximo do ramo escolhido na faixa anterior. Isso evita calcular a
    media entre os dois lados de um circulo ou de uma intersecao.
    """
    centro_robo = config.camera_x / 2.
    invalido = PontoFuturo(centro_robo, float(config.camera_y), False)
    if contorno is None or len(contorno) < 3:
        return invalido

    x, y, largura, altura = cv2.boundingRect(contorno)
    if (
        largura <= 0
        or altura < config.camera_y * config.LINE_PATH_MIN_VERTICAL_SPAN_RATIO
    ):
        return invalido

    if mascara_linha is not None:
        if mascara_linha.ndim != 2:
            raise ValueError("mascara_linha deve ser uma imagem 2D")
        mascara = np.asarray(
            mascara_linha[y:y + altura, x:x + largura], dtype=np.uint8)
    else:
        # Fallback para ferramentas/testes que possuem apenas o contorno.
        mascara = np.zeros((altura, largura), dtype=np.uint8)
        local = np.asarray(contorno, dtype=np.int32).copy()
        local[:, 0, 0] -= x
        local[:, 0, 1] -= y
        cv2.drawContours(mascara, [local], -1, 255, thickness=cv2.FILLED)

    bandas = int(config.LINE_PATH_SAMPLE_BANDS)
    limites = np.linspace(0, altura, bandas + 1, dtype=np.int32)
    anterior = float(
        centro_robo if origem_x is None else origem_x) - float(x)
    pontos = []
    ultimo_indice_ramificado = None
    falhas = 0

    for indice in range(bandas - 1, -1, -1):
        y0, y1 = int(limites[indice]), int(limites[indice + 1])
        if y1 <= y0:
            continue
        faixa = mascara[y0:y1]
        intervalos = _intervalos(np.any(faixa != 0, axis=0))
        if not intervalos:
            falhas += 1
            if pontos and falhas > config.LINE_PATH_MAX_BAND_GAP:
                break
            continue

        centros = [(.5 * (inicio + fim), inicio, fim)
                   for inicio, fim in intervalos]
        centro, inicio, fim = min(
            centros, key=lambda item: abs(item[0] - anterior))
        if (
            pontos
            and abs(centro - anterior)
            > config.LINE_PATH_MAX_LATERAL_JUMP_PX
        ):
            break

        regiao = faixa[:, inicio:fim + 1]
        ys, xs = np.nonzero(regiao)
        if not xs.size:
            continue
        ponto_x = float(x + inicio + np.mean(xs))
        ponto_y = float(y + y0 + np.mean(ys))
        pontos.append((ponto_x, ponto_y))
        if len(intervalos) > 1:
            ultimo_indice_ramificado = len(pontos) - 1
        anterior = ponto_x - x
        falhas = 0

    if len(pontos) < config.LINE_PATH_MIN_SAMPLES:
        return invalido

    # Em circulos/intersecoes, nao mire depois do ponto em que os dois ramos
    # voltam a se fundir: isso atravessaria o vazio entre eles. O verde ainda
    # decide qual ramo seguir; aqui apenas mantemos o alvo sobre a faixa.
    limite_futuro = (
        len(pontos) - 1
        if ultimo_indice_ramificado is None
        else ultimo_indice_ramificado
    )
    indice_futuro = int(round(
        limite_futuro * config.LINE_FUTURE_FRACTION))
    raio = int(config.LINE_FUTURE_SMOOTH_RADIUS)
    inicio = max(0, indice_futuro - raio)
    fim = min(limite_futuro + 1, indice_futuro + raio + 1)
    vizinhanca = np.asarray(pontos[inicio:fim], dtype=np.float64)
    return PontoFuturo(
        float(np.median(vizinhanca[:, 0])),
        float(np.median(vizinhanca[:, 1])),
        True,
    )
