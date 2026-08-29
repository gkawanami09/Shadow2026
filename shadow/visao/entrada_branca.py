"""Geometria da entrada do resgate por corredor branco entre laterais pretas."""

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class MedicaoEntradaBranca:
    candidata: bool
    preto_esquerda: float
    preto_centro: float
    preto_direita: float


def _razao_preto(mask, x0, x1, y0, y1):
    recorte = mask[y0:y1, x0:x1]
    if not recorte.size:
        return 0.0
    return float(np.count_nonzero(recorte) / recorte.size)


def detectar_entrada_branca(black_mask, *, linha_a_frente=False):
    """Exige preto nas duas laterais e centro branco, sem linha adiante."""
    if black_mask is None or black_mask.ndim != 2:
        raise ValueError("detectar_entrada_branca exige mascara preta 2D")
    height, width = black_mask.shape
    y0 = int(height * config.ENTRY_WHITE_GATE_Y_MIN)
    y1 = int(height * config.ENTRY_WHITE_GATE_Y_MAX)

    def x(fracao):
        return int(width * fracao)

    esquerda = _razao_preto(
        black_mask,
        x(config.ENTRY_WHITE_GATE_LEFT_X_MIN),
        x(config.ENTRY_WHITE_GATE_LEFT_X_MAX), y0, y1)
    centro = _razao_preto(
        black_mask,
        x(config.ENTRY_WHITE_GATE_CENTER_X_MIN),
        x(config.ENTRY_WHITE_GATE_CENTER_X_MAX), y0, y1)
    direita = _razao_preto(
        black_mask,
        x(config.ENTRY_WHITE_GATE_RIGHT_X_MIN),
        x(config.ENTRY_WHITE_GATE_RIGHT_X_MAX), y0, y1)
    candidata = bool(
        not linha_a_frente
        and esquerda >= config.ENTRY_WHITE_GATE_SIDE_MIN_BLACK_RATIO
        and direita >= config.ENTRY_WHITE_GATE_SIDE_MIN_BLACK_RATIO
        and centro <= config.ENTRY_WHITE_GATE_CENTER_MAX_BLACK_RATIO
    )
    return MedicaoEntradaBranca(candidata, esquerda, centro, direita)
