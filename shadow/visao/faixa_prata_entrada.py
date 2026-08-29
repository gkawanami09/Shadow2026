"""Deteccao leve da faixa prata da entrada na camera de linha.

Nao classifica uma cor isolada como prata. A decisao combina neutralidade,
contraste especular, largura transversal e a linha preta que chega ate a
faixa e termina ali.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class MedicaoFaixaPrata:
    score: int
    saturacao_media: float
    desvio_brilho: float
    pct_claro: float
    pct_escuro: float
    largura_ratio: float
    linha_fim: bool
    bbox: tuple[int, int, int, int] | None

    @property
    def candidata(self):
        return bool(
            self.bbox is not None
            and self.linha_fim
            and self.score >= config.ENTRY_SILVER_SCORE_MIN)


def _maior_faixa_contigua(rows):
    indices = np.flatnonzero(rows)
    if not indices.size:
        return None
    cortes = np.flatnonzero(np.diff(indices) > 1) + 1
    grupos = np.split(indices, cortes)
    grupo = max(grupos, key=len)
    return int(grupo[0]), int(grupo[-1]) + 1


def _linha_chega_e_termina(mask, bbox, line_aligned):
    if not line_aligned or mask is None or mask.ndim != 2 or bbox is None:
        return False
    height, width = mask.shape
    _x, y, _w, h = bbox
    x0 = int(width * config.ENTRY_SILVER_LINE_X_MIN)
    x1 = int(width * config.ENTRY_SILVER_LINE_X_MAX)
    if x1 <= x0:
        return False
    abaixo_inicio = min(height, max(0, y + h))
    abaixo_fim = min(height, abaixo_inicio + int(
        height * config.ENTRY_SILVER_LINE_APPROACH_RATIO))
    approach = mask[abaixo_inicio:abaixo_fim, x0:x1]
    if not approach.size:
        return False
    row_fill = np.count_nonzero(approach, axis=1) / approach.shape[1]
    min_rows = max(1, int(round(
        approach.shape[0] * config.ENTRY_SILVER_LINE_MIN_ROWS_RATIO)))
    chega = np.count_nonzero(
        row_fill >= config.ENTRY_SILVER_LINE_MIN_ROW_FILL) >= min_rows

    # Preto acima da faixa e' continuidade da pista, nao entrada da sala.
    acima = mask[:max(0, y - 2), x0:x1]
    continua = False
    if acima.size:
        fill_acima = np.count_nonzero(acima, axis=1) / acima.shape[1]
        continua = bool(np.count_nonzero(
            fill_acima >= config.ENTRY_SILVER_LINE_MIN_ROW_FILL)
            >= min_rows)
    return bool(chega and not continua)


def detectar_faixa_prata(frame_bgr, black_mask=None, *, line_aligned=False):
    """Mede a evidencia de uma faixa prata sem alterar o segue-linha."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame BGR invalido para detectar_faixa_prata")
    height, width = frame_bgr.shape[:2]
    x0 = int(width * config.ENTRY_SILVER_ROI_X_MIN)
    x1 = int(width * config.ENTRY_SILVER_ROI_X_MAX)
    y0 = int(height * config.ENTRY_SILVER_ROI_Y_MIN)
    y1 = int(height * config.ENTRY_SILVER_ROI_Y_MAX)
    roi = frame_bgr[y0:y1, x0:x1]
    if not roi.size:
        return MedicaoFaixaPrata(0, 255., 0., 0., 0., 0., False, None)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    low_sat = saturation <= config.ENTRY_SILVER_MAX_SATURATION
    bright = value >= config.ENTRY_SILVER_BRIGHT_VALUE
    dark = value <= config.ENTRY_SILVER_DARK_VALUE
    neutral_contrast = low_sat & (bright | dark)
    row_fill = np.mean(neutral_contrast, axis=1)
    bright_row = np.mean(low_sat & bright, axis=1)
    dark_row = np.mean(low_sat & dark, axis=1)
    band = _maior_faixa_contigua(
        (row_fill >= config.ENTRY_SILVER_MIN_CONTRAST_ROW_FILL)
        & (bright_row >= config.ENTRY_SILVER_MIN_BRIGHT_RATIO)
        & (dark_row >= config.ENTRY_SILVER_MIN_DARK_ROW_RATIO))

    bbox = None
    width_ratio = 0.
    if band is not None:
        band_y0, band_y1 = band
        min_height = max(1, int(round(
            height * config.ENTRY_SILVER_MIN_BAND_HEIGHT_RATIO)))
        if band_y1 - band_y0 >= min_height:
            active = neutral_contrast[band_y0:band_y1]
            columns = np.flatnonzero(np.mean(active, axis=0) >= .08)
            if columns.size:
                left, right = int(columns[0]), int(columns[-1]) + 1
                width_ratio = (right - left) / width
                bbox = (x0 + left, y0 + band_y0,
                        right - left, band_y1 - band_y0)

    pct_bright = float(np.mean(bright))
    pct_dark = float(np.mean(dark))
    sat_mean = float(np.mean(saturation))
    std_value = float(np.std(value))
    line_end = _linha_chega_e_termina(black_mask, bbox, line_aligned)

    score = 0
    score += int(sat_mean <= config.ENTRY_SILVER_MAX_SATURATION)
    score += 2 * int(std_value >= config.ENTRY_SILVER_MIN_STD_VALUE)
    score += int(pct_bright >= config.ENTRY_SILVER_MIN_BRIGHT_RATIO)
    score += int(pct_dark >= config.ENTRY_SILVER_MIN_DARK_RATIO)
    score += int(
        pct_bright >= config.ENTRY_SILVER_MIN_BRIGHT_RATIO
        and pct_dark >= config.ENTRY_SILVER_MIN_DARK_RATIO)
    score += 2 * int(width_ratio >= config.ENTRY_SILVER_MIN_WIDE_RATIO)
    score += 3 * int(line_end)
    return MedicaoFaixaPrata(
        score, sat_mean, std_value, pct_bright, pct_dark,
        width_ratio, line_end, bbox)
