"""Faixa PRETA de saída da sala de resgate, vista pela CÂMERA DE RESGATE.

Risco central deste detector: a arena contém uma vítima PRETA. Cor sozinha
não distingue as duas. A separação implementada aqui é geométrica:

* a vítima é compacta — proporção largura/espessura próxima de 1;
* a faixa é alongada e atravessa o campo de visão.

O veto de proporção (``EXIT_BLACK_MIN_ASPECT``) é o que impede que a esfera
preta seja lida como soleira de saída. Além dele, o contraste com o piso é
medido COM SINAL: o entorno precisa ser mais claro que a faixa, o que rejeita
uma sombra sobre piso já escuro.

Este módulo é consultado exclusivamente no estado ``FIND_BLACK_EXIT``. Quem
garante isso é o coordenador da missão, que só instancia o detector nesse
estado — fora dele a faixa preta não pode interromper a busca de vítimas.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

import config_resgate as cfg
from visao.faixa_transversal import (BandGeometry, StripeConfirmer,
                                     find_transversal_band)


@dataclass(frozen=True)
class ExitStripeDetection:
    """Candidato de faixa preta já aprovado em forma e contraste."""

    center_x: float
    center_y: float
    width: int
    height: int
    top_y: int
    bottom_y: int
    span_ratio: float
    thickness_ratio: float
    aspect: float
    value: float
    surround_contrast: float
    confidence: float
    timestamp: float
    bbox: tuple


def default_geometry():
    return BandGeometry(
        min_row_fill=cfg.EXIT_BLACK_MIN_ROW_FILL,
        min_span_ratio=cfg.EXIT_BLACK_MIN_SPAN_RATIO,
        max_span_ratio=cfg.EXIT_BLACK_MAX_SPAN_RATIO,
        min_thickness_ratio=cfg.EXIT_BLACK_MIN_THICKNESS_RATIO,
        max_thickness_ratio=cfg.EXIT_BLACK_MAX_THICKNESS_RATIO,
        min_fill_ratio=cfg.EXIT_BLACK_MIN_FILL_RATIO,
        min_aspect=cfg.EXIT_BLACK_MIN_ASPECT,
    )


def dark_mask(frame_bgr, hsv_min=None, hsv_max=None):
    """Máscara escura da faixa. Função pura, reutilizável no replay."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.asarray(
            cfg.EXIT_BLACK_HSV_MIN if hsv_min is None else hsv_min,
            dtype=np.uint8),
        np.asarray(
            cfg.EXIT_BLACK_HSV_MAX if hsv_max is None else hsv_max,
            dtype=np.uint8),
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


class BlackExitDetector:
    """Encontra a soleira preta de saída em um frame da câmera de resgate."""

    def __init__(self, hsv_min=None, hsv_max=None, geometry=None):
        self.hsv_min = tuple(
            cfg.EXIT_BLACK_HSV_MIN if hsv_min is None else hsv_min)
        self.hsv_max = tuple(
            cfg.EXIT_BLACK_HSV_MAX if hsv_max is None else hsv_max)
        self.geometry = default_geometry() if geometry is None else geometry
        self.last_reason = "inicio"
        self.last_mask = None
        self.last_band = None

    def detect(self, frame_bgr, timestamp=None):
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("BlackExitDetector exige um frame BGR")

        timestamp = 0.0 if timestamp is None else float(timestamp)
        self.last_band = None

        mask = dark_mask(frame_bgr, self.hsv_min, self.hsv_max)
        self.last_mask = mask

        band, reason = find_transversal_band(
            mask,
            self.geometry,
            roi_top_ratio=cfg.EXIT_BLACK_ROI_TOP,
            roi_bottom_ratio=cfg.EXIT_BLACK_ROI_BOTTOM,
        )
        if band is None:
            perspectiva = self._detect_perspective_line(
                frame_bgr, timestamp)
            if perspectiva is not None:
                self.last_reason = ""
                return perspectiva
            # "compacta" é o motivo devolvido quando uma esfera preta chega
            # até aqui: ela é escura e está na ROI, mas não é alongada.
            self.last_reason = reason
            return None
        self.last_band = band

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)

        inside = np.zeros(mask.shape, dtype=bool)
        inside[band.top_y:band.bottom_y + 1,
               band.left_x:band.right_x + 1] = True
        inside &= mask > 0
        if not inside.any():
            self.last_reason = "faixa_vazia"
            return None

        inside_value = float(np.median(value[inside]))
        if inside_value > cfg.EXIT_BLACK_MAX_INSIDE_VALUE:
            self.last_reason = "clara"
            return None

        surround_contrast = self._surround_contrast(value, band, inside_value)
        if surround_contrast < cfg.EXIT_BLACK_MIN_SURROUND_CONTRAST:
            self.last_reason = "sem_contraste"
            return None

        confidence = self._confidence(band, inside_value, surround_contrast)
        if confidence < cfg.EXIT_BLACK_MIN_CONFIDENCE:
            self.last_reason = "confianca"
            return None

        self.last_reason = ""
        return ExitStripeDetection(
            center_x=band.center_x,
            center_y=band.center_y,
            width=band.width,
            height=band.height,
            top_y=band.top_y,
            bottom_y=band.bottom_y,
            span_ratio=band.span_ratio,
            thickness_ratio=band.thickness_ratio,
            aspect=band.aspect,
            value=inside_value,
            surround_contrast=surround_contrast,
            confidence=confidence,
            timestamp=timestamp,
            bbox=band.bbox,
        )

    @staticmethod
    def _detect_perspective_line(frame_bgr, timestamp):
        """Encontra a soleira fina e inclinada vista de longe.

        O caminho principal continua sendo a mascara preta. Este fallback so
        procura segmentos na parte inferior, exige comprimento proporcional
        e confirma que pelo menos um dos lados da borda e escuro. A borda
        curva de uma esfera nao produz um trecho reto desse comprimento.
        """
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            cfg.EXIT_LINE_CANNY_LOW,
            cfg.EXIT_LINE_CANNY_HIGH,
        )

        top = int(round(height * cfg.EXIT_LINE_ROI_TOP))
        bottom = int(round(height * cfg.EXIT_LINE_ROI_BOTTOM))
        edges[:top, :] = 0
        edges[bottom:, :] = 0

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(
                12, int(round(width * cfg.EXIT_LINE_HOUGH_THRESHOLD_RATIO))),
            minLineLength=max(
                20, int(round(width * cfg.EXIT_LINE_MIN_LENGTH_RATIO))),
            maxLineGap=max(
                2, int(round(width * cfg.EXIT_LINE_MAX_GAP_RATIO))),
        )
        if lines is None:
            return None

        candidates = []
        margin = max(3, int(round(height * 0.018)))
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < width * cfg.EXIT_LINE_MIN_LENGTH_RATIO:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) > cfg.EXIT_LINE_MAX_ANGLE_DEG:
                continue

            center_y = (float(y1) + float(y2)) / 2.0
            center_y_px = int(round(center_y))
            left = max(min(int(x1), int(x2)), 0)
            right = min(max(int(x1), int(x2)) + 1, width)
            above = value[
                max(center_y_px - margin, 0):center_y_px,
                left:right,
            ]
            below = value[
                center_y_px:min(center_y_px + margin, height),
                left:right,
            ]
            medians = [
                float(np.median(sample)) for sample in (above, below)
                if sample.size
            ]
            if not medians:
                continue
            dark_value = min(medians)
            if dark_value > cfg.EXIT_LINE_MAX_DARK_SIDE_VALUE:
                continue
            contrast = (
                abs(medians[0] - medians[-1])
                if len(medians) > 1 else 0.0)

            y_ratio = center_y / max(float(height), 1.0)
            length_ratio = length / max(float(width), 1.0)
            score = (
                length_ratio
                + 0.80 * y_ratio
                - 0.30 * abs(angle)
                / max(cfg.EXIT_LINE_MAX_ANGLE_DEG, 1.0)
            )
            candidates.append(
                (score, x1, y1, x2, y2, length, dark_value, contrast))

        if not candidates:
            return None

        _, x1, y1, x2, y2, length, dark_value, contrast = max(
            candidates, key=lambda item: item[0])
        left = max(min(int(x1), int(x2)), 0)
        right = min(max(int(x1), int(x2)) + 1, width)
        center_y = int(round((float(y1) + float(y2)) / 2.0))
        span_ratio = abs(float(x2 - x1)) / max(float(width), 1.0)
        thickness = margin * 2 + 1
        confidence = _clip01(
            0.55
            + 0.30 * (span_ratio - cfg.EXIT_LINE_MIN_LENGTH_RATIO)
            + 0.15 * contrast / 80.0
        )
        top_y = max(center_y - margin, 0)
        bottom_y = min(center_y + margin, height - 1)
        box_width = max(right - left, 1)

        return ExitStripeDetection(
            center_x=(float(x1) + float(x2)) / 2.0,
            center_y=(float(y1) + float(y2)) / 2.0,
            width=box_width,
            height=bottom_y - top_y + 1,
            top_y=top_y,
            bottom_y=bottom_y,
            span_ratio=span_ratio,
            thickness_ratio=thickness / max(float(height), 1.0),
            aspect=length / max(float(thickness), 1.0),
            value=dark_value,
            surround_contrast=contrast,
            confidence=confidence,
            timestamp=float(timestamp),
            bbox=(left, top_y, box_width, bottom_y - top_y + 1),
        )

    @staticmethod
    def _surround_contrast(value, band, inside_value):
        """Entorno menos faixa, COM SINAL: o piso precisa ser mais claro."""
        height = value.shape[0]
        margin = max(
            int(round(height * cfg.EXIT_BLACK_SURROUND_MARGIN_RATIO)), 1)
        samples = []
        above_top = max(band.top_y - margin, 0)
        if above_top < band.top_y:
            samples.append(
                value[above_top:band.top_y, band.left_x:band.right_x + 1])
        below_bottom = min(band.bottom_y + 1 + margin, height)
        if below_bottom > band.bottom_y + 1:
            samples.append(
                value[band.bottom_y + 1:below_bottom,
                      band.left_x:band.right_x + 1])
        contrasts = [
            float(np.median(sample)) - inside_value
            for sample in samples if sample.size
        ]
        return max(contrasts) if contrasts else 0.0

    @staticmethod
    def _confidence(band, inside_value, surround_contrast):
        span_score = _clip01(
            (band.span_ratio - cfg.EXIT_BLACK_MIN_SPAN_RATIO)
            / max(1.0 - cfg.EXIT_BLACK_MIN_SPAN_RATIO, 1e-6))
        fill_score = _clip01(
            (band.fill_ratio - cfg.EXIT_BLACK_MIN_FILL_RATIO)
            / max(1.0 - cfg.EXIT_BLACK_MIN_FILL_RATIO, 1e-6))
        dark_score = _clip01(
            1.0 - inside_value / max(cfg.EXIT_BLACK_MAX_INSIDE_VALUE, 1e-6))
        contrast_score = _clip01(
            surround_contrast
            / max(cfg.EXIT_BLACK_MIN_SURROUND_CONTRAST * 3.0, 1e-6))
        aspect_score = _clip01(
            band.aspect / max(cfg.EXIT_BLACK_MIN_ASPECT * 2.0, 1e-6))
        return float(
            0.28 * span_score
            + 0.16 * fill_score
            + 0.20 * dark_score
            + 0.24 * contrast_score
            + 0.12 * aspect_score
        )


class BlackExitGate:
    """Detector + votação temporal da soleira de saída."""

    def __init__(self, detector=None, confirmer=None, max_age_s=None):
        self.detector = BlackExitDetector() if detector is None else detector
        self.confirmer = (
            StripeConfirmer(
                votes_needed=cfg.EXIT_BLACK_VOTES_NEEDED,
                window=cfg.EXIT_BLACK_VOTE_WINDOW,
                max_age_s=(
                    cfg.BALL_FRAME_STALE_S if max_age_s is None
                    else float(max_age_s)),
                cooldown_s=cfg.EXIT_BLACK_COOLDOWN_S,
            )
            if confirmer is None else confirmer
        )
        self.last_detection = None

    @property
    def confirmed(self):
        return self.confirmer.confirmed

    @property
    def votes(self):
        return self.confirmer.votes

    def reset(self, now=None):
        self.confirmer.reset(now=now)
        self.last_detection = None

    def update(self, frame_bgr, timestamp=None, now=None):
        detection = self.detector.detect(frame_bgr, timestamp=timestamp)
        self.last_detection = detection
        confirmed = self.confirmer.update(
            detection is not None,
            timestamp=0.0 if timestamp is None else timestamp,
            now=now,
        )
        return confirmed, detection
