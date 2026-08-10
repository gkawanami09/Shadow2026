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
    angle_deg: float = 0.0
    dark_support: float = 1.0
    contrast_support: float = 1.0


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

    def __init__(
        self,
        hsv_min=None,
        hsv_max=None,
        geometry=None,
        max_inside_value=None,
        min_surround_contrast=None,
        min_confidence=None,
    ):
        self.hsv_min = tuple(
            cfg.EXIT_BLACK_HSV_MIN if hsv_min is None else hsv_min)
        self.hsv_max = tuple(
            cfg.EXIT_BLACK_HSV_MAX if hsv_max is None else hsv_max)
        self.geometry = default_geometry() if geometry is None else geometry
        self.max_inside_value = (
            cfg.EXIT_BLACK_MAX_INSIDE_VALUE
            if max_inside_value is None else float(max_inside_value)
        )
        self.min_surround_contrast = (
            cfg.EXIT_BLACK_MIN_SURROUND_CONTRAST
            if min_surround_contrast is None
            else float(min_surround_contrast)
        )
        self.min_confidence = (
            cfg.EXIT_BLACK_MIN_CONFIDENCE
            if min_confidence is None else float(min_confidence)
        )
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
                if self._green_ratio(frame_bgr, perspectiva.bbox) > (
                    cfg.EXIT_BLACK_GREEN_VETO_MAX_RATIO
                ):
                    self.last_reason = "verde"
                    return None
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
        if inside_value > self.max_inside_value:
            self.last_reason = "clara"
            return None

        surround_contrast = self._surround_contrast(value, band, inside_value)
        if surround_contrast < self.min_surround_contrast:
            self.last_reason = "sem_contraste"
            return None

        confidence = self._confidence(band, inside_value, surround_contrast)
        if confidence < self.min_confidence:
            self.last_reason = "confianca"
            return None

        if self._green_ratio(frame_bgr, band.bbox) > (
            cfg.EXIT_BLACK_GREEN_VETO_MAX_RATIO
        ):
            self.last_reason = "verde"
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
            angle_deg=self._band_angle_deg(mask, band),
        )

    @staticmethod
    def _band_angle_deg(mask, band):
        """Mede a inclinação do eixo longo usando os pixels da própria fita."""
        recorte = mask[
            band.top_y:band.bottom_y + 1,
            band.left_x:band.right_x + 1,
        ]
        ys, xs = np.nonzero(recorte)
        if xs.size < 12:
            return 0.0
        pontos = np.column_stack((
            xs.astype(np.float64) + float(band.left_x),
            ys.astype(np.float64) + float(band.top_y),
        ))
        pontos -= np.mean(pontos, axis=0, keepdims=True)
        covariancia = np.cov(pontos, rowvar=False)
        valores, vetores = np.linalg.eigh(covariancia)
        eixo = vetores[:, int(np.argmax(valores))]
        angulo = math.degrees(math.atan2(float(eixo[1]), float(eixo[0])))
        while angulo > 90.0:
            angulo -= 180.0
        while angulo < -90.0:
            angulo += 180.0
        return float(angulo)

    @staticmethod
    def _green_ratio(frame_bgr, bbox):
        """Mede verde saturado dentro e ao redor do candidato escuro."""
        height, width = frame_bgr.shape[:2]
        x, y, w, h = bbox
        margin_x = max(
            int(round(width * cfg.EXIT_BLACK_GREEN_VETO_MARGIN_RATIO)),
            1,
        )
        margin_y = max(
            int(round(height * cfg.EXIT_BLACK_GREEN_VETO_MARGIN_RATIO)),
            int(h),
            1,
        )
        left = max(int(x) - margin_x, 0)
        right = min(int(x + w) + margin_x, width)
        top = max(int(y) - margin_y, 0)
        bottom = min(int(y + h) + margin_y, height)
        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.asarray(cfg.EXIT_BLACK_GREEN_VETO_HSV_MIN, dtype=np.uint8),
            np.asarray(cfg.EXIT_BLACK_GREEN_VETO_HSV_MAX, dtype=np.uint8),
        )
        return float(np.mean(green > 0))

    @staticmethod
    def _detect_perspective_line(frame_bgr, timestamp):
        """Encontra a soleira fina e inclinada vista de longe.

        O caminho principal continua sendo a mascara preta. Este fallback
        agrupa pedacos colineares da mesma borda. Um risco ou uma mancha
        isolada nao pode ser aprovado apenas por possuir um pequeno trecho
        reto: o conjunto precisa atravessar boa parte da imagem e manter um
        lado escuro e contrastante ao longo de quase toda a borda.
        """
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        value = cv2.cvtColor(
            frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
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
                20, int(round(width * cfg.EXIT_LINE_MIN_SEGMENT_RATIO))),
            maxLineGap=max(
                2, int(round(width * cfg.EXIT_LINE_MAX_GAP_RATIO))),
        )
        if lines is None:
            return None

        segments = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < width * cfg.EXIT_LINE_MIN_SEGMENT_RATIO:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) > cfg.EXIT_LINE_MAX_ANGLE_DEG:
                continue
            if abs(dx) < 1.0:
                continue
            slope = dy / dx
            segments.append({
                "x1": int(x1),
                "x2": int(x2),
                "slope": slope,
                "intercept": float(y1) - slope * float(x1),
                "angle": angle,
                "length": length,
            })

        if not segments:
            return None

        # O Hough costuma quebrar a mesma borda em varios pedacos por causa
        # dos reflexos. So juntamos pedacos com inclinacao e altura parecidas.
        groups = []
        center_x = float(width) / 2.0
        max_y_distance = (
            float(height) * cfg.EXIT_LINE_MAX_GROUP_Y_DISTANCE_RATIO)
        for segment in sorted(
                segments, key=lambda item: item["length"], reverse=True):
            segment_y = (
                segment["slope"] * center_x + segment["intercept"])
            selected = None
            for group in groups:
                group_y = group["slope"] * center_x + group["intercept"]
                if (
                    abs(segment["angle"] - group["angle"])
                    <= cfg.EXIT_LINE_MAX_GROUP_ANGLE_DIFF_DEG
                    and abs(segment_y - group_y) <= max_y_distance
                ):
                    selected = group
                    break
            if selected is None:
                selected = {
                    "segments": [],
                    "slope": segment["slope"],
                    "intercept": segment["intercept"],
                    "angle": segment["angle"],
                }
                groups.append(selected)
            selected["segments"].append(segment)
            weights = np.asarray(
                [item["length"] for item in selected["segments"]],
                dtype=np.float64,
            )
            selected["slope"] = float(np.average(
                [item["slope"] for item in selected["segments"]],
                weights=weights,
            ))
            selected["intercept"] = float(np.average(
                [item["intercept"] for item in selected["segments"]],
                weights=weights,
            ))
            selected["angle"] = math.degrees(
                math.atan(selected["slope"]))

        candidates = []
        join_gap = float(width) * cfg.EXIT_LINE_MAX_JOIN_GAP_RATIO
        margin = max(3, int(round(height * 0.018)))
        for group in groups:
            intervals = sorted(
                (item["x1"], item["x2"])
                for item in group["segments"]
            )
            joined = []
            for left, right in intervals:
                if joined and left <= joined[-1][1] + join_gap:
                    joined[-1][1] = max(joined[-1][1], right)
                else:
                    joined.append([left, right])
            left, right = max(
                joined, key=lambda interval: interval[1] - interval[0])
            span_ratio = (
                float(right - left) / max(float(width), 1.0))
            if span_ratio < cfg.EXIT_LINE_MIN_LENGTH_RATIO:
                continue

            slope = group["slope"]
            intercept = group["intercept"]
            sample_count = max(32, int(round((right - left) / 5.0)))
            sample_x = np.linspace(left, right, sample_count)
            sample_y = slope * sample_x + intercept
            normal_size = math.sqrt(1.0 + slope * slope)
            normal_x = -slope / normal_size
            normal_y = 1.0 / normal_size

            positive_side = []
            negative_side = []
            center_side = []
            for point_x, point_y in zip(sample_x, sample_y):
                positive_values = []
                negative_values = []
                center_values = []
                for offset in (-1.0, 0.0, 1.0):
                    x = int(round(point_x + normal_x * offset))
                    y = int(round(point_y + normal_y * offset))
                    if 0 <= x < width and 0 <= y < height:
                        center_values.append(value[y, x])
                for offset in np.linspace(2.0, float(margin), 3):
                    for values, direction in (
                        (positive_values, 1.0),
                        (negative_values, -1.0),
                    ):
                        x = int(round(
                            point_x + direction * normal_x * offset))
                        y = int(round(
                            point_y + direction * normal_y * offset))
                        if 0 <= x < width and 0 <= y < height:
                            values.append(value[y, x])
                if positive_values and negative_values and center_values:
                    positive_side.append(float(np.median(positive_values)))
                    negative_side.append(float(np.median(negative_values)))
                    # A linha do Hough costuma cair em uma das duas bordas.
                    # O menor valor entre centro e um pixel de cada lado
                    # alcança a fita mesmo quando ela tem só 2 ou 3 pixels.
                    center_side.append(float(np.min(center_values)))

            if not positive_side:
                continue
            positive_side = np.asarray(positive_side, dtype=np.float32)
            negative_side = np.asarray(negative_side, dtype=np.float32)
            center_side = np.asarray(center_side, dtype=np.float32)
            if np.median(positive_side) <= np.median(negative_side):
                dark_side, light_side = positive_side, negative_side
            else:
                dark_side, light_side = negative_side, positive_side

            dark_value = float(np.median(dark_side))
            dark_support = float(np.mean(
                dark_side <= cfg.EXIT_LINE_MAX_DARK_SIDE_VALUE))
            differences = light_side - dark_side
            contrast = float(np.median(differences))
            contrast_support = float(np.mean(
                differences >= cfg.EXIT_LINE_MIN_SIDE_CONTRAST))
            thick_ok = not (
                dark_value > cfg.EXIT_LINE_MAX_DARK_SIDE_VALUE
                or dark_support < cfg.EXIT_LINE_MIN_DARK_SUPPORT
                or contrast < cfg.EXIT_LINE_MIN_SIDE_CONTRAST
                or contrast_support < cfg.EXIT_LINE_MIN_CONTRAST_SUPPORT
            )

            # Para a soleira muito distante, a fita pode ter só alguns
            # pixels: nesse caso ambos os lados são claros e apenas o centro
            # da linha é escuro. Este segundo teste cobre exatamente isso.
            thin_light_side = np.maximum(positive_side, negative_side)
            thin_dark_value = float(np.median(center_side))
            thin_dark_support = float(np.mean(
                center_side <= cfg.EXIT_LINE_MAX_DARK_SIDE_VALUE))
            thin_differences = thin_light_side - center_side
            thin_contrast = float(np.median(thin_differences))
            thin_contrast_support = float(np.mean(
                thin_differences >= cfg.EXIT_LINE_MIN_SIDE_CONTRAST))
            thin_ok = (
                thin_dark_value <= cfg.EXIT_LINE_MAX_DARK_SIDE_VALUE
                and thin_dark_support >= cfg.EXIT_LINE_MIN_DARK_SUPPORT
                and thin_contrast >= cfg.EXIT_LINE_MIN_SIDE_CONTRAST
                and thin_contrast_support
                >= cfg.EXIT_LINE_MIN_CONTRAST_SUPPORT
            )
            if not thick_ok and not thin_ok:
                continue
            if thin_ok and (
                not thick_ok or thin_contrast_support > contrast_support
            ):
                dark_value = thin_dark_value
                dark_support = thin_dark_support
                contrast = thin_contrast
                contrast_support = thin_contrast_support

            middle_y = slope * center_x + intercept
            y_ratio = middle_y / max(float(height), 1.0)
            score = (
                span_ratio
                + 0.55 * dark_support
                + 0.35 * contrast_support
                + 0.35 * y_ratio
                - 0.20 * abs(group["angle"])
                / max(cfg.EXIT_LINE_MAX_ANGLE_DEG, 1.0)
            )
            candidates.append({
                "score": score,
                "left": int(left),
                "right": int(right),
                "slope": slope,
                "intercept": intercept,
                "angle": group["angle"],
                "span_ratio": span_ratio,
                "dark_value": dark_value,
                "contrast": contrast,
                "dark_support": dark_support,
                "contrast_support": contrast_support,
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item["score"])
        left = max(best["left"], 0)
        right = min(best["right"] + 1, width)
        y_left = best["slope"] * float(left) + best["intercept"]
        y_right = best["slope"] * float(right - 1) + best["intercept"]
        center_y = (
            best["slope"] * ((left + right - 1) / 2.0)
            + best["intercept"])
        thickness = margin * 2 + 1
        confidence = _clip01(
            0.45
            + 0.30 * (
                best["span_ratio"] - cfg.EXIT_LINE_MIN_LENGTH_RATIO)
            + 0.15 * best["dark_support"]
            + 0.10 * best["contrast_support"]
        )
        top_y = max(int(round(min(y_left, y_right))) - margin, 0)
        bottom_y = min(
            int(round(max(y_left, y_right))) + margin, height - 1)
        box_width = max(right - left, 1)

        return ExitStripeDetection(
            center_x=(left + right - 1) / 2.0,
            center_y=center_y,
            width=box_width,
            height=bottom_y - top_y + 1,
            top_y=top_y,
            bottom_y=bottom_y,
            span_ratio=best["span_ratio"],
            thickness_ratio=thickness / max(float(height), 1.0),
            aspect=box_width / max(float(thickness), 1.0),
            value=best["dark_value"],
            surround_contrast=best["contrast"],
            confidence=confidence,
            timestamp=float(timestamp),
            bbox=(left, top_y, box_width, bottom_y - top_y + 1),
            angle_deg=best["angle"],
            dark_support=best["dark_support"],
            contrast_support=best["contrast_support"],
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

    def __init__(
        self,
        detector=None,
        confirmer=None,
        max_age_s=None,
        locked_detector=None,
    ):
        self.detector = BlackExitDetector() if detector is None else detector
        if locked_detector is not None:
            self.locked_detector = locked_detector
        elif detector is None:
            self.locked_detector = BlackExitDetector(
                hsv_max=cfg.EXIT_BLACK_LOCK_HSV_MAX,
                geometry=BandGeometry(
                    min_row_fill=cfg.EXIT_BLACK_LOCK_MIN_ROW_FILL,
                    min_span_ratio=cfg.EXIT_BLACK_LOCK_MIN_SPAN_RATIO,
                    max_span_ratio=cfg.EXIT_BLACK_MAX_SPAN_RATIO,
                    min_thickness_ratio=(
                        cfg.EXIT_BLACK_LOCK_MIN_THICKNESS_RATIO),
                    max_thickness_ratio=(
                        cfg.EXIT_BLACK_LOCK_MAX_THICKNESS_RATIO),
                    min_fill_ratio=cfg.EXIT_BLACK_LOCK_MIN_FILL_RATIO,
                    min_aspect=cfg.EXIT_BLACK_LOCK_MIN_ASPECT,
                ),
                max_inside_value=cfg.EXIT_BLACK_LOCK_MAX_INSIDE_VALUE,
                min_surround_contrast=(
                    cfg.EXIT_BLACK_LOCK_MIN_SURROUND_CONTRAST),
                min_confidence=cfg.EXIT_BLACK_LOCK_MIN_CONFIDENCE,
            )
        else:
            # Detectores falsos/externos mantêm a assinatura antiga.
            self.locked_detector = self.detector
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
        self._vote_reference = None
        self._locked_reference = None
        self._just_locked = False

    @property
    def confirmed(self):
        return self.confirmer.confirmed

    @property
    def track_locked(self):
        """A mesma faixa preta continua sendo o único alvo aceito?"""
        return self._locked_reference is not None

    @property
    def votes(self):
        return self.confirmer.votes

    @property
    def just_locked(self):
        """Verdadeiro somente no frame que confirmou a saída."""
        return self._just_locked

    def reset(self, now=None):
        self.confirmer.reset(now=now)
        self.last_detection = None
        self._vote_reference = None
        self._locked_reference = None
        self._just_locked = False

    def preview(self, frame_bgr, timestamp=None):
        """Detecta durante o giro sem permitir que o frame some um voto.

        A prévia serve somente para mandar frear assim que a faixa entra na
        imagem. A confirmação continua acontecendo depois que o chassi para,
        evitando aprovar borrão de movimento ou reflexo passageiro.
        """
        preview = getattr(self.detector, "preview", None)
        detection = (
            preview(frame_bgr, timestamp=timestamp)
            if callable(preview)
            else self.detector.detect(frame_bgr, timestamp=timestamp)
        )
        self.last_detection = detection
        return detection

    def update(self, frame_bgr, timestamp=None, now=None):
        self._just_locked = False
        detection = self.detector.detect(frame_bgr, timestamp=timestamp)
        if (
            detection is None
            and self.track_locked
            and self.locked_detector is not self.detector
        ):
            detection = self.locked_detector.detect(
                self._locked_search_frame(frame_bgr),
                timestamp=timestamp,
            )
        current_reference = None
        if detection is not None:
            height, width = frame_bgr.shape[:2]
            current_reference = (
                float(detection.center_x) / max(float(width), 1.0),
                float(detection.center_y) / max(float(height), 1.0),
                float(detection.span_ratio),
            )
            if self.track_locked:
                if not self._same_locked_candidate(
                    self._locked_reference, current_reference
                ):
                    # Não troque a soleira já confirmada por uma sombra,
                    # parede ou objeto preto que apareceu durante o pulso.
                    self.last_detection = None
                    return True, None
                self._locked_reference = current_reference
                self._vote_reference = current_reference
                self.last_detection = detection
                return True, detection
            if (
                self._vote_reference is not None
                and not self._same_candidate(
                    self._vote_reference, current_reference)
            ):
                # O chassi esta parado durante os votos. Se o candidato pula
                # para outra parte da imagem, e outra borda/mancha e a
                # contagem anterior nao pode ser aproveitada.
                self.confirmer.reset(now=now)
            self._vote_reference = current_reference
        elif self.track_locked:
            # Um frame borrado não desfaz o lock. O controlador permanece
            # parado e aguarda a faixa reaparecer dentro do timeout próprio.
            self.last_detection = None
            return True, None

        self.last_detection = detection
        fast_lock_confidence = getattr(
            self.detector, "fast_lock_confidence", None)
        confirmed = bool(
            detection is not None
            and fast_lock_confidence is not None
            and float(detection.confidence) >= float(fast_lock_confidence)
        )
        if not confirmed:
            confirmed = self.confirmer.update(
                detection is not None,
                timestamp=0.0 if timestamp is None else timestamp,
                now=now,
            )
        if confirmed and detection is not None:
            self._locked_reference = current_reference
            self._just_locked = True
        if detection is None and self.confirmer.votes == 0:
            self._vote_reference = None
        return confirmed, detection

    @staticmethod
    def _same_candidate(reference, current):
        return (
            abs(reference[0] - current[0])
            <= cfg.EXIT_BLACK_MAX_VOTE_CENTER_X_DRIFT_RATIO
            and abs(reference[1] - current[1])
            <= cfg.EXIT_BLACK_MAX_VOTE_CENTER_Y_DRIFT_RATIO
            and abs(reference[2] - current[2])
            <= cfg.EXIT_BLACK_MAX_VOTE_SPAN_DRIFT_RATIO
        )

    @staticmethod
    def _same_locked_candidate(reference, current):
        return (
            abs(reference[0] - current[0])
            <= cfg.EXIT_BLACK_LOCK_MAX_CENTER_X_DRIFT_RATIO
            and abs(reference[1] - current[1])
            <= cfg.EXIT_BLACK_LOCK_MAX_CENTER_Y_DRIFT_RATIO
            and abs(reference[2] - current[2])
            <= cfg.EXIT_BLACK_LOCK_MAX_SPAN_DRIFT_RATIO
        )

    def _locked_search_frame(self, frame_bgr):
        """Apaga tudo fora da vizinhança da soleira já travada."""
        height, width = frame_bgr.shape[:2]
        center_x = self._locked_reference[0] * width
        center_y = self._locked_reference[1] * height
        margin_x = max(
            cfg.EXIT_BLACK_LOCK_SEARCH_X_MARGIN_RATIO * width,
            self._locked_reference[2] * width,
        )
        margin_y = cfg.EXIT_BLACK_LOCK_SEARCH_Y_MARGIN_RATIO * height
        left = max(int(round(center_x - margin_x)), 0)
        right = min(int(round(center_x + margin_x)), width)
        top = max(int(round(center_y - margin_y)), 0)
        bottom = min(int(round(center_y + margin_y)), height)
        limitado = np.full_like(frame_bgr, 255)
        limitado[top:bottom, left:right] = frame_bgr[top:bottom, left:right]
        return limitado
