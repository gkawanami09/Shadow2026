"""Detecta e acompanha marcadores triangulares na camera de resgate.

O modulo nao importa o pipeline do segue-linha e nao acessa estado
compartilhado. ``color_masks`` e propositalmente uma funcao pura: outros
detectores do resgate podem reutilizar as mesmas mascaras para excluir os
marcadores sem acoplar o rastreador abaixo.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

import config_resgate as cfg


@dataclass(frozen=True)
class MarkerDetection:
    """Marcador confirmado ou ainda em aquisicao temporal."""

    kind: str
    center_x: float
    center_y: float
    width: float
    height: float
    bottom_y: float
    area: float
    confidence: float
    confirmed: bool
    hits: int
    timestamp: float
    track_locked: bool = False
    bbox: object = None

    def __post_init__(self):
        if self.bbox is None:
            object.__setattr__(
                self,
                "bbox",
                (
                    float(self.center_x) - float(self.width) / 2.0,
                    float(self.center_y) - float(self.height) / 2.0,
                    float(self.width),
                    float(self.height),
                ),
            )

    @property
    def center(self):
        return self.center_x, self.center_y

    def horizontal_error(self, frame_width):
        half_width = max(float(frame_width) / 2.0, 1.0)
        return float(np.clip(
            (self.center_x - half_width) / half_width,
            -1.0,
            1.0,
        ))


@dataclass(frozen=True)
class _MarkerCandidate:
    kind: str
    center_x: float
    center_y: float
    bbox: tuple
    bottom_y: float
    area: float
    confidence: float

    @property
    def size(self):
        _x, _y, width, height = self.bbox
        return math.hypot(width, height)


def _as_hsv_bound(values):
    return np.asarray(values, dtype=np.uint8)


def color_masks(frame_bgr):
    """Retorna mascaras HSV cruas ``{"green", "red"}`` para um frame BGR.

    A funcao nao aplica ROI nem morfologia. Assim o detector da bola pode
    reutiliza-la para exclusao de cor sem herdar decisões geometricas dos
    marcadores.
    """
    if (
        frame_bgr is None
        or not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError("color_masks exige um frame BGR de tres canais")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        _as_hsv_bound(cfg.MARKER_GREEN_HSV_MIN),
        _as_hsv_bound(cfg.MARKER_GREEN_HSV_MAX),
    )
    red_low = cv2.inRange(
        hsv,
        _as_hsv_bound(cfg.MARKER_RED_HSV_MIN_1),
        _as_hsv_bound(cfg.MARKER_RED_HSV_MAX_1),
    )
    red_high = cv2.inRange(
        hsv,
        _as_hsv_bound(cfg.MARKER_RED_HSV_MIN_2),
        _as_hsv_bound(cfg.MARKER_RED_HSV_MAX_2),
    )
    return {
        "green": green,
        "red": cv2.bitwise_or(red_low, red_high),
    }


def _pixel_scale(width, height):
    return max(min(
        float(width) / cfg.MARKER_BASE_WIDTH,
        float(height) / cfg.MARKER_BASE_HEIGHT,
    ), 0.25)


def _odd_kernel_size(base_size, scale):
    size = max(int(round(float(base_size) * scale)), 1)
    if size % 2 == 0:
        size += 1
    return size


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


class MarkerDetector:
    """Seleciona um unico triangulo verde ou vermelho e preserva seu lock."""

    VALID_TARGETS = ("green", "red")

    def __init__(self, target_kind):
        if target_kind not in self.VALID_TARGETS:
            raise ValueError(
                "marcador de resgate deve ser 'green' ou 'red'")
        self.target_kind = target_kind
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._track_locked = False
        self._last_hit_timestamp = None
        self._pixel_scale = 1.0
        self.last_candidates = ()
        self.last_mask = None
        self.last_rejections = {}

    def reset(self):
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._track_locked = False
        self._last_hit_timestamp = None
        self._pixel_scale = 1.0
        self.last_candidates = ()
        self.last_mask = None
        self.last_rejections = {}

    def detect(self, frame_bgr, timestamp=None, masks=None):
        timestamp = (
            time.monotonic() if timestamp is None else float(timestamp))
        height, width = frame_bgr.shape[:2]
        scale = _pixel_scale(width, height)
        self._pixel_scale = scale

        if masks is None:
            masks = color_masks(frame_bgr)
        if self.target_kind not in masks:
            raise ValueError(
                f"mascara precomputada ausente: {self.target_kind}")
        mask = np.asarray(masks[self.target_kind], dtype=np.uint8)
        if mask.shape != (height, width):
            raise ValueError(
                "mascara precomputada possui resolucao diferente do frame")
        mask = mask.copy()
        chroma = self._chroma_map(frame_bgr)
        # Um banho ciano pode cair na banda HSV verde em toda a parede. Exigir
        # dominancia cromatica ja na proposta separa o triangulo saturado do
        # iluminante; o contraste com o anel ainda valida o candidato depois.
        # Limiar POR PIXEL: baixo de proposito. Subi-lo corroeria a borda do
        # blob e mudaria a forma do candidato. O rigor cromatico fica no gate
        # por blob (MARKER_MIN_INSIDE_CHROMA), aplicado depois da geometria.
        mask[chroma < cfg.MARKER_MASK_MIN_CHROMA] = 0
        roi_top = int(round(height * cfg.MARKER_ROI_TOP))
        mask[:roi_top, :] = 0

        open_size = _odd_kernel_size(
            cfg.MARKER_MORPH_OPEN_PX, scale)
        close_size = _odd_kernel_size(
            cfg.MARKER_MORPH_CLOSE_PX, scale)
        if open_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (open_size, open_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if close_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.last_mask = mask

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rejections = {}
        candidates = []
        for contour in contours:
            candidate, reason = self._candidate_from_contour(
                contour,
                mask,
                chroma,
                frame_bgr.shape,
                scale,
            )
            if candidate is None:
                rejections[reason] = rejections.get(reason, 0) + 1
            else:
                candidates.append(candidate)

        self.last_rejections = rejections
        self.last_candidates = tuple(sorted(
            candidates,
            key=lambda item: (item.confidence, item.area),
            reverse=True,
        ))
        selected = self._select_candidate(candidates, scale)
        return self._update_track(selected, timestamp)

    def _candidate_from_contour(
        self,
        contour,
        mask,
        chroma,
        frame_shape,
        scale,
    ):
        height, width = frame_shape[:2]
        frame_area = float(max(width * height, 1))
        area = float(cv2.contourArea(contour))
        area_ratio = area / frame_area
        if area_ratio < cfg.MARKER_MIN_AREA_RATIO:
            return None, "area_small"
        if area_ratio > cfg.MARKER_MAX_AREA_RATIO:
            return None, "area_large"

        x, y, box_width, box_height = cv2.boundingRect(contour)
        min_side = cfg.MARKER_MIN_SIDE_PX * scale
        if min(box_width, box_height) < min_side:
            return None, "side"

        # Blob encostado na borda lateral: parte dele ficou fora do quadro e
        # sua forma nao descreve o objeto inteiro. Ver config_resgate.
        margin = cfg.MARKER_EDGE_MARGIN_PX
        touches_edge = (
            x <= margin or x + box_width >= width - margin)
        # "Chegada": grande e encostando — o robo esta em cima do marcador.
        # Julgar forma aqui perderia o alvo justamente na hora de depositar.
        arrival = (
            touches_edge and area_ratio >= cfg.MARKER_NEAR_AREA_RATIO)
        if touches_edge and not arrival:
            # Fragmento: o giro de procura continua ate o marcador aparecer
            # inteiro. Rejeitar aqui e o comportamento correto.
            return None, "incompleto"

        aspect = max(box_width, box_height) / max(
            min(box_width, box_height), 1)
        if not arrival and aspect > cfg.MARKER_MAX_ASPECT_RATIO:
            return None, "aspect"

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0.0:
            return None, "hull"
        solidity = area / hull_area
        if solidity < cfg.MARKER_MIN_SOLIDITY:
            return None, "solidity"

        # SANIDADE DE FORMA, nao classificacao de forma. A camera olha quase
        # rente ao piso e o escorco faz o triangulo real medir 0.577 de
        # triangularidade — entre um quadrado (0.500) e um circulo (0.605),
        # e ABAIXO da cadeira vermelha do laboratorio (0.677). Julgar forma
        # aqui rejeitava o marcador verdadeiro; quem discrimina e a
        # cromaticidade, mais abaixo.
        #
        # Compacidade = area / area do retangulo envolvente. Serve so para
        # excluir filamento, borda de parede e mancha difusa.
        compactness = area / float(max(box_width * box_height, 1))
        if not arrival and compactness < cfg.MARKER_MIN_COMPACTNESS:
            return None, "compactness"

        # Mantidos apenas como telemetria: aparecem no overlay e no dataset,
        # mas nao reprovam mais nada.
        perimeter = float(cv2.arcLength(hull, True))
        approximation = cv2.approxPolyDP(
            hull,
            cfg.MARKER_APPROX_EPSILON_RATIO * perimeter,
            True,
        )
        vertex_count = len(approximation)
        triangle_area, _triangle = cv2.minEnclosingTriangle(hull)
        triangularity = (
            hull_area / float(triangle_area)
            if triangle_area and triangle_area > 0.0 else 0.0)

        contour_fill = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_fill, [contour], -1, 255, -1)
        contour_pixels = int(np.count_nonzero(contour_fill))
        if contour_pixels <= 0:
            return None, "empty"
        mask_fill = (
            np.count_nonzero(
                cv2.bitwise_and(mask, contour_fill))
            / contour_pixels
        )
        if mask_fill < cfg.MARKER_MIN_MASK_FILL:
            return None, "mask_fill"

        ring_size = _odd_kernel_size(
            cfg.MARKER_RING_WIDTH_PX * 2 + 1, scale)
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ring_size, ring_size))
        expanded = cv2.dilate(contour_fill, ring_kernel)
        ring = cv2.subtract(expanded, contour_fill)
        ring[:int(round(height * cfg.MARKER_ROI_TOP)), :] = 0

        inside_selector = cv2.bitwise_and(mask, contour_fill) > 0
        ring_selector = ring > 0
        if (
            np.count_nonzero(inside_selector) == 0
            or np.count_nonzero(ring_selector)
            < cfg.MARKER_RING_MIN_PIXELS
        ):
            return None, "ring"
        inside_chroma = float(np.median(chroma[inside_selector]))
        outside_chroma = float(np.median(chroma[ring_selector]))
        contrast = inside_chroma - outside_chroma
        if inside_chroma < cfg.MARKER_MIN_INSIDE_CHROMA:
            return None, "chroma"
        if contrast < cfg.MARKER_MIN_CHROMA_CONTRAST:
            return None, "contrast"

        # Na rota de chegada a forma nao foi julgada, entao ela nao pode nem
        # somar nem subtrair: recebe um valor neutro. Sem isso o marcador
        # correto seria aceito na geometria e reprovado logo depois em
        # "confidence" — o alvo se perderia na hora de depositar.
        # A forma NAO entra mais na confianca. Ela media triangularidade, e
        # medi que triangularidade nao separa marcador de circulo nem de
        # cadeira nesta camera. Manter esse termo faria a confianca variar
        # com um sinal que e ruido. O peso foi para cromaticidade e
        # contraste, que sao os termos com separacao medida.
        compactness_score = _clip01(
            (compactness - cfg.MARKER_MIN_COMPACTNESS)
            / max(1.0 - cfg.MARKER_MIN_COMPACTNESS, 1e-6)
        )
        contrast_score = _clip01(
            (contrast - cfg.MARKER_MIN_CHROMA_CONTRAST) / 100.0)
        # O zero da pontuacao e o limiar da MASCARA, nao o do gate. Se fosse o
        # do gate, endurecer o gate baixaria a confianca do proprio marcador
        # verdadeiro e ele cairia em "confidence" — foi exatamente o que
        # aconteceu ao medir com as capturas reais da arena.
        chroma_score = _clip01(
            (inside_chroma - cfg.MARKER_MASK_MIN_CHROMA) / 140.0)
        solidity_score = _clip01(
            (solidity - cfg.MARKER_MIN_SOLIDITY)
            / max(1.0 - cfg.MARKER_MIN_SOLIDITY, 1e-6)
        )
        fill_score = _clip01(
            (mask_fill - cfg.MARKER_MIN_MASK_FILL)
            / max(1.0 - cfg.MARKER_MIN_MASK_FILL, 1e-6)
        )
        area_score = _clip01(
            area_ratio / max(cfg.MARKER_MIN_AREA_RATIO * 5.0, 1e-6))
        # Cromaticidade e contraste somam 62% do peso: sao os unicos termos
        # com separacao medida (marcador 124-148 contra cadeira 63-79).
        confidence = (
            0.34 * contrast_score
            + 0.28 * chroma_score
            + 0.14 * solidity_score
            + 0.10 * fill_score
            + 0.09 * compactness_score
            + 0.05 * area_score
        )
        if confidence < cfg.MARKER_MIN_CONFIDENCE:
            return None, "confidence"

        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
        bbox = (int(x), int(y), int(box_width), int(box_height))
        return _MarkerCandidate(
            self.target_kind,
            float(center_x),
            float(center_y),
            bbox,
            float(y + box_height - 1),
            area,
            float(confidence),
        ), None

    def _chroma_map(self, frame_bgr):
        channels = frame_bgr.astype(np.float32)
        blue = channels[:, :, 0]
        green = channels[:, :, 1]
        red = channels[:, :, 2]
        if self.target_kind == "green":
            return green - np.maximum(red, blue)
        return red - np.maximum(green, blue)

    def _select_candidate(self, candidates, scale):
        if not candidates:
            return None
        if self._tracked is None:
            return max(
                candidates,
                key=lambda item: (
                    item.confidence
                    + min(
                        item.area
                        / (
                            cfg.MARKER_BASE_WIDTH
                            * cfg.MARKER_BASE_HEIGHT
                            * scale * scale
                        ),
                        0.20,
                    ),
                    item.area,
                ),
            )

        matches = []
        for candidate in candidates:
            compatible, normalized_distance = self._track_match(
                candidate, scale)
            if compatible:
                matches.append((candidate, normalized_distance))
        if matches:
            return min(
                matches,
                key=lambda item: (
                    item[1],
                    -item[0].confidence,
                ),
            )[0]
        if self._track_locked:
            return None
        return max(
            candidates,
            key=lambda item: (item.confidence, item.area),
        )

    def _track_match(self, candidate, scale):
        if self._tracked is None:
            return False, float("inf")
        distance = math.hypot(
            candidate.center_x - self._tracked.center_x,
            candidate.center_y - self._tracked.center_y,
        )
        gate = max(
            cfg.MARKER_ASSOCIATION_MIN_PX * scale,
            cfg.MARKER_ASSOCIATION_SIZE_FACTOR
            * max(candidate.size, self._tracked.size),
        )
        area_ratio = candidate.area / max(self._tracked.area, 1.0)
        compatible = (
            distance <= gate
            and cfg.MARKER_AREA_RATIO_MIN
            <= area_ratio
            <= cfg.MARKER_AREA_RATIO_MAX
        )
        return compatible, distance / max(gate, 1.0)

    def _update_track(self, selected, timestamp):
        if selected is None:
            self._misses += 1
            if self._misses > cfg.MARKER_MAX_TRACK_MISSES:
                self._tracked = None
                self._hits = 0
                self._track_locked = False
                self._last_hit_timestamp = None
            return None

        compatible = (
            self._tracked is not None
            and self._track_match(
                selected,
                self._pixel_scale,
            )[0]
        )
        is_new_timestamp = (
            self._last_hit_timestamp is None
            or timestamp > self._last_hit_timestamp + 1e-9
        )

        if compatible:
            if is_new_timestamp:
                self._hits += 1
            self._tracked = selected
        else:
            self._tracked = selected
            self._hits = 1
            self._track_locked = False
        self._misses = 0
        if is_new_timestamp:
            self._last_hit_timestamp = timestamp
        if self._hits >= cfg.MARKER_ACQUIRE_HITS:
            self._track_locked = True

        return MarkerDetection(
            kind=self._tracked.kind,
            center_x=self._tracked.center_x,
            center_y=self._tracked.center_y,
            width=float(self._tracked.bbox[2]),
            height=float(self._tracked.bbox[3]),
            area=self._tracked.area,
            bbox=self._tracked.bbox,
            bottom_y=self._tracked.bottom_y,
            confidence=self._tracked.confidence,
            confirmed=self._hits >= cfg.MARKER_ACQUIRE_HITS,
            hits=self._hits,
            timestamp=timestamp,
            track_locked=self._track_locked,
        )
