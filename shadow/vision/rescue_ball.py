"""Deteccao e rastreamento temporal das esferas da zona de resgate.

O detector usa apenas OpenCV e NumPy, que ja sao dependencias de ``shadow``.
Ele combina propostas de Hough, contornos de borda e mascara escura; depois
valida geometria, borda radial e contraste local. A classificacao de aparencia
sempre usa o frame original, sem gamma.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

import rescue_config as cfg


@dataclass(frozen=True)
class BallDetection:
    kind: str
    center_x: float
    center_y: float
    radius: float
    confidence: float
    confirmed: bool
    hits: int
    timestamp: float

    @property
    def diameter(self):
        return self.radius * 2.0

    @property
    def bottom_y(self):
        return self.center_y + self.radius

    def horizontal_error(self, frame_width):
        half_width = max(float(frame_width) / 2.0, 1.0)
        return float(np.clip(
            (self.center_x - half_width) / half_width, -1.0, 1.0))


@dataclass
class _Proposal:
    center_x: float
    center_y: float
    radius: float
    circularity: float
    fill_ratio: float
    source: str
    edge_support: float = 0.0


@dataclass
class _Candidate:
    kind: str
    center_x: float
    center_y: float
    radius: float
    confidence: float


class RescueEnhancer:
    """CLAHE + gamma somente na luminosidade LAB."""

    def __init__(self, gamma=cfg.RESCUE_GAMMA):
        if gamma <= 0:
            raise ValueError("gamma precisa ser maior que zero")
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.RESCUE_CLAHE_CLIP,
            tileGridSize=cfg.RESCUE_CLAHE_GRID)
        values = np.arange(256, dtype=np.float32) / 255.0
        self.gamma_lut = np.clip(
            np.power(values, 1.0 / gamma) * 255.0,
            0, 255).astype(np.uint8)

    def apply(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = self.clahe.apply(lightness)
        lightness = cv2.LUT(lightness, self.gamma_lut)
        return cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR)


class BallDetector:
    """Detector stateful: uma esfera so e confirmada apos varios frames."""

    def __init__(self, target_kind="any", enhance=True):
        if target_kind not in ("any", "black", "silver"):
            raise ValueError("target_kind deve ser any, black ou silver")
        self.target_kind = target_kind
        self.enhancer = RescueEnhancer() if enhance else None
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._pixel_scale = 1.0
        self.last_candidates = []
        self.last_enhanced = None
        self.last_edges = None

    def reset(self):
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self.last_candidates = []

    def detect(self, frame, timestamp=None):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame BGR invalido")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)

        height, width = frame.shape[:2]
        self._pixel_scale = cfg.ball_pixel_scale(width, height)
        enhanced = (
            self.enhancer.apply(frame) if self.enhancer is not None
            else frame.copy())
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur_size = cfg.BALL_MEDIAN_BLUR
        if blur_size % 2 == 0:
            blur_size += 1
        gray_blur = cv2.medianBlur(gray, blur_size)

        median = float(np.median(gray_blur))
        lower = int(max(20, (1.0 - cfg.BALL_CANNY_SIGMA) * median))
        upper = int(min(255, max(lower + 30, (1.0 + cfg.BALL_CANNY_SIGMA) * median)))
        edges = cv2.Canny(gray_blur, lower, upper)

        roi_top = int(height * cfg.BALL_ROI_TOP)
        roi_bottom = int(height * cfg.BALL_ROI_BOTTOM)
        edges[:roi_top, :] = 0
        edges[roi_bottom:, :] = 0
        closed_edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, dark_mask = cv2.threshold(
            hsv[:, :, 2],
            cfg.BALL_BLACK_V_MAX,
            255,
            cv2.THRESH_BINARY_INV)
        dark_mask[:roi_top, :] = 0
        dark_mask[roi_bottom:, :] = 0
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        proposals = []
        proposals.extend(self._hough_proposals(
            gray_blur, roi_top, roi_bottom, self._pixel_scale))
        proposals.extend(self._contour_proposals(
            closed_edges, "edge", self._pixel_scale))
        proposals.extend(self._contour_proposals(
            dark_mask, "dark", self._pixel_scale))
        proposals = self._deduplicate(proposals)

        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        candidates = []
        for proposal in proposals:
            if not self._inside_roi(proposal, width, height, roi_top, roi_bottom):
                continue
            proposal.edge_support = self._radial_edge_support(
                edge_dilated, proposal)
            if proposal.edge_support < cfg.BALL_MIN_EDGE_SUPPORT:
                continue
            candidate = self._classify(frame, hsv, proposal)
            if candidate is None:
                continue
            if self.target_kind != "any" and candidate.kind != self.target_kind:
                continue
            candidates.append(candidate)

        self.last_candidates = candidates
        self.last_enhanced = enhanced
        self.last_edges = edges
        selected = self._select_candidate(candidates)
        return self._update_track(selected, timestamp)

    def _hough_proposals(self, gray, roi_top, roi_bottom, pixel_scale):
        roi = gray[roi_top:roi_bottom, :]
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=cfg.BALL_HOUGH_DP,
            minDist=max(
                2, int(round(cfg.BALL_HOUGH_MIN_DIST_PX * pixel_scale))),
            param1=cfg.BALL_HOUGH_PARAM1,
            param2=cfg.BALL_HOUGH_PARAM2,
            minRadius=max(
                2, int(round(cfg.BALL_MIN_RADIUS_PX * pixel_scale))),
            maxRadius=max(
                3, int(round(cfg.BALL_MAX_RADIUS_PX * pixel_scale))),
        )
        if circles is None:
            return []
        return [
            _Proposal(
                float(circle[0]),
                float(circle[1] + roi_top),
                float(circle[2]),
                cfg.BALL_MIN_CIRCULARITY,
                cfg.BALL_MIN_FILL_RATIO,
                "hough")
            for circle in circles[0]
        ]

    def _contour_proposals(self, mask, source, pixel_scale):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area <= 0 or perimeter <= 0:
                continue

            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            min_radius = cfg.BALL_MIN_RADIUS_PX * pixel_scale
            max_radius = cfg.BALL_MAX_RADIUS_PX * pixel_scale
            if not min_radius <= radius <= max_radius:
                continue

            _, _, box_width, box_height = cv2.boundingRect(contour)
            short_side = max(min(box_width, box_height), 1)
            aspect = max(box_width, box_height) / short_side
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            fill_ratio = area / max(math.pi * radius * radius, 1.0)

            if aspect > cfg.BALL_MAX_ASPECT_RATIO:
                continue
            if circularity < cfg.BALL_MIN_CIRCULARITY:
                continue
            if fill_ratio < cfg.BALL_MIN_FILL_RATIO:
                continue

            proposals.append(_Proposal(
                float(center_x), float(center_y), float(radius),
                float(min(circularity, 1.0)),
                float(min(fill_ratio, 1.0)),
                source))
        return proposals

    @staticmethod
    def _inside_roi(proposal, width, height, roi_top, roi_bottom):
        margin = 2
        return (
            proposal.center_x - proposal.radius >= margin
            and proposal.center_x + proposal.radius < width - margin
            and proposal.center_y - proposal.radius >= roi_top
            and proposal.center_y + proposal.radius < min(roi_bottom, height) - margin
        )

    @staticmethod
    def _radial_edge_support(edges, proposal):
        angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        supported = np.zeros(angles.shape, dtype=bool)
        for radius_factor in (0.88, 1.0, 1.12):
            radius = proposal.radius * radius_factor
            xs = np.rint(
                proposal.center_x + radius * np.cos(angles)).astype(int)
            ys = np.rint(
                proposal.center_y + radius * np.sin(angles)).astype(int)
            valid = (
                (xs >= 0) & (xs < edges.shape[1])
                & (ys >= 0) & (ys < edges.shape[0]))
            supported[valid] |= edges[ys[valid], xs[valid]] > 0
        return float(np.mean(supported))

    @staticmethod
    def _circle_samples(hsv, proposal):
        height, width = hsv.shape[:2]
        outer = int(math.ceil(proposal.radius * 1.45))
        x0 = max(int(proposal.center_x) - outer, 0)
        x1 = min(int(proposal.center_x) + outer + 1, width)
        y0 = max(int(proposal.center_y) - outer, 0)
        y1 = min(int(proposal.center_y) + outer + 1, height)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance_sq = (
            (xx - proposal.center_x) ** 2
            + (yy - proposal.center_y) ** 2)
        inner = distance_sq <= (proposal.radius * 0.78) ** 2
        annulus = (
            (distance_sq >= (proposal.radius * 1.08) ** 2)
            & (distance_sq <= (proposal.radius * 1.42) ** 2))
        crop = hsv[y0:y1, x0:x1]
        return crop[:, :, 1][inner], crop[:, :, 2][inner], crop[:, :, 2][annulus]

    def _classify(self, frame, hsv, proposal):
        del frame  # reservado para futuros descritores sem alterar a API
        inner_s, inner_v, annulus_v = self._circle_samples(hsv, proposal)
        if inner_v.size < 20 or annulus_v.size < 20:
            return None

        inner_mean = float(np.mean(inner_v))
        annulus_mean = float(np.mean(annulus_v))
        dark_fraction = float(np.mean(inner_v <= cfg.BALL_BLACK_V_MAX))
        local_dark_contrast = annulus_mean - inner_mean
        low_sat_fraction = float(np.mean(inner_s <= cfg.BALL_SILVER_S_MAX))
        dynamic_range = float(
            np.percentile(inner_v, 90) - np.percentile(inner_v, 10))
        highlight_fraction = float(
            np.mean(inner_v >= cfg.BALL_SILVER_HIGHLIGHT_V))

        geometry = float(np.clip(
            0.35 * proposal.circularity
            + 0.20 * proposal.fill_ratio
            + 0.45 * min(proposal.edge_support / 0.65, 1.0),
            0.0, 1.0))

        black_valid = (
            dark_fraction >= cfg.BALL_BLACK_DARK_FRACTION_MIN
            and (
                local_dark_contrast >= cfg.BALL_BLACK_LOCAL_CONTRAST_MIN
                or inner_mean <= cfg.BALL_BLACK_V_MAX * 0.62))
        black_score = float(np.clip(
            0.42 * dark_fraction
            + 0.25 * np.clip(local_dark_contrast / 55.0, 0.0, 1.0)
            + 0.33 * geometry,
            0.0, 1.0))

        silver_valid = (
            inner_mean > cfg.BALL_BLACK_V_MAX * 0.62
            and low_sat_fraction >= cfg.BALL_SILVER_LOW_SAT_FRACTION_MIN
            and dynamic_range >= cfg.BALL_SILVER_DYNAMIC_RANGE_MIN
            and (
                highlight_fraction >= cfg.BALL_SILVER_HIGHLIGHT_FRACTION_MIN
                or abs(annulus_mean - inner_mean)
                >= cfg.BALL_BLACK_LOCAL_CONTRAST_MIN))
        silver_score = float(np.clip(
            0.25 * low_sat_fraction
            + 0.25 * np.clip(dynamic_range / 100.0, 0.0, 1.0)
            + 0.15 * np.clip(highlight_fraction / 0.20, 0.0, 1.0)
            + 0.35 * geometry,
            0.0, 1.0))

        if black_valid and black_score >= silver_score:
            kind, confidence = "black", black_score
        elif silver_valid:
            kind, confidence = "silver", silver_score
        else:
            return None

        required_confidence = (
            cfg.BALL_HOUGH_MIN_CONFIDENCE
            if proposal.source == "hough"
            else cfg.BALL_MIN_CONFIDENCE)
        if confidence < required_confidence:
            return None
        return _Candidate(
            kind,
            proposal.center_x,
            proposal.center_y,
            proposal.radius,
            confidence)

    @staticmethod
    def _deduplicate(proposals):
        proposals = sorted(
            proposals,
            key=lambda item: (
                item.source != "hough",
                item.circularity + item.fill_ratio,
                item.radius),
            reverse=True)
        unique = []
        for proposal in proposals:
            duplicate = False
            for kept in unique:
                distance = math.hypot(
                    proposal.center_x - kept.center_x,
                    proposal.center_y - kept.center_y)
                if distance <= 0.55 * max(proposal.radius, kept.radius):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(proposal)
        return unique

    def _select_candidate(self, candidates):
        if not candidates:
            return None
        if self._tracked is None:
            return max(
                candidates,
                key=lambda item: (
                    item.confidence
                    + min(item.radius / (160.0 * self._pixel_scale), 0.35)))

        matches = []
        for candidate in candidates:
            if candidate.kind != self._tracked.kind:
                continue
            distance = math.hypot(
                candidate.center_x - self._tracked.center_x,
                candidate.center_y - self._tracked.center_y)
            gate = max(
                cfg.BALL_ASSOCIATION_MIN_PX * self._pixel_scale,
                cfg.BALL_ASSOCIATION_RADIUS_FACTOR
                * max(candidate.radius, self._tracked.radius))
            radius_ratio = candidate.radius / max(self._tracked.radius, 1.0)
            if (
                distance <= gate
                and cfg.BALL_RADIUS_RATIO_MIN <= radius_ratio
                <= cfg.BALL_RADIUS_RATIO_MAX
            ):
                matches.append((candidate, distance / gate))
        if matches:
            return max(
                matches,
                key=lambda item: item[0].confidence - 0.25 * item[1])[0]
        return max(
            candidates,
            key=lambda item: (
                item.confidence
                + min(item.radius / (160.0 * self._pixel_scale), 0.35)))

    def _update_track(self, selected, timestamp):
        if selected is None:
            self._misses += 1
            # Qualquer perda ja colocou o controle em PARAR. Para voltar a
            # mover, a esfera precisa cumprir novamente todos os hits.
            self._hits = 0
            if self._misses > cfg.BALL_MAX_TRACK_MISSES:
                self.reset()
            return None

        compatible = False
        if self._tracked is not None and selected.kind == self._tracked.kind:
            distance = math.hypot(
                selected.center_x - self._tracked.center_x,
                selected.center_y - self._tracked.center_y)
            gate = max(
                cfg.BALL_ASSOCIATION_MIN_PX * self._pixel_scale,
                cfg.BALL_ASSOCIATION_RADIUS_FACTOR
                * max(selected.radius, self._tracked.radius))
            radius_ratio = selected.radius / max(self._tracked.radius, 1.0)
            compatible = (
                distance <= gate
                and cfg.BALL_RADIUS_RATIO_MIN <= radius_ratio
                <= cfg.BALL_RADIUS_RATIO_MAX)

        if compatible:
            alpha = cfg.BALL_TRACK_EMA_ALPHA
            self._tracked = _Candidate(
                selected.kind,
                (1.0 - alpha) * self._tracked.center_x
                + alpha * selected.center_x,
                (1.0 - alpha) * self._tracked.center_y
                + alpha * selected.center_y,
                (1.0 - alpha) * self._tracked.radius
                + alpha * selected.radius,
                (1.0 - alpha) * self._tracked.confidence
                + alpha * selected.confidence,
            )
            self._hits += 1
        else:
            self._tracked = selected
            self._hits = 1

        self._misses = 0
        return BallDetection(
            self._tracked.kind,
            self._tracked.center_x,
            self._tracked.center_y,
            self._tracked.radius,
            self._tracked.confidence,
            self._hits >= cfg.BALL_ACQUIRE_HITS,
            self._hits,
            timestamp,
        )


def annotate_rescue_frame(
    frame,
    detection,
    state,
    detail="",
    distance_mm=None,
    motors_enabled=False,
):
    """Retorna uma copia anotada para debug; nao participa da decisao."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    roi_top = int(height * cfg.BALL_ROI_TOP)
    roi_bottom = int(height * cfg.BALL_ROI_BOTTOM)
    cv2.line(annotated, (0, roi_top), (width, roi_top), (90, 90, 90), 1)
    cv2.line(annotated, (0, roi_bottom), (width, roi_bottom), (90, 90, 90), 1)
    cv2.line(
        annotated,
        (width // 2, 0),
        (width // 2, height),
        (0, 180, 255),
        1)

    if detection is not None:
        color = (0, 255, 0) if detection.confirmed else (0, 180, 255)
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        radius = int(round(detection.radius))
        cv2.circle(annotated, center, radius, color, 2)
        cv2.circle(annotated, center, 3, color, -1)
        label = (
            f"{detection.kind} {detection.confidence:.2f} "
            f"r={detection.radius:.0f} hits={detection.hits}")
        cv2.putText(
            annotated, label,
            (max(center[0] - radius, 3), max(center[1] - radius - 7, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    cv2.putText(
        annotated, f"estado: {state}", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if detail:
        cv2.putText(
            annotated, detail, (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    if distance_mm is not None:
        cv2.putText(
            annotated, f"ultrassom: {distance_mm} mm", (8, 63),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    motor_label = (
        "MOTORES: ATIVOS (--drive)"
        if motors_enabled else
        "MOTORES: DESATIVADOS (adicione --drive)")
    motor_color = (0, 210, 0) if motors_enabled else (0, 0, 255)
    cv2.putText(
        annotated, motor_label, (8, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, motor_color, 2, cv2.LINE_AA)
    return annotated
