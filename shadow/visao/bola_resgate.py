"""Detecção e rastreamento das esferas da área de resgate.

O detector usa apenas OpenCV e NumPy, que ja sao dependencias de ``shadow``.
Ele combina propostas de Hough, contornos de borda e mascara escura; depois
valida geometria, borda radial e contraste local. A classificacao de aparencia
sempre usa o frame original, sem gamma.
"""

from dataclasses import dataclass, replace
import math
import time

import cv2
import numpy as np

import config_resgate as cfg
from visao.arena_resgate import MonocularArenaGuardian
from visao.marcador_resgate import MarkerDetector, color_masks


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
    track_locked: bool = False

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


@dataclass(frozen=True)
class CloseCrescentEvidence:
    """Evidencia normalizada da borda larga da esfera cortada pelo quadro."""

    accepted: bool
    confidence: float
    support: float
    left_support: float
    center_support: float
    right_support: float
    contrast: float
    center_x_ratio: float
    top_y_ratio: float
    halfspan_ratio: float
    bottom_y_ratio: float
    timestamp: float
    gradient_polarity: float = 0.0
    profile_support: float = 0.0
    profile_polarity: float = 0.0
    coherent_run: float = 0.0
    circle_rmse_ratio: float = 1.0
    curvature_score: float = 0.0
    foil_fallback: bool = False
    foil_texture_bins: int = 0
    foil_valid_bins: int = 0
    interior_edge_density: float = 0.0
    background_edge_density: float = 1.0


def _polarity_floor(values, valid, sector_masks):
    """Coerencia por trechos, permitindo um reflexo central de outra cor."""
    supports = []
    masks = (np.ones(valid.shape, dtype=bool),) + tuple(sector_masks)
    for mask in masks:
        indices = np.flatnonzero(valid & mask)
        if indices.size < 2:
            return 0.0
        signs = values[indices] >= 0
        supports.append(float(np.mean(signs[1:] == signs[:-1])))
    return min(supports)


def _support_floor(valid, eligible, sector_masks):
    """Menor suporte relativo somente entre amostras geometricamente validas."""
    eligible_valid = eligible & valid
    if not np.any(eligible):
        return 0.0
    supports = [float(
        np.count_nonzero(eligible_valid)
        / max(np.count_nonzero(eligible), 1)
    )]
    for sector_mask in sector_masks:
        sector_eligible = eligible & sector_mask
        if not np.any(sector_eligible):
            return 0.0
        supports.append(float(
            np.count_nonzero(valid & sector_eligible)
            / np.count_nonzero(sector_eligible)
        ))
    return min(supports)


def _longest_coherent_run(hits, residuals, max_residual_jump):
    """Fração da maior cadeia adjacente que segue a mesma curva."""
    longest = 0
    current = 0
    previous_residual = None
    for hit, residual in zip(hits, residuals):
        if not hit:
            current = 0
            previous_residual = None
            continue
        if (
            previous_residual is None
            or abs(float(residual) - previous_residual)
            <= max_residual_jump
        ):
            current += 1
        else:
            current = 1
        previous_residual = float(residual)
        longest = max(longest, current)
    return float(longest) / max(len(hits), 1)


def _circle_fit_rmse_ratio(xs, ys, frame_height):
    """Erro geometrico do melhor circulo para a borda visivel."""
    if len(xs) < 6:
        return 1.0
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    matrix = np.column_stack((
        2.0 * xs,
        2.0 * ys,
        np.ones(xs.size, dtype=np.float64),
    ))
    squared = np.square(xs) + np.square(ys)
    try:
        center_x, center_y, constant = np.linalg.lstsq(
            matrix, squared, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0
    radius_squared = (
        constant
        + center_x * center_x
        + center_y * center_y
    )
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return 1.0
    radius = math.sqrt(radius_squared)
    radial_error = (
        np.hypot(xs - center_x, ys - center_y) - radius)
    rmse = float(np.sqrt(np.mean(np.square(radial_error))))
    return rmse / max(float(frame_height), 1.0)


def _smooth_curve_samples(values, hits):
    """Suaviza somente a forma larga, preservando V e quinas extensas."""
    values = np.asarray(values, dtype=np.float64)
    hits = np.asarray(hits, dtype=bool)
    indices = np.arange(values.size, dtype=np.float64)
    valid_indices = indices[hits]
    if valid_indices.size < 6:
        return values

    filled = np.interp(
        indices,
        valid_indices,
        values[hits],
    )
    window = max(int(cfg.BALL_CRESCENT_SMOOTH_SAMPLES), 1)
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 else values.size - 1)
    if window < 3:
        return filled

    radius = window // 2
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(window / 4.0, 1.0)
    kernel = np.exp(-0.5 * np.square(positions / sigma))
    kernel /= np.sum(kernel)
    padded = np.pad(filled, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _distributed_curvature_score(
    normalized_x,
    xs,
    ys,
    hits,
):
    """Mede se a inclinacao cresce ao longo de todo o arco, nao so nos cantos."""
    bin_count = max(int(cfg.BALL_CRESCENT_CURVATURE_BINS), 5)
    bin_edges = np.linspace(-1.0, 1.0, bin_count + 1)
    slopes = []
    for index in range(bin_count):
        if index == bin_count - 1:
            in_bin = (
                hits
                & (normalized_x >= bin_edges[index])
                & (normalized_x <= bin_edges[index + 1])
            )
        else:
            in_bin = (
                hits
                & (normalized_x >= bin_edges[index])
                & (normalized_x < bin_edges[index + 1])
            )
        if np.count_nonzero(in_bin) < 4:
            continue
        bin_x = np.asarray(xs[in_bin], dtype=np.float64)
        bin_y = np.asarray(ys[in_bin], dtype=np.float64)
        if float(np.ptp(bin_x)) < 2.0:
            continue
        centered_x = bin_x - float(np.mean(bin_x))
        denominator = float(np.dot(centered_x, centered_x))
        if denominator <= 1e-9:
            continue
        slope = float(np.dot(
            centered_x,
            bin_y - float(np.mean(bin_y)),
        ) / denominator)
        slopes.append((index, slope))

    if len(slopes) < bin_count - 1:
        return 0.0
    increments = []
    for (left_index, left_slope), (
        right_index,
        right_slope,
    ) in zip(slopes, slopes[1:]):
        gap = max(right_index - left_index, 1)
        increments.append(
            right_slope - left_slope
            > cfg.BALL_CRESCENT_MIN_SLOPE_STEP * gap
        )
    if not increments:
        return 0.0
    distributed = float(np.mean(increments))
    slope_span = slopes[-1][1] - slopes[0][1]
    span_score = float(np.clip(
        slope_span / max(cfg.BALL_CRESCENT_MIN_SLOPE_SPAN, 1e-6),
        0.0,
        1.0,
    ))
    return min(distributed, span_score)


def _circular_crescent_geometry(
    width,
    height,
    center_x_ratio,
    top_y_ratio,
    halfspan_ratio,
    bottom_y_ratio,
    normalized_x,
):
    """Arco circular definido pelo ápice e pelos dois ombros inferiores."""
    center_x = float(center_x_ratio) * width
    top_y = float(top_y_ratio) * height
    bottom_y = float(bottom_y_ratio) * height
    halfspan = float(halfspan_ratio) * width
    vertical_delta = max(bottom_y - top_y, 1.0)
    radius = (
        halfspan * halfspan + vertical_delta * vertical_delta
    ) / (2.0 * vertical_delta)
    circle_center_y = top_y + radius
    offset_x = np.asarray(normalized_x, dtype=np.float64) * halfspan
    root = np.sqrt(np.maximum(
        radius * radius - np.square(offset_x),
        1e-6,
    ))
    xs = center_x + offset_x
    ys = circle_center_y - root
    slope = offset_x / root
    return xs, ys, slope


def _crescent_fill_metrics(
    profile_gray,
    x_indices,
    actual_y,
    template_valid,
    hits,
    normalized_x,
    sector_masks,
    height,
    contrast_offset,
    outside_contrast_offset,
    deep_contrast_offset,
):
    """Contraste da borda e preenchimento real abaixo do caminho encontrado."""
    samples = len(x_indices)
    above_y = actual_y - outside_contrast_offset
    below_y = actual_y + contrast_offset
    deep_y = actual_y + deep_contrast_offset
    contrast_valid = (
        template_valid
        & hits
        & (above_y >= 0)
        & (below_y < height)
    )
    if not np.any(contrast_valid):
        return 0.0, 0.0, 0.0

    profile_delta = np.zeros(samples, dtype=np.float32)
    profile_delta[contrast_valid] = (
        profile_gray[
            below_y[contrast_valid],
            x_indices[contrast_valid],
        ].astype(np.float32)
        - profile_gray[
            above_y[contrast_valid],
            x_indices[contrast_valid],
        ].astype(np.float32)
    )
    strong_profile = (
        contrast_valid
        & (
            np.abs(profile_delta)
            >= cfg.BALL_CRESCENT_MIN_CONTRAST
        )
    )
    contrast = float(np.median(np.abs(
        profile_delta[contrast_valid])))
    profile_support = _support_floor(
        strong_profile,
        contrast_valid,
        sector_masks,
    )
    profile_polarity = _polarity_floor(
        profile_delta,
        strong_profile,
        sector_masks,
    )

    # Uma linha curva fina volta ao fundo poucos pixels abaixo. A esfera
    # continua ocupando a faixa interna profunda.
    deep_valid = (
        template_valid
        & hits
        & (above_y >= 0)
        & (deep_y < height)
        & (
            np.abs(normalized_x)
            <= cfg.BALL_CRESCENT_DEEP_INNER_X_RATIO
        )
    )
    deep_delta = np.zeros(samples, dtype=np.float32)
    if np.any(deep_valid):
        deep_delta[deep_valid] = (
            profile_gray[
                deep_y[deep_valid],
                x_indices[deep_valid],
            ].astype(np.float32)
            - profile_gray[
                above_y[deep_valid],
                x_indices[deep_valid],
            ].astype(np.float32)
        )
    strong_deep = (
        deep_valid
        & (
            np.abs(deep_delta)
            >= cfg.BALL_CRESCENT_MIN_CONTRAST
        )
    )
    deep_support = _support_floor(
        strong_deep,
        deep_valid,
        sector_masks,
    )
    deep_polarity = _polarity_floor(
        deep_delta,
        strong_deep,
        sector_masks,
    )
    deep_contrast = (
        float(np.median(np.abs(deep_delta[deep_valid])))
        if np.any(deep_valid)
        else 0.0
    )
    return (
        min(contrast, deep_contrast),
        min(profile_support, deep_support),
        min(profile_polarity, deep_polarity),
    )


def _crescent_texture_metrics(
    gray,
    edge_binary,
    x_indices,
    path_y,
    template_valid,
    normalized_x,
    height,
):
    """Reflexos distribuídos no domo e fundo mais limpo que o interior."""
    bin_count = max(int(cfg.BALL_CRESCENT_FOIL_TEXTURE_BINS), 3)
    limit = float(cfg.BALL_CRESCENT_FOIL_INNER_X_RATIO)
    bin_edges = np.linspace(-limit, limit, bin_count + 1)
    textured_bins = 0
    valid_bins = 0
    interior_edge_densities = []
    background_edge_densities = []

    for index in range(bin_count):
        if index == bin_count - 1:
            in_bin = (
                template_valid
                & (normalized_x >= bin_edges[index])
                & (normalized_x <= bin_edges[index + 1])
            )
        else:
            in_bin = (
                template_valid
                & (normalized_x >= bin_edges[index])
                & (normalized_x < bin_edges[index + 1])
            )
        if np.count_nonzero(in_bin) < 3:
            continue

        inside_values = []
        inside_edges = []
        for offset_ratio in cfg.BALL_CRESCENT_FOIL_INSIDE_OFFSETS:
            sample_y = (
                path_y
                + int(round(float(offset_ratio) * height))
            )
            valid = in_bin & (sample_y >= 0) & (sample_y < height)
            if np.any(valid):
                inside_values.append(gray[
                    sample_y[valid],
                    x_indices[valid],
                ])
                inside_edges.append(edge_binary[
                    sample_y[valid],
                    x_indices[valid],
                ])

        outside_edges = []
        for offset_ratio in cfg.BALL_CRESCENT_FOIL_OUTSIDE_OFFSETS:
            sample_y = (
                path_y
                - int(round(float(offset_ratio) * height))
            )
            valid = in_bin & (sample_y >= 0) & (sample_y < height)
            if np.any(valid):
                outside_edges.append(edge_binary[
                    sample_y[valid],
                    x_indices[valid],
                ])

        if not inside_values or not inside_edges or not outside_edges:
            continue
        interior = np.concatenate(inside_values).astype(np.float32)
        if interior.size < 8:
            continue
        valid_bins += 1
        dynamic_range = float(
            np.percentile(interior, 90)
            - np.percentile(interior, 10)
        )
        highlight_range = float(
            np.percentile(interior, 99)
            - np.median(interior)
        )
        if max(dynamic_range, highlight_range) >= (
            cfg.BALL_CRESCENT_FOIL_MIN_DYNAMIC_RANGE
        ):
            textured_bins += 1
        interior_edge_densities.append(float(np.mean(
            np.concatenate(inside_edges))))
        background_edge_densities.append(float(np.mean(
            np.concatenate(outside_edges))))

    if valid_bins == 0:
        return 0, 0, 0.0, 1.0
    return (
        textured_bins,
        valid_bins,
        float(np.median(interior_edge_densities)),
        float(np.median(background_edge_densities)),
    )


def _detect_close_crescent(gray, edges, timestamp):
    """Procura o arco circular largo da esfera cortada pelo quadro."""
    if (
        gray is None
        or edges is None
        or gray.ndim != 2
        or edges.shape != gray.shape
    ):
        raise ValueError("gray/edges invalidos para detectar meia-lua")

    height, width = gray.shape
    samples = max(int(cfg.BALL_CRESCENT_SAMPLES), 9)
    normalized_x = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    left_mask = normalized_x < (-1.0 / 3.0)
    center_mask = np.abs(normalized_x) <= (1.0 / 3.0)
    right_mask = normalized_x > (1.0 / 3.0)

    # A forma externa e avaliada na imagem suavizada para que os reflexos do
    # papel-aluminio nao troquem a polaridade da silhueta a cada amostra.
    sigma = max(float(height) * 0.008, 1.0)
    profile_gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    gradient_x = cv2.Sobel(
        profile_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(
        profile_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_binary = edges > 0
    connected_edges = cv2.morphologyEx(
        edge_binary.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )
    _, component_labels = cv2.connectedComponents(
        connected_edges, connectivity=8)
    band_px = max(float(height) * cfg.BALL_CRESCENT_BAND_RATIO, 1.5)
    band_steps = max(int(np.ceil(band_px)), 2)
    offsets = np.arange(
        -band_steps, band_steps + 1, dtype=np.int32)
    contrast_offset = max(
        int(round(height * cfg.BALL_CRESCENT_CONTRAST_OFFSET_RATIO)),
        2,
    )
    outside_contrast_offset = max(
        int(round(
            height * cfg.BALL_CRESCENT_OUTSIDE_CONTRAST_OFFSET_RATIO
        )),
        contrast_offset + 2,
    )
    deep_contrast_offset = max(
        int(round(
            height * cfg.BALL_CRESCENT_DEEP_CONTRAST_OFFSET_RATIO
        )),
        contrast_offset + 2,
    )

    best = None
    best_key = None
    foil_candidates_by_shape = {}
    for center_ratio in cfg.BALL_CRESCENT_CENTER_RATIOS:
        center_error = (float(center_ratio) - 0.5) / 0.5
        for top_ratio in cfg.BALL_CRESCENT_TOP_RATIOS:
            for halfspan_ratio in cfg.BALL_CRESCENT_HALFSPAN_RATIOS:
                xs, ys, slope = _circular_crescent_geometry(
                    width,
                    height,
                    center_ratio,
                    top_ratio,
                    halfspan_ratio,
                    cfg.BALL_CRESCENT_BOTTOM_RATIO,
                    normalized_x,
                )
                x_indices = np.clip(
                    np.rint(xs).astype(np.int32), 0, width - 1)
                y_indices = np.clip(
                    np.rint(ys).astype(np.int32), 0, height - 1)
                template_valid = (
                    (xs >= 0)
                    & (xs < width)
                    & (ys >= 0)
                    & (ys < height)
                )

                # A derivada do arco fornece a normal esperada da borda.
                # Um mosaico/grade pode ter muitos Canny pixels por perto, mas
                # nao mantem essa orientacao mudando suavemente de um ombro ao
                # outro nem poucos trechos coerentes de fundo->esfera. A troca
                # de sinal no miolo e permitida por causa dos reflexos do foil.
                normal_scale = np.sqrt(1.0 + np.square(slope))
                normal_x = -slope / normal_scale
                normal_y = 1.0 / normal_scale

                search_y = y_indices[:, None] + offsets[None, :]
                search_valid = (
                    template_valid[:, None]
                    & (search_y >= 0)
                    & (search_y < height)
                )
                search_y_clipped = np.clip(
                    search_y, 0, height - 1)
                search_x = np.broadcast_to(
                    x_indices[:, None], search_y.shape)
                gx = gradient_x[search_y_clipped, search_x]
                gy = gradient_y[search_y_clipped, search_x]
                magnitude = np.hypot(gx, gy)
                normal_gradient = (
                    gx * normal_x[:, None]
                    + gy * normal_y[:, None]
                )
                alignment = (
                    np.abs(normal_gradient)
                    / np.maximum(magnitude, 1e-6)
                )
                aligned_edges = (
                    search_valid
                    & edge_binary[search_y_clipped, search_x]
                    & (magnitude >= cfg.BALL_CRESCENT_MIN_GRADIENT)
                    & (
                        alignment
                        >= cfg.BALL_CRESCENT_MIN_GRADIENT_ALIGNMENT
                    )
                )
                search_labels = component_labels[
                    search_y_clipped, search_x]
                candidate_labels = np.unique(
                    search_labels[aligned_edges])
                candidate_labels = candidate_labels[
                    candidate_labels > 0]
                dominant_label = None
                dominant_coverage = -1
                for label in candidate_labels:
                    coverage = int(np.count_nonzero(np.any(
                        aligned_edges & (search_labels == label),
                        axis=1,
                    )))
                    if coverage > dominant_coverage:
                        dominant_label = int(label)
                        dominant_coverage = coverage
                component_edges = (
                    aligned_edges
                    & (search_labels == dominant_label)
                    if dominant_label is not None
                    else np.zeros_like(aligned_edges)
                )
                edge_scores = np.where(
                    component_edges, np.abs(normal_gradient), -1.0)
                best_offsets = np.argmax(edge_scores, axis=1)
                hits = np.any(component_edges, axis=1)
                selected_gradient = normal_gradient[
                    np.arange(samples), best_offsets]
                selected_residual = offsets[best_offsets]
                actual_y = y_indices + selected_residual
                circle_rmse_ratio = 1.0
                curvature_score = 0.0

                support = float(np.mean(hits))
                left_support = float(np.mean(hits[left_mask]))
                center_support = float(np.mean(hits[center_mask]))
                right_support = float(np.mean(hits[right_mask]))
                sector_masks = (left_mask, center_mask, right_mask)
                gradient_polarity = _polarity_floor(
                    selected_gradient,
                    hits,
                    sector_masks,
                )
                coherent_run = _longest_coherent_run(
                    hits,
                    selected_residual,
                    max_residual_jump=max(band_px * 0.60, 2.0),
                )

                # As amostras de preenchimento partem da borda realmente
                # encontrada, não do centro teórico da faixa. Isso permite
                # tolerar o contorno amassado sem transformar uma linha curva
                # fina, deslocada dentro da faixa, em uma esfera preenchida.
                contrast, profile_support, profile_polarity = (
                    _crescent_fill_metrics(
                        profile_gray,
                        x_indices,
                        actual_y,
                        template_valid,
                        hits,
                        normalized_x,
                        sector_masks,
                        height,
                        contrast_offset,
                        outside_contrast_offset,
                        deep_contrast_offset,
                    )
                )
                sector_floor = min(
                    left_support,
                    center_support,
                    right_support,
                )
                contrast_score = float(np.clip(
                    contrast / max(
                        cfg.BALL_CRESCENT_MIN_CONTRAST * 2.0,
                        1.0,
                    ),
                    0.0,
                    1.0,
                ))
                confidence = float(np.clip(
                    0.30 * support
                    + 0.15 * sector_floor
                    + 0.15 * gradient_polarity
                    + 0.10 * profile_support
                    + 0.15 * profile_polarity
                    + 0.10 * min(
                        coherent_run
                        / max(
                            cfg.BALL_CRESCENT_MIN_COHERENT_RUN,
                            1e-6,
                        ),
                        1.0,
                    )
                    + 0.05 * contrast_score,
                    0.0,
                    1.0,
                ))
                foil_rank = (
                    0.35 * support
                    + 0.20 * sector_floor
                    + 0.15 * profile_support
                    + 0.15 * profile_polarity
                    + 0.10 * min(
                        coherent_run
                        / max(
                            cfg.BALL_CRESCENT_FOIL_MIN_COHERENT_RUN,
                            1e-6,
                        ),
                        1.0,
                    )
                    + 0.05 * contrast_score
                )
                base_geometry_accepted = bool(
                    support >= cfg.BALL_CRESCENT_MIN_SUPPORT
                    and left_support
                    >= cfg.BALL_CRESCENT_MIN_SHOULDER_SUPPORT
                    and center_support
                    >= cfg.BALL_CRESCENT_MIN_CENTER_SUPPORT
                    and right_support
                    >= cfg.BALL_CRESCENT_MIN_SHOULDER_SUPPORT
                    and contrast >= cfg.BALL_CRESCENT_MIN_CONTRAST
                    and gradient_polarity
                    >= cfg.BALL_CRESCENT_MIN_GRADIENT_POLARITY
                    and profile_support
                    >= cfg.BALL_CRESCENT_MIN_PROFILE_SUPPORT
                    and profile_polarity
                    >= cfg.BALL_CRESCENT_MIN_PROFILE_POLARITY
                    and coherent_run
                    >= cfg.BALL_CRESCENT_MIN_COHERENT_RUN
                    and abs(center_error)
                    <= cfg.BALL_CRESCENT_MAX_CENTER_ERROR + 1e-9
                )
                foil_shape_accepted = bool(
                    support >= cfg.BALL_CRESCENT_FOIL_MIN_SUPPORT
                    and left_support
                    >= cfg.BALL_CRESCENT_FOIL_MIN_SHOULDER_SUPPORT
                    and center_support
                    >= cfg.BALL_CRESCENT_FOIL_MIN_CENTER_SUPPORT
                    and right_support
                    >= cfg.BALL_CRESCENT_FOIL_MIN_SHOULDER_SUPPORT
                    and contrast >= cfg.BALL_CRESCENT_MIN_CONTRAST
                    and profile_support
                    >= cfg.BALL_CRESCENT_MIN_PROFILE_SUPPORT
                    and profile_polarity
                    >= cfg.BALL_CRESCENT_MIN_PROFILE_POLARITY
                    and coherent_run
                    >= cfg.BALL_CRESCENT_FOIL_MIN_COHERENT_RUN
                    and abs(center_error)
                    <= cfg.BALL_CRESCENT_MAX_CENTER_ERROR + 1e-9
                )
                if base_geometry_accepted or foil_shape_accepted:
                    smooth_y = _smooth_curve_samples(actual_y, hits)
                    circle_rmse_ratio = _circle_fit_rmse_ratio(
                        x_indices[hits],
                        smooth_y[hits],
                        height,
                    )
                    curvature_score = _distributed_curvature_score(
                        normalized_x,
                        x_indices,
                        smooth_y,
                        hits,
                    )
                strict_accepted = bool(
                    base_geometry_accepted
                    and circle_rmse_ratio
                    <= cfg.BALL_CRESCENT_MAX_CIRCLE_RMSE_RATIO
                    and curvature_score
                    >= cfg.BALL_CRESCENT_MIN_CURVATURE_SCORE
                )
                foil_geometry_accepted = bool(
                    foil_shape_accepted
                    and circle_rmse_ratio
                    <= cfg.BALL_CRESCENT_FOIL_MAX_CIRCLE_RMSE_RATIO
                )
                evidence = CloseCrescentEvidence(
                    accepted=strict_accepted,
                    confidence=confidence,
                    support=support,
                    left_support=left_support,
                    center_support=center_support,
                    right_support=right_support,
                    contrast=contrast,
                    center_x_ratio=float(center_ratio),
                    top_y_ratio=float(top_ratio),
                    halfspan_ratio=float(halfspan_ratio),
                    bottom_y_ratio=cfg.BALL_CRESCENT_BOTTOM_RATIO,
                    timestamp=float(timestamp),
                    gradient_polarity=gradient_polarity,
                    profile_support=profile_support,
                    profile_polarity=profile_polarity,
                    coherent_run=coherent_run,
                    circle_rmse_ratio=circle_rmse_ratio,
                    curvature_score=curvature_score,
                )
                key = (int(strict_accepted), confidence)
                if best is None or key > best_key:
                    best = evidence
                    best_key = key
                if foil_geometry_accepted:
                    foil_key = (foil_rank, support, sector_floor)
                    # A textura e relativamente cara. Guardamos somente as
                    # melhores formas plausiveis e as medimos depois das 40
                    # hipoteses. Variacoes apenas de centro disputam a mesma
                    # vaga, evitando um top-3 de templates quase duplicados.
                    shape_key = (
                        float(top_ratio),
                        float(halfspan_ratio),
                    )
                    candidate = (
                        foil_key,
                        evidence,
                        x_indices.copy(),
                        np.where(hits, actual_y, y_indices).copy(),
                        template_valid.copy(),
                    )
                    previous = foil_candidates_by_shape.get(shape_key)
                    if previous is None or foil_key > previous[0]:
                        foil_candidates_by_shape[shape_key] = candidate

    foil_candidates = sorted(
        foil_candidates_by_shape.values(),
        key=lambda item: item[0],
        reverse=True,
    )[:cfg.BALL_CRESCENT_FOIL_MAX_CANDIDATES]

    if (
        foil_candidates
        and (best is None or not best.accepted)
    ):
        for (
            _foil_geometry_key,
            foil_evidence,
            foil_x_indices,
            foil_path_y,
            foil_template_valid,
        ) in foil_candidates:
            (
                foil_texture_bins,
                foil_valid_bins,
                interior_edge_density,
                background_edge_density,
            ) = _crescent_texture_metrics(
                gray,
                edge_binary,
                foil_x_indices,
                foil_path_y,
                foil_template_valid,
                normalized_x,
                height,
            )
            foil_texture_accepted = bool(
                foil_valid_bins
                >= cfg.BALL_CRESCENT_FOIL_MIN_TEXTURE_BINS
                and foil_texture_bins
                >= cfg.BALL_CRESCENT_FOIL_MIN_TEXTURE_BINS
                and interior_edge_density
                >= cfg.BALL_CRESCENT_FOIL_MIN_INTERIOR_EDGE_DENSITY
                and background_edge_density
                <= cfg.BALL_CRESCENT_FOIL_MAX_BACKGROUND_EDGE_DENSITY
                and background_edge_density
                <= (
                    cfg.BALL_CRESCENT_FOIL_BACKGROUND_EDGE_RATIO
                    * interior_edge_density
                )
            )
            foil_evidence = replace(
                foil_evidence,
                accepted=foil_texture_accepted,
                foil_fallback=foil_texture_accepted,
                foil_texture_bins=foil_texture_bins,
                foil_valid_bins=foil_valid_bins,
                interior_edge_density=interior_edge_density,
                background_edge_density=background_edge_density,
            )
            foil_key = (
                int(foil_texture_accepted),
                foil_evidence.confidence,
            )
            if best is None or foil_key >= best_key:
                best = foil_evidence
                best_key = foil_key
            if foil_texture_accepted:
                break

    return best


@dataclass
class _Proposal:
    center_x: float
    center_y: float
    radius: float
    circularity: float
    fill_ratio: float
    source: str
    edge_support: float = 0.0
    edge_sector_count: int = 0
    radial_dispersion: float = 1.0


@dataclass
class _Candidate:
    kind: str
    center_x: float
    center_y: float
    radius: float
    confidence: float
    arena_evidence: object = None


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

    def __init__(
        self,
        target_kind="any",
        enhance=True,
        enforce_arena=False,
        exclude_markers=False,
        arena_guardian=None,
    ):
        if target_kind not in ("any", "black", "silver"):
            raise ValueError("target_kind deve ser any, black ou silver")
        self.target_kind = target_kind
        self.enhancer = RescueEnhancer() if enhance else None
        self.enforce_arena = bool(
            enforce_arena or arena_guardian is not None)
        self.arena_guardian = (
            arena_guardian
            if arena_guardian is not None
            else (
                MonocularArenaGuardian(
                    max_work_width=cfg.ARENA_GUARD_WORK_WIDTH,
                    max_work_height=cfg.ARENA_GUARD_WORK_HEIGHT,
                )
                if self.enforce_arena else None
            )
        )
        self.marker_guards = (
            (MarkerDetector("green"), MarkerDetector("red"))
            if exclude_markers else ()
        )
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._track_locked = False
        self._pixel_scale = 1.0
        self._arena_frame_index = 0
        self._arena_cached_model = None
        self._marker_guard_frame_index = 0
        self._hough_idle_frame_index = 0
        self.last_candidates = []
        self.last_enhanced = None
        self.last_edges = None
        self.last_hough_used = False
        self.last_hough_deferred = False
        self.last_contour_proposals = 0
        self.last_hough_proposals = 0
        self.last_diagnostic = "inicio"
        self.last_crescent_evidence = None
        self.last_locked_detection = None
        self.last_arena_model = None
        self.last_arena_evidence = None
        self.last_arena_vetoed = False
        self.last_marker_exclusions = ()
        self._frame_rejections = {}

    def reset(self):
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._track_locked = False
        self._arena_frame_index = 0
        self._arena_cached_model = None
        self._marker_guard_frame_index = 0
        self._hough_idle_frame_index = 0
        self.last_candidates = []
        self.last_hough_used = False
        self.last_hough_deferred = False
        self.last_contour_proposals = 0
        self.last_hough_proposals = 0
        self.last_diagnostic = "reset"
        self.last_crescent_evidence = None
        self.last_locked_detection = None
        self.last_arena_model = None
        self.last_arena_evidence = None
        self.last_arena_vetoed = False
        self.last_marker_exclusions = ()
        for marker_guard in self.marker_guards:
            marker_guard.reset()
        self._frame_rejections = {}

    def detect(self, frame, timestamp=None):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame BGR invalido")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        self._frame_rejections = {}
        self.last_contour_proposals = 0
        self.last_hough_proposals = 0
        self.last_hough_deferred = False

        height, width = frame.shape[:2]
        self._pixel_scale = cfg.ball_pixel_scale(width, height)
        acquiring = not (
            self._track_locked
            or self._hits >= cfg.BALL_ACQUIRE_HITS
        )
        self.last_arena_model = None
        self.last_arena_evidence = None
        self.last_arena_vetoed = False
        if (
            acquiring
            and self.enforce_arena
            and self.arena_guardian is not None
        ):
            interval = max(
                int(cfg.ARENA_GUARD_INTERVAL_FRAMES), 1)
            cached = self._arena_cached_model
            refresh_arena = (
                cached is None
                or getattr(cached, "frame_width", width) != width
                or getattr(cached, "frame_height", height) != height
                or self._arena_frame_index % interval == 0
            )
            if refresh_arena:
                cached = self.arena_guardian.build_model(frame)
                self._arena_cached_model = cached
            self.last_arena_model = cached
            self._arena_frame_index += 1

        marker_exclusions = []
        marker_interval = max(
            int(cfg.MARKER_GUARD_INTERVAL_FRAMES), 1)
        run_marker_guards = (
            acquiring
            and self.marker_guards
            and self._marker_guard_frame_index % marker_interval == 0
        )
        if acquiring and self.marker_guards:
            self._marker_guard_frame_index += 1
        if run_marker_guards:
            precomputed_masks = color_masks(frame)
            minimum_marker_pixels = max(
                8,
                int(round(
                    cfg.MARKER_MIN_AREA_RATIO
                    * width
                    * height
                    * 0.50
                )),
            )
            for marker_guard in self.marker_guards:
                marker_mask = precomputed_masks[
                    marker_guard.target_kind]
                if (
                    np.count_nonzero(marker_mask)
                    < minimum_marker_pixels
                ):
                    marker_guard.reset()
                    continue
                marker_guard.detect(
                    frame,
                    timestamp=timestamp,
                    masks=precomputed_masks,
                )
                marker_exclusions.extend(marker_guard.last_candidates)
        self.last_marker_exclusions = tuple(marker_exclusions)

        enhanced = (
            self.enhancer.apply(frame) if self.enhancer is not None
            else frame.copy())
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        appearance_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_size = cfg.BALL_MEDIAN_BLUR
        if blur_size % 2 == 0:
            blur_size += 1
        gray_blur = cv2.medianBlur(gray, blur_size)

        median = float(np.median(gray_blur))
        lower = int(max(20, (1.0 - cfg.BALL_CANNY_SIGMA) * median))
        upper = int(min(255, max(lower + 30, (1.0 + cfg.BALL_CANNY_SIGMA) * median)))
        raw_edges = cv2.Canny(gray_blur, lower, upper)
        gradient_x = cv2.Sobel(
            gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(
            gray_blur, cv2.CV_32F, 0, 1, ksize=3)
        crescent_evidence = _detect_close_crescent(
            appearance_gray,
            raw_edges,
            timestamp,
        )

        roi_top = int(height * cfg.BALL_ROI_TOP)
        roi_bottom = int(height * cfg.BALL_ROI_BOTTOM)
        edges = raw_edges.copy()
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

        contour_proposals = []
        contour_proposals.extend(self._contour_proposals(
            closed_edges, "edge", self._pixel_scale))
        contour_proposals.extend(self._contour_proposals(
            dark_mask, "dark", self._pixel_scale))
        contour_proposals = self._deduplicate(contour_proposals)
        self.last_contour_proposals = len(contour_proposals)

        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        candidates = self._evaluate_proposals(
            contour_proposals,
            frame,
            hsv,
            edge_dilated,
            width,
            height,
            roi_top,
            roi_bottom,
            gradient_x,
            gradient_y,
            self.last_arena_model,
            self.last_marker_exclusions,
            acquiring,
        )

        # Hough e, de longe, o trecho mais caro no Raspberry Pi. Um contorno
        # forte ja passou por circularidade, suporte de borda e classificacao
        # de aparencia; os 3 hits temporais continuam sendo obrigatorios. Hough
        # fica como fallback para contorno ausente/fraco ou incompatibilidade
        # com o alvo rastreado.
        strong_contours = [
            candidate for candidate in candidates
            if candidate.confidence >= cfg.BALL_CONTOUR_FAST_CONFIDENCE
        ]
        if self._tracked is None:
            strong_track_match = bool(strong_contours)
        else:
            strong_track_match = any(
                self._track_match(candidate)[0]
                for candidate in strong_contours
            )
        run_hough = not strong_track_match
        if (
            run_hough
            and self._tracked is None
            and not candidates
        ):
            idle_interval = max(
                int(cfg.BALL_HOUGH_IDLE_INTERVAL_FRAMES), 1)
            run_hough = (
                self._hough_idle_frame_index % idle_interval == 0)
            self._hough_idle_frame_index += 1
            self.last_hough_deferred = not run_hough
        self.last_hough_used = run_hough
        if self.last_hough_used:
            hough_proposals = self._hough_proposals(
                gray_blur, roi_top, roi_bottom, self._pixel_scale)
            self.last_hough_proposals = len(hough_proposals)
            # Nao colapsar circulos Hough antes da aparencia. Um halo grande
            # invalido poderia apagar o perimetro verdadeiro (ou vice-versa)
            # sem que ambos tivessem a chance de ser classificados. Em
            # 320x240 o Hough produz poucas propostas e todas podem ser
            # avaliadas dentro do orcamento.
            hough_candidates = self._evaluate_proposals(
                hough_proposals,
                frame,
                hsv,
                edge_dilated,
                width,
                height,
                roi_top,
                roi_bottom,
                gradient_x,
                gradient_y,
                self.last_arena_model,
                self.last_marker_exclusions,
                acquiring,
            )
            candidates.extend(hough_candidates)

        self.last_candidates = candidates
        self.last_enhanced = enhanced
        self.last_edges = edges
        selected = self._select_candidate(candidates)
        if (
            selected is not None
            and selected.arena_evidence is not None
        ):
            # A telemetria deve descrever exatamente a proposta escolhida,
            # nao outra circunferencia avaliada antes/depois no mesmo frame.
            self.last_arena_evidence = selected.arena_evidence
            self.last_arena_vetoed = self._arena_evidence_vetoes(
                selected.arena_evidence)
        detection = self._update_track(selected, timestamp)
        self.last_crescent_evidence = crescent_evidence
        self.last_diagnostic = self._diagnostic(selected)
        return detection

    def _evaluate_proposals(
        self,
        proposals,
        frame,
        hsv,
        edge_dilated,
        width,
        height,
        roi_top,
        roi_bottom,
        gradient_x=None,
        gradient_y=None,
        arena_model=None,
        marker_exclusions=(),
        acquiring=True,
    ):
        candidates = []
        for proposal in proposals:
            arena_evidence = None
            if not self._inside_roi(proposal, width, height, roi_top, roi_bottom):
                self._reject("roi")
                continue
            if (
                acquiring
                and self._overlaps_marker(
                    proposal, marker_exclusions)
            ):
                self._reject("marcador")
                continue
            if (
                acquiring
                and self.enforce_arena
                and self.arena_guardian is not None
            ):
                arena_evidence = self.arena_guardian.evaluate(
                    arena_model, proposal)
                arena_vetoed = self._arena_evidence_vetoes(
                    arena_evidence)
                current_evidence = self.last_arena_evidence
                current_priority = (
                    2
                    if self._arena_evidence_vetoes(current_evidence)
                    else (
                        1
                        if (
                            current_evidence is not None
                            and current_evidence.accepted
                        )
                        else 0
                    )
                )
                arena_priority = (
                    2 if arena_vetoed
                    else (1 if arena_evidence.accepted else 0)
                )
                if (
                    current_evidence is None
                    or arena_priority > current_priority
                ):
                    self.last_arena_evidence = arena_evidence
                if arena_vetoed:
                    self.last_arena_vetoed = True
                    self._reject(arena_evidence.reason)
                    continue
            (
                proposal.edge_support,
                proposal.edge_sector_count,
                proposal.radial_dispersion,
                anchor_sectors_valid,
            ) = self._radial_perimeter_metrics(
                edge_dilated,
                gradient_x,
                gradient_y,
                proposal,
            )
            if (
                proposal.edge_support < cfg.BALL_MIN_EDGE_SUPPORT
                or proposal.edge_sector_count
                < cfg.BALL_RADIAL_MIN_GOOD_SECTORS
                or proposal.radial_dispersion
                > cfg.BALL_RADIAL_MAX_DISPERSION_RATIO
                or not anchor_sectors_valid
            ):
                self._reject("perimetro")
                continue
            candidate = self._classify(frame, hsv, proposal)
            if candidate is None:
                continue
            if self.target_kind != "any" and candidate.kind != self.target_kind:
                self._reject("tipo")
                continue
            candidate.arena_evidence = arena_evidence
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _arena_evidence_vetoes(evidence):
        """Usa a arena como veto forte, nunca como requisito de aquisicao.

        A lente larga curva a junção parede-piso e a porta interrompe essa
        linha. Sem um modelo confiavel, os filtros independentes de metade
        inferior, perimetro, aparencia e confirmacao temporal continuam
        decidindo. Quando a soleira e forte, objetos acima dela ainda sao
        descartados; falta de apoio no piso so veta quando o suporte e quase
        zero.
        """
        if evidence is None or evidence.accepted:
            return False
        confidence = float(
            getattr(evidence, "boundary_confidence", 0.0))
        reason = str(getattr(evidence, "reason", ""))
        if (
            reason == "fora_arena"
            and confidence >= cfg.ARENA_VETO_MIN_CONFIDENCE
        ):
            return True
        return (
            reason == "sem_apoio_piso"
            and confidence >= cfg.ARENA_VETO_MIN_CONFIDENCE
            and float(getattr(evidence, "floor_support", 1.0))
            <= cfg.ARENA_VETO_MAX_FLOOR_SUPPORT
        )

    @staticmethod
    def _overlaps_marker(proposal, marker_exclusions):
        """Triangulos validados nunca entram como propostas de esfera."""
        for marker in marker_exclusions:
            x, y, width, height = marker.bbox
            padding = max(
                2.0,
                0.08 * max(float(width), float(height)),
            )
            nearest_x = float(np.clip(
                proposal.center_x,
                float(x) - padding,
                float(x + width) + padding,
            ))
            nearest_y = float(np.clip(
                proposal.center_y,
                float(y) - padding,
                float(y + height) + padding,
            ))
            distance = math.hypot(
                proposal.center_x - nearest_x,
                proposal.center_y - nearest_y,
            )
            if distance <= proposal.radius:
                return True
        return False

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
                0.0,
                0.0,
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
        bottom_overflow = (
            height * cfg.BALL_ROI_BOTTOM_OVERFLOW_RATIO)
        return (
            proposal.center_x - proposal.radius >= margin
            and proposal.center_x + proposal.radius < width - margin
            and proposal.center_y
            >= height * cfg.BALL_TARGET_MIN_CENTER_Y_RATIO
            and proposal.center_y - proposal.radius >= roi_top
            and proposal.center_y + proposal.radius
            < min(roi_bottom, height) - margin + bottom_overflow
        )

    @staticmethod
    def _radial_perimeter_metrics(
        edges,
        gradient_x,
        gradient_y,
        proposal,
    ):
        """Mede cobertura, distribuicao e forma real do perimetro proposto."""
        if (
            gradient_x is None
            or gradient_y is None
            or gradient_x.shape != edges.shape
            or gradient_y.shape != edges.shape
        ):
            float_edges = edges.astype(np.float32)
            gradient_x = cv2.Sobel(
                float_edges, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(
                float_edges, cv2.CV_32F, 0, 1, ksize=3)

        radius = max(float(proposal.radius), 1.0)
        sample_count = min(
            max(int(cfg.BALL_RADIAL_SAMPLES), 24),
            max(24, int(round(2.0 * math.pi * radius))),
        )
        angles = np.linspace(
            0.0, 2.0 * math.pi, sample_count, endpoint=False)
        cosines = np.cos(angles)
        sines = np.sin(angles)

        band = max(
            radius * cfg.BALL_RADIAL_SEARCH_BAND_RATIO,
            1.0,
        )
        radial_steps = min(
            max(int(math.ceil(2.0 * band)) + 1, 5),
            max(int(cfg.BALL_RADIAL_MAX_STEPS), 5),
        )
        offsets = np.linspace(-band, band, radial_steps)
        sampled_radii = np.maximum(radius + offsets, 1.0)
        xs = np.rint(
            proposal.center_x
            + cosines[:, None] * sampled_radii[None, :]
        ).astype(np.int32)
        ys = np.rint(
            proposal.center_y
            + sines[:, None] * sampled_radii[None, :]
        ).astype(np.int32)

        valid = (
            (xs >= 0)
            & (xs < edges.shape[1])
            & (ys >= 0)
            & (ys < edges.shape[0])
        )
        safe_xs = np.clip(xs, 0, edges.shape[1] - 1)
        safe_ys = np.clip(ys, 0, edges.shape[0] - 1)
        gx = gradient_x[safe_ys, safe_xs]
        gy = gradient_y[safe_ys, safe_xs]
        magnitude = np.hypot(gx, gy)
        radial_gradient = (
            gx * cosines[:, None]
            + gy * sines[:, None]
        )
        alignment = (
            np.abs(radial_gradient)
            / np.maximum(magnitude, 1e-6)
        )
        eligible = (
            valid
            & (edges[safe_ys, safe_xs] > 0)
            & (magnitude >= cfg.BALL_RADIAL_MIN_GRADIENT)
            & (alignment >= cfg.BALL_RADIAL_MIN_ALIGNMENT)
        )
        scores = np.where(eligible, np.abs(radial_gradient), -1.0)
        best_indices = np.argmax(scores, axis=1)
        hits = np.any(eligible, axis=1)
        support = float(np.mean(hits))

        sector_count = max(int(cfg.BALL_RADIAL_SECTORS), 4)
        sector_width = 2.0 * math.pi / sector_count
        # Deslocar meia largura faz os setores 0, 2, 4 e 6 ficarem centrados
        # em direita, baixo, esquerda e topo, respectivamente.
        sector_indices = np.floor(
            ((angles + sector_width / 2.0) % (2.0 * math.pi))
            / sector_width
        ).astype(np.int32)
        sector_supports = np.asarray([
            float(np.mean(hits[sector_indices == index]))
            if np.any(sector_indices == index) else 0.0
            for index in range(sector_count)
        ])
        good_sectors = int(np.count_nonzero(
            sector_supports >= cfg.BALL_RADIAL_MIN_SECTOR_SUPPORT))

        cardinal_indices = (
            0,
            sector_count // 2,
            (3 * sector_count) // 4,
        )
        anchor_sectors_valid = all(
            sector_supports[index]
            >= cfg.BALL_RADIAL_MIN_SECTOR_SUPPORT
            for index in cardinal_indices
        )

        if np.any(hits):
            selected_offsets = offsets[best_indices[hits]]
            centered_offsets = (
                selected_offsets - np.median(selected_offsets))
            radial_dispersion = float(
                np.sqrt(np.mean(np.square(centered_offsets)))
                / radius
            )
        else:
            radial_dispersion = 1.0
        return (
            support,
            good_sectors,
            radial_dispersion,
            anchor_sectors_valid,
        )

    @staticmethod
    def _radial_edge_support(edges, proposal):
        """Compatibilidade: suporte simples usado por ferramentas antigas."""
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

    @staticmethod
    def _smooth_sphere_radial_metrics(hsv, proposal):
        """Contraste radial de uma esfera prata lisa.

        A esfera perolada das fotos quase nao possui a textura distribuida do
        papel-aluminio amassado. Mesmo assim, sua iluminacao e esferica: a
        regiao central tende a ser mais clara que um anel interno proximo da
        borda. Medianas tornam a medida resistente ao reflexo especular.
        """
        height, width = hsv.shape[:2]
        outer = int(math.ceil(proposal.radius * 0.82))
        x0 = max(int(proposal.center_x) - outer, 0)
        x1 = min(int(proposal.center_x) + outer + 1, width)
        y0 = max(int(proposal.center_y) - outer, 0)
        y1 = min(int(proposal.center_y) + outer + 1, height)
        crop = hsv[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0, 0

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = xx - proposal.center_x
        dy = yy - proposal.center_y
        distance = np.hypot(dx, dy)
        center = distance <= proposal.radius * 0.42
        rim = (
            (distance >= proposal.radius * 0.58)
            & (distance <= proposal.radius * 0.80)
        )
        center_values = crop[:, :, 2][center]
        rim_values = crop[:, :, 2][rim]
        if center_values.size < 12 or rim_values.size < 12:
            return 0.0, 0

        center_rim = float(
            np.median(center_values) - np.median(rim_values))
        value = crop[:, :, 2]
        quadrants = (
            (dx < 0) & (dy < 0),
            (dx >= 0) & (dy < 0),
            (dx < 0) & (dy >= 0),
            (dx >= 0) & (dy >= 0),
        )
        radial_quadrants = 0
        for quadrant in quadrants:
            quadrant_center = value[center & quadrant]
            quadrant_rim = value[rim & quadrant]
            if (
                quadrant_center.size < 3
                or quadrant_rim.size < 3
            ):
                continue
            contrast = float(
                np.median(quadrant_center)
                - np.median(quadrant_rim)
            )
            if contrast >= (
                cfg.BALL_SILVER_SMOOTH_QUADRANT_CONTRAST_MIN
            ):
                radial_quadrants += 1
        return center_rim, radial_quadrants

    @staticmethod
    def _distributed_appearance_metrics(hsv, proposal):
        """Mede textura interna e contraste de borda por setores."""
        height, width = hsv.shape[:2]
        outer = int(math.ceil(proposal.radius * 1.42))
        x0 = max(int(proposal.center_x) - outer, 0)
        x1 = min(int(proposal.center_x) + outer + 1, width)
        y0 = max(int(proposal.center_y) - outer, 0)
        y1 = min(int(proposal.center_y) + outer + 1, height)
        crop = hsv[y0:y1, x0:x1]
        if crop.size == 0:
            return 0, 0, 0, 0, 0.0

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = xx - proposal.center_x
        dy = yy - proposal.center_y
        distance = np.sqrt(np.square(dx) + np.square(dy))
        angles = (
            np.arctan2(dy, dx) + 2.0 * math.pi
        ) % (2.0 * math.pi)
        saturation = crop[:, :, 1]
        value = crop[:, :, 2]
        min_samples = max(
            int(cfg.BALL_APPEARANCE_MIN_SECTOR_SAMPLES), 1)

        texture_sectors = 0
        reflective_sectors = 0
        silver_sector_count = max(
            int(cfg.BALL_SILVER_TEXTURE_SECTORS), 1)
        for index in range(silver_sector_count):
            lower_angle = 2.0 * math.pi * index / silver_sector_count
            upper_angle = (
                2.0 * math.pi * (index + 1) / silver_sector_count)
            sector = (
                (angles >= lower_angle)
                & (angles < upper_angle)
                & (distance >= proposal.radius * 0.15)
                & (distance <= proposal.radius * 0.78)
            )
            sector_values = value[sector]
            sector_saturation = saturation[sector]
            if sector_values.size < min_samples:
                continue
            dynamic_range = float(
                np.percentile(sector_values, 90)
                - np.percentile(sector_values, 10)
            )
            if (
                dynamic_range
                >= cfg.BALL_SILVER_SECTOR_DYNAMIC_RANGE_MIN
            ):
                texture_sectors += 1
            neutral_highlights = (
                (sector_values >= cfg.BALL_SILVER_HIGHLIGHT_V)
                & (
                    sector_saturation
                    <= cfg.BALL_SILVER_TINTED_NEUTRAL_S_MAX
                )
            )
            if float(np.mean(neutral_highlights)) >= (
                cfg.BALL_SILVER_SECTOR_HIGHLIGHT_FRACTION_MIN
            ):
                reflective_sectors += 1

        contrast_sectors = 0
        usable_contrast_sectors = 0
        sector_contrasts = []
        contrast_sector_count = max(
            int(cfg.BALL_APPEARANCE_SECTORS), 1)
        for index in range(contrast_sector_count):
            lower_angle = 2.0 * math.pi * index / contrast_sector_count
            upper_angle = (
                2.0 * math.pi * (index + 1) / contrast_sector_count)
            in_angle = (
                (angles >= lower_angle)
                & (angles < upper_angle)
            )
            inner = (
                in_angle
                & (distance >= proposal.radius * 0.45)
                & (distance <= proposal.radius * 0.82)
            )
            exterior = (
                in_angle
                & (distance >= proposal.radius * 1.12)
                & (distance <= proposal.radius * 1.38)
            )
            inner_values = value[inner]
            exterior_values = value[exterior]
            if (
                inner_values.size < min_samples
                or exterior_values.size < min_samples
            ):
                continue
            usable_contrast_sectors += 1
            contrast = float(
                np.median(exterior_values)
                - np.median(inner_values)
            )
            sector_contrasts.append(contrast)
            if contrast >= cfg.BALL_BLACK_LOCAL_CONTRAST_MIN:
                contrast_sectors += 1

        median_contrast = (
            float(np.median(sector_contrasts))
            if sector_contrasts else 0.0
        )
        return (
            texture_sectors,
            reflective_sectors,
            contrast_sectors,
            usable_contrast_sectors,
            median_contrast,
        )

    def _classify(self, frame, hsv, proposal):
        del frame  # reservado para futuros descritores sem alterar a API
        inner_s, inner_v, annulus_v = self._circle_samples(hsv, proposal)
        if inner_v.size < 20 or annulus_v.size < 20:
            self._reject("amostra")
            return None

        inner_mean = float(np.mean(inner_v))
        annulus_mean = float(np.mean(annulus_v))
        dark_fraction = float(np.mean(inner_v <= cfg.BALL_BLACK_V_MAX))
        low_sat_fraction = float(np.mean(inner_s <= cfg.BALL_SILVER_S_MAX))
        dynamic_range = float(
            np.percentile(inner_v, 90) - np.percentile(inner_v, 10))
        highlight_fraction = float(
            np.mean(inner_v >= cfg.BALL_SILVER_HIGHLIGHT_V))
        neutral_highlight_fraction = float(np.mean(
            (inner_v >= cfg.BALL_SILVER_HIGHLIGHT_V)
            & (inner_s <= cfg.BALL_SILVER_TINTED_NEUTRAL_S_MAX)
        ))
        (
            center_rim_contrast,
            radial_quadrants,
        ) = self._smooth_sphere_radial_metrics(hsv, proposal)
        (
            texture_sectors,
            reflective_sectors,
            black_contrast_sectors,
            usable_contrast_sectors,
            radial_dark_contrast,
        ) = self._distributed_appearance_metrics(hsv, proposal)

        sector_geometry = float(np.clip(
            proposal.edge_sector_count
            / max(float(cfg.BALL_RADIAL_SECTORS), 1.0),
            0.0,
            1.0,
        ))
        dispersion_geometry = float(np.clip(
            1.0
            - proposal.radial_dispersion
            / max(cfg.BALL_RADIAL_MAX_DISPERSION_RATIO, 1e-6),
            0.0,
            1.0,
        ))
        perimeter_geometry = float(np.clip(
            0.55 * proposal.edge_support
            + 0.25 * sector_geometry
            + 0.20 * dispersion_geometry,
            0.0,
            1.0,
        ))
        if proposal.source == "hough":
            # Hough propoe a circunferencia, mas nao mede area nem
            # circularidade de um objeto. So evidencia realmente observada
            # participa do score.
            geometry = perimeter_geometry
        else:
            geometry = float(np.clip(
                0.25 * proposal.circularity
                + 0.15 * proposal.fill_ratio
                + 0.60 * perimeter_geometry,
                0.0,
                1.0,
            ))

        black_valid = (
            dark_fraction >= cfg.BALL_BLACK_DARK_FRACTION_MIN
            and usable_contrast_sectors
            >= cfg.BALL_BLACK_MIN_USABLE_CONTRAST_SECTORS
            and black_contrast_sectors
            >= cfg.BALL_BLACK_MIN_CONTRAST_SECTORS
        )
        black_score = float(np.clip(
            0.42 * dark_fraction
            + 0.25 * np.clip(
                radial_dark_contrast / 55.0, 0.0, 1.0)
            + 0.33 * geometry,
            0.0, 1.0))

        neutral_silver_valid = (
            low_sat_fraction >= cfg.BALL_SILVER_LOW_SAT_FRACTION_MIN)
        tinted_reflective_valid = (
            inner_mean >= cfg.BALL_SILVER_TINTED_INNER_V_MIN
            and dynamic_range >= cfg.BALL_SILVER_TINTED_DYNAMIC_RANGE_MIN
            and highlight_fraction
            >= cfg.BALL_SILVER_TINTED_HIGHLIGHT_FRACTION_MIN
            and neutral_highlight_fraction
            >= cfg.BALL_SILVER_TINTED_NEUTRAL_HIGHLIGHT_MIN
            and proposal.edge_support
            >= cfg.BALL_SILVER_TINTED_EDGE_SUPPORT_MIN)
        textured_silver_valid = (
            inner_mean > cfg.BALL_BLACK_V_MAX * 0.62
            and dynamic_range >= cfg.BALL_SILVER_DYNAMIC_RANGE_MIN
            and texture_sectors >= cfg.BALL_SILVER_MIN_TEXTURE_SECTORS
            and (
                reflective_sectors
                >= cfg.BALL_SILVER_MIN_REFLECTIVE_SECTORS
                or (
                    tinted_reflective_valid
                    and reflective_sectors >= 2
                )
            )
            and (neutral_silver_valid or tinted_reflective_valid)
            and (
                highlight_fraction >= cfg.BALL_SILVER_HIGHLIGHT_FRACTION_MIN
                or abs(annulus_mean - inner_mean)
                >= cfg.BALL_BLACK_LOCAL_CONTRAST_MIN))
        smooth_silver_valid = (
            inner_mean >= cfg.BALL_SILVER_SMOOTH_INNER_V_MIN
            and low_sat_fraction
            >= cfg.BALL_SILVER_SMOOTH_LOW_SAT_FRACTION_MIN
            and dynamic_range
            >= cfg.BALL_SILVER_SMOOTH_DYNAMIC_RANGE_MIN
            and dynamic_range
            <= cfg.BALL_SILVER_SMOOTH_DYNAMIC_RANGE_MAX
            and cfg.BALL_SILVER_SMOOTH_HIGHLIGHT_MIN
            <= highlight_fraction
            <= cfg.BALL_SILVER_SMOOTH_HIGHLIGHT_MAX
            and center_rim_contrast
            >= cfg.BALL_SILVER_SMOOTH_CENTER_RIM_MIN
            and reflective_sectors
            >= cfg.BALL_SILVER_SMOOTH_MIN_REFLECTIVE_SECTORS
            and radial_quadrants
            >= cfg.BALL_SILVER_SMOOTH_MIN_RADIAL_QUADRANTS
            and geometry >= cfg.BALL_SILVER_SMOOTH_GEOMETRY_MIN
        )
        bright_silver_valid = (
            inner_mean >= cfg.BALL_SILVER_BRIGHT_INNER_V_MIN
            and low_sat_fraction
            >= cfg.BALL_SILVER_SMOOTH_LOW_SAT_FRACTION_MIN
            and highlight_fraction
            >= cfg.BALL_SILVER_BRIGHT_HIGHLIGHT_MIN
            and highlight_fraction
            <= cfg.BALL_SILVER_BRIGHT_HIGHLIGHT_MAX
            and dynamic_range
            >= cfg.BALL_SILVER_BRIGHT_DYNAMIC_RANGE_MIN
            and center_rim_contrast
            >= cfg.BALL_SILVER_BRIGHT_CENTER_RIM_MIN
            and radial_quadrants
            >= cfg.BALL_SILVER_SMOOTH_MIN_RADIAL_QUADRANTS
            and reflective_sectors
            >= cfg.BALL_SILVER_BRIGHT_MIN_REFLECTIVE_SECTORS
            and geometry >= cfg.BALL_SILVER_BRIGHT_GEOMETRY_MIN
            and abs(annulus_mean - inner_mean)
            >= cfg.BALL_BLACK_LOCAL_CONTRAST_MIN
        )
        silver_valid = (
            textured_silver_valid
            or smooth_silver_valid
            or bright_silver_valid
        )
        neutrality_score = float(np.clip(
            low_sat_fraction
            / max(cfg.BALL_SILVER_LOW_SAT_FRACTION_MIN, 0.01),
            0.0,
            1.0,
        ))
        silver_score = float(np.clip(
            0.25 * neutrality_score
            + 0.25 * np.clip(dynamic_range / 100.0, 0.0, 1.0)
            + 0.15 * np.clip(highlight_fraction / 0.20, 0.0, 1.0)
            + 0.35 * geometry,
            0.0, 1.0))
        smooth_silver_score = float(np.clip(
            0.50 * geometry
            + 0.18 * neutrality_score
            + 0.12 * np.clip(
                inner_mean / 180.0, 0.0, 1.0)
            + 0.10 * np.clip(
                highlight_fraction / 0.05, 0.0, 1.0)
            + 0.10 * np.clip(
                center_rim_contrast / 20.0, 0.0, 1.0),
            0.0,
            1.0,
        ))
        bright_silver_score = float(np.clip(
            0.40 * geometry
            + 0.20 * neutrality_score
            + 0.20 * np.clip(
                highlight_fraction
                / max(cfg.BALL_SILVER_BRIGHT_HIGHLIGHT_MIN, 0.01),
                0.0,
                1.0,
            )
            + 0.20 * np.clip(inner_mean / 220.0, 0.0, 1.0),
            0.0,
            1.0,
        ))
        if smooth_silver_valid:
            silver_score = max(silver_score, smooth_silver_score)
        if bright_silver_valid:
            silver_score = max(silver_score, bright_silver_score)

        if black_valid and black_score >= silver_score:
            kind, confidence = "black", black_score
        elif silver_valid:
            kind, confidence = "silver", silver_score
        else:
            if inner_mean <= cfg.BALL_BLACK_V_MAX * 0.62:
                self._reject("escura")
            if (
                not neutral_silver_valid
                and not tinted_reflective_valid
                and low_sat_fraction
                < cfg.BALL_SILVER_SMOOTH_LOW_SAT_FRACTION_MIN
            ):
                self._reject("saturacao")
            if (
                not textured_silver_valid
                and not smooth_silver_valid
                and not bright_silver_valid
                and (
                    dynamic_range < cfg.BALL_SILVER_DYNAMIC_RANGE_MIN
                    or texture_sectors
                    < cfg.BALL_SILVER_MIN_TEXTURE_SECTORS
                )
            ):
                self._reject("textura")
            if (
                not textured_silver_valid
                and not smooth_silver_valid
                and not bright_silver_valid
                and (
                    reflective_sectors
                    < cfg.BALL_SILVER_SMOOTH_MIN_REFLECTIVE_SECTORS
                    or (
                        highlight_fraction
                        < cfg.BALL_SILVER_SMOOTH_HIGHLIGHT_MIN
                        and abs(annulus_mean - inner_mean)
                        < cfg.BALL_BLACK_LOCAL_CONTRAST_MIN
                    )
                )
            ):
                self._reject("reflexo")
            return None

        required_confidence = (
            cfg.BALL_HOUGH_MIN_CONFIDENCE
            if proposal.source == "hough"
            else cfg.BALL_MIN_CONFIDENCE)
        if confidence < required_confidence:
            self._reject("confianca")
            return None
        return _Candidate(
            kind,
            proposal.center_x,
            proposal.center_y,
            proposal.radius,
            confidence)

    def _reject(self, reason):
        self._frame_rejections[reason] = (
            self._frame_rejections.get(reason, 0) + 1)

    def _diagnostic(self, selected):
        if selected is not None:
            return "ok"
        if self.last_hough_deferred:
            return "hough_intercalado"
        if self.last_hough_used and self.last_hough_proposals == 0:
            return "sem_circulo"
        if self._frame_rejections:
            # Aparencia e mais util para calibrar a esfera real do que dezenas
            # de contornos do fundo rejeitados por borda/ROI.
            priority = {
                "saturacao": 8,
                "textura": 7,
                "reflexo": 6,
                "confianca": 5,
                "borda": 4,
                "perimetro": 4,
                "marcador": 4,
                "fora_arena": 4,
                "sem_apoio_piso": 4,
                "sem_area_apoio": 4,
                "sem_piso_conectado": 4,
                "sem_limite_confiavel": 4,
                "tipo": 3,
                "roi": 2,
                "amostra": 1,
                "escura": 0,
            }
            appearance_reasons = {
                reason: count
                for reason, count in self._frame_rejections.items()
                if reason in {
                    "saturacao", "textura", "reflexo",
                    "confianca", "tipo", "escura",
                }
            }
            reasons = appearance_reasons or self._frame_rejections
            return max(
                reasons,
                key=lambda reason: (
                    reasons[reason],
                    priority.get(reason, -1),
                ),
            )
        if (
            self.last_arena_model is not None
            and not self.last_arena_model.valid
        ):
            return f"arena_incerta_{self.last_arena_model.reason}"
        if self.last_contour_proposals == 0:
            return "sem_contorno"
        return "sem_candidato"

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
                max_radius = max(proposal.radius, kept.radius)
                radius_ratio = (
                    min(proposal.radius, kept.radius)
                    / max(max_radius, 1.0)
                )
                if (
                    distance
                    <= cfg.BALL_DUPLICATE_CENTER_FACTOR * max_radius
                    and radius_ratio
                    >= cfg.BALL_DUPLICATE_RADIUS_RATIO_MIN
                ):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(proposal)
        return unique

    @staticmethod
    def _prefer_outer_candidates(candidates):
        """Remove reflexos contidos quando ha envelope externo confiavel.

        A regra e local: candidatos separados continuam competindo pelo score
        normal, e um halo externo de confianca muito inferior nunca elimina um
        circulo interno forte.
        """
        preferred = []
        for inner in candidates:
            internal_reflection = False
            for outer in candidates:
                if outer is inner or outer.kind != inner.kind:
                    continue
                radius_ratio = outer.radius / max(inner.radius, 1.0)
                if not (
                    cfg.BALL_OUTER_MIN_RADIUS_RATIO
                    <= radius_ratio
                    <= cfg.BALL_OUTER_MAX_RADIUS_RATIO
                ):
                    continue
                distance = math.hypot(
                    outer.center_x - inner.center_x,
                    outer.center_y - inner.center_y)
                if (
                    distance
                    > cfg.BALL_OUTER_CENTER_FACTOR * outer.radius
                    or distance + inner.radius
                    > cfg.BALL_OUTER_CONTAINMENT_SLACK * outer.radius
                ):
                    continue
                if (
                    outer.confidence
                    < inner.confidence
                    - cfg.BALL_OUTER_CONFIDENCE_TOLERANCE
                ):
                    continue
                internal_reflection = True
                break
            if not internal_reflection:
                preferred.append(inner)
        return preferred or list(candidates)

    def _select_candidate(self, candidates):
        if not candidates:
            return None
        if self._tracked is None:
            candidates = self._prefer_outer_candidates(candidates)
            return max(
                candidates,
                key=lambda item: (item.confidence, item.radius))

        matches = []
        for candidate in candidates:
            compatible, distance, gate = self._track_match(candidate)
            if compatible:
                matches.append((candidate, distance / gate))
        if matches:
            preferred_ids = {
                id(candidate)
                for candidate in self._prefer_outer_candidates(
                    [item[0] for item in matches])
            }
            matches = [
                item for item in matches
                if id(item[0]) in preferred_ids
            ]
            # Com dois ou mais circulos compativeis, preserve a identidade
            # espacial do alvo ja rastreado. Confianca serve apenas como
            # desempate; um brilho vizinho nao pode roubar a aproximacao.
            return min(
                matches,
                key=lambda item: (item[1], -item[0].confidence))[0]
        if self._track_locked or self._hits >= cfg.BALL_ACQUIRE_HITS:
            # Depois de confirmar a vitima, um brilho incompatível nao pode
            # roubar o circulo. A perda para o robo imediatamente; o lock fica
            # vivo apenas pelo pequeno limite de misses.
            return None
        candidates = self._prefer_outer_candidates(candidates)
        return max(
            candidates,
            key=lambda item: (item.confidence, item.radius))

    def _track_match(self, candidate):
        if self._tracked is None:
            return False, float("inf"), 1.0
        if (
            candidate.kind != self._tracked.kind
            and self.target_kind != "any"
        ):
            return False, float("inf"), 1.0
        distance = math.hypot(
            candidate.center_x - self._tracked.center_x,
            candidate.center_y - self._tracked.center_y)
        acquiring = not (
            self._track_locked
            or self._hits >= cfg.BALL_ACQUIRE_HITS
        )
        if acquiring:
            association_min = cfg.BALL_ACQUIRE_ASSOCIATION_MIN_PX
            radius_factor = cfg.BALL_ACQUIRE_ASSOCIATION_RADIUS_FACTOR
            association_max = cfg.BALL_ACQUIRE_ASSOCIATION_MAX_PX
            radius_ratio_min = cfg.BALL_ACQUIRE_RADIUS_RATIO_MIN
            radius_ratio_max = cfg.BALL_ACQUIRE_RADIUS_RATIO_MAX
        else:
            association_min = cfg.BALL_ASSOCIATION_MIN_PX
            radius_factor = cfg.BALL_ASSOCIATION_RADIUS_FACTOR
            association_max = None
            radius_ratio_min = cfg.BALL_RADIUS_RATIO_MIN
            radius_ratio_max = cfg.BALL_RADIUS_RATIO_MAX
        gate = max(
            association_min * self._pixel_scale,
            radius_factor * max(candidate.radius, self._tracked.radius))
        if association_max is not None:
            gate = min(
                gate,
                association_max * self._pixel_scale,
            )
        radius_ratio = candidate.radius / max(self._tracked.radius, 1.0)
        compatible = (
            distance <= gate
            and radius_ratio_min <= radius_ratio <= radius_ratio_max)
        return compatible, distance, gate

    def _update_track(self, selected, timestamp):
        if selected is None:
            self._misses += 1
            # O controle recebe None e manda PARAR. Uma falha isolada preserva
            # a identidade/confirmacao visual. Na segunda, o lock e o circulo
            # antigo sao liberados; qualquer alvo precisa de tres hits novos.
            if (
                self._track_locked
                and self._misses > cfg.BALL_TRACK_COAST_MISSES
            ):
                self._tracked = None
                self._hits = 0
                self._track_locked = False
                self.last_locked_detection = None
            elif not self._track_locked:
                self._hits = 0
            if self._misses > cfg.BALL_MAX_TRACK_MISSES:
                frame_metrics = (
                    self.last_hough_used,
                    self.last_hough_deferred,
                    self.last_contour_proposals,
                    self.last_hough_proposals,
                    dict(self._frame_rejections),
                    self.last_arena_model,
                    self.last_arena_evidence,
                    self.last_arena_vetoed,
                    self.last_marker_exclusions,
                    self._arena_frame_index,
                    self._arena_cached_model,
                    self._marker_guard_frame_index,
                    self._hough_idle_frame_index,
                )
                self.reset()
                # reset() externo limpa telemetria; aqui o frame atual acabou
                # de usar Hough e o overlay deve preservar o motivo da rejeicao.
                (
                    self.last_hough_used,
                    self.last_hough_deferred,
                    self.last_contour_proposals,
                    self.last_hough_proposals,
                    self._frame_rejections,
                    self.last_arena_model,
                    self.last_arena_evidence,
                    self.last_arena_vetoed,
                    self.last_marker_exclusions,
                    self._arena_frame_index,
                    self._arena_cached_model,
                    self._marker_guard_frame_index,
                    self._hough_idle_frame_index,
                ) = frame_metrics
            return None

        compatible = False
        if self._tracked is not None:
            compatible = self._track_match(selected)[0]

        if compatible:
            alpha = cfg.BALL_TRACK_EMA_ALPHA
            self._tracked = _Candidate(
                self._tracked.kind,
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
            self._track_locked = False

        self._misses = 0
        if self._hits >= cfg.BALL_ACQUIRE_HITS:
            self._track_locked = True
        detection = BallDetection(
            self._tracked.kind,
            self._tracked.center_x,
            self._tracked.center_y,
            self._tracked.radius,
            self._tracked.confidence,
            self._hits >= cfg.BALL_ACQUIRE_HITS,
            self._hits,
            timestamp,
            track_locked=self._track_locked,
        )
        if detection.track_locked:
            self.last_locked_detection = detection
        return detection


def _crescent_band_points(
    width,
    height,
    center_x_ratio,
    top_y_ratio,
    halfspan_ratio,
    bottom_y_ratio,
):
    normalized_x = np.linspace(
        -1.0,
        1.0,
        max(int(cfg.BALL_CRESCENT_SAMPLES), 9),
        dtype=np.float32,
    )
    xs, ys, _ = _circular_crescent_geometry(
        width,
        height,
        center_x_ratio,
        top_y_ratio,
        halfspan_ratio,
        bottom_y_ratio,
        normalized_x,
    )
    band = max(height * cfg.BALL_CRESCENT_BAND_RATIO, 1.0)

    def points(y_values):
        return np.column_stack((
            np.clip(np.rint(xs), 0, width - 1),
            np.clip(np.rint(y_values), 0, height - 1),
        )).astype(np.int32)

    return points(ys - band), points(ys + band)


def annotate_rescue_frame(
    frame,
    detection,
    state,
    detail="",
    distance_mm=None,
    motors_enabled=False,
    performance_text="",
    pickup_in_range=False,
    pickup_confirmations=0,
    crescent_evidence=None,
):
    """Retorna uma copia anotada para debug; nao participa da decisao."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    roi_top = int(height * cfg.BALL_ROI_TOP)
    roi_bottom = int(height * cfg.BALL_ROI_BOTTOM)
    target_center_min_y = int(round(
        height * cfg.BALL_TARGET_MIN_CENTER_Y_RATIO))
    cv2.line(annotated, (0, roi_top), (width, roi_top), (90, 90, 90), 1)
    cv2.line(annotated, (0, roi_bottom), (width, roi_bottom), (90, 90, 90), 1)
    cv2.line(
        annotated,
        (0, target_center_min_y),
        (width, target_center_min_y),
        (0, 80, 255),
        2,
    )
    cv2.putText(
        annotated,
        "IGNORAR CENTROS ACIMA",
        (8, max(target_center_min_y - 7, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 80, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        annotated,
        (width // 2, 0),
        (width // 2, height),
        (0, 180, 255),
        1)

    confirmation_count = int(np.clip(
        pickup_confirmations, 0, cfg.BALL_STOP_CONFIRM_FRAMES))
    if confirmation_count >= cfg.BALL_STOP_CONFIRM_FRAMES:
        gate_color = (0, 255, 0)
    elif pickup_in_range:
        gate_color = (0, 165, 255)
    else:
        gate_color = (0, 255, 255)
    gate_thickness = 2 if pickup_in_range else 1
    pickup_point = (
        int(round(
            cfg.BALL_LOCKED_CIRCLE_POINT_X_RATIO * width)),
        int(round(
            cfg.BALL_LOCKED_CIRCLE_POINT_Y_RATIO * height)),
    )
    gate_center = (
        float(crescent_evidence.center_x_ratio)
        if crescent_evidence is not None else 0.5)
    gate_top = (
        float(crescent_evidence.top_y_ratio)
        if crescent_evidence is not None
        else cfg.BALL_CRESCENT_DEFAULT_TOP_RATIO)
    gate_halfspan = (
        float(crescent_evidence.halfspan_ratio)
        if crescent_evidence is not None
        else cfg.BALL_CRESCENT_DEFAULT_HALFSPAN_RATIO)
    gate_bottom = (
        float(crescent_evidence.bottom_y_ratio)
        if crescent_evidence is not None
        else cfg.BALL_CRESCENT_BOTTOM_RATIO)
    upper_crescent, lower_crescent = _crescent_band_points(
        width,
        height,
        gate_center,
        gate_top,
        gate_halfspan,
        gate_bottom,
    )
    cv2.polylines(
        annotated,
        [upper_crescent],
        False,
        gate_color,
        gate_thickness,
    )
    cv2.polylines(
        annotated,
        [lower_crescent],
        False,
        gate_color,
        gate_thickness,
    )
    cv2.line(
        annotated,
        tuple(upper_crescent[0]),
        tuple(lower_crescent[0]),
        gate_color,
        gate_thickness,
    )
    cv2.line(
        annotated,
        tuple(upper_crescent[-1]),
        tuple(lower_crescent[-1]),
        gate_color,
        gate_thickness,
    )
    crescent_metrics = ""
    if crescent_evidence is not None:
        shape_metrics = (
            " f="
            f"{crescent_evidence.foil_texture_bins}/"
            f"{crescent_evidence.foil_valid_bins}"
            if crescent_evidence.foil_valid_bins > 0
            else (
                " q="
                f"{crescent_evidence.curvature_score * 100:.0f}%"
            )
        )
        crescent_metrics = (
            f" s={crescent_evidence.support * 100:.0f}%"
            f" c={crescent_evidence.contrast:.0f}"
            " p="
            f"{min(
                crescent_evidence.gradient_polarity,
                crescent_evidence.profile_polarity,
            ) * 100:.0f}%"
            f"{shape_metrics}"
        )
    label_x = max(int(round(
        width * (
            gate_center - gate_halfspan
        )
    )), 3)
    label_y = max(int(round(
        height * (
            gate_top
            - cfg.BALL_CRESCENT_BAND_RATIO
        )
    )) - 7, 15)
    cv2.putText(
        annotated,
        (
            "MEIA-LUA GARRA "
            f"{confirmation_count}/{cfg.BALL_STOP_CONFIRM_FRAMES}"
            f"{crescent_metrics}"
        ),
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        gate_color,
        1,
        cv2.LINE_AA,
    )

    if detection is not None:
        color = (0, 255, 0) if detection.confirmed else (0, 180, 255)
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        radius = int(round(detection.radius))
        cv2.circle(annotated, center, radius, color, 2)
        cv2.circle(annotated, center, 3, color, -1)
        label = (
            f"{detection.kind} {detection.confidence:.2f} "
            f"r={detection.radius:.0f} hits={detection.hits}"
            f"{' LOCK' if detection.track_locked else ''}")
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
    if performance_text:
        performance_y = 84 if distance_mm is not None else 63
        cv2.putText(
            annotated, performance_text, (8, performance_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 80), 1,
            cv2.LINE_AA)
    motor_label = (
        "MOTORES: ATIVOS (--drive)"
        if motors_enabled else
        "MOTORES: DESATIVADOS (adicione --drive)")
    motor_color = (0, 210, 0) if motors_enabled else (0, 0, 255)
    cv2.putText(
        annotated, motor_label, (8, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, motor_color, 2, cv2.LINE_AA)
    # O ponto fica por ultimo para continuar visivel mesmo muito perto da
    # base, onde o texto de estado dos motores ocupa a mesma faixa.
    cv2.drawMarker(
        annotated,
        pickup_point,
        gate_color,
        cv2.MARKER_CROSS,
        15,
        gate_thickness,
    )
    cv2.putText(
        annotated,
        "PONTO GARRA",
        (
            min(pickup_point[0] + 9, max(width - 95, 0)),
            max(pickup_point[1] - 8, 15),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        gate_color,
        1,
        cv2.LINE_AA,
    )
    return annotated
