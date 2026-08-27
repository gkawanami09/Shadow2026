"""Topologia de intersecoes e marcadores verdes orientada pelo trajeto.

Este modulo nao conhece ``config`` nem os dados compartilhados do robo.  A
entrada e formada pelas mascaras *brutas* (antes dos recortes do seguidor) e o
resultado e uma observacao imutavel, apropriada para ser publicada de forma
atomica pelo processo de visao.

Convencoes geometricas
----------------------

* a tangente aponta da linha de entrada para a intersecao;
* no plano de analise, X aponta para a direita e Y para frente;
* ``cross(tangente, deslocamento) > 0`` e o lado esquerdo;
* uma homografia, quando fornecida, converte pixels diretamente para esse
  plano (normalmente em milimetros). Sem homografia, X e o pixel e Y e o pixel
  vertical com sinal invertido.

Assim, nenhuma decisao depende de topo/esquerda/direita da imagem ou da
orientacao ambigua retornada por ``minAreaRect`` para um quadrado.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import IntEnum
import math
from typing import Deque, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


class GreenDecision(IntEnum):
    """Decisao semantica de uma unica cena."""

    NONE = 0
    PENDING = 1
    STRAIGHT = 2
    LEFT = 3
    RIGHT = 4
    UTURN = 5


class MarkerPhase(IntEnum):
    """Posicao longitudinal do marcador em relacao a intersecao."""

    UNKNOWN = 0
    PRE = 1
    POST = 2
    AMBIGUOUS = 3


class PathSide(IntEnum):
    UNKNOWN = 0
    LEFT = -1
    RIGHT = 1


class BranchKind(IntEnum):
    INCOMING = 0
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3


@dataclass(frozen=True)
class TopologyConfig:
    """Limiarizacao geometrica, independente da configuracao global."""

    min_black_area_px: int = 80
    min_green_area_px: int = 28
    marker_min_aspect: float = 0.55
    marker_max_aspect: float = 1.82
    marker_min_fill: float = 0.55
    marker_min_mm: float = 18.0
    marker_max_mm: float = 35.0
    marker_min_line_widths: float = 0.55
    marker_max_line_widths: float = 2.10
    marker_junction_max_sides: float = 3.2
    pre_post_margin_sides: float = 0.35
    min_lateral_sides: float = 0.22
    min_branch_length_widths: float = 1.50
    branch_straight_limit_deg: float = 52.0
    incoming_limit_deg: float = 125.0
    black_required_support: float = 0.43
    black_forbidden_support: float = 0.52
    tangent_history_frames: int = 5
    tangent_history_weight: float = 0.40
    tangent_min_confidence: float = 0.65
    tangent_max_step_deg: float = 55.0
    entry_corridor_half_width_ratio: float = 0.14
    entry_contact_height_ratio: float = 0.15
    entry_propagation_diagonal_ratio: float = 0.18
    entry_propagation_frames: int = 2
    marker_id_ttl_frames: int = 2


@dataclass(frozen=True)
class BranchObservation:
    kind: BranchKind
    angle_deg: float
    direction: Point
    target_image: Point
    length_widths: float
    branch_token: int = 0


@dataclass(frozen=True)
class MarkerObservation:
    center_image: Point
    center_ground: Point
    side_length: float
    phase: MarkerPhase
    side: PathSide
    plausible: bool
    associated: bool
    black_to_junction: bool
    black_inward: bool
    clear_outward: bool
    clear_behind: bool
    valid: bool
    confidence: float
    touches_border: bool = False
    reason: str = ""
    marker_id: int = 0


@dataclass(frozen=True)
class TopologyObservation:
    decision: GreenDecision
    confidence: float = 0.0
    entry_tangent: Point = (0.0, 1.0)
    entry_tangent_image: Optional[Point] = None
    entry_image: Optional[Point] = None
    junction_image: Optional[Point] = None
    junction_ground: Optional[Point] = None
    branches: Tuple[BranchObservation, ...] = ()
    markers: Tuple[MarkerObservation, ...] = ()
    target_branch: Optional[BranchObservation] = None
    ready_to_turn: bool = False
    line_width_px: float = 0.0
    reason: str = ""
    junction_id: int = 0
    entry_propagated: bool = False
    geometry_truncated: bool = False

    @property
    def marker_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(marker.marker_id for marker in self.markers
                            if marker.marker_id > 0
                            and marker.plausible and marker.associated))

    @property
    def pre_markers(self) -> Tuple[MarkerObservation, ...]:
        return tuple(marker for marker in self.markers
                     if marker.phase == MarkerPhase.PRE)

    @property
    def post_markers(self) -> Tuple[MarkerObservation, ...]:
        return tuple(marker for marker in self.markers
                     if marker.phase == MarkerPhase.POST)

    @property
    def ambiguous_markers(self) -> Tuple[MarkerObservation, ...]:
        return tuple(marker for marker in self.markers
                     if marker.phase == MarkerPhase.AMBIGUOUS)


def _normalize(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return np.array((0.0, 1.0), dtype=np.float64)
    return value / norm


def _cross(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _as_binary(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        raise ValueError("a mascara nao pode ser vazia")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


class _PlaneTransform:
    def __init__(self, homography: Optional[np.ndarray]):
        self.metric = homography is not None
        if homography is None:
            self.h = None
            self.inverse = None
            return
        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("image_to_ground deve ser uma matriz 3x3 finita")
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise ValueError("image_to_ground nao pode ser singular")
        self.h = matrix
        self.inverse = np.linalg.inv(matrix)

    @staticmethod
    def _perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    def to_ground(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.h is None:
            result = values.copy()
            result[:, 1] *= -1.0
            return result
        return self._perspective(values, self.h)

    def to_image(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.inverse is None:
            result = values.copy()
            result[:, 1] *= -1.0
            return result
        return self._perspective(values, self.inverse)


def _nearest_nonzero(mask: np.ndarray, point: Point,
                     max_distance: float) -> Optional[Tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    dx = xs.astype(np.float64) - float(point[0])
    dy = ys.astype(np.float64) - float(point[1])
    distances = dx * dx + dy * dy
    index = int(np.argmin(distances))
    if float(distances[index]) > max_distance * max_distance:
        return None
    return int(xs[index]), int(ys[index])


def _entry_component(
    mask: np.ndarray,
    point: Point,
    config: TopologyConfig,
    *,
    allow_propagation: bool = False,
) -> Tuple[np.ndarray, Optional[Point], bool]:
    """Seleciona somente a componente que realmente chega ao robo.

    A busca normal fica num corredor curto ao redor do ponto de entrada. Uma
    perda temporaria pode ampliar esse alcance por poucos frames, mas essa
    permissao so e concedida pelo ``GreenTopologyTracker`` depois de contato
    real com a base.
    """
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros_like(mask), None, False
    half_width = max(8.0, width * config.entry_corridor_half_width_ratio)
    half_height = max(8.0, height * config.entry_contact_height_ratio)
    dx = xs.astype(np.float64) - float(point[0])
    dy = ys.astype(np.float64) - float(point[1])
    in_corridor = ((np.abs(dx) <= half_width)
                   & (np.abs(dy) <= half_height))
    propagated = False
    if np.any(in_corridor):
        indices = np.flatnonzero(in_corridor)
        local_distances = dx[indices] * dx[indices] + dy[indices] * dy[indices]
        selected = int(indices[int(np.argmin(local_distances))])
        seed = int(xs[selected]), int(ys[selected])
    elif allow_propagation:
        max_distance = (
            math.hypot(width, height)
            * config.entry_propagation_diagonal_ratio
        )
        seed = _nearest_nonzero(mask, point, max_distance)
        propagated = seed is not None
    else:
        seed = None
    if seed is None:
        return np.zeros_like(mask), None, False
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    label = int(labels[seed[1], seed[0]])
    if label <= 0 or label >= count:
        return np.zeros_like(mask), None, False
    if int(stats[label, cv2.CC_STAT_AREA]) < config.min_black_area_px:
        return np.zeros_like(mask), None, False
    return np.where(labels == label, 255, 0).astype(np.uint8), (
        float(seed[0]), float(seed[1])), propagated


def _component_line_width(component: np.ndarray) -> float:
    """Estima espessura por 2*area/perimetro de uma faixa longa."""
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 3.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return 3.0
    return max(3.0, 2.0 * float(cv2.contourArea(contour)) / perimeter)


def _local_ground_per_pixel(transform: _PlaneTransform, point: Point) -> float:
    """Escala local isotropica em unidades do chão por pixel."""
    if not transform.metric:
        return 1.0
    local = transform.to_ground((
        point,
        (point[0] + 1.0, point[1]),
        (point[0], point[1] - 1.0),
    ))
    scale_x = float(np.linalg.norm(local[1] - local[0]))
    scale_y = float(np.linalg.norm(local[2] - local[0]))
    return math.sqrt(max(scale_x * scale_y, 1e-9))


def _circular_runs(values: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(values, dtype=bool)
    if values.all():
        return [np.arange(len(values), dtype=np.int32)]
    if not values.any():
        return []
    start = int(np.flatnonzero(~values)[0])
    ordered = np.roll(values, -start - 1)
    runs = []
    index = 0
    while index < len(values):
        if not ordered[index]:
            index += 1
            continue
        end = index + 1
        while end < len(values) and ordered[end]:
            end += 1
        original = (np.arange(index, end) + start + 1) % len(values)
        runs.append(original.astype(np.int32))
        index = end
    return runs


def _sample_arms(component: np.ndarray, center: Point, radius: float,
                 transform: _PlaneTransform) -> list[Tuple[np.ndarray, Point]]:
    samples = 180
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    occupied = np.zeros(samples, dtype=bool)
    height, width = component.shape
    center_ground = transform.to_ground((center,))[0]
    if transform.metric:
        radius_ground = radius * _local_ground_per_pixel(transform, center)

    for scale in (0.92, 1.0, 1.08):
        if transform.metric:
            ground_ring = center_ground + radius_ground * scale * np.column_stack(
                (np.cos(angles), np.sin(angles)))
            image_ring = transform.to_image(ground_ring)
            xs = np.rint(image_ring[:, 0]).astype(int)
            ys = np.rint(image_ring[:, 1]).astype(int)
        else:
            xs = np.rint(center[0] + radius * scale * np.cos(angles)).astype(int)
            ys = np.rint(center[1] - radius * scale * np.sin(angles)).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        occupied[inside] |= component[ys[inside], xs[inside]] > 0

    arms: list[Tuple[np.ndarray, Point]] = []
    for run in _circular_runs(occupied):
        if len(run) < 3:
            continue
        vectors = np.column_stack((np.cos(angles[run]), np.sin(angles[run])))
        direction_image_plane = _normalize(vectors.mean(axis=0))
        if transform.metric:
            endpoint_ground_hint = (
                center_ground + radius_ground * direction_image_plane)
            endpoint_array = transform.to_image((endpoint_ground_hint,))[0]
            endpoint = (float(endpoint_array[0]), float(endpoint_array[1]))
        else:
            endpoint = (
                center[0] + radius * float(direction_image_plane[0]),
                center[1] - radius * float(direction_image_plane[1]),
            )
        endpoint_ground = transform.to_ground((endpoint,))[0]
        direction_ground = _normalize(endpoint_ground - center_ground)
        arms.append((direction_ground, endpoint))
    return arms


def _distance_junction_candidates(
    component: np.ndarray,
    entry_image: Point,
    line_width: float,
    transform: _PlaneTransform,
    tangent_hint: Optional[Sequence[float]],
    config: TopologyConfig,
):
    """Encontra alargamentos reais e confirma tres bracos num anel.

    Esta e a geometria equivalente ao pequeno grafo: o centro de uma faixa
    comum tem raio de aproximadamente meia largura; uma juncao tem raio maior.
    A prova de tres bracos no anel elimina curvas L e cantos arredondados. O
    metodo permanece conectado pela componente de entrada e custa bem menos
    que afinar a imagem inteira na Raspberry Pi.
    """
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    widened = np.where(
        distance >= max(2.0, 0.62 * line_width), 255, 0).astype(np.uint8)
    # Separa maximos proximos sem fragmentar o miolo de uma mesma juncao.
    close_radius = max(1, int(round(0.18 * line_width)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
    widened = cv2.morphologyEx(widened, cv2.MORPH_CLOSE, kernel)
    count, labels = cv2.connectedComponents((widened > 0).astype(np.uint8), 8)
    entry_ground = transform.to_ground((entry_image,))[0]
    hint = None if tangent_hint is None else _normalize(tangent_hint)
    probe_radius = max(6.0, line_width * config.min_branch_length_widths)
    results = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if not len(xs):
            continue
        weights = distance[ys, xs].astype(np.float64)
        center = (
            float(np.average(xs, weights=weights)),
            float(np.average(ys, weights=weights)),
        )
        arms = _sample_arms(component, center, probe_radius, transform)
        if len(arms) < 3:
            continue
        ground = transform.to_ground((center,))[0]
        delta = ground - entry_ground
        euclidean = float(np.linalg.norm(delta))
        if hint is not None:
            forward = float(np.dot(delta, hint))
            line_width_ground = (
                line_width * _local_ground_per_pixel(transform, center)
            )
            if forward <= 0.35 * line_width_ground:
                continue
            lateral = abs(_cross(hint, delta))
            score = forward + 0.20 * lateral
        else:
            score = euclidean
        results.append((score, center, arms, euclidean))
    results.sort(key=lambda item: item[0])
    return results


def _tangent_from_entry(entry_image: Point, junction_image: Point,
                        transform: _PlaneTransform,
                        hint: Optional[Sequence[float]],
                        history_weight: float) -> np.ndarray:
    points = transform.to_ground((entry_image, junction_image))
    observed = _normalize(points[1] - points[0])
    if hint is None:
        return observed
    prior = _normalize(hint)
    if np.dot(prior, observed) < 0.0:
        observed *= -1.0
    weight = float(np.clip(history_weight, 0.0, 1.0))
    return _normalize((1.0 - weight) * observed + weight * prior)


def _classify_branches(arms, tangent: np.ndarray, radius: float,
                       config: TopologyConfig) -> Tuple[BranchObservation, ...]:
    classified = []
    for direction, endpoint in arms:
        dot = float(np.clip(np.dot(tangent, direction), -1.0, 1.0))
        angle = math.degrees(math.atan2(_cross(tangent, direction), dot))
        if abs(angle) >= config.incoming_limit_deg:
            kind = BranchKind.INCOMING
        elif abs(angle) <= config.branch_straight_limit_deg:
            kind = BranchKind.STRAIGHT
        elif angle > 0.0:
            kind = BranchKind.LEFT
        else:
            kind = BranchKind.RIGHT
        classified.append(BranchObservation(
            kind=kind,
            angle_deg=float(angle),
            direction=(float(direction[0]), float(direction[1])),
            target_image=(float(endpoint[0]), float(endpoint[1])),
            length_widths=float(radius),
        ))
    # A amostragem circular pode dividir uma faixa grossa. Mantem o ramo mais
    # proximo do angulo ideal para cada classe.
    ideals = {
        BranchKind.INCOMING: 180.0,
        BranchKind.STRAIGHT: 0.0,
        BranchKind.LEFT: 90.0,
        BranchKind.RIGHT: -90.0,
    }
    unique = {}
    for branch in classified:
        ideal = ideals[branch.kind]
        error = abs(abs(branch.angle_deg) - 180.0) if branch.kind == BranchKind.INCOMING \
            else abs(branch.angle_deg - ideal)
        current = unique.get(branch.kind)
        if current is None or error < current[0]:
            unique[branch.kind] = (error, branch)
    return tuple(item[1] for item in unique.values())


def _marker_contours(
    green_mask: np.ndarray,
    *,
    repair_kernel_px: int = 3,
) -> Iterable[np.ndarray]:
    # A segmentacao HSV frequentemente abre uma fenda no meio do mesmo
    # quadrado por reflexo. O kernel e limitado e proporcional a largura da
    # linha: recompõe fragmentos do marcador sem unir os dois lados da pista.
    size = max(3, min(int(repair_kernel_px), 11))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    repaired = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        repaired, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _strip_support(mask: np.ndarray, transform: _PlaneTransform,
                   origin: np.ndarray, direction: np.ndarray,
                   start: float, length: float, half_width: float) -> float:
    """Fracao longitudinal sustentada por preto (tolera pequenos buracos)."""
    direction = _normalize(direction)
    normal = np.array((-direction[1], direction[0]), dtype=np.float64)
    along = np.linspace(max(0.0, start), max(start + 1e-3, start + length), 12)
    across = np.linspace(-half_width, half_width, 5)
    ground = origin + along[:, None, None] * direction[None, None, :]
    ground = ground + across[None, :, None] * normal[None, None, :]
    image = transform.to_image(ground.reshape(-1, 2)).reshape(12, 5, 2)
    xs = np.rint(image[:, :, 0]).astype(int)
    ys = np.rint(image[:, :, 1]).astype(int)
    inside = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    values = np.zeros_like(inside, dtype=bool)
    values[inside] = mask[ys[inside], xs[inside]] > 0
    supported = values.any(axis=1)
    # Fecha somente uma falha isolada entre dois pontos sustentados.
    if len(supported) >= 3:
        supported[1:-1] |= supported[:-2] & supported[2:]

    # ``start`` parte do centro do marcador usando um raio conservador. Num
    # quadrado, a borda na diagonal fica mais longe que a borda lateral e a
    # separacao branca regulamentar tambem ocupa alguns pixels. Por isso os
    # tres primeiros pontos podem legitimamente estar vazios (o terceiro
    # cobre a diagonal do quadrado e arredondamentos de subpixel). Depois do
    # primeiro contato, porem, o preto precisa permanecer continuo (salvo a
    # falha isolada reparada acima). Isto preserva o marcador verdadeiro e
    # rejeita um reflexo distante que so encontra a linha perto da juncao.
    occupied = np.flatnonzero(supported)
    if not len(occupied) or int(occupied[0]) > 3:
        return 0.0
    connected = supported[int(occupied[0]):]
    if np.any(~connected[:-1] & ~connected[1:]):
        return 0.0
    return float(np.mean(supported))


def _observe_markers(green_mask: np.ndarray, black_component: np.ndarray,
                     junction_image: Point, junction_ground: np.ndarray,
                     tangent: np.ndarray, branches: Tuple[BranchObservation, ...],
                     line_width: float, transform: _PlaneTransform,
                     config: TopologyConfig) -> Tuple[MarkerObservation, ...]:
    observations = []
    branch_kinds = {branch.kind for branch in branches}
    left_normal = np.array((-tangent[1], tangent[0]), dtype=np.float64)

    repair_kernel = 2 * int(math.ceil(0.22 * line_width)) + 1
    for contour in _marker_contours(
        green_mask,
        repair_kernel_px=repair_kernel,
    ):
        area_px = float(cv2.contourArea(contour))
        # O mesmo limiar que define um candidato plausivel tambem limita o
        # custo: speckles menores nao podem virar marcador nem evidencia de
        # fragmento confiavel, portanto nao executam homografia/strips.
        if area_px < config.min_green_area_px:
            continue
        # O fechamento morfologico serve apenas para recompor reflexos. Seus
        # pixels dilatados nao podem inventar contato com a borda: medimos o
        # corte somente nos pixels HSV originais pertencentes a este contorno.
        # A ROI local evita alocar uma mascara de frame inteiro por candidato.
        x_box, y_box, width_box, height_box = cv2.boundingRect(contour)
        contour_region = np.zeros((height_box, width_box), dtype=np.uint8)
        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x_box
        local_contour[:, 0, 1] -= y_box
        cv2.drawContours(contour_region, (local_contour,), -1, 255, -1)
        source_roi = green_mask[
            y_box:y_box + height_box,
            x_box:x_box + width_box,
        ]
        source_y, source_x = np.nonzero(
            (source_roi > 0) & (contour_region > 0))
        if len(source_x):
            source_left = x_box + int(np.min(source_x))
            source_right = x_box + int(np.max(source_x))
            source_top = y_box + int(np.min(source_y))
            source_bottom = y_box + int(np.max(source_y))
            touches_border = bool(
                source_left <= 1
                or source_top <= 1
                or source_right >= green_mask.shape[1] - 2
                or source_bottom >= green_mask.shape[0] - 2
            )
        else:
            touches_border = False
        if transform.metric:
            # Forma e preenchimento pertencem ao chão. Um quadrado físico
            # pode ser um trapézio alongado na imagem mesmo após remover a
            # distorção fisheye.
            contour_ground = transform.to_ground(
                contour.reshape(-1, 2)).astype(np.float32).reshape(-1, 1, 2)
            (center_ground_raw, (width_ground, height_ground), _ground_angle
             ) = cv2.minAreaRect(contour_ground)
            area_ground = abs(float(cv2.contourArea(contour_ground)))
            rect_area = max(float(width_ground * height_ground), 1e-9)
            aspect = float(width_ground / max(height_ground, 1e-9))
            fill = area_ground / rect_area
            side = float((width_ground + height_ground) * 0.5)
            plausible = bool(
                area_px >= config.min_green_area_px
                and config.marker_min_aspect <= aspect <= config.marker_max_aspect
                and fill >= config.marker_min_fill
                and config.marker_min_mm <= width_ground <= config.marker_max_mm
                and config.marker_min_mm <= height_ground <= config.marker_max_mm
            )
            center_ground = np.asarray(center_ground_raw, dtype=np.float64)
            center_image_array = transform.to_image((center_ground,))[0]
            center_image = (
                float(center_image_array[0]), float(center_image_array[1]))
        else:
            center_image_raw, (width_px, height_px), _angle = cv2.minAreaRect(
                contour)
            rect_area = max(float(width_px * height_px), 1.0)
            aspect = float(width_px / max(height_px, 1e-6))
            fill = area_px / rect_area
            side = float((width_px + height_px) * 0.5)
            plausible = bool(
                area_px >= config.min_green_area_px
                and config.marker_min_aspect <= aspect <= config.marker_max_aspect
                and fill >= config.marker_min_fill
                and config.marker_min_line_widths * line_width
                <= side <= config.marker_max_line_widths * line_width
            )
            center_image = (
                float(center_image_raw[0]), float(center_image_raw[1]))
            center_ground = transform.to_ground((center_image,))[0]
        offset = center_ground - junction_ground
        longitudinal = float(np.dot(offset, tangent))
        lateral = _cross(tangent, offset)
        margin = config.pre_post_margin_sides * max(side, 1.0)
        if longitudinal < -margin:
            phase = MarkerPhase.PRE
        elif longitudinal > margin:
            phase = MarkerPhase.POST
        else:
            phase = MarkerPhase.AMBIGUOUS
        if lateral > config.min_lateral_sides * max(side, 1.0):
            marker_side = PathSide.LEFT
            side_normal = left_normal
        elif lateral < -config.min_lateral_sides * max(side, 1.0):
            marker_side = PathSide.RIGHT
            side_normal = -left_normal
        else:
            marker_side = PathSide.UNKNOWN
            side_normal = np.zeros(2, dtype=np.float64)

        distance = float(np.linalg.norm(offset))
        associated = bool(distance <= config.marker_junction_max_sides * max(side, 1.0))
        inward = -side_normal
        to_junction = _normalize(junction_ground - center_ground)
        sample_start = 0.47 * side
        sample_length = max(0.75 * side, distance - sample_start)
        strip_width = max(1.0, 0.16 * side)
        support_junction = _strip_support(
            black_component, transform, center_ground, to_junction,
            sample_start, sample_length, strip_width)
        support_inward = 0.0 if marker_side == PathSide.UNKNOWN else _strip_support(
            black_component, transform, center_ground, inward,
            sample_start, 1.05 * side, strip_width)
        support_outward = 1.0 if marker_side == PathSide.UNKNOWN else _strip_support(
            black_component, transform, center_ground, side_normal,
            sample_start, 0.90 * side, strip_width)
        support_behind = _strip_support(
            black_component, transform, center_ground, -tangent,
            sample_start, 0.90 * side, strip_width)
        black_to_junction = support_junction >= config.black_required_support
        black_inward = support_inward >= config.black_required_support
        clear_outward = support_outward < config.black_forbidden_support
        clear_behind = support_behind < config.black_forbidden_support
        desired_branch = (BranchKind.LEFT if marker_side == PathSide.LEFT
                          else BranchKind.RIGHT)
        branch_exists = desired_branch in branch_kinds
        valid = bool(
            plausible and associated and phase == MarkerPhase.PRE
            and marker_side != PathSide.UNKNOWN
            and black_to_junction and black_inward
            and clear_outward and clear_behind and branch_exists
        )
        checks = (plausible, associated, marker_side != PathSide.UNKNOWN,
                  black_to_junction, black_inward, clear_outward, clear_behind,
                  branch_exists)
        confidence = float(sum(bool(value) for value in checks) / len(checks))
        if not plausible:
            reason = "forma/tamanho"
        elif not associated:
            reason = "outra intersecao"
        elif phase == MarkerPhase.POST:
            reason = "depois da intersecao"
        elif phase == MarkerPhase.AMBIGUOUS:
            reason = "margem PRE/POST"
        elif marker_side == PathSide.UNKNOWN:
            reason = "lado ambiguo"
        elif not branch_exists:
            reason = "ramo indicado ausente"
        elif not (black_to_junction and black_inward):
            reason = "preto obrigatorio ausente"
        elif not (clear_outward and clear_behind):
            reason = "preto no lado proibido"
        else:
            reason = "valido"
        observations.append(MarkerObservation(
            center_image=(float(center_image[0]), float(center_image[1])),
            center_ground=(float(center_ground[0]), float(center_ground[1])),
            side_length=side,
            phase=phase,
            side=marker_side,
            plausible=plausible,
            associated=associated,
            black_to_junction=black_to_junction,
            black_inward=black_inward,
            clear_outward=clear_outward,
            clear_behind=clear_behind,
            valid=valid,
            confidence=confidence,
            touches_border=touches_border,
            reason=reason,
        ))
    return tuple(observations)


def _find_branch(branches: Sequence[BranchObservation],
                 kind: BranchKind) -> Optional[BranchObservation]:
    return next((branch for branch in branches if branch.kind == kind), None)


def analyze_green_intersection(
    black_mask: np.ndarray,
    green_mask: np.ndarray,
    *,
    image_to_ground: Optional[np.ndarray] = None,
    entry_point: Optional[Point] = None,
    entry_tangent: Optional[Sequence[float]] = None,
    allow_entry_propagation: bool = False,
    config: TopologyConfig = TopologyConfig(),
) -> TopologyObservation:
    """Analisa uma cena sem usar os eixos da tela para decidir a curva.

    ``entry_point`` permanece em pixels. ``entry_tangent`` usa o plano de
    analise (X direita/Y frente); isto permite alimentar a media dos cinco
    quadros anteriores sem reconverter vetores.
    """
    black = _as_binary(black_mask)
    green = _as_binary(green_mask)
    if black.shape != green.shape:
        raise ValueError("as mascaras preta e verde precisam ter o mesmo tamanho")
    height, width = black.shape
    if entry_point is None:
        entry_point = (width * 0.5, height - 1.0)
    transform = _PlaneTransform(image_to_ground)

    # Fecha fendas de um pixel, mas nao aplica os recortes do seguidor.
    component_source = cv2.morphologyEx(
        black, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    component, actual_entry, entry_propagated = _entry_component(
        component_source,
        entry_point,
        config,
        allow_propagation=allow_entry_propagation,
    )
    if actual_entry is None:
        has_green = any(cv2.contourArea(c) >= config.min_green_area_px
                        for c in _marker_contours(green))
        return TopologyObservation(
            decision=GreenDecision.PENDING if has_green else GreenDecision.NONE,
            confidence=0.0,
            entry_tangent=tuple(_normalize(
                (0.0, 1.0) if entry_tangent is None else entry_tangent)),
            reason="verde sem linha de entrada" if has_green else "sem linha de entrada",
            entry_propagated=False,
        )

    line_width = _component_line_width(component)
    candidates = _distance_junction_candidates(
        component, actual_entry, line_width, transform, entry_tangent, config)
    has_plausible_green = any(
        cv2.contourArea(contour) >= config.min_green_area_px
        for contour in _marker_contours(green)
    )
    if not candidates:
        return TopologyObservation(
            decision=(GreenDecision.PENDING if has_plausible_green
                      else GreenDecision.NONE),
            confidence=0.1 if has_plausible_green else 0.0,
            entry_tangent=tuple(_normalize(
                (0.0, 1.0) if entry_tangent is None else entry_tangent)),
            entry_image=actual_entry,
            line_width_px=line_width,
            reason=("marcador plausivel aguardando intersecao"
                    if has_plausible_green else "sem primeira juncao"),
            entry_propagated=entry_propagated,
        )

    _junction_score, junction_image, arms, _entry_distance = candidates[0]
    tangent = _tangent_from_entry(
        actual_entry,
        junction_image,
        transform,
        entry_tangent,
        config.tangent_history_weight,
    )
    probe_widths = config.min_branch_length_widths
    branches = _classify_branches(arms, tangent, probe_widths, config)
    border_margin = max(6.0, line_width * 1.8)
    junction_truncated = bool(
        junction_image[0] <= border_margin
        or junction_image[0] >= width - 1 - border_margin
        or junction_image[1] <= border_margin
        or junction_image[1] >= height - 1 - border_margin
    )
    junction_ground = transform.to_ground((junction_image,))[0]
    markers = _observe_markers(
        green, component, junction_image, junction_ground, tangent,
        branches, line_width, transform, config)

    pre_left = [marker for marker in markers
                if marker.valid and marker.side == PathSide.LEFT]
    pre_right = [marker for marker in markers
                 if marker.valid and marker.side == PathSide.RIGHT]
    fragment_min_side = (
        config.marker_min_mm * 0.45
        if transform.metric
        else line_width * 0.45
    )
    pre_evidence = [
        marker for marker in markers
        if (
            marker.associated
            and marker.phase == MarkerPhase.PRE
            and marker.side != PathSide.UNKNOWN
            and marker.side_length >= fragment_min_side
        )
    ]
    border_marker_evidence = [
        marker for marker in markers
        if (
            marker.touches_border
            and marker.associated
            and marker.side_length >= fragment_min_side
        )
    ]
    geometry_truncated = bool(
        junction_truncated or border_marker_evidence)
    uncertain_left = [
        marker for marker in pre_evidence
        if marker.side == PathSide.LEFT and not marker.valid
    ]
    uncertain_right = [
        marker for marker in pre_evidence
        if marker.side == PathSide.RIGHT and not marker.valid
    ]
    plausible_associated = [marker for marker in markers
                            if marker.plausible and marker.associated]
    ambiguous = [marker for marker in plausible_associated
                 if marker.phase == MarkerPhase.AMBIGUOUS]
    invalid_pre = [marker for marker in plausible_associated
                   if marker.phase == MarkerPhase.PRE and not marker.valid]
    post = [marker for marker in plausible_associated
            if marker.phase == MarkerPhase.POST]
    straight_target = _find_branch(branches, BranchKind.STRAIGHT)
    truncated_marker_evidence = bool(
        border_marker_evidence
        or (junction_truncated and pre_evidence)
    )

    target = None
    if truncated_marker_evidence:
        # A borda pode esconder parte do quadrado ou um segundo marcador.
        # Mesmo um fragmento que isoladamente parece valido nao autoriza uma
        # direcao: a cena precisa voltar completa antes de votar 3/5.
        decision = GreenDecision.PENDING
        confidence = max(
            (marker.confidence for marker in markers),
            default=0.25,
        )
        reason = "geometria de marcador/ramo truncada pela borda"
    elif pre_left and pre_right:
        target = _find_branch(branches, BranchKind.INCOMING)
        if target is None:
            decision = GreenDecision.PENDING
            confidence = min(max(marker.confidence for marker in pre_left),
                             max(marker.confidence for marker in pre_right))
            reason = "dois PRE, mas ramo de entrada incompleto"
        else:
            decision = GreenDecision.UTURN
            confidence = min(max(marker.confidence for marker in pre_left),
                             max(marker.confidence for marker in pre_right))
            reason = "dois marcadores PRE opostos"
    elif pre_left and uncertain_right:
        decision = GreenDecision.PENDING
        confidence = min(
            max(marker.confidence for marker in pre_left),
            max(marker.confidence for marker in uncertain_right),
        )
        reason = "possivel segundo marcador PRE direito incompleto"
    elif pre_right and uncertain_left:
        decision = GreenDecision.PENDING
        confidence = min(
            max(marker.confidence for marker in pre_right),
            max(marker.confidence for marker in uncertain_left),
        )
        reason = "possivel segundo marcador PRE esquerdo incompleto"
    elif pre_left:
        decision = GreenDecision.LEFT
        target = _find_branch(branches, BranchKind.LEFT)
        confidence = max(marker.confidence for marker in pre_left)
        reason = "marcador PRE esquerdo"
    elif pre_right:
        decision = GreenDecision.RIGHT
        target = _find_branch(branches, BranchKind.RIGHT)
        confidence = max(marker.confidence for marker in pre_right)
        reason = "marcador PRE direito"
    elif ambiguous:
        decision = GreenDecision.PENDING
        confidence = max(marker.confidence for marker in ambiguous)
        reason = "marcador na margem PRE/POST"
    elif invalid_pre and straight_target is not None and all(
            marker.side != PathSide.UNKNOWN for marker in invalid_pre):
        # A intersecao esta completa e o marcador PRE falhou numa regra
        # objetiva (preto obrigatorio/proibido ou ramo indicado). Isso e um
        # falso verde, nao uma ordem incompleta: trava a continuacao reta.
        decision = GreenDecision.STRAIGHT
        target = straight_target
        confidence = min(0.78, max(
            marker.confidence for marker in invalid_pre))
        reason = "marcador PRE falso; ramo reto confirmado"
    elif invalid_pre:
        decision = GreenDecision.PENDING
        confidence = max(marker.confidence for marker in invalid_pre)
        reason = "geometria PRE incompleta"
    elif post and straight_target is not None:
        decision = GreenDecision.STRAIGHT
        target = straight_target
        confidence = max(marker.confidence for marker in post)
        reason = "somente marcador POST"
    elif post:
        decision = GreenDecision.PENDING
        confidence = max(marker.confidence for marker in post)
        reason = "marcador POST sem ramo reto completo"
    elif straight_target is not None:
        decision = GreenDecision.STRAIGHT
        target = straight_target
        confidence = 0.78
        reason = "intersecao sem marcador PRE"
    else:
        decision = GreenDecision.PENDING
        confidence = 0.52
        reason = "intersecao sem ramo reto completo"

    tangent_scale_px = max(24.0, line_width * 2.2)
    tangent_scale_ground = (
        tangent_scale_px * _local_ground_per_pixel(transform, junction_image)
    )
    tangent_target_ground = junction_ground + tangent * tangent_scale_ground
    tangent_target_image_array = transform.to_image((tangent_target_ground,))[0]
    tangent_target_image = (
        float(tangent_target_image_array[0]),
        float(tangent_target_image_array[1]),
    )

    return TopologyObservation(
        decision=decision,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        entry_tangent=(float(tangent[0]), float(tangent[1])),
        entry_tangent_image=tangent_target_image,
        entry_image=actual_entry,
        junction_image=junction_image,
        junction_ground=(float(junction_ground[0]), float(junction_ground[1])),
        branches=branches,
        markers=markers,
        target_branch=target,
        ready_to_turn=False,
        line_width_px=line_width,
        reason=reason,
        entry_propagated=entry_propagated,
        geometry_truncated=geometry_truncated,
    )


@dataclass
class _MarkerTrack:
    marker_id: int
    relative_ground: np.ndarray
    side: PathSide
    side_length: float
    missed_frames: int = 0


@dataclass
class GreenTopologyTracker:
    """Guarda apenas a tangente estavel dos ultimos quadros.

    Confirmacao temporal e imutabilidade da decisao pertencem ao controle; o
    historico aqui serve exclusivamente para estabilizar o referencial local.
    """

    config: TopologyConfig = field(default_factory=TopologyConfig)
    _tangents: Deque[np.ndarray] = field(init=False, repr=False)
    _next_junction_id: int = field(init=False, default=1, repr=False)
    _junction_id: int = field(init=False, default=0, repr=False)
    _last_junction_image: Optional[np.ndarray] = field(
        init=False, default=None, repr=False)
    _marker_tracks: list = field(init=False, default_factory=list, repr=False)
    _next_marker_id: int = field(init=False, default=1, repr=False)
    _missing_junction_frames: int = field(init=False, default=0, repr=False)
    _entry_grace_remaining: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        self._tangents = deque(maxlen=max(1, self.config.tangent_history_frames))

    @property
    def stable_tangent(self) -> Optional[Point]:
        if not self._tangents:
            return None
        reference = self._tangents[-1]
        aligned = [(-value if np.dot(value, reference) < 0.0 else value)
                   for value in self._tangents]
        result = _normalize(np.mean(aligned, axis=0))
        return float(result[0]), float(result[1])

    def reset(self):
        self._tangents.clear()
        self._junction_id = 0
        self._last_junction_image = None
        self._marker_tracks.clear()
        self._missing_junction_frames = 0
        self._entry_grace_remaining = 0

    def _assign_ids(self, observation: TopologyObservation) -> TopologyObservation:
        if observation.junction_image is None:
            self._missing_junction_frames += 1
            retained_tracks = []
            for track in self._marker_tracks:
                track.missed_frames += 1
                if track.missed_frames <= self.config.marker_id_ttl_frames:
                    retained_tracks.append(track)
            self._marker_tracks = retained_tracks
            if self._missing_junction_frames > 5:
                self._junction_id = 0
                self._last_junction_image = None
                self._marker_tracks.clear()
            return observation

        current = np.asarray(observation.junction_image, dtype=np.float64)
        same = False
        if self._last_junction_image is not None:
            threshold = max(18.0, 3.2 * observation.line_width_px)
            same = float(np.linalg.norm(current - self._last_junction_image)) <= threshold
        if not same:
            self._junction_id = self._next_junction_id
            self._next_junction_id += 1
            self._marker_tracks.clear()
        self._last_junction_image = current
        self._missing_junction_frames = 0

        available = list(self._marker_tracks)
        updated_tracks = []
        markers = []
        junction_ground = np.asarray(
            observation.junction_ground, dtype=np.float64)
        for marker in observation.markers:
            marker_id = 0
            if marker.plausible and marker.associated:
                relative_ground = (
                    np.asarray(marker.center_ground, dtype=np.float64)
                    - junction_ground
                )
                best_index = None
                best_distance = float("inf")
                for index, track in enumerate(available):
                    if (track.side != marker.side
                            and track.side != PathSide.UNKNOWN
                            and marker.side != PathSide.UNKNOWN):
                        continue
                    distance = float(np.linalg.norm(
                        relative_ground - track.relative_ground))
                    limit = 2.5 * max(track.side_length, marker.side_length)
                    if distance <= limit and distance < best_distance:
                        best_index = index
                        best_distance = distance
                if best_index is None:
                    marker_id = self._next_marker_id
                    self._next_marker_id += 1
                else:
                    marker_id = int(available.pop(best_index).marker_id)
                updated_tracks.append(_MarkerTrack(
                    marker_id=marker_id,
                    relative_ground=relative_ground,
                    side=marker.side,
                    side_length=marker.side_length,
                    missed_frames=0,
                ))
            markers.append(replace(marker, marker_id=marker_id))
        for track in available:
            track.missed_frames += 1
            if track.missed_frames <= self.config.marker_id_ttl_frames:
                updated_tracks.append(track)
        self._marker_tracks = updated_tracks
        branches = tuple(
            replace(
                branch,
                branch_token=(self._junction_id << 3) | (int(branch.kind) + 1),
            )
            for branch in observation.branches
        )
        target = None
        if observation.target_branch is not None:
            target = next(
                (branch for branch in branches
                 if branch.kind == observation.target_branch.kind),
                None,
            )
        return replace(
            observation,
            junction_id=self._junction_id,
            markers=tuple(markers),
            branches=branches,
            target_branch=target,
        )

    def update(self, black_mask: np.ndarray, green_mask: np.ndarray, *,
               image_to_ground: Optional[np.ndarray] = None,
               entry_point: Optional[Point] = None,
               entry_tangent: Optional[Sequence[float]] = None) -> TopologyObservation:
        prior = entry_tangent if entry_tangent is not None else self.stable_tangent
        observation = analyze_green_intersection(
            black_mask,
            green_mask,
            image_to_ground=image_to_ground,
            entry_point=entry_point,
            entry_tangent=prior,
            allow_entry_propagation=(self._entry_grace_remaining > 0),
            config=self.config,
        )
        if observation.entry_image is None:
            self._entry_grace_remaining = max(
                0, self._entry_grace_remaining - 1)
        elif observation.entry_propagated:
            self._entry_grace_remaining = max(
                0, self._entry_grace_remaining - 1)
        else:
            self._entry_grace_remaining = max(
                0, int(self.config.entry_propagation_frames))

        tangent_stable = bool(
            observation.junction_image is not None
            and not observation.entry_propagated
            and observation.decision != GreenDecision.PENDING
            and observation.confidence >= self.config.tangent_min_confidence
        )
        value = _normalize(observation.entry_tangent)
        if tangent_stable and self._tangents:
            cosine = float(np.clip(
                np.dot(value, self._tangents[-1]), -1.0, 1.0))
            tangent_stable = bool(
                math.degrees(math.acos(cosine))
                <= self.config.tangent_max_step_deg
            )
        if tangent_stable:
            self._tangents.append(value)
        return self._assign_ids(observation)


def draw_topology_debug(image: np.ndarray, observation: TopologyObservation) -> np.ndarray:
    """Desenha o referencial e as classificacoes sem alterar a observacao."""
    if observation.junction_image is not None:
        junction = tuple(
            int(round(value)) for value in observation.junction_image)
        cv2.circle(image, junction, 6, (0, 180, 255), 2)
        if observation.entry_tangent_image is not None:
            tangent_end = tuple(
                int(round(value))
                for value in observation.entry_tangent_image)
        else:
            scale = max(24.0, observation.line_width_px * 2.2)
            tangent = observation.entry_tangent
            tangent_end = (
                int(round(junction[0] + scale * tangent[0])),
                int(round(junction[1] - scale * tangent[1])),
            )
        cv2.arrowedLine(image, junction, tangent_end, (255, 255, 0), 2,
                        tipLength=0.22)
        colors = {
            BranchKind.INCOMING: (120, 120, 120),
            BranchKind.STRAIGHT: (255, 255, 0),
            BranchKind.LEFT: (255, 0, 255),
            BranchKind.RIGHT: (0, 128, 255),
        }
        for branch in observation.branches:
            target = tuple(
                int(round(value)) for value in branch.target_image)
            cv2.line(image, junction, target, colors[branch.kind], 2)
    phase_names = {
        MarkerPhase.UNKNOWN: "?",
        MarkerPhase.PRE: "PRE",
        MarkerPhase.POST: "POST",
        MarkerPhase.AMBIGUOUS: "AMB",
    }
    for marker in observation.markers:
        center = tuple(int(round(value)) for value in marker.center_image)
        color = (0, 255, 0) if marker.valid else (0, 0, 255)
        cv2.circle(image, center, 4, color, -1)
        cv2.putText(image, phase_names[marker.phase], (center[0] + 5, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return image


__all__ = [
    "BranchKind",
    "BranchObservation",
    "GreenDecision",
    "GreenTopologyTracker",
    "MarkerObservation",
    "MarkerPhase",
    "PathSide",
    "TopologyConfig",
    "TopologyObservation",
    "analyze_green_intersection",
    "draw_topology_debug",
]
