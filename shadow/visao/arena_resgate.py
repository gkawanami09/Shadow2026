"""Guardiao monocular conservador para a area de resgate.

O modulo nao tenta reconhecer uma esfera. Ele responde a uma pergunta mais
simples: o ponto de contato de uma proposta circular parece estar apoiado no
piso da arena, do lado interno do limite parede-piso?

O limite e estimado por uma mudanca de aparencia sustentada acima/abaixo da
linha. Segmentos coerentes dos dois lados de uma abertura sustentam a mesma
reta; assim, a reta interpolada funciona como uma soleira virtual para uma
porta. O piso permitido e a parte dessa regiao que continua conectada a
sementes na base da imagem.

Todos os limiares geometricos sao proporcoes da imagem de trabalho. A imagem
e reduzida somente para esta analise, portanto a API aceita qualquer resolucao
e devolve coordenadas na resolucao original.

Limitacao fisica: se um objeto externo projeta seu ponto de contato no mesmo
lado da soleira que o piso interno, uma unica camera nao fornece informacao
suficiente para provar de que lado ele esta. Nesse caso, este guardiao deve ser
combinado com uma soleira visual conhecida, odometria ou outro sensor.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


# Defaults internos intencionalmente isolados de config_resgate.py. Eles
# descrevem geometria normalizada, nao pixels de uma camera especifica.
_MAX_WORK_WIDTH = 320
_MAX_WORK_HEIGHT = 240
_BOUNDARY_SEARCH_TOP = 0.34
_BOUNDARY_SEARCH_BOTTOM = 0.88
_BOUNDARY_SAMPLE_OFFSET = 0.018
_BOUNDARY_MAX_SLOPE = 0.65
_BOUNDARY_MIN_SEGMENT = 0.10
_BOUNDARY_MAX_LINE_GAP = 0.07
_BOUNDARY_MIN_COVERAGE = 0.26
_BOUNDARY_MIN_SPAN = 0.52
_BOUNDARY_MIN_CONFIDENCE = 0.52
_BOUNDARY_MIN_CONTRAST = 9.0
# O contraste sustentado forma uma faixa dos dois lados da juncao. Usar a
# borda inferior dessa faixa e conservador: a soleira nao invade a parede e a
# tolerancia de contato abaixo ainda preserva uma esfera encostada nela.
_BOUNDARY_FLOOR_SIDE_BIAS = 0.012
_FLOOR_SEED_TOP = 0.80
_FLOOR_CONNECT_TOP = 0.90
_FLOOR_CHROMA_TOLERANCE = 32.0
_FLOOR_LIGHTNESS_TOLERANCE = 92.0
_CONTACT_HALF_WIDTH = 0.30
_CONTACT_DEPTH = 0.18
_CONTACT_MIN_DEPTH_RATIO = 0.014
_CONTACT_BOUNDARY_TOLERANCE = 0.020
_CONTACT_MIN_FLOOR_SUPPORT = 0.38


@dataclass
class ArenaFloorModel:
    """Modelo por quadro, reutilizavel para todas as propostas circulares."""

    frame_width: int
    frame_height: int
    work_width: int
    work_height: int
    boundary_y: object
    observed_columns: object
    floor_mask: object
    confidence: float
    reason: str

    @property
    def valid(self):
        return (
            self.reason == "ok"
            and self.boundary_y is not None
            and self.floor_mask is not None
        )

    def boundary_points(self, count=33):
        """Pontos da soleira em coordenadas do frame original."""
        if self.boundary_y is None:
            return ()
        count = max(2, min(int(count), self.work_width))
        xs = np.linspace(0, self.work_width - 1, count)
        indices = np.clip(np.rint(xs).astype(int), 0, self.work_width - 1)
        scale_x = self.frame_width / max(float(self.work_width), 1.0)
        scale_y = self.frame_height / max(float(self.work_height), 1.0)
        return tuple(
            (
                int(round(float(x) * scale_x)),
                int(round(float(self.boundary_y[index]) * scale_y)),
            )
            for x, index in zip(xs, indices)
        )


@dataclass
class ArenaSupportEvidence:
    """Resultado conservador para uma proposta.

    ``model`` pode ser reutilizado para avaliar os demais candidatos do mesmo
    quadro sem repetir Sobel, Hough, segmentacao ou componentes conexos.
    """

    valid: bool
    reason: str
    boundary_confidence: float
    floor_support: float
    boundary_y: object
    contact_y: float
    support_box: object
    model: ArenaFloorModel

    @property
    def accepted(self):
        return self.valid

    def boundary_points(self, count=33):
        return self.model.boundary_points(count)


class MonocularArenaGuardian:
    """Estima a soleira da arena e valida apoio de candidatos no piso."""

    def __init__(
        self,
        max_work_width=_MAX_WORK_WIDTH,
        max_work_height=_MAX_WORK_HEIGHT,
        min_boundary_confidence=_BOUNDARY_MIN_CONFIDENCE,
        min_floor_support=_CONTACT_MIN_FLOOR_SUPPORT,
    ):
        self.max_work_width = max(int(max_work_width), 64)
        self.max_work_height = max(int(max_work_height), 48)
        self.min_boundary_confidence = float(min_boundary_confidence)
        self.min_floor_support = float(min_floor_support)

    def build_model(self, frame):
        """Cria um modelo fail-closed para um frame BGR."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] < 24
            or frame.shape[1] < 32
        ):
            raise ValueError("frame BGR invalido")

        frame_height, frame_width = frame.shape[:2]
        work, _, _ = self._working_frame(frame)
        work_height, work_width = work.shape[:2]
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        contrast = self._sustained_vertical_contrast(lab, gray)
        boundary, observed, confidence = self._estimate_boundary(contrast)
        if boundary is None or confidence < self.min_boundary_confidence:
            return ArenaFloorModel(
                frame_width,
                frame_height,
                work_width,
                work_height,
                None,
                None,
                None,
                float(confidence),
                "sem_limite_confiavel",
            )

        floor_mask = self._connected_floor_mask(lab, gray, boundary)
        if floor_mask is None or not np.any(floor_mask):
            return ArenaFloorModel(
                frame_width,
                frame_height,
                work_width,
                work_height,
                boundary,
                observed,
                np.zeros((work_height, work_width), dtype=bool),
                float(confidence),
                "sem_piso_conectado",
            )

        return ArenaFloorModel(
            frame_width,
            frame_height,
            work_width,
            work_height,
            boundary,
            observed,
            floor_mask,
            float(confidence),
            "ok",
        )

    def evaluate(
        self,
        model,
        proposal=None,
        *,
        center_x=None,
        center_y=None,
        radius=None,
    ):
        """Avalia uma proposta contra um ``ArenaFloorModel`` existente."""
        if not isinstance(model, ArenaFloorModel):
            raise TypeError("model precisa ser ArenaFloorModel")
        center_x, center_y, radius = self._proposal_values(
            proposal, center_x, center_y, radius)

        if not model.valid:
            return ArenaSupportEvidence(
                False,
                model.reason,
                model.confidence,
                0.0,
                None,
                center_y + radius,
                None,
                model,
            )

        scale_x = model.work_width / max(float(model.frame_width), 1.0)
        scale_y = model.work_height / max(float(model.frame_height), 1.0)
        cx = center_x * scale_x
        cy = center_y * scale_y
        # O resize preserva proporcao, mas min() evita que uma entrada com
        # pixels nao quadrados transforme o raio em uma elipse artificial.
        work_radius = radius * min(scale_x, scale_y)
        contact_y = cy + work_radius
        x_index = int(np.clip(round(cx), 0, model.work_width - 1))
        boundary_y = float(model.boundary_y[x_index])
        boundary_tolerance = max(
            2.0,
            _CONTACT_BOUNDARY_TOLERANCE * model.work_height,
            0.10 * work_radius,
        )

        original_contact_y = center_y + radius
        original_boundary_y = (
            boundary_y
            * model.frame_height
            / max(float(model.work_height), 1.0)
        )
        if contact_y < boundary_y - boundary_tolerance:
            return ArenaSupportEvidence(
                False,
                "fora_arena",
                model.confidence,
                0.0,
                original_boundary_y,
                original_contact_y,
                None,
                model,
            )

        half_width = max(2, int(round(
            _CONTACT_HALF_WIDTH * max(work_radius, 1.0))))
        depth = max(
            3,
            int(round(_CONTACT_DEPTH * max(work_radius, 1.0))),
            int(round(_CONTACT_MIN_DEPTH_RATIO * model.work_height)),
        )
        # Comeca um pixel abaixo da tangencia. A faixa e central e estreita:
        # uma parede ao lado da esfera nao elimina o apoio que existe embaixo.
        x0 = max(0, int(math.floor(cx)) - half_width)
        x1 = min(model.work_width, int(math.ceil(cx)) + half_width + 1)
        y0 = max(0, int(math.floor(contact_y)) + 1)
        y1 = min(model.work_height, y0 + depth)
        if x1 <= x0 or y1 <= y0:
            return ArenaSupportEvidence(
                False,
                "sem_area_apoio",
                model.confidence,
                0.0,
                original_boundary_y,
                original_contact_y,
                self._box_to_original(model, x0, y0, x1, y1),
                model,
            )

        support = float(np.mean(model.floor_mask[y0:y1, x0:x1]))
        support_box = self._box_to_original(model, x0, y0, x1, y1)
        if support < self.min_floor_support:
            return ArenaSupportEvidence(
                False,
                "sem_apoio_piso",
                model.confidence,
                support,
                original_boundary_y,
                original_contact_y,
                support_box,
                model,
            )

        return ArenaSupportEvidence(
            True,
            "ok",
            model.confidence,
            support,
            original_boundary_y,
            original_contact_y,
            support_box,
            model,
        )

    def inspect(
        self,
        frame,
        proposal=None,
        *,
        center_x=None,
        center_y=None,
        radius=None,
    ):
        """Atalho que constroi o modelo e avalia uma unica proposta."""
        model = self.build_model(frame)
        return self.evaluate(
            model,
            proposal,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )

    def _working_frame(self, frame):
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            self.max_work_width / max(float(width), 1.0),
            self.max_work_height / max(float(height), 1.0),
        )
        work_width = max(32, int(round(width * scale)))
        work_height = max(24, int(round(height * scale)))
        if work_width == width and work_height == height:
            return frame, 1.0, 1.0
        work = cv2.resize(
            frame,
            (work_width, work_height),
            interpolation=cv2.INTER_AREA,
        )
        return work, work_width / width, work_height / height

    @staticmethod
    def _sustained_vertical_contrast(lab, gray):
        height, width = gray.shape
        offset = max(2, int(round(_BOUNDARY_SAMPLE_OFFSET * height)))
        smoothed_lab = cv2.GaussianBlur(lab, (5, 5), 0)
        contrast = np.zeros((height, width), dtype=np.float32)
        if height > 2 * offset:
            delta = (
                smoothed_lab[2 * offset:, :, :]
                - smoothed_lab[:-2 * offset, :, :]
            )
            contrast[offset:-offset, :] = np.linalg.norm(
                delta, axis=2)

        # Sobel ajuda quando parede e piso tem cromaticidade parecida, mas
        # recebe peso baixo para que um rejunte fino nao pareca uma soleira.
        sobel_y = np.abs(cv2.Sobel(
            gray, cv2.CV_32F, 0, 1, ksize=3))
        contrast += 0.08 * sobel_y
        sigma_x = max(1.0, 0.012 * width)
        return cv2.GaussianBlur(contrast, (0, 0), sigma_x, 0.8)

    def _estimate_boundary(self, contrast):
        height, width = contrast.shape
        top = max(2, int(round(_BOUNDARY_SEARCH_TOP * height)))
        bottom = min(
            height - 2,
            int(round(_BOUNDARY_SEARCH_BOTTOM * height)),
        )
        if bottom <= top + 3:
            return None, None, 0.0

        band = contrast[top:bottom, :]
        positive = band[np.isfinite(band)]
        if positive.size == 0:
            return None, None, 0.0
        percentile = float(np.percentile(positive, 82))
        threshold = max(_BOUNDARY_MIN_CONTRAST, percentile)
        binary = np.zeros_like(contrast, dtype=np.uint8)
        binary[top:bottom, :] = np.where(
            band >= threshold, 255, 0).astype(np.uint8)
        close_width = max(3, int(round(0.022 * width)))
        if close_width % 2 == 0:
            close_width += 1
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones((1, close_width), dtype=np.uint8),
        )

        lines = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 180.0,
            threshold=max(10, int(round(0.055 * width))),
            minLineLength=max(8, int(round(_BOUNDARY_MIN_SEGMENT * width))),
            maxLineGap=max(3, int(round(_BOUNDARY_MAX_LINE_GAP * width))),
        )
        if lines is None:
            return None, None, 0.0

        hypotheses = []
        for packed in lines[:, 0, :]:
            x1, y1, x2, y2 = (float(value) for value in packed)
            dx = x2 - x1
            if abs(dx) < _BOUNDARY_MIN_SEGMENT * width:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) > _BOUNDARY_MAX_SLOPE:
                continue
            intercept = y1 - slope * x1
            predicted = slope * np.arange(width, dtype=np.float32) + intercept
            if (
                np.mean((predicted >= top) & (predicted < bottom))
                < 0.80
            ):
                continue
            hypotheses.append((slope, intercept))
        if not hypotheses:
            return None, None, 0.0

        best = None
        search_radius = max(2, int(round(0.014 * height)))
        xs = np.arange(width)
        for slope, intercept in hypotheses:
            predicted = slope * xs + intercept
            local_values = np.zeros(width, dtype=np.float32)
            local_ys = np.rint(predicted).astype(np.int32)
            for delta_y in range(-search_radius, search_radius + 1):
                ys = np.clip(
                    np.rint(predicted + delta_y).astype(np.int32),
                    top,
                    bottom - 1,
                )
                values = contrast[ys, xs]
                replace = values > local_values
                local_values[replace] = values[replace]
                local_ys[replace] = ys[replace]

            observed = local_values >= threshold
            coverage = float(np.mean(observed))
            indices = np.flatnonzero(observed)
            span = (
                float(indices[-1] - indices[0]) / max(width - 1, 1)
                if indices.size >= 2 else 0.0
            )
            strength = float(np.mean(np.clip(
                local_values[observed] / max(2.0 * threshold, 1.0),
                0.0,
                1.0,
            ))) if np.any(observed) else 0.0
            # Uma porta reduz cobertura, mas deve deixar evidencia nos dois
            # lados. O span impede extrapolar um pedaco isolado de parede.
            quality = 0.58 * coverage + 0.27 * span + 0.15 * strength
            if (
                best is None
                or quality > best[0] + 1e-6
                or (
                    abs(quality - best[0]) <= 1e-6
                    and float(np.median(predicted))
                    < float(np.median(best[2]))
                )
            ):
                best = (
                    quality,
                    local_ys,
                    predicted.astype(np.float32),
                    observed,
                    coverage,
                    span,
                    strength,
                )

        if best is None:
            return None, None, 0.0
        _, local_ys, predicted, observed, coverage, span, strength = best
        if (
            coverage < _BOUNDARY_MIN_COVERAGE
            or span < _BOUNDARY_MIN_SPAN
            or np.count_nonzero(observed) < 8
        ):
            confidence = (
                0.55 * min(coverage / _BOUNDARY_MIN_COVERAGE, 1.0)
                + 0.30 * min(span / _BOUNDARY_MIN_SPAN, 1.0)
                + 0.15 * strength
            ) * 0.49
            return None, None, float(np.clip(confidence, 0.0, 0.49))

        # Reajuste robusto com os pixels realmente observados. A reta final
        # interpola naturalmente a lacuna da porta.
        points = np.column_stack((
            xs[observed].astype(np.float32),
            local_ys[observed].astype(np.float32),
        ))
        vx, vy, x0, y0 = (
            float(value)
            for value in cv2.fitLine(
                points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        )
        if abs(vx) < 1e-5:
            return None, None, 0.0
        refined_slope = vy / vx
        if abs(refined_slope) > _BOUNDARY_MAX_SLOPE:
            return None, None, 0.0
        refined = (
            refined_slope * (xs.astype(np.float32) - x0) + y0
        )
        refined += max(1.0, _BOUNDARY_FLOOR_SIDE_BIAS * height)
        refined = np.clip(refined, top, bottom - 1).astype(np.float32)

        # Cobertura de 0,26 e span de 0,52 sao o minimo; uma linha que cruza
        # quase toda a imagem chega perto de 1.0. Falhas na porta reduzem
        # cobertura, nao a validade da interpolacao.
        confidence = (
            0.50 * min(coverage / 0.65, 1.0)
            + 0.30 * span
            + 0.20 * strength
        )
        return refined, observed.astype(bool), float(np.clip(
            confidence, 0.0, 1.0))

    @staticmethod
    def _connected_floor_mask(lab, gray, boundary):
        height, width = gray.shape
        yy = np.arange(height, dtype=np.float32)[:, None]
        geometric_floor = yy >= (
            boundary[None, :] - max(1.0, 0.010 * height))
        seed_top = int(round(_FLOOR_SEED_TOP * height))
        seed_top = min(max(seed_top, 0), height - 1)
        seed_region = geometric_floor.copy()
        seed_region[:seed_top, :] = False

        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        seed_gradients = gradient[seed_region]
        gradient_limit = (
            float(np.percentile(seed_gradients, 80))
            if seed_gradients.size else 255.0
        )

        prototypes = []
        stripe_count = 5
        for stripe in range(stripe_count):
            x0 = int(round(stripe * width / stripe_count))
            x1 = int(round((stripe + 1) * width / stripe_count))
            local_mask = seed_region[:, x0:x1].copy()
            local_mask &= (
                gradient[:, x0:x1]
                <= max(gradient_limit, 12.0)
            )
            values = lab[:, x0:x1, :][local_mask]
            if values.shape[0] < 12:
                values = lab[:, x0:x1, :][
                    seed_region[:, x0:x1]]
            if values.shape[0] >= 8:
                prototypes.append(np.median(values, axis=0))
        if not prototypes:
            return None

        floor_like = np.zeros((height, width), dtype=bool)
        lightness_tolerance = _FLOOR_LIGHTNESS_TOLERANCE
        chroma_tolerance = _FLOOR_CHROMA_TOLERANCE
        for prototype in prototypes:
            lightness_delta = np.abs(lab[:, :, 0] - prototype[0])
            chroma_delta = np.linalg.norm(
                lab[:, :, 1:3] - prototype[1:3],
                axis=2,
            )
            floor_like |= (
                (lightness_delta <= lightness_tolerance)
                & (chroma_delta <= chroma_tolerance)
            )
        floor_like &= geometric_floor

        close_size = max(3, int(round(0.018 * min(width, height))))
        if close_size % 2 == 0:
            close_size += 1
        mask_u8 = (floor_like.astype(np.uint8) * 255)
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), dtype=np.uint8),
            iterations=2,
        )
        mask_u8 &= (geometric_floor.astype(np.uint8) * 255)

        count, labels = cv2.connectedComponents(
            (mask_u8 > 0).astype(np.uint8), connectivity=8)
        if count <= 1:
            return None
        connect_top = int(round(_FLOOR_CONNECT_TOP * height))
        connect_top = min(max(connect_top, 0), height - 1)
        seed_labels = np.unique(labels[connect_top:, :])
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            return None
        return np.isin(labels, seed_labels) & geometric_floor

    @staticmethod
    def _proposal_values(proposal, center_x, center_y, radius):
        if proposal is not None:
            if all(hasattr(proposal, field) for field in (
                "center_x", "center_y", "radius"
            )):
                center_x = proposal.center_x
                center_y = proposal.center_y
                radius = proposal.radius
            elif isinstance(proposal, (tuple, list, np.ndarray)) and len(
                    proposal) >= 3:
                center_x, center_y, radius = proposal[:3]
            else:
                raise TypeError(
                    "proposal deve expor center_x/center_y/radius "
                    "ou ser uma sequencia (cx, cy, r)")
        if center_x is None or center_y is None or radius is None:
            raise ValueError("center_x, center_y e radius sao obrigatorios")
        values = tuple(float(value) for value in (
            center_x, center_y, radius))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("proposta contem valor nao finito")
        if values[2] <= 0:
            raise ValueError("radius precisa ser positivo")
        return values

    @staticmethod
    def _box_to_original(model, x0, y0, x1, y1):
        scale_x = model.frame_width / max(float(model.work_width), 1.0)
        scale_y = model.frame_height / max(float(model.work_height), 1.0)
        return (
            int(round(x0 * scale_x)),
            int(round(y0 * scale_y)),
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
        )


def evaluate_arena_support(
    frame,
    proposal=None,
    *,
    center_x=None,
    center_y=None,
    radius=None,
    guardian=None,
):
    """API funcional para quem nao precisa manter uma instancia."""
    guardian = guardian or MonocularArenaGuardian()
    return guardian.inspect(
        frame,
        proposal,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )


def annotate_arena_evidence(frame, evidence, alpha=0.20):
    """Desenha piso, soleira e faixa de apoio sem alterar o frame original."""
    if not isinstance(evidence, ArenaSupportEvidence):
        raise TypeError("evidence precisa ser ArenaSupportEvidence")
    annotated = frame.copy()
    model = evidence.model
    if (
        model.floor_mask is not None
        and model.floor_mask.shape[:2]
        == (model.work_height, model.work_width)
    ):
        floor = cv2.resize(
            model.floor_mask.astype(np.uint8),
            (model.frame_width, model.frame_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        tint = annotated.copy()
        tint[floor] = (
            0.35 * tint[floor]
            + 0.65 * np.array((40, 150, 40), dtype=np.float32)
        ).astype(np.uint8)
        annotated = cv2.addWeighted(
            tint, float(alpha), annotated, 1.0 - float(alpha), 0.0)

    points = evidence.boundary_points()
    if points:
        cv2.polylines(
            annotated,
            [np.asarray(points, dtype=np.int32)],
            False,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if evidence.support_box is not None:
        x0, y0, x1, y1 = evidence.support_box
        color = (0, 220, 0) if evidence.valid else (0, 0, 255)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        annotated,
        (
            f"arena {evidence.reason} "
            f"c={evidence.boundary_confidence:.2f} "
            f"piso={evidence.floor_support:.0%}"
        ),
        (8, max(18, int(round(0.045 * annotated.shape[0])))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, annotated.shape[1] / 1600.0),
        (0, 255, 255) if evidence.valid else (0, 80, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated
