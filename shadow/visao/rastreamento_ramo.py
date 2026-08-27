"""Rastreamento fail-closed do ramo fisico travado para o retorno de 180.

O token nasce no grafo topologico. Pontos visuais sao semeados somente no
ramo INCOMING e propagados quadro a quadro por fluxo optico com verificacao
ida-e-volta. Se os pontos se perdem, nenhuma outra faixa herda o token.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class LockedBranchResult:
    token: int = 0
    valid: bool = False
    bottom_x: float = -1.0
    bottom_y: float = -1.0
    tracked_points: int = 0


class LockedBranchTracker:
    """Mantem identidade visual sem jamais reacoplar por proximidade apenas."""

    def __init__(self, *, min_points: int = 4, max_fb_error: float = 1.8):
        self.min_points = max(int(min_points), 2)
        self.max_fb_error = max(float(max_fb_error), .1)
        self.reset()

    def reset(self):
        self.decision_id = 0
        self.branch_token = 0
        self._previous_gray = None
        self._points = None
        self._last_sequence = -1
        self._predicted_junction = None
        self._predicted_target = None
        self._line_width_px = 0.0
        self._pose_valid = False
        self._last_motion = None
        self._predicted_motion_frames = 0
        self._result = LockedBranchResult()

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("frame do rastreador precisa ser cinza ou BGR")
        return np.ascontiguousarray(gray, dtype=np.uint8)

    @staticmethod
    def _binary(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        if mask is None or mask.shape[:2] != shape:
            raise ValueError("mascara preta incompatível com o frame")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    @staticmethod
    def _bottom_point(points: np.ndarray) -> Point:
        flat = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if not len(flat):
            return -1.0, -1.0
        limit = float(np.percentile(flat[:, 1], 60.0))
        lower = flat[flat[:, 1] >= limit]
        if not len(lower):
            lower = flat
        return float(np.median(lower[:, 0])), float(np.median(lower[:, 1]))

    def arm(
        self,
        frame: np.ndarray,
        black_mask: np.ndarray,
        *,
        sequence: int,
        decision_id: int,
        branch_token: int,
        junction: Sequence[float],
        target: Sequence[float],
        line_width_px: float,
    ) -> LockedBranchResult:
        """Semeia pontos apenas no corredor do INCOMING identificado."""

        self.reset()
        decision_id = int(decision_id)
        branch_token = int(branch_token)
        if decision_id <= 0 or branch_token <= 0:
            return self._result

        gray = self._gray(frame)
        black = self._binary(black_mask, gray.shape)
        junction = np.asarray(junction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        direction = target - junction
        length = float(np.linalg.norm(direction))
        if (
            junction.shape != (2,)
            or target.shape != (2,)
            or not np.all(np.isfinite(junction))
            or not np.all(np.isfinite(target))
            or length < 4.0
        ):
            return self._result
        direction /= length

        height, width = gray.shape
        extension = max(length * 2.4, height * .34)
        start = junction + direction * max(float(line_width_px) * .45, 3.0)
        end = target + direction * extension
        start_i = tuple(np.rint(start).astype(int))
        end_i = tuple(np.rint(end).astype(int))
        thickness = max(9, int(round(max(float(line_width_px), 6.0) * 1.35)))
        corridor = np.zeros_like(gray)
        cv2.line(corridor, start_i, end_i, 255, thickness, cv2.LINE_AA)
        black_near = cv2.dilate(
            black,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        seed_mask = cv2.bitwise_and(corridor, black_near)

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=48,
            qualityLevel=.008,
            minDistance=3.0,
            mask=seed_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if points is None or len(points) < self.min_points:
            # Linhas perfeitamente uniformes podem nao produzir cantos. Os
            # pixels da borda fornecem sementes adicionais; a checagem
            # ida-e-volta do proximo frame decide se sao rastreaveis.
            edges = cv2.bitwise_and(cv2.Canny(gray, 35, 110), seed_mask)
            ys, xs = np.nonzero(edges)
            if len(xs):
                step = max(1, len(xs) // 48)
                sampled = np.column_stack((xs[::step], ys[::step]))[:48]
                points = sampled.astype(np.float32).reshape(-1, 1, 2)

        count = 0 if points is None else int(len(points))
        self.decision_id = decision_id
        self.branch_token = branch_token
        self._previous_gray = gray
        self._points = points
        self._last_sequence = int(sequence)
        self._predicted_junction = junction.astype(np.float64)
        self._predicted_target = target.astype(np.float64)
        self._line_width_px = max(float(line_width_px), 1.0)
        self._pose_valid = True
        self._last_motion = np.eye(3, dtype=np.float64)
        self._predicted_motion_frames = 0
        valid = count >= self.min_points
        bottom = self._bottom_point(points) if valid else (-1.0, -1.0)
        self._result = LockedBranchResult(
            token=branch_token,
            valid=valid,
            bottom_x=bottom[0],
            bottom_y=bottom[1],
            tracked_points=count,
        )
        return self._result

    def _estimate_frame_motion(self, previous, current):
        """Estima o movimento do plano do chao entre dois frames retificados."""

        previous_small = cv2.resize(
            previous, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        current_small = cv2.resize(
            current, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        features = cv2.goodFeaturesToTrack(
            previous_small,
            maxCorners=80,
            qualityLevel=.006,
            minDistance=5.0,
            blockSize=5,
        )
        if features is None or len(features) < 6:
            return None
        moved, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            previous_small,
            current_small,
            features,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                .02,
            ),
        )
        if moved is None or status_fwd is None:
            return None
        backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
            current_small,
            previous_small,
            moved,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                .02,
            ),
        )
        if backward is None or status_back is None:
            return None
        source = features.reshape(-1, 2)
        destination = moved.reshape(-1, 2)
        fb_error = np.linalg.norm(
            source - backward.reshape(-1, 2), axis=1)
        good = (
            status_fwd.reshape(-1).astype(bool)
            & status_back.reshape(-1).astype(bool)
            & (fb_error <= max(self.max_fb_error, 2.0))
        )
        source = source[good]
        destination = destination[good]
        if len(source) < 6:
            return None
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.2,
            maxIters=500,
            confidence=.995,
            refineIters=10,
        )
        if affine is None or inliers is None:
            return None
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 6 or inlier_count / len(source) < .50:
            return None
        inlier_mask = inliers.reshape(-1).astype(bool)
        source_inliers = source[inlier_mask]
        destination_inliers = destination[inlier_mask]
        height, width = current_small.shape
        cells = {
            (
                min(int(point[0] * 3 / max(width, 1)), 2),
                min(int(point[1] * 3 / max(height, 1)), 2),
            )
            for point in source_inliers
        }
        hull_area = (
            cv2.contourArea(cv2.convexHull(
                source_inliers.astype(np.float32)))
            if len(source_inliers) >= 3
            else 0.0
        )
        if len(cells) < 3 or hull_area < width * height * .008:
            return None
        linear = np.asarray(affine[:, :2], dtype=np.float64)
        scale = math.sqrt(abs(float(np.linalg.det(linear))))
        translation = float(np.linalg.norm(affine[:, 2]))
        rotation_deg = abs(math.degrees(math.atan2(
            float(affine[1, 0]), float(affine[0, 0]))))
        predicted = cv2.transform(
            source_inliers.reshape(-1, 1, 2), affine).reshape(-1, 2)
        residual = np.linalg.norm(predicted - destination_inliers, axis=1)
        if (
            not np.all(np.isfinite(affine))
            or not .90 <= scale <= 1.10
            or translation > 45.0
            or rotation_deg > 18.0
            or float(np.percentile(residual, 95)) > 2.5
        ):
            return None
        motion = np.eye(3, dtype=np.float64)
        motion[:2, :] = affine
        scale_to_small = np.diag((.5, .5, 1.0))
        return np.linalg.inv(scale_to_small) @ motion @ scale_to_small

    def _move_prediction(self, motion):
        if (
            not self._pose_valid
            or self._predicted_junction is None
            or self._predicted_target is None
        ):
            return
        points = np.asarray((
            self._predicted_junction,
            self._predicted_target,
        ), dtype=np.float64).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(points, motion).reshape(-1, 2)
        if not np.all(np.isfinite(moved)):
            self._pose_valid = False
            return
        self._predicted_junction = moved[0]
        self._predicted_target = moved[1]

    @staticmethod
    def _motion_step_plausible(motion):
        try:
            matrix = np.asarray(motion, dtype=np.float64).reshape(3, 3)
            linear = matrix[:2, :2]
            scale = math.sqrt(abs(float(np.linalg.det(linear))))
            translation = float(np.linalg.norm(matrix[:2, 2]))
            rotation = abs(math.degrees(math.atan2(
                float(matrix[1, 0]), float(matrix[0, 0]))))
        except (TypeError, ValueError):
            return False
        return bool(
            np.all(np.isfinite(matrix))
            and .90 <= scale <= 1.10
            and translation <= 45.0
            and rotation <= 18.0
        )

    def _seed_from_prediction(self, gray, black, *, sequence):
        if (
            not self._pose_valid
            or self._predicted_junction is None
            or self._predicted_target is None
        ):
            return None
        candidate = LockedBranchTracker(
            min_points=self.min_points,
            max_fb_error=self.max_fb_error,
        )
        result = candidate.arm(
            gray,
            black,
            sequence=sequence,
            decision_id=self.decision_id,
            branch_token=self.branch_token,
            junction=self._predicted_junction,
            target=self._predicted_target,
            line_width_px=self._line_width_px,
        )
        return candidate if result.valid else None

    def update(
        self,
        frame: np.ndarray,
        black_mask: np.ndarray,
        *,
        sequence: int,
        decision_id: int,
        branch_token: int,
    ) -> LockedBranchResult:
        """Propaga o mesmo token; perda nunca permite semear em outra linha."""

        if (
            int(decision_id) != self.decision_id
            or int(branch_token) != self.branch_token
            or self.decision_id <= 0
            or self.branch_token <= 0
        ):
            self.reset()
            return self._result
        sequence = int(sequence)
        if sequence <= self._last_sequence:
            return self._result
        gray = self._gray(frame)
        black = self._binary(black_mask, gray.shape)
        previous = self._previous_gray
        measured_motion = self._estimate_frame_motion(previous, gray)
        if not self._motion_step_plausible(measured_motion):
            measured_motion = None
        if self._pose_valid:
            motion = measured_motion
            if motion is not None:
                self._last_motion = motion
                self._predicted_motion_frames = 0
            else:
                # Uma lacuna quebra a cadeia de pose. Pontos do proprio ramo
                # ainda podem continuar por LK, mas nunca havera reacoplamento
                # espacial posterior sem nova geometria topologica real.
                self._pose_valid = False
                motion = None
            if motion is not None:
                self._move_prediction(motion)

        good = np.empty((0, 1, 2), dtype=np.float32)
        if self._points is not None and len(self._points) >= self.min_points:
            next_points, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                gray,
                self._points,
                None,
                winSize=(31, 31),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    .02,
                ),
            )
            if next_points is not None and status_fwd is not None:
                back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    previous,
                    next_points,
                    None,
                    winSize=(31, 31),
                    maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        20,
                        .02,
                    ),
                )
                if back_points is not None and status_back is not None:
                    original = self._points.reshape(-1, 2)
                    current = next_points.reshape(-1, 2)
                    backward = back_points.reshape(-1, 2)
                    fb_error = np.linalg.norm(original - backward, axis=1)
                    inside = (
                        (current[:, 0] >= 0)
                        & (current[:, 0] < gray.shape[1])
                        & (current[:, 1] >= 0)
                        & (current[:, 1] < gray.shape[0])
                    )
                    candidate = (
                        status_fwd.reshape(-1).astype(bool)
                        & status_back.reshape(-1).astype(bool)
                        & (fb_error <= self.max_fb_error)
                        & inside
                    )
                    indices = np.flatnonzero(candidate)
                    if len(indices):
                        black_near = cv2.dilate(
                            black,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (9, 9)),
                        )
                        rounded = np.rint(current[indices]).astype(int)
                        supported = black_near[
                            rounded[:, 1], rounded[:, 0]
                        ] > 0
                        indices = indices[supported]
                    good = current[indices].astype(
                        np.float32).reshape(-1, 1, 2)

        if len(good) < self.min_points and measured_motion is not None:
            candidate = self._seed_from_prediction(
                gray, black, sequence=sequence)
            if candidate is not None:
                good = candidate._points

        self._previous_gray = gray
        self._points = good
        self._last_sequence = sequence
        count = int(len(good))
        valid = count >= self.min_points
        bottom = self._bottom_point(good) if valid else (-1.0, -1.0)
        self._result = LockedBranchResult(
            token=self.branch_token,
            valid=valid,
            bottom_x=bottom[0],
            bottom_y=bottom[1],
            tracked_points=count,
        )
        return self._result

    def refresh_from_verified_geometry(
        self,
        frame: np.ndarray,
        black_mask: np.ndarray,
        *,
        sequence: int,
        decision_id: int,
        branch_token: int,
        junction: Sequence[float],
        target: Sequence[float],
        line_width_px: float,
    ) -> LockedBranchResult:
        """Renova sementes somente com o mesmo ramo topologico real.

        O chamador deve usar este metodo apenas quando a observacao atomica da
        mesma sequencia ainda declara a juncao visivel (nao propagada). Um
        candidato ruim nao apaga pontos antigos; um candidato bom substitui o
        conjunto por pontos atuais do mesmo token, evitando que a aproximacao
        os carregue para fora do FOV antes do giro.
        """

        decision_id = int(decision_id)
        branch_token = int(branch_token)
        if (
            decision_id != self.decision_id
            or branch_token != self.branch_token
        ):
            return self.arm(
                frame,
                black_mask,
                sequence=sequence,
                decision_id=decision_id,
                branch_token=branch_token,
                junction=junction,
                target=target,
                line_width_px=line_width_px,
            )
        candidate = LockedBranchTracker(
            min_points=self.min_points,
            max_fb_error=self.max_fb_error,
        )
        result = candidate.arm(
            frame,
            black_mask,
            sequence=sequence,
            decision_id=decision_id,
            branch_token=branch_token,
            junction=junction,
            target=target,
            line_width_px=line_width_px,
        )
        self._predicted_junction = np.asarray(
            junction, dtype=np.float64).reshape(2)
        self._predicted_target = np.asarray(
            target, dtype=np.float64).reshape(2)
        self._line_width_px = max(float(line_width_px), 1.0)
        self._pose_valid = True
        self._predicted_motion_frames = 0
        if result.valid:
            self.decision_id = candidate.decision_id
            self.branch_token = candidate.branch_token
            self._previous_gray = candidate._previous_gray
            self._points = candidate._points
            self._last_sequence = candidate._last_sequence
            self._result = candidate._result
        return self._result


__all__ = ["LockedBranchResult", "LockedBranchTracker"]
