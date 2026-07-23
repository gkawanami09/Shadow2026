"""Maquina de estados pura para alinhar e aproximar da esfera detectada."""

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

import rescue_config as cfg


@dataclass(frozen=True)
class MotionCommand:
    state: str
    angle: int = 190
    speed: float = 0.0
    detail: str = ""
    terminal: bool = False
    pickup_in_range: bool = False
    pickup_confirmations: int = 0


class BallApproachController:
    """Transforma uma deteccao confirmada em comandos da lei steer existente."""

    WAIT_TARGET = "WAIT_TARGET"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    NEAR_CONFIRM = "NEAR_CONFIRM"
    LOST = "LOST"
    NEAR = "NEAR"
    FAULT = "FAULT"

    def __init__(self, start_time=None):
        now = time.monotonic() if start_time is None else float(start_time)
        self.state = self.WAIT_TARGET
        self.wait_started = now
        self.active_started = None
        self.last_seen = None
        self.visual_near_count = 0
        self.progress = deque()
        self._pixel_scale = 1.0
        self._terminal_detail = ""

    def update(
        self,
        detection,
        frame_shape,
        candidate_count=0,
        candidate_circles=(),
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        height, width = frame_shape[:2]
        self._pixel_scale = cfg.ball_pixel_scale(width, height)

        if self.state == self.NEAR:
            return MotionCommand(
                self.state,
                detail=self._terminal_detail,
                terminal=True,
                pickup_in_range=True,
                pickup_confirmations=cfg.BALL_STOP_CONFIRM_FRAMES,
            )
        if self.state == self.FAULT:
            return MotionCommand(
                self.state, detail=self._terminal_detail, terminal=True)

        if detection is None or not detection.confirmed:
            self.visual_near_count = 0
            self.progress.clear()

            if self.active_started is None:
                if now - self.wait_started >= cfg.BALL_MAX_WAIT_S:
                    return self._fault("tempo maximo esperando uma bolinha")
                self.state = self.WAIT_TARGET
                return MotionCommand(
                    self.state, detail="parado; aguardando confirmacao temporal")

            self.state = self.LOST
            lost_for = now - (self.last_seen or now)
            if lost_for >= cfg.BALL_REACQUIRE_TIMEOUT_S:
                self.active_started = None
                self.wait_started = now
                self.state = self.WAIT_TARGET
                return MotionCommand(
                    self.state,
                    detail="alvo perdido; aproximacao cancelada e robo parado")
            return MotionCommand(
                self.state,
                detail=f"alvo ausente ha {lost_for:.2f}s; robo parado")

        if now - detection.timestamp > cfg.BALL_FRAME_STALE_S:
            self.visual_near_count = 0
            self.progress.clear()
            self.state = self.LOST
            return MotionCommand(
                self.state, detail="deteccao antiga; robo parado")

        if self.active_started is None:
            self.active_started = now
        elif now - self.active_started >= cfg.BALL_MAX_ACTIVE_S:
            return self._fault("timeout da aproximacao")

        self.last_seen = now
        error = detection.horizontal_error(width)

        normal_near = (
            detection.radius
            >= cfg.BALL_STOP_RADIUS_PX * self._pixel_scale
            and detection.bottom_y >= height * cfg.BALL_STOP_BOTTOM_Y_RATIO
            and abs(error) <= cfg.BALL_STOP_CENTER_ERROR)
        clipped_silver_near = self._clipped_silver_near(
            detection,
            height,
            width,
            error,
            candidate_count,
            candidate_circles,
        )
        visual_near = normal_near or clipped_silver_near
        self.visual_near_count = (
            self.visual_near_count + 1 if visual_near else 0)

        if self.visual_near_count >= cfg.BALL_STOP_CONFIRM_FRAMES:
            reason = (
                "faixa de coleta confirmada; esfera muito proxima"
                if clipped_silver_near
                else "proximidade visual confirmada")
            self.state = self.NEAR
            self._terminal_detail = reason
            return MotionCommand(
                self.state,
                detail=reason,
                terminal=True,
                pickup_in_range=True,
                pickup_confirmations=self.visual_near_count,
            )

        if visual_near:
            # O reflexo interno pode ficar fora do centro mesmo com a esfera
            # externa alinhada. Parar durante a confirmacao evita um pivo curto
            # em cima da bolinha e suspende o watchdog de progresso.
            self.state = self.NEAR_CONFIRM
            self.progress.clear()
            return MotionCommand(
                self.state,
                detail=(
                    "esfera na faixa de coleta; confirmando "
                    f"{self.visual_near_count}/"
                    f"{cfg.BALL_STOP_CONFIRM_FRAMES}"),
                pickup_in_range=True,
                pickup_confirmations=self.visual_near_count,
            )

        if abs(error) > cfg.BALL_ALIGN_THRESHOLD:
            self.state = self.ALIGN
            self.progress.clear()
            angle = cfg.BALL_ALIGN_ANGLE if error > 0 else -cfg.BALL_ALIGN_ANGLE
            return MotionCommand(
                self.state,
                angle=angle,
                speed=cfg.BALL_ALIGN_SPEED,
                detail=f"centralizando; erro horizontal={error:+.3f}")

        self.state = self.APPROACH
        self._record_progress(now, detection.radius, detection.bottom_y)
        if self._progress_stalled(now):
            return self._fault("sem progresso visual durante a aproximacao")

        if abs(error) <= cfg.BALL_CENTER_DEADBAND:
            angle = 0
        else:
            angle = int(round(np.clip(
                error / cfg.BALL_ALIGN_THRESHOLD * cfg.BALL_STEER_MAX_ANGLE,
                -cfg.BALL_STEER_MAX_ANGLE,
                cfg.BALL_STEER_MAX_ANGLE)))

        slow_radius = cfg.BALL_SLOW_RADIUS_PX * self._pixel_scale
        stop_radius = cfg.BALL_STOP_RADIUS_PX * self._pixel_scale
        slow_span = max(stop_radius - slow_radius, 1)
        near_fraction = float(np.clip(
            (detection.radius - slow_radius) / slow_span,
            0.0, 1.0))
        speed = (
            cfg.BALL_APPROACH_SPEED_FAR * (1.0 - near_fraction)
            + cfg.BALL_APPROACH_SPEED_NEAR * near_fraction)
        return MotionCommand(
            self.state,
            angle=angle,
            speed=float(speed),
            detail=(
                f"aproximando; erro={error:+.3f}, "
                f"raio={detection.radius:.1f}px"))

    def _clipped_silver_near(
        self,
        detection,
        frame_height,
        frame_width,
        horizontal_error,
        candidate_count,
        candidate_circles,
    ):
        """Reconhece a esfera prateada proxima mesmo seguindo reflexo interno."""
        associated_candidates = []
        for circle in candidate_circles or ():
            if len(circle) < 5:
                continue
            center_x, center_y, radius = map(float, circle[:3])
            kind = str(circle[3])
            confidence = float(circle[4])
            if (
                radius <= 0
                or kind != "silver"
                or confidence < cfg.BALL_CLOSE_OUTER_MIN_CONFIDENCE
            ):
                continue
            center_distance = float(np.hypot(
                detection.center_x - center_x,
                detection.center_y - center_y,
            ))
            if (
                center_distance
                <= radius * cfg.BALL_CLOSE_ASSOCIATION_RADIUS_RATIO
            ):
                associated_candidates.append(
                    (radius, center_x, center_y))
        associated_candidates.sort(reverse=True)
        if len(associated_candidates) < 2:
            return False
        large_candidates = [
            candidate for candidate in associated_candidates
            if candidate[0]
            >= cfg.BALL_CLOSE_SECOND_RADIUS_PX * self._pixel_scale
        ]
        if len(large_candidates) < 2:
            return False
        outer_center_x = float(np.median(
            [candidate[1] for candidate in large_candidates]))
        outer_half_width = max(float(frame_width) / 2.0, 1.0)
        outer_center_error = (
            outer_center_x - outer_half_width) / outer_half_width
        return (
            detection.kind == "silver"
            and detection.confidence >= cfg.BALL_CLOSE_MIN_CONFIDENCE
            and detection.hits >= cfg.BALL_CLOSE_MIN_HITS
            and detection.radius
            >= cfg.BALL_CLOSE_MIN_RADIUS_PX * self._pixel_scale
            and detection.center_y
            >= frame_height * cfg.BALL_CLOSE_CENTER_Y_RATIO
            and detection.bottom_y
            >= frame_height * cfg.BALL_CLOSE_BOTTOM_Y_RATIO
            and abs(horizontal_error) <= cfg.BALL_CLOSE_CENTER_ERROR
            and int(candidate_count) >= cfg.BALL_CLOSE_MIN_CANDIDATES
            and abs(outer_center_error)
            <= cfg.BALL_CLOSE_OUTER_CENTER_ERROR
            and associated_candidates[0][0]
            >= cfg.BALL_CLOSE_LARGEST_RADIUS_PX * self._pixel_scale
            and associated_candidates[1][0]
            >= cfg.BALL_CLOSE_SECOND_RADIUS_PX * self._pixel_scale
        )

    def _record_progress(self, now, radius, bottom_y):
        self.progress.append((
            now,
            float(radius),
            float(bottom_y),
        ))
        while (
            self.progress
            and now - self.progress[0][0] > cfg.BALL_PROGRESS_WINDOW_S
        ):
            self.progress.popleft()

    def _progress_stalled(self, now):
        if len(self.progress) < 2:
            return False
        elapsed = now - self.progress[0][0]
        if elapsed < cfg.BALL_PROGRESS_WINDOW_S * 0.90:
            return False
        radius_progress = (
            max(radius for _, radius, _ in self.progress)
            - self.progress[0][1])
        bottom_progress = (
            max(bottom_y for _, _, bottom_y in self.progress)
            - self.progress[0][2])
        return (
            radius_progress
            < cfg.BALL_PROGRESS_MIN_RADIUS_PX * self._pixel_scale
            and bottom_progress
            < cfg.BALL_PROGRESS_MIN_BOTTOM_Y_PX * self._pixel_scale
        )

    def _fault(self, detail):
        self.state = self.FAULT
        self._terminal_detail = detail
        return MotionCommand(
            self.state, detail=detail, terminal=True)
