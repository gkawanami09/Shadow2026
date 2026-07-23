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


class BallApproachController:
    """Transforma uma deteccao confirmada em comandos da lei steer existente."""

    WAIT_TARGET = "WAIT_TARGET"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    PROXIMITY_HOLD = "PROXIMITY_HOLD"
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
        self.ultrasonic_near_count = 0
        self.ultrasonic_hold_started = None
        self.progress = deque()
        self._terminal_detail = ""

    def update(
        self,
        detection,
        frame_shape,
        distance_mm=None,
        ultrasonic_polled=False,
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        height, width = frame_shape[:2]

        if self.state in (self.NEAR, self.FAULT):
            return MotionCommand(
                self.state, detail=self._terminal_detail, terminal=True)

        if detection is None or not detection.confirmed:
            self.visual_near_count = 0
            self.ultrasonic_near_count = 0
            self.ultrasonic_hold_started = None
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
            self.state = self.LOST
            return MotionCommand(
                self.state, detail="deteccao antiga; robo parado")

        if self.active_started is None:
            self.active_started = now
        elif now - self.active_started >= cfg.BALL_MAX_ACTIVE_S:
            return self._fault("timeout da aproximacao")

        self.last_seen = now
        error = detection.horizontal_error(width)

        visual_near = (
            detection.radius >= cfg.BALL_STOP_RADIUS_PX
            and detection.bottom_y >= height * cfg.BALL_STOP_BOTTOM_Y_RATIO
            and abs(error) <= cfg.BALL_STOP_CENTER_ERROR)
        self.visual_near_count = (
            self.visual_near_count + 1 if visual_near else 0)

        if ultrasonic_polled and distance_mm is not None:
            if 0 < distance_mm < cfg.BALL_ULTRASONIC_MIN_VALID_MM:
                return self._fault(
                    "eco abaixo da faixa confiavel do ultrassom; PARAR")
            if distance_mm <= cfg.BALL_ULTRASONIC_STOP_MM:
                if self.ultrasonic_near_count == 0:
                    self.ultrasonic_hold_started = now
                self.ultrasonic_near_count += 1
            else:
                self.ultrasonic_near_count = 0
                self.ultrasonic_hold_started = None

        if self.visual_near_count >= cfg.BALL_STOP_CONFIRM_FRAMES:
            reason = "proximidade visual confirmada"
            self.state = self.NEAR
            self._terminal_detail = reason
            return MotionCommand(
                self.state, detail=reason, terminal=True)

        if (
            self.ultrasonic_near_count
            >= cfg.BALL_ULTRASONIC_CONFIRM_READS
        ):
            if abs(error) <= cfg.BALL_STOP_CENTER_ERROR:
                reason = (
                    "barreira ultrassonica confirmada com alvo centralizado")
                self.state = self.NEAR
                self._terminal_detail = reason
                return MotionCommand(
                    self.state, detail=reason, terminal=True)
            return self._fault(
                "obstaculo proximo no ultrassom fora do eixo da bolinha")

        if self.ultrasonic_near_count > 0:
            hold_for = now - (self.ultrasonic_hold_started or now)
            if hold_for >= cfg.BALL_ULTRASONIC_HOLD_TIMEOUT_S:
                return self._fault(
                    "nao foi possivel confirmar a leitura ultrassonica proxima")
            self.state = self.PROXIMITY_HOLD
            self.progress.clear()
            detail = (
                f"primeiro eco proximo ({distance_mm} mm); PARAR para confirmar"
                if distance_mm is not None
                else "eco proximo pendente; PARAR para confirmar")
            return MotionCommand(self.state, detail=detail)

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
        self._record_progress(now, detection.radius)
        if self._progress_stalled(now):
            return self._fault("sem progresso visual durante a aproximacao")

        if abs(error) <= cfg.BALL_CENTER_DEADBAND:
            angle = 0
        else:
            angle = int(round(np.clip(
                error / cfg.BALL_ALIGN_THRESHOLD * cfg.BALL_STEER_MAX_ANGLE,
                -cfg.BALL_STEER_MAX_ANGLE,
                cfg.BALL_STEER_MAX_ANGLE)))

        slow_span = max(
            cfg.BALL_STOP_RADIUS_PX - cfg.BALL_SLOW_RADIUS_PX, 1)
        near_fraction = float(np.clip(
            (detection.radius - cfg.BALL_SLOW_RADIUS_PX) / slow_span,
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

    def _record_progress(self, now, radius):
        self.progress.append((now, float(radius)))
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
        return (
            max(radius for _, radius in self.progress)
            - self.progress[0][1]
            < cfg.BALL_PROGRESS_MIN_RADIUS_PX)

    def _fault(self, detail):
        self.state = self.FAULT
        self._terminal_detail = detail
        return MotionCommand(
            self.state, detail=detail, terminal=True)
