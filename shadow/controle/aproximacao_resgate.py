"""Controla o alinhamento e a aproximação da esfera detectada."""

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

import config_resgate as cfg


@dataclass(frozen=True)
class MotionCommand:
    state: str
    angle: int = 190
    speed: float = 0.0
    detail: str = ""
    terminal: bool = False
    pickup_in_range: bool = False
    pickup_confirmations: int = 0
    target_kind: object = None


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
        self.approach_history = deque()
        self._history_kind = None
        self._crescent_token_until = None
        self._crescent_token_center = None
        self._crescent_token_kind = None
        self._last_history_timestamp = None
        self._last_near_timestamp = None
        self._near_source = None
        self._near_first_at = None
        self._near_hold_until = None
        self._near_misses = 0
        self._near_kind = None
        self._pixel_scale = 1.0
        self._terminal_detail = ""

    def update(
        self,
        detection,
        frame_shape,
        crescent_evidence=None,
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
                target_kind=self._near_kind,
            )
        if self.state == self.FAULT:
            return MotionCommand(
                self.state, detail=self._terminal_detail, terminal=True)

        detection_confirmed = bool(
            detection is not None and detection.confirmed)
        if (
            detection_confirmed
            and now - detection.timestamp > cfg.BALL_FRAME_STALE_S
        ):
            self._reset_near_confirmation()
            self.progress.clear()
            self.state = self.LOST
            return MotionCommand(
                self.state, detail="deteccao antiga; robo parado")

        if (
            self.active_started is not None
            and now - self.active_started >= cfg.BALL_MAX_ACTIVE_S
        ):
            return self._fault("timeout da aproximacao")

        if detection_confirmed:
            if self.active_started is None:
                self.active_started = now
            self.last_seen = now
            # O estado anterior descreve o comando que estava nas rodas entre
            # este frame e o anterior. So esse trecho de avanco pode autorizar
            # a transicao do circulo inteiro para a meia-lua cortada.
            if self.state == self.APPROACH:
                self._record_approach_history(
                    detection,
                    height,
                    width,
                    now,
                )

        self._refresh_crescent_token(now)

        locked_circle_near = self._locked_circle_near(
            detection if detection_confirmed else None,
            height,
            width,
            now,
        )
        crescent_near = self._close_crescent_near(
            crescent_evidence,
            detection if detection_confirmed else None,
            height,
            width,
            now,
        )
        near_source = None
        near_label = None
        near_timestamp = None
        near_kind = None
        near_required = cfg.BALL_STOP_CONFIRM_FRAMES
        if locked_circle_near:
            near_source = "contato inferior"
            near_label = "circulo no ponto inferior"
            near_timestamp = float(detection.timestamp)
            near_kind = detection.kind
            near_required = cfg.BALL_LOCKED_CIRCLE_CONFIRM_FRAMES
        elif (
            crescent_near
            and self._near_source == "contato inferior"
            and self.visual_near_count > 0
            and self._near_kind == self._crescent_token_kind
            and (
                not detection_confirmed
                or detection.kind == self._near_kind
            )
            and self._near_hold_until is not None
            and now <= self._near_hold_until
            and self._near_misses
            <= cfg.BALL_NEAR_CONFIRM_MAX_MISSES
        ):
            # A meia-lua nao inicia coleta sozinha. Ela apenas confirma o
            # contato que o circulo travado acabou de fazer com a borda
            # inferior, caso o perimetro seja cortado no frame seguinte.
            near_source = "contato inferior"
            near_label = "meia-lua apos contato"
            near_timestamp = float(crescent_evidence.timestamp)
            near_kind = self._near_kind
            near_required = cfg.BALL_LOCKED_CIRCLE_CONFIRM_FRAMES

        # Uma meia-lua nao possui cor propria. Se o mesmo frame ainda traz
        # uma deteccao confirmada de outra cor, ela nao pode confirmar o latch
        # anterior nem sobreviver pela tolerancia de um miss.
        if (
            near_source is None
            and detection_confirmed
            and self._near_kind is not None
            and detection.kind != self._near_kind
        ):
            self._reset_near_confirmation()

        # A janela curta existe para uma unica falha visual, nao para unir
        # confirmacoes de tracks diferentes depois de uma reacquisicao.
        if (
            self.visual_near_count > 0
            and (
                self._near_hold_until is None
                or now > self._near_hold_until
            )
        ):
            self._reset_near_confirmation()

        if near_source is not None:
            if self.active_started is None:
                self.active_started = now
            self.last_seen = now
            if (
                self._near_source != near_source
                or (
                    self._near_kind is not None
                    and near_kind != self._near_kind
                )
            ):
                self._reset_near_confirmation()
            if (
                self._near_first_at is not None
                and now - self._near_first_at
                > cfg.BALL_NEAR_CONFIRM_WINDOW_S
            ):
                self._reset_near_confirmation()
            self._near_source = near_source
            self._near_kind = near_kind
            if (
                self._last_near_timestamp is None
                or near_timestamp
                > self._last_near_timestamp + 1e-9
            ):
                if self._near_first_at is None:
                    self._near_first_at = now
                self.visual_near_count += 1
                self._last_near_timestamp = near_timestamp
                self._near_misses = 0
                self._near_hold_until = (
                    now + cfg.BALL_NEAR_CONFIRM_GRACE_S)
            if self.visual_near_count >= near_required:
                reason = (
                    "contato inferior confirmado; "
                    "esfera na posicao de coleta")
                self.state = self.NEAR
                self._terminal_detail = reason
                return MotionCommand(
                    self.state,
                    detail=reason,
                    terminal=True,
                    pickup_in_range=True,
                    pickup_confirmations=cfg.BALL_STOP_CONFIRM_FRAMES,
                    target_kind=self._near_kind,
                )

            self.state = self.NEAR_CONFIRM
            self.progress.clear()
            return MotionCommand(
                self.state,
                detail=(
                    f"{near_label} proximo; confirmando "
                    f"{self.visual_near_count}/"
                    f"{near_required}"),
                pickup_in_range=True,
                pickup_confirmations=self.visual_near_count,
                target_kind=self._near_kind,
            )

        if self.visual_near_count > 0:
            self._near_misses += 1
        if (
            self.visual_near_count > 0
            and self._near_misses
            <= cfg.BALL_NEAR_CONFIRM_MAX_MISSES
            and self._near_hold_until is not None
            and now <= self._near_hold_until
            and self._crescent_token_until is not None
            and now <= self._crescent_token_until
        ):
            self.state = self.NEAR_CONFIRM
            self.progress.clear()
            required = (
                cfg.BALL_LOCKED_CIRCLE_CONFIRM_FRAMES
                if self._near_source == "contato inferior"
                else cfg.BALL_STOP_CONFIRM_FRAMES
            )
            return MotionCommand(
                self.state,
                detail=(
                    f"{self._near_source}: mantendo trava visual "
                    f"{self.visual_near_count}/{required}"),
                pickup_in_range=True,
                pickup_confirmations=self.visual_near_count,
                target_kind=self._near_kind,
            )

        self._reset_near_confirmation()
        if not detection_confirmed:
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
                self._clear_approach_history()
                self.state = self.WAIT_TARGET
                return MotionCommand(
                    self.state,
                    detail="alvo perdido; aproximacao cancelada e robo parado")
            return MotionCommand(
                self.state,
                detail=f"alvo ausente ha {lost_for:.2f}s; robo parado")

        error = detection.horizontal_error(width)

        align_threshold = (
            cfg.BALL_ALIGN_EXIT_THRESHOLD
            if self.state == self.ALIGN
            else cfg.BALL_ALIGN_THRESHOLD
        )
        if abs(error) > align_threshold:
            self.state = self.ALIGN
            self.progress.clear()
            severity = float(np.clip(
                (
                    abs(error) - cfg.BALL_ALIGN_EXIT_THRESHOLD
                ) / max(
                    1.0 - cfg.BALL_ALIGN_EXIT_THRESHOLD,
                    1e-6,
                ),
                0.0,
                1.0,
            ))
            angle_magnitude = int(round(
                cfg.BALL_ALIGN_ARC_MIN_ANGLE
                + severity
                * (
                    cfg.BALL_ALIGN_ARC_MAX_ANGLE
                    - cfg.BALL_ALIGN_ARC_MIN_ANGLE
                )
            ))
            angle = (
                angle_magnitude if error > 0
                else -angle_magnitude
            )
            speed = (
                cfg.BALL_ALIGN_SPEED_MIN
                + severity
                * (
                    cfg.BALL_ALIGN_SPEED_MAX
                    - cfg.BALL_ALIGN_SPEED_MIN
                )
            )
            return MotionCommand(
                self.state,
                angle=angle,
                speed=float(speed),
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

    def _reset_near_confirmation(self):
        self.visual_near_count = 0
        self._last_near_timestamp = None
        self._near_source = None
        self._near_first_at = None
        self._near_hold_until = None
        self._near_misses = 0
        self._near_kind = None

    def _locked_circle_near(
        self,
        detection,
        frame_height,
        frame_width,
        now,
    ):
        """Circulo rastreado cobrindo o ponto fisico de coleta."""
        if (
            detection is None
            or not detection.confirmed
            or not bool(getattr(detection, "track_locked", False))
        ):
            return False
        age = now - float(detection.timestamp)
        if not (-0.05 <= age <= cfg.BALL_FRAME_STALE_S):
            return False
        if (
            self._crescent_token_until is None
            or now > self._crescent_token_until
            or self._crescent_token_center is None
            or self._crescent_token_kind != detection.kind
        ):
            return False

        center_ratio = (
            float(detection.center_x) / max(frame_width, 1))
        if (
            abs(detection.horizontal_error(frame_width))
            > cfg.BALL_LOCKED_CIRCLE_MAX_CENTER_ERROR
            or abs(
                center_ratio - self._crescent_token_center
            ) > cfg.BALL_CRESCENT_ASSOCIATION_X_RATIO
        ):
            return False
        radius = float(detection.radius)
        if (
            radius / max(frame_height, 1)
            < cfg.BALL_LOCKED_CIRCLE_MIN_RADIUS_RATIO
        ):
            return False

        point_x = (
            cfg.BALL_LOCKED_CIRCLE_POINT_X_RATIO * frame_width)
        point_y = (
            cfg.BALL_LOCKED_CIRCLE_POINT_Y_RATIO * frame_height)
        point_distance = float(np.hypot(
            float(detection.center_x) - point_x,
            float(detection.center_y) - point_y,
        ))
        margin = (
            cfg.BALL_LOCKED_CIRCLE_POINT_SLACK_RATIO
            * frame_height
        )
        return point_distance <= radius + margin

    def _record_approach_history(
        self,
        detection,
        frame_height,
        frame_width,
        now,
    ):
        kind = detection.kind
        if kind not in ("silver", "black"):
            return
        if self._history_kind is not None and kind != self._history_kind:
            self._clear_approach_history()
        self._history_kind = kind
        timestamp = float(detection.timestamp)
        if (
            self._last_history_timestamp is not None
            and timestamp <= self._last_history_timestamp + 1e-9
        ):
            return
        center_error = detection.horizontal_error(frame_width)
        if abs(center_error) > cfg.BALL_CRESCENT_ARM_MAX_CENTER_ERROR:
            return
        self._last_history_timestamp = timestamp
        self.approach_history.append((
            timestamp,
            float(detection.center_x) / max(frame_width, 1),
            float(detection.radius) / max(frame_height, 1),
            float(detection.bottom_y) / max(frame_height, 1),
        ))
        self._prune_approach_history(now)

    def _prune_approach_history(self, now):
        while (
            self.approach_history
            and now - self.approach_history[0][0]
            > cfg.BALL_CRESCENT_HISTORY_S
        ):
            self.approach_history.popleft()
        if not self.approach_history:
            self._history_kind = None
            self._last_history_timestamp = None

    def _refresh_crescent_token(self, now):
        self._prune_approach_history(now)
        samples = list(self.approach_history)
        if len(samples) < cfg.BALL_CRESCENT_HISTORY_MIN_SAMPLES:
            return
        span = samples[-1][0] - samples[0][0]
        if span < max(
            cfg.BALL_CRESCENT_HISTORY_MIN_SPAN_S,
            cfg.BALL_CRESCENT_HISTORY_MIN_FORWARD_S,
        ):
            return

        third = max(len(samples) // 3, 1)
        first = samples[:third]
        last = samples[-third:]

        first_radius = float(np.median(
            [sample[2] for sample in first]))
        last_radius = float(np.median(
            [sample[2] for sample in last]))
        first_bottom = float(np.median(
            [sample[3] for sample in first]))
        last_bottom = float(np.median(
            [sample[3] for sample in last]))
        close_endpoint = (
            last_radius >= cfg.BALL_CRESCENT_ARM_RADIUS_RATIO
            and last_bottom >= cfg.BALL_CRESCENT_ARM_BOTTOM_RATIO
        )
        grew = (
            last_radius - first_radius
            >= cfg.BALL_CRESCENT_ARM_RADIUS_GROWTH_RATIO
            or last_bottom - first_bottom
            >= cfg.BALL_CRESCENT_ARM_BOTTOM_GROWTH_RATIO
        )
        if not (close_endpoint and grew):
            return

        recent_centers = [sample[1] for sample in last]
        center = float(np.median(recent_centers))
        if max(
            abs(sample_center - center)
            for sample_center in recent_centers
        ) > cfg.BALL_CRESCENT_ASSOCIATION_X_RATIO:
            return
        self._crescent_token_center = center
        self._crescent_token_kind = self._history_kind
        self._crescent_token_until = (
            samples[-1][0] + cfg.BALL_CRESCENT_TOKEN_TTL_S)

    def _close_crescent_near(
        self,
        evidence,
        detection,
        frame_height,
        frame_width,
        now,
    ):
        if evidence is None or not bool(
            getattr(evidence, "accepted", False)
        ):
            return False
        evidence_age = now - float(evidence.timestamp)
        if not (-0.05 <= evidence_age <= cfg.BALL_FRAME_STALE_S):
            return False
        if (
            self._crescent_token_until is None
            or now > self._crescent_token_until
            or self._crescent_token_center is None
            or self._crescent_token_kind is None
        ):
            return False
        if (
            abs(
                float(evidence.center_x_ratio)
                - self._crescent_token_center
            )
            > cfg.BALL_CRESCENT_ASSOCIATION_X_RATIO
        ):
            return False

        # Se o Hough ainda existe, ele deve confirmar que o mesmo alvo ja esta
        # baixo e grande. Uma bolinha distante jamais pode "emprestar" o token
        # para um arco de parede/piso.
        if detection is not None:
            if detection.kind != self._crescent_token_kind:
                return False
            center_ratio = (
                float(detection.center_x) / max(frame_width, 1))
            radius_ratio = (
                float(detection.radius) / max(frame_height, 1))
            bottom_ratio = (
                float(detection.bottom_y) / max(frame_height, 1))
            close_inner_reflection = (
                bottom_ratio >= cfg.BALL_CRESCENT_INNER_BOTTOM_RATIO
                or (
                    radius_ratio >= cfg.BALL_CRESCENT_ARM_RADIUS_RATIO
                    and bottom_ratio
                    >= cfg.BALL_CRESCENT_ARM_BOTTOM_RATIO
                )
            )
            if (
                not close_inner_reflection
                or abs(
                    center_ratio - float(evidence.center_x_ratio)
                ) > cfg.BALL_CRESCENT_INNER_ASSOCIATION_X_RATIO
            ):
                return False
        return True

    def _clear_approach_history(self):
        self.approach_history.clear()
        self._last_history_timestamp = None
        self._history_kind = None
        self._crescent_token_until = None
        self._crescent_token_center = None
        self._crescent_token_kind = None

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
