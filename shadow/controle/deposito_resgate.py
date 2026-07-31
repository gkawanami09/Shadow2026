"""Navega ate o triangulo de evacuacao sem usar ultrassom ou IMU."""

import time

import numpy as np

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


class DepositMarkerController:
    """Procura, confirma, centraliza e aproxima um marcador de uma cor."""

    START = "DEPOSIT_SEARCH_START"
    ROTATING = "DEPOSIT_SEARCH"
    TARGET_STOP = "DEPOSIT_TARGET_STOP"
    VERIFY = "DEPOSIT_VERIFY"
    TURN_STOP = "DEPOSIT_TURN_STOP"
    FINAL_VERIFY = "DEPOSIT_FINAL_VERIFY"
    ALIGN = "DEPOSIT_ALIGN"
    APPROACH = "DEPOSIT_APPROACH"
    LOST_STOP = "DEPOSIT_LOST_STOP"
    ARRIVAL_STOP = "DEPOSIT_ARRIVAL_STOP"
    ARRIVED = "DEPOSIT_ARRIVED"
    FAULT = "DEPOSIT_FAULT"

    def __init__(
        self,
        target_kind,
        start_time=None,
        near_confirm_frames=None,
        align_tank_speed=None,
        approach_speed=None,
    ):
        if target_kind not in ("green", "red"):
            raise ValueError("marcador deve ser green ou red")
        if near_confirm_frames is None:
            near_confirm_frames = cfg.DEPOSIT_NEAR_CONFIRM_FRAMES
        near_confirm_frames = int(near_confirm_frames)
        if near_confirm_frames < 1:
            raise ValueError("near_confirm_frames deve ser positivo")
        if align_tank_speed is not None and float(align_tank_speed) <= 0.0:
            raise ValueError("align_tank_speed deve ser positivo")
        if approach_speed is not None and float(approach_speed) <= 0.0:
            raise ValueError("approach_speed deve ser positivo")
        self.target_kind = target_kind
        self.near_confirm_frames = near_confirm_frames
        self.align_tank_speed = (
            None if align_tank_speed is None else float(align_tank_speed)
        )
        self.approach_speed = (
            None if approach_speed is None else float(approach_speed)
        )
        self.state = self.START
        self._created_at = (
            time.monotonic()
            if start_time is None
            else float(start_time)
        )
        self._rotation_started_at = None
        self._rotation_elapsed_s = 0.0
        self._active_started_at = None
        self._stopped_at = None
        self._last_seen_at = None
        self._tentative_target = False
        self._tracking_reset_requested = False
        self._near_count = 0
        self._near_first_at = None
        self._last_near_timestamp = None
        self._progress_at = None
        self._progress_error = None
        self._progress_width = None
        self._progress_bottom = None
        self._terminal_detail = ""

    @property
    def terminal(self):
        return self.state in (self.ARRIVED, self.FAULT)

    @property
    def arrived(self):
        return self.state == self.ARRIVED

    def update(self, detection, frame_shape, now=None):
        now = time.monotonic() if now is None else float(now)
        height, width = frame_shape[:2]

        if self.state == self.ARRIVED:
            return self._stop(
                self.ARRIVED,
                "marcador correto alcancado; deposito autorizado",
                terminal=True,
            )
        if self.state == self.FAULT:
            return self._stop(
                self.FAULT,
                self._terminal_detail,
                terminal=True,
            )
        if (
            self._active_started_at is not None
            and now - self._active_started_at
            >= cfg.DEPOSIT_MAX_ACTIVE_S
        ):
            return self._fault(
                "tempo maximo navegando ate o triangulo; "
                "esfera mantida na garra"
            )
        if self.state == self.TARGET_STOP:
            return self._stop(
                self.TARGET_STOP,
                "marcador encontrado; aguardando confirmacao de PARAR",
            )
        if self.state == self.TURN_STOP:
            return self._stop(
                self.TURN_STOP,
                "aguardando PARAR depois da busca de 360 graus",
            )
        if self.state == self.LOST_STOP:
            return self._stop(
                self.LOST_STOP,
                "marcador perdido; aguardando PARAR antes de retomar",
            )
        if self.state == self.ARRIVAL_STOP:
            return self._stop(
                self.ARRIVAL_STOP,
                "chegada confirmada; aguardando PARAR antes do deposito",
            )

        if self.state == self.START:
            if self._valid_target(detection, now):
                return self._request_target_stop(detection)
            if self._plausible_target(detection, now):
                return self._request_target_stop(detection, tentative=True)
            return self._tank(
                self.START,
                f"procurando triangulo {self.target_kind}",
            )

        if self.state == self.ROTATING:
            if self._valid_target(detection, now):
                return self._request_target_stop(detection)
            if self._plausible_target(detection, now):
                return self._request_target_stop(detection, tentative=True)
            if (
                self._rotation_started_at is not None
                and self._rotation_elapsed(now)
                >= cfg.DEPOSIT_SEARCH_FULL_TURN_S - 1e-9
            ):
                self.state = self.TURN_STOP
                return self._stop(
                    self.TURN_STOP,
                    "360 concluido; verificando o marcador com o robo parado",
                )
            return self._tank(
                self.ROTATING,
                f"girando devagar; procurando triangulo {self.target_kind}",
            )

        if self.state in (self.VERIFY, self.FINAL_VERIFY):
            valid = self._valid_target(
                detection,
                now,
                captured_after=self._stopped_at,
            )
            if valid:
                self._tentative_target = False
                self._last_seen_at = now
                self._reset_near()
                return self._drive_to_target(
                    detection, width, height, now)

            if (
                self._stopped_at is not None
                and now - self._stopped_at
                >= cfg.DEPOSIT_SEARCH_VERIFY_TIMEOUT_S
            ):
                if self.state == self.FINAL_VERIFY:
                    return self._fault(
                        f"triangulo {self.target_kind} nao encontrado "
                        "apos um giro completo; esfera mantida na garra"
                    )
                self.state = (
                    self.TURN_STOP
                    if self._rotation_elapsed_s
                    >= cfg.DEPOSIT_SEARCH_FULL_TURN_S - 1e-9
                    else self.START
                )
                self._stopped_at = None
                self._tentative_target = False
                self._tracking_reset_requested = True
                self._reset_progress()
                if self.state == self.TURN_STOP:
                    return self._stop(
                        self.TURN_STOP,
                        "falso marcador descartado no fim do 360",
                    )
                return self._tank(
                    self.START,
                    "falso marcador descartado; retomando a busca",
                )
            return self._stop(
                self.state,
                f"reconfirmando triangulo {self.target_kind} parado",
            )

        if self.state in (self.ALIGN, self.APPROACH):
            if not self._valid_target(detection, now):
                self._reset_near()
                lost_for = now - (
                    self._last_seen_at
                    if self._last_seen_at is not None
                    else now
                )
                if lost_for >= cfg.DEPOSIT_REACQUIRE_TIMEOUT_S:
                    self.state = self.LOST_STOP
                    return self._stop(
                        self.LOST_STOP,
                        "marcador perdido; parando antes de nova busca",
                    )
                return self._stop(
                    self.state,
                    f"marcador ausente ha {lost_for:.2f}s; robo parado",
                )
            self._last_seen_at = now
            return self._drive_to_target(detection, width, height, now)

        raise RuntimeError(
            f"estado de deposito desconhecido: {self.state}")

    def mark_rotation_started(self, now=None):
        if self.state != self.START:
            raise RuntimeError(
                "confirmacao do giro de deposito fora do estado inicial")
        now = time.monotonic() if now is None else float(now)
        self.state = self.ROTATING
        self._rotation_started_at = now
        self._mark_active(now)

    def mark_target_stopped(self, now=None):
        if self.state != self.TARGET_STOP:
            raise RuntimeError(
                "confirmacao de PARAR fora do marcador encontrado")
        now = time.monotonic() if now is None else float(now)
        self._mark_active(now)
        self._finish_rotation_segment(now)
        self.state = self.VERIFY
        self._stopped_at = now

    def mark_full_turn_stopped(self, now=None):
        if self.state != self.TURN_STOP:
            raise RuntimeError(
                "confirmacao de PARAR fora do fim da busca de marcador")
        now = time.monotonic() if now is None else float(now)
        self._mark_active(now)
        self._finish_rotation_segment(now)
        self.state = self.FINAL_VERIFY
        self._stopped_at = now

    def mark_lost_stopped(self, now=None):
        if self.state != self.LOST_STOP:
            raise RuntimeError(
                "confirmacao de PARAR fora da perda do marcador")
        now = time.monotonic() if now is None else float(now)
        self._mark_active(now)
        self.state = self.START
        self._stopped_at = now
        self._tracking_reset_requested = True
        self._reset_progress()

    def mark_arrival_stopped(self, now=None):
        if self.state != self.ARRIVAL_STOP:
            raise RuntimeError(
                "confirmacao de PARAR fora da chegada ao marcador")
        self._stopped_at = (
            time.monotonic() if now is None else float(now))
        self.state = self.ARRIVED

    def frame_allowed(self, captured_at):
        if (
            self.state not in (self.VERIFY, self.FINAL_VERIFY)
            or self._stopped_at is None
        ):
            return True
        return float(captured_at) > self._stopped_at + 1e-9

    def consume_tracking_reset(self):
        requested = self._tracking_reset_requested
        self._tracking_reset_requested = False
        return requested

    def _drive_to_target(self, detection, width, height, now):
        error = float(np.clip(
            (float(detection.center_x) - width / 2.0)
            / max(width / 2.0, 1.0),
            -1.0,
            1.0,
        ))
        width_ratio = float(detection.width) / max(width, 1)
        bottom_ratio = float(detection.bottom_y) / max(height, 1)
        near = (
            width_ratio >= cfg.DEPOSIT_NEAR_MIN_WIDTH_RATIO
            and bottom_ratio >= cfg.DEPOSIT_NEAR_MIN_BOTTOM_RATIO
            and abs(error) <= cfg.DEPOSIT_NEAR_MAX_CENTER_ERROR
        )
        if near:
            if (
                self._near_first_at is None
                or now - self._near_first_at
                > cfg.DEPOSIT_NEAR_CONFIRM_WINDOW_S
            ):
                self._reset_near()
            if (
                self._last_near_timestamp is None
                or float(detection.timestamp)
                > self._last_near_timestamp + 1e-9
            ):
                if self._near_first_at is None:
                    self._near_first_at = now
                self._near_count += 1
                self._last_near_timestamp = float(detection.timestamp)
            if self._near_count >= self.near_confirm_frames:
                self.state = self.ARRIVAL_STOP
                return self._stop(
                    self.ARRIVAL_STOP,
                    f"triangulo {self.target_kind} proximo e centralizado",
                )
            return self._stop(
                self.APPROACH,
                "marcador no ponto de deposito; confirmando "
                f"{self._near_count}/{self.near_confirm_frames}",
            )
        self._reset_near()
        if not self._observe_progress(
            error,
            width_ratio,
            bottom_ratio,
            now,
        ):
            return self._fault(
                "sem progresso visual ate o triangulo; "
                "motores parados e esfera mantida na garra"
            )

        align_threshold = (
            cfg.DEPOSIT_ALIGN_EXIT_ERROR
            if self.state == self.ALIGN
            else cfg.DEPOSIT_ALIGN_ENTER_ERROR
        )
        if abs(error) > align_threshold:
            self.state = self.ALIGN
            if self.align_tank_speed is not None:
                angle = (
                    cfg.DEPOSIT_SEARCH_TANK_ANGLE
                    if error > 0
                    else -cfg.DEPOSIT_SEARCH_TANK_ANGLE
                )
                return MotionCommand(
                    self.ALIGN,
                    angle=angle,
                    speed=self.align_tank_speed,
                    detail=(
                        f"centralizando marcador {self.target_kind} em tanque; "
                        f"erro={error:+.3f}"
                    ),
                )
            severity = float(np.clip(
                (
                    abs(error) - cfg.DEPOSIT_ALIGN_EXIT_ERROR
                ) / max(1.0 - cfg.DEPOSIT_ALIGN_EXIT_ERROR, 1e-6),
                0.0,
                1.0,
            ))
            magnitude = int(round(
                cfg.DEPOSIT_ALIGN_ANGLE_MIN
                + severity
                * (
                    cfg.DEPOSIT_ALIGN_ANGLE_MAX
                    - cfg.DEPOSIT_ALIGN_ANGLE_MIN
                )
            ))
            speed = (
                cfg.DEPOSIT_ALIGN_SPEED_MIN
                + severity
                * (
                    cfg.DEPOSIT_ALIGN_SPEED_MAX
                    - cfg.DEPOSIT_ALIGN_SPEED_MIN
                )
            )
            return MotionCommand(
                self.ALIGN,
                angle=magnitude if error > 0 else -magnitude,
                speed=float(speed),
                detail=(
                    f"centralizando triangulo {self.target_kind}; "
                    f"erro={error:+.3f}"
                ),
            )

        self.state = self.APPROACH
        if abs(error) <= cfg.DEPOSIT_APPROACH_CENTER_DEADBAND:
            angle = 0
        else:
            angle = int(round(np.clip(
                error
                / max(cfg.DEPOSIT_ALIGN_ENTER_ERROR, 1e-6)
                * cfg.DEPOSIT_APPROACH_STEER_MAX_ANGLE,
                -cfg.DEPOSIT_APPROACH_STEER_MAX_ANGLE,
                cfg.DEPOSIT_APPROACH_STEER_MAX_ANGLE,
            )))
        near_fraction = float(np.clip(
            width_ratio / max(cfg.DEPOSIT_NEAR_MIN_WIDTH_RATIO, 1e-6),
            0.0,
            1.0,
        ))
        speed = self.approach_speed
        if speed is None:
            speed = (
                cfg.DEPOSIT_APPROACH_SPEED_FAR * (1.0 - near_fraction)
                + cfg.DEPOSIT_APPROACH_SPEED_NEAR * near_fraction
            )
        return MotionCommand(
            self.APPROACH,
            angle=angle,
            speed=float(speed),
            detail=(
                f"aproximando triangulo {self.target_kind}; "
                f"erro={error:+.3f}, largura={width_ratio:.2f}W"
            ),
        )

    def _request_target_stop(self, detection, tentative=False):
        self.state = self.TARGET_STOP
        self._tentative_target = bool(tentative)
        self._reset_progress()
        return self._stop(
            self.TARGET_STOP,
            (
                "candidato de marcador encontrado; freando para confirmar"
                if tentative
                else "marcador travado; freando para confirmar"
            ),
        )

    def _plausible_target(self, detection, now):
        if (
            detection is None
            or detection.kind != self.target_kind
            or float(detection.confidence)
            < float(getattr(cfg, "MARKER_MIN_CONFIDENCE", 0.60))
        ):
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    def _valid_target(self, detection, now, captured_after=None):
        if (
            detection is None
            or detection.kind != self.target_kind
            or not bool(detection.confirmed)
            or not bool(getattr(detection, "track_locked", False))
        ):
            return False
        if (
            captured_after is not None
            and float(detection.timestamp)
            <= float(captured_after) + 1e-9
        ):
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    def _rotation_elapsed(self, now):
        elapsed = self._rotation_elapsed_s
        if self._rotation_started_at is not None:
            elapsed += max(float(now) - self._rotation_started_at, 0.0)
        return elapsed

    def _finish_rotation_segment(self, now):
        if self._rotation_started_at is None:
            return
        self._rotation_elapsed_s = self._rotation_elapsed(now)
        self._rotation_started_at = None

    def _reset_near(self):
        self._near_count = 0
        self._near_first_at = None
        self._last_near_timestamp = None

    def _mark_active(self, now):
        if self._active_started_at is None:
            self._active_started_at = float(now)

    def _reset_progress(self):
        self._progress_at = None
        self._progress_error = None
        self._progress_width = None
        self._progress_bottom = None

    def _observe_progress(
        self,
        error,
        width_ratio,
        bottom_ratio,
        now,
    ):
        """Exige que o chassi mude a geometria vista do marcador."""
        error = abs(float(error))
        width_ratio = float(width_ratio)
        bottom_ratio = float(bottom_ratio)
        now = float(now)
        if self._progress_at is None:
            self._progress_at = now
            self._progress_error = error
            self._progress_width = width_ratio
            self._progress_bottom = bottom_ratio
            return True

        improved = (
            error
            <= self._progress_error - cfg.DEPOSIT_PROGRESS_MIN_ERROR
            or width_ratio
            >= (
                self._progress_width
                + cfg.DEPOSIT_PROGRESS_MIN_WIDTH_RATIO
            )
            or bottom_ratio
            >= (
                self._progress_bottom
                + cfg.DEPOSIT_PROGRESS_MIN_BOTTOM_RATIO
            )
        )
        if improved:
            self._progress_at = now
            self._progress_error = error
            self._progress_width = width_ratio
            self._progress_bottom = bottom_ratio
            return True
        return (
            now - self._progress_at
            < cfg.DEPOSIT_PROGRESS_TIMEOUT_S
        )

    def _fault(self, detail):
        self.state = self.FAULT
        self._terminal_detail = str(detail)
        return self._stop(
            self.FAULT,
            self._terminal_detail,
            terminal=True,
        )

    @staticmethod
    def _tank(state, detail):
        return MotionCommand(
            state,
            angle=cfg.DEPOSIT_SEARCH_TANK_ANGLE,
            speed=cfg.DEPOSIT_SEARCH_TANK_SPEED,
            detail=detail,
        )

    @staticmethod
    def _stop(state, detail, terminal=False):
        return MotionCommand(
            state,
            angle=190,
            speed=0.0,
            detail=detail,
            terminal=terminal,
        )
