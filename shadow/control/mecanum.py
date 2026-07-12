"""Controle visual fechado da camera frontal usando as quatro rodas mecanum."""

import time

import numpy as np

from config import (MECANUM_DERIVATIVE_FILTER, MECANUM_FOLLOW_MAX_PWM,
                    MECANUM_LATERAL_KD, MECANUM_LATERAL_KP,
                    MECANUM_LATERAL_MAX_PWM, MECANUM_ROTATION_KD,
                    MECANUM_ROTATION_KP, MECANUM_ROTATION_MAX_PWM)
from control.steer import drive_mecanum


class MecanumCameraController:
    def __init__(self):
        self.reset()

    def reset(self):
        self._last_t = None
        self._last_lateral = 0.0
        self._last_heading = 0.0
        self._d_lateral = 0.0
        self._d_heading = 0.0

    def update(self, lateral_error, heading_error):
        now = time.monotonic()
        if self._last_t is None:
            dt = None
        else:
            dt = max(now - self._last_t, 1e-3)

        if dt is not None:
            raw_dl = (lateral_error - self._last_lateral) / dt
            raw_dh = (heading_error - self._last_heading) / dt
            alpha = MECANUM_DERIVATIVE_FILTER
            self._d_lateral = alpha * raw_dl + (1 - alpha) * self._d_lateral
            self._d_heading = alpha * raw_dh + (1 - alpha) * self._d_heading

        lateral_pwm = np.clip(
            MECANUM_LATERAL_KP * lateral_error
            + MECANUM_LATERAL_KD * self._d_lateral,
            -MECANUM_LATERAL_MAX_PWM, MECANUM_LATERAL_MAX_PWM)
        rotation_pwm = np.clip(
            MECANUM_ROTATION_KP * heading_error
            + MECANUM_ROTATION_KD * self._d_heading,
            -MECANUM_ROTATION_MAX_PWM, MECANUM_ROTATION_MAX_PWM)

        self._last_t = now
        self._last_lateral = lateral_error
        self._last_heading = heading_error
        drive_mecanum(MECANUM_FOLLOW_MAX_PWM, lateral_pwm, rotation_pwm,
                      MECANUM_FOLLOW_MAX_PWM)
        return float(lateral_pwm), float(rotation_pwm)
