"""
control/steer.py — steer(angle, speed): the P-only steering law.
Ported from Overengineering² Reading Dossier, Hotspot 2
  Original source: robot_v.3/Python/main/control.py
    - steer(angle=190., speed=.8)   (lines 134-187) — law verbatim
Shadow2026 adaptations:
  - GPIO writes (gpiozero LED/PWMLED on the L298N) replaced by SPEC 01 serial
    commands to the Uno: stop → PARAR, everything else → LADO <esq> <dir>
    (signed left/right wheel speeds; the firmware maps sign → direction pins,
    so the OE² pin-name inversion of dossier §9.1 does not carry over).
  - speed ∈ [0, 1] exactly like OE²; converted to pwm = round(speed × 120)
    at the very end. |pwm| ≤ 120 is asserted in serial_link/arduino.py.
  - sleep_steering(duration): drop-in replacement for time.sleep() in ported
    maneuver code — re-sends the last command every 0.25 s so the Uno's 1 s
    watchdog never fires mid-maneuver (OE² had no watchdog and slept freely).

This module deliberately does NOT import shared/mp_manager, so bench tools
(tools/steer_test.py) can use it without spawning a multiprocessing Manager.
"""

import time

from config import (FRONT_ANCHORED_STEERING, FRONT_ANCHOR_FULL_ANGLE,
                    FRONT_ANCHOR_MAX_BLEND, FRONT_ANCHOR_REAR_SCALE,
                    FRONT_ANCHOR_START_ANGLE,
                    MAX_PWM, left_correction, max_turn_angle, right_correction)

# Instancia definida por init_steering() no processo de controle (ou nos tools).
arduino = None


def init_steering(arduino_instance):
    global arduino
    arduino = arduino_instance


def steer(angle=190., speed=.8):
    """Command vocabulary (dossier Hotspot 2):
    angle == 190 -> full stop; angle == 200 -> straight backward at `speed`;
    angle in [-180, 180] -> forward, positive = steer right;
    |angle| <= max_turn_angle (110) -> arc (inner wheel scaled down linearly);
    |angle| >  max_turn_angle       -> pivot in place at 1.2x speed."""

    # stop
    if angle == 190:
        arduino.parar()
        return

    # backward
    elif angle == 200:
        speed_left = -max(speed * left_correction, 0)
        speed_right = -max(speed * right_correction, 0)

    # forward
    elif -180 <= angle <= 180:

        # right
        if angle >= 0:
            if angle > max_turn_angle:
                # pivot: roda interna (direita) inverte o sentido
                speed_left = min(speed * left_correction * 1.2, 1)
                speed_right = -min(speed * right_correction * 1.2, 1)
            else:
                speed_left = min(speed * left_correction, 1)
                speed_right = min(speed * right_correction * ((max_turn_angle - angle) / (max_turn_angle - 1)), 1)

        # left
        else:
            if angle < -max_turn_angle:
                # pivot: roda interna (esquerda) inverte o sentido
                speed_left = -min(speed * left_correction * 1.2, 1)
                speed_right = min(speed * right_correction * 1.2, 1)
            else:
                speed_left = min(speed * left_correction * ((max_turn_angle + angle) / (max_turn_angle - 1)), 1)
                speed_right = min(speed * right_correction, 1)

    else:
        # angulo fora do vocabulario: para por seguranca
        arduino.parar()
        return

    # Para erros grandes, desloca progressivamente o centro de giro para a
    # frente do chassi. No limite, as rodas dianteiras ficam quase paradas e
    # somente a traseira gira em sentidos opostos. Isso faz a traseira buscar
    # o alinhamento apontado pela bolinha inferior sem um caso especial de 90°.
    if FRONT_ANCHORED_STEERING and -180 <= angle <= 180 and \
            abs(angle) > FRONT_ANCHOR_START_ANGLE:
        span = max(FRONT_ANCHOR_FULL_ANGLE - FRONT_ANCHOR_START_ANGLE, 1)
        blend = min((abs(angle) - FRONT_ANCHOR_START_ANGLE) / span,
                    FRONT_ANCHOR_MAX_BLEND)
        rear_speed = min(speed * FRONT_ANCHOR_REAR_SCALE, 1.)

        if angle > 0:  # direita: traseira esquerda avanca, direita recua
            anchor_te, anchor_td = rear_speed, -rear_speed
        else:          # esquerda: traseira direita avanca, esquerda recua
            anchor_te, anchor_td = -rear_speed, rear_speed

        front_left = speed_left * (1 - blend)
        front_right = speed_right * (1 - blend)
        rear_left = speed_left * (1 - blend) + anchor_te * blend
        rear_right = speed_right * (1 - blend) + anchor_td * blend

        arduino.rodas(round(front_left * MAX_PWM),
                      round(rear_left * MAX_PWM),
                      round(front_right * MAX_PWM),
                      round(rear_right * MAX_PWM))
    else:
        arduino.lado(round(speed_left * MAX_PWM), round(speed_right * MAX_PWM))


def sleep_steering(duration):
    """time.sleep() que mantém o watchdog do Uno alimentado (keepalive)."""
    end = time.monotonic() + duration
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        if arduino is not None:
            arduino.refresh()
        time.sleep(min(.05, remaining))
