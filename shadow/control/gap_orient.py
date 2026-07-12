"""
control/gap_orient.py — gap validation, square-up and commit.
Ported from Overengineering² Reading Dossier, Hotspot 3
  Original source: robot_v.3/Python/main/control.py
    - drive_back_until_line   (lines 452-464) — verbatim
    - ensure_line_detected    (lines 467-478) — verbatim
    - orientate_gap           (lines 481-676) — 3 phases:
        1. validate  (483-509): reverse until the line stub is visible again
        2. square-up (511-643): <= 7 correction cycles against gap_angle /
           gap_center_x, plus the lateral-offset fix when |x_gap| > 55
        3. commit    (638-643): line_status = "gap_avoid", drive 0.8 s @ .7
Shadow2026 adaptations:
  - silver_detected() / obstacle_detected() call sites replaced with literal
    False (no silver AI, no IR sensors) — the surrounding logic is unchanged.
  - update_sensor_average() dropped (IR-only).
  - program_continue() (run switch) -> `terminate.value` checks.
  - time.sleep -> sleep_steering (Uno watchdog keepalive).
  # IMU_REPLACEMENT: the gyro search sweep (lines 653-676, turn_to_angle
    +/-45/90° stop-on-black) is replaced by TIMED pivots: right T_SWEEP_RIGHT,
    left 2x, right T_SWEEP_RIGHT back, each aborting early when
    line_detected flips true (mission §3.2); the final 1.2 s creep is kept.
  - min_line_size audit (dossier §9 item 10): every `return False` path here
    may leave min_line_size at 9000; the caller (control/loop.py gap dispatch)
    restores it to 3000 on ANY False return — same as OE² lines 1944-1945.
  - y_gap is deliberately NOT refreshed inside the correction loop (only angle
    and x_gap are) — that matches the original code exactly.
"""

import time

import numpy as np

from config import (GAP_BLACK_AVG_MAX, GAP_COMMIT_SPEED, GAP_COMMIT_TIME,
                    GAP_CORRECTION_CYCLES, GAP_MIN_LINE_SIZE_COMMIT,
                    GAP_MAX_END_WIDTH_PX, GAP_MIN_LINE_SIZE_ORIENT, GAP_NOT_A_STUB_SIZE,
                    LINE_SEARCH_CREEP, MIN_LINE_SIZE_DEFAULT, SWEEP_SPEED,
                    T_SWEEP_RIGHT, camera_y)
from control.steer import sleep_steering, steer
from shared.mp_manager import (black_average, gap_angle, gap_center_x, gap_center_y,
                               gap_end_width, line_ahead, line_detected, line_size,
                               line_status, min_line_size,
                               status, terminate, timer)


def drive_back_until_line(max_time, speed=.7):

    timer.set_timer("find_line_again", max_time)
    while not line_detected.value and not timer.get_timer("find_line_again"):
        steer(200, speed)
        time.sleep(.001)
    min_line_size.value = MIN_LINE_SIZE_DEFAULT

    steer(0, .7)
    sleep_steering(.2)
    steer()

    return line_detected.value


def ensure_line_detected():
    sleep_steering(.25)
    if not line_detected.value:
        steer(200, .7)
        sleep_steering(.15)
        steer()
        sleep_steering(.1)

        if not line_detected.value:
            return False

    return True


def _timed_sweep(direction, duration):
    """IMU_REPLACEMENT: pivot temporizado; aborta cedo se a linha aparecer.
    Retorna True se a linha foi detectada durante a varredura."""
    steer(180 if direction == "r" else -180, SWEEP_SPEED)
    end = time.monotonic() + duration
    while time.monotonic() < end:
        if line_detected.value or terminate.value:
            steer()
            return line_detected.value
        sleep_steering(.01)
    steer()
    return False


def orientate_gap():
    # A validacao nunca pode movimentar o robo se a camera ainda enxerga uma
    # continuacao vertical material. Isso tambem cobre o frame que chega entre
    # a decisao do loop de controle e a entrada nesta funcao.
    steer()
    status.value = 'Validando gap — aguardando confirmacao visual'
    sleep_steering(.1)
    if line_ahead.value:
        status.value = 'Validacao falhou — linha continua a frente'
        return False

    if gap_end_width.value > GAP_MAX_END_WIDTH_PX:
        status.value = 'Validacao falhou — ponta larga demais para gap'
        return False

    if not line_detected.value or line_size.value < GAP_NOT_A_STUB_SIZE:
        status.value = 'Validando gap'

        steer(200, .7)
        sleep_steering(.15)
        steer()
        sleep_steering(.2)

        if line_ahead.value:
            status.value = 'Validacao falhou — linha continua a frente'
            return False

        if gap_end_width.value > GAP_MAX_END_WIDTH_PX:
            status.value = 'Validacao falhou — ponta larga demais para gap'
            return False

        steer(200, .7)
        sleep_steering(.3)
        if not line_detected.value:
            sleep_steering(.2)

        steer(0, .7)
        sleep_steering(.25)
        steer()

    if line_detected.value and black_average.value < GAP_BLACK_AVG_MAX:
        status.value = 'Orientando no gap'

        angle = gap_angle.value
        x_gap = gap_center_x.value
        y_gap = gap_center_y.value

        correction_counter = 0
        while correction_counter < GAP_CORRECTION_CYCLES:

            if gap_end_width.value > GAP_MAX_END_WIDTH_PX:
                status.value = 'Validacao falhou — ponta larga demais para gap'
                return False

            if y_gap < 10:
                return False

            time_foreward = .25

            if (0 < angle < 173 and x_gap < 0) or (angle < -7 and x_gap > 0) or (0 < angle < 155) or (angle < -25):
                min_time = .35

                if (0 < angle < 173 and x_gap < 0) or (angle < -7 and x_gap > 0):
                    x_gap_perc = pow(abs(x_gap) / (180 + 40), .7)
                else:
                    x_gap_perc = 0

                y_gap_perc = y_gap / camera_y

                if angle > 0:
                    angle_perc = (angle - 90) / 90
                else:
                    angle_perc = abs(-angle / 90)

                time_foreward = min_time + .3 * x_gap_perc + .1 * y_gap_perc + .3 * angle_perc

            if not (0 >= angle > -1 or angle > 179):
                steer(0, .7)
                sleep_steering(time_foreward)

                if terminate.value:
                    return False

                if angle > 0:
                    steer(180, .65)
                    sleep_steering(abs(.9 - .85 * ((angle - 90) / 90)))
                else:
                    steer(-180, .65)
                    sleep_steering(abs(.05 + .85 * abs(-angle / 90)))

                steer()

                if terminate.value:
                    return False

                min_line_size.value = GAP_MIN_LINE_SIZE_ORIENT
                steer(200, .7)
                sleep_steering(time_foreward + np.clip(((time_foreward - .25) / .4) * .15, 0, .15))

                if not drive_back_until_line(.6, .7):
                    return False

                if line_size.value > GAP_NOT_A_STUB_SIZE:
                    steer(200, .7)
                    sleep_steering(.2)
                    steer()
                    return False

            if not ensure_line_detected() or terminate.value:
                return False

            angle = gap_angle.value
            x_gap = gap_center_x.value

            if y_gap < 10:
                return False

            if abs(x_gap) > 55:
                steer(180 if x_gap > 0 else -180, .6)
                sleep_steering(.4)
                steer()
                sleep_steering(.2)

                time_foreward = .35 + .35 * ((abs(x_gap) - 55) / 100)

                steer(0, .7)
                sleep_steering(time_foreward)
                steer()

                if terminate.value:
                    return False

                steer(-180 if x_gap > 0 else 180, .6)
                sleep_steering(.3)
                steer()
                sleep_steering(.2)

                if terminate.value:
                    return False

                min_line_size.value = GAP_MIN_LINE_SIZE_ORIENT
                steer(200, .7)
                sleep_steering(time_foreward + np.clip(((time_foreward - .35) / .35) * .2, 0, .2))

                if not drive_back_until_line(.5, .7):
                    return False

                if line_size.value > GAP_NOT_A_STUB_SIZE:
                    steer(200, .7)
                    sleep_steering(.2)
                    steer()
                    return False

                if not ensure_line_detected() or terminate.value:
                    return False

                angle = gap_angle.value
                x_gap = gap_center_x.value

                if y_gap < 10:
                    return False

            if (0 >= angle > -1 or angle > 179) and abs(x_gap) < 140:
                break

            if terminate.value:
                return False

            correction_counter += 1

        if gap_end_width.value > GAP_MAX_END_WIDTH_PX:
            status.value = 'Validacao falhou — ponta larga demais para gap'
            return False

        status.value = 'Gap orientado — cruzando'
        line_status.value = "gap_avoid"
        min_line_size.value = GAP_MIN_LINE_SIZE_COMMIT
        steer(0, GAP_COMMIT_SPEED)
        sleep_steering(GAP_COMMIT_TIME)
        return True

    elif line_detected.value and black_average.value > GAP_BLACK_AVG_MAX:
        status.value = 'Validação falhou — não é gap'
        steer(200, .7)
        sleep_steering(.2)
        steer()
        return False

    else:
        status.value = 'Procurando a linha'

        # IMU_REPLACEMENT: varredura temporizada no lugar do gyro sweep ±45/90°
        if _timed_sweep("r", T_SWEEP_RIGHT) or terminate.value:
            return False

        if _timed_sweep("l", 2 * T_SWEEP_RIGHT) or terminate.value:
            return False

        if _timed_sweep("r", T_SWEEP_RIGHT) or terminate.value:
            return False

        timer.set_timer("line_search", LINE_SEARCH_CREEP)
        while not line_detected.value and not timer.get_timer("line_search") and not terminate.value:
            steer(0, .7)
            time.sleep(.001)

        steer()
        return False
