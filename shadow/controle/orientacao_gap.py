"""Valida, alinha e atravessa um gap com limites de segurança."""

import time

import numpy as np

from config import (GAP_BLACK_AVG_MAX, GAP_COMMIT_SPEED, GAP_COMMIT_TIME,
                    GAP_CORRECTION_CYCLES, GAP_MIN_LINE_SIZE_COMMIT,
                    GAP_MAX_END_WIDTH_PX, GAP_MIN_LINE_SIZE_ORIENT, GAP_NOT_A_STUB_SIZE,
                    LINE_SEARCH_CREEP, MIN_LINE_SIZE_DEFAULT, SWEEP_SPEED,
                    T_SWEEP_RIGHT, camera_y)
from controle.direcao import sleep_steering, steer
from shared.dados_compartilhados import (black_average, gap_angle, gap_center_x, gap_center_y,
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
    """Faz uma busca temporizada e para quando a linha aparece.
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

        # Varredura temporizada porque o robô não possui giroscópio.
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
