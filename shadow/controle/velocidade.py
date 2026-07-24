"""Escolhe a velocidade do segue-linha de acordo com a curva."""

from config import (LINE_FOLLOW_SPEED, RAMP_AHEAD_HOLD, RAMP_AHEAD_SPEED_ARC,
                    RAMP_AHEAD_SPEED_PIVOT, RAMP_AHEAD_SPEED_STRAIGHT,
                    max_turn_angle)
from shared.dados_compartilhados import ramp_ahead, timer


def get_speed(angle):
    if ramp_ahead.value or not timer.get_timer("ramp_ahead"):
        if ramp_ahead.value:
            timer.set_timer("ramp_ahead", RAMP_AHEAD_HOLD)

        if abs(angle) > max_turn_angle:
            return RAMP_AHEAD_SPEED_PIVOT
        elif abs(angle) > max_turn_angle / 2:
            return RAMP_AHEAD_SPEED_ARC
        else:
            return RAMP_AHEAD_SPEED_STRAIGHT

    return LINE_FOLLOW_SPEED
