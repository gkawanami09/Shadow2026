"""Manobra temporizada para curva preta de 90 graus sem marcador verde."""

from config import (CORNER_90_FORWARD_TIME, CORNER_90_PIVOT_TIME,
                    CORNER_90_REVERSE_TIME, CORNER_90_SPEED)
from control.steer import sleep_steering, steer
from shared.mp_manager import status


def execute_corner_90(direction):
    """Leva o centro ao canto, faz pivot e recua para reposicionar a camera."""
    status.value = f'90 graus {direction}: centralizando'
    steer(0, CORNER_90_SPEED)
    sleep_steering(CORNER_90_FORWARD_TIME)

    status.value = f'90 graus {direction}: girando'
    steer(-180 if direction == "left" else 180, CORNER_90_SPEED)
    sleep_steering(CORNER_90_PIVOT_TIME)
    steer()

    status.value = f'90 graus {direction}: reposicionando camera'
    steer(200, CORNER_90_SPEED)
    sleep_steering(CORNER_90_REVERSE_TIME)
    steer()
    sleep_steering(.1)
