"""
control/turn_around.py — 180° turn on double green (timed-pivot replacement).
Ported from Overengineering² Reading Dossier, Hotspot 4 (180° execution)
  Original source: robot_v.3/Python/main/control.py
    - turn_around()   (lines 679-729; general path lines 711-729)
Shadow2026 adaptations:
  # IMU_REPLACEMENT: the gyro turn `turn_to_angle(round_angle(yaw+180, 90))`
  is replaced by a timed pivot of T_180 s at T_180_SPEED in the direction of
  `last_turn_dir` (mission §3.2 — tune T_180 during commissioning).
  Kept verbatim from OE²: the forward pre-roll over the marker (0.55 s), the
  reverse line-reacquisition tail (0.3 s, +0.4 s if line_size < 5500), the
  stuck-cooldown re-arm, and the l/r ALTERNATION of the turn direction.
  Dropped: the ramp-side open-loop wiggle (lines 681-709, sensor_z-gated —
  can never fire without an IMU; dossier §9 item 8) and the was_ramp_up
  timing variants.
"""

from config import (T_180, T_180_SPEED, TURN_AROUND_PREROLL, TURN_AROUND_REVERSE,
                    TURN_AROUND_REVERSE_EXTRA, TURN_AROUND_SMALL_LINE)
from control.steer import sleep_steering, steer
from shared.mp_manager import line_size, timer


def turn_around(last_turn_dir):
    """Executes the 180° and returns the NEXT turn direction ("l"/"r")."""
    # avanca por cima do marcador duplo
    steer(0, .7)
    sleep_steering(TURN_AROUND_PREROLL)

    # IMU_REPLACEMENT: pivot temporizado no lugar do giro por giroscopio
    steer(180 if last_turn_dir == "r" else -180, T_180_SPEED)
    sleep_steering(T_180)
    steer()

    # re-aquisicao da linha (cauda identica ao OE²)
    steer(200, .7)
    sleep_steering(TURN_AROUND_REVERSE)
    steer()

    if line_size.value < TURN_AROUND_SMALL_LINE:
        steer(200, .7)
        sleep_steering(TURN_AROUND_REVERSE_EXTRA)
        steer()

    timer.set_timer("stuck_cooldown", 5)

    return "r" if last_turn_dir == "l" else "l"
