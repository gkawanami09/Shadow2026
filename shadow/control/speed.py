"""
control/speed.py — get_speed(angle): the speed scheduler.
Ported from Overengineering² Reading Dossier, Hotspot 2
  Original source: robot_v.3/Python/main/control.py
    - get_speed(angle)   (lines 260-297)
Shadow2026 adaptations:
  # IMU_REPLACEMENT: rotation_y ramp branches (ramp_up lines 261-276,
  ramp_down lines 278-284) removed — rotation_y is constant "none" without a
  BNO055, so those branches could never fire. Kept: flat = 1.0 and the
  camera-only `ramp_ahead` branch (lines 286-295), which is fed by the
  Hotspot 1 dark-ahead detector and is fully portable.
"""

from config import (RAMP_AHEAD_HOLD, RAMP_AHEAD_SPEED_ARC, RAMP_AHEAD_SPEED_PIVOT,
                    RAMP_AHEAD_SPEED_STRAIGHT, max_turn_angle)
from shared.mp_manager import ramp_ahead, timer


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

    return 1
