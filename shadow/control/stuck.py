"""
control/stuck.py — avoid_stuck(): open-loop recovery when the camera image freezes.
Ported from Overengineering² Reading Dossier, Hotspot 2 (stuck recovery)
  Original source: robot_v.3/Python/main/control.py
    - avoid_stuck()   (lines 980-1008)
Shadow2026 adaptations:
  # IMU_REPLACEMENT: rotation_y == "none" is always true, so the big-angle
  jiggle keeps its original gate minus the rotation check; the ramp_down
  branch (lines 997-1002) is dead and was removed. The stuck_detected timer
  keeps the flat value (.85 s; the 1.2 s ramp_up variant is dead).
  - time.sleep -> sleep_steering (Uno watchdog keepalive).
"""

from control.steer import sleep_steering, steer
from shared.mp_manager import line_angle, line_status, status, timer


def avoid_stuck():
    status.value = 'Preso na linha — recuperando'

    angle = line_angle.value

    if line_status.value == "line_detected" and abs(angle) > 120:
        steer()
        sleep_steering(1)
        steer(180 if angle < 0 else -180, .7)
        sleep_steering(.35)
        steer(0, .7)
        sleep_steering(.45)
        steer(-180 if angle < 0 else 180, .7)
        sleep_steering(.45)
        steer(200, .7)
        sleep_steering(.5)

    else:
        steer()
        sleep_steering(.5)

    timer.set_timer("stuck_detected", .85)
