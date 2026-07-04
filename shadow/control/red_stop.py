"""
control/red_stop.py — stop_for_red(): the mandatory stop on a red line.
Ported from Overengineering² Reading Dossier, Hotspot 5
  Original source: robot_v.3/Python/main/control.py
    - stop_for_red()   (lines 1024-1039)
Shadow2026 adaptations:
  - The final-second forward nudge `steer(0, 55)` was DROPPED (dossier §9
    item 2: leftover from the old 0-100 duty scale; mission Phase F says the
    robot simply holds the stop for 9 s and, if red is still visible when it
    resumes, it stops again).
  - The GUI run-timer reset at second 5 (`run_start_time = -1`) was dropped
    (no GUI).
  - program_continue() (physical run switch) -> terminate flag.
  - The 1 s sleeps use sleep_steering so PARAR keepalives keep flowing.
"""

from config import wait_time_red
from control.steer import sleep_steering, steer
from shared.mp_manager import status, terminate


def stop_for_red():
    steer()
    for i in range(wait_time_red):
        if terminate.value:
            break

        status.value = f'Parada por vermelho: {wait_time_red - i} s restantes'
        sleep_steering(1)
