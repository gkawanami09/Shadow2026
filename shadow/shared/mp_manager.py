"""
shared/mp_manager.py — shared scalars (Manager proxies) + time-window helpers.
Ported from Overengineering² Reading Dossier, Section 7 + direct source
  Original source: robot_v.3/Python/main/mp_manager.py
    - Manager().Value declarations       (lines 12-98, in-scope subset only)
    - empty_time_arr / fill_array        (lines 144-152) — verbatim
    - add_time_value / get_time_average  (lines 155-164) — verbatim
Shadow2026 adaptations:
  - Only the mission §4.1 scalar set is declared (no IR/gyro/zone/GUI values).
  - `rotation_y`/gyro fusion (lines 101-208) removed — no IMU on Shadow2026.
  - average_line_point / average_line_angle / last_bottom_point are published
    as shared scalars (in OE² they were process-local time arrays inside
    line_cam; the mission's shared-variable API lists them explicitly).
  - `timer` lives here so every module of a process shares one instance
    (post-fork each process gets its own independent copy, matching OE²'s
    per-module Timer() instances).

IMPORTANT: this module instantiates Manager() at import time (same as OE²).
It must only be imported by main.py and by the two worker processes it forks —
never by the bench tools.
"""

import time
from multiprocessing import Manager

import numpy as np

import config
from shared.managers import ConfigManager, Timer

config_manager = ConfigManager(str(config.CONFIG_INI_PATH))

manager = Manager()

terminate = manager.Value("i", False)
vision_ready = manager.Value("i", False)   # SHADOW: controle espera a visao no boot

min_line_size = manager.Value("i", config.MIN_LINE_SIZE_DEFAULT)

line_angle = manager.Value("i", 0.)
line_angle_y = manager.Value("i", -1)
line_detected = manager.Value("i", False)
line_ahead = manager.Value("i", False)
line_crop = manager.Value("i", config.LINE_CROP_INITIAL)
line_similarity = manager.Value("i", 0.)
line_size = manager.Value("i", 0.)
gap_angle = manager.Value("i", 0.)
gap_center_x = manager.Value("i", -180)
gap_center_y = manager.Value("i", -1)
ramp_ahead = manager.Value("i", False)
red_detected = manager.Value("i", False)
turn_dir = manager.Value("i", "straight")  # "straight"; "left"; "right"; "turn_around"
black_average = manager.Value("i", 0.)

last_bottom_point = manager.Value("i", config.camera_x / 2)
average_line_point = manager.Value("i", config.camera_x / 2)
average_line_angle = manager.Value("i", 0.)

line_status = manager.Value("i", "line_detected")  # "line_detected"; "gap_detected"; "gap_avoid"; "stop"

status = manager.Value("i", "Parado")

timer = Timer()


def empty_time_arr(length: int = 240):
    return np.zeros((length, 2))


def fill_array(value: int, length: int = 240, fill_time: int = 0):
    arr = np.zeros((length, 2))
    arr[fill_time:, 0] = time.perf_counter()
    arr[:, 1] = value
    return arr


def add_time_value(time_value_array, value):
    return np.delete(np.vstack((time_value_array, [time.perf_counter(), value])), 0, axis=0)


def get_time_average(time_value_array, time_range):
    time_value_array = time_value_array[np.where(time_value_array[:, 0] > time.perf_counter() - time_range)]
    if time_value_array.size > 0:
        return np.mean(time_value_array[:, 1])
    else:
        return -1
