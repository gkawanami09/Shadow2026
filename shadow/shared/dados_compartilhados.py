"""Guarda os valores compartilhados pelos processos de visão e controle."""

import time
from multiprocessing import Manager

import numpy as np

import config
from shared.gerenciadores import ConfigManager, Timer

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
line_size = manager.Value("i", 0.)
gap_angle = manager.Value("i", 0.)
gap_center_x = manager.Value("i", -180)
gap_center_y = manager.Value("i", -1)
gap_end_width = manager.Value("i", -1)
ramp_ahead = manager.Value("i", False)
red_detected = manager.Value("i", False)
turn_dir = manager.Value("i", "straight")  # "straight"; "left"; "right"; "turn_around"
black_average = manager.Value("i", 0.)

last_bottom_point = manager.Value("i", config.camera_x / 2)

line_status = manager.Value("i", "line_detected")  # "line_detected"; "gap_detected"; "gap_avoid"; "stop"

status = manager.Value("i", "Parado")

timer = Timer()


def empty_time_arr(length: int = 240):
    return np.zeros((length, 2))


def add_time_value(time_value_array, value):
    return np.delete(np.vstack((time_value_array, [time.perf_counter(), value])), 0, axis=0)


def get_time_average(time_value_array, time_range):
    time_value_array = time_value_array[np.where(time_value_array[:, 0] > time.perf_counter() - time_range)]
    if time_value_array.size > 0:
        return np.mean(time_value_array[:, 1])
    else:
        return -1
