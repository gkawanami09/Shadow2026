#!/usr/bin/env python3
"""
tools/color_slider.py — calibração de cores ao vivo com sliders.
Adapted from robot_v.3/Python/debug/color_slider.py (capture backend swapped
to vision/capture.py; saves straight into shadow/config.ini).

Grupos (teclas 1-6):
    1  black_max_normal_top       (teto BGR — faixa 0-40 % da imagem)
    2  black_max_normal_bottom    (teto BGR — faixa 40-100 %)
    3  black_max_ramp_down_top    (teto BGR — troca de rampa)
    4  green_min / green_max      (HSV)
    5  red_min_1 / red_max_1      (HSV, lado baixo do hue)
    6  red_min_2 / red_max_2      (HSV, lado alto do hue)

Teclas:  s = salvar grupo atual no config.ini   |   q = sair
Um bom ajuste: pixels da linha/marcador BRANCOS na máscara, fundo limpo.
Requer ambiente gráfico (monitor ou X forwarding).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from shared.managers import ConfigManager  # noqa: E402
from vision.capture import LineCamera  # noqa: E402

WINDOW = "Shadow2026 - calibracao de cores"

config_manager = ConfigManager(str(config.CONFIG_INI_PATH))


def read_ini(name, fallback):
    value = config_manager.read_variable('color_values_line', name)
    return list(value) if value is not None else list(fallback)


# Cada grupo: (titulo, [(nome_ini, valores, rotulos_trackbar, maximos)], modo)
GROUPS = {
    "1": ("black_max_normal_top", "bgr_ceiling"),
    "2": ("black_max_normal_bottom", "bgr_ceiling"),
    "3": ("black_max_ramp_down_top", "bgr_ceiling"),
    "4": ("green", "hsv_range"),
    "5": ("red_1", "hsv_range"),
    "6": ("red_2", "hsv_range"),
}

HSV_KEYS = {"green": ("green_min", "green_max"),
            "red_1": ("red_min_1", "red_max_1"),
            "red_2": ("red_min_2", "red_max_2")}

HSV_DEFAULTS = {"green_min": config.GREEN_MIN_DEFAULT, "green_max": config.GREEN_MAX_DEFAULT,
                "red_min_1": config.RED_MIN_1_DEFAULT, "red_max_1": config.RED_MAX_1_DEFAULT,
                "red_min_2": config.RED_MIN_2_DEFAULT, "red_max_2": config.RED_MAX_2_DEFAULT}

BGR_DEFAULTS = {"black_max_normal_top": config.BLACK_MAX_NORMAL_TOP_DEFAULT,
                "black_max_normal_bottom": config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT,
                "black_max_ramp_down_top": config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT}


def build_trackbars(group_name, mode):
    cv2.destroyAllWindows()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, config.camera_x * 2, config.camera_y * 2 + 80)

    if mode == "bgr_ceiling":
        values = read_ini(group_name, BGR_DEFAULTS[group_name])
        for label, val, vmax in zip(("B max", "G max", "R max"), values, (255, 255, 255)):
            cv2.createTrackbar(label, WINDOW, int(val), vmax, lambda _v: None)
    else:
        key_min, key_max = HSV_KEYS[group_name]
        vmin = read_ini(key_min, HSV_DEFAULTS[key_min])
        vmax = read_ini(key_max, HSV_DEFAULTS[key_max])
        for label, val, top in (("H min", vmin[0], 180), ("S min", vmin[1], 255), ("V min", vmin[2], 255),
                                ("H max", vmax[0], 180), ("S max", vmax[1], 255), ("V max", vmax[2], 255)):
            cv2.createTrackbar(label, WINDOW, int(val), top, lambda _v: None)


def get_bgr_values():
    return [cv2.getTrackbarPos("B max", WINDOW),
            cv2.getTrackbarPos("G max", WINDOW),
            cv2.getTrackbarPos("R max", WINDOW)]


def get_hsv_values():
    vmin = [cv2.getTrackbarPos("H min", WINDOW),
            cv2.getTrackbarPos("S min", WINDOW),
            cv2.getTrackbarPos("V min", WINDOW)]
    vmax = [cv2.getTrackbarPos("H max", WINDOW),
            cv2.getTrackbarPos("S max", WINDOW),
            cv2.getTrackbarPos("V max", WINDOW)]
    return vmin, vmax


def main():
    print("Abrindo câmera…")
    camera = LineCamera()

    group_key = "1"
    group_name, mode = GROUPS[group_key]
    build_trackbars(group_name, mode)
    print(__doc__)

    try:
        while True:
            frame = camera.get_frame()

            if mode == "bgr_ceiling":
                ceiling = get_bgr_values()
                mask = cv2.inRange(frame, np.array(config.BLACK_MIN_DEFAULT), np.array(ceiling))
            else:
                vmin, vmax = get_hsv_values()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array(vmin), np.array(vmax))

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            stacked = np.vstack((frame, mask_bgr))
            cv2.putText(stacked, f"[{group_key}] {group_name}  (s=salvar, q=sair)",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            cv2.imshow(WINDOW, stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if mode == "bgr_ceiling":
                    values = get_bgr_values()
                    config_manager.write_variable('color_values_line', group_name, values)
                    print(f"salvo: {group_name} = {values}")
                else:
                    key_min, key_max = HSV_KEYS[group_name]
                    vmin, vmax = get_hsv_values()
                    config_manager.write_variable('color_values_line', key_min, vmin)
                    config_manager.write_variable('color_values_line', key_max, vmax)
                    print(f"salvo: {key_min} = {vmin} | {key_max} = {vmax}")
            elif key != 255 and chr(key) in GROUPS:
                group_key = chr(key)
                group_name, mode = GROUPS[group_key]
                build_trackbars(group_name, mode)
                print(f"grupo: {group_name}")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
