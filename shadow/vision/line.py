"""
vision/line.py — line detection (contour choice → POIs → steering error).
Ported from Overengineering² Reading Dossier, Hotspot 1
  Original source: robot_v.3/Python/main/line_cam.py
    - determine_correct_line   (lines 198-240) — verbatim (GUI drawing dropped)
    - calculate_angle_numba    (lines 243-340) — verbatim, @njit(cache=True) kept
    - calculate_angle          (lines 343-424) — verbatim minus the `entry`
      (zone-entry positioning) branch, which is out of scope
Shadow2026 adaptations:
  - Runs on 448×252 BGR frames captured by Picamera2 (vision/capture.py).
  - No silver / zone-entry paths.
  - x_last / y_last / multiple_bottom_side stay module-level globals exactly
    like OE²; init_tracker() must be called once by the vision process.
Output: line_angle = int((poi_x − 224) / 224 × 180) ∈ [−180, +180] — a scaled
horizontal pixel offset, positive = steer right.
"""

import math

import cv2
import numpy as np
from numba import njit

from config import (CAMERA_GROUND_DEPTH_CM, CAMERA_GROUND_WIDTH_CM,
                    CAMERA_TO_ROBOT_CENTER_CM, FRONT_CAMERA_CONTROL,
                    FRONT_HEADING_GAIN, FRONT_LATERAL_GAIN, camera_x,
                    camera_y)
from shared.mp_manager import line_crop, timer, turn_dir

x_last = camera_x / 2
y_last = camera_y / 2
multiple_bottom_side = camera_x / 2
front_camera_debug = None


def init_tracker():
    global x_last, y_last, multiple_bottom_side, front_camera_debug
    x_last = camera_x / 2
    y_last = camera_y / 2
    multiple_bottom_side = camera_x / 2
    front_camera_debug = None


def get_front_camera_debug():
    return front_camera_debug


def _pixel_to_ground(point):
    x_cm = (float(point[0]) - camera_x / 2) / camera_x * CAMERA_GROUND_WIDTH_CM
    y_cm = (camera_y - float(point[1])) / camera_y * CAMERA_GROUND_DEPTH_CM
    return x_cm, y_cm


def _front_camera_angle(final_poi, bottom_point, legacy_angle):
    """Combina posicao proxima e direcao da linha a partir do centro do robo."""
    global front_camera_debug

    front_camera_debug = None
    if not FRONT_CAMERA_CONTROL:
        return legacy_angle

    near_x, near_y = _pixel_to_ground(bottom_point)
    far_x, far_y = _pixel_to_ground(final_poi)
    forward_delta = far_y - near_y
    if forward_delta <= .05:
        return legacy_angle

    lateral_angle = math.degrees(math.atan2(
        near_x, CAMERA_TO_ROBOT_CENTER_CM + near_y))
    heading_angle = math.degrees(math.atan2(far_x - near_x, forward_delta))
    combined = int(round(np.clip(
        FRONT_LATERAL_GAIN * lateral_angle
        + FRONT_HEADING_GAIN * heading_angle,
        -180, 180)))

    front_camera_debug = {
        "near": (int(bottom_point[0]), int(bottom_point[1])),
        "far": (int(final_poi[0]), int(final_poi[1])),
        "lateral": lateral_angle,
        "heading": heading_angle,
        "legacy": legacy_angle,
        "combined": combined,
    }
    return combined


def determine_correct_line(contours_blk):
    global x_last, y_last
    candidates = np.zeros((len(contours_blk), 5), dtype=np.int32)
    off_bottom = 0

    for i, contour in enumerate(contours_blk):
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        box = box[box[:, 1].argsort()[::-1]]  # Sort them by their y values and reverse
        bottom_y = box[0][1]
        y_mean = (np.clip(box[0][1], 0, camera_y) + np.clip(box[3][1], 0, camera_y)) / 2

        if box[0][1] >= (camera_y * 0.75):
            off_bottom += 1

        box = box[box[:, 0].argsort()]
        x_mean = (np.clip(box[0][0], 0, camera_x) + np.clip(box[3][0], 0, camera_x)) / 2
        x_y_distance = abs(x_last - x_mean) + abs(y_last - y_mean)  # Distance between the last x/y and current x/y

        candidates[i] = i, bottom_y, x_y_distance, x_mean, y_mean

    if off_bottom < 2:
        candidates = candidates[candidates[:, 1].argsort()[::-1]]  # Sort candidates by their bottom_y
    else:
        off_bottom_candidates = candidates[np.where(candidates[:, 1] >= (camera_y * 0.75))]
        candidates = off_bottom_candidates[off_bottom_candidates[:, 2].argsort()]

    if turn_dir.value == "left":
        x_last = np.clip(candidates[0][3] - 150, 0, camera_x)
    elif turn_dir.value == "right":
        x_last = np.clip(candidates[0][3] + 150, 0, camera_x)
    else:
        x_last = candidates[0][3]

    y_last = candidates[0][4]
    blackline = contours_blk[candidates[0][0]]
    blackline_crop = blackline[np.where(blackline[:, 0, 1] > camera_y * line_crop.value)]

    return blackline, blackline_crop


@njit(cache=True)
def calculate_angle_numba(blackline, blackline_crop, last_bottom_point, average_line_point):
    max_gap = 1
    max_line_width = camera_x * .19

    poi_no_crop = np.zeros((4, 2), dtype=np.int32)  # [t, l, r, b]

    # Top without crop
    blackline_y_min = np.amin(blackline[:, :, 1])
    blackline_top = blackline[np.where(blackline[:, 0, 1] == blackline_y_min)][:, :, 0]

    blackline_top = blackline_top[blackline_top[:, 0].argsort()]
    blackline_top_gap_fill = (blackline_top + max_gap + 1)[:-1]

    blackline_gap_mask = blackline_top_gap_fill < blackline_top[1:]

    top_mean = (int(np.mean(blackline_top)), blackline_y_min)

    if np.sum(blackline_gap_mask) == 1:
        gap_index = np.where(blackline_gap_mask)[0][0]

        if blackline_top[:gap_index].size > 0 and blackline_top[gap_index:].size > 0:
            top_mean_l = int(np.mean(blackline_top[:gap_index]))
            top_mean_r = int(np.mean(blackline_top[gap_index:]))

            top_mean = (top_mean_l, blackline_y_min) if np.abs(top_mean_l - average_line_point) < np.abs(top_mean_r - average_line_point) else (top_mean_r, blackline_y_min)

    poi_no_crop[0] = [top_mean[0], top_mean[1]]

    # Bottom without crop
    blackline_y_max = np.amax(blackline[:, :, 1])
    blackline_bottom = blackline[np.where(blackline[:, 0, 1] == blackline_y_max)][:, :, 0]
    blackline_bottom = blackline_bottom[blackline_bottom[:, 0].argsort()]
    blackline_bottom_gap_fill = (blackline_bottom + max_gap + 1)[:-1]

    blackline_gap_mask = blackline_bottom_gap_fill < blackline_bottom[1:]

    bottom_point_mean = (int(np.mean(blackline_bottom)), blackline_y_max)

    if np.sum(blackline_gap_mask) == 1:
        gap_index = np.where(blackline_gap_mask)[0][0]

        if blackline_bottom[:gap_index].size > 0 and blackline_bottom[gap_index:].size > 0:
            bottom_mean_l = int(np.mean(blackline_bottom[:gap_index]))
            bottom_mean_r = int(np.mean(blackline_bottom[gap_index:]))

            if np.abs(bottom_mean_l - bottom_mean_r) > 80:
                if np.abs(bottom_mean_l - last_bottom_point) < np.abs(bottom_mean_r - last_bottom_point):
                    bottom_point_mean = (bottom_mean_l, blackline_y_max)
                    bottom_mean = (bottom_mean_r, blackline_y_max)
                else:
                    bottom_point_mean = (bottom_mean_r, blackline_y_max)
                    bottom_mean = (bottom_mean_l, blackline_y_max)

                poi_no_crop[3] = [bottom_mean[0], bottom_mean[1]]

    bottom_point = [bottom_point_mean[0], bottom_point_mean[1]]

    # Left without crop
    blackline_x_min = np.amin(blackline[:, :, 0])
    blackline_left = blackline[np.where(blackline[:, 0, 0] == blackline_x_min)]
    left_mean = (blackline_x_min, int(np.mean(blackline_left[:, :, 1])))
    poi_no_crop[1] = [left_mean[0], left_mean[1]]

    # Right without crop
    blackline_x_max = np.amax(blackline[:, :, 0])
    blackline_right = blackline[np.where(blackline[:, 0, 0] == blackline_x_max)]
    right_mean = (blackline_x_max, int(np.mean(blackline_right[:, :, 1])))
    poi_no_crop[2] = [right_mean[0], right_mean[1]]

    poi = np.zeros((3, 2), dtype=np.int32)  # [t, l, r]
    is_crop = blackline_crop.size > 0

    max_black_top = False

    if is_crop:
        # Top
        blackline_y_min = np.amin(blackline_crop[:, :, 1])
        blackline_top = blackline_crop[np.where(blackline_crop[:, 0, 1] == blackline_y_min)][:, :, 0]
        top_mean = (int(np.mean(blackline_top)), blackline_y_min)
        poi[0] = [top_mean[0], top_mean[1]]

        blackline_top = blackline_top[blackline_top[:, 0].argsort()]
        max_black_top = bool(np.abs(blackline_top[0] - blackline_top[-1]) > max_line_width)

        # Left
        blackline_x_min = np.amin(blackline_crop[:, :, 0])
        blackline_left = blackline_crop[np.where(blackline_crop[:, 0, 0] == blackline_x_min)]
        left_mean = (blackline_x_min, int(np.mean(blackline_left[:, :, 1])))
        poi[1] = [left_mean[0], left_mean[1]]

        # Right
        blackline_x_max = np.amax(blackline_crop[:, :, 0])
        blackline_right = blackline_crop[np.where(blackline_crop[:, 0, 0] == blackline_x_max)]
        right_mean = (blackline_x_max, int(np.mean(blackline_right[:, :, 1])))
        poi[2] = [right_mean[0], right_mean[1]]

    return poi, poi_no_crop, is_crop, max_black_top, bottom_point


def calculate_angle(blackline, blackline_crop, average_line_angle, turn_direction, last_bottom_point, average_line_point):
    global multiple_bottom_side

    poi, poi_no_crop, is_crop, max_black_top, bottom_point = calculate_angle_numba(blackline, blackline_crop, last_bottom_point, average_line_point)

    black_top = poi_no_crop[0][1] < camera_y * .1

    multiple_bottom = not (poi_no_crop[3][0] == 0 and poi_no_crop[3][1] == 0)

    black_l_high = poi_no_crop[1][1] < camera_y * .5
    black_r_high = poi_no_crop[2][1] < camera_y * .5

    if not timer.get_timer("multiple_bottom"):
        final_poi = [multiple_bottom_side, camera_y]

    elif turn_direction in ["left", "right"]:
        index = 1 if turn_direction == "left" else 2
        final_poi = poi[index] if is_crop else poi_no_crop[index]

    else:
        if black_top:
            final_poi = poi[0] if is_crop and not max_black_top else poi_no_crop[0]

            if (poi_no_crop[1][0] < camera_x * 0.02 and poi_no_crop[1][1] > camera_y * (line_crop.value * .75)) or (poi_no_crop[2][0] > camera_x * 0.98 and poi_no_crop[2][1] > camera_y * (line_crop.value * .75)):
                final_poi = poi_no_crop[0]

                if black_l_high or black_r_high:
                    near_high_index = 0
                    if black_l_high and not black_r_high:
                        near_high_index = 1
                    elif not black_l_high and black_r_high:
                        near_high_index = 2
                    elif black_l_high and black_r_high:
                        if np.abs(poi_no_crop[1][0] - average_line_point) < np.abs(poi_no_crop[2][0] - average_line_point):
                            near_high_index = 1
                        else:
                            near_high_index = 2

                    if np.abs(poi_no_crop[near_high_index][0] - average_line_point) < np.abs(poi_no_crop[0][0] - average_line_point):
                        final_poi = poi_no_crop[near_high_index]

        else:
            final_poi = poi[0] if is_crop else poi_no_crop[0]

            if poi_no_crop[1][0] < camera_x * 0.02 and poi_no_crop[2][0] > camera_x * 0.98 and timer.get_timer("multiple_side_r") and timer.get_timer("multiple_side_l"):
                if average_line_angle >= 0:
                    index = 2
                    timer.set_timer("multiple_side_r", .6)
                else:
                    index = 1
                    timer.set_timer("multiple_side_l", .6)
                final_poi = poi[index] if is_crop else poi_no_crop[index]

            elif not timer.get_timer("multiple_side_l"):
                final_poi = poi[1] if is_crop else poi_no_crop[1]

            elif not timer.get_timer("multiple_side_r"):
                final_poi = poi[2] if is_crop else poi_no_crop[2]

            elif poi_no_crop[1][0] < camera_x * 0.02:
                final_poi = poi[1] if is_crop else poi_no_crop[1]

            elif poi_no_crop[2][0] > camera_x * 0.98:
                final_poi = poi[2] if is_crop else poi_no_crop[2]

            elif multiple_bottom and timer.get_timer("multiple_bottom"):
                if poi_no_crop[3][0] < bottom_point[0]:
                    final_poi = [0, camera_y]
                    multiple_bottom_side = 0
                else:
                    final_poi = [camera_x, camera_y]
                    multiple_bottom_side = camera_x
                timer.set_timer("multiple_bottom", .6)

    legacy_angle = int((final_poi[0] - camera_x / 2) / (camera_x / 2) * 180)
    line_angle = _front_camera_angle(final_poi, bottom_point, legacy_angle)
    return line_angle, final_poi, bottom_point
