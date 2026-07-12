"""
vision/pipeline.py — vision process entry: masks → contours → shared scalars.
Ported from Overengineering² Reading Dossier, Hotspot 1 (+3/4/5 call sites)
  Original source: robot_v.3/Python/main/line_cam.py
    - update_color_values          (lines 68-93, line colors only)
    - loop scaffolding + timers    (lines 527-600)
    - mask block + ramp swap       (lines 618-654) — silver/zone-entry
      threshold branches removed; normal split threshold + ramp_down swap kept
    - black_average + SSIM         (lines 656-663)
    - side/gap blanking            (lines 676-682; obstacle rectangles dropped)
    - morphology                   (lines 684-703; position_entry_2 variant dropped)
    - contours + min_line_size     (lines 714-721)
    - red / green / latch          (lines 723-758 via vision/red.py, vision/green.py)
    - line + gap + averages        (lines 760-816 via vision/line.py, vision/gap.py)
Shadow2026 adaptations:
  - Capture via vision/capture.py (Picamera2 640×480 → 448×252 BGR).
  - No silver AI, no zone loop, no calibration-UI branches, no `objective`
    (this robot only follows the line).
  # IMU_REPLACEMENT: rotation_y is gone — do_inference/ramp threshold logic
    that depended on it never existed here; line_crop ramp variants dropped
    inside vision/green.py.
  - Debug frame goes to multiprocessing.shared_memory (created by main.py)
    ONLY when --debug is on; annotations mirror OE²'s GUI drawings.
  - Publishes vision_ready after the first processed frame (control waits).
"""

import time
from multiprocessing import shared_memory

import cv2
import numpy as np
from skimage.metrics import structural_similarity

import config
from config import (BLACK_AVG_SIDE_MASK, BOTTOM_CENTER_MIN_Y, DEBUG_SHM_NAME, RAMP_SWAP_MARGIN,
                    RAMP_SWAP_TRIGGER, SIMILARITY_CHECK_EVERY, VISION_MAX_FRAMES,
                    camera_x, camera_y)
from shared.mp_manager import (add_time_value, average_line_angle, average_line_point,
                               black_average, config_manager, empty_time_arr,
                               get_time_average, last_bottom_point, line_angle,
                               line_angle_y, line_crop, line_detected,
                               line_geometry_valid, line_heading_error,
                               line_lateral_error,
                               line_similarity, line_size, line_status,
                               min_line_size, ramp_ahead, red_detected, status,
                               terminate, timer, turn_dir, vision_ready)
from vision import line as line_module
from vision.capture import LineCamera
from vision.gap import apply_gap_avoid_mask, publish_gap_geometry, reset_gap_values
from vision.green import check_green, latch_turn_direction
from vision.line import calculate_angle, determine_correct_line
from vision.red import check_contour_size

# Cores carregadas do config.ini (fallback: valores do config.py)
black_min = np.array(config.BLACK_MIN_DEFAULT)
black_max_normal_top = np.array(config.BLACK_MAX_NORMAL_TOP_DEFAULT)
black_max_normal_bottom = np.array(config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT)
black_max_ramp_down_top = np.array(config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT)
green_min = np.array(config.GREEN_MIN_DEFAULT)
green_max = np.array(config.GREEN_MAX_DEFAULT)
red_min_1 = np.array(config.RED_MIN_1_DEFAULT)
red_max_1 = np.array(config.RED_MAX_1_DEFAULT)
red_min_2 = np.array(config.RED_MIN_2_DEFAULT)
red_max_2 = np.array(config.RED_MAX_2_DEFAULT)


def update_color_values():
    global black_max_normal_top, black_max_normal_bottom, black_max_ramp_down_top, \
        green_min, green_max, red_min_1, red_max_1, red_min_2, red_max_2

    def read(name, fallback):
        value = config_manager.read_variable('color_values_line', name)
        return np.array(value) if value is not None else np.array(fallback)

    black_max_normal_top = read('black_max_normal_top', config.BLACK_MAX_NORMAL_TOP_DEFAULT)
    black_max_normal_bottom = read('black_max_normal_bottom', config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT)
    black_max_ramp_down_top = read('black_max_ramp_down_top', config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT)

    green_min = read('green_min', config.GREEN_MIN_DEFAULT)
    green_max = read('green_max', config.GREEN_MAX_DEFAULT)

    red_min_1 = read('red_min_1', config.RED_MIN_1_DEFAULT)
    red_max_1 = read('red_max_1', config.RED_MAX_1_DEFAULT)
    red_min_2 = read('red_min_2', config.RED_MIN_2_DEFAULT)
    red_max_2 = read('red_max_2', config.RED_MAX_2_DEFAULT)


def vision_loop(debug=False):
    line_module.init_tracker()

    bottom_y = camera_y

    time_line_angle = empty_time_arr()
    time_turn_direction = empty_time_arr()
    time_last_bottom_point_x = empty_time_arr()
    time_last_average_line_point = empty_time_arr()

    camera = LineCamera()

    shm = None
    shm_array = None
    if debug:
        shm = shared_memory.SharedMemory(name=DEBUG_SHM_NAME)
        shm_array = np.ndarray((camera_y, camera_x, 3), dtype=np.uint8, buffer=shm.buf)

    update_color_values()

    # Kernal for noise reduction
    kernal = np.ones((3, 3), np.uint8)

    # FPS counter / limiter (OE²: max_frames_line = 90; a captura a 40 fps
    # bloqueia por frame, então na prática todo frame é processado)
    fps_time = time.perf_counter()
    counter = 0
    fps = 0
    fps_limit_time = time.perf_counter()

    last_image = np.zeros((camera_y, camera_x), dtype=np.uint8)

    timer.set_timer("multiple_bottom", .05)
    timer.set_timer("multiple_side_l", .05)
    timer.set_timer("multiple_side_r", .05)
    timer.set_timer("right_marker", .05)
    timer.set_timer("left_marker", .05)

    check_similarity_counter = 0

    try:
        while not terminate.value:
            cv2_img = camera.get_frame()

            if time.perf_counter() - fps_limit_time <= 1 / VISION_MAX_FRAMES:
                continue
            fps_limit_time = time.perf_counter()

            hsv_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
            green_image = cv2.inRange(hsv_image, green_min, green_max)
            red_image = cv2.inRange(hsv_image, red_min_1, red_max_1) + cv2.inRange(hsv_image, red_min_2, red_max_2)

            black_image = cv2.inRange(cv2_img, black_min, black_max_normal_bottom)
            black_image[0:int(camera_y * .4), 0:camera_x] = cv2.inRange(cv2_img, black_min, black_max_normal_top)[0:int(camera_y * .4), 0:camera_x]

            black_image -= green_image
            black_image[black_image < 2] = 0

            # Change black_max to black_max_ramp_down_top if the top section of the image is too dark
            dark_ahead = False
            black_mean = round(np.mean(black_image[0:int(camera_y * .25), 0:camera_x]), 2)
            if black_mean > RAMP_SWAP_TRIGGER:
                black_image_2 = cv2.inRange(cv2_img, black_min, black_max_ramp_down_top)
                black_image_2 -= green_image
                black_image_2[black_image_2 < 2] = 0

                black_mean_2 = round(np.mean(black_image_2[0:int(camera_y * .25), 0:camera_x]), 2)

                if black_mean_2 + RAMP_SWAP_MARGIN < black_mean:
                    cv2.circle(cv2_img, (10, 10), 5, (0, 0, 0), -1, cv2.LINE_AA)
                    black_image[0:int(camera_y * .4), 0:camera_x] = black_image_2[0:int(camera_y * .4), 0:camera_x]
                    dark_ahead = True

            ramp_ahead.value = dark_ahead

            black_average.value = np.mean(black_image[:])

            # SSIM entre máscaras consecutivas (detecção de robô preso)
            if check_similarity_counter >= SIMILARITY_CHECK_EVERY:
                line_similarity.value = structural_similarity(black_image, last_image)
                last_image = black_image.copy()
                check_similarity_counter = 0
            check_similarity_counter += 1

            # Cut out certain parts of the image
            if line_status.value == "gap_avoid":
                apply_gap_avoid_mask(black_image)

            if bottom_y < camera_y * .95 and black_average.value < BLACK_AVG_SIDE_MASK and line_status.value == "line_detected":
                cv2.rectangle(black_image, (0, 0), (int(camera_x * .25), camera_y), 0, -1)
                cv2.rectangle(black_image, (int(camera_x * .75), 0), (camera_x, camera_y), 0, -1)

            # Noise reduction
            if line_status.value == "gap_avoid":
                black_image = cv2.erode(black_image, kernal, iterations=5)
                black_image = cv2.dilate(black_image, kernal, iterations=8)
            else:
                black_image = cv2.erode(black_image, kernal, iterations=5)
                black_image = cv2.dilate(black_image, kernal, iterations=17)
                black_image = cv2.erode(black_image, kernal, iterations=9)

            green_image = cv2.erode(green_image, kernal, iterations=1)
            green_image = cv2.dilate(green_image, kernal, iterations=11)
            green_image = cv2.erode(green_image, kernal, iterations=9)

            red_image = cv2.erode(red_image, kernal, iterations=1)
            red_image = cv2.dilate(red_image, kernal, iterations=11)
            red_image = cv2.erode(red_image, kernal, iterations=9)

            # Find contours in the image
            contours_grn, _ = cv2.findContours(green_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(red_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_blk, _ = cv2.findContours(black_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            blk_contour_area = np.array([cv2.contourArea(i) for i in contours_blk])
            blk_mask = blk_contour_area > min_line_size.value
            contours_blk = [c for c, m in zip(contours_blk, blk_mask) if m]

            # Check for red line
            red_detected.value = check_contour_size(contours_red, "red", debug_img=cv2_img if debug else None)

            # Check for green turn signs
            if len(contours_grn) > 0:
                turn_direction = check_green(
                    contours_grn, black_image, debug_img=cv2_img if debug else None)
            else:
                turn_direction = "straight"

            time_turn_direction = latch_turn_direction(turn_direction, time_turn_direction)

            # Determine the correct line
            if len(contours_blk) > 0:
                line_detected.value = True
                blackline, black_line_crop = determine_correct_line(contours_blk)
                line_size.value = cv2.contourArea(blackline)

                # Calculate the gap angle
                if line_status.value == "gap_detected":
                    publish_gap_geometry(blackline, cv2_img if debug else None)
                else:
                    reset_gap_values()

                # Calculate the angle to turn
                last_bottom_point_x = float(get_time_average(time_last_bottom_point_x, .15))
                last_average_line_point = float(get_time_average(time_last_average_line_point, .15))

                line_angle.value, poi, bottom_point = calculate_angle(
                    blackline, black_line_crop,
                    float(get_time_average(time_line_angle, .3)),
                    turn_dir.value, last_bottom_point_x, last_average_line_point)
                line_angle_y.value = int(poi[1])

                dy = float(bottom_point[1] - poi[1])
                geometry_valid = (bottom_point[1] >= camera_y * BOTTOM_CENTER_MIN_Y
                                  and dy > 5)
                if geometry_valid:
                    line_lateral_error.value = float(np.clip(
                        (bottom_point[0] - camera_x / 2) / (camera_x / 2), -1, 1))
                    line_heading_error.value = float(np.clip(
                        np.arctan2(poi[0] - bottom_point[0], dy) / (np.pi / 2),
                        -1, 1))
                    line_geometry_valid.value = True
                else:
                    line_lateral_error.value = 0.0
                    line_heading_error.value = 0.0
                    line_geometry_valid.value = False


                time_line_angle = add_time_value(time_line_angle, line_angle.value)
                time_last_bottom_point_x = add_time_value(time_last_bottom_point_x, bottom_point[0])

                # projeção do vetor bottom→POI até y=0 (line_cam.py 795-801)
                if bottom_point[0] != poi[0] and bottom_point[1] != poi[1]:
                    slope = (bottom_point[1] - poi[1]) / (bottom_point[0] - poi[0])
                    x = min(max(poi[0] + (0 - poi[1]) / slope, 0), camera_x)
                else:
                    x = poi[0]

                time_last_average_line_point = add_time_value(time_last_average_line_point, x)

                bottom_y = bottom_point[1]

                # publica os escalares da API compartilhada (mission §4.1)
                last_bottom_point.value = bottom_point[0]
                average_line_point.value = last_average_line_point
                average_line_angle.value = float(get_time_average(time_line_angle, .3))

                if debug:
                    cv2.drawContours(cv2_img, [blackline], -1, (255, 0, 0), 2)
                    cv2.circle(cv2_img, (int(last_average_line_point), 0), 5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (int(poi[0]), int(poi[1])), 5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (int(bottom_point[0]), int(bottom_point[1])), 5, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (camera_x // 2, camera_y - 4),
                               5, (255, 0, 0), -1, cv2.LINE_AA)
                    if line_geometry_valid.value:
                        cv2.putText(
                            cv2_img,
                            f'mec lat={line_lateral_error.value:+.2f} '
                            f'head={line_heading_error.value:+.2f}',
                            (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .4,
                            (255, 0, 255), 1)

            else:
                line_detected.value = False
                line_geometry_valid.value = False
                line_lateral_error.value = 0.0
                line_heading_error.value = 0.0
                line_angle.value = 0
                line_size.value = 0
                line_angle_y.value = -1
                reset_gap_values()

            if not vision_ready.value:
                vision_ready.value = True
                print("[visão] primeiro frame processado — pipeline ativo")

            # FPS
            counter += 1
            if time.perf_counter() - fps_time > 1:
                fps = int(counter / (time.perf_counter() - fps_time))
                fps_time = time.perf_counter()
                counter = 0

            if debug:
                cv2.putText(cv2_img, f"{fps} fps  ang={line_angle.value}  {line_status.value}",
                            (5, camera_y - 8), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.putText(cv2_img, str(status.value), (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                shm_array[:] = cv2_img

    finally:
        camera.close()
        if shm is not None:
            shm.close()
