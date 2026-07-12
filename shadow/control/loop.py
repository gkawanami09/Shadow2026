"""
control/loop.py — control process entry: the line_status state machine.
Ported from Overengineering² Reading Dossier, Hotspots 2/3/5 (dispatch)
  Original source: robot_v.3/Python/main/control.py
    - control_loop scaffolding + 60 it/s cap  (lines 1731-1778, 2296-2298)
    - gap trigger                              (lines 1885-1888)
    - red trigger                              (lines 1890-1891)
    - "line_detected" branch                   (lines 1913-1929)
    - "stop" branch                            (lines 1932-1935)
    - "gap_detected" dispatch                  (lines 1938-1949)
    - "gap_avoid" branch                       (lines 1952-1972)
Shadow2026 adaptations:
  - GPIO/servo/LED/button setup replaced by the SPEC 01 serial link (Arduino
    class); this process is the only owner of the serial port.
  # IMU_REPLACEMENT: `rotation_y.value == "none"` in the gap trigger is
    always true and was dropped; stuck cooldown keeps the flat 4 s value.
  - silver/obstacle/seesaw/zone states and predicates removed (out of scope);
    the state machine keeps exactly 4 states:
    "line_detected" | "gap_detected" | "gap_avoid" | "stop".
  - program_continue() (run switch) -> terminate flag; on shutdown the finally
    block sends PARAR and closes the port.
  - Waits for the vision process (vision_ready) before entering the state
    machine, so a dark camera at boot doesn't trigger gap handling.
"""

import time

from config import (CONTROL_MAX_ITERATIONS, GAP_AVOID_RETREAT_TIME, GAP_AVOID_SPEED,
                    GAP_AVOID_TIMEOUT, GAP_MIN_LINE_SIZE_RETREAT,
                    GAP_MISSING_CONFIRM_TIME, FRONT_ANCHOR_READY_X_PX,
                    FRONT_ANCHOR_READY_Y, FRONT_ANCHOR_START_ANGLE,
                    MIN_LINE_SIZE_DEFAULT, VISION_READY_TIMEOUT, camera_x, camera_y)
from control.gap_orient import drive_back_until_line, orientate_gap
from control.red_stop import stop_for_red
from control.speed import get_speed
from control.steer import init_steering, sleep_steering, steer
from control.turn_around import turn_around
from serial_link.arduino import Arduino
from shared.mp_manager import (add_time_value, empty_time_arr, last_bottom_point,
                               line_ahead, line_angle, line_bottom_y, line_detected,
                               line_status, min_line_size,
                               ramp_ahead, red_detected, status, terminate,
                               timer, turn_dir, vision_ready)


def control_loop():
    arduino = Arduino()
    init_steering(arduino)
    steer()  # motores parados desde o inicio

    last_turn_dir = "l"

    time_last_angles = empty_time_arr()

    timer.set_timer("ramp_ahead", .01)
    timer.set_timer("stuck_detected", .01)
    timer.set_timer("stuck_cooldown", 5)

    # espera a visao publicar o primeiro frame processado
    wait_start = time.perf_counter()
    while not vision_ready.value and not terminate.value:
        if time.perf_counter() - wait_start > VISION_READY_TIMEOUT:
            print("[controle] AVISO: visão não ficou pronta em "
                  f"{VISION_READY_TIMEOUT} s — seguindo mesmo assim")
            break
        arduino.refresh()
        time.sleep(.05)

    line_status.value = "line_detected"
    status.value = "Shadow2026 pronto — aguardando linha"
    print("Shadow2026 ready — awaiting line")

    iteration_limit_time = time.perf_counter()
    max_iterations = CONTROL_MAX_ITERATIONS
    line_missing_since = None

    try:
        while not terminate.value:

            # detected line on last frame
            if line_status.value == "line_detected":

                # IMU_REPLACEMENT: clausula `rotation_y == "none"` removida
                if not line_detected.value and not line_ahead.value and not ramp_ahead.value:
                    if line_missing_since is None:
                        line_missing_since = time.monotonic()
                    elif time.monotonic() - line_missing_since >= GAP_MISSING_CONFIRM_TIME:
                        line_status.value = "gap_detected"
                        line_missing_since = None
                else:
                    line_missing_since = None

                if red_detected.value:
                    line_status.value = "stop"

            # still line detected
            if line_status.value == "line_detected":
                if turn_dir.value == "turn_around":
                    status.value = f'Girando 180° para a {"direita" if last_turn_dir == "r" else "esquerda"}'

                    last_turn_dir = turn_around(last_turn_dir)
                    continue

                status.value = 'Seguindo Linha'

                command_angle = line_angle.value
                anchor_ready = (
                    line_bottom_y.value >= camera_y * FRONT_ANCHOR_READY_Y
                    and abs(last_bottom_point.value - camera_x / 2)
                    <= FRONT_ANCHOR_READY_X_PX
                )
                # Enquanto a linha/canto ainda nao chegou a bolinha, mantem um
                # arco para frente. O pivô traseiro so e liberado quando o
                # preto realmente ocupa a regiao inferior central.
                if abs(command_angle) > FRONT_ANCHOR_START_ANGLE and not anchor_ready:
                    command_angle = (FRONT_ANCHOR_START_ANGLE
                                     if command_angle > 0
                                     else -FRONT_ANCHOR_START_ANGLE)

                steer(command_angle, get_speed(command_angle))

                time_last_angles = add_time_value(time_last_angles, line_angle.value)
            elif line_status.value == "stop":
                stop_for_red()
                line_status.value = "line_detected"
                continue

            elif line_status.value == "gap_detected":
                verified_gap = orientate_gap()

                if verified_gap:
                    timer.set_timer("gap_avoid", GAP_AVOID_TIMEOUT)
                else:
                    line_status.value = "line_detected"
                    min_line_size.value = MIN_LINE_SIZE_DEFAULT
                    sleep_steering(.1)

                timer.set_timer("stuck_cooldown", 4)
                continue

            elif line_status.value == "gap_avoid":
                status.value = 'Cruzando o gap'

                if line_detected.value:
                    min_line_size.value = MIN_LINE_SIZE_DEFAULT
                    line_status.value = "line_detected"
                    timer.set_timer("stuck_cooldown", 4)
                    continue
                else:
                    steer(0, GAP_AVOID_SPEED)

                if timer.get_timer("gap_avoid"):
                    min_line_size.value = GAP_MIN_LINE_SIZE_RETREAT
                    steer(200, GAP_AVOID_SPEED)
                    sleep_steering(GAP_AVOID_RETREAT_TIME)
                    drive_back_until_line(.3, GAP_AVOID_SPEED)

                    line_status.value = "line_detected"
                    sleep_steering(.1)
                    timer.set_timer("stuck_cooldown", 4)
                    continue

            # limitador de 60 it/s (identico ao OE², control.py:2296-2298)
            if time.perf_counter() - iteration_limit_time < 1 / max_iterations:
                time.sleep(abs(1 / max_iterations - (time.perf_counter() - iteration_limit_time)))
            iteration_limit_time = time.perf_counter()

    finally:
        status.value = "Parado"
        try:
            steer()  # PARAR
        finally:
            arduino.close()
