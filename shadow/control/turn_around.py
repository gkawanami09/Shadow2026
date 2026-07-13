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

import time

from config import (T_180, T_180_CONFIRM_TIME, T_180_EXIT_ANGLE,
                    T_180_EXIT_BOTTOM_PX, T_180_SEARCH_SPEED,
                    T_180_SEARCH_TIMEOUT, T_180_SPEED, T_180_TEST_STOP,
                    TURN_AROUND_PREROLL, TURN_AROUND_REVERSE,
                    TURN_AROUND_REVERSE_EXTRA, TURN_AROUND_SMALL_LINE,
                    camera_x)
from control.steer import sleep_steering, steer
from shared.mp_manager import (last_bottom_point, line_angle, line_detected,
                               line_size, status, terminate, timer)


def turn_around(last_turn_dir):
    """Executes the 180° and returns the NEXT turn direction ("l"/"r")."""
    # avanca por cima do marcador duplo
    steer(0, .7)
    sleep_steering(TURN_AROUND_PREROLL)

    # IMU_REPLACEMENT: pivot temporizado no lugar do giro por giroscopio
    steer(180 if last_turn_dir == "r" else -180, T_180_SPEED)
    sleep_steering(T_180)
    steer()

    # Modo temporario de afericao: isola somente o giro cronometrado. Mantem
    # PARAR ate o operador encerrar o programa, sem busca visual, re ou
    # retomada automatica do segue-linha mascararem o angulo obtido.
    if T_180_TEST_STOP:
        status.value = 'Teste 180 concluido — parado apos o giro'
        while not terminate.value:
            sleep_steering(.05)
        return last_turn_dir

    # Depois da parte cega, reduz a velocidade e continua no mesmo sentido ate
    # a camera confirmar a linha centralizada. A posicao inferior e o sinal
    # principal porque representa diretamente a bolinha azul; o angulo fica
    # como alternativa para linhas que ainda nao alcancaram a borda inferior.
    steer(180 if last_turn_dir == "r" else -180, T_180_SEARCH_SPEED)
    status.value = 'Completando 180 — procurando linha no centro'
    search_end = time.monotonic() + T_180_SEARCH_TIMEOUT
    aligned_since = None
    while time.monotonic() < search_end:
        bottom_aligned = abs(last_bottom_point.value - camera_x / 2) <= T_180_EXIT_BOTTOM_PX
        angle_aligned = abs(line_angle.value) <= T_180_EXIT_ANGLE
        aligned = line_detected.value and (bottom_aligned or angle_aligned)
        if aligned:
            if aligned_since is None:
                aligned_since = time.monotonic()
            elif time.monotonic() - aligned_since >= T_180_CONFIRM_TIME:
                break
        else:
            aligned_since = None
        sleep_steering(.01)

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
