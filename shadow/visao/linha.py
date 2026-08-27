"""Escolhe o contorno da linha e calcula o erro de direção."""

import cv2
import numpy as np
try:
    from numba import njit
except ModuleNotFoundError:  # ferramentas/testes fora da Raspberry
    def njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

from config import (BOTTOM_CENTER_CONTROL, BOTTOM_CENTER_MIN_Y,
                    BOTTOM_CENTER_WEIGHT,
                    GREEN_BRANCH_TRACKER_OFFSET_PX,
                    OBSTACLE_LEFT_PREFERENCE_MIN_SPAN_RATIO,
                    camera_x, camera_y)
from shared.dados_compartilhados import line_crop, timer, turn_dir
from visao.cache_numba import aquecer_com_cache_recuperavel

x_last = camera_x / 2
y_last = camera_y / 2
multiple_bottom_side = camera_x / 2


def init_tracker():
    global x_last, y_last, multiple_bottom_side
    x_last = camera_x / 2
    y_last = camera_y / 2
    multiple_bottom_side = camera_x / 2


def contorno_atravessa_laterais(contorno):
    if contorno is None or np.asarray(contorno).size == 0:
        return False
    pontos_x = np.asarray(contorno)[:, 0, 0]
    return bool(
        np.min(pontos_x) < camera_x * .02
        and np.max(pontos_x) > camera_x * .98
    )


def determine_correct_line(contours_blk, preferir_esquerda=False,
                           turn_direction=None):
    """Escolhe o contorno mantendo o ramo verde ja marcado."""
    global x_last, y_last
    candidates = np.zeros((len(contours_blk), 5), dtype=np.int32)
    off_bottom = 0

    # A intencao do verde precisa participar da escolha deste quadro. Antes,
    # o deslocamento era aplicado apenas ao ``x_last`` depois de selecionar o
    # contorno; em uma intersecao, a correcao/historico podia vencer e o robo
    # seguir reto. A visao ja confirmou o marcador neste ponto, portanto o
    # mesmo deslocamento passa a ser a referencia da pontuacao atual.
    direcao_marcada = turn_direction or turn_dir.value
    x_referencia = x_last
    if direcao_marcada == "left":
        x_referencia = np.clip(
            x_last - GREEN_BRANCH_TRACKER_OFFSET_PX, 0, camera_x)
    elif direcao_marcada == "right":
        x_referencia = np.clip(
            x_last + GREEN_BRANCH_TRACKER_OFFSET_PX, 0, camera_x)

    for i, contour in enumerate(contours_blk):
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        box = box[box[:, 1].argsort()[::-1]]  # Sort them by their y values and reverse
        bottom_y = box[0][1]
        y_mean = (np.clip(box[0][1], 0, camera_y) + np.clip(box[3][1], 0, camera_y)) / 2

        if box[0][1] >= (camera_y * 0.75):
            off_bottom += 1

        box = box[box[:, 0].argsort()]
        x_mean = (np.clip(box[0][0], 0, camera_x) + np.clip(box[3][0], 0, camera_x)) / 2
        x_y_distance = abs(x_referencia - x_mean) + abs(y_last - y_mean)  # Distance between the last x/y and current x/y

        candidates[i] = i, bottom_y, x_y_distance, x_mean, y_mean

    if off_bottom < 2:
        candidates = candidates[candidates[:, 1].argsort()[::-1]]  # Sort candidates by their bottom_y
    else:
        off_bottom_candidates = candidates[np.where(candidates[:, 1] >= (camera_y * 0.75))]
        candidates = off_bottom_candidates[off_bottom_candidates[:, 2].argsort()]

    if preferir_esquerda:
        # Peso para o próximo quadro sem transformar a preferência em uma
        # ordem de giro: contornos mais à esquerda ficam mais próximos do
        # histórico e vencem apenas quando houver ambiguidade.
        x_last = np.clip(
            candidates[0][3] - GREEN_BRANCH_TRACKER_OFFSET_PX, 0, camera_x)
    elif direcao_marcada == "left":
        x_last = np.clip(
            candidates[0][3] - GREEN_BRANCH_TRACKER_OFFSET_PX, 0, camera_x)
    elif direcao_marcada == "right":
        x_last = np.clip(
            candidates[0][3] + GREEN_BRANCH_TRACKER_OFFSET_PX, 0, camera_x)
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


def aquecer_numba():
    """Compila a geometria com o robô parado, antes do primeiro comando."""
    contorno = np.array(
        [[180, 100], [268, 100], [268, 240], [180, 240]],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    aquecer_com_cache_recuperavel(
        calculate_angle_numba,
        contorno,
        contorno,
        camera_x / 2,
        camera_x / 2,
    )


def calculate_angle(
    blackline,
    blackline_crop,
    average_line_angle,
    turn_direction,
    last_bottom_point,
    average_line_point,
    preferir_esquerda=False,
    preferir_reto=False,
):
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

    elif preferir_reto:
        final_poi = poi_no_crop[0]

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

            atravessa_os_dois_lados = (
                poi_no_crop[1][0] < camera_x * 0.02
                and poi_no_crop[2][0] > camera_x * 0.98
            )
            largura_lateral = (
                poi_no_crop[2][0] - poi_no_crop[1][0]
            )
            preferencia_transversal = (
                preferir_esquerda
                and largura_lateral
                >= camera_x * OBSTACLE_LEFT_PREFERENCE_MIN_SPAN_RATIO
            )
            if preferencia_transversal:
                index = 1
                # Mantém a escolha por mais alguns quadros depois que a
                # preferência temporária for retirada pelo controle.
                timer.set_timer("multiple_side_r", 0)
                timer.set_timer("multiple_side_l", .6)
                final_poi = poi[index] if is_crop else poi_no_crop[index]

            elif (atravessa_os_dois_lados
                    and timer.get_timer("multiple_side_r")
                    and timer.get_timer("multiple_side_l")):
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
                if preferir_esquerda:
                    final_poi = [0, camera_y]
                    multiple_bottom_side = 0
                elif poi_no_crop[3][0] < bottom_point[0]:
                    final_poi = [0, camera_y]
                    multiple_bottom_side = 0
                else:
                    final_poi = [camera_x, camera_y]
                    multiple_bottom_side = camera_x
                timer.set_timer("multiple_bottom", .6)

    legacy_angle = int((final_poi[0] - camera_x / 2) / (camera_x / 2) * 180)

    if BOTTOM_CENTER_CONTROL and bottom_point[1] >= camera_y * BOTTOM_CENTER_MIN_Y:
        bottom_angle = int((bottom_point[0] - camera_x / 2) / (camera_x / 2) * 180)
        line_angle = int(round(np.clip(
            BOTTOM_CENTER_WEIGHT * bottom_angle
            + (1 - BOTTOM_CENTER_WEIGHT) * legacy_angle,
            -180, 180)))
    else:
        line_angle = legacy_angle

    return line_angle, final_poi, bottom_point
