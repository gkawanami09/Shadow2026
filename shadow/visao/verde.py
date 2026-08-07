"""Detecta e confirma os marcadores verdes do percurso."""

import cv2
import numpy as np
from numba import njit

from config import (GREEN_MARKER_MEMORY, GREEN_MIN_AREA, GREEN_ROI_MEAN,
                    GREEN_ROI_TOP, GREEN_VOTE_THRESHOLD, GREEN_VOTE_WINDOW,
                    LINE_CROP_GREEN, LINE_CROP_NORMAL, camera_x, camera_y)
from shared.dados_compartilhados import (add_time_value, get_time_average, line_crop,
                               timer, turn_dir)


def perto_do_robo(contour, altura_frame=None, corte=None):
    """O contorno alcança a parte de baixo do quadro?

    Mede pela BASE do contorno, não pelo centro: um marcador que está
    entrando no quadro por baixo já é acionável, e um que ainda está inteiro
    lá em cima não é. Esta é a única pergunta que separa o marcador verde
    real do reflexo verde-oliva da fita prata da entrada, que aparece grande
    e alto no quadro.
    """
    altura_frame = camera_y if altura_frame is None else int(altura_frame)
    corte = GREEN_ROI_TOP if corte is None else float(corte)
    _x, y, _w, h = cv2.boundingRect(contour)
    return (y + h) >= altura_frame * corte


def filtrar_verdes_proximos(contours_grn, altura_frame=None, corte=None):
    """Separa os contornos verdes em (aceitos, descartados por estarem longe).

    Devolver também os descartados é de propósito: o overlay de debug os
    desenha em outra cor, para a tela mostrar "vi verde e ignorei" em vez de
    simplesmente não mostrar nada — foi a ausência dessa informação que
    tornou o reflexo da fita prata difícil de diagnosticar.
    """
    aceitos = []
    descartados = []
    for contour in contours_grn:
        destino = (
            aceitos if perto_do_robo(contour, altura_frame, corte)
            else descartados
        )
        destino.append(contour)
    return aceitos, descartados


def check_green(contours_grn, black_image, debug_img=None):
    black_around_sign = np.zeros((len(contours_grn), 5), dtype=np.int16)  # [[b,t,l,r,lp], [b,t,l,r,lp]]

    for i, contour in enumerate(contours_grn):
        area = cv2.contourArea(contour)
        if area <= GREEN_MIN_AREA:
            continue

        green_box = cv2.boxPoints(cv2.minAreaRect(contour))
        if debug_img is not None:
            draw_box = np.intp(green_box)
            cv2.drawContours(debug_img, [draw_box], -1, (0, 0, 255), 2)
        # ``check_black`` apenas lê a máscara. Copiá-la para cada contorno
        # verde gastava memória e tempo sem proteger nenhum dado.
        black_around_sign = check_black(
            black_around_sign, i, green_box, black_image)

    turn_left, turn_right, left_bottom, right_bottom = determine_turn_direction(black_around_sign)

    if turn_left and not turn_right and not left_bottom:
        return "left"
    elif turn_right and not turn_left and not right_bottom:
        return "right"
    elif turn_left and turn_right and not (left_bottom and right_bottom):
        return "turn_around"
    else:
        return "straight"


@njit(cache=True)
def check_black(black_around_sign, i, green_box, black_image):
    green_box = green_box[green_box[:, 1].argsort()]

    marker_height = green_box[-1][1] - green_box[0][1]

    black_around_sign[i, 4] = int(green_box[2][1])

    # Bottom
    roi_b = black_image[int(green_box[2][1]):np.minimum(int(green_box[2][1] + (marker_height * 0.8)), camera_y), np.minimum(int(green_box[2][0]), int(green_box[3][0])):np.maximum(int(green_box[2][0]), int(green_box[3][0]))]
    if roi_b.size > 0:
        if np.mean(roi_b[:]) > GREEN_ROI_MEAN:
            black_around_sign[i, 0] = 1

    # Top
    roi_t = black_image[np.maximum(int(green_box[1][1] - (marker_height * 0.8)), 0):int(green_box[1][1]), np.minimum(np.maximum(int(green_box[0][0]), 0), np.maximum(int(green_box[1][0]), 0)):np.maximum(np.maximum(int(green_box[0][0]), 0), np.maximum(int(green_box[1][0]), 0))]
    if roi_t.size > 0:
        if np.mean(roi_t[:]) > GREEN_ROI_MEAN:
            black_around_sign[i, 1] = 1

    green_box = green_box[green_box[:, 0].argsort()]

    # Left
    roi_l = black_image[np.minimum(int(green_box[0][1]), int(green_box[1][1])):np.maximum(int(green_box[0][1]), int(green_box[1][1])), np.maximum(int(green_box[1][0] - (marker_height * 0.8)), 0):int(green_box[1][0])]
    if roi_l.size > 0:
        if np.mean(roi_l[:]) > GREEN_ROI_MEAN:
            black_around_sign[i, 2] = 1

    # Right
    roi_r = black_image[np.minimum(int(green_box[2][1]), int(green_box[3][1])):np.maximum(int(green_box[2][1]), int(green_box[3][1])), int(green_box[2][0]):np.minimum(int(green_box[2][0] + (marker_height * 0.8)), camera_x)]
    if roi_r.size > 0:
        if np.mean(roi_r[:]) > GREEN_ROI_MEAN:
            black_around_sign[i, 3] = 1

    return black_around_sign


def aquecer_numba():
    """Compila a análise antes de o primeiro verde aparecer na pista."""
    acumulador = np.zeros((1, 5), dtype=np.int16)
    caixa = np.array(
        [[180, 120], [220, 120], [180, 160], [220, 160]],
        dtype=np.float32,
    )
    mascara = np.zeros((camera_y, camera_x), dtype=np.uint8)
    check_black(acumulador, 0, caixa, mascara)


def determine_turn_direction(black_around_sign):
    turn_left = False
    turn_right = False
    left_bottom = False
    right_bottom = False

    for i in black_around_sign:
        if np.sum(i[:4]) == 2:
            if i[1] == 1 and i[2] == 1:
                turn_right = True
                if i[4] > camera_y * 0.95:
                    right_bottom = True
            elif i[1] == 1 and i[3] == 1:
                turn_left = True
                if i[4] > camera_y * 0.95:
                    left_bottom = True

    return turn_left, turn_right, left_bottom, right_bottom


def average_direction(turn_direction):
    turn_dir_num = 0

    if turn_direction == "left":
        turn_dir_num = -1
    elif turn_direction == "right":
        turn_dir_num = 1

    return turn_dir_num


def latch_turn_direction(turn_direction, time_turn_direction):
    """Confirma a direção por vários quadros e guarda uma memória curta."""
    time_turn_direction = add_time_value(time_turn_direction, average_direction(turn_direction))
    avg_turn_dir = get_time_average(time_turn_direction, GREEN_VOTE_WINDOW)

    if avg_turn_dir > GREEN_VOTE_THRESHOLD:
        timer.set_timer("right_marker", GREEN_MARKER_MEMORY)
    elif avg_turn_dir < -GREEN_VOTE_THRESHOLD:
        timer.set_timer("left_marker", GREEN_MARKER_MEMORY)

    if not timer.get_timer("right_marker") and not turn_direction == "turn_around" and avg_turn_dir >= 0:
        turn_dir.value = "right"
        line_crop.value = LINE_CROP_GREEN
    elif not timer.get_timer("left_marker") and not turn_direction == "turn_around" and avg_turn_dir <= 0:
        turn_dir.value = "left"
        line_crop.value = LINE_CROP_GREEN
    else:
        turn_dir.value = turn_direction
        line_crop.value = LINE_CROP_NORMAL

    return time_turn_direction
