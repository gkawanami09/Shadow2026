"""Detecta e confirma os marcadores verdes do percurso."""

import cv2
import numpy as np

from config import (GREEN_BLACK_MAX_GAP_RATIO, GREEN_BLACK_MIN_RUN_RATIO,
                    GREEN_BLACK_ROI_SCALE, GREEN_CONFIRM_FRAMES,
                    GREEN_MARKER_MAX_ASPECT, GREEN_MARKER_MEMORY,
                    GREEN_MARKER_MIN_ASPECT, GREEN_MARKER_MIN_RECT_FILL,
                    GREEN_MIN_AREA, GREEN_VOTE_THRESHOLD, GREEN_VOTE_WINDOW,
                    LINE_CROP_GREEN, LINE_CROP_NORMAL, camera_y)
from shared.dados_compartilhados import (add_time_value, get_time_average,
                                         line_crop, timer, turn_dir)


class ConfirmadorVerde:
    """So libera uma direcao repetida em quadros consecutivos."""

    def __init__(self, frames=GREEN_CONFIRM_FRAMES):
        self.frames = max(1, int(frames))
        self._direcao = "straight"
        self._contagem = 0

    def atualizar(self, direcao):
        if direcao == "straight":
            self._direcao = "straight"
            self._contagem = 0
            return "straight"
        if direcao == self._direcao:
            self._contagem += 1
        else:
            self._direcao = direcao
            self._contagem = 1
        return direcao if self._contagem >= self.frames else "straight"


def _marcador_plausivel(contour):
    """Rejeita manchas verdes que nao podem ser um quadrado da pista."""
    area = float(cv2.contourArea(contour))
    if area <= GREEN_MIN_AREA:
        return False
    _centro, (largura, altura), _angulo = cv2.minAreaRect(contour)
    largura = float(largura)
    altura = float(altura)
    if largura <= 0. or altura <= 0.:
        return False
    aspecto = largura / altura
    preenchimento = area / max(largura * altura, 1.)
    return bool(
        GREEN_MARKER_MIN_ASPECT <= aspecto <= GREEN_MARKER_MAX_ASPECT
        and preenchimento >= GREEN_MARKER_MIN_RECT_FILL
    )


def _tem_segmento_continuo(roi, orientacao, borda_interna, minimo):
    """Confirma uma linha continua e proxima da borda do marcador."""
    if roi.size == 0 or minimo < 2:
        return False
    mascara = (roi > 0).astype(np.uint8) * 255
    kernel = (
        np.ones((1, minimo), dtype=np.uint8)
        if orientacao == "horizontal"
        else np.ones((minimo, 1), dtype=np.uint8)
    )
    segmento = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    if not np.any(segmento):
        return False

    tamanho = (
        roi.shape[0]
        if borda_interna in ("top", "bottom")
        else roi.shape[1]
    )
    alcance = max(3, int(round(tamanho * GREEN_BLACK_MAX_GAP_RATIO)))
    if borda_interna == "bottom":
        proximo = segmento[-alcance:, :]
    elif borda_interna == "top":
        proximo = segmento[:alcance, :]
    elif borda_interna == "right":
        proximo = segmento[:, -alcance:]
    else:
        proximo = segmento[:, :alcance]
    return bool(np.any(proximo))


def check_green(contours_grn, black_image, debug_img=None):
    """Retorna apenas direcoes sustentadas pela geometria preta obrigatoria."""
    black_around_sign = np.zeros((len(contours_grn), 5), dtype=np.int16)

    for i, contour in enumerate(contours_grn):
        if not _marcador_plausivel(contour):
            continue

        green_box = cv2.boxPoints(cv2.minAreaRect(contour))
        if debug_img is not None:
            cv2.drawContours(
                debug_img, [np.intp(green_box)], -1, (0, 0, 255), 2)
        check_black(black_around_sign, i, green_box, black_image)

    turn_left, turn_right, left_bottom, right_bottom = (
        determine_turn_direction(black_around_sign)
    )
    if turn_left and not turn_right and not left_bottom:
        return "left"
    if turn_right and not turn_left and not right_bottom:
        return "right"
    if turn_left and turn_right and not (left_bottom and right_bottom):
        return "turn_around"
    return "straight"


def check_black(black_around_sign, i, green_box, black_image):
    """Mede linhas continuas acima e dos lados do quadrado verde."""
    x0 = max(int(np.floor(np.min(green_box[:, 0]))), 0)
    x1 = min(
        int(np.ceil(np.max(green_box[:, 0]))) + 1,
        black_image.shape[1],
    )
    y0 = max(int(np.floor(np.min(green_box[:, 1]))), 0)
    y1 = min(
        int(np.ceil(np.max(green_box[:, 1]))) + 1,
        black_image.shape[0],
    )
    largura = x1 - x0
    altura = y1 - y0
    if largura < 3 or altura < 3:
        return black_around_sign

    alcance = max(
        4,
        int(round(max(largura, altura) * GREEN_BLACK_ROI_SCALE)),
    )
    margem_x = max(1, int(round(largura * .08)))
    margem_y = max(1, int(round(altura * .08)))
    minimo_horizontal = max(
        2, int(round(largura * GREEN_BLACK_MIN_RUN_RATIO)))
    minimo_vertical = max(
        2, int(round(altura * GREEN_BLACK_MIN_RUN_RATIO)))

    topo = black_image[
        max(0, y0 - alcance):y0,
        x0 + margem_x:max(x0 + margem_x + 1, x1 - margem_x),
    ]
    baixo = black_image[
        y1:min(black_image.shape[0], y1 + alcance),
        x0 + margem_x:max(x0 + margem_x + 1, x1 - margem_x),
    ]
    esquerda = black_image[
        y0 + margem_y:max(y0 + margem_y + 1, y1 - margem_y),
        max(0, x0 - alcance):x0,
    ]
    direita = black_image[
        y0 + margem_y:max(y0 + margem_y + 1, y1 - margem_y),
        x1:min(black_image.shape[1], x1 + alcance),
    ]

    black_around_sign[i, 0] = int(_tem_segmento_continuo(
        baixo, "horizontal", "top", minimo_horizontal))
    black_around_sign[i, 1] = int(_tem_segmento_continuo(
        topo, "horizontal", "bottom", minimo_horizontal))
    black_around_sign[i, 2] = int(_tem_segmento_continuo(
        esquerda, "vertical", "right", minimo_vertical))
    black_around_sign[i, 3] = int(_tem_segmento_continuo(
        direita, "vertical", "left", minimo_vertical))
    black_around_sign[i, 4] = y1
    return black_around_sign


def aquecer_numba():
    """Mantem a API de inicializacao; a validacao agora usa OpenCV."""


def determine_turn_direction(black_around_sign):
    turn_left = False
    turn_right = False
    left_bottom = False
    right_bottom = False

    for leitura in black_around_sign:
        # Exatamente topo + um lado. Preto embaixo ou dos dois lados torna a
        # cena ambigua e, portanto, nunca autoriza uma curva.
        if np.sum(leitura[:4]) == 2:
            if leitura[1] == 1 and leitura[2] == 1:
                turn_right = True
                if leitura[4] > camera_y * .95:
                    right_bottom = True
            elif leitura[1] == 1 and leitura[3] == 1:
                turn_left = True
                if leitura[4] > camera_y * .95:
                    left_bottom = True

    return turn_left, turn_right, left_bottom, right_bottom


def average_direction(turn_direction):
    if turn_direction == "left":
        return -1
    if turn_direction == "right":
        return 1
    return 0


def latch_turn_direction(turn_direction, time_turn_direction):
    """Confirma a direcao por varios quadros e guarda uma memoria curta."""
    time_turn_direction = add_time_value(
        time_turn_direction, average_direction(turn_direction))
    avg_turn_dir = get_time_average(time_turn_direction, GREEN_VOTE_WINDOW)

    if avg_turn_dir > GREEN_VOTE_THRESHOLD:
        timer.set_timer("right_marker", GREEN_MARKER_MEMORY)
    elif avg_turn_dir < -GREEN_VOTE_THRESHOLD:
        timer.set_timer("left_marker", GREEN_MARKER_MEMORY)

    if (not timer.get_timer("right_marker")
            and turn_direction != "turn_around" and avg_turn_dir >= 0):
        turn_dir.value = "right"
        line_crop.value = LINE_CROP_GREEN
    elif (not timer.get_timer("left_marker")
          and turn_direction != "turn_around" and avg_turn_dir <= 0):
        turn_dir.value = "left"
        line_crop.value = LINE_CROP_GREEN
    else:
        turn_dir.value = turn_direction
        line_crop.value = LINE_CROP_NORMAL

    return time_turn_direction
