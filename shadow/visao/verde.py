"""Detecta e confirma os marcadores verdes do percurso."""

from collections import deque

import cv2
import numpy as np

from config import (GREEN_BLACK_MAX_GAP_RATIO, GREEN_BLACK_MIN_RUN_RATIO,
                    GREEN_BLACK_ROI_SCALE, GREEN_CONFIRM_FRAMES,
                    GREEN_CONFIRM_WINDOW_FRAMES,
                    GREEN_MARKER_MAX_ASPECT, GREEN_MARKER_MEMORY,
                    GREEN_MARKER_MIN_ASPECT, GREEN_MARKER_MIN_RECT_FILL,
                    GREEN_MIN_AREA, GREEN_TURN_AROUND_CONFIRM_FRAMES,
                    GREEN_VOTE_THRESHOLD, GREEN_VOTE_WINDOW,
                    LINE_CROP_GREEN, LINE_CROP_NORMAL)
from shared.dados_compartilhados import (add_time_value, get_time_average,
                                         line_crop, timer, turn_dir)


class ConfirmadorVerde:
    """Libera uma maioria coerente dentro de uma janela curta de quadros."""

    def __init__(self, frames=GREEN_CONFIRM_FRAMES,
                 frames_180=GREEN_TURN_AROUND_CONFIRM_FRAMES,
                 window=GREEN_CONFIRM_WINDOW_FRAMES):
        self.frames = max(1, int(frames))
        self.frames_180 = max(1, int(frames_180))
        self.window = max(self.frames, int(window))
        self._historico = deque(maxlen=self.window)

    def atualizar(self, direcao):
        direcao = (
            direcao
            if direcao in ("left", "right", "turn_around")
            else "straight"
        )
        self._historico.append(direcao)

        # Dois verdes sempre possuem prioridade. Enquanto ha evidencia dupla
        # recente, um unico contorno perdido nao pode degradar 180 para 90.
        votos_180 = self._historico.count("turn_around")
        if votos_180 >= self.frames_180:
            self._historico.clear()
            return "turn_around"
        if votos_180:
            return "straight"

        votos_esquerda = self._historico.count("left")
        votos_direita = self._historico.count("right")
        if votos_esquerda >= self.frames and votos_esquerda > votos_direita:
            return "left"
        if votos_direita >= self.frames and votos_direita > votos_esquerda:
            return "right"
        return "straight"


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
    """Confirma preto conectado e adjacente sem exigir linha horizontal."""
    if roi.size == 0 or minimo < 2:
        return False
    mascara = (roi > 0).astype(np.uint8) * 255
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    quantidade, _rotulos, estatisticas, _centros = (
        cv2.connectedComponentsWithStats(mascara, connectivity=8)
    )
    altura_roi, largura_roi = mascara.shape
    normal = altura_roi if borda_interna in ("top", "bottom") else largura_roi
    alcance = max(2, int(round(normal * GREEN_BLACK_MAX_GAP_RATIO)))
    for indice in range(1, quantidade):
        x, y, largura, altura, area = estatisticas[indice]
        toca = (
            y + altura >= altura_roi - alcance
            if borda_interna == "bottom" else
            y <= alcance
            if borda_interna == "top" else
            x + largura >= largura_roi - alcance
            if borda_interna == "right" else
            x <= alcance
        )
        extensao = largura if orientacao == "horizontal" else altura
        if toca and extensao >= minimo and area >= max(4, minimo * .12):
            return max(1, int(round(
                1000. * np.count_nonzero(mascara) / mascara.size
            )))
    return 0


def check_green(contours_grn, black_image, debug_img=None):
    """Retorna apenas direcoes sustentadas pela geometria preta obrigatoria."""
    black_around_sign = np.zeros((len(contours_grn), 5), dtype=np.int16)

    for i, contour in enumerate(contours_grn):
        if not _marcador_plausivel(contour):
            continue

        green_box = cv2.boxPoints(cv2.minAreaRect(contour))
        check_black(black_around_sign, i, green_box, black_image)
        if debug_img is not None:
            leitura = black_around_sign[i]
            leitura_esquerda, leitura_direita = (
                determine_turn_direction((leitura,)))
            geometria_valida = leitura_esquerda != leitura_direita
            # Vermelho significa apenas quadrado verde plausivel; verde
            # significa que topo + lado ja autorizaram uma ordem de curva.
            cor = (0, 255, 0) if geometria_valida else (0, 0, 255)
            cv2.drawContours(
                debug_img, [np.intp(green_box)], -1, cor, 2)

    turn_left, turn_right = determine_turn_direction(black_around_sign)
    # O par tem prioridade absoluta. Sem esta ordem, perder um dos contornos
    # por um frame pode deixar a memoria de 90 graus vencer o retorno.
    if turn_left and turn_right:
        return "turn_around"
    if turn_left and not turn_right:
        return "left"
    if turn_right and not turn_left:
        return "right"
    return "straight"


def check_black(black_around_sign, i, green_box, black_image):
    """Mede acima/lados no eixo da camera, como a logica antiga."""
    altura_imagem, largura_imagem = black_image.shape[:2]
    x, y, largura, altura = cv2.boundingRect(
        np.asarray(green_box, dtype=np.float32))
    lado = max(largura, altura, 3)
    margem = max(4, int(round(lado * GREEN_BLACK_ROI_SCALE)))
    folga = max(2, int(round(lado * .12)))
    x0 = max(0, x - folga)
    x1 = min(largura_imagem, x + largura + folga)
    y0 = max(0, y - folga)
    y1 = min(altura_imagem, y + altura + folga)
    topo = black_image[max(0, y - margem):y, x0:x1]
    baixo = black_image[y + altura:min(altura_imagem, y + altura + margem), x0:x1]
    esquerda = black_image[y0:y1, max(0, x - margem):x]
    direita = black_image[y0:y1, x + largura:min(largura_imagem, x + largura + margem)]
    minimo = max(2, int(round(lado * GREEN_BLACK_MIN_RUN_RATIO)))

    black_around_sign[i, 0] = _tem_segmento_continuo(
        baixo, "horizontal", "top", minimo)
    black_around_sign[i, 1] = _tem_segmento_continuo(
        topo, "horizontal", "bottom", minimo)
    black_around_sign[i, 2] = _tem_segmento_continuo(
        esquerda, "vertical", "right", minimo)
    black_around_sign[i, 3] = _tem_segmento_continuo(
        direita, "vertical", "left", minimo)
    black_around_sign[i, 4] = int(np.ceil(np.max(green_box[:, 1])))
    return black_around_sign


def aquecer_numba():
    """Mantem a API de inicializacao; a validacao agora usa OpenCV."""


def determine_turn_direction(black_around_sign):
    turn_left = False
    turn_right = False

    for leitura in black_around_sign:
        # A regra da pista e topo + o lado oposto ao giro. Preto abaixo pode
        # ser a propria faixa de entrada quando o marcador ja chegou perto da
        # base; ele nao contradiz a ordem. Os dois lados continuam ambiguos e
        # nao autorizam um 90.
        tem_topo = leitura[1] > 0
        suporte_esquerda = int(leitura[2])
        suporte_direita = int(leitura[3])
        # Em chegada diagonal a barra superior pode invadir as duas ROIs.
        # O lado realmente adjacente ocupa claramente mais pixels; se a
        # diferenca for pequena, a cena continua ambigua e o robo vai reto.
        esquerda_domina = bool(
            suporte_esquerda > 0
            and (suporte_direita == 0
                 or suporte_esquerda >= suporte_direita * 1.15)
        )
        direita_domina = bool(
            suporte_direita > 0
            and (suporte_esquerda == 0
                 or suporte_direita >= suporte_esquerda * 1.15)
        )
        if tem_topo and esquerda_domina and not direita_domina:
            turn_right = True
        elif tem_topo and direita_domina and not esquerda_domina:
            turn_left = True

    return turn_left, turn_right


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
