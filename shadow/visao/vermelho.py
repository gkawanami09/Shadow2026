"""Detecta a faixa vermelha de chegada."""

from collections import deque

import cv2

from config import (RED_CONFIRM_READINGS, RED_CONFIRM_WINDOW_FRAMES,
                    RED_FAR_MAX_ANGLE_DEG, RED_FAR_MIN_ASPECT_RATIO,
                    RED_FAR_MIN_CONTOUR, RED_FAR_MIN_SPAN_RATIO,
                    RED_MIN_CONTOUR, camera_x, camera_y)


class ConfirmadorVermelho:
    """Exige 2-de-3 frames para uma mancha vermelha parar o robô."""

    def __init__(self, confirmacoes=RED_CONFIRM_READINGS,
                 tamanho_janela=RED_CONFIRM_WINDOW_FRAMES):
        if not 1 <= confirmacoes <= tamanho_janela:
            raise ValueError(
                "confirmacoes deve ficar entre 1 e tamanho_janela")
        self.confirmacoes = int(confirmacoes)
        self._votos = deque(maxlen=int(tamanho_janela))

    def atualizar(self, candidato):
        self._votos.append(bool(candidato))
        return sum(self._votos) >= self.confirmacoes

    def reiniciar(self):
        self._votos.clear()


def _faixa_vermelha_distante(contour, frame_shape):
    """Aceita uma faixa pequena somente se sua geometria for transversal."""
    area = float(cv2.contourArea(contour))
    if area < RED_FAR_MIN_CONTOUR:
        return False

    altura, largura = frame_shape[:2]
    (_centro, (lado_a, lado_b), angulo) = cv2.minAreaRect(contour)
    lado_longo = max(float(lado_a), float(lado_b))
    lado_curto = max(min(float(lado_a), float(lado_b)), 1.0)
    proporcao = lado_longo / lado_curto
    if (
        lado_longo < largura * RED_FAR_MIN_SPAN_RATIO
        or proporcao < RED_FAR_MIN_ASPECT_RATIO
    ):
        return False

    # Converte o angulo do maior eixo para [-90, 90]. Zero e horizontal.
    angulo_longo = float(angulo)
    if lado_b > lado_a:
        angulo_longo += 90.0
    while angulo_longo > 90.0:
        angulo_longo -= 180.0
    while angulo_longo < -90.0:
        angulo_longo += 180.0
    return abs(angulo_longo) <= RED_FAR_MAX_ANGLE_DEG


def check_contour_size(
    contours,
    contour_color="red",
    size=RED_MIN_CONTOUR,
    debug_img=None,
    frame_shape=None,
):
    if contour_color == "red":
        color = (0, 255, 0)
    elif contour_color == "green":
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    if frame_shape is None:
        frame_shape = (
            debug_img.shape
            if debug_img is not None else (camera_y, camera_x)
        )

    for contour in contours:
        contour_size = cv2.contourArea(contour)

        candidato = contour_size > size
        if contour_color == "red" and not candidato:
            candidato = _faixa_vermelha_distante(contour, frame_shape)

        if candidato:
            if debug_img is not None:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            return True

    return False
