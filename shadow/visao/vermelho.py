"""Detecta a faixa vermelha de chegada."""

from collections import deque

import cv2

from config import (RED_CONFIRM_READINGS, RED_CONFIRM_WINDOW_FRAMES,
                    RED_MIN_CONTOUR)


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


def check_contour_size(contours, contour_color="red", size=RED_MIN_CONTOUR, debug_img=None):
    if contour_color == "red":
        color = (0, 255, 0)
    elif contour_color == "green":
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    for contour in contours:
        contour_size = cv2.contourArea(contour)

        if contour_size > size:
            if debug_img is not None:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            return True

    return False
