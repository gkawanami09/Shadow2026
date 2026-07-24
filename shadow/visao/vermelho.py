"""Detecta a faixa vermelha de chegada."""

import cv2

from config import RED_MIN_CONTOUR


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
