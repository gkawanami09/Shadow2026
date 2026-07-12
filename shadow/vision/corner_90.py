"""Deteccao conservadora de um unico canto preto em L (90 graus)."""

import math

import cv2
import numpy as np

from config import (CORNER_90_CLUSTER_TOLERANCE_PX,
                    CORNER_90_HOUGH_THRESHOLD, CORNER_90_MAX_AXIS_ERROR_DEG,
                    CORNER_90_MIN_ARM_PX, CORNER_90_VERTEX_TOLERANCE_PX)


def detect_corner_90(mask, debug_img=None):
    """Retorna ``left``, ``right`` ou ``none``.

    Exige uma perna aproximadamente vertical chegando de baixo e exatamente
    um braco horizontal longo saindo do vertice. Mais de um vertice separado
    (zig-zag), direcoes intermediarias (curva) ou continuacao vertical fazem a
    deteccao falhar fechada: o controle normal da bolinha permanece ativo.
    """
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180, CORNER_90_HOUGH_THRESHOLD,
        minLineLength=CORNER_90_MIN_ARM_PX, maxLineGap=18)
    if lines is None:
        return "none"

    horizontal = []
    vertical = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        axis_error = min(angle, abs(180 - angle))
        if axis_error <= CORNER_90_MAX_AXIS_ERROR_DEG:
            horizontal.append((x1, y1, x2, y2, length))
        elif abs(angle - 90) <= CORNER_90_MAX_AXIS_ERROR_DEG:
            vertical.append((x1, y1, x2, y2, length))

    candidates = []
    tol = CORNER_90_VERTEX_TOLERANCE_PX
    for hx1, hy1, hx2, hy2, _ in horizontal:
        h_min_x, h_max_x = sorted((hx1, hx2))
        h_y = (hy1 + hy2) / 2
        for vx1, vy1, vx2, vy2, _ in vertical:
            v_x = (vx1 + vx2) / 2
            v_min_y, v_max_y = sorted((vy1, vy2))
            if not (h_min_x - tol <= v_x <= h_max_x + tol
                    and v_min_y - tol <= h_y <= v_max_y + tol):
                continue

            # A perna de entrada precisa chegar ao vertice a partir de baixo.
            if v_max_y - h_y < CORNER_90_MIN_ARM_PX:
                continue

            left_extent = v_x - h_min_x
            right_extent = h_max_x - v_x
            left_long = left_extent >= CORNER_90_MIN_ARM_PX
            right_long = right_extent >= CORNER_90_MIN_ARM_PX
            if left_long == right_long:  # nenhum braco ou linha atravessando
                continue

            direction = "left" if left_long else "right"
            candidates.append((v_x, h_y, direction))

    if not candidates:
        return "none"

    # Linhas duplicadas do Hough sao aceitas somente se descreverem o mesmo
    # vertice e a mesma direcao. Vertices separados caracterizam zig-zag.
    directions = {c[2] for c in candidates}
    xs = [c[0] for c in candidates]
    ys = [c[1] for c in candidates]
    if (len(directions) != 1
            or max(xs) - min(xs) > CORNER_90_CLUSTER_TOLERANCE_PX
            or max(ys) - min(ys) > CORNER_90_CLUSTER_TOLERANCE_PX):
        return "none"

    direction = candidates[0][2]
    if debug_img is not None:
        center = (int(round(np.mean(xs))), int(round(np.mean(ys))))
        cv2.circle(debug_img, center, 8, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_img, f"90 {direction}", (center[0] + 8, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 165, 255), 1)
    return direction
