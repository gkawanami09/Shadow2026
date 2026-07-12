"""
vision/gap.py — gap geometry (line-stub top edge) + gap-mode image masking.
Ported from Overengineering² Reading Dossier, Hotspot 3 (geometry) + Hotspot 1
  Original source: robot_v.3/Python/main/line_cam.py
    - get_gap_angle            (lines 427-437) — verbatim
    - in-loop publication      (lines 766-783) — wrapped in publish_gap_geometry /
      reset_gap_values
    - gap_avoid tunnel mask    (lines 676-678) — apply_gap_avoid_mask
Semantics: while line_status == "gap_detected", fit a min-area rectangle
around the re-found line stub, take its two HIGHEST corners and publish
gap_angle (top-edge angle vs. horizontal, 0° = square to the robot) and
gap_center_x/y (edge midpoint, x scaled to ±180). Both corners must sit above
the bottom 5 % of the frame.
"""

import cv2
import numpy as np

from config import camera_x, camera_y
from shared.mp_manager import gap_angle, gap_center_x, gap_center_y, gap_end_width


def get_gap_angle(box):
    box = box[box[:, 1].argsort()]

    vector = box[0] - box[1]
    angle = np.arccos(np.dot(vector, [1, 0]) / (np.linalg.norm(vector) * np.linalg.norm([1, 0]))) * 180 / np.pi
    angle = angle if box[0][0] < box[1][0] else -angle

    if angle == 180:
        angle = 0

    return box[0], box[1], angle


def publish_gap_geometry(blackline, debug_img=None):
    p1, p2, angle = get_gap_angle(cv2.boxPoints(cv2.minAreaRect(blackline)))
    if p1[1] < camera_y * 0.95 and p2[1] < camera_y * 0.95:
        gap_angle.value = angle
        gap_end_width.value = int(round(np.linalg.norm(p1 - p2)))

        center_gap_ponit = (p1 - p2) / 2 + p2

        gap_center_x.value = int((center_gap_ponit[0] - camera_x / 2) / (camera_x / 2) * 180)
        gap_center_y.value = center_gap_ponit[1]

        if debug_img is not None:
            cv2.line(debug_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(center_gap_ponit[0]), int(center_gap_ponit[1])), 5, (0, 255, 0), 1, cv2.LINE_AA)


def reset_gap_values():
    gap_angle.value = -181
    gap_center_x.value = -181
    gap_center_y.value = -1
    gap_end_width.value = -1


def apply_gap_avoid_mask(black_image):
    """Tunnel vision while blind-crossing: blank the left/right 35 %."""
    cv2.rectangle(black_image, (0, 0), (int(camera_x * .35), camera_y), 0, -1)
    cv2.rectangle(black_image, (int(camera_x * .65), 0), (camera_x, camera_y), 0, -1)
