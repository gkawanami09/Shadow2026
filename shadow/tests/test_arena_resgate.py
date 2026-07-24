"""Testes sinteticos do guardiao monocular da arena."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.arena_resgate import (  # noqa: E402
    MonocularArenaGuardian,
    annotate_arena_evidence,
    evaluate_arena_support,
)


def _boundary_values(width, height, slope=0.0, base_ratio=0.56):
    xs = np.arange(width, dtype=np.float32)
    return (
        base_ratio * height
        + float(slope) * (xs - 0.5 * width)
    )


def _arena_frame(
    width=640,
    height=480,
    *,
    slope=0.0,
    door=None,
    grout=False,
    shadow=False,
):
    """Parede acima e piso iluminado abaixo de uma soleira sintetica."""
    boundary = _boundary_values(width, height, slope=slope)
    yy, xx = np.indices((height, width))

    wall = np.empty((height, width, 3), dtype=np.float32)
    wall[:, :, 0] = 196 + 7 * xx / max(width - 1, 1)
    wall[:, :, 1] = 198 + 4 * yy / max(height - 1, 1)
    wall[:, :, 2] = 202

    floor = np.empty_like(wall)
    floor[:, :, 0] = (
        118 + 20 * xx / max(width - 1, 1)
        + 9 * yy / max(height - 1, 1)
    )
    floor[:, :, 1] = (
        142 + 14 * xx / max(width - 1, 1)
        + 7 * yy / max(height - 1, 1)
    )
    floor[:, :, 2] = (
        160 + 8 * xx / max(width - 1, 1)
        + 4 * yy / max(height - 1, 1)
    )

    frame = wall
    floor_pixels = yy >= boundary[None, :]
    frame[floor_pixels] = floor[floor_pixels]

    if door is not None:
        x0, x1 = door
        # O piso externo tem a mesma aparencia do interno e apaga a linha
        # parede-piso no vao. A soleira precisa vir dos trechos laterais.
        door_mask = (
            (xx >= x0)
            & (xx < x1)
            & (yy < boundary[None, :])
        )
        external = floor.copy()
        external[:, :, 0] -= 4
        external[:, :, 1] -= 3
        frame[door_mask] = external[door_mask]

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Reforca somente os trechos fisicos da juncao. Nenhuma linha e desenhada
    # dentro da porta.
    points = np.column_stack((
        np.arange(width, dtype=np.int32),
        np.rint(boundary).astype(np.int32),
    ))
    if door is None:
        cv2.polylines(frame, [points], False, (72, 78, 82), 2)
    else:
        x0, x1 = door
        if x0 > 1:
            cv2.polylines(
                frame, [points[:x0]], False, (72, 78, 82), 2)
        if x1 < width - 1:
            cv2.polylines(
                frame, [points[x1:]], False, (72, 78, 82), 2)
        cv2.line(
            frame,
            (x0, int(round(boundary[x0]))),
            (x0, max(0, int(round(boundary[x0])) - height // 4)),
            (80, 85, 88),
            3,
        )
        cv2.line(
            frame,
            (x1, int(round(boundary[min(x1, width - 1)]))),
            (x1, max(
                0,
                int(round(boundary[min(x1, width - 1)]))
                - height // 4,
            )),
            (80, 85, 88),
            3,
        )

    if grout:
        for ratio in (0.71, 0.84, 0.94):
            y = int(round(ratio * height))
            cv2.line(frame, (0, y), (width - 1, y), (80, 92, 98), 2)
        cv2.line(
            frame,
            (int(0.18 * width), height - 1),
            (int(0.42 * width), int(0.58 * height)),
            (86, 95, 100),
            2,
        )

    if shadow:
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            (int(0.50 * width), int(0.74 * height)),
            (int(0.22 * width), int(0.08 * height)),
            -8,
            0,
            360,
            (55, 67, 76),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    return frame, boundary


def _draw_ball(frame, center, radius, silver=True):
    x, y = (int(round(value)) for value in center)
    radius = int(round(radius))
    cv2.ellipse(
        frame,
        (x, y + radius),
        (max(2, int(0.72 * radius)), max(1, int(0.16 * radius))),
        0,
        0,
        360,
        (65, 70, 75),
        -1,
    )
    color = (210, 218, 224) if silver else (35, 36, 38)
    cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(
        frame,
        (x - radius // 4, y - radius // 4),
        max(2, radius // 6),
        (250, 250, 250) if silver else (72, 72, 74),
        -1,
        cv2.LINE_AA,
    )


class MonocularArenaGuardianTests(unittest.TestCase):
    def setUp(self):
        self.guardian = MonocularArenaGuardian()

    def test_internal_ball_has_connected_floor_support_with_grout_and_shadow(self):
        frame, boundary = _arena_frame(grout=True, shadow=True)
        center = (320, float(boundary[320] + 44))
        radius = 27
        _draw_ball(frame, center, radius)

        evidence = self.guardian.inspect(frame, (center[0], center[1], radius))

        self.assertTrue(evidence.valid, evidence.reason)
        self.assertEqual(evidence.reason, "ok")
        self.assertGreaterEqual(evidence.floor_support, 0.38)
        self.assertGreaterEqual(evidence.boundary_confidence, 0.52)
        self.assertAlmostEqual(
            evidence.boundary_y,
            float(boundary[320]),
            delta=14,
        )

    def test_round_object_above_wall_is_rejected(self):
        frame, boundary = _arena_frame()
        center = (390, float(boundary[390] - 49))
        radius = 20
        _draw_ball(frame, center, radius)

        evidence = evaluate_arena_support(
            frame,
            center_x=center[0],
            center_y=center[1],
            radius=radius,
            guardian=self.guardian,
        )

        self.assertFalse(evidence.valid)
        self.assertEqual(evidence.reason, "fora_arena")

    def test_door_gap_is_interpolated_and_external_object_is_blocked(self):
        door = (245, 395)
        frame, boundary = _arena_frame(door=door)
        center = (320, float(boundary[320] - 43))
        radius = 22
        _draw_ball(frame, center, radius)

        model = self.guardian.build_model(frame)
        evidence = self.guardian.evaluate(
            model, (center[0], center[1], radius))

        self.assertTrue(model.valid, model.reason)
        door_start = int(round(door[0] * model.work_width / frame.shape[1]))
        door_end = int(round(door[1] * model.work_width / frame.shape[1]))
        door_support = float(np.mean(
            model.observed_columns[door_start:door_end]))
        side_support = float(np.mean(np.concatenate((
            model.observed_columns[:door_start],
            model.observed_columns[door_end:],
        ))))
        self.assertLess(door_support, side_support * 0.70)
        self.assertFalse(evidence.valid)
        self.assertEqual(evidence.reason, "fora_arena")
        self.assertAlmostEqual(
            evidence.boundary_y,
            float(boundary[320]),
            delta=15,
        )

    def test_internal_ball_in_front_of_same_door_is_allowed(self):
        door = (245, 395)
        frame, boundary = _arena_frame(door=door)
        center = (320, float(boundary[320] + 39))
        radius = 25
        _draw_ball(frame, center, radius)

        evidence = self.guardian.inspect(
            frame, (center[0], center[1], radius))

        self.assertTrue(evidence.valid, evidence.reason)
        self.assertGreaterEqual(evidence.floor_support, 0.38)

    def test_inclined_boundary_uses_local_height(self):
        frame, boundary = _arena_frame(slope=0.18)
        inside_center = (525, float(boundary[525] + 35))
        outside_center = (105, float(boundary[105] - 42))
        _draw_ball(frame, inside_center, 23)
        _draw_ball(frame, outside_center, 19)

        model = self.guardian.build_model(frame)
        inside = self.guardian.evaluate(
            model, (*inside_center, 23))
        outside = self.guardian.evaluate(
            model, (*outside_center, 19))

        self.assertTrue(model.valid, model.reason)
        self.assertTrue(inside.valid, inside.reason)
        self.assertFalse(outside.valid)
        self.assertEqual(outside.reason, "fora_arena")
        self.assertLess(
            float(model.boundary_y[model.work_width // 5]),
            float(model.boundary_y[4 * model.work_width // 5]),
        )

    def test_ball_near_horizontal_corner_needs_only_central_support(self):
        frame, boundary = _arena_frame(grout=True)
        center = (27, float(boundary[27] + 37))
        radius = 19
        _draw_ball(frame, center, radius)

        evidence = self.guardian.inspect(frame, (*center, radius))

        self.assertTrue(evidence.valid, evidence.reason)

    def test_ball_touching_wall_is_kept_by_contact_tolerance(self):
        frame, boundary = _arena_frame(door=(245, 395))
        radius = 24
        center = (205, float(boundary[205] + 2 - radius))
        _draw_ball(frame, center, radius)

        evidence = self.guardian.inspect(frame, (*center, radius))

        self.assertTrue(evidence.valid, evidence.reason)
        self.assertGreaterEqual(evidence.floor_support, 0.38)

    def test_uniform_scene_without_boundary_fails_closed(self):
        frame = np.full((480, 640, 3), (145, 150, 155), dtype=np.uint8)

        evidence = self.guardian.inspect(frame, (320, 330, 25))

        self.assertFalse(evidence.valid)
        self.assertEqual(evidence.reason, "sem_limite_confiavel")
        self.assertEqual(evidence.floor_support, 0.0)

    def test_normalized_geometry_is_resolution_independent(self):
        for width, height in ((320, 240), (960, 540)):
            with self.subTest(size=(width, height)):
                frame, boundary = _arena_frame(width=width, height=height)
                x = int(round(0.68 * width))
                radius = 0.052 * height
                center = (x, float(boundary[x] + 0.085 * height))
                _draw_ball(frame, center, radius)

                evidence = self.guardian.inspect(
                    frame, (*center, radius))

                self.assertTrue(evidence.valid, evidence.reason)
                self.assertAlmostEqual(
                    evidence.boundary_y / height,
                    float(boundary[x]) / height,
                    delta=0.035,
                )

    def test_model_is_reused_and_debug_overlay_preserves_shape(self):
        frame, boundary = _arena_frame(door=(250, 390))
        model = self.guardian.build_model(frame)
        first = self.guardian.evaluate(
            model, (180, float(boundary[180] + 38), 22))
        second = self.guardian.evaluate(
            model, (470, float(boundary[470] + 40), 24))

        self.assertIs(first.model, model)
        self.assertIs(second.model, model)
        self.assertGreater(len(first.boundary_points()), 2)
        overlay = annotate_arena_evidence(frame, first)
        self.assertEqual(overlay.shape, frame.shape)
        self.assertFalse(np.shares_memory(overlay, frame))


if __name__ == "__main__":
    unittest.main()
