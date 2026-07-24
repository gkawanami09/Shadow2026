import sys
from pathlib import Path
import unittest
from unittest import mock

import cv2
import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg
import vision.rescue_ball as rescue_ball
from vision.rescue_ball import (
    BallDetection,
    BallDetector,
    CloseCrescentEvidence,
    RescueEnhancer,
    _Candidate,
    _Proposal,
    _crescent_band_points,
    annotate_rescue_frame,
)


def base_frame():
    frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    cv2.rectangle(frame, (0, 80), (639, 479), (145, 145, 145), -1)
    return frame


def black_ball_frame():
    frame = base_frame()
    cv2.circle(frame, (320, 300), 43, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.circle(frame, (307, 286), 7, (75, 75, 75), -1, cv2.LINE_AA)
    return frame


def silver_ball_frame():
    frame = base_frame()
    center = (320, 300)
    for radius, value in (
        (44, 70), (40, 95), (34, 125), (27, 160), (19, 195)):
        cv2.circle(frame, center, radius, (value, value, value), -1, cv2.LINE_AA)
    cv2.circle(frame, (307, 286), 8, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(frame, center, 44, (65, 65, 65), 2, cv2.LINE_AA)
    return frame


def with_color_cast(frame, bgr_gains):
    gains = np.asarray(bgr_gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(
        frame.astype(np.float32) * gains,
        0,
        255,
    ).astype(np.uint8)


def close_crescent_frame(
    center_x_ratio=0.50,
    top_y_ratio=0.74,
    halfspan_ratio=0.46,
    background_value=165,
    sphere_value=35,
):
    height, width = 240, 320
    frame = np.full(
        (height, width, 3), background_value, dtype=np.uint8)
    normalized_x = np.linspace(-1.0, 1.0, 201)
    xs = (
        center_x_ratio * width
        + normalized_x * halfspan_ratio * width
    )
    top_y = top_y_ratio * height
    bottom_y = cfg.BALL_CRESCENT_BOTTOM_RATIO * height
    halfspan = halfspan_ratio * width
    vertical_delta = bottom_y - top_y
    radius = (
        halfspan * halfspan
        + vertical_delta * vertical_delta
    ) / max(2.0 * vertical_delta, 1.0)
    center_y = top_y + radius
    offset_x = normalized_x * halfspan
    ys = center_y - np.sqrt(np.maximum(
        radius * radius - np.square(offset_x),
        0.0,
    ))
    curve = np.column_stack((xs, ys)).astype(np.int32)
    polygon = np.vstack((
        curve,
        (curve[-1, 0], height - 1),
        (curve[0, 0], height - 1),
    )).astype(np.int32)
    cv2.fillPoly(
        frame,
        [polygon],
        (sphere_value, sphere_value, sphere_value),
    )
    return frame


def checkerboard_frame(cell_size):
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    values = np.where(
        ((xx // cell_size + yy // cell_size) % 2) == 0,
        30,
        210,
    ).astype(np.uint8)
    return cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)


def close_circle_frame(
    top_y_ratio=0.74,
    halfspan_ratio=0.46,
):
    """Círculo real cuja interseção inferior coincide com o gate."""
    height, width = 240, 320
    top_y = top_y_ratio * height
    bottom_y = cfg.BALL_CRESCENT_BOTTOM_RATIO * height
    halfspan = halfspan_ratio * width
    vertical_delta = bottom_y - top_y
    radius = (
        halfspan * halfspan
        + vertical_delta * vertical_delta
    ) / max(2.0 * vertical_delta, 1.0)
    center_y = top_y + radius
    frame = np.full((height, width, 3), 165, dtype=np.uint8)
    cv2.circle(
        frame,
        (width // 2, int(round(center_y))),
        int(round(radius)),
        (35, 35, 35),
        -1,
        cv2.LINE_AA,
    )
    return frame


def thick_grid_frame(angle=0.0):
    height, width = 240, 320
    gray = np.full((height, width), 210, dtype=np.uint8)
    for x in range(0, width, 18):
        cv2.line(gray, (x, 130), (x, height - 1), 25, 3)
    for y in range(130, height, 18):
        cv2.line(gray, (0, y), (width - 1, y), 25, 3)
    if angle:
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0), angle, 1.0)
        gray = cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=210,
        )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def disconnected_crescent_islands(connect_at_floor=False):
    height, width = 240, 320
    frame = np.full((height, width, 3), 165, dtype=np.uint8)
    center = 0.50 * width
    halfspan = 0.46 * width
    left = int(round(center - halfspan))
    right = int(round(center + halfspan))
    for start_x in range(left, right, 28):
        end_x = min(start_x + 20, right)
        xs = np.arange(start_x, end_x + 1)
        normalized = (xs - center) / halfspan
        top_y = 0.70 * height
        bottom_y = cfg.BALL_CRESCENT_BOTTOM_RATIO * height
        vertical_delta = bottom_y - top_y
        radius = (
            halfspan * halfspan
            + vertical_delta * vertical_delta
        ) / max(2.0 * vertical_delta, 1.0)
        circle_center_y = top_y + radius
        ys = circle_center_y - np.sqrt(np.maximum(
            radius * radius - np.square(xs - center),
            0.0,
        ))
        curve = np.column_stack((xs, ys)).astype(np.int32)
        polygon = np.vstack((
            curve,
            (end_x, height - 3),
            (start_x, height - 3),
        )).astype(np.int32)
        cv2.fillPoly(frame, [polygon], (35, 35, 35))
    if connect_at_floor:
        cv2.rectangle(
            frame,
            (left, height - 5),
            (right, height - 1),
            (35, 35, 35),
            -1,
        )
    return frame


def irregular_foil_dome_frame():
    """Domo largo e amassado, semelhante aos brutos enviados pelo robo."""
    height, width = 240, 320
    boundary = np.asarray(
        (
            (30, 239),
            (38, 226),
            (52, 209),
            (70, 196),
            (95, 180),
            (125, 165),
            (160, 153),
            (186, 162),
            (210, 176),
            (240, 199),
            (270, 224),
            (292, 239),
        ),
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [boundary], 255)

    rng = np.random.default_rng(20260723)
    gray = np.full((height, width), 165, dtype=np.int16)
    gray[mask > 0] = np.clip(
        65 + rng.normal(0, 35, np.count_nonzero(mask)),
        0,
        255,
    )
    frame = cv2.cvtColor(
        gray.astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    for _ in range(80):
        x = int(rng.integers(25, 295))
        y = int(rng.integers(150, 239))
        if mask[y, x]:
            cv2.circle(
                frame,
                (x, y),
                int(rng.integers(1, 4)),
                (240, 240, 240),
                -1,
            )
    return frame


class RescueOverlayTests(unittest.TestCase):
    def test_pickup_gate_shows_outside_confirming_and_ready(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = BallDetection(
            "silver", 424, 379, 37, 0.91, True, 181, 1.0)
        upper, _ = _crescent_band_points(
            frame.shape[1],
            frame.shape[0],
            0.5,
            cfg.BALL_CRESCENT_DEFAULT_TOP_RATIO,
            cfg.BALL_CRESCENT_DEFAULT_HALFSPAN_RATIO,
            cfg.BALL_CRESCENT_BOTTOM_RATIO,
        )
        sample_x, sample_y = upper[len(upper) // 2]
        pickup_point = (
            int(round(
                cfg.BALL_LOCKED_CIRCLE_POINT_X_RATIO
                * frame.shape[1])),
            int(round(
                cfg.BALL_LOCKED_CIRCLE_POINT_Y_RATIO
                * frame.shape[0])),
        )

        outside = annotate_rescue_frame(
            frame, detection, "APPROACH")
        confirming = annotate_rescue_frame(
            frame,
            detection,
            "NEAR_CONFIRM",
            pickup_in_range=True,
            pickup_confirmations=2,
        )
        ready = annotate_rescue_frame(
            frame,
            detection,
            "NEAR",
            pickup_in_range=True,
            pickup_confirmations=cfg.BALL_STOP_CONFIRM_FRAMES,
        )

        self.assertTupleEqual(
            tuple(outside[sample_y, sample_x]),
            (0, 255, 255),
        )
        self.assertTupleEqual(
            tuple(confirming[sample_y, sample_x]),
            (0, 165, 255),
        )
        self.assertTupleEqual(
            tuple(ready[sample_y, sample_x]),
            (0, 255, 0),
        )
        self.assertTupleEqual(
            tuple(outside[pickup_point[1], pickup_point[0]]),
            (0, 255, 255),
        )
        self.assertTupleEqual(
            tuple(ready[pickup_point[1], pickup_point[0]]),
            (0, 255, 0),
        )

    def test_overlay_accepts_crescent_metrics(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        evidence = CloseCrescentEvidence(
            True, 0.90, 0.80, 0.75, 0.90, 0.76,
            42.0, 0.56, 0.70, 0.34, 0.98, 1.0,
            gradient_polarity=0.90,
            profile_support=0.85,
            profile_polarity=0.88,
            coherent_run=0.70,
        )
        annotated = annotate_rescue_frame(
            frame,
            None,
            "NEAR_CONFIRM",
            crescent_evidence=evidence,
        )
        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated, frame))
        upper, _ = _crescent_band_points(
            frame.shape[1],
            frame.shape[0],
            evidence.center_x_ratio,
            evidence.top_y_ratio,
            evidence.halfspan_ratio,
            evidence.bottom_y_ratio,
        )
        sample_x, sample_y = upper[len(upper) // 2]
        self.assertTupleEqual(
            tuple(annotated[sample_y, sample_x]),
            (0, 255, 255),
        )


class RescueBallDetectorTests(unittest.TestCase):
    def _confirmed(self, detector, frame):
        result = None
        for index in range(cfg.BALL_ACQUIRE_HITS):
            result = detector.detect(frame, timestamp=index * 0.03)
        return result

    def test_enhancer_preserves_shape_and_type(self):
        frame = black_ball_frame()
        enhanced = RescueEnhancer().apply(frame)
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_close_crescent_accepts_huge_bottom_clipped_sphere(self):
        detector = BallDetector("any", enhance=False)
        detector.detect(close_crescent_frame(), timestamp=0.1)

        evidence = detector.last_crescent_evidence
        self.assertIsNotNone(evidence)
        self.assertTrue(evidence.accepted)
        self.assertGreaterEqual(
            evidence.support,
            cfg.BALL_CRESCENT_MIN_SUPPORT,
        )
        self.assertGreaterEqual(
            evidence.left_support,
            cfg.BALL_CRESCENT_MIN_SHOULDER_SUPPORT,
        )
        self.assertGreaterEqual(
            evidence.right_support,
            cfg.BALL_CRESCENT_MIN_SHOULDER_SUPPORT,
        )

    def test_close_crescent_accepts_true_circle_arcs(self):
        for top_y_ratio, halfspan_ratio in (
            (0.62, 0.40),
            (0.66, 0.46),
            (0.70, 0.40),
            (0.70, 0.46),
            (0.74, 0.40),
            (0.74, 0.46),
        ):
            with self.subTest(
                top_y_ratio=top_y_ratio,
                halfspan_ratio=halfspan_ratio,
            ):
                detector = BallDetector("any", enhance=False)
                detector.detect(
                    close_circle_frame(
                        top_y_ratio,
                        halfspan_ratio,
                    ),
                    timestamp=0.1,
                )
                self.assertTrue(
                    detector.last_crescent_evidence.accepted,
                    detector.last_crescent_evidence,
                )

    def test_close_crescent_accepts_both_contrast_polarities(self):
        detector = BallDetector("any", enhance=False)
        detector.detect(
            close_crescent_frame(
                background_value=45,
                sphere_value=215,
            ),
            timestamp=0.1,
        )
        self.assertTrue(detector.last_crescent_evidence.accepted)

    def test_close_crescent_accepts_metallic_texture_inside_silhouette(self):
        frame = close_crescent_frame(
            background_value=170,
            sphere_value=65,
        )
        mask = frame[:, :, 0] < 100
        rng = np.random.default_rng(2026)
        textured = frame[:, :, 0].astype(np.int16)
        textured[mask] += rng.normal(
            0, 24, np.count_nonzero(mask)).astype(np.int16)
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(textured, cv2.COLOR_GRAY2BGR)
        for _ in range(50):
            x = int(rng.integers(25, 295))
            y = int(rng.integers(180, 238))
            if mask[y, x]:
                cv2.circle(
                    frame,
                    (x, y),
                    int(rng.integers(1, 4)),
                    (230, 230, 230),
                    -1,
                )

        detector = BallDetector("any", enhance=False)
        detector.detect(frame, timestamp=0.1)
        self.assertTrue(detector.last_crescent_evidence.accepted)

    def test_close_crescent_accepts_irregular_realistic_foil_dome(self):
        detector = BallDetector("any", enhance=False)
        detector.detect(
            irregular_foil_dome_frame(),
            timestamp=0.1,
        )

        evidence = detector.last_crescent_evidence
        self.assertTrue(evidence.accepted, evidence)
        self.assertTrue(evidence.foil_fallback, evidence)
        self.assertGreaterEqual(
            evidence.foil_texture_bins,
            cfg.BALL_CRESCENT_FOIL_MIN_TEXTURE_BINS,
        )

    def test_foil_texture_is_only_measured_on_shortlist(self):
        original = rescue_ball._crescent_texture_metrics
        with mock.patch.object(
            rescue_ball,
            "_crescent_texture_metrics",
            wraps=original,
        ) as texture_metrics:
            detector = BallDetector("any", enhance=False)
            detector.detect(
                irregular_foil_dome_frame(),
                timestamp=0.1,
            )

        self.assertGreaterEqual(texture_metrics.call_count, 1)
        self.assertLessEqual(
            texture_metrics.call_count,
            cfg.BALL_CRESCENT_FOIL_MAX_CANDIDATES,
        )

    def test_close_crescent_accepts_center_hotspot_with_opposite_polarity(self):
        for sides, center in ((35, 215), (215, 35)):
            for hotspot_width in (
                40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140
            ):
                with self.subTest(
                    sides=sides,
                    center=center,
                    hotspot_width=hotspot_width,
                ):
                    frame = close_crescent_frame(
                        background_value=165,
                        sphere_value=sides,
                    )
                    sphere_mask = frame[:, :, 0] == sides
                    center_columns = np.zeros((240, 320), dtype=bool)
                    left = 160 - hotspot_width // 2
                    right = left + hotspot_width
                    center_columns[:, left:right] = True
                    frame[sphere_mask & center_columns] = (
                        center, center, center)

                    detector = BallDetector("any", enhance=False)
                    detector.detect(frame, timestamp=0.1)
                    self.assertTrue(
                        detector.last_crescent_evidence.accepted,
                        detector.last_crescent_evidence,
                    )

    def test_close_crescent_rejects_small_ball_joined_to_v_shadow(self):
        for shadow_value, ball_value in (
            (15, 70),
            (15, 100),
            (35, 70),
            (70, 100),
        ):
            with self.subTest(
                shadow_value=shadow_value,
                ball_value=ball_value,
            ):
                frame = np.full((240, 320, 3), 165, dtype=np.uint8)
                cv2.fillPoly(
                    frame,
                    [np.asarray(
                        ((160, 180), (310, 239), (10, 239)),
                        dtype=np.int32,
                    )],
                    (shadow_value,) * 3,
                )
                cv2.circle(
                    frame,
                    (160, 165),
                    29,
                    (ball_value,) * 3,
                    -1,
                    cv2.LINE_AA,
                )

                detector = BallDetector("any", enhance=False)
                detector.detect(frame, timestamp=0.1)
                evidence = detector.last_crescent_evidence
                self.assertFalse(
                    evidence.accepted,
                    evidence,
                )
                self.assertLess(
                    evidence.curvature_score,
                    cfg.BALL_CRESCENT_MIN_CURVATURE_SCORE,
                )

    def test_close_crescent_rejects_small_distant_ball_inside_old_box(self):
        frame = np.full((240, 320, 3), 165, dtype=np.uint8)
        cv2.circle(
            frame, (160, 190), 25,
            (35, 35, 35), -1, cv2.LINE_AA)
        detector = BallDetector("any", enhance=False)
        detector.detect(frame, timestamp=0.1)

        evidence = detector.last_crescent_evidence
        self.assertFalse(evidence.accepted)
        self.assertLess(
            min(evidence.left_support, evidence.right_support),
            cfg.BALL_CRESCENT_MIN_SHOULDER_SUPPORT,
        )

    def test_close_crescent_rejects_horizontal_floor_line(self):
        frame = np.full((240, 320, 3), 165, dtype=np.uint8)
        cv2.rectangle(
            frame, (0, 180), (319, 239),
            (35, 35, 35), -1)
        detector = BallDetector("any", enhance=False)
        detector.detect(frame, timestamp=0.1)
        self.assertFalse(detector.last_crescent_evidence.accepted)

    def test_close_crescent_rejects_misaligned_large_arc(self):
        detector = BallDetector("any", enhance=False)
        detector.detect(
            close_crescent_frame(center_x_ratio=0.70),
            timestamp=0.1,
        )
        self.assertFalse(detector.last_crescent_evidence.accepted)

    def test_close_crescent_rejects_checkerboards(self):
        for cell_size in (8, 12, 16, 24, 32):
            with self.subTest(cell_size=cell_size):
                detector = BallDetector("any", enhance=False)
                detector.detect(
                    checkerboard_frame(cell_size),
                    timestamp=0.1,
                )
                self.assertFalse(
                    detector.last_crescent_evidence.accepted)

    def test_close_crescent_rejects_straight_and_rotated_grids(self):
        for angle in (0.0, -15.0, 15.0):
            with self.subTest(angle=angle):
                detector = BallDetector("any", enhance=False)
                detector.detect(
                    thick_grid_frame(angle),
                    timestamp=0.1,
                )
                self.assertFalse(
                    detector.last_crescent_evidence.accepted)

    def test_close_crescent_rejects_disconnected_arc_islands(self):
        for connected_at_floor in (False, True):
            with self.subTest(connected_at_floor=connected_at_floor):
                detector = BallDetector("any", enhance=False)
                detector.detect(
                    disconnected_crescent_islands(
                        connect_at_floor=connected_at_floor),
                    timestamp=0.1,
                )
                evidence = detector.last_crescent_evidence
                self.assertFalse(evidence.accepted)
                self.assertLess(
                    evidence.coherent_run,
                    cfg.BALL_CRESCENT_MIN_COHERENT_RUN,
                )

    def test_close_crescent_rejects_curve_without_filled_sphere(self):
        frame = np.full((240, 320, 3), 165, dtype=np.uint8)
        upper, _ = _crescent_band_points(
            320, 240, 0.50, 0.74, 0.46,
            cfg.BALL_CRESCENT_BOTTOM_RATIO)
        cv2.polylines(
            frame, [upper], False, (35, 35, 35), 4, cv2.LINE_AA)
        detector = BallDetector("any", enhance=False)
        detector.detect(frame, timestamp=0.1)

        evidence = detector.last_crescent_evidence
        self.assertFalse(evidence.accepted)
        self.assertTrue(
            evidence.profile_support
            < cfg.BALL_CRESCENT_MIN_PROFILE_SUPPORT
            or evidence.profile_polarity
            < cfg.BALL_CRESCENT_MIN_PROFILE_POLARITY
        )

    def test_close_crescent_rejects_filled_v_and_trapezoids(self):
        polygons = (
            ((160, 180), (310, 239), (10, 239)),
            ((80, 180), (240, 180), (310, 239), (10, 239)),
            ((100, 184), (220, 184), (310, 239), (10, 239)),
            ((100, 192), (220, 192), (310, 239), (10, 239)),
            ((123, 184), (195, 184), (300, 239), (16, 239)),
        )
        for points in polygons:
            with self.subTest(points=points):
                frame = np.full((240, 320, 3), 165, dtype=np.uint8)
                cv2.fillPoly(
                    frame,
                    [np.asarray(points, dtype=np.int32)],
                    (35, 35, 35),
                )
                detector = BallDetector("any", enhance=False)
                detector.detect(frame, timestamp=0.1)
                evidence = detector.last_crescent_evidence

                self.assertFalse(evidence.accepted)
                self.assertTrue(
                    evidence.curvature_score
                    < cfg.BALL_CRESCENT_MIN_CURVATURE_SCORE
                    or evidence.circle_rmse_ratio
                    > cfg.BALL_CRESCENT_MAX_CIRCLE_RMSE_RATIO
                )

    def test_detects_and_confirms_black_ball(self):
        result = self._confirmed(BallDetector("black"), black_ball_frame())
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.kind, "black")
        self.assertAlmostEqual(result.center_x, 320, delta=10)
        self.assertAlmostEqual(result.center_y, 300, delta=10)

    def test_detects_and_confirms_silver_ball(self):
        detector = BallDetector("silver")
        result = self._confirmed(detector, silver_ball_frame())
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.kind, "silver")
        self.assertAlmostEqual(result.center_x, 320, delta=12)
        detector.detect(silver_ball_frame(), timestamp=0.2)
        self.assertFalse(detector.last_hough_used)

    def test_hough_remains_a_fallback_when_contours_do_not_propose(self):
        detector = BallDetector("silver")
        detector._contour_proposals = lambda *_args: []
        result = self._confirmed(detector, silver_ball_frame())
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertTrue(detector.last_hough_used)

    def test_close_dome_keeps_hough_distance_veto_available(self):
        detector = BallDetector("any", enhance=False)
        detector._contour_proposals = lambda *_args: []
        detector._hough_proposals = mock.Mock(return_value=[])

        detector.detect(
            irregular_foil_dome_frame(),
            timestamp=0.1,
        )

        self.assertTrue(detector.last_crescent_evidence.accepted)
        self.assertTrue(detector.last_hough_used)
        detector._hough_proposals.assert_called_once()

    def test_strong_contour_skips_hough_during_acquisition(self):
        detector = BallDetector("silver")
        strong = _Candidate("silver", 320, 300, 42, 0.90)
        detector._evaluate_proposals = lambda *_args: [strong]

        def unexpected_hough(*_args):
            raise AssertionError(
                "Hough nao deve atrasar um contorno forte na aquisicao")

        detector._hough_proposals = unexpected_hough
        result = None
        for index in range(cfg.BALL_ACQUIRE_HITS):
            result = detector.detect(
                base_frame(), timestamp=0.1 + index * 0.03)

        self.assertFalse(detector.last_hough_used)
        self.assertEqual(result.hits, cfg.BALL_ACQUIRE_HITS)
        self.assertTrue(result.confirmed)

    def test_weak_contour_still_uses_hough_fallback(self):
        detector = BallDetector("silver")
        weak = _Candidate("silver", 320, 300, 42, 0.65)
        from_hough = _Candidate("silver", 321, 300, 43, 0.90)
        evaluations = iter(([weak], [from_hough]))
        detector._evaluate_proposals = lambda *_args: next(evaluations)
        detector._hough_proposals = lambda *_args: [object()]
        detector._deduplicate = lambda proposals: proposals

        result = detector.detect(base_frame(), timestamp=0.1)

        self.assertTrue(detector.last_hough_used)
        self.assertAlmostEqual(result.confidence, 0.90)

    def test_incompatible_contour_cannot_block_hough_reacquisition(self):
        detector = BallDetector("any")
        detector._tracked = _Candidate("silver", 100, 300, 40, 0.9)
        detector._hits = 5
        detector._pixel_scale = 1.0
        spurious = _Candidate("black", 500, 300, 40, 0.9)
        reacquired = _Candidate("silver", 102, 300, 41, 0.92)
        evaluations = iter(([spurious], [reacquired]))
        detector._evaluate_proposals = lambda *_args: next(evaluations)
        detector._hough_proposals = lambda *_args: [object()]
        detector._deduplicate = lambda proposals: proposals

        result = detector.detect(base_frame(), timestamp=0.1)

        self.assertTrue(detector.last_hough_used)
        self.assertEqual(result.kind, "silver")
        self.assertTrue(result.confirmed)
        self.assertEqual(result.hits, 6)
        self.assertAlmostEqual(result.center_x, 100.8)

    def test_compatible_confirmed_contour_skips_hough(self):
        detector = BallDetector("silver")
        detector._tracked = _Candidate("silver", 320, 300, 40, 0.9)
        detector._hits = cfg.BALL_ACQUIRE_HITS
        compatible = _Candidate("silver", 321, 300, 41, 0.92)
        detector._evaluate_proposals = lambda *_args: [compatible]

        def unexpected_hough(*_args):
            raise AssertionError("Hough nao deveria rodar com track compativel")

        detector._hough_proposals = unexpected_hough
        result = detector.detect(base_frame(), timestamp=0.1)
        self.assertFalse(detector.last_hough_used)
        self.assertTrue(result.confirmed)

    def test_roi_allows_only_small_clip_after_bottom_contact(self):
        width, height = 320, 240
        roi_top = int(height * cfg.BALL_ROI_TOP)
        roi_bottom = int(height * cfg.BALL_ROI_BOTTOM)
        just_clipped = _Proposal(
            160, 219, 25, 0.80, 0.80, "hough")
        too_far_out = _Proposal(
            160, 221, 25, 0.80, 0.80, "hough")

        self.assertTrue(BallDetector._inside_roi(
            just_clipped,
            width,
            height,
            roi_top,
            roi_bottom,
        ))
        self.assertFalse(BallDetector._inside_roi(
            too_far_out,
            width,
            height,
            roi_top,
            roi_bottom,
        ))

    def test_pickup_point_precedes_bottom_track_loss_with_real_margin(self):
        detector_height = cfg.RESCUE_DETECTOR_MAX_HEIGHT
        point_y = (
            cfg.BALL_LOCKED_CIRCLE_POINT_Y_RATIO * detector_height)
        last_allowed_bottom = (
            detector_height
            - 2
            + detector_height * cfg.BALL_ROI_BOTTOM_OVERFLOW_RATIO
        )

        # O antigo 0,98 deixava pouca imagem util antes de o Hough perder a
        # esfera real. O ponto operacional precisa conservar uma margem
        # vertical suficiente para duas imagens distintas.
        self.assertGreaterEqual(last_allowed_bottom - point_y, 17.0)

    def test_deduplicate_keeps_nested_radii_but_merges_near_duplicates(self):
        detector = BallDetector("silver")
        proposals = [
            _Proposal(100, 100, 20, 0.56, 0.50, "hough"),
            _Proposal(100.8, 100.5, 19.5, 0.56, 0.50, "hough"),
            _Proposal(102, 101, 13, 0.56, 0.50, "hough"),
        ]

        unique = detector._deduplicate(proposals)

        self.assertEqual(len(unique), 2)
        self.assertEqual(
            sorted(round(item.radius, 1) for item in unique),
            [13.0, 20.0],
        )

    def test_all_hough_radii_are_evaluated_before_selection(self):
        detector = BallDetector("silver")
        detector._contour_proposals = lambda *_args: []
        proposals = [
            _Proposal(320, 300, 40, 0.56, 0.50, "hough"),
            _Proposal(322, 299, 28, 0.56, 0.50, "hough"),
        ]
        detector._hough_proposals = lambda *_args: proposals
        evaluated_radii = []

        def evaluate(items, *_args):
            evaluated_radii.append([item.radius for item in items])
            if not items:
                return []
            # Simula halo externo invalido e perimetro menor valido.
            return [_Candidate("silver", 322, 299, 28, 0.82)]

        detector._evaluate_proposals = evaluate

        result = detector.detect(base_frame(), timestamp=0.1)

        self.assertEqual(evaluated_radii[-1], [40, 28])
        self.assertIsNotNone(result)
        self.assertEqual(result.radius, 28)

    def test_nested_reflections_do_not_beat_credible_outer_circle(self):
        detector = BallDetector("silver")
        detector._pixel_scale = 0.5
        outer = _Candidate("silver", 178, 163, 18.5, 0.70)
        reflection = _Candidate("silver", 181, 159, 14.5, 0.86)
        inner = _Candidate("silver", 180, 158, 12.0, 0.87)

        selected = detector._select_candidate(
            [outer, reflection, inner])

        self.assertIs(selected, outer)

    def test_weak_outer_halo_does_not_hide_strong_inner_candidate(self):
        detector = BallDetector("silver")
        detector._pixel_scale = 0.5
        halo = _Candidate("silver", 178, 163, 18.5, 0.60)
        inner = _Candidate("silver", 181, 159, 14.5, 0.90)

        selected = detector._select_candidate([halo, inner])

        self.assertIs(selected, inner)

    def test_outer_preference_does_not_bias_spatially_separate_candidates(self):
        detector = BallDetector("silver")
        detector._pixel_scale = 0.5
        large = _Candidate("silver", 100, 100, 20, 0.70)
        separate = _Candidate("silver", 130, 100, 14, 0.90)

        selected = detector._select_candidate([large, separate])

        self.assertIs(selected, separate)

    def test_stable_outer_circle_confirms_despite_changing_reflections(self):
        detector = BallDetector("silver")
        detector._pixel_scale = 0.5
        result = None
        for index in range(cfg.BALL_ACQUIRE_HITS):
            outer = _Candidate(
                "silver", 178 + index * 0.5, 163, 18.5, 0.70)
            reflection = _Candidate(
                "silver", 181 - index, 159 + index, 14.5, 0.86)
            selected = detector._select_candidate([outer, reflection])
            self.assertIs(selected, outer)
            result = detector._update_track(
                selected, timestamp=0.1 + index * 0.03)

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.hits, cfg.BALL_ACQUIRE_HITS)
        self.assertAlmostEqual(result.radius, 18.5)

    def test_acquisition_association_is_stricter_than_confirmed_tracking(self):
        detector = BallDetector("silver")
        detector._pixel_scale = 0.5
        detector._tracked = _Candidate("silver", 100, 100, 20, 0.9)
        moved = _Candidate("silver", 120, 100, 20, 0.9)
        grown = _Candidate("silver", 100, 100, 31, 0.9)

        detector._hits = 1
        self.assertFalse(detector._track_match(moved)[0])
        self.assertFalse(detector._track_match(grown)[0])

        detector._hits = cfg.BALL_ACQUIRE_HITS
        self.assertTrue(detector._track_match(moved)[0])
        self.assertTrue(detector._track_match(grown)[0])

    def test_confirmed_track_does_not_jump_to_incompatible_reflection(self):
        detector = BallDetector("any")
        detector._pixel_scale = 1.0
        detector._tracked = _Candidate(
            "silver", 320, 300, 50, 0.90)
        detector._hits = cfg.BALL_ACQUIRE_HITS
        detector._track_locked = True

        far_reflection = _Candidate(
            "black", 520, 390, 18, 0.96)
        huge_jump = _Candidate(
            "silver", 322, 302, 128, 0.96)

        self.assertIsNone(detector._select_candidate(
            [far_reflection, huge_jump]))
        self.assertIsNone(
            detector._update_track(None, timestamp=0.1))
        self.assertTrue(detector._track_locked)
        self.assertEqual(detector._hits, cfg.BALL_ACQUIRE_HITS)

    def test_any_target_lock_survives_silver_black_class_flip(self):
        detector = BallDetector("any")
        detector._pixel_scale = 1.0
        detector._tracked = _Candidate(
            "silver", 320, 300, 50, 0.90)
        detector._hits = cfg.BALL_ACQUIRE_HITS
        detector._track_locked = True
        changed_appearance = _Candidate(
            "black", 324, 303, 52, 0.88)

        self.assertTrue(
            detector._track_match(changed_appearance)[0])
        selected = detector._select_candidate(
            [changed_appearance])
        result = detector._update_track(
            selected, timestamp=0.1)

        self.assertTrue(result.confirmed)
        self.assertTrue(result.track_locked)
        self.assertAlmostEqual(result.center_x, 321.6)

    def test_detector_thresholds_scale_at_960_pixels_wide(self):
        frame = cv2.resize(
            silver_ball_frame(), (960, 720), interpolation=cv2.INTER_LINEAR)
        result = self._confirmed(BallDetector("silver"), frame)
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.kind, "silver")
        self.assertAlmostEqual(result.center_x, 480, delta=18)

    def test_detects_silver_at_runtime_detector_resolution(self):
        frame = cv2.resize(
            silver_ball_frame(),
            (cfg.RESCUE_DETECTOR_MAX_WIDTH,
             cfg.RESCUE_DETECTOR_MAX_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        detector = BallDetector("silver")
        result = self._confirmed(detector, frame)
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.kind, "silver")
        self.assertAlmostEqual(
            result.center_x,
            cfg.RESCUE_DETECTOR_MAX_WIDTH / 2,
            delta=8,
        )
        self.assertFalse(detector.last_hough_used)

    def test_detects_silver_under_cyan_and_green_color_cast(self):
        for gains in ((1.45, 1.35, 0.65), (1.05, 1.50, 0.65)):
            cast = with_color_cast(silver_ball_frame(), gains)
            frame = cv2.resize(
                cast,
                (cfg.RESCUE_DETECTOR_MAX_WIDTH,
                 cfg.RESCUE_DETECTOR_MAX_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            detector = BallDetector("silver")
            result = self._confirmed(detector, frame)
            self.assertIsNotNone(result)
            self.assertTrue(result.confirmed)
            self.assertEqual(result.kind, "silver")
            self.assertEqual(detector.last_diagnostic, "ok")

    def test_rejects_solid_cyan_circle_without_metallic_texture(self):
        frame = base_frame()
        cv2.circle(
            frame, (320, 300), 44, (230, 190, 70),
            -1, cv2.LINE_AA)
        frame = cv2.resize(
            frame,
            (cfg.RESCUE_DETECTOR_MAX_WIDTH,
             cfg.RESCUE_DETECTOR_MAX_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        detector = BallDetector("silver")
        results = [
            detector.detect(frame, timestamp=index * 0.03)
            for index in range(cfg.BALL_ACQUIRE_HITS + 1)
        ]
        self.assertFalse(any(
            result is not None and result.confirmed for result in results))

    def test_hough_fallback_still_detects_at_runtime_resolution(self):
        frame = cv2.resize(
            silver_ball_frame(),
            (cfg.RESCUE_DETECTOR_MAX_WIDTH,
             cfg.RESCUE_DETECTOR_MAX_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        detector = BallDetector("silver")
        detector._contour_proposals = lambda *_args: []
        result = self._confirmed(detector, frame)
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertTrue(detector.last_hough_used)

    def test_runtime_resolution_rejects_existing_negative_scenes(self):
        rectangle = base_frame()
        cv2.rectangle(
            rectangle, (250, 250), (390, 315), (20, 20, 20), -1)
        line = base_frame()
        cv2.rectangle(line, (0, 280), (639, 310), (20, 20, 20), -1)

        for original in (base_frame(), line, rectangle):
            frame = cv2.resize(
                original,
                (cfg.RESCUE_DETECTOR_MAX_WIDTH,
                 cfg.RESCUE_DETECTOR_MAX_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            detector = BallDetector("any")
            results = [
                detector.detect(frame, timestamp=index * 0.03)
                for index in range(cfg.BALL_ACQUIRE_HITS + 1)
            ]
            self.assertFalse(any(
                result is not None and result.confirmed
                for result in results
            ))

    def test_hough_telemetry_survives_internal_track_reset(self):
        detector = BallDetector("any")
        for index in range(cfg.BALL_MAX_TRACK_MISSES + 1):
            self.assertIsNone(
                detector.detect(base_frame(), timestamp=index * 0.03))
        self.assertTrue(detector.last_hough_used)
        self.assertEqual(detector.last_diagnostic, "sem_circulo")

    def test_detector_scales_correctly_for_wide_full_fov(self):
        resized = cv2.resize(
            silver_ball_frame(), (720, 540), interpolation=cv2.INTER_LINEAR)
        frame = np.full((540, 960, 3), 150, dtype=np.uint8)
        frame[:, 120:840] = resized
        result = self._confirmed(BallDetector("silver"), frame)
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.kind, "silver")
        self.assertAlmostEqual(result.center_x, 480, delta=18)

    def test_rejects_plain_rectangle(self):
        frame = base_frame()
        cv2.rectangle(frame, (250, 250), (390, 315), (20, 20, 20), -1)
        detector = BallDetector("any")
        results = [
            detector.detect(frame, timestamp=index * 0.03)
            for index in range(cfg.BALL_ACQUIRE_HITS + 1)
        ]
        self.assertFalse(any(
            result is not None and result.confirmed for result in results))

    def test_rejects_empty_floor_and_black_line(self):
        empty = base_frame()
        line = base_frame()
        cv2.rectangle(line, (0, 280), (639, 310), (20, 20, 20), -1)
        for frame in (empty, line):
            detector = BallDetector("any")
            results = [
                detector.detect(frame, timestamp=index * 0.03)
                for index in range(cfg.BALL_ACQUIRE_HITS + 1)
            ]
            self.assertFalse(any(
                result is not None and result.confirmed for result in results))

    def test_requires_temporal_confirmation(self):
        detector = BallDetector("black")
        first = detector.detect(black_ball_frame(), timestamp=0.0)
        self.assertIsNotNone(first)
        self.assertFalse(first.confirmed)
        self.assertEqual(first.hits, 1)

    def test_one_missing_frame_preserves_locked_confirmation(self):
        detector = BallDetector("black")
        frame = black_ball_frame()
        result = self._confirmed(detector, frame)
        self.assertTrue(result.confirmed)
        self.assertIsNone(detector.detect(base_frame(), timestamp=0.2))
        reacquired = detector.detect(frame, timestamp=0.23)
        self.assertIsNotNone(reacquired)
        self.assertTrue(reacquired.confirmed)
        self.assertTrue(reacquired.track_locked)

    def test_two_missing_frames_require_full_reconfirmation(self):
        detector = BallDetector("black")
        frame = black_ball_frame()
        result = self._confirmed(detector, frame)
        self.assertTrue(result.confirmed)
        self.assertIsNone(
            detector.detect(base_frame(), timestamp=0.20))
        self.assertIsNone(
            detector.detect(base_frame(), timestamp=0.23))
        self.assertFalse(detector._track_locked)
        self.assertIsNone(detector._tracked)
        self.assertIsNone(detector.last_locked_detection)

        reacquired = detector.detect(frame, timestamp=0.26)

        self.assertIsNotNone(reacquired)
        self.assertFalse(reacquired.confirmed)
        self.assertFalse(reacquired.track_locked)
        self.assertEqual(reacquired.hits, 1)

    def test_target_filter_does_not_accept_other_class(self):
        detector = BallDetector("silver")
        results = [
            detector.detect(black_ball_frame(), timestamp=index * 0.03)
            for index in range(cfg.BALL_ACQUIRE_HITS + 1)
        ]
        self.assertFalse(any(
            result is not None and result.confirmed for result in results))


if __name__ == "__main__":
    unittest.main()
