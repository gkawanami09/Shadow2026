import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg
from vision.rescue_ball import (
    BallDetection,
    BallDetector,
    RescueEnhancer,
    _Candidate,
    _Proposal,
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


class RescueOverlayTests(unittest.TestCase):
    def test_pickup_gate_shows_outside_confirming_and_ready(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = BallDetection(
            "silver", 424, 379, 37, 0.91, True, 181, 1.0)
        gate_y = int(round(
            frame.shape[0] * cfg.BALL_CLOSE_BOTTOM_Y_RATIO))
        gate_left = int(round(
            frame.shape[1] / 2
            - frame.shape[1] / 2 * cfg.BALL_CLOSE_OUTER_CENTER_ERROR
        ))
        sample_x = gate_left + 10

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
            tuple(outside[gate_y, sample_x]),
            (0, 255, 255),
        )
        self.assertTupleEqual(
            tuple(confirming[gate_y, sample_x]),
            (0, 165, 255),
        )
        self.assertTupleEqual(
            tuple(ready[gate_y, sample_x]),
            (0, 255, 0),
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

    def test_missing_frame_requires_full_reconfirmation(self):
        detector = BallDetector("black")
        frame = black_ball_frame()
        result = self._confirmed(detector, frame)
        self.assertTrue(result.confirmed)
        self.assertIsNone(detector.detect(base_frame(), timestamp=0.2))
        reacquired = detector.detect(frame, timestamp=0.23)
        self.assertIsNotNone(reacquired)
        self.assertFalse(reacquired.confirmed)
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
