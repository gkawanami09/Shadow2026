import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg
from vision.rescue_ball import BallDetector, RescueEnhancer, _Candidate


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
