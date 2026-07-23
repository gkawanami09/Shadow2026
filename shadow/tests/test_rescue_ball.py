import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg
from vision.rescue_ball import BallDetector, RescueEnhancer


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
        result = self._confirmed(BallDetector("silver"), silver_ball_frame())
        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.kind, "silver")
        self.assertAlmostEqual(result.center_x, 320, delta=12)

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
