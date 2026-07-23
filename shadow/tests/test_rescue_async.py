import sys
from pathlib import Path
import threading
import time
import unittest

import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from vision.rescue_async import (  # noqa: E402
    LatestFrameBallDetector,
    LatestFrameSource,
    _fit_detector_size,
    _scale_detection,
)
from vision.rescue_ball import BallDetection  # noqa: E402


def wait_result(worker, after_sequence=0, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll(after_sequence)
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("worker nao publicou resultado no prazo")


class ImmediateDetector:
    def detect(self, frame, timestamp):
        height, width = frame.shape[:2]
        return BallDetection(
            "silver",
            width / 2,
            height / 2,
            40,
            0.9,
            True,
            3,
            timestamp,
        )


class LatestFrameDetectorTests(unittest.TestCase):
    def test_fit_preserves_aspect_without_upscale(self):
        self.assertEqual(
            _fit_detector_size((720, 960, 3), 640, 480),
            (640, 480))
        self.assertEqual(
            _fit_detector_size((540, 960, 3), 640, 480),
            (640, 360))
        self.assertEqual(
            _fit_detector_size((240, 320, 3), 640, 480),
            (320, 240))

    def test_detection_is_mapped_back_to_preview_resolution(self):
        detection = BallDetection(
            "silver", 320, 240, 40, 0.9, True, 3, 1.0)
        scaled = _scale_detection(
            detection, (480, 640, 3), (720, 960, 3))
        self.assertEqual(scaled.center_x, 480)
        self.assertEqual(scaled.center_y, 360)
        self.assertEqual(scaled.radius, 60)
        self.assertEqual(scaled.timestamp, 1.0)

    def test_worker_downscales_and_publishes_result(self):
        worker = LatestFrameBallDetector(
            ImmediateDetector(), max_width=640, max_height=480)
        try:
            frame = np.zeros((720, 960, 3), dtype=np.uint8)
            sequence = worker.submit(frame, captured_at=10.0)
            result = wait_result(worker)
            self.assertEqual(result.sequence, sequence)
            self.assertEqual(result.detector_shape, (480, 640, 3))
            self.assertEqual(result.frame_shape, (720, 960, 3))
            self.assertEqual(result.detection.center_x, 480)
            self.assertEqual(result.detection.center_y, 360)
            self.assertIsNone(worker.poll(after_sequence=result.sequence))
        finally:
            worker.close()

    def test_pending_frame_is_replaced_instead_of_building_backlog(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingDetector:
            def __init__(self):
                self.values = []

            def detect(self, frame, timestamp):
                self.values.append(int(frame[0, 0, 0]))
                if len(self.values) == 1:
                    started.set()
                    release.wait(1.0)
                return None

        detector = BlockingDetector()
        worker = LatestFrameBallDetector(
            detector, max_width=640, max_height=480)
        try:
            worker.submit(np.full((40, 40, 3), 1, dtype=np.uint8), 1.0)
            self.assertTrue(started.wait(0.5))
            worker.submit(np.full((40, 40, 3), 2, dtype=np.uint8), 2.0)
            last_sequence = worker.submit(
                np.full((40, 40, 3), 3, dtype=np.uint8), 3.0)
            release.set()

            deadline = time.monotonic() + 1.0
            final = None
            while time.monotonic() < deadline:
                candidate = worker.poll()
                if (
                    candidate is not None
                    and candidate.sequence == last_sequence
                ):
                    final = candidate
                    break
                time.sleep(0.005)
            self.assertIsNotNone(final)
            self.assertEqual(final.sequence, last_sequence)
            self.assertEqual(detector.values, [1, 3])
            self.assertGreaterEqual(final.dropped_frames, 1)
        finally:
            release.set()
            worker.close()

    def test_reset_during_detection_discards_old_generation(self):
        started = threading.Event()
        release = threading.Event()

        class ResetAwareDetector:
            def __init__(self):
                self.calls = 0
                self.hits = 0
                self.reset_count = 0

            def reset(self):
                self.hits = 0
                self.reset_count += 1

            def detect(self, frame, timestamp):
                self.calls += 1
                if self.calls == 1:
                    started.set()
                    release.wait(1.0)
                self.hits += 1
                return BallDetection(
                    "silver", 20, 20, 10, 0.9,
                    self.hits >= 3, self.hits, timestamp)

        detector = ResetAwareDetector()
        worker = LatestFrameBallDetector(
            detector, max_width=640, max_height=480)
        try:
            worker.submit(np.zeros((40, 40, 3), dtype=np.uint8), 1.0)
            self.assertTrue(started.wait(0.5))
            worker.reset_tracking()
            new_sequence = worker.submit(
                np.zeros((40, 40, 3), dtype=np.uint8), 2.0)
            release.set()

            result = wait_result(worker)
            self.assertEqual(result.sequence, new_sequence)
            self.assertEqual(result.generation, 1)
            self.assertEqual(result.detection.hits, 1)
            self.assertFalse(result.detection.confirmed)
            self.assertEqual(detector.reset_count, 1)
        finally:
            release.set()
            worker.close()

    def test_reset_exception_is_reported_by_poll(self):
        class BrokenResetDetector(ImmediateDetector):
            def reset(self):
                raise ValueError("reset quebrado")

        worker = LatestFrameBallDetector(
            BrokenResetDetector(), max_width=640, max_height=480)
        try:
            worker.reset_tracking()
            worker.submit(np.zeros((40, 40, 3), dtype=np.uint8), 1.0)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    worker.poll()
                except RuntimeError as err:
                    self.assertIn("reset quebrado", str(err))
                    break
                time.sleep(0.005)
            else:
                self.fail("erro do reset nao foi publicado")
        finally:
            worker.close()

    def test_close_during_detection_prevents_late_result(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingDetector(ImmediateDetector):
            def detect(self, frame, timestamp):
                started.set()
                release.wait(1.0)
                return super().detect(frame, timestamp)

        worker = LatestFrameBallDetector(
            BlockingDetector(), max_width=640, max_height=480)
        worker.submit(np.zeros((40, 40, 3), dtype=np.uint8), 1.0)
        self.assertTrue(started.wait(0.5))
        self.assertFalse(worker.close(timeout=0.01))
        release.set()
        self.assertTrue(worker.close(timeout=0.5))
        self.assertIsNone(worker.poll())


class LatestFrameSourceTests(unittest.TestCase):
    def test_source_publishes_latest_frame_and_eos(self):
        class FiniteSource:
            def __init__(self):
                self.values = iter((1, 2, 3))

            def get_frame(self):
                try:
                    value = next(self.values)
                except StopIteration:
                    return None
                return np.full((4, 4, 3), value, dtype=np.uint8)

            def close(self):
                pass

        worker = LatestFrameSource(FiniteSource())
        try:
            deadline = time.monotonic() + 1.0
            while not worker.ended and time.monotonic() < deadline:
                time.sleep(0.005)
            frame = worker.poll()
            self.assertTrue(worker.ended)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.sequence, 3)
            self.assertEqual(int(frame.frame[0, 0, 0]), 3)
        finally:
            worker.close()

    def test_close_unblocks_capture(self):
        class BlockingSource:
            def __init__(self):
                self.released = threading.Event()

            def get_frame(self):
                self.released.wait(1.0)
                return None

            def close(self):
                self.released.set()

        source = BlockingSource()
        worker = LatestFrameSource(source)
        self.assertTrue(worker.close(timeout=0.5))
        self.assertTrue(worker.ended)
        self.assertIsNone(worker.poll())

    def test_close_timeout_does_not_freeze_caller(self):
        class StuckSource:
            def __init__(self):
                self.released = threading.Event()

            def get_frame(self):
                self.released.wait()
                return None

            def close(self):
                self.released.wait()

        source = StuckSource()
        worker = LatestFrameSource(source)
        started_at = time.monotonic()
        self.assertFalse(worker.close(timeout=0.03))
        self.assertLess(time.monotonic() - started_at, 0.20)
        source.released.set()
        self.assertTrue(worker.close(timeout=0.5))

    def test_capture_exception_is_reported(self):
        class BrokenSource:
            def get_frame(self):
                raise ValueError("camera quebrada")

            def close(self):
                pass

        worker = LatestFrameSource(BrokenSource())
        try:
            deadline = time.monotonic() + 1.0
            while not worker.ended and time.monotonic() < deadline:
                time.sleep(0.005)
            with self.assertRaisesRegex(RuntimeError, "camera quebrada"):
                worker.poll()
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
