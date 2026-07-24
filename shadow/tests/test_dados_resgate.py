"""Testes da gravação de imagens para calibração."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.dados_resgate import (  # noqa: E402
    RescueDatasetWriter,
    SCHEMA_VERSION,
)
from controle.aproximacao_resgate import MotionCommand  # noqa: E402
from resgate import _dataset_metadata  # noqa: E402
from visao.resgate_assincrono import AsyncDetectionResult  # noqa: E402
from visao.bola_resgate import (  # noqa: E402
    BallDetection,
    CloseCrescentEvidence,
)


class RescueDatasetWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix=".rescue-dataset-test-",
            dir=str(SHADOW_ROOT),
        )
        self.dataset_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def sample_frame():
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :8] = (0, 0, 255)
        frame[3:15, 10:25] = (17, 123, 241)
        frame[18:, :] = np.arange(32, dtype=np.uint8)[None, :, None]
        return frame

    def test_writes_lossless_raw_png_and_versioned_sidecar(self):
        writer = RescueDatasetWriter(
            self.dataset_root, session_id="lossless")
        frame = self.sample_frame()
        metadata = {
            "capture_sequence": 42,
            "position": {"distance_mm": 500, "offset_mm": -80},
        }

        submitted = writer.submit(frame, metadata)
        self.assertTrue(submitted.accepted)
        self.assertTrue(writer.close(timeout=2.0))

        png_path = writer.session_dir / f"{submitted.capture_id}.png"
        json_path = writer.session_dir / f"{submitted.capture_id}.json"
        restored = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(restored, frame))

        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["schema_version"], SCHEMA_VERSION)
        self.assertEqual(sidecar["capture_id"], submitted.capture_id)
        self.assertEqual(sidecar["image"]["format"], "png")
        self.assertTrue(sidecar["image"]["lossless"])
        self.assertEqual(sidecar["image"]["color_space"], "BGR8_opencv")
        self.assertEqual(sidecar["metadata"], metadata)
        self.assertEqual(writer.completed_count, 1)
        self.assertEqual(writer.failed_count, 0)

    def test_mailbox_has_only_one_pending_position(self):
        writer = RescueDatasetWriter(
            self.dataset_root, session_id="bounded")
        started = threading.Event()
        release = threading.Event()
        original_write = writer._write_snapshot

        def blocking_write(snapshot):
            started.set()
            release.wait(1.0)
            original_write(snapshot)

        writer._write_snapshot = blocking_write
        try:
            first = writer.submit(self.sample_frame(), {"sample": 1})
            self.assertTrue(first.accepted)
            self.assertTrue(started.wait(0.5))

            second = writer.submit(self.sample_frame(), {"sample": 2})
            third = writer.submit(self.sample_frame(), {"sample": 3})
            self.assertTrue(second.accepted)
            self.assertEqual(third.status, "mailbox_full")
        finally:
            release.set()
            self.assertTrue(writer.close(timeout=2.0))

        self.assertEqual(writer.completed_count, 2)
        self.assertEqual(
            len(list(writer.session_dir.glob("*.png"))), 2)
        self.assertEqual(
            len(list(writer.session_dir.glob("*.json"))), 2)

    def test_submit_snapshots_frame_and_metadata(self):
        writer = RescueDatasetWriter(
            self.dataset_root, session_id="snapshot")
        frame = self.sample_frame()
        original_frame = frame.copy()
        metadata = {"position": {"id": "centro"}}

        submitted = writer.submit(frame, metadata)
        frame[:] = 255
        metadata["position"]["id"] = "alterado"
        self.assertTrue(writer.close(timeout=2.0))

        restored = cv2.imread(
            str(writer.session_dir / f"{submitted.capture_id}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        sidecar = json.loads(
            (writer.session_dir / f"{submitted.capture_id}.json")
            .read_text(encoding="utf-8"))
        self.assertTrue(np.array_equal(restored, original_frame))
        self.assertEqual(
            sidecar["metadata"]["position"]["id"], "centro")

    def test_close_timeout_is_bounded_and_rejects_new_submit(self):
        writer = RescueDatasetWriter(
            self.dataset_root, session_id="close-timeout")
        started = threading.Event()
        release = threading.Event()

        def blocking_write(snapshot):
            del snapshot
            started.set()
            release.wait(2.0)

        writer._write_snapshot = blocking_write
        writer.submit(self.sample_frame())
        self.assertTrue(started.wait(0.5))

        started_close = time.monotonic()
        self.assertFalse(writer.close(timeout=0.02))
        self.assertLess(time.monotonic() - started_close, 0.25)
        self.assertEqual(
            writer.submit(self.sample_frame()).status, "closed")

        release.set()
        self.assertTrue(writer.close(timeout=1.0))

    def test_write_failure_does_not_kill_worker(self):
        writer = RescueDatasetWriter(
            self.dataset_root, session_id="recover")
        original_write = writer._write_snapshot
        first_attempt = threading.Event()
        calls = {"count": 0}

        def fail_once(snapshot):
            calls["count"] += 1
            if calls["count"] == 1:
                first_attempt.set()
                raise OSError("disco simulado")
            original_write(snapshot)

        writer._write_snapshot = fail_once
        first = writer.submit(self.sample_frame(), {"sample": 1})
        self.assertTrue(first.accepted)
        self.assertTrue(first_attempt.wait(0.5))

        deadline = time.monotonic() + 0.5
        while writer.failed_count == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        second = writer.submit(self.sample_frame(), {"sample": 2})
        self.assertTrue(second.accepted)
        self.assertTrue(writer.close(timeout=2.0))

        self.assertEqual(writer.failed_count, 1)
        self.assertEqual(writer.completed_count, 1)
        self.assertIsNone(writer.last_error)
        self.assertTrue(
            (writer.session_dir / f"{second.capture_id}.png").exists())

    def test_rejects_output_outside_shadow_without_creating_it(self):
        outside = SHADOW_ROOT.parent / (
            f"outside-rescue-dataset-{time.time_ns()}")
        self.assertFalse(outside.exists())
        with self.assertRaises(ValueError):
            RescueDatasetWriter(outside, session_id="forbidden")
        self.assertFalse(outside.exists())

    def test_rejects_invalid_frame_and_metadata(self):
        writer = RescueDatasetWriter(
            self.dataset_root, session_id="invalid")
        try:
            with self.assertRaises(ValueError):
                writer.submit(np.zeros((2, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                writer.submit(self.sample_frame(), {"bad": float("nan")})
        finally:
            self.assertTrue(writer.close(timeout=1.0))

    def test_metadata_only_marks_exact_capture_result_as_same_frame(self):
        args = SimpleNamespace(
            camera_index=0,
            target="silver",
            no_enhance=False,
            drive=False,
        )
        detection = BallDetection(
            "silver", 320, 300, 42, 0.86,
            True, 3, 10.0,
            track_locked=True,
        )
        result = AsyncDetectionResult(
            sequence=9,
            source_sequence=41,
            detection=detection,
            frame_shape=(480, 640, 3),
            detector_shape=(240, 320, 3),
            captured_at=10.0,
            completed_at=10.012,
            processing_s=0.012,
            dropped_frames=0,
            generation=0,
            hough_used=True,
            contour_proposals=2,
            hough_proposals=5,
            candidate_count=3,
            candidate_radii=(42.0, 31.0, 24.0),
            diagnostic="ok",
            candidate_circles=(
                (320.0, 300.0, 42.0, "silver", 0.90),
                (315.0, 298.0, 31.0, "silver", 0.82),
                (330.0, 302.0, 24.0, "silver", 0.76),
            ),
            crescent_evidence=CloseCrescentEvidence(
                True, 0.90, 0.80, 0.75, 0.90, 0.76,
                42.0, 0.50, 0.74, 0.46, 0.98, 10.0,
                foil_fallback=True,
                foil_texture_bins=4,
                foil_valid_bins=5,
                interior_edge_density=0.08,
                background_edge_density=0.01,
            ),
            locked_detection=detection,
        )
        command = MotionCommand("ALIGN", angle=180, speed=0.35)

        exact = _dataset_metadata(
            args, command, 41, 10.0, result, 10.02)
        newer_frame = _dataset_metadata(
            args, command, 42, 10.03, result, 10.04)

        self.assertTrue(
            exact["latest_detector_result"]["same_frame"])
        self.assertFalse(
            newer_frame["latest_detector_result"]["same_frame"])
        self.assertTrue(exact["raw_unannotated"])
        self.assertEqual(
            exact["latest_detector_result"]["candidate_radii"],
            [42.0, 31.0, 24.0],
        )
        self.assertEqual(
            exact["latest_detector_result"]["candidate_circles"][0],
            [320.0, 300.0, 42.0, "silver", 0.90],
        )
        self.assertTrue(
            exact["latest_detector_result"]["detection"][
                "track_locked"
            ]
        )
        self.assertEqual(
            exact["latest_detector_result"]["locked_detection"][
                "radius"
            ],
            42.0,
        )
        self.assertTrue(
            exact["latest_detector_result"]["crescent_evidence"]["accepted"])
        self.assertEqual(
            exact["latest_detector_result"]["crescent_evidence"]["support"],
            0.80,
        )
        self.assertTrue(
            exact["latest_detector_result"]["crescent_evidence"][
                "foil_fallback"
            ]
        )
        self.assertEqual(
            exact["latest_detector_result"]["crescent_evidence"][
                "foil_texture_bins"
            ],
            4,
        )


if __name__ == "__main__":
    unittest.main()
