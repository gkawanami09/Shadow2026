"""Testes da câmera de resgate."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.captura_resgate import (  # noqa: E402
    RescueCamera,
    _fit_output_size,
    _known_sensor_mode,
    _select_widest_sensor_mode,
)
import config_resgate  # noqa: E402


class RescueCaptureConfigurationTests(unittest.TestCase):
    @staticmethod
    def _camera_with_frame(frame, rotate=False):
        class FakePicamera:
            @staticmethod
            def capture_array(_stream):
                return frame.copy()

        camera = RescueCamera.__new__(RescueCamera)
        camera.picam2 = FakePicamera()
        camera._software_rotate = rotate
        return camera

    def test_output_preserves_full_sensor_aspect(self):
        self.assertEqual(_fit_output_size(4 / 3, 960, 720), (960, 720))
        self.assertEqual(_fit_output_size(16 / 9, 960, 720), (960, 540))

    def test_ov5647_uses_known_full_fov_mode_without_discovery(self):
        mode = _known_sensor_mode({"Model": "ov5647"})
        self.assertEqual(mode["size"], (1296, 972))
        self.assertEqual(mode["bit_depth"], 10)
        self.assertEqual(mode["crop_limits"], (0, 0, 2592, 1944))

    def test_prefers_full_fov_mode_that_sustains_target_fps(self):
        cropped_fast = {
            "size": (1920, 1080),
            "crop_limits": (680, 692, 1920, 1080),
            "fps": 60.0,
            "bit_depth": 10,
        }
        full_binned = {
            "size": (1640, 1232),
            "crop_limits": (0, 0, 3280, 2464),
            "fps": 40.0,
            "bit_depth": 10,
        }
        full_photo = {
            "size": (3280, 2464),
            "crop_limits": (0, 0, 3280, 2464),
            "fps": 21.0,
            "bit_depth": 10,
        }

        selected = _select_widest_sensor_mode(
            [cropped_fast, full_photo, full_binned],
            target_fps=30,
            max_output_width=960,
            max_output_height=720,
        )

        self.assertIs(selected, full_binned)

    def test_discovery_does_not_call_98_percent_crop_full_fov(self):
        almost_full_fast = {
            "size": (640, 480),
            "crop_limits": (16, 0, 2560, 1920),
            "fps": 90.0,
            "bit_depth": 10,
        }
        full_fov = {
            "size": (1296, 972),
            "crop_limits": (0, 0, 2592, 1944),
            "fps": 43.25,
            "bit_depth": 10,
        }
        selected = _select_widest_sensor_mode(
            [almost_full_fast, full_fov],
            target_fps=30,
            max_output_width=640,
            max_output_height=480,
        )
        self.assertIs(selected, full_fov)

    def test_rgb888_memory_order_is_already_bgr_for_opencv(self):
        frame = np.array([[[7, 31, 211]]], dtype=np.uint8)
        result = self._camera_with_frame(frame).get_frame()
        np.testing.assert_array_equal(result, frame)

    def test_software_fallback_rotates_180_degrees(self):
        frame = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], dtype=np.uint8)
        result = self._camera_with_frame(frame, rotate=True).get_frame()
        np.testing.assert_array_equal(
            result,
            np.array([
                [[10, 11, 12], [7, 8, 9]],
                [[4, 5, 6], [1, 2, 3]],
            ], dtype=np.uint8))

    def test_configuration_requires_fresh_frames_and_exact_sensor_mode(self):
        class FakePicamera:
            def __init__(self):
                self.kwargs = None

            def create_video_configuration(self, **kwargs):
                self.kwargs = kwargs
                return kwargs

        camera = RescueCamera.__new__(RescueCamera)
        camera.picam2 = FakePicamera()
        camera.output_size = (640, 480)
        camera._software_rotate = False
        mode = {
            "size": (1296, 972),
            "bit_depth": 10,
        }
        with patch.object(config_resgate, "RESCUE_ROTATE_180", False):
            configuration = camera._create_configuration(mode, 33333)

        self.assertFalse(configuration["queue"])
        self.assertEqual(configuration["buffer_count"], 4)
        self.assertEqual(
            configuration["sensor"],
            {"output_size": (1296, 972), "bit_depth": 10})


if __name__ == "__main__":
    unittest.main()
