import sys
from pathlib import Path
import unittest

import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from vision.rescue_capture import (  # noqa: E402
    RescueCamera,
    _fit_output_size,
    _select_widest_sensor_mode,
)


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


if __name__ == "__main__":
    unittest.main()
