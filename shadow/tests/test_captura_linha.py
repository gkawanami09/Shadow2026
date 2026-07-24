"""Testes da seleção da câmera de linha."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config
import config_resgate
from visao.captura import LineCamera


class LineCameraSelectionTests(unittest.TestCase):
    def test_line_and_rescue_use_different_fixed_indices(self):
        self.assertEqual(config.LINE_CAMERA_INDEX, 1)
        self.assertEqual(config_resgate.RESCUE_CAMERA_INDEX, 0)
        self.assertNotEqual(
            config.LINE_CAMERA_INDEX,
            config_resgate.RESCUE_CAMERA_INDEX,
        )

    def test_line_camera_opens_explicit_flat_2_index(self):
        opened_indices = []

        class FakePicamera2:
            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}, {"Model": "line"}]

            def __init__(self, camera_num):
                opened_indices.append(camera_num)

            def create_video_configuration(self, **kwargs):
                return kwargs

            def configure(self, _configuration):
                pass

            def start(self):
                pass

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(
                sys.modules,
                {"picamera2": fake_module},
            ),
            mock.patch("visao.captura.time.sleep"),
        ):
            LineCamera()

        self.assertEqual(opened_indices, [config.LINE_CAMERA_INDEX])

    def test_missing_line_camera_fails_instead_of_opening_rescue(self):
        class FakePicamera2:
            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}]

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with mock.patch.dict(
            sys.modules,
            {"picamera2": fake_module},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "camera de segue-linha",
            ):
                LineCamera()


if __name__ == "__main__":
    unittest.main()
