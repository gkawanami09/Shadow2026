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
from visao.captura import LineCamera, escolher_fps_captura


class LineCameraSelectionTests(unittest.TestCase):
    def test_fps_nunca_ultrapassa_modo_vga_anunciado(self):
        casos = (
            ([{"size": (640, 480), "fps": 90}], 60.),
            ([{"size": (640, 480), "fps": 55}], 55.),
            ([{"size": (640, 480), "fps": 40}], 40.),
            ([{"size": (640, 480), "fps": 30}], 30.),
            ([
                {"size": (320, 240), "fps": 120},
                {"size": (640, 480), "fps": 40},
            ], 40.),
        )

        for modos, esperado in casos:
            with self.subTest(modos=modos):
                self.assertEqual(escolher_fps_captura(modos), esperado)

    def test_modos_ausentes_ou_invalidos_usam_fallback(self):
        for modos in (
            None,
            [],
            [{}],
            [{"size": (640, 480), "fps": 0}],
            [{"size": (640, 480), "fps": float("nan")}],
        ):
            with self.subTest(modos=modos):
                self.assertEqual(
                    escolher_fps_captura(modos),
                    float(config.CAPTURE_FPS_FALLBACK),
                )

    def test_modo_vga_rapido_configura_sessenta_fps(self):
        configuracoes = []

        class FakePicamera2:
            sensor_modes = [{"size": (640, 480), "fps": 90}]

            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}, {"Model": "line"}]

            def __init__(self, camera_num):
                self.camera_num = camera_num

            def create_video_configuration(self, **kwargs):
                configuracoes.append(kwargs)
                return kwargs

            def configure(self, _configuration):
                pass

            def start(self):
                pass

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(sys.modules, {"picamera2": fake_module}),
            mock.patch("visao.captura.time.sleep"),
        ):
            camera = LineCamera()

        self.assertEqual(camera.capture_fps, 60.)
        self.assertEqual(
            configuracoes[0]["controls"]["FrameDurationLimits"],
            (16667, 16667),
        )
        self.assertFalse(configuracoes[0]["queue"])

    def test_driver_que_recusa_modo_rapido_reabre_em_quarenta_fps(self):
        configuracoes = []
        aberturas = []

        class FakePicamera2:
            sensor_modes = [{"size": (640, 480), "fps": 90}]

            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}, {"Model": "line"}]

            def __init__(self, camera_num):
                self.camera_num = camera_num
                aberturas.append(camera_num)

            def create_video_configuration(self, **kwargs):
                configuracoes.append(kwargs)
                return kwargs

            def configure(self, _configuration):
                pass

            def start(self):
                duracao = configuracoes[-1]["controls"][
                    "FrameDurationLimits"][0]
                if duracao == 16667:
                    raise RuntimeError("modo recusado")

            def stop(self):
                pass

            def close(self):
                pass

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(sys.modules, {"picamera2": fake_module}),
            mock.patch("visao.captura.time.sleep"),
        ):
            camera = LineCamera()

        self.assertEqual(aberturas, [config.LINE_CAMERA_INDEX] * 2)
        self.assertEqual(camera.capture_fps, 40.)
        self.assertEqual(
            configuracoes[-1]["controls"]["FrameDurationLimits"],
            (25000, 25000),
        )

    def test_picamera_antigo_sem_queue_ainda_recebe_controle_de_fps(self):
        configuracoes = []

        class FakePicamera2:
            sensor_modes = [{"size": (640, 480), "fps": 90}]

            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}, {"Model": "line"}]

            def __init__(self, camera_num):
                self.camera_num = camera_num

            def create_video_configuration(self, **kwargs):
                if "queue" in kwargs:
                    raise TypeError("queue desconhecido")
                configuracoes.append(kwargs)
                return kwargs

            def configure(self, _configuration):
                pass

            def start(self):
                pass

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(sys.modules, {"picamera2": fake_module}),
            mock.patch("visao.captura.time.sleep"),
        ):
            camera = LineCamera()

        self.assertEqual(camera.capture_fps, 60.)
        self.assertEqual(
            configuracoes[0]["controls"]["FrameDurationLimits"],
            (16667, 16667),
        )

    def test_pwm_fixo_mantem_captura_em_sessenta_fps(self):
        configuracoes = []

        class FakePicamera2:
            sensor_modes = [{"size": (640, 480), "fps": 90}]

            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}, {"Model": "line"}]

            def __init__(self, camera_num):
                self.camera_num = camera_num

            def create_video_configuration(self, **kwargs):
                configuracoes.append(kwargs)
                return kwargs

            def configure(self, _configuration):
                pass

            def start(self):
                pass

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(sys.modules, {"picamera2": fake_module}),
            mock.patch("visao.captura.time.sleep"),
        ):
            camera = LineCamera()

        self.assertEqual(camera.capture_fps, 60.)
        self.assertEqual(
            configuracoes[0]["controls"]["FrameDurationLimits"],
            (16667, 16667),
        )

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
