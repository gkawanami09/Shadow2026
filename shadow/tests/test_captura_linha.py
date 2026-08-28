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
from visao.captura import (LineCamera, escolher_fps_captura,
                           escolher_modo_sensor_campo_aberto,
                           normalizar_recorte_metadata,
                           normalizar_identidade_sensor,
                           obter_recorte_maximo)


class LineCameraSelectionTests(unittest.TestCase):
    def test_identidade_real_do_sensor_e_canonica(self):
        self.assertEqual(
            normalizar_identidade_sensor({"Model": " IMX708-Wide "}),
            "imx708_wide",
        )
        self.assertEqual(
            normalizar_identidade_sensor({"SensorModel": "IMX 708 Wide"}),
            "imx_708_wide",
        )
        self.assertEqual(normalizar_identidade_sensor({}), "unknown")
        self.assertEqual(normalizar_identidade_sensor(None), "unknown")

    def test_captura_e_processamento_preservam_aspecto_16_por_9(self):
        self.assertEqual(config.CAPTURE_WIDTH * 9, config.CAPTURE_HEIGHT * 16)
        self.assertEqual(config.camera_x * 9, config.camera_y * 16)
        self.assertEqual(
            (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
            (config.camera_x, config.camera_y),
        )

    def test_recorte_maximo_aceita_apenas_retangulo_valido(self):
        self.assertEqual(
            obter_recorte_maximo({"ScalerCrop": ((0, 0, 1, 1),
                                                  (0, 0, 4608, 2592))}),
            (0, 0, 4608, 2592),
        )
        self.assertEqual(
            normalizar_recorte_metadata(SimpleNamespace(
                x=0, y=0, width=4608, height=2592,
            )),
            (0, 0, 4608, 2592),
        )

    def test_modo_sem_crop_prefere_sensor_inteiro_ao_modo_rapido_recortado(self):
        modos = (
            {
                "size": (1536, 864), "bit_depth": 10, "fps": 120.13,
                "crop_limits": (768, 432, 3072, 1728),
            },
            {
                "size": (2304, 1296), "bit_depth": 10, "fps": 56.03,
                "crop_limits": (0, 0, 4608, 2592),
            },
            {
                "size": (4608, 2592), "bit_depth": 10, "fps": 14.35,
                "crop_limits": (0, 0, 4608, 2592),
            },
        )
        self.assertEqual(
            escolher_modo_sensor_campo_aberto(modos, 50),
            {"output_size": (2304, 1296), "bit_depth": 10},
        )
        self.assertIsNone(escolher_modo_sensor_campo_aberto(modos, 60))
        self.assertIsNone(obter_recorte_maximo({}))
        self.assertIsNone(
            obter_recorte_maximo({"ScalerCrop": ((0, 0, 1, 1),
                                                  (0, 0, 0, 2592))})
        )

    def test_fps_nunca_ultrapassa_modo_vga_anunciado(self):
        casos = (
            ([{"size": (640, 480), "fps": 90}], 40.),
            ([{"size": (640, 480), "fps": 55}], 40.),
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

    def test_modo_full_fov_configura_quarenta_fps(self):
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

        self.assertEqual(camera.capture_fps, 40.)
        self.assertEqual(
            configuracoes[0]["controls"]["FrameDurationLimits"],
            (25000, 25000),
        )
        self.assertFalse(configuracoes[0]["queue"])

    def test_camera_expoe_sensor_modo_e_crop_reais(self):
        configuracoes = []
        controles = []

        class FakePicamera2:
            sensor_modes = (
                {
                    "size": (1536, 864),
                    "bit_depth": 10,
                    "fps": 120.0,
                    "crop_limits": (768, 432, 3072, 1728),
                },
                {
                    "size": (2304, 1296),
                    "bit_depth": 10,
                    "fps": 56.0,
                    "crop_limits": (0, 0, 4608, 2592),
                },
            )
            camera_controls = {
                "ScalerCrop": ((0, 0, 1, 1), (0, 0, 4608, 2592)),
            }

            @staticmethod
            def global_camera_info():
                return [
                    {"Model": "ov5647"},
                    {"Model": "IMX708 Wide"},
                ]

            def __init__(self, camera_num):
                self.camera_num = camera_num
                self.configuration = None

            def create_video_configuration(self, **kwargs):
                configuracoes.append(kwargs)
                return kwargs

            def configure(self, configuration):
                self.configuration = configuration

            def camera_configuration(self):
                return self.configuration

            def start(self):
                pass

            def set_controls(self, values):
                controles.append(values)

            @staticmethod
            def capture_metadata():
                return {
                    "ScalerCrop": (0, 0, 4608, 2592),
                    "FrameDuration": 25000,
                }

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(sys.modules, {"picamera2": fake_module}),
            mock.patch("visao.captura.time.sleep"),
        ):
            camera = LineCamera()

        self.assertEqual(camera.camera_index, config.LINE_CAMERA_INDEX)
        self.assertEqual(camera.sensor_id, "imx708_wide")
        self.assertEqual(
            camera.sensor_mode,
            {"output_size": (2304, 1296), "bit_depth": 10},
        )
        self.assertEqual(camera.scaler_crop, (0, 0, 4608, 2592))
        self.assertEqual(
            camera.capture_mode_id,
            "LineCamera:448x252@40.00:full-fov;"
            "sensor-mode=2304x1296x10;crop=0,0,4608,2592",
        )
        self.assertIn({"ScalerCrop": (0, 0, 4608, 2592)}, controles)
        self.assertEqual(
            configuracoes[0]["sensor"],
            {"output_size": (2304, 1296), "bit_depth": 10},
        )

    def test_assinatura_competitiva_exige_readback_de_crop_fps_e_sensor(self):
        camera = LineCamera.__new__(LineCamera)
        camera.sensor_id = config.LINE_CAMERA_SENSOR_ID
        camera._sensor_mode_applied = {
            "output_size": (2304, 1296), "bit_depth": 10,
        }
        camera._main_stream_applied = None
        camera._scaler_crop_applied = None
        camera._capture_fps_confirmed = None
        camera.capture_fps = 40.
        with self.assertRaisesRegex(RuntimeError, "stream principal"):
            _ = camera.capture_mode_id

        camera._main_stream_applied = {
            "size": (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
            "format": "RGB888",
        }
        with self.assertRaisesRegex(RuntimeError, "ScalerCrop"):
            _ = camera.capture_mode_id

        camera._scaler_crop_applied = (0, 0, 4608, 2592)
        with self.assertRaisesRegex(RuntimeError, "FrameDuration"):
            _ = camera.capture_mode_id

        camera._capture_fps_confirmed = 40.
        camera.sensor_id = "ov5647"
        with self.assertRaisesRegex(RuntimeError, "sensor"):
            _ = camera.capture_mode_id

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
                if duracao == 20000:
                    raise RuntimeError("modo recusado")

            def stop(self):
                pass

            def close(self):
                pass

        fake_module = SimpleNamespace(Picamera2=FakePicamera2)
        with (
            mock.patch.dict(sys.modules, {"picamera2": fake_module}),
            mock.patch("visao.captura.CAPTURE_FPS", 50),
            mock.patch("visao.captura.CAPTURE_FPS_FALLBACK", 40),
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

        self.assertEqual(camera.capture_fps, 40.)
        self.assertEqual(
            configuracoes[0]["controls"]["FrameDurationLimits"],
            (25000, 25000),
        )

    def test_pwm_fixo_mantem_captura_em_quarenta_fps(self):
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

        self.assertEqual(camera.capture_fps, 40.)
        self.assertEqual(
            configuracoes[0]["controls"]["FrameDurationLimits"],
            (25000, 25000),
        )

    def test_autofocus_roda_uma_vez_e_depois_fica_manual(self):
        controles_aplicados = []
        ciclos_af = []

        class FakePicamera2:
            sensor_modes = [{"size": (640, 480), "fps": 90}]
            camera_controls = {}

            @staticmethod
            def global_camera_info():
                return [{"Model": "rescue"}, {"Model": "line"}]

            def __init__(self, camera_num):
                self.camera_num = camera_num

            def create_video_configuration(self, **kwargs):
                return kwargs

            def configure(self, _configuration):
                pass

            def start(self):
                pass

            def set_controls(self, values):
                controles_aplicados.append(values)

            def autofocus_cycle(self):
                ciclos_af.append(True)
                return True

            def capture_metadata(self):
                return {"LensPosition": 12.5}

        af_mode = SimpleNamespace(Continuous="continuous", Manual="manual")
        controls = SimpleNamespace(
            AfModeEnum=af_mode,
            AfRangeEnum=SimpleNamespace(Full="full"),
        )
        with (
            mock.patch.dict(sys.modules, {
                "picamera2": SimpleNamespace(Picamera2=FakePicamera2),
                "libcamera": SimpleNamespace(controls=controls),
            }),
            mock.patch("visao.captura.LENS_POSITION", None),
            mock.patch("visao.captura.time.sleep"),
        ):
            camera = LineCamera()

        self.assertEqual(ciclos_af, [True])
        self.assertIn({"AfRange": "full"}, controles_aplicados)
        self.assertIn(
            {"AfMode": "manual", "LensPosition": 12.5},
            controles_aplicados,
        )
        self.assertNotIn({"AfMode": "continuous"}, controles_aplicados)
        self.assertAlmostEqual(camera.lens_position, 12.5)

    def test_aplica_e_confirma_lens_position_salva(self):
        controles_aplicados = []

        class FakeCamera:
            def set_controls(self, values):
                controles_aplicados.append(values)

            @staticmethod
            def capture_metadata():
                return {"LensPosition": 13.625}

        camera = LineCamera.__new__(LineCamera)
        camera.picam2 = FakeCamera()
        camera._lens_position_confirmed = None
        controls = SimpleNamespace(
            AfModeEnum=SimpleNamespace(Manual="manual"),
        )
        with mock.patch.dict(
            sys.modules,
            {"libcamera": SimpleNamespace(controls=controls)},
        ):
            confirmada = camera.aplicar_posicao_lente(13.64)

        self.assertAlmostEqual(confirmada, 13.625)
        self.assertAlmostEqual(camera.lens_position, 13.625)
        self.assertEqual(
            controles_aplicados,
            [{"AfMode": "manual", "LensPosition": 13.64}],
        )

    def test_lens_position_nao_confirmada_falha_fechado(self):
        class FakeCamera:
            @staticmethod
            def set_controls(_values):
                pass

            @staticmethod
            def capture_metadata():
                return {"LensPosition": 8.0}

        camera = LineCamera.__new__(LineCamera)
        camera.picam2 = FakeCamera()
        camera._lens_position_confirmed = None
        controls = SimpleNamespace(
            AfModeEnum=SimpleNamespace(Manual="manual"),
        )
        with (
            mock.patch.dict(
                sys.modules,
                {"libcamera": SimpleNamespace(controls=controls)},
            ),
            mock.patch("visao.captura.time.sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "não foi confirmada"):
                camera.aplicar_posicao_lente(13.5, tentativas=2)

        self.assertIsNone(camera.lens_position)

    def test_assinatura_competitiva_exige_modo_bruto_confirmado(self):
        camera = LineCamera.__new__(LineCamera)
        camera._sensor_mode_applied = None
        camera._scaler_crop_applied = None
        camera._capture_fps_confirmed = None
        camera.sensor_id = config.LINE_CAMERA_SENSOR_ID
        camera.capture_fps = 40.
        with self.assertRaisesRegex(RuntimeError, "modo bruto"):
            _ = camera.capture_mode_id

    def test_frame_no_tamanho_do_algoritmo_nao_e_redimensionado(self):
        class FrameFalso:
            ndim = 3
            shape = (config.camera_y, config.camera_x, 3)

        frame = FrameFalso()
        camera = LineCamera.__new__(LineCamera)
        camera.picam2 = SimpleNamespace(
            capture_array=lambda _stream: frame,
        )

        devolvido = camera.get_frame()

        self.assertIs(devolvido, frame)

    def test_frame_com_geometria_inesperada_falha_sem_esticar(self):
        class FrameFalso:
            ndim = 3
            shape = (480, 640, 3)

        camera = LineCamera.__new__(LineCamera)
        camera.picam2 = SimpleNamespace(
            capture_array=lambda _stream: FrameFalso(),
        )

        with self.assertRaisesRegex(RuntimeError, "mudou de geometria"):
            camera.get_frame()

    def test_frame_com_quatro_canais_falha_sem_mascarar_formato(self):
        class FrameFalso:
            ndim = 3
            shape = (config.camera_y, config.camera_x, 4)

        camera = LineCamera.__new__(LineCamera)
        camera.picam2 = SimpleNamespace(
            capture_array=lambda _stream: FrameFalso(),
        )

        with self.assertRaisesRegex(RuntimeError, "mudou de geometria"):
            camera.get_frame()

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
