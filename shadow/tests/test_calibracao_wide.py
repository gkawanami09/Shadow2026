"""Testes sem hardware da calibração geométrica da câmera wide."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from tools.calibrar_camera_wide import (  # noqa: E402
    board_object_points,
    build_parser,
    calibrate_fisheye,
    calculate_ground_homography,
    ground_board_points,
    normalize_corner_order,
    run,
)
from visao.calibracao_wide import (  # noqa: E402
    BOARD_SQUARES,
    CALIBRATION_SCHEMA_VERSION,
    INNER_CORNERS,
    MAX_HOMOGRAPHY_RMS_MM,
    WideCalibration,
    WideCalibrationError,
    WideCalibrationMetadata,
    build_capture_mode_id,
    carregar_calibracao,
    load_wide_calibration,
)


TEST_CAPTURE_MODE = (
    "LineCamera:64x48@40.00:full-fov;"
    "sensor-mode=2304x1296x10;crop=0,0,4608,2592"
)


def make_metadata(**changes):
    base = WideCalibrationMetadata(
        image_size=(64, 48),
        camera_index=1,
        sensor="imx708_wide",
        capture_mode=TEST_CAPTURE_MODE,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        rms_px=0.42,
        homography_rms_mm=0.3,
        lens_position=13.5,
        view_count=20,
        balance=1.0,
    )
    return replace(base, **changes)


def make_calibration(metadata=None, **matrix_changes):
    camera_matrix = np.array(
        [[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    values = {
        "camera_matrix": camera_matrix,
        "distortion": np.zeros((4, 1), dtype=np.float64),
        "rectified_matrix": camera_matrix.copy(),
        "pixel_to_ground": np.array(
            [[0.5, 0.0, -16.0], [0.0, -0.5, 24.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "metadata": metadata or make_metadata(),
    }
    values.update(matrix_changes)
    return WideCalibration(**values)


class WideCalibrationArtifactTests(unittest.TestCase):
    def test_cli_usa_o_caminho_central_do_config(self):
        args = build_parser().parse_args([])
        self.assertEqual(
            Path(args.output),
            Path(config.GREEN_WIDE_CALIBRATION_PATH),
        )

    def test_modo_offline_exige_assinatura_real_do_modo(self):
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images="capturas",
            homography_image="plano.png",
            capture_mode=None,
            sensor="imx708_wide",
            lens_position=13.5,
        )
        with self.assertRaisesRegex(WideCalibrationError, "--capture-mode"):
            run(args)

    def test_modo_offline_exige_identidade_real_do_sensor(self):
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images="capturas",
            homography_image="plano.png",
            capture_mode="LineCamera:448x252@40.00:full-fov",
            sensor=None,
            lens_position=13.5,
        )
        with self.assertRaisesRegex(WideCalibrationError, "--sensor"):
            run(args)

    def test_modo_offline_exige_lens_position(self):
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images="capturas",
            homography_image="plano.png",
            capture_mode=(
                "LineCamera:448x252@40.00:full-fov;"
                "sensor-mode=2304x1296x10"
            ),
            sensor="imx708_wide",
            lens_position=None,
        )
        with self.assertRaisesRegex(WideCalibrationError, "--lens-position"):
            run(args)

    def test_sensor_unknown_nunca_gera_calibracao_offline(self):
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images="capturas",
            homography_image="plano.png",
            capture_mode=(
                "LineCamera:448x252@40.00:full-fov;"
                "sensor-mode=2304x1296x10"
            ),
            sensor="unknown",
            lens_position=13.5,
        )
        with self.assertRaisesRegex(WideCalibrationError, "sensor"):
            run(args)

    def test_sensor_diferente_da_wide_nunca_gera_artefato(self):
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images="capturas",
            homography_image="plano.png",
            capture_mode=(
                "LineCamera:448x252@40.00:full-fov;"
                "sensor-mode=2304x1296x10"
            ),
            sensor="ov5647",
            lens_position=13.5,
        )
        with self.assertRaisesRegex(WideCalibrationError, "Wide esperada"):
            run(args)

    def test_sensor_unknown_da_camera_nao_pode_ser_sobrescrito(self):
        camera = SimpleNamespace(
            sensor_id="unknown",
            sensor_mode={"output_size": (2304, 1296), "bit_depth": 10},
            lens_position=13.5,
            close=mock.Mock(),
        )
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images=None,
            homography_image=None,
            capture_mode=None,
            sensor="imx708_wide",
            lens_position=None,
        )
        with mock.patch(
            "tools.calibrar_camera_wide.LineCamera",
            return_value=camera,
        ), self.assertRaisesRegex(WideCalibrationError, "sensor"):
            run(args)
        camera.close.assert_called_once_with()

    def test_modo_bruto_nao_confirmado_bloqueia_calibracao_ao_vivo(self):
        camera = SimpleNamespace(
            sensor_id="imx708_wide",
            sensor_mode=None,
            lens_position=13.5,
            close=mock.Mock(),
        )
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images=None,
            homography_image=None,
            capture_mode=None,
            sensor=None,
            lens_position=None,
        )
        with mock.patch(
            "tools.calibrar_camera_wide.LineCamera",
            return_value=camera,
        ), self.assertRaisesRegex(WideCalibrationError, "modo bruto"):
            run(args)
        camera.close.assert_called_once_with()

    def test_foco_nao_confirmado_bloqueia_calibracao_ao_vivo(self):
        camera = SimpleNamespace(
            sensor_id="imx708_wide",
            sensor_mode={"output_size": (2304, 1296), "bit_depth": 10},
            lens_position=None,
            close=mock.Mock(),
        )
        args = SimpleNamespace(
            views=20,
            balance=1.0,
            images=None,
            homography_image=None,
            capture_mode=None,
            sensor=None,
            lens_position=None,
        )
        with mock.patch(
            "tools.calibrar_camera_wide.LineCamera",
            return_value=camera,
        ), self.assertRaisesRegex(WideCalibrationError, "LensPosition"):
            run(args)
        camera.close.assert_called_once_with()

    def test_roundtrip_npz_preserva_matrizes_e_metadados(self):
        calibration = make_calibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.npz"
            calibration.save(path)
            loaded = load_wide_calibration(
                path,
                expected_image_size=(64, 48),
                expected_camera_index=1,
                expected_sensor="IMX708_WIDE",
                expected_capture_mode=TEST_CAPTURE_MODE,
                expected_lens_position=13.5,
            )

        np.testing.assert_allclose(loaded.camera_matrix, calibration.camera_matrix)
        np.testing.assert_allclose(loaded.distortion, calibration.distortion)
        np.testing.assert_allclose(
            loaded.rectified_matrix,
            calibration.rectified_matrix,
        )
        np.testing.assert_allclose(
            loaded.pixel_to_ground,
            calibration.pixel_to_ground,
        )
        self.assertEqual(loaded.metadata.sensor, "imx708_wide")
        self.assertEqual(loaded.metadata.camera_index, 1)
        self.assertAlmostEqual(loaded.metadata.homography_rms_mm, 0.3)
        self.assertAlmostEqual(loaded.metadata.lens_position, 13.5)
        self.assertEqual(loaded.metadata.board_squares, BOARD_SQUARES)
        self.assertEqual(loaded.metadata.inner_corners, INNER_CORNERS)

    def test_api_curta_valida_toda_a_compatibilidade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.npz"
            make_calibration().save(path)
            loaded = carregar_calibracao(
                path,
                resolution=(64, 48),
                camera_index=1,
                sensor="imx708_wide",
                mode=TEST_CAPTURE_MODE,
                lens_position=13.5,
            )
        self.assertEqual(loaded.image_size, (64, 48))

    def test_npz_nao_contem_objetos_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.npz"
            make_calibration().save(path)
            with np.load(path, allow_pickle=False) as artifact:
                self.assertEqual(
                    int(np.asarray(artifact["schema_version"]).item()),
                    CALIBRATION_SCHEMA_VERSION,
                )
                self.assertTrue(all(array.dtype.kind != "O" for array in artifact.values()))

    def test_carregamento_rejeita_arquivo_incompleto(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incompleto.npz"
            np.savez(path, schema_version=CALIBRATION_SCHEMA_VERSION)
            with self.assertRaises(WideCalibrationError):
                WideCalibration.load(path)

    def test_carregamento_rejeita_schema_desconhecido(self):
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.npz"
            invalid_path = Path(directory) / "invalid.npz"
            make_calibration().save(valid_path)
            with np.load(valid_path, allow_pickle=False) as artifact:
                values = {key: np.asarray(value).copy() for key, value in artifact.items()}
            values["schema_version"] = np.int32(99)
            np.savez_compressed(invalid_path, **values)
            with self.assertRaisesRegex(WideCalibrationError, "versão"):
                WideCalibration.load(invalid_path)

    def test_qualidade_e_quantidade_minimas_sao_obrigatorias(self):
        for metadata in (
            make_metadata(rms_px=0.8001),
            make_metadata(rms_px=float("nan")),
            make_metadata(view_count=19),
            make_metadata(view_count=20.5),
            make_metadata(homography_rms_mm=MAX_HOMOGRAPHY_RMS_MM + 0.001),
            make_metadata(homography_rms_mm=float("nan")),
            make_metadata(lens_position=float("nan")),
            make_metadata(lens_position=float("inf")),
            make_metadata(sensor="unknown"),
            make_metadata(sensor=" "),
            make_metadata(capture_mode="LineCamera:64x48@40.00:full-fov"),
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(WideCalibrationError):
                    make_calibration(metadata)

    def test_matrizes_invalidas_sao_rejeitadas(self):
        cases = (
            {"camera_matrix": np.eye(2)},
            {"distortion": np.zeros(5)},
            {"rectified_matrix": np.zeros((3, 3))},
            {"pixel_to_ground": np.zeros((3, 3))},
            {"pixel_to_ground": np.full((3, 3), np.nan)},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(WideCalibrationError):
                    make_calibration(**changes)

    def test_compatibilidade_exige_resolucao_sensor_e_modo(self):
        calibration = make_calibration()
        self.assertTrue(calibration.validate_compatibility((64, 48)))
        for kwargs in (
            {"image_size": (65, 48)},
            {"image_size": (64.5, 48)},
            {"image_size": (64, 48), "sensor": "ov5647"},
            {"image_size": (64, 48), "sensor": "unknown"},
            {"image_size": (64, 48), "camera_index": 0},
            {"image_size": (64, 48), "capture_mode": "crop"},
            {"image_size": (64, 48), "lens_position": 14.0},
        ):
            with self.subTest(kwargs=kwargs):
                image_size = kwargs.pop("image_size")
                with self.assertRaises(WideCalibrationError):
                    calibration.validate_compatibility(image_size, **kwargs)

    def test_identificador_de_modo_e_estavel(self):
        self.assertEqual(
            build_capture_mode_id((448, 252), 40),
            "LineCamera:448x252@40.00:full-fov",
        )
        self.assertEqual(
            build_capture_mode_id(
                (448, 252),
                40,
                sensor_mode={
                    "output_size": (2304, 1296),
                    "bit_depth": 10,
                },
                scaler_crop=(0, 0, 4608, 2592),
            ),
            "LineCamera:448x252@40.00:full-fov;"
            "sensor-mode=2304x1296x10;crop=0,0,4608,2592",
        )

    def test_limites_publicos_vem_do_config(self):
        self.assertEqual(
            MAX_HOMOGRAPHY_RMS_MM,
            config.GREEN_WIDE_HOMOGRAPHY_MAX_ERROR_MM,
        )


class WideCalibrationGeometryTests(unittest.TestCase):
    def test_calibracao_fisheye_recupera_camera_sintetica(self):
        object_points = board_object_points()
        expected_camera = np.array(
            [[350.0, 0.0, 320.0], [0.0, 350.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        expected_distortion = np.array([[-0.03], [0.005], [0.0], [0.0]])
        corner_sets = []
        for index in range(20):
            rotation = np.array(
                [
                    -0.35 + 0.035 * index,
                    0.18 * np.sin(index * 0.7),
                    -0.25 + 0.025 * index,
                ],
                dtype=np.float64,
            )
            translation = np.array(
                [
                    [-55.0 + (index % 5) * 20.0],
                    [-35.0 + (index // 5) * 18.0],
                    [280.0 + (index % 4) * 20.0],
                ],
                dtype=np.float64,
            )
            pixels, _ = cv2.fisheye.projectPoints(
                object_points,
                rotation,
                translation,
                expected_camera,
                expected_distortion,
            )
            corner_sets.append(pixels)

        rms_px, camera, distortion, rectified = calibrate_fisheye(
            corner_sets,
            (640, 480),
        )
        self.assertLess(rms_px, 1e-6)
        np.testing.assert_allclose(camera, expected_camera, atol=1e-5)
        np.testing.assert_allclose(distortion, expected_distortion, atol=1e-5)
        self.assertEqual(rectified.shape, (3, 3))

    def test_pixel_chao_pixel_faz_roundtrip(self):
        calibration = make_calibration()
        pixels = np.array([[32.0, 48.0], [42.0, 28.0], [12.5, 6.25]])
        ground = calibration.pixels_to_ground_mm(pixels)
        expected = np.array([[0.0, 0.0], [5.0, 10.0], [-9.75, 20.875]])
        np.testing.assert_allclose(ground, expected, atol=1e-10)
        np.testing.assert_allclose(
            calibration.ground_mm_to_pixels(ground),
            pixels,
            atol=1e-10,
        )

    def test_retificar_e_desretificar_pontos_faz_roundtrip(self):
        calibration = make_calibration(
            distortion=np.array(
                [[-0.04], [0.006], [-0.001], [0.0002]],
                dtype=np.float64,
            ),
            rectified_matrix=np.array(
                [[54.0, 0.0, 31.0], [0.0, 55.0, 23.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )
        raw_pixels = np.array(
            [[5.0, 4.0], [32.0, 24.0], [58.0, 43.0], [16.0, 35.0]],
            dtype=np.float64,
        )
        rectified = calibration.rectify_points(raw_pixels)
        recovered = calibration.unrectify_points(rectified)

        self.assertEqual(recovered.shape, raw_pixels.shape)
        np.testing.assert_allclose(recovered, raw_pixels, atol=1e-8)

    def test_retificacao_preserva_shape_e_rejeita_resolucao_errada(self):
        calibration = make_calibration()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        rectified = calibration.rectify(frame, interpolation=cv2.INTER_NEAREST)
        self.assertEqual(rectified.shape, frame.shape)
        mask = calibration.rectify_mask(frame[:, :, 0])
        self.assertEqual(mask.shape, frame.shape[:2])
        with self.assertRaisesRegex(WideCalibrationError, "resolução incompatível"):
            calibration.rectify(np.zeros((47, 64, 3), dtype=np.uint8))

    def test_warp_ground_respeita_frente_para_cima(self):
        calibration = make_calibration()
        rectified = np.zeros((48, 64), dtype=np.uint8)
        # chão (0, 10) corresponde ao pixel retificado (32, 28)
        rectified[28, 32] = 255
        top_view = calibration.warp_ground(
            rectified,
            x_limits_mm=(-10, 10),
            y_limits_mm=(0, 20),
            pixels_per_mm=1,
            frame_is_rectified=True,
            interpolation=cv2.INTER_NEAREST,
        )
        self.assertEqual(top_view.shape, (20, 20))
        self.assertEqual(int(top_view[10, 10]), 255)

    def test_pontos_do_tabuleiro_usam_x_direita_y_frente(self):
        points = ground_board_points().reshape(INNER_CORNERS[1], INNER_CORNERS[0], 2)
        self.assertEqual(tuple(points[0, 0]), (-30.0, 40.0))
        self.assertEqual(tuple(points[-1, -1]), (30.0, 0.0))
        self.assertTrue(np.all(np.diff(points[0, :, 0]) > 0))
        self.assertTrue(np.all(np.diff(points[:, 0, 1]) < 0))

    def test_ordem_dos_cantos_e_normalizada_pelos_eixos_da_imagem(self):
        columns, rows = INNER_CORNERS
        canonical = np.asarray(
            [(20 + column * 10, 15 + row * 12)
             for row in range(rows) for column in range(columns)],
            dtype=np.float64,
        ).reshape(rows, columns, 2)
        reversed_both = canonical[::-1, ::-1, :].reshape(-1, 1, 2)
        normalized = normalize_corner_order(reversed_both)
        np.testing.assert_allclose(normalized.reshape(rows, columns, 2), canonical)

    def test_homografia_sintetica_recupera_milimetros(self):
        ground = ground_board_points().reshape(-1, 2)
        pixels = np.column_stack((
            100.0 + 3.0 * ground[:, 0],
            200.0 - 3.0 * ground[:, 1],
        )).reshape(-1, 1, 2)
        homography, rms_mm = calculate_ground_homography(pixels)
        projected = cv2.perspectiveTransform(pixels, homography)
        np.testing.assert_allclose(projected, ground.reshape(-1, 1, 2), atol=1e-8)
        self.assertLess(rms_mm, 1e-8)


if __name__ == "__main__":
    unittest.main()
