#!/usr/bin/env python3
"""Calibra a Camera Module 3 Wide para topologia de interseções.

Tabuleiro físico exigido: 8 x 6 quadrados de 10 mm. Isso corresponde a
7 x 5 cantos internos no OpenCV.

Uso no robô (interativo):

    python3 tools/calibrar_camera_wide.py \
        --output calibracao_camera_wide.npz

Na primeira etapa, varie posição, distância e inclinação do tabuleiro e
pressione ESPAÇO em 20 vistas diferentes. Na segunda etapa, coloque o
tabuleiro plano no chão, centralizado e alinhado ao eixo do robô, com a borda
mais próxima aparecendo embaixo da imagem, e pressione ESPAÇO novamente.

Também é possível calibrar sem abrir a câmera:

    python3 tools/calibrar_camera_wide.py --images captures/calibracao \
        --homography-image captures/tabuleiro_plano.png \
        --sensor imx708_wide \
        --capture-mode 'LineCamera:448x252@40.00:full-fov;sensor-mode=2304x1296x10;crop=0,0,4608,2592' \
        --lens-position 13.641 \
        --output calibracao_camera_wide.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import (GREEN_WIDE_CALIBRATION_PATH,  # noqa: E402
                    LINE_CAMERA_INDEX, LINE_CAMERA_SENSOR_ID)
from visao.calibracao_wide import (  # noqa: E402
    BOARD_SQUARES,
    INNER_CORNERS,
    MAX_CALIBRATION_RMS_PX,
    MAX_HOMOGRAPHY_RMS_MM,
    MIN_CALIBRATION_VIEWS,
    SQUARE_SIZE_MM,
    WideCalibration,
    WideCalibrationError,
    WideCalibrationMetadata,
    validate_capture_mode_id,
    validate_lens_position,
    validate_sensor_identity,
)
from visao.captura import (LineCamera,  # noqa: E402
                           normalizar_identidade_sensor)


WINDOW = "Shadow2026 - calibracao wide"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MIN_BOARD_AREA_RATIO = 0.025
DEFAULT_OUTPUT = GREEN_WIDE_CALIBRATION_PATH


def _fisheye_flag(name):
    """Compatibilidade entre os namespaces do OpenCV 4 e 5."""
    value = getattr(cv2.fisheye, name, None)
    if value is None:
        value = getattr(cv2, name)
    return value


def board_object_points():
    """Pontos 3D do padrão, no plano Z=0, usados na calibração fisheye."""
    columns, rows = INNER_CORNERS
    points = np.zeros((1, columns * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[0, :, :2] = grid * SQUARE_SIZE_MM
    return points


def ground_board_points(*, center_x_mm=0.0, near_y_mm=0.0):
    """Cantos do tabuleiro no referencial do robô.

    A primeira linha detectada fica no alto da imagem e, com o tabuleiro
    alinhado conforme instruído, é a mais distante do robô. A última linha é
    a mais próxima e recebe ``near_y_mm``.
    """
    columns, rows = INNER_CORNERS
    x_values = (
        np.arange(columns, dtype=np.float64) - (columns - 1) / 2.0
    ) * SQUARE_SIZE_MM + float(center_x_mm)
    y_values = (
        np.arange(rows - 1, -1, -1, dtype=np.float64) * SQUARE_SIZE_MM
        + float(near_y_mm)
    )
    return np.asarray(
        [(x_value, y_value) for y_value in y_values for x_value in x_values],
        dtype=np.float64,
    ).reshape(-1, 1, 2)


def normalize_corner_order(corners):
    """Ordena cantos como longe->perto e esquerda->direita na imagem."""
    array = np.asarray(corners, dtype=np.float64).reshape(
        INNER_CORNERS[1], INNER_CORNERS[0], 2
    )
    if float(np.mean(array[0, :, 1])) > float(np.mean(array[-1, :, 1])):
        array = array[::-1, :, :]
    if float(np.mean(array[:, 0, 0])) > float(np.mean(array[:, -1, 0])):
        array = array[:, ::-1, :]
    return np.ascontiguousarray(array.reshape(-1, 1, 2))


def detect_board_corners(frame):
    """Detecta e refina os 7x5 cantos internos; retorna ``None`` se falhar."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return None

    corners = None
    found = False
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags |= cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(
            gray,
            INNER_CORNERS,
            flags=flags,
        )
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(
            gray,
            INNER_CORNERS,
            flags=flags,
        )
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                40,
                0.01,
            )
            corners = cv2.cornerSubPix(
                gray,
                np.asarray(corners, dtype=np.float32),
                (5, 5),
                (-1, -1),
                criteria,
            )
    if not found or corners is None or len(corners) != math.prod(INNER_CORNERS):
        return None
    return normalize_corner_order(corners)


def board_area_ratio(corners, image_size):
    """Área convexa relativa; rejeita tabuleiro distante demais."""
    width, height = image_size
    hull = cv2.convexHull(np.asarray(corners, dtype=np.float32))
    return float(cv2.contourArea(hull) / max(1.0, float(width * height)))


def _view_signature(corners, image_size):
    width, height = image_size
    grid = np.asarray(corners, dtype=np.float64).reshape(
        INNER_CORNERS[1], INNER_CORNERS[0], 2
    )
    center = np.mean(grid.reshape(-1, 2), axis=0)
    area = max(board_area_ratio(corners, image_size), 1e-9)
    horizontal = np.mean(grid[:, -1, :] - grid[:, 0, :], axis=0)
    angle = math.atan2(float(horizontal[1]), float(horizontal[0]))
    return np.array(
        [center[0] / width, center[1] / height, math.log(area), angle],
        dtype=np.float64,
    )


def view_is_distinct(corners, previous_signatures, image_size):
    """Evita contar 20 cópias praticamente iguais como 20 vistas."""
    signature = _view_signature(corners, image_size)
    for previous in previous_signatures:
        center_delta = float(np.linalg.norm(signature[:2] - previous[:2]))
        area_delta = abs(float(signature[2] - previous[2]))
        angle_delta = abs(
            math.atan2(
                math.sin(float(signature[3] - previous[3])),
                math.cos(float(signature[3] - previous[3])),
            )
        )
        if center_delta < 0.035 and area_delta < 0.10 and angle_delta < math.radians(7):
            return False, signature
    return True, signature


def calibrate_fisheye(corner_sets, image_size, *, balance=1.0):
    """Calcula K/D fisheye e a matriz retificada usada em produção."""
    if len(corner_sets) < MIN_CALIBRATION_VIEWS:
        raise WideCalibrationError(
            f"são necessárias pelo menos {MIN_CALIBRATION_VIEWS} vistas válidas"
        )
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise WideCalibrationError("resolução das vistas é inválida")
    if not 0 <= float(balance) <= 1:
        raise WideCalibrationError("balance deve estar entre 0 e 1")

    object_template = board_object_points()
    object_points = [object_template.copy() for _ in corner_sets]
    image_points = []
    expected_count = math.prod(INNER_CORNERS)
    for index, corners in enumerate(corner_sets):
        points = np.asarray(corners, dtype=np.float64)
        if points.size != expected_count * 2 or not np.all(np.isfinite(points)):
            raise WideCalibrationError(f"vista {index + 1} possui cantos inválidos")
        image_points.append(points.reshape(1, expected_count, 2))

    camera_matrix = np.eye(3, dtype=np.float64)
    camera_matrix[0, 0] = max(width, height)
    camera_matrix[1, 1] = max(width, height)
    camera_matrix[0, 2] = width / 2.0
    camera_matrix[1, 2] = height / 2.0
    distortion = np.zeros((4, 1), dtype=np.float64)
    flags = (
        _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC")
        | _fisheye_flag("CALIB_CHECK_COND")
        | _fisheye_flag("CALIB_FIX_SKEW")
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        120,
        1e-7,
    )
    try:
        rms_px, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
            object_points,
            image_points,
            (width, height),
            camera_matrix,
            distortion,
            flags=flags,
            criteria=criteria,
        )
    except cv2.error as error:
        raise WideCalibrationError(
            "OpenCV recusou as vistas; capture ângulos e posições mais variados"
        ) from error
    if not np.isfinite(rms_px):
        raise WideCalibrationError("calibração produziu RMS não finito")

    rectified_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        distortion,
        (width, height),
        np.eye(3, dtype=np.float64),
        balance=float(balance),
        new_size=(width, height),
        fov_scale=1.0,
    )
    return (
        float(rms_px),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(4, 1),
        np.asarray(rectified_matrix, dtype=np.float64),
    )


def rectify_corner_points(corners, camera_matrix, distortion, rectified_matrix):
    points = normalize_corner_order(corners)
    return cv2.fisheye.undistortPoints(
        points,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        P=np.asarray(rectified_matrix, dtype=np.float64),
    )


def calculate_ground_homography(
    rectified_corners,
    *,
    center_x_mm=0.0,
    near_y_mm=0.0,
):
    """Calcula pixel retificado -> milímetros e retorna seu RMS em mm."""
    image_points = normalize_corner_order(rectified_corners)
    ground_points = ground_board_points(
        center_x_mm=center_x_mm,
        near_y_mm=near_y_mm,
    )
    homography, inliers = cv2.findHomography(
        image_points.reshape(-1, 2),
        ground_points.reshape(-1, 2),
        method=0,
    )
    if homography is None or inliers is None or np.linalg.matrix_rank(homography) != 3:
        raise WideCalibrationError("não foi possível calcular a homografia do chão")
    homography = np.asarray(homography, dtype=np.float64)
    homography /= homography[2, 2]
    projected = cv2.perspectiveTransform(image_points, homography)
    differences = projected.reshape(-1, 2) - ground_points.reshape(-1, 2)
    rms_mm = float(np.sqrt(np.mean(np.sum(differences * differences, axis=1))))
    if not np.isfinite(rms_mm):
        raise WideCalibrationError("homografia produziu erro não finito")
    return homography, rms_mm


def _draw_capture_status(frame, corners, captured, requested):
    display = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            display,
            INNER_CORNERS,
            np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            True,
        )
    color = (0, 255, 0) if corners is not None else (0, 0, 255)
    lines = (
        f"INTRINSECA {captured}/{requested}",
        "ESPACO=aceitar vista  Q=sair",
        "detectado" if corners is not None else "tabuleiro nao detectado",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (8, 20 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color if index == 2 else (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def capture_intrinsic_views(camera, requested, save_directory=None):
    corner_sets = []
    signatures = []
    image_size = None
    save_path = Path(save_directory) if save_directory else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    print(
        f"[wide] capture {requested} vistas diferentes; "
        "inclua centro, quatro cantos, perto, longe e inclinações"
    )
    while len(corner_sets) < requested:
        frame = camera.get_frame()
        height, width = frame.shape[:2]
        current_size = (width, height)
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise WideCalibrationError("a câmera mudou de resolução durante a captura")
        corners = detect_board_corners(frame)
        display = _draw_capture_status(frame, corners, len(corner_sets), requested)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise WideCalibrationError("calibração cancelada pelo operador")
        if key != ord(" "):
            continue
        if corners is None:
            print("[wide] vista ignorada: cantos internos não detectados")
            continue
        area_ratio = board_area_ratio(corners, image_size)
        if area_ratio < MIN_BOARD_AREA_RATIO:
            print(
                "[wide] vista ignorada: tabuleiro pequeno demais "
                f"({area_ratio:.1%} da imagem)"
            )
            continue
        distinct, signature = view_is_distinct(corners, signatures, image_size)
        if not distinct:
            print("[wide] vista muito parecida; mova ou incline o tabuleiro")
            continue
        corner_sets.append(corners.copy())
        signatures.append(signature)
        if save_path is not None:
            filename = save_path / f"wide_{len(corner_sets):02d}.png"
            if not cv2.imwrite(str(filename), frame):
                raise WideCalibrationError(f"não foi possível salvar {filename}")
        print(f"[wide] vista aceita: {len(corner_sets)}/{requested}")
    return corner_sets, image_size


def capture_ground_frame(camera):
    print(
        "[wide] agora deixe o tabuleiro PLANO, CENTRALIZADO e ALINHADO; "
        "a borda próxima deve aparecer embaixo. ESPAÇO aceita."
    )
    while True:
        frame = camera.get_frame()
        corners = detect_board_corners(frame)
        display = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(
                display,
                INNER_CORNERS,
                np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
                True,
            )
        cv2.putText(
            display,
            "HOMOGRAFIA: plano + alinhado  ESPACO=aceitar  Q=sair",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise WideCalibrationError("calibração cancelada pelo operador")
        if key == ord(" "):
            if corners is None:
                print("[wide] tabuleiro ainda não foi detectado")
                continue
            return frame, corners


def load_offline_views(directory, requested):
    directory = Path(directory)
    if not directory.is_dir():
        raise WideCalibrationError(f"diretório de imagens não existe: {directory}")
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    corner_sets = []
    signatures = []
    image_size = None
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[wide] ignorada (não abriu): {path.name}")
            continue
        height, width = frame.shape[:2]
        current_size = (width, height)
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            print(f"[wide] ignorada (resolução diferente): {path.name}")
            continue
        corners = detect_board_corners(frame)
        if corners is None:
            print(f"[wide] ignorada (tabuleiro ausente): {path.name}")
            continue
        area_ratio = board_area_ratio(corners, image_size)
        if area_ratio < MIN_BOARD_AREA_RATIO:
            print(f"[wide] ignorada (tabuleiro pequeno): {path.name}")
            continue
        distinct, signature = view_is_distinct(corners, signatures, image_size)
        if not distinct:
            print(f"[wide] ignorada (vista repetida): {path.name}")
            continue
        corner_sets.append(corners)
        signatures.append(signature)
        print(f"[wide] vista válida: {path.name}")
        if len(corner_sets) >= requested:
            break
    if len(corner_sets) < requested:
        raise WideCalibrationError(
            f"somente {len(corner_sets)} vistas válidas; necessárias {requested}"
        )
    return corner_sets, image_size


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Calibra fisheye + chão com tabuleiro de "
            f"{BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} quadrados de "
            f"{SQUARE_SIZE_MM:g} mm"
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="artefato NPZ de saída",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=MIN_CALIBRATION_VIEWS,
        help=f"quantidade de vistas (mínimo {MIN_CALIBRATION_VIEWS})",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1.0,
        help="0 corta bordas pretas; 1 preserva o campo de visão máximo",
    )
    parser.add_argument(
        "--sensor",
        default=None,
        help=(
            "identidade real do sensor; no modo câmera vem do Picamera2 e "
            "no modo --images é obrigatória"
        ),
    )
    parser.add_argument(
        "--capture-mode",
        default=None,
        help="identificador do modo; no modo câmera é preenchido automaticamente",
    )
    parser.add_argument(
        "--lens-position",
        type=float,
        default=None,
        help=(
            "foco manual da calibração; obrigatório com --images e opcional "
            "ao vivo (sem ele usa o autofocus confirmado)"
        ),
    )
    parser.add_argument(
        "--images",
        help="diretório com vistas intrínsecas; não abre a câmera",
    )
    parser.add_argument(
        "--homography-image",
        help="imagem do tabuleiro plano/alinhado; obrigatória com --images",
    )
    parser.add_argument(
        "--save-captures",
        help="diretório opcional para salvar as vistas aceitas",
    )
    parser.add_argument(
        "--center-x-mm",
        type=float,
        default=0.0,
        help="X físico do centro do tabuleiro alinhado",
    )
    parser.add_argument(
        "--near-y-mm",
        type=float,
        default=0.0,
        help="Y físico da fileira interna mais próxima do robô",
    )
    return parser


def run(args):
    if args.views < MIN_CALIBRATION_VIEWS:
        raise WideCalibrationError(
            f"--views deve ser pelo menos {MIN_CALIBRATION_VIEWS}"
        )
    if not 0 <= args.balance <= 1:
        raise WideCalibrationError("--balance deve estar entre 0 e 1")
    if args.images and not args.homography_image:
        raise WideCalibrationError("--homography-image é obrigatória com --images")
    if args.homography_image and not args.images:
        raise WideCalibrationError("--homography-image só pode ser usada com --images")
    if args.images and not args.capture_mode:
        raise WideCalibrationError(
            "--capture-mode é obrigatória com --images; use exatamente a "
            "assinatura exibida por LineCamera.capture_mode_id na câmera "
            "que gerou as imagens"
        )
    if args.images and not args.sensor:
        raise WideCalibrationError(
            "--sensor é obrigatório com --images; use a identidade exibida "
            "por LineCamera.sensor_id"
        )
    if args.images and args.lens_position is None:
        raise WideCalibrationError(
            "--lens-position é obrigatória com --images; use a posição "
            "confirmada na captura original"
        )
    if not args.images and args.capture_mode:
        raise WideCalibrationError(
            "--capture-mode não deve ser forçado na captura ao vivo; o modo "
            "real será lido do driver"
        )

    capture_mode = None
    sensor_identity = None
    lens_position = None
    if args.images:
        sensor_identity = normalizar_identidade_sensor({"Model": args.sensor})
        validate_sensor_identity(sensor_identity)
        sensor_esperado = normalizar_identidade_sensor({
            "Model": LINE_CAMERA_SENSOR_ID,
        })
        if sensor_identity != sensor_esperado:
            raise WideCalibrationError(
                "sensor informado difere da Camera Module 3 Wide esperada: "
                f"{sensor_identity!r} != {sensor_esperado!r}"
            )
        capture_mode = validate_capture_mode_id(args.capture_mode)
        lens_position = validate_lens_position(args.lens_position)

    camera = None
    try:
        if args.images:
            corner_sets, image_size = load_offline_views(args.images, args.views)
        else:
            camera = LineCamera()
            sensor_identity = validate_sensor_identity(camera.sensor_id)
            sensor_esperado = normalizar_identidade_sensor({
                "Model": LINE_CAMERA_SENSOR_ID,
            })
            if sensor_identity != sensor_esperado:
                raise WideCalibrationError(
                    "Picamera2 abriu um sensor diferente da Camera Module 3 "
                    f"Wide esperada: {sensor_identity!r} != {sensor_esperado!r}"
                )
            if args.sensor:
                informed_sensor = normalizar_identidade_sensor(
                    {"Model": args.sensor})
                validate_sensor_identity(informed_sensor)
                if informed_sensor != sensor_identity:
                    raise WideCalibrationError(
                        "--sensor diverge do Picamera2: "
                        f"{informed_sensor!r} != {sensor_identity!r}"
                    )
            if camera.sensor_mode is None:
                raise WideCalibrationError(
                    "Picamera2 não confirmou o modo bruto do sensor; "
                    "atualize o driver antes de calibrar"
                )
            if args.lens_position is not None:
                lens_position = camera.aplicar_posicao_lente(
                    args.lens_position)
            else:
                lens_position = validate_lens_position(camera.lens_position)
            capture_mode = validate_capture_mode_id(camera.capture_mode_id)
            print(
                "[wide] identidade confirmada: "
                f"sensor={sensor_identity}, LensPosition={lens_position:.4f}, "
                f"modo={capture_mode}"
            )
            corner_sets, image_size = capture_intrinsic_views(
                camera,
                args.views,
                args.save_captures,
            )

        print("[wide] calculando modelo fisheye...")
        rms_px, camera_matrix, distortion, rectified_matrix = calibrate_fisheye(
            corner_sets,
            image_size,
            balance=args.balance,
        )
        print(f"[wide] RMS intrínseco: {rms_px:.4f} px")
        if rms_px > MAX_CALIBRATION_RMS_PX:
            raise WideCalibrationError(
                f"RMS {rms_px:.4f} px excede {MAX_CALIBRATION_RMS_PX:g} px; "
                "refaça vistas borradas ou repetidas"
            )

        if args.images:
            ground_frame = cv2.imread(args.homography_image, cv2.IMREAD_COLOR)
            if ground_frame is None:
                raise WideCalibrationError(
                    f"não foi possível abrir {args.homography_image}"
                )
            ground_size = (ground_frame.shape[1], ground_frame.shape[0])
            if ground_size != image_size:
                raise WideCalibrationError(
                    "imagem da homografia tem resolução diferente: "
                    f"{ground_size} != {image_size}"
                )
            ground_corners = detect_board_corners(ground_frame)
            if ground_corners is None:
                raise WideCalibrationError(
                    "tabuleiro não foi detectado na imagem da homografia"
                )
        else:
            ground_frame, ground_corners = capture_ground_frame(camera)

        rectified_corners = rectify_corner_points(
            ground_corners,
            camera_matrix,
            distortion,
            rectified_matrix,
        )
        pixel_to_ground, homography_rms = calculate_ground_homography(
            rectified_corners,
            center_x_mm=args.center_x_mm,
            near_y_mm=args.near_y_mm,
        )
        print(f"[wide] RMS da homografia: {homography_rms:.4f} mm")
        if homography_rms > MAX_HOMOGRAPHY_RMS_MM:
            raise WideCalibrationError(
                f"erro da homografia {homography_rms:.3f} mm excede "
                f"{MAX_HOMOGRAPHY_RMS_MM:g} mm; alinhe melhor o tabuleiro"
            )

        metadata = WideCalibrationMetadata(
            image_size=image_size,
            camera_index=(
                camera.camera_index if camera is not None
                else LINE_CAMERA_INDEX
            ),
            sensor=sensor_identity,
            capture_mode=capture_mode,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            rms_px=rms_px,
            homography_rms_mm=homography_rms,
            lens_position=lens_position,
            view_count=len(corner_sets),
            balance=args.balance,
        )
        calibration = WideCalibration(
            camera_matrix=camera_matrix,
            distortion=distortion,
            rectified_matrix=rectified_matrix,
            pixel_to_ground=pixel_to_ground,
            metadata=metadata,
        )
        target = calibration.save(args.output)
        loaded = WideCalibration.load(
            target,
            expected_image_size=image_size,
            expected_camera_index=metadata.camera_index,
            expected_sensor=sensor_identity,
            expected_capture_mode=capture_mode,
            expected_lens_position=lens_position,
        )
        # Força a criação dos mapas agora; um artefato matematicamente aceito
        # mas recusado pelo OpenCV não chega à pista.
        loaded.rectify(np.zeros((image_size[1], image_size[0]), dtype=np.uint8))
        print(f"[wide] calibração validada e salva em: {target.resolve()}")
        print(
            "[wide] convenção: X+ direita, Y+ frente; "
            "homografia recebe pixels já retificados"
        )
        return target
    finally:
        if camera is not None:
            camera.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # OpenCV headless não implementa HighGUI; isso não deve ocultar
            # o erro de identidade/calibração que encerrou a ferramenta.
            pass


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (WideCalibrationError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"ERRO: {error}\n")


if __name__ == "__main__":
    main()
