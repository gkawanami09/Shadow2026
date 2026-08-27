"""Calibração geométrica da câmera wide usada nas interseções.

O artefato salvo por este módulo contém duas transformações distintas:

1. câmera fisheye -> imagem retificada, mantendo a resolução de captura;
2. pixel da imagem retificada -> chão em milímetros.

No chão, ``X`` cresce para a direita do robô e ``Y`` cresce para a frente.
A transformação não é aplicada automaticamente ao segue-linha: o consumidor
decide em quais detectores usar a geometria retificada.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import uuid

import cv2
import numpy as np

import config


CALIBRATION_SCHEMA_VERSION = 3
BOARD_SQUARES = tuple(config.GREEN_WIDE_BOARD_SQUARES)
INNER_CORNERS = tuple(config.GREEN_WIDE_BOARD_INNER_CORNERS)
SQUARE_SIZE_MM = float(config.GREEN_WIDE_BOARD_SQUARE_MM)
MIN_CALIBRATION_VIEWS = int(config.GREEN_WIDE_CALIBRATION_MIN_VIEWS)
MAX_CALIBRATION_RMS_PX = float(config.GREEN_WIDE_CALIBRATION_MAX_ERROR_PX)
MAX_HOMOGRAPHY_RMS_MM = float(config.GREEN_WIDE_HOMOGRAPHY_MAX_ERROR_MM)
LENS_POSITION_TOLERANCE = .05

_INVALID_SENSOR_IDENTITIES = frozenset((
    "",
    "unknown",
    "none",
    "null",
    "unavailable",
    "indisponivel",
    "indisponível",
))
_CAPTURE_MODE_PATTERN = re.compile(
    r"^LineCamera:(?P<width>[1-9]\d*)x(?P<height>[1-9]\d*)"
    r"@(?P<fps>(?:\d+(?:\.\d*)?|\.\d+)):(?P<fov>full-fov|cropped)"
    r";sensor-mode=(?P<sensor_width>[1-9]\d*)x"
    r"(?P<sensor_height>[1-9]\d*)x(?P<bit_depth>[1-9]\d*)"
    r"(?:;crop=(?P<crop_x>-?\d+),(?P<crop_y>-?\d+),"
    r"(?P<crop_width>[1-9]\d*),(?P<crop_height>[1-9]\d*))?$"
)


class WideCalibrationError(RuntimeError):
    """Artefato ausente, incompatível ou geometricamente inválido."""


def validate_sensor_identity(sensor):
    """Recusa identidades que não comprovam qual sensor está ativo."""
    if not isinstance(sensor, str):
        raise WideCalibrationError("sensor competitivo não foi informado")
    value = sensor.strip()
    if value.casefold() in _INVALID_SENSOR_IDENTITIES:
        raise WideCalibrationError(
            "Picamera2 não confirmou a identidade do sensor competitivo"
        )
    return value


def validate_lens_position(lens_position):
    """Normaliza uma posição de lente finita usada pela calibração."""
    if isinstance(lens_position, bool):
        raise WideCalibrationError("LensPosition da calibração é inválida")
    try:
        value = float(lens_position)
    except (TypeError, ValueError) as error:
        raise WideCalibrationError(
            "LensPosition da calibração é inválida") from error
    if not np.isfinite(value):
        raise WideCalibrationError("LensPosition da calibração deve ser finita")
    return value


def validate_capture_mode_id(capture_mode, *, image_size=None):
    """Exige assinatura canônica contendo o modo bruto confirmado."""
    if not isinstance(capture_mode, str):
        raise WideCalibrationError("modo de captura não foi informado")
    match = _CAPTURE_MODE_PATTERN.fullmatch(capture_mode.strip())
    if match is None:
        raise WideCalibrationError(
            "modo bruto do sensor não foi confirmado na assinatura de captura"
        )
    values = match.groupdict()
    fps = float(values["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise WideCalibrationError("FPS da assinatura de captura é inválido")
    mode_image_size = (int(values["width"]), int(values["height"]))
    if image_size is not None and mode_image_size != tuple(image_size):
        raise WideCalibrationError(
            "resolução da assinatura de captura difere do artefato: "
            f"{mode_image_size} != {tuple(image_size)}"
        )
    return capture_mode.strip()


def build_capture_mode_id(
    image_size,
    fps,
    *,
    full_fov=True,
    sensor_mode=None,
    scaler_crop=None,
):
    """Gera a assinatura canônica compartilhada entre CLI e runtime.

    Chamadas antigas, sem detalhes do sensor, conservam exatamente a string
    anterior. ``LineCamera.capture_mode_id`` informa ``sensor_mode`` e
    ``scaler_crop`` quando o driver os confirmou, impedindo que um modo
    recortado reutilize silenciosamente uma homografia full-FoV.
    """
    try:
        width, height = (int(value) for value in image_size)
        fps_value = float(fps)
    except (TypeError, ValueError) as error:
        raise WideCalibrationError("modo de captura possui valores inválidos") from error
    if width <= 0 or height <= 0 or not np.isfinite(fps_value) or fps_value <= 0:
        raise WideCalibrationError("modo de captura possui valores inválidos")
    crop_name = "full-fov" if full_fov else "cropped"
    signature = f"LineCamera:{width}x{height}@{fps_value:.2f}:{crop_name}"

    if sensor_mode is not None:
        try:
            mode_width, mode_height = (
                int(value) for value in sensor_mode["output_size"]
            )
            bit_depth = int(sensor_mode["bit_depth"])
        except (KeyError, TypeError, ValueError) as error:
            raise WideCalibrationError("sensor_mode inválido") from error
        if mode_width <= 0 or mode_height <= 0 or bit_depth <= 0:
            raise WideCalibrationError("sensor_mode inválido")
        signature += (
            f";sensor-mode={mode_width}x{mode_height}x{bit_depth}"
        )

    if scaler_crop is not None:
        try:
            crop = tuple(int(value) for value in scaler_crop)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("ScalerCrop inválido") from error
        if len(crop) != 4 or crop[2] <= 0 or crop[3] <= 0:
            raise WideCalibrationError("ScalerCrop inválido")
        signature += ";crop=" + ",".join(str(value) for value in crop)
    return signature


@dataclass(frozen=True)
class WideCalibrationMetadata:
    """Metadados que vinculam a calibração ao modo real de captura."""

    image_size: tuple[int, int]
    camera_index: int
    sensor: str
    capture_mode: str
    created_at_utc: str
    rms_px: float
    homography_rms_mm: float
    lens_position: float
    view_count: int
    board_squares: tuple[int, int] = BOARD_SQUARES
    inner_corners: tuple[int, int] = INNER_CORNERS
    square_size_mm: float = SQUARE_SIZE_MM
    balance: float = 1.0


def _array_f64(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise WideCalibrationError(f"{name} não é uma matriz numérica") from error
    if array.shape != shape:
        raise WideCalibrationError(
            f"{name} deve ter shape {shape}, recebido {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise WideCalibrationError(f"{name} contém valor não finito")
    return np.ascontiguousarray(array)


def _points_f64(points, name):
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise WideCalibrationError(f"{name} não contém pontos numéricos") from error
    if array.ndim < 1 or array.shape[-1] != 2:
        raise WideCalibrationError(
            f"{name} deve terminar em duas coordenadas, recebido {array.shape}"
        )
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise WideCalibrationError(f"{name} está vazio ou contém valor não finito")
    return np.ascontiguousarray(array), array.shape


def _transform_points(matrix, points, name):
    array, original_shape = _points_f64(points, name)
    flat = array.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(flat, matrix)
    if not np.all(np.isfinite(transformed)):
        raise WideCalibrationError(f"{name} projetou ponto no infinito")
    return transformed.reshape(original_shape)


def _scalar(npz, key, cast):
    if key not in npz:
        raise WideCalibrationError(f"campo obrigatório ausente: {key}")
    array = np.asarray(npz[key])
    if array.size != 1:
        raise WideCalibrationError(f"campo {key} deve ser escalar")
    try:
        return cast(array.reshape(()).item())
    except (TypeError, ValueError, OverflowError) as error:
        raise WideCalibrationError(f"campo {key} inválido") from error


def _integer_scalar(npz, key):
    raw_value = _scalar(npz, key, float)
    if (
        not np.isfinite(raw_value)
        or not float(raw_value).is_integer()
    ):
        raise WideCalibrationError(f"campo {key} deve ser inteiro")
    return int(raw_value)


def _text_scalar(npz, key):
    if key not in npz:
        raise WideCalibrationError(f"campo obrigatório ausente: {key}")
    array = np.asarray(npz[key])
    if array.size != 1 or array.dtype.kind not in ("U", "S"):
        raise WideCalibrationError(f"campo {key} deve ser texto")
    value = array.reshape(()).item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WideCalibrationError(f"campo {key} não é UTF-8") from error
    return str(value)


@dataclass
class WideCalibration:
    """Calibração validada e pronta para transformar frames e pontos."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    rectified_matrix: np.ndarray
    pixel_to_ground: np.ndarray
    metadata: WideCalibrationMetadata

    def __post_init__(self):
        self.camera_matrix = _array_f64(
            self.camera_matrix, (3, 3), "camera_matrix"
        )
        try:
            distortion = np.asarray(self.distortion, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("distortion não é uma matriz numérica") from error
        if distortion.shape == (4,):
            distortion = distortion.reshape(4, 1)
        self.distortion = _array_f64(distortion, (4, 1), "distortion")
        self.rectified_matrix = _array_f64(
            self.rectified_matrix, (3, 3), "rectified_matrix"
        )
        self.pixel_to_ground = _array_f64(
            self.pixel_to_ground, (3, 3), "pixel_to_ground"
        )
        self._ground_to_pixel = None
        self._map_1 = None
        self._map_2 = None
        self.validate()

    @property
    def image_size(self):
        """Resolução ``(largura, altura)`` para a qual o artefato vale."""
        return self.metadata.image_size

    @property
    def ground_to_pixel(self):
        """Homografia chão em mm -> pixel da imagem retificada."""
        if self._ground_to_pixel is None:
            self._ground_to_pixel = np.linalg.inv(self.pixel_to_ground)
        return self._ground_to_pixel

    def validate(self):
        """Valida conteúdo, convenções e limites de qualidade do artefato."""
        metadata = self.metadata
        if not isinstance(metadata, WideCalibrationMetadata):
            raise WideCalibrationError("metadata possui tipo inválido")
        try:
            width, height = metadata.image_size
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("image_size deve ser (largura, altura)") from error
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or int(width) != width
            or int(height) != height
            or int(width) <= 0
            or int(height) <= 0
        ):
            raise WideCalibrationError("resolução da calibração é inválida")

        validate_sensor_identity(metadata.sensor)
        validate_capture_mode_id(
            metadata.capture_mode,
            image_size=(int(width), int(height)),
        )
        validate_lens_position(metadata.lens_position)
        try:
            created_at = datetime.fromisoformat(
                str(metadata.created_at_utc).replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("created_at_utc não é uma data ISO válida") from error
        if created_at.utcoffset() is None or created_at.utcoffset().total_seconds() != 0:
            raise WideCalibrationError("created_at_utc deve possuir fuso UTC")
        try:
            camera_index = float(metadata.camera_index)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("índice da câmera é inválido") from error
        if (
            not np.isfinite(camera_index)
            or isinstance(metadata.camera_index, bool)
            or not camera_index.is_integer()
            or int(camera_index) < 0
        ):
            raise WideCalibrationError("índice da câmera é inválido")

        try:
            board_squares = tuple(metadata.board_squares)
            inner_corners = tuple(metadata.inner_corners)
            square_size_mm = float(metadata.square_size_mm)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("descrição do tabuleiro é inválida") from error
        if board_squares != BOARD_SQUARES:
            raise WideCalibrationError(
                f"tabuleiro deve possuir {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} quadrados"
            )
        if inner_corners != INNER_CORNERS:
            raise WideCalibrationError(
                f"padrão deve possuir {INNER_CORNERS[0]}x{INNER_CORNERS[1]} cantos internos"
            )
        if not np.isfinite(square_size_mm) or not np.isclose(
            square_size_mm,
            SQUARE_SIZE_MM,
        ):
            raise WideCalibrationError(
                f"quadrados devem medir {SQUARE_SIZE_MM:g} mm"
            )
        try:
            view_count = float(metadata.view_count)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("quantidade de vistas é inválida") from error
        if (
            not np.isfinite(view_count)
            or isinstance(metadata.view_count, bool)
            or not view_count.is_integer()
            or int(view_count) < MIN_CALIBRATION_VIEWS
        ):
            raise WideCalibrationError(
                f"calibração precisa de pelo menos {MIN_CALIBRATION_VIEWS} vistas"
            )
        if (
            isinstance(metadata.rms_px, bool)
            or isinstance(metadata.homography_rms_mm, bool)
            or isinstance(metadata.balance, bool)
        ):
            raise WideCalibrationError("RMS ou balance não é numérico")
        try:
            rms = float(metadata.rms_px)
            homography_rms = float(metadata.homography_rms_mm)
            balance = float(metadata.balance)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("RMS ou balance não é numérico") from error
        if not np.isfinite(rms) or not 0 <= rms <= MAX_CALIBRATION_RMS_PX:
            raise WideCalibrationError(
                f"RMS deve estar entre 0 e {MAX_CALIBRATION_RMS_PX:g} px"
            )
        if (
            not np.isfinite(homography_rms)
            or not 0 <= homography_rms <= MAX_HOMOGRAPHY_RMS_MM
        ):
            raise WideCalibrationError(
                "RMS da homografia deve estar entre 0 e "
                f"{MAX_HOMOGRAPHY_RMS_MM:g} mm"
            )
        if not np.isfinite(balance) or not 0 <= balance <= 1:
            raise WideCalibrationError("balance deve estar entre 0 e 1")

        for name, matrix in (
            ("camera_matrix", self.camera_matrix),
            ("rectified_matrix", self.rectified_matrix),
            ("pixel_to_ground", self.pixel_to_ground),
        ):
            if np.linalg.matrix_rank(matrix) != 3:
                raise WideCalibrationError(f"{name} é singular")
        for name, matrix in (
            ("camera_matrix", self.camera_matrix),
            ("rectified_matrix", self.rectified_matrix),
        ):
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                raise WideCalibrationError(f"{name} possui distância focal inválida")
            if not np.isclose(matrix[2, 2], 1.0):
                raise WideCalibrationError(f"{name}[2,2] deve ser 1")

        scale = self.pixel_to_ground[2, 2]
        if np.isclose(scale, 0.0):
            raise WideCalibrationError("pixel_to_ground possui escala inválida")
        self.pixel_to_ground /= scale
        return self

    def validate_compatibility(
        self,
        image_size,
        *,
        camera_index=None,
        sensor=None,
        capture_mode=None,
        lens_position=None,
    ):
        """Falha cedo se câmera, resolução ou modo diferirem do artefato."""
        try:
            raw_size = tuple(image_size)
            received_size = tuple(int(value) for value in raw_size)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("resolução recebida é inválida") from error
        if (
            len(received_size) != 2
            or any(
                isinstance(value, bool) or int(value) != value
                for value in raw_size
            )
        ):
            raise WideCalibrationError("resolução recebida é inválida")
        if received_size != tuple(self.image_size):
            raise WideCalibrationError(
                "resolução incompatível: artefato "
                f"{self.image_size}, câmera {received_size}"
            )
        if camera_index is not None:
            try:
                received_index = float(camera_index)
            except (TypeError, ValueError) as error:
                raise WideCalibrationError("índice da câmera recebido é inválido") from error
            if (
                not np.isfinite(received_index)
                or not received_index.is_integer()
                or int(received_index) != int(self.metadata.camera_index)
            ):
                raise WideCalibrationError(
                    "índice de câmera incompatível: artefato "
                    f"{self.metadata.camera_index}, câmera {camera_index}"
                )
        if sensor is not None:
            received_sensor = validate_sensor_identity(sensor)
            if received_sensor.casefold() != self.metadata.sensor.strip().casefold():
                raise WideCalibrationError(
                    f"sensor incompatível: artefato {self.metadata.sensor!r}, "
                    f"câmera {sensor!r}"
                )
        if capture_mode is not None:
            received_mode = validate_capture_mode_id(
                capture_mode,
                image_size=received_size,
            )
            if received_mode != self.metadata.capture_mode.strip():
                raise WideCalibrationError(
                    "modo de captura incompatível: artefato "
                    f"{self.metadata.capture_mode!r}, câmera {capture_mode!r}"
                )
        if lens_position is not None:
            received_lens = validate_lens_position(lens_position)
            expected_lens = validate_lens_position(
                self.metadata.lens_position)
            if not np.isclose(
                received_lens,
                expected_lens,
                rtol=0.,
                atol=LENS_POSITION_TOLERANCE,
            ):
                raise WideCalibrationError(
                    "LensPosition incompatível: artefato "
                    f"{expected_lens:.4f}, câmera {received_lens:.4f}"
                )
        return True

    def _validate_frame(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise WideCalibrationError("frame deve ser uma imagem NumPy 2D ou 3D")
        height, width = frame.shape[:2]
        self.validate_compatibility((width, height))

    def _ensure_maps(self):
        if self._map_1 is None or self._map_2 is None:
            self._map_1, self._map_2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion,
                np.eye(3, dtype=np.float64),
                self.rectified_matrix,
                tuple(self.image_size),
                cv2.CV_16SC2,
            )

    def rectify(self, frame, interpolation=cv2.INTER_LINEAR):
        """Remove distorção fisheye sem alterar a resolução do frame."""
        self._validate_frame(frame)
        self._ensure_maps()
        return cv2.remap(
            frame,
            self._map_1,
            self._map_2,
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def rectify_mask(self, mask):
        """Retifica máscara binária sem criar níveis pela interpolação."""
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise WideCalibrationError("máscara deve ser uma imagem NumPy 2D")
        return self.rectify(mask, interpolation=cv2.INTER_NEAREST)

    def rectify_points(self, raw_pixels):
        """Converte pixels fisheye crus em pixels da imagem retificada."""
        points, original_shape = _points_f64(raw_pixels, "raw_pixels")
        rectified = cv2.fisheye.undistortPoints(
            points.reshape(-1, 1, 2),
            self.camera_matrix,
            self.distortion,
            R=np.eye(3, dtype=np.float64),
            P=self.rectified_matrix,
        )
        if not np.all(np.isfinite(rectified)):
            raise WideCalibrationError("retificação projetou ponto no infinito")
        return rectified.reshape(original_shape)

    def unrectify_points(self, rectified_pixels):
        """Converte pixels retificados de volta ao frame fisheye cru."""
        points, original_shape = _points_f64(
            rectified_pixels, "rectified_pixels")
        flat = points.reshape(-1, 2)
        homogeneous = np.column_stack((
            flat,
            np.ones(len(flat), dtype=np.float64),
        ))
        normalized_h = (
            np.linalg.inv(self.rectified_matrix) @ homogeneous.T
        ).T
        denominator = normalized_h[:, 2]
        if np.any(np.isclose(denominator, 0.0)):
            raise WideCalibrationError(
                "pixel retificado projetou ponto no infinito")
        normalized = (
            normalized_h[:, :2] / denominator[:, np.newaxis]
        ).reshape(-1, 1, 2)
        raw = cv2.fisheye.distortPoints(
            normalized,
            self.camera_matrix,
            self.distortion,
        )
        if not np.all(np.isfinite(raw)):
            raise WideCalibrationError(
                "distorção inversa projetou ponto no infinito")
        return raw.reshape(original_shape)

    def pixels_to_ground_mm(self, pixels, *, raw=False):
        """Projeta pixels crus ou retificados no plano do chão em milímetros."""
        if raw:
            pixels = self.rectify_points(pixels)
        return _transform_points(self.pixel_to_ground, pixels, "pixels")

    def ground_mm_to_pixels(self, ground_points):
        """Projeta pontos do chão em pixels da imagem já retificada."""
        return _transform_points(
            self.ground_to_pixel,
            ground_points,
            "ground_points",
        )

    def warp_ground(
        self,
        frame,
        *,
        x_limits_mm,
        y_limits_mm,
        pixels_per_mm=2.0,
        frame_is_rectified=False,
        interpolation=cv2.INTER_LINEAR,
    ):
        """Cria visão superior com frente para cima e direita para a direita."""
        try:
            x_min, x_max = (float(value) for value in x_limits_mm)
            y_min, y_max = (float(value) for value in y_limits_mm)
            scale = float(pixels_per_mm)
        except (TypeError, ValueError) as error:
            raise WideCalibrationError("limites da visão de chão são inválidos") from error
        if not (
            np.isfinite((x_min, x_max, y_min, y_max, scale)).all()
            and x_max > x_min
            and y_max > y_min
            and scale > 0
        ):
            raise WideCalibrationError("limites da visão de chão são inválidos")

        if frame_is_rectified:
            self._validate_frame(frame)
            rectified = frame
        else:
            rectified = self.rectify(frame, interpolation=interpolation)

        width = max(1, int(round((x_max - x_min) * scale)))
        height = max(1, int(round((y_max - y_min) * scale)))
        ground_to_canvas = np.array(
            [
                [scale, 0.0, -x_min * scale],
                [0.0, -scale, y_max * scale],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pixel_to_canvas = ground_to_canvas @ self.pixel_to_ground
        return cv2.warpPerspective(
            rectified,
            pixel_to_canvas,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def save(self, path):
        """Salva o artefato NPZ de forma atômica e sem objetos pickle."""
        self.validate()
        target = Path(path)
        if target.suffix.lower() != ".npz":
            raise WideCalibrationError("artefato de calibração deve terminar em .npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.stem}.{uuid.uuid4().hex}.tmp.npz"
        )
        metadata = self.metadata
        try:
            np.savez_compressed(
                temporary,
                schema_version=np.int32(CALIBRATION_SCHEMA_VERSION),
                camera_matrix=self.camera_matrix,
                distortion=self.distortion,
                rectified_matrix=self.rectified_matrix,
                pixel_to_ground=self.pixel_to_ground,
                image_size=np.asarray(metadata.image_size, dtype=np.int32),
                camera_index=np.int32(metadata.camera_index),
                sensor=np.asarray(metadata.sensor),
                capture_mode=np.asarray(metadata.capture_mode),
                created_at_utc=np.asarray(metadata.created_at_utc),
                rms_px=np.float64(metadata.rms_px),
                homography_rms_mm=np.float64(metadata.homography_rms_mm),
                lens_position=np.float64(metadata.lens_position),
                view_count=np.int32(metadata.view_count),
                board_squares=np.asarray(metadata.board_squares, dtype=np.int32),
                inner_corners=np.asarray(metadata.inner_corners, dtype=np.int32),
                square_size_mm=np.float64(metadata.square_size_mm),
                balance=np.float64(metadata.balance),
            )
            os.replace(temporary, target)
        except (OSError, ValueError) as error:
            raise WideCalibrationError(
                f"não foi possível salvar calibração em {target}: {error}"
            ) from error
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return target

    @classmethod
    def load(
        cls,
        path,
        *,
        expected_image_size=None,
        expected_camera_index=None,
        expected_sensor=None,
        expected_capture_mode=None,
        expected_lens_position=None,
    ):
        """Carrega um NPZ sem pickle e opcionalmente confere a câmera ativa."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as npz:
                schema_version = _integer_scalar(npz, "schema_version")
                if schema_version != CALIBRATION_SCHEMA_VERSION:
                    raise WideCalibrationError(
                        "versão de calibração incompatível: "
                        f"{schema_version}, esperada {CALIBRATION_SCHEMA_VERSION}"
                    )

                def pair(key, cast):
                    if key not in npz:
                        raise WideCalibrationError(
                            f"campo obrigatório ausente: {key}"
                        )
                    array = np.asarray(npz[key])
                    if array.shape != (2,):
                        raise WideCalibrationError(f"campo {key} deve ter 2 valores")
                    raw_values = array.tolist()
                    try:
                        converted = tuple(cast(value) for value in raw_values)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise WideCalibrationError(f"campo {key} inválido") from error
                    if cast is int and any(
                        isinstance(value, bool) or int(value) != value
                        for value in raw_values
                    ):
                        raise WideCalibrationError(f"campo {key} deve conter inteiros")
                    return converted

                metadata = WideCalibrationMetadata(
                    image_size=pair("image_size", int),
                    camera_index=_integer_scalar(npz, "camera_index"),
                    sensor=_text_scalar(npz, "sensor"),
                    capture_mode=_text_scalar(npz, "capture_mode"),
                    created_at_utc=_text_scalar(npz, "created_at_utc"),
                    rms_px=_scalar(npz, "rms_px", float),
                    homography_rms_mm=_scalar(
                        npz, "homography_rms_mm", float),
                    lens_position=_scalar(npz, "lens_position", float),
                    view_count=_integer_scalar(npz, "view_count"),
                    board_squares=pair("board_squares", int),
                    inner_corners=pair("inner_corners", int),
                    square_size_mm=_scalar(npz, "square_size_mm", float),
                    balance=_scalar(npz, "balance", float),
                )
                calibration = cls(
                    camera_matrix=np.asarray(npz["camera_matrix"]).copy(),
                    distortion=np.asarray(npz["distortion"]).copy(),
                    rectified_matrix=np.asarray(npz["rectified_matrix"]).copy(),
                    pixel_to_ground=np.asarray(npz["pixel_to_ground"]).copy(),
                    metadata=metadata,
                )
        except WideCalibrationError:
            raise
        except (OSError, ValueError, KeyError, EOFError) as error:
            raise WideCalibrationError(
                f"não foi possível carregar calibração {source}: {error}"
            ) from error

        if expected_image_size is not None:
            calibration.validate_compatibility(
                expected_image_size,
                camera_index=expected_camera_index,
                sensor=expected_sensor,
                capture_mode=expected_capture_mode,
                lens_position=expected_lens_position,
            )
        elif (
            expected_camera_index is not None
            or expected_sensor is not None
            or expected_capture_mode is not None
            or expected_lens_position is not None
        ):
            calibration.validate_compatibility(
                calibration.image_size,
                camera_index=expected_camera_index,
                sensor=expected_sensor,
                capture_mode=expected_capture_mode,
                lens_position=expected_lens_position,
            )
        return calibration


def load_wide_calibration(path, **compatibility):
    """Atalho explícito para consumidores que não precisam conhecer a classe."""
    return WideCalibration.load(path, **compatibility)


def carregar_calibracao(
    path,
    *,
    resolution=None,
    camera_index=None,
    sensor=None,
    mode=None,
    lens_position=None,
):
    """API curta: carrega e confere a câmera antes de armar o detector."""
    return WideCalibration.load(
        path,
        expected_image_size=resolution,
        expected_camera_index=camera_index,
        expected_sensor=sensor,
        expected_capture_mode=mode,
        expected_lens_position=lens_position,
    )


__all__ = [
    "BOARD_SQUARES",
    "CALIBRATION_SCHEMA_VERSION",
    "INNER_CORNERS",
    "LENS_POSITION_TOLERANCE",
    "MAX_CALIBRATION_RMS_PX",
    "MAX_HOMOGRAPHY_RMS_MM",
    "MIN_CALIBRATION_VIEWS",
    "SQUARE_SIZE_MM",
    "WideCalibration",
    "WideCalibrationError",
    "WideCalibrationMetadata",
    "build_capture_mode_id",
    "carregar_calibracao",
    "load_wide_calibration",
    "validate_capture_mode_id",
    "validate_lens_position",
    "validate_sensor_identity",
]
