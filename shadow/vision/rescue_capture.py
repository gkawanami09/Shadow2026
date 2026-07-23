"""Captura exclusiva da camera frontal de resgate.

O modo de sensor e escolhido pelo campo de visao, nao apenas pela resolucao.
Assim, a camera nao perde as laterais por causa de um recorte 16:9/4:3.
"""

import time

import cv2

import rescue_config


def _mode_size(mode):
    size = mode.get("size", (0, 0))
    if len(size) != 2:
        return 0, 0
    return int(size[0]), int(size[1])


def _mode_crop(mode):
    """Retorna a area fisica do sensor usada pelo modo."""
    crop = mode.get("crop_limits")
    if crop is not None and len(crop) >= 4:
        return tuple(int(value) for value in crop[:4])
    width, height = _mode_size(mode)
    return 0, 0, width, height


def _fit_output_size(aspect, max_width, max_height):
    """Maior tamanho par que cabe nos limites sem deformar a imagem."""
    if aspect <= 0:
        raise ValueError("proporcao da camera invalida")
    width = int(max_width)
    height = int(round(width / aspect))
    if height > max_height:
        height = int(max_height)
        width = int(round(height * aspect))
    width = max(2, width - width % 2)
    height = max(2, height - height % 2)
    return width, height


def _select_widest_sensor_mode(
    sensor_modes,
    target_fps,
    max_output_width,
    max_output_height,
):
    """Seleciona o modo que enxerga a maior area fisica do sensor.

    Entre modos com o mesmo campo de visao, prefere um que sustente o FPS
    desejado e que tenha pixels suficientes para a saida, evitando processar
    o modo fotografico gigante sem ganho para o detector.
    """
    modes = [
        mode for mode in sensor_modes
        if _mode_size(mode)[0] > 0 and _mode_size(mode)[1] > 0
    ]
    if not modes:
        return None

    def crop_area(mode):
        _, _, width, height = _mode_crop(mode)
        return width * height

    largest_crop = max(crop_area(mode) for mode in modes)
    widest = [
        mode for mode in modes
        # Tolerar somente arredondamento/alinhamento de poucos pixels. O modo
        # OV5647 640x480 usa 98,77% da area e nao e full-FoV.
        if crop_area(mode) >= largest_crop * 0.999
    ]
    fast_enough = [
        mode for mode in widest
        if float(mode.get("fps", 0.0)) + 0.25 >= target_fps
    ]
    candidates = fast_enough or widest

    _, _, crop_width, crop_height = _mode_crop(candidates[0])
    aspect = crop_width / max(crop_height, 1)
    wanted_width, wanted_height = _fit_output_size(
        aspect, max_output_width, max_output_height)
    wanted_area = wanted_width * wanted_height
    adequate = [
        mode for mode in candidates
        if _mode_size(mode)[0] * _mode_size(mode)[1] >= wanted_area
    ]

    if adequate:
        return min(
            adequate,
            key=lambda mode: (
                _mode_size(mode)[0] * _mode_size(mode)[1],
                -float(mode.get("fps", 0.0))))
    return max(
        candidates,
        key=lambda mode: (
            _mode_size(mode)[0] * _mode_size(mode)[1],
            float(mode.get("fps", 0.0))))


def _known_sensor_mode(camera_info):
    model = str(
        camera_info.get("Model", camera_info.get("model", ""))).lower()
    for model_name, mode in rescue_config.RESCUE_KNOWN_SENSOR_MODES.items():
        if model_name.lower() in model:
            return dict(mode)
    return None


class RescueCamera:
    """Abre uma camera CSI por indice explicito e devolve BGR corrigido."""

    def __init__(self, camera_index=None):
        from picamera2 import Picamera2

        self.camera_index = (
            rescue_config.RESCUE_CAMERA_INDEX
            if camera_index is None else int(camera_index))
        self._software_rotate = False

        camera_info = Picamera2.global_camera_info()
        print("[resgate] cameras publicadas pelo libcamera:")
        for index, info in enumerate(camera_info):
            print(f"  [{index}] {info}")

        if rescue_config.RESCUE_REQUIRE_TWO_CAMERAS and len(camera_info) < 2:
            raise RuntimeError(
                "o modo de resgate exige duas cameras conectadas; "
                f"o libcamera publicou somente {len(camera_info)}")
        if not 0 <= self.camera_index < len(camera_info):
            raise RuntimeError(
                f"indice de camera de resgate {self.camera_index} invalido; "
                f"indices disponiveis: 0..{len(camera_info) - 1}")

        print(
            f"[resgate] abrindo camera explicita {self.camera_index}. "
            "Confirme no --debug que esta e a camera frontal de resgate.")

        self.picam2 = Picamera2(camera_num=self.camera_index)
        sensor_mode = _known_sensor_mode(camera_info[self.camera_index])
        used_known_mode = sensor_mode is not None
        if used_known_mode:
            print(
                "[resgate] partida rapida: modo full-FoV conhecido para "
                f"{camera_info[self.camera_index].get('Model', 'camera')}")
        else:
            print(
                "[resgate] sensor desconhecido; consultando modos uma vez")
            sensor_mode = self._discover_sensor_mode()

        if sensor_mode is None:
            raise RuntimeError(
                "a camera de resgate nao publicou modos de sensor validos")

        camera_config = self._prepare_configuration(sensor_mode)
        try:
            self.picam2.configure(camera_config)
        except (TypeError, ValueError, RuntimeError) as err:
            if not used_known_mode:
                raise
            print(
                "[resgate] modo rapido recusado; descobrindo modos como "
                f"fallback: {err}")
            self._software_rotate = False
            sensor_mode = self._discover_sensor_mode()
            if sensor_mode is None:
                raise RuntimeError(
                    "fallback nao encontrou modo de sensor valido") from err
            camera_config = self._prepare_configuration(sensor_mode)
            self.picam2.configure(camera_config)

        self.picam2.start()

        if rescue_config.RESCUE_LENS_POSITION is not None:
            try:
                from libcamera import controls
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": rescue_config.RESCUE_LENS_POSITION,
                })
            except Exception as err:
                print(f"[resgate] foco manual ignorado: {err}")

        time.sleep(0.15)

    def _discover_sensor_mode(self):
        return _select_widest_sensor_mode(
            self.picam2.sensor_modes,
            rescue_config.RESCUE_CAMERA_FPS,
            rescue_config.RESCUE_CAMERA_MAX_WIDTH,
            rescue_config.RESCUE_CAMERA_MAX_HEIGHT,
        )

    def _prepare_configuration(self, sensor_mode):
        sensor_width, sensor_height = _mode_size(sensor_mode)
        crop = _mode_crop(sensor_mode)
        crop_aspect = crop[2] / max(crop[3], 1)
        output_width, output_height = _fit_output_size(
            crop_aspect,
            min(rescue_config.RESCUE_CAMERA_MAX_WIDTH, sensor_width),
            min(rescue_config.RESCUE_CAMERA_MAX_HEIGHT, sensor_height),
        )
        self.output_size = (output_width, output_height)

        mode_fps = float(
            sensor_mode.get("fps", rescue_config.RESCUE_CAMERA_FPS))
        requested_fps = min(rescue_config.RESCUE_CAMERA_FPS, mode_fps)
        frame_us = int(round(1_000_000 / max(requested_fps, 1.0)))

        print(
            "[resgate] modo de sensor com maior campo de visao: "
            f"{sensor_width}x{sensor_height}, crop={crop}, "
            f"max={mode_fps:.1f} fps")
        print(
            "[resgate] saida de video: "
            f"{output_width}x{output_height} a {requested_fps:.1f} fps; "
            f"rotacao 180 graus={'sim' if rescue_config.RESCUE_ROTATE_180 else 'nao'}")

        return self._create_configuration(sensor_mode, frame_us)

    def _create_configuration(self, sensor_mode, frame_us):
        main = {"size": self.output_size, "format": "RGB888"}
        common = {
            "main": main,
            "controls": {"FrameDurationLimits": (frame_us, frame_us)},
            "buffer_count": 4,
            # Cada iteracao deve receber um frame novo. O worker ja descarta
            # backlog; repetir o ultimo buffer confirmaria a mesma imagem.
            "queue": False,
        }

        transform = None
        if rescue_config.RESCUE_ROTATE_180:
            try:
                from libcamera import Transform
                transform = Transform(hflip=True, vflip=True)
            except (ImportError, TypeError):
                self._software_rotate = True

        if transform is not None:
            common["transform"] = transform

        sensor_request = {
            "output_size": _mode_size(sensor_mode),
            "bit_depth": int(sensor_mode.get("bit_depth", 0)),
        }
        if sensor_request["bit_depth"] > 0:
            try:
                return self.picam2.create_video_configuration(
                    sensor=sensor_request, **common)
            except (TypeError, ValueError, RuntimeError) as err:
                print(
                    "[resgate] API sensor exata indisponivel; "
                    f"tentando compatibilidade raw: {err}")

        try:
            return self.picam2.create_video_configuration(
                raw={"size": _mode_size(sensor_mode)}, **common)
        except (TypeError, ValueError, RuntimeError) as err:
            print(
                "[resgate] modo raw exato indisponivel; "
                f"usando configuracao compativel: {err}")

        fallback = {"main": main}
        if transform is not None:
            fallback["transform"] = transform
        try:
            return self.picam2.create_video_configuration(**fallback)
        except (TypeError, ValueError, RuntimeError):
            fallback.pop("transform", None)
            if rescue_config.RESCUE_ROTATE_180:
                self._software_rotate = True
            return self.picam2.create_video_configuration(**fallback)

    def get_frame(self):
        frame = self.picam2.capture_array("main")
        if frame.ndim == 3 and frame.shape[2] == 4:
            # Picamera2 entrega RGB888/XRGB888 em memoria na ordem B,G,R[,X],
            # que ja e a ordem nativa esperada pelo OpenCV.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"formato inesperado da camera de resgate: {frame.shape}")
        if self._software_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass
