"""Captura exclusiva da camera de resgate, mantendo a proporcao 4:3."""

import time

import cv2

import rescue_config


class RescueCamera:
    """Abre uma camera CSI por indice explicito e devolve BGR nativo."""

    def __init__(self, camera_index=None):
        from picamera2 import Picamera2

        self.camera_index = (
            rescue_config.RESCUE_CAMERA_INDEX
            if camera_index is None else int(camera_index))

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
        frame_us = int(1_000_000 / rescue_config.RESCUE_CAMERA_FPS)
        try:
            camera_config = self.picam2.create_video_configuration(
                main={
                    "size": (
                        rescue_config.RESCUE_CAMERA_WIDTH,
                        rescue_config.RESCUE_CAMERA_HEIGHT),
                    "format": "RGB888",
                },
                controls={"FrameDurationLimits": (frame_us, frame_us)},
                buffer_count=4,
            )
        except TypeError:
            camera_config = self.picam2.create_video_configuration(
                main={
                    "size": (
                        rescue_config.RESCUE_CAMERA_WIDTH,
                        rescue_config.RESCUE_CAMERA_HEIGHT),
                    "format": "RGB888",
                })

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

    def get_frame(self):
        frame = self.picam2.capture_array("main")
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass

