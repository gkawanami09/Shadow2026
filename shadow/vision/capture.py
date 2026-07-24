"""
vision/capture.py — Picamera2 wrapper: 640×480 RGB capture → 448×252 BGR.
Ported from Overengineering² Reading Dossier, Hotspot 1 (capture settings)
  Original source: robot_v.3/Python/main/line_cam.py (lines 543-550, 587-589)
Shadow2026 adaptations:
  - Camera Module 3 Wide → generic CSI module; the sensor-mode/bit-depth
    config is replaced by the video configuration proven on this exact
    hardware in ../Shadow2026/raspberry/camera_test.py (RGB888 main stream,
    FrameDurationLimits, buffer_count=4).
  - Capture at 640×480, downscale to 448×252 and convert RGB→BGR IMMEDIATELY
    (mission §2.4) — all dossier constants are calibrated to 448×252.
  - 40 fps target (FrameDurationLimits = 25000 µs).
  - LensPosition: OE² used 6.5 (manual focus ≈ 15 cm); Shadow2026's module is
    fixed-focus by default → config.LENS_POSITION = None disables the call.
    Set it in config.py if your module has AF.
"""

import time

import cv2

from config import (CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_WIDTH,
                    LENS_POSITION, LINE_CAMERA_INDEX, camera_x, camera_y)


class LineCamera:
    def __init__(self):
        from picamera2 import Picamera2  # import local: so existe no Pi

        camera_info = Picamera2.global_camera_info()
        if not 0 <= LINE_CAMERA_INDEX < len(camera_info):
            raise RuntimeError(
                "camera de segue-linha no indice "
                f"{LINE_CAMERA_INDEX} indisponivel; detectadas: "
                f"{camera_info}"
            )
        print(
            "[camera] abrindo camera de segue-linha explicita "
            f"{LINE_CAMERA_INDEX} (flat 2)"
        )
        self.picam2 = Picamera2(camera_num=LINE_CAMERA_INDEX)

        frame_us = int(1_000_000 / CAPTURE_FPS)
        try:
            video_config = self.picam2.create_video_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"},
                controls={"FrameDurationLimits": (frame_us, frame_us)},
                buffer_count=4,
            )
        except TypeError:
            # compatibilidade com Picamera2 antigo (mesmo fallback do Shadow2026)
            video_config = self.picam2.create_video_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"})

        self.picam2.configure(video_config)
        self.picam2.start()

        if LENS_POSITION is not None:
            try:
                from libcamera import controls
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                          "LensPosition": LENS_POSITION})
            except Exception as err:
                print(f"[camera] LensPosition ignorado (módulo sem AF?): {err}")

        time.sleep(0.1)

    def sensor_modes(self):
        return self.picam2.sensor_modes

    def get_frame(self):
        """Returns one 448×252 BGR frame (blocks until the next frame, ~25 ms)."""
        raw = self.picam2.capture_array("main")
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB)
        raw = cv2.resize(raw, (camera_x, camera_y))
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass
