#!/usr/bin/env python3
"""
tools/camera_smoke.py — sanidade da Picamera2 (Fase B).
Adapted from robot_v.3/Python/debug/cam_debug_1.py.

Imprime os sensor modes, captura 1 frame em 448×252 BGR, salva um JPEG e
mede o FPS efetivo por ~3 s.

Uso:  python3 -m shadow.tools.camera_smoke [--out camera_smoke.jpg]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from config import CAPTURE_FPS, camera_x, camera_y  # noqa: E402
from vision.capture import LineCamera  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Sanidade da câmera de linha")
    parser.add_argument("--out", default="camera_smoke.jpg", help="arquivo JPEG de saída")
    args = parser.parse_args()

    print("Abrindo câmera…")
    camera = LineCamera()

    try:
        print("\nSensor modes:")
        for i, mode in enumerate(camera.sensor_modes()):
            print(f"  [{i}] {mode}")

        frame = camera.get_frame()
        assert frame.shape == (camera_y, camera_x, 3), f"shape inesperado: {frame.shape}"
        if not cv2.imwrite(args.out, frame):
            raise RuntimeError(f"não consegui salvar {args.out}")
        print(f"\nFrame {camera_x}×{camera_y} BGR salvo em: {args.out}")

        print(f"Medindo FPS (~3 s; alvo {CAPTURE_FPS})…")
        n = 0
        start = time.monotonic()
        while time.monotonic() - start < 3:
            camera.get_frame()
            n += 1
        fps = n / (time.monotonic() - start)
        print(f"FPS efetivo: {fps:.1f}")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
