#!/usr/bin/env python3
"""
Mostra duas câmeras CSI simultaneamente, sem executar a visão do robô.

Uso:
    python3 -m shadow.tools.dual_camera_viewer
    python3 -m shadow.tools.dual_camera_viewer --camera-a 1 --camera-b 0

Pressione q, Esc ou Ctrl-C para fechar.
"""

import argparse
import time

import cv2
import numpy as np


WINDOW = "Shadow2026 - duas cameras"


def open_camera(camera_num, width, height, fps):
    """Abre uma Picamera2 pelo índice publicado pelo libcamera."""
    from picamera2 import Picamera2

    camera = Picamera2(camera_num=camera_num)
    frame_us = int(1_000_000 / fps)

    try:
        camera_config = camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
            buffer_count=4,
        )
    except TypeError:
        # Compatibilidade com versões antigas do Picamera2.
        camera_config = camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"})

    camera.configure(camera_config)
    camera.start()
    return camera


def get_bgr_frame(camera):
    """Captura o frame sem aplicar qualquer algoritmo de visão."""
    frame = camera.capture_array("main")
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def close_camera(camera):
    if camera is None:
        return
    try:
        camera.stop()
    except Exception:
        pass
    try:
        camera.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Mostra duas câmeras CSI lado a lado, sem processamento")
    parser.add_argument("--camera-a", type=int, default=0,
                        help="índice da câmera mostrada à esquerda (padrão: 0)")
    parser.add_argument("--camera-b", type=int, default=1,
                        help="índice da câmera mostrada à direita (padrão: 1)")
    parser.add_argument("--width", type=int, default=640,
                        help="largura de cada imagem (padrão: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="altura de cada imagem (padrão: 480)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS solicitado para cada câmera (padrão: 30)")
    args = parser.parse_args()

    if args.camera_a == args.camera_b:
        parser.error("--camera-a e --camera-b precisam ser diferentes")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height e --fps precisam ser maiores que zero")

    from picamera2 import Picamera2

    camera_info = Picamera2.global_camera_info()
    print("Câmeras detectadas:")
    for index, info in enumerate(camera_info):
        print(f"  [{index}] {info}")

    required_index = max(args.camera_a, args.camera_b)
    if args.camera_a < 0 or args.camera_b < 0 or required_index >= len(camera_info):
        raise RuntimeError(
            f"foram detectadas {len(camera_info)} câmera(s), mas foram pedidos "
            f"os índices {args.camera_a} e {args.camera_b}")

    camera_a = None
    camera_b = None

    try:
        print(f"Abrindo câmeras {args.camera_a} e {args.camera_b}...")
        camera_a = open_camera(
            args.camera_a, args.width, args.height, args.fps)
        camera_b = open_camera(
            args.camera_b, args.width, args.height, args.fps)
        time.sleep(.2)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("Visualização ativa — pressione q ou Esc para fechar.")

        while True:
            frame_a = get_bgr_frame(camera_a)
            frame_b = get_bgr_frame(camera_b)
            combined = np.hstack((frame_a, frame_b))

            cv2.imshow(WINDOW, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("\nCtrl-C — encerrando...")
    finally:
        cv2.destroyAllWindows()
        close_camera(camera_b)
        close_camera(camera_a)
        print("Câmeras fechadas.")


if __name__ == "__main__":
    main()
