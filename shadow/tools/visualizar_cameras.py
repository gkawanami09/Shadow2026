#!/usr/bin/env python3
"""
Exibe as duas câmeras CSI sem executar a detecção do robô.

A imagem esquerda é a câmera de resgate. Por padrão, somente ela recebe CLAHE
e correção gamma no canal de luminosidade para recuperar detalhes escuros.

Uso:
    python3 -m shadow.tools.visualizar_cameras
    python3 -m shadow.tools.visualizar_cameras --camera-a 1 --camera-b 0

Pressione q, Esc ou Ctrl-C para fechar.
"""

import argparse
import time

import cv2
import numpy as np


WINDOW = "Shadow2026 - duas cameras"


class RescueEnhancer:
    """Melhora iluminação local preservando os canais de cor."""

    def __init__(self, gamma):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        values = np.arange(256, dtype=np.float32) / 255.0
        self.gamma_lut = np.clip(
            np.power(values, 1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)

    def apply(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = self.clahe.apply(lightness)
        lightness = cv2.LUT(lightness, self.gamma_lut)
        enhanced = cv2.merge((lightness, channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


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
        description="Mostra duas câmeras CSI lado a lado, sem detecção")
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
    parser.add_argument("--rescue-gamma", type=float, default=1.5,
                        help="clareamento da câmera esquerda; maior clareia "
                             "mais (padrão: 1.5)")
    parser.add_argument("--no-rescue-filter", action="store_true",
                        help="mostra a câmera esquerda sem CLAHE/gamma")
    args = parser.parse_args()

    if args.camera_a == args.camera_b:
        parser.error("--camera-a e --camera-b precisam ser diferentes")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height e --fps precisam ser maiores que zero")
    if args.rescue_gamma <= 0:
        parser.error("--rescue-gamma precisa ser maior que zero")

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
    rescue_enhancer = (
        None if args.no_rescue_filter else RescueEnhancer(args.rescue_gamma))

    try:
        print(f"Abrindo câmeras {args.camera_a} e {args.camera_b}...")
        camera_a = open_camera(
            args.camera_a, args.width, args.height, args.fps)
        camera_b = open_camera(
            args.camera_b, args.width, args.height, args.fps)
        time.sleep(.2)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if rescue_enhancer is not None:
            print("Filtro de resgate ativo na imagem esquerda: "
                  f"CLAHE + gamma {args.rescue_gamma:.1f}")
        print("Visualização ativa — pressione q ou Esc para fechar.")

        while True:
            frame_a = get_bgr_frame(camera_a)
            frame_b = get_bgr_frame(camera_b)
            if rescue_enhancer is not None:
                frame_a = rescue_enhancer.apply(frame_a)
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
