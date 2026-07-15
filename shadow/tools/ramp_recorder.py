#!/usr/bin/env python3
"""Grava amostras da camera para calibrar a deteccao visual de rampa.

Esta ferramenta e independente do controle: nao abre a serial e nunca envia
comandos aos motores. A imagem vem da mesma ``LineCamera`` usada pelo
segue-linha, com a mesma resolucao, orientacao e configuracao de captura.

Exemplos::

    python3 -m shadow.tools.ramp_recorder flat_1
    python3 -m shadow.tools.ramp_recorder subida_1 --duration 15

O video, os tempos de cada frame (CSV) e os metadados (JSON) sao salvos em
``ramp_recordings/`` na raiz do projeto.
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Mantem o mesmo modelo de imports das outras ferramentas executadas com -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

import config  # noqa: E402
from vision.capture import LineCamera  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ramp_recordings"


def safe_label(value):
    """Converte o rotulo informado em uma parte segura do nome do arquivo."""
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return label or "ramp"


def open_video_writer(output_dir, stem, fps, frame_size):
    """Tenta MP4 primeiro e usa AVI/MJPEG se o codec nao estiver disponivel."""
    candidates = ((".mp4", "mp4v"), (".avi", "MJPG"))

    for suffix, codec in candidates:
        path = output_dir / f"{stem}{suffix}"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            return writer, path, codec

        writer.release()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    raise RuntimeError(
        "OpenCV nao conseguiu abrir os codecs mp4v nem MJPG para gravacao")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grava a camera do Shadow2026 sem movimentar os motores.")
    parser.add_argument(
        "label", nargs="?", default="ramp",
        help="identificacao da passagem, por exemplo flat_1 ou subida_1")
    parser.add_argument(
        "--duration", type=float, default=12.0,
        help="duracao da gravacao em segundos; use 0 para gravar ate Ctrl+C")
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="contagem regressiva antes de iniciar (padrao: 3 s)")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="diretorio de destino (padrao: ramp_recordings na raiz)")
    args = parser.parse_args()

    if args.duration < 0:
        parser.error("--duration deve ser maior ou igual a zero")
    if args.delay < 0:
        parser.error("--delay deve ser maior ou igual a zero")
    return args


def countdown(delay):
    if delay <= 0:
        return

    print("Camera pronta. Posicione o robo; a gravacao vai iniciar em:")
    deadline = time.monotonic() + delay
    last_second = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        second = max(1, int(remaining) + 1)
        if second != last_second:
            print(f"  {second}...")
            last_second = second
        time.sleep(min(.05, remaining))


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    stem = f"{timestamp:%Y%m%d_%H%M%S}_{safe_label(args.label)}"

    camera = None
    writer = None
    csv_file = None
    video_path = None
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    codec = None
    frames = 0
    recording_started = None
    recording_finished = None

    try:
        print("Abrindo camera...")
        camera = LineCamera()
        countdown(args.delay)

        first_frame = camera.get_frame()
        height, width = first_frame.shape[:2]
        writer, video_path, codec = open_video_writer(
            output_dir, stem, float(config.CAPTURE_FPS), (width, height))

        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(("frame", "elapsed_s", "capture_interval_ms"))

        print(f"Gravando em {video_path}")
        print("Mova o robo manualmente pelo trecho. Ctrl+C encerra e salva.")

        recording_started = time.monotonic()
        last_capture = recording_started
        next_report = recording_started + 1.0
        frame = first_frame

        while True:
            captured_at = time.monotonic()
            elapsed = captured_at - recording_started
            if args.duration > 0 and elapsed >= args.duration:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
            csv_writer.writerow((frames, f"{elapsed:.6f}",
                                 f"{(captured_at - last_capture) * 1000:.3f}"))
            frames += 1
            last_capture = captured_at

            if captured_at >= next_report:
                print(f"  {elapsed:5.1f} s | {frames} frames")
                next_report = captured_at + 1.0

            frame = camera.get_frame()

    except KeyboardInterrupt:
        print("\nGravacao encerrada pelo operador.")
    finally:
        recording_finished = time.monotonic()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if camera is not None:
            camera.close()

    if video_path is None or recording_started is None:
        raise RuntimeError("A gravacao nao chegou a ser iniciada")

    elapsed_total = max(recording_finished - recording_started, 0.0)
    metadata = {
        "label": args.label,
        "recorded_at": timestamp.isoformat(),
        "video": video_path.name,
        "timestamps": csv_path.name,
        "codec": codec,
        "width": config.camera_x,
        "height": config.camera_y,
        "requested_fps": config.CAPTURE_FPS,
        "frames": frames,
        "elapsed_s": round(elapsed_total, 6),
        "measured_fps": round(frames / elapsed_total, 3) if elapsed_total else 0,
        "camera_mount_must_remain_fixed": True,
        "motors_commanded": False,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Gravacao salva:")
    print(f"  video: {video_path}")
    print(f"  tempos: {csv_path}")
    print(f"  dados:  {json_path}")
    print(f"  frames: {frames} | FPS medido: {metadata['measured_fps']}")


if __name__ == "__main__":
    main()
