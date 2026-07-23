#!/usr/bin/env python3
"""Primeira etapa autonoma do resgate: detectar, alinhar e chegar perto.

Este executavel e independente do segue-linha. Nunca rode ``shadow/main.py`` e
``shadow/rescue_main.py`` ao mesmo tempo: cada um precisa ser o unico dono de
sua camera e da serial do Arduino.

Exemplos:
    python3 shadow/rescue_main.py --debug
    python3 shadow/rescue_main.py --camera-index 0 --drive --debug
    python3 shadow/rescue_main.py --video shadow/captures/resgate.mp4 --debug
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

import rescue_config as cfg  # noqa: E402
from control.rescue_approach import BallApproachController  # noqa: E402
from vision.rescue_ball import (BallDetector, annotate_rescue_frame)  # noqa: E402


WINDOW = "Shadow2026 - aproximacao da bolinha"


class VideoSource:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"nao foi possivel abrir o video: {path}")

    def get_frame(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self):
        self.capture.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "detecta uma esfera de resgate e, com --drive, aproxima o robo"))
    parser.add_argument(
        "--camera-index", type=int,
        help=(
            "indice libcamera da camera de resgate; sem --drive usa "
            f"{cfg.RESCUE_CAMERA_INDEX} como padrao"))
    parser.add_argument(
        "--target", choices=("any", "black", "silver"), default="any",
        help="tipo de esfera aceito (padrao: any)")
    parser.add_argument(
        "--drive", action="store_true",
        help="AUTORIZA movimento; sem esta opcao o Arduino nem e aberto")
    parser.add_argument(
        "--debug", action="store_true",
        help="mostra a camera anotada; q ou Esc encerra")
    parser.add_argument(
        "--video", type=Path,
        help="processa um video gravado em vez da camera; sempre sem motores")
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="desativa CLAHE + gamma nas propostas visuais")
    parser.add_argument(
        "--no-ultrasonic", action="store_true",
        help="nao usa o HC-SR04 como barreira auxiliar")
    args = parser.parse_args()
    if args.video is not None and args.drive:
        parser.error("--drive nao pode ser usado junto com --video")
    if args.drive and args.camera_index is None:
        parser.error(
            "--drive exige --camera-index explicito; valide antes qual imagem "
            "e da camera frontal de resgate usando --debug")
    if args.camera_index is None:
        args.camera_index = cfg.RESCUE_CAMERA_INDEX
    return args


def main():
    args = parse_args()
    source = None
    arduino = None
    motor_lock = None
    detector = BallDetector(
        target_kind=args.target, enhance=not args.no_enhance)
    controller = BallApproachController()
    last_state = None
    last_detail = None
    last_ultrasonic_poll = 0.0
    distance_mm = None

    try:
        if args.drive:
            from control.motor_lock import MotorLockError, MotorOwnerLock
            motor_lock = MotorOwnerLock("aproximacao-resgate")
            try:
                motor_lock.acquire()
            except MotorLockError as err:
                raise RuntimeError(
                    f"modo de resgate recusado: {err}") from err

        if args.video is not None:
            source = VideoSource(args.video)
            print(f"[resgate] replay sem motores: {args.video}")
        else:
            from vision.rescue_capture import RescueCamera
            source = RescueCamera(args.camera_index)

        if args.drive:
            from serial_link.arduino import Arduino
            from control.steer import init_steering, steer

            arduino = Arduino()
            init_steering(arduino)
            steer()
            print(
                "[resgate] MOVIMENTO AUTORIZADO. Mantendo PARAR por 3 s; "
                "afaste as maos e tenha acesso ao desligamento.")
            end_countdown = time.monotonic() + 3.0
            while time.monotonic() < end_countdown:
                arduino.refresh()
                time.sleep(0.05)
        else:
            print(
                "[resgate] modo de visao: motores desativados. "
                "Use --drive somente depois de validar o --debug.")

        while True:
            loop_started = time.monotonic()
            frame = source.get_frame()
            if frame is None:
                print("[resgate] fim da fonte de imagem")
                break

            detection = detector.detect(frame, timestamp=loop_started)

            distance_for_update = None
            ultrasonic_polled = False
            if (
                arduino is not None
                and not args.no_ultrasonic
                and detection is not None
                and detection.confirmed
                and loop_started - last_ultrasonic_poll
                >= cfg.BALL_ULTRASONIC_POLL_S
            ):
                distance_mm = arduino.distancia_ultrassom()
                distance_for_update = distance_mm
                ultrasonic_polled = True
                last_ultrasonic_poll = loop_started

            command = controller.update(
                detection,
                frame.shape,
                distance_mm=distance_for_update,
                ultrasonic_polled=ultrasonic_polled,
                now=time.monotonic())

            if arduino is not None:
                from control.steer import steer
                steer(command.angle, command.speed)
                arduino.refresh()

            if command.state != last_state or command.detail != last_detail:
                print(f"[resgate] {command.state}: {command.detail}")
                last_state = command.state
                last_detail = command.detail

            if args.debug:
                annotated = annotate_rescue_frame(
                    frame, detection, command.state, command.detail, distance_mm)
                cv2.imshow(WINDOW, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if command.terminal:
                if arduino is not None:
                    print(
                        f"[resgate] estado terminal {command.state}; "
                        "motores parados")
                    time.sleep(0.15)
                if not args.debug or arduino is not None:
                    break

    except RuntimeError as err:
        print(f"[resgate] ERRO: {err}")
    except KeyboardInterrupt:
        print("\n[resgate] Ctrl-C")
    finally:
        if arduino is not None:
            try:
                from control.steer import steer
                steer()
            finally:
                arduino.close()
        if source is not None:
            source.close()
        if motor_lock is not None:
            motor_lock.release()
        cv2.destroyAllWindows()
        if arduino is not None:
            print("[resgate] encerrado com PARAR")
        else:
            print("[resgate] encerrado; motores nunca foram habilitados")


if __name__ == "__main__":
    main()
