#!/usr/bin/env python3
"""Primeira etapa autonoma do resgate: detectar, alinhar e chegar perto.

Este executavel e independente do segue-linha. Nunca rode ``shadow/main.py`` e
``shadow/rescue_main.py`` ao mesmo tempo: cada um precisa ser o unico dono de
sua camera e da serial do Arduino.

Captura e detector pesado possuem threads proprias e caixas de apenas um frame.
O preview/controle sempre usa o dado mais recente, sem acumular imagens antigas.

Exemplos:
    python3 shadow/rescue_main.py --debug
    python3 shadow/rescue_main.py --camera-index 0 --drive --debug
    python3 shadow/rescue_main.py --video shadow/captures/resgate.mp4 --debug
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

import rescue_config as cfg  # noqa: E402
from control.rescue_approach import (  # noqa: E402
    BallApproachController,
    MotionCommand,
)
from vision.rescue_async import (  # noqa: E402
    FreshDetectionGate,
    LatestFrameBallDetector,
    LatestFrameSource,
)
from vision.rescue_ball import (BallDetector, annotate_rescue_frame)  # noqa: E402
from vision.rescue_dataset import RescueDatasetWriter  # noqa: E402


WINDOW = "Shadow2026 - aproximacao da bolinha"
IDLE_CONTROL_INTERVAL_S = 0.25
LOG_INTERVAL_S = 0.50
MAIN_TICK_S = 0.005


class VideoSource:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"nao foi possivel abrir o video: {path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_period = 1.0 / fps if 1.0 <= fps <= 120.0 else 1.0 / 30.0
        self.next_frame_at = time.monotonic()

    def get_frame(self):
        remaining = self.next_frame_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        ok, frame = self.capture.read()
        self.next_frame_at = max(
            self.next_frame_at + self.frame_period,
            time.monotonic(),
        )
        return frame if ok else None

    def close(self):
        self.capture.release()


def _rate(timestamps):
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def _sequence_rate(samples):
    """FPS da fonte mesmo quando o loop principal pula frames intermediarios."""
    if len(samples) < 2:
        return 0.0
    elapsed = samples[-1][1] - samples[0][1]
    return (
        (samples[-1][0] - samples[0][0]) / elapsed
        if elapsed > 0 else 0.0
    )


def _best_effort(label, action):
    try:
        return action()
    except Exception as err:
        print(f"[resgate] falha ao {label}: {err}")
        return None


def _dataset_metadata(
    args,
    command,
    frame_sequence,
    frame_captured_at,
    result,
    distance_mm,
    now,
):
    same_frame = bool(
        result is not None
        and result.source_sequence == frame_sequence
    )
    detection = result.detection if result is not None else None
    detection_data = None
    if detection is not None:
        detection_data = {
            "kind": detection.kind,
            "center_x": float(detection.center_x),
            "center_y": float(detection.center_y),
            "radius": float(detection.radius),
            "confidence": float(detection.confidence),
            "confirmed": bool(detection.confirmed),
            "hits": int(detection.hits),
        }

    detector_data = None
    if result is not None:
        detector_data = {
            # Candidatos/deteccao so descrevem este PNG quando esta flag e
            # true. Caso contrario servem apenas como contexto do loop.
            "same_frame": same_frame,
            "source_capture_sequence": result.source_sequence,
            "result_sequence": int(result.sequence),
            "age_s": float(max(now - result.captured_at, 0.0)),
            "processing_ms": float(result.processing_s * 1000.0),
            "hough_used": bool(result.hough_used),
            "contour_proposals": int(result.contour_proposals),
            "hough_proposals": int(result.hough_proposals),
            "candidate_count": int(result.candidate_count),
            "candidate_radii": list(result.candidate_radii),
            "diagnostic": result.diagnostic,
            "detection": detection_data,
        }

    return {
        "purpose": "rescue_ball_calibration",
        "raw_unannotated": True,
        "frame": {
            "capture_sequence": int(frame_sequence),
            "captured_monotonic_s": float(frame_captured_at),
        },
        "run": {
            "camera_index": int(args.camera_index),
            "target": args.target,
            "enhance": not args.no_enhance,
            "motors_enabled": bool(args.drive),
        },
        "control": {
            "state": command.state,
            "detail": command.detail,
            "angle": int(command.angle),
            "speed": float(command.speed),
            "terminal": bool(command.terminal),
            "ultrasonic_distance_mm": distance_mm,
        },
        "latest_detector_result": detector_data,
    }


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
        help=(
            "AUTORIZA movimento; sem esta opcao o Arduino permanece em PARAR "
            "(o LED ainda e apagado no uso da camera real)"))
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "mostra a camera anotada; s salva PNG bruto para calibracao; "
            "q ou Esc encerra"))
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
    capture_worker = None
    detector_worker = None
    dataset_worker = None
    controller = None
    hardware_session = args.video is None

    last_state = None
    last_logged_detail = None
    last_log_at = 0.0
    last_ultrasonic_poll = 0.0
    last_idle_control = 0.0
    distance_mm = None
    ultrasonic_sample_ready = False
    ultrasonic_sample_at = 0.0
    capture_samples = deque(maxlen=60)
    detection_times = deque(maxlen=30)

    try:
        if hardware_session:
            from control.motor_lock import MotorLockError, MotorOwnerLock
            motor_lock = MotorOwnerLock(
                "aproximacao-resgate" if args.drive else "visao-resgate")
            try:
                motor_lock.acquire()
            except MotorLockError as err:
                raise RuntimeError(
                    f"modo de resgate recusado: {err}") from err

            from serial_link.arduino import Arduino
            from control.steer import init_steering, steer

            arduino = Arduino()
            init_steering(arduino)
            steer()
            arduino.led("APAGADO")
            print(
                "[resgate] LED APAGADO antes de abrir a camera frontal; "
                "motores em PARAR")

        if args.video is not None:
            source = VideoSource(args.video)
            print(f"[resgate] replay sem motores: {args.video}")
        else:
            from vision.rescue_capture import RescueCamera
            source = RescueCamera(args.camera_index)

        capture_worker = LatestFrameSource(source)
        detector_worker = LatestFrameBallDetector(
            BallDetector(
                target_kind=args.target,
                enhance=not args.no_enhance,
            ),
            max_width=cfg.RESCUE_DETECTOR_MAX_WIDTH,
            max_height=cfg.RESCUE_DETECTOR_MAX_HEIGHT,
        )
        fresh_gate = FreshDetectionGate(cfg.BALL_ACQUIRE_HITS)

        loop_started = time.monotonic()
        armed_at = (
            loop_started + cfg.RESCUE_ARM_DELAY_S
            if args.drive else loop_started)
        controller = BallApproachController(start_time=armed_at)
        command = MotionCommand(
            "ARMING" if args.drive else controller.WAIT_TARGET,
            detail=(
                "camera ativa; mantendo PARAR durante a contagem"
                if args.drive else
                "parado; aguardando confirmacao temporal"),
        )
        last_frame_sequence = 0
        last_result_sequence = 0
        latest_frame = None
        latest_frame_captured_at = None
        latest_result = None
        latest_detection = None
        last_metrics_result = None

        if args.drive:
            print(
                "[resgate] MOVIMENTO AUTORIZADO. A camera ja esta ativa; "
                f"mantendo PARAR por {cfg.RESCUE_ARM_DELAY_S:.0f} s.")
        else:
            print(
                "[resgate] modo de visao: motores desativados. "
                "No --debug, pressione s para salvar um PNG bruto; "
                "use --drive somente depois de validar a visao.")

        while True:
            frame_packet = capture_worker.poll(last_frame_sequence)
            new_frame = frame_packet is not None
            if new_frame:
                last_frame_sequence = frame_packet.sequence
                latest_frame = frame_packet.frame
                latest_frame_captured_at = frame_packet.captured_at
                capture_samples.append((
                    frame_packet.sequence,
                    frame_packet.captured_at,
                ))
            elif capture_worker.ended:
                print("[resgate] fim da fonte de imagem")
                break

            now = time.monotonic()
            armed = now >= armed_at
            if new_frame and armed:
                detector_worker.submit(
                    latest_frame,
                    captured_at=frame_packet.captured_at,
                    source_sequence=frame_packet.sequence,
                )

            if (
                args.drive
                and arduino is not None
                and not args.no_ultrasonic
            ):
                sample_done, sample_value = arduino.poll_ultrassom()
                if sample_done:
                    distance_mm = sample_value
                    ultrasonic_sample_ready = True
                    ultrasonic_sample_at = now

            result = detector_worker.poll(last_result_sequence)
            if not detector_worker.is_alive:
                # poll() ja transforma uma excecao do worker em RuntimeError.
                detector_worker.poll(last_result_sequence)
                raise RuntimeError("detector assincrono encerrou inesperadamente")

            # Uma unica referencia temporal decide frescor e alimenta o
            # controlador; assim nao existe divergencia perto do limite stale.
            now = time.monotonic()
            command_updated = False
            if not armed:
                remaining = max(armed_at - now, 0.0)
                command = MotionCommand(
                    "ARMING",
                    detail=(
                        f"camera fluida; PARAR por mais {remaining:.1f} s"),
                )
            elif result is not None:
                last_result_sequence = result.sequence
                last_metrics_result = result
                detection_times.append(result.completed_at)
                result_age = now - result.captured_at

                if result_age > cfg.BALL_FRAME_STALE_S:
                    # Nunca consultar sensor nem mover com uma imagem que ja
                    # venceu enquanto era processada.
                    command = controller.update(
                        result.detection,
                        result.frame_shape,
                        now=now,
                    )
                    command_updated = True
                    latest_result = None
                    latest_detection = None
                    distance_mm = None
                    ultrasonic_sample_ready = False
                    fresh_gate.reset()
                    if arduino is not None:
                        arduino.cancelar_ultrassom()
                    last_idle_control = now
                else:
                    latest_result = result
                    control_detection = fresh_gate.accept(result.detection)
                    latest_detection = control_detection
                    if (
                        control_detection is None
                        or not control_detection.confirmed
                    ):
                        distance_mm = None
                        ultrasonic_sample_ready = False
                        if arduino is not None:
                            arduino.cancelar_ultrassom()

                    distance_for_update = None
                    ultrasonic_polled = False
                    if (
                        ultrasonic_sample_ready
                        and now - ultrasonic_sample_at
                        <= cfg.BALL_FRAME_STALE_S
                    ):
                        distance_for_update = distance_mm
                        ultrasonic_polled = True
                        ultrasonic_sample_ready = False
                    elif ultrasonic_sample_ready:
                        ultrasonic_sample_ready = False
                        distance_mm = None

                    command = controller.update(
                        control_detection,
                        result.frame_shape,
                        distance_mm=distance_for_update,
                        ultrasonic_polled=ultrasonic_polled,
                        now=now,
                    )
                    command_updated = True
                    if (
                        args.drive
                        and arduino is not None
                        and not args.no_ultrasonic
                        and control_detection is not None
                        and control_detection.confirmed
                        and not command.terminal
                        and not ultrasonic_sample_ready
                        and now - last_ultrasonic_poll
                        >= cfg.BALL_ULTRASONIC_POLL_S
                        and arduino.iniciar_ultrassom(
                            timeout=cfg.BALL_ULTRASONIC_TIMEOUT_S)
                    ):
                        last_ultrasonic_poll = now
            elif (
                latest_result is not None
                and now - latest_result.captured_at
                > cfg.BALL_FRAME_STALE_S
            ):
                command = controller.update(
                    latest_result.detection,
                    latest_result.frame_shape,
                    now=now,
                )
                command_updated = True
                latest_result = None
                latest_detection = None
                distance_mm = None
                ultrasonic_sample_ready = False
                fresh_gate.reset()
                if arduino is not None:
                    arduino.cancelar_ultrassom()
                last_idle_control = now
            elif (
                latest_result is None
                and now - last_idle_control >= IDLE_CONTROL_INTERVAL_S
            ):
                frame_shape = (
                    latest_frame.shape
                    if latest_frame is not None
                    else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                          cfg.RESCUE_CAMERA_MAX_WIDTH, 3)
                )
                command = controller.update(None, frame_shape, now=now)
                command_updated = True
                last_idle_control = now

            if args.drive and arduino is not None and command_updated:
                from control.steer import steer
                steer(command.angle, command.speed)
            if arduino is not None:
                arduino.refresh(fail_closed=True)

            log_now = time.monotonic()
            should_log = (
                command.state != last_state
                or (
                    command.detail != last_logged_detail
                    and log_now - last_log_at >= LOG_INTERVAL_S
                )
            )
            if should_log:
                print(f"[resgate] {command.state}: {command.detail}")
                last_state = command.state
                last_logged_detail = command.detail
                last_log_at = log_now

            result_age = (
                time.monotonic() - latest_result.captured_at
                if latest_result is not None else None)
            overlay_detection = (
                latest_detection
                if (
                    latest_result is not None
                    and latest_detection is not None
                    and result_age <= cfg.BALL_FRAME_STALE_S
                )
                else None)
            detector_fps = _rate(detection_times)
            processing_ms = (
                last_metrics_result.processing_s * 1000.0
                if last_metrics_result is not None else 0.0)
            dropped = (
                last_metrics_result.dropped_frames
                if last_metrics_result is not None else 0)
            vision_mode = (
                "H" if (
                    last_metrics_result is not None
                    and last_metrics_result.hough_used
                ) else "C"
            )
            candidate_count = (
                last_metrics_result.candidate_count
                if last_metrics_result is not None else 0)
            hough_proposals = (
                last_metrics_result.hough_proposals
                if last_metrics_result is not None else 0)
            candidate_radii = (
                last_metrics_result.candidate_radii
                if last_metrics_result is not None else ())
            diagnostic = (
                last_metrics_result.diagnostic
                if last_metrics_result is not None else "inicio")
            radii_text = (
                " r" + "/".join(
                    f"{radius:.0f}" for radius in candidate_radii)
                if candidate_radii else "")
            performance_text = (
                f"cam {_sequence_rate(capture_samples):.1f} | "
                f"vis {detector_fps:.1f} | "
                f"{processing_ms:.0f}ms | "
                f"{vision_mode}{candidate_count}/{hough_proposals}:"
                f"{diagnostic}{radii_text} | "
                f"d{dropped}")

            if (
                args.debug
                and latest_frame is not None
                and (new_frame or command_updated)
            ):
                annotated = annotate_rescue_frame(
                    latest_frame,
                    overlay_detection,
                    command.state,
                    command.detail,
                    distance_mm,
                    motors_enabled=args.drive,
                    performance_text=performance_text,
                )
                cv2.imshow(WINDOW, annotated)

            if args.debug:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    if args.drive:
                        print(
                            "[dataset] captura recusada com --drive; "
                            "colecione imagens com os motores desativados")
                    elif (
                        latest_frame is None
                        or latest_frame_captured_at is None
                    ):
                        print("[dataset] nenhum frame disponivel para salvar")
                    else:
                        if dataset_worker is None:
                            dataset_worker = RescueDatasetWriter()
                            print(
                                "[dataset] sessao criada em "
                                f"{dataset_worker.session_dir}")
                        submitted = dataset_worker.submit(
                            latest_frame,
                            _dataset_metadata(
                                args,
                                command,
                                last_frame_sequence,
                                latest_frame_captured_at,
                                last_metrics_result,
                                distance_mm,
                                time.monotonic(),
                            ),
                        )
                        if submitted.accepted:
                            print(
                                "[dataset] frame bruto aceito: "
                                f"{submitted.capture_id}")
                        elif submitted.status == "mailbox_full":
                            print(
                                "[dataset] gravacao ocupada; aguarde antes "
                                "de pressionar s novamente")
                        else:
                            print(
                                "[dataset] captura recusada: "
                                f"{submitted.status}")

            if command.terminal:
                if args.drive:
                    print(
                        f"[resgate] estado terminal {command.state}; "
                        "motores parados")
                if not args.debug or args.drive:
                    break

            time.sleep(MAIN_TICK_S)

    except RuntimeError as err:
        print(f"[resgate] ERRO: {err}")
    except KeyboardInterrupt:
        print("\n[resgate] Ctrl-C")
    finally:
        # PARAR vem antes de aguardar worker/camera, inclusive em excecoes.
        if arduino is not None:
            from control.steer import steer
            _best_effort("parar os motores", steer)
        if dataset_worker is not None:
            dataset_closed = _best_effort(
                "encerrar a gravacao do dataset",
                lambda: dataset_worker.close(
                    timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S),
            )
            if dataset_closed is False:
                print(
                    "[dataset] AVISO: gravacao nao encerrou no prazo; "
                    "motores ja estao em PARAR")
            if dataset_worker.failed_count:
                print(
                    "[dataset] AVISO: "
                    f"{dataset_worker.failed_count} captura(s) falharam; "
                    f"ultimo erro: {dataset_worker.last_error}")
        if detector_worker is not None:
            detector_closed = _best_effort(
                "encerrar o detector",
                lambda: detector_worker.close(
                    timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S),
            )
            if detector_closed is False:
                print(
                    "[resgate] AVISO: detector nao encerrou no prazo; "
                    "processo permanecera com a thread daemon")
        if capture_worker is not None:
            capture_closed = _best_effort(
                "encerrar a captura",
                lambda: capture_worker.close(
                    timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S),
            )
            if capture_closed is False:
                print(
                    "[resgate] AVISO: captura nao encerrou no prazo; "
                    "motores ja estao em PARAR")
        elif source is not None:
            _best_effort("fechar a fonte de imagem", source.close)
        if arduino is not None:
            _best_effort("fechar o Arduino", arduino.close)
        if motor_lock is not None:
            _best_effort("liberar a trava dos motores", motor_lock.release)
        _best_effort("fechar a janela", cv2.destroyAllWindows)
        if args.drive:
            print("[resgate] encerrado com PARAR")
        else:
            print("[resgate] encerrado; motores nunca foram habilitados")


if __name__ == "__main__":
    main()
