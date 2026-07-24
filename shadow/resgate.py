#!/usr/bin/env python3
"""Resgate: procura, aproxima, coleta e deposita todas as esferas visiveis.

Este executavel e independente do segue-linha. Nunca rode ``shadow/main.py`` e
``shadow/resgate.py`` ao mesmo tempo: cada um precisa ser o unico dono de
sua camera e da serial do Arduino.

Captura e detector pesado possuem threads proprias e caixas de apenas um frame.
O preview/controle sempre usa o dado mais recente, sem acumular imagens antigas.

Exemplos:
    python3 shadow/resgate.py --debug
    python3 shadow/resgate.py --camera-index 0 --drive --debug
    python3 shadow/resgate.py --video shadow/captures/resgate.mp4 --debug
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config_resgate as cfg  # noqa: E402
from controle.aproximacao_resgate import (  # noqa: E402
    BallApproachController,
    MotionCommand,
)
from controle.busca_resgate import BallSearchController  # noqa: E402
from controle.coleta_resgate import BallPickupSequencer  # noqa: E402
from controle.deposito_resgate import DepositMarkerController  # noqa: E402
from visao.resgate_assincrono import (  # noqa: E402
    FreshDetectionGate,
    LatestFrameBallDetector,
    LatestFrameSource,
)
from visao.bola_resgate import (BallDetector, annotate_rescue_frame)  # noqa: E402
from visao.dados_resgate import RescueDatasetWriter  # noqa: E402
from visao.marcador_resgate import MarkerDetector  # noqa: E402


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


def _annotate_marker(frame, detection, target_kind):
    """Desenha somente a telemetria do triangulo usado no transporte."""
    if target_kind not in ("green", "red"):
        return frame
    color = (0, 210, 0) if target_kind == "green" else (0, 0, 255)
    cv2.putText(
        frame,
        f"DESTINO: {target_kind.upper()}",
        (max(frame.shape[1] - 235, 8), 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    if detection is None:
        return frame
    x, y, width, height = detection.bbox
    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + width))
    y1 = int(round(y + height))
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.circle(
        frame,
        (int(round(detection.center_x)),
         int(round(detection.center_y))),
        4,
        color,
        -1,
    )
    cv2.putText(
        frame,
        (
            f"{detection.kind} {detection.confidence:.2f} "
            f"hits={detection.hits}"
            f"{' LOCK' if detection.track_locked else ''}"
        ),
        (max(x0, 4), max(y0 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def _annotate_arena_guard(frame, result):
    """Mostra a soleira virtual e as exclusoes usadas na aquisicao."""
    if result is None:
        return frame
    points = tuple(getattr(result, "arena_boundary_points", ()))
    reason = str(getattr(result, "arena_reason", ""))
    accepted = getattr(result, "arena_accepted", None)
    blocked = (
        accepted is False
        or (accepted is None and reason not in ("", "ok"))
    )
    arena_color = (0, 80, 255) if blocked else (255, 210, 0)
    if points:
        cv2.polylines(
            frame,
            [np.asarray(points, dtype=np.int32)],
            False,
            arena_color,
            2,
            cv2.LINE_AA,
        )
        label_y = max(min(point[1] for point in points) - 8, 18)
        if not blocked:
            cv2.putText(
                frame,
                (
                    "SOLEIRA ARENA "
                    f"{getattr(result, 'arena_confidence', 0.0):.2f}"
                ),
                (8, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                arena_color,
                2,
                cv2.LINE_AA,
            )
    if blocked:
        cv2.putText(
            frame,
            f"ARENA BLOQUEADA: {reason}",
            (
                8,
                (
                    max(min(point[1] for point in points) - 8, 18)
                    if points
                    else max(int(round(frame.shape[0] * 0.46)), 20)
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            arena_color,
            2,
            cv2.LINE_AA,
        )

    support_box = getattr(result, "arena_support_box", None)
    if support_box is not None:
        x0, y0, x1, y1 = support_box
        cv2.rectangle(
            frame, (x0, y0), (x1, y1), arena_color, 1)
    for x, y, width, height, kind in getattr(
        result, "marker_exclusion_boxes", ()
    ):
        color = (0, 190, 0) if kind == "green" else (0, 0, 220)
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            2,
        )
        cv2.putText(
            frame,
            f"{kind}: NAO E BOLA",
            (x, max(y - 6, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def _reset_for_next_search(detector_worker, fresh_gate, now):
    """Invalida o alvo anterior e cria um ciclo independente de busca."""
    detector_worker.reset_tracking()
    fresh_gate.reset()
    return (
        BallSearchController(start_time=now),
        BallPickupSequencer(),
    )


def _new_deposit_navigation(pickup, now):
    """Cria o destino imutavel somente com a esfera presa e elevada."""
    if not pickup.ready_for_deposit:
        raise RuntimeError(
            "navegacao de deposito exige esfera presa e elevada")
    try:
        marker_kind = cfg.DEPOSIT_MARKER_BY_BALL_KIND[
            pickup.target_kind]
    except KeyError as err:
        raise RuntimeError(
            f"cor de esfera sem destino: {pickup.target_kind}") from err
    return (
        MarkerDetector(marker_kind),
        DepositMarkerController(marker_kind, start_time=now),
    )


def _authorize_marker_deposit(pickup, deposit_controller):
    """Impede liberar antes de PARAR no triangulo da cor congelada."""
    expected_marker = cfg.DEPOSIT_MARKER_BY_BALL_KIND.get(
        pickup.target_kind)
    if (
        not pickup.ready_for_deposit
        or deposit_controller is None
        or not deposit_controller.arrived
        or deposit_controller.target_kind != expected_marker
    ):
        return False
    return pickup.resume_deposit()


def _apply_pickup_actions(
    step,
    arduino,
    steer_action,
    expected_connection_epoch=None,
):
    """Aplica somente os eventos one-shot emitidos pelo sequenciador."""
    forward_started = False
    motor_stopped = False

    def link_changed():
        return (
            expected_connection_epoch is not None
            and (
                not arduino.connected
                or arduino.connection_epoch != expected_connection_epoch
            )
        )

    def link_error():
        return "serial mudou durante a coleta; sequencia cancelada"

    def abort(detail):
        # No passo das garras o avanco ja foi iniciado no estado anterior.
        # Qualquer falha precisa cortar as rodas aqui, sem esperar o proximo
        # tick; o caller ainda repete PARAR ao entrar em PICKUP_FAULT.
        if (
            not motor_stopped
            and (forward_started or step.gripper_action is not None)
        ):
            try:
                steer_action()
            except Exception:
                pass
        return detail

    try:
        if link_changed():
            return abort(link_error())
        if step.motor_action == "hold":
            # PARAR tambem cortaria o Futaba no firmware. LADO 0 0 zera as
            # quatro rodas e vira um keepalive que nao toca no canal CH3.
            if arduino.lado(0, 0) is False:
                return "LADO 0 0 nao foi enviado pela serial"
        elif step.motor_action == "stop":
            if steer_action() is False:
                return "PARAR nao foi enviado pela serial"
            motor_stopped = True
        elif step.motor_action not in ("", "forward"):
            return f"acao de motor desconhecida: {step.motor_action}"
        if link_changed():
            return abort(link_error())

        if step.stop_futaba:
            if arduino.parar_futaba() is False:
                return "FUTABA PARAR nao foi enviado pela serial"
            if link_changed():
                return abort(link_error())

        if step.futaba_action is not None:
            potencia, tempo_ms = step.futaba_action
            if arduino.futaba(potencia, tempo_ms) is False:
                return "FUTABA nao foi enviado pela serial"
            if link_changed():
                return abort(link_error())

        # Motores sao aplicados antes dos atuadores: isso garante PARAR antes
        # de fechar. Nos passos do elevador, FUTABA PARAR vem antes do novo
        # pulso; LADO 0 0 preserva o keepalive sem cortar o Futaba ativo.
        if step.motor_action == "forward":
            if steer_action(step.angle, step.speed) is False:
                return "comando de avanco nao foi enviado pela serial"
            forward_started = True
            if link_changed():
                return abort(link_error())

        if step.gripper_action is not None:
            esquerda, direita = step.gripper_action
            if arduino.garras(esquerda, direita) is False:
                return abort(
                    "comando simultaneo das garras nao foi enviado")
            if link_changed():
                return abort(link_error())
    except Exception as err:
        return abort(f"falha ao comandar coleta: {err}")
    return None


def _dataset_metadata(
    args,
    command,
    frame_sequence,
    frame_captured_at,
    result,
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
            "track_locked": bool(detection.track_locked),
        }

    detector_data = None
    if result is not None:
        locked_detection = result.locked_detection
        locked_detection_data = None
        if locked_detection is not None:
            locked_detection_data = {
                "kind": locked_detection.kind,
                "center_x": float(locked_detection.center_x),
                "center_y": float(locked_detection.center_y),
                "radius": float(locked_detection.radius),
                "confidence": float(locked_detection.confidence),
                "confirmed": bool(locked_detection.confirmed),
                "hits": int(locked_detection.hits),
                "timestamp": float(locked_detection.timestamp),
                "track_locked": bool(
                    locked_detection.track_locked),
            }
        crescent = result.crescent_evidence
        crescent_data = None
        if crescent is not None:
            crescent_data = {
                "accepted": bool(crescent.accepted),
                "confidence": float(crescent.confidence),
                "support": float(crescent.support),
                "left_support": float(crescent.left_support),
                "center_support": float(crescent.center_support),
                "right_support": float(crescent.right_support),
                "contrast": float(crescent.contrast),
                "center_x_ratio": float(crescent.center_x_ratio),
                "top_y_ratio": float(crescent.top_y_ratio),
                "halfspan_ratio": float(crescent.halfspan_ratio),
                "gradient_polarity": float(
                    crescent.gradient_polarity),
                "profile_support": float(crescent.profile_support),
                "profile_polarity": float(crescent.profile_polarity),
                "coherent_run": float(crescent.coherent_run),
                "circle_rmse_ratio": float(
                    crescent.circle_rmse_ratio),
                "curvature_score": float(crescent.curvature_score),
                "foil_fallback": bool(crescent.foil_fallback),
                "foil_texture_bins": int(
                    crescent.foil_texture_bins),
                "foil_valid_bins": int(crescent.foil_valid_bins),
                "interior_edge_density": float(
                    crescent.interior_edge_density),
                "background_edge_density": float(
                    crescent.background_edge_density),
            }
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
            "candidate_circles": [
                list(circle) for circle in result.candidate_circles
            ],
            "arena_guard": {
                "reason": result.arena_reason,
                "confidence": float(result.arena_confidence),
                "accepted": result.arena_accepted,
                "floor_support": float(result.arena_floor_support),
                "boundary_points": [
                    list(point)
                    for point in result.arena_boundary_points
                ],
                "support_box": (
                    list(result.arena_support_box)
                    if result.arena_support_box is not None
                    else None
                ),
            },
            "marker_exclusion_boxes": [
                list(box) for box in result.marker_exclusion_boxes
            ],
            "crescent_evidence": crescent_data,
            "diagnostic": result.diagnostic,
            "detection": detection_data,
            "locked_detection": locked_detection_data,
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
            "pickup_in_range": bool(command.pickup_in_range),
            "pickup_confirmations": int(command.pickup_confirmations),
        },
        "latest_detector_result": detector_data,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "detecta uma esfera e, com --drive, aproxima e aciona a coleta"))
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
    # Compatibilidade com comandos antigos. O sensor permanece desativado
    # independentemente desta flag.
    parser.add_argument(
        "--no-ultrasonic", action="store_true", help=argparse.SUPPRESS)
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
    search = None
    pickup = None
    marker_detector = None
    deposit_controller = None
    hardware_session = args.video is None

    last_state = None
    last_logged_detail = None
    last_log_at = 0.0
    last_idle_control = 0.0
    pickup_connection_epoch = None
    search_connection_epoch = None
    completed_pickups = 0
    latest_marker_detection = None
    latest_marker_captured_at = None
    capture_samples = deque(maxlen=60)
    detection_times = deque(maxlen=30)

    try:
        if hardware_session:
            from controle.trava_motores import MotorLockError, MotorOwnerLock
            motor_lock = MotorOwnerLock(
                "aproximacao-resgate" if args.drive else "visao-resgate")
            try:
                motor_lock.acquire()
            except MotorLockError as err:
                raise RuntimeError(
                    f"modo de resgate recusado: {err}") from err

            from comunicacao_serial.arduino import Arduino
            from controle.direcao import init_steering, steer

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
            from visao.captura_resgate import RescueCamera
            source = RescueCamera(args.camera_index)

        capture_worker = LatestFrameSource(source)
        detector_worker = LatestFrameBallDetector(
            BallDetector(
                target_kind=args.target,
                enhance=not args.no_enhance,
                enforce_arena=True,
                exclude_markers=True,
            ),
            max_width=cfg.RESCUE_DETECTOR_MAX_WIDTH,
            max_height=cfg.RESCUE_DETECTOR_MAX_HEIGHT,
        )
        fresh_gate = FreshDetectionGate(
            cfg.BALL_ACQUIRE_HITS,
            max_misses=cfg.BALL_FRESH_GATE_MAX_MISSES,
        )

        loop_started = time.monotonic()
        armed_at = (
            loop_started + cfg.RESCUE_ARM_DELAY_S
            if args.drive else loop_started)
        controller = (
            None
            if args.drive
            else BallApproachController(start_time=armed_at)
        )
        search = (
            BallSearchController(start_time=armed_at)
            if args.drive
            else None
        )
        pickup = BallPickupSequencer()
        command = MotionCommand(
            (
                "ARMING"
                if args.drive
                else BallApproachController.WAIT_TARGET
            ),
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
            if new_frame and armed and not pickup.started:
                detector_worker.submit(
                    latest_frame,
                    captured_at=frame_packet.captured_at,
                    source_sequence=frame_packet.sequence,
                )

            result = detector_worker.poll(last_result_sequence)
            if not detector_worker.is_alive:
                # poll() ja transforma uma excecao do worker em RuntimeError.
                detector_worker.poll(last_result_sequence)
                raise RuntimeError("detector assincrono encerrou inesperadamente")

            # Uma unica referencia temporal decide frescor e alimenta o
            # controlador; assim nao existe divergencia perto do limite stale.
            now = time.monotonic()
            command_updated = False
            pickup_step = None
            if pickup.ready_for_deposit:
                # A cor fica congelada desde o fechamento das garras. Enquanto
                # a esfera esta elevada, a visao de bolas permanece desligada
                # e somente o triangulo correspondente pode comandar o robo.
                if result is not None:
                    last_result_sequence = result.sequence
                    last_metrics_result = result
                    detection_times.append(result.completed_at)
                if (
                    arduino is not None
                    and pickup_connection_epoch is not None
                    and arduino.connection_epoch
                    != pickup_connection_epoch
                ):
                    pickup_step = pickup.fail(
                        "serial reconectou durante o transporte; "
                        "esfera mantida e sequencia cancelada")
                    command = pickup_step.motion_command()
                    command_updated = True
                else:
                    if marker_detector is None:
                        (
                            marker_detector,
                            deposit_controller,
                        ) = _new_deposit_navigation(
                            pickup,
                            now,
                        )
                        marker_kind = deposit_controller.target_kind
                        detector_worker.reset_tracking()
                        fresh_gate.reset()
                        latest_result = None
                        latest_detection = None
                        latest_marker_detection = None
                        latest_marker_captured_at = None
                        last_idle_control = 0.0
                        print(
                            f"[deposito] esfera {pickup.target_kind} presa; "
                            f"procurando triangulo {marker_kind}")

                    marker_frame_shape = (
                        latest_frame.shape
                        if latest_frame is not None
                        else (
                            cfg.RESCUE_CAMERA_MAX_HEIGHT,
                            cfg.RESCUE_CAMERA_MAX_WIDTH,
                            3,
                        )
                    )
                    marker_result = None
                    if (
                        new_frame
                        and deposit_controller.frame_allowed(
                            frame_packet.captured_at)
                    ):
                        marker_result = marker_detector.detect(
                            latest_frame,
                            timestamp=frame_packet.captured_at,
                        )
                        marker_now = time.monotonic()
                        if (
                            marker_now - frame_packet.captured_at
                            > cfg.BALL_FRAME_STALE_S
                        ):
                            marker_result = None
                        latest_marker_detection = marker_result
                        latest_marker_captured_at = frame_packet.captured_at
                        command = deposit_controller.update(
                            marker_result,
                            marker_frame_shape,
                            now=marker_now,
                        )
                        command_updated = True
                        last_idle_control = marker_now
                    elif (
                        latest_marker_captured_at is not None
                        and now - latest_marker_captured_at
                        > cfg.BALL_FRAME_STALE_S
                    ):
                        latest_marker_detection = None
                        latest_marker_captured_at = None
                        command = deposit_controller.update(
                            None,
                            marker_frame_shape,
                            now=now,
                        )
                        command_updated = True
                        last_idle_control = now
                    elif (
                        now - last_idle_control
                        >= IDLE_CONTROL_INTERVAL_S
                    ):
                        command = deposit_controller.update(
                            None,
                            marker_frame_shape,
                            now=now,
                        )
                        command_updated = True
                        last_idle_control = now
            elif pickup.started:
                # Consumir um eventual resultado que ja estava em voo somente
                # para telemetria. A visao nunca volta a comandar os motores
                # depois que a coleta foi armada.
                if result is not None:
                    last_result_sequence = result.sequence
                    last_metrics_result = result
                    detection_times.append(result.completed_at)
                if (
                    arduino is not None
                    and pickup_connection_epoch is not None
                    and arduino.connection_epoch
                    != pickup_connection_epoch
                ):
                    pickup_step = pickup.fail(
                        "serial reconectou durante a coleta; "
                        "sequencia cancelada")
                else:
                    pickup_step = pickup.update(now)
                command = pickup_step.motion_command()
                command_updated = True
            elif not armed:
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
                    # Nunca mover com uma imagem que venceu durante o
                    # processamento.
                    if search is not None:
                        command = search.update(None, now=now)
                    else:
                        command = controller.update(
                            result.detection,
                            result.frame_shape,
                            crescent_evidence=result.crescent_evidence,
                            now=now,
                        )
                    command_updated = True
                    latest_result = None
                    latest_detection = None
                    fresh_gate.reset()
                    last_idle_control = now
                else:
                    latest_result = result
                    if (
                        search is not None
                        and not search.frame_allowed(result.captured_at)
                    ):
                        # Um frame capturado enquanto o chassi ainda girava
                        # nao pode reconfirmar o alvo depois de PARAR.
                        fresh_gate.reset()
                        control_detection = None
                    else:
                        control_detection = fresh_gate.accept(
                            result.detection)
                    latest_detection = control_detection

                    if search is not None:
                        command = search.update(
                            control_detection,
                            now=now,
                        )
                        if search.target_acquired:
                            controller = BallApproachController(
                                start_time=now)
                            command = controller.update(
                                control_detection,
                                result.frame_shape,
                                crescent_evidence=(
                                    result.crescent_evidence),
                                now=now,
                            )
                            search = None
                            search_connection_epoch = None
                    else:
                        command = controller.update(
                            control_detection,
                            result.frame_shape,
                            crescent_evidence=result.crescent_evidence,
                            now=now,
                        )
                    command_updated = True
            elif (
                latest_result is not None
                and now - latest_result.captured_at
                > cfg.BALL_FRAME_STALE_S
            ):
                if search is not None:
                    command = search.update(None, now=now)
                else:
                    command = controller.update(
                        latest_result.detection,
                        latest_result.frame_shape,
                        crescent_evidence=(
                            latest_result.crescent_evidence),
                        now=now,
                    )
                command_updated = True
                latest_result = None
                latest_detection = None
                fresh_gate.reset()
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
                if search is not None:
                    command = search.update(None, now=now)
                else:
                    command = controller.update(
                        None, frame_shape, now=now)
                command_updated = True
                last_idle_control = now

            if (
                args.drive
                and search is None
                and controller is not None
                and not pickup.started
                and command.state == BallApproachController.WAIT_TARGET
            ):
                # Se um alvo sumiu durante a aproximacao, nao espere parado
                # por 30 s: descarte esse track e volte a procurar.
                search, pickup = _reset_for_next_search(
                    detector_worker,
                    fresh_gate,
                    now,
                )
                search_connection_epoch = None
                controller = None
                latest_result = None
                latest_detection = None
                last_idle_control = 0.0
                command = search.update(None, now=now)
                command_updated = True

            motor_result = None
            motion_connection_epoch = None
            if args.drive and arduino is not None:
                from controle.direcao import steer
                if pickup_step is not None:
                    pickup_error = _apply_pickup_actions(
                        pickup_step,
                        arduino,
                        steer,
                        expected_connection_epoch=pickup_connection_epoch,
                    )
                    if (
                        pickup_error is None
                        and not pickup_step.terminal
                        and pickup_connection_epoch is not None
                        and arduino.connection_epoch
                        != pickup_connection_epoch
                    ):
                        pickup_error = (
                            "serial reconectou durante a coleta; "
                            "sequencia cancelada")
                    if pickup_error is None:
                        action_completed_at = time.monotonic()
                        if pickup_step.futaba_action is not None:
                            pickup.mark_futaba_started(action_completed_at)
                        if pickup_step.motor_action == "forward":
                            pickup.mark_forward_started(action_completed_at)
                        if pickup_step.gripper_action is not None:
                            pickup.mark_grippers_started(action_completed_at)
                    if pickup_error is not None:
                        pickup_step = pickup.fail(pickup_error)
                        command = pickup_step.motion_command()
                        _apply_pickup_actions(
                            pickup_step,
                            arduino,
                            steer,
                        )
                        command_updated = True
                    elif (
                        pickup_step.state
                        == BallPickupSequencer.COMPLETE
                    ):
                        completed_pickups += 1
                        search, pickup = _reset_for_next_search(
                            detector_worker,
                            fresh_gate,
                            action_completed_at,
                        )
                        search_connection_epoch = None
                        controller = None
                        marker_detector = None
                        deposit_controller = None
                        pickup_connection_epoch = None
                        latest_result = None
                        latest_detection = None
                        latest_marker_detection = None
                        latest_marker_captured_at = None
                        detection_times.clear()
                        last_idle_control = 0.0
                        command = search.update(
                            None, now=action_completed_at)
                        command_updated = True
                        pickup_step = None
                        print(
                            f"[resgate] deposito {completed_pickups} "
                            "concluido; procurando a proxima esfera")
                elif command_updated:
                    motion_connection_epoch = arduino.connection_epoch
                    motor_result = steer(command.angle, command.speed)
                    if motor_result is False:
                        raise RuntimeError(
                            "comando de movimento nao foi enviado "
                            "pela serial")
                    if deposit_controller is not None:
                        if (
                            pickup_connection_epoch is not None
                            and (
                                not arduino.connected
                                or arduino.connection_epoch
                                != pickup_connection_epoch
                            )
                        ):
                            raise RuntimeError(
                                "serial mudou durante o transporte; "
                                "motores parados e deposito bloqueado")
                        action_completed_at = time.monotonic()
                        active_deposit = deposit_controller
                        if active_deposit.consume_tracking_reset():
                            marker_detector.reset()
                            latest_marker_detection = None
                            latest_marker_captured_at = None
                        if (
                            command.state
                            == DepositMarkerController.START
                        ):
                            active_deposit.mark_rotation_started(
                                action_completed_at)
                        elif (
                            command.state
                            == DepositMarkerController.TARGET_STOP
                        ):
                            active_deposit.mark_target_stopped(
                                action_completed_at)
                            marker_detector.reset()
                            latest_marker_detection = None
                            latest_marker_captured_at = None
                            last_idle_control = action_completed_at
                        elif (
                            command.state
                            == DepositMarkerController.TURN_STOP
                        ):
                            active_deposit.mark_full_turn_stopped(
                                action_completed_at)
                            marker_detector.reset()
                            latest_marker_detection = None
                            latest_marker_captured_at = None
                            last_idle_control = action_completed_at
                        elif (
                            command.state
                            == DepositMarkerController.LOST_STOP
                        ):
                            active_deposit.mark_lost_stopped(
                                action_completed_at)
                            marker_detector.reset()
                            latest_marker_detection = None
                            latest_marker_captured_at = None
                            last_idle_control = action_completed_at
                        elif (
                            command.state
                            == DepositMarkerController.ARRIVAL_STOP
                        ):
                            active_deposit.mark_arrival_stopped(
                                action_completed_at)
                            if not _authorize_marker_deposit(
                                pickup, active_deposit
                            ):
                                raise RuntimeError(
                                    "deposito recusado: chegada/cor/estado "
                                    "nao autorizados")
                            print(
                                f"[deposito] triangulo "
                                f"{active_deposit.target_kind} alcancado; "
                                "executando o movimento de garra preservado")
                            marker_detector = None
                            deposit_controller = None
                            latest_marker_detection = None
                            latest_marker_captured_at = None
                            last_idle_control = action_completed_at
                    elif search is not None:
                        action_completed_at = time.monotonic()
                        if search.consume_tracking_reset():
                            detector_worker.reset_tracking()
                            fresh_gate.reset()
                            latest_result = None
                            latest_detection = None
                        if (
                            command.state
                            == BallSearchController.START
                        ):
                            search.mark_rotation_started(
                                action_completed_at)
                            search_connection_epoch = (
                                arduino.connection_epoch)
                        elif (
                            command.state
                            == BallSearchController.TARGET_STOP
                        ):
                            search.mark_target_stopped(
                                action_completed_at)
                            fresh_gate.reset()
                            latest_result = None
                            latest_detection = None
                            last_idle_control = action_completed_at
                        elif (
                            command.state
                            == BallSearchController.TURN_STOP
                        ):
                            search.mark_full_turn_stopped(
                                action_completed_at)
                            fresh_gate.reset()
                            latest_result = None
                            latest_detection = None
                            last_idle_control = action_completed_at
            if arduino is not None:
                arduino.refresh(fail_closed=True)
                if (
                    motion_connection_epoch is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch
                        != motion_connection_epoch
                    )
                ):
                    raise RuntimeError(
                        "serial mudou depois do comando visual; "
                        "alvo invalidado e motores parados")
                if (
                    search is not None
                    and search_connection_epoch is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch
                        != search_connection_epoch
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante o giro de busca; "
                        "360 invalidado e motores parados")

            if (
                args.drive
                and command.state == BallApproachController.NEAR
                and not pickup.started
            ):
                if (
                    motor_result is not True
                    or motion_connection_epoch is None
                    or arduino is None
                    or not arduino.connected
                    or arduino.connection_epoch
                    != motion_connection_epoch
                ):
                    raise RuntimeError(
                        "coleta recusada: PARAR da aproximacao nao teve "
                        "escrita serial estavel")
                if command.target_kind not in ("silver", "black"):
                    raise RuntimeError(
                        "coleta recusada: cor da esfera nao foi confirmada")
                pickup.start(command.target_kind)
                pickup_connection_epoch = (
                    arduino.connection_epoch
                    if arduino is not None else None
                )
                print(
                    f"[coleta] esfera {command.target_kind} no ponto "
                    "inferior: avancando 1,5 s, prendendo e elevando; "
                    "a liberacao so ocorrera no triangulo correto")

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
                else (
                    latest_result.locked_detection
                    if (
                        latest_result is not None
                        and latest_result.locked_detection is not None
                        and time.monotonic()
                        - latest_result.locked_detection.timestamp
                        <= cfg.BALL_FRAME_STALE_S
                    )
                    else None
                )
            )
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
            crescent_metrics = (
                last_metrics_result.crescent_evidence
                if last_metrics_result is not None else None)
            crescent_marker = (
                "F" if (
                    crescent_metrics is not None
                    and crescent_metrics.foil_fallback
                ) else (
                    "*" if (
                        crescent_metrics is not None
                        and crescent_metrics.accepted
                    ) else ""
                )
            )
            crescent_text = (
                (
                    " lua"
                    f"{crescent_metrics.support * 100:.0f}%"
                    f"/{crescent_metrics.contrast:.0f}"
                    f"{crescent_marker}"
                )
                if crescent_metrics is not None
                else ""
            )
            radii_text = (
                " r" + "/".join(
                    f"{radius:.0f}" for radius in candidate_radii)
                if candidate_radii else "")
            performance_text = (
                f"cam {_sequence_rate(capture_samples):.1f} | "
                f"vis {detector_fps:.1f} | "
                f"{processing_ms:.0f}ms | "
                f"{vision_mode}{candidate_count}/{hough_proposals}:"
                f"{diagnostic}{radii_text}{crescent_text} | "
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
                    None,
                    motors_enabled=args.drive,
                    performance_text=performance_text,
                    pickup_in_range=command.pickup_in_range,
                    pickup_confirmations=command.pickup_confirmations,
                    crescent_evidence=(
                        latest_result.crescent_evidence
                        if latest_result is not None else None
                    ),
                )
                annotated = _annotate_arena_guard(
                    annotated, latest_result)
                if deposit_controller is not None:
                    marker_overlay = (
                        latest_marker_detection
                        if (
                            latest_marker_captured_at is not None
                            and time.monotonic()
                            - latest_marker_captured_at
                            <= cfg.BALL_FRAME_STALE_S
                        )
                        else None
                    )
                    annotated = _annotate_marker(
                        annotated,
                        marker_overlay,
                        deposit_controller.target_kind,
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
                approach_handoff = (
                    args.drive
                    and command.state == BallApproachController.NEAR
                    and pickup.started
                )
                if args.drive and not approach_handoff:
                    print(
                        f"[resgate] estado terminal {command.state}; "
                        "motores parados")
                if (
                    not approach_handoff
                    and (not args.debug or args.drive)
                ):
                    break

            time.sleep(MAIN_TICK_S)

    except RuntimeError as err:
        print(f"[resgate] ERRO: {err}")
    except KeyboardInterrupt:
        print("\n[resgate] Ctrl-C")
    finally:
        # PARAR vem antes de aguardar worker/camera, inclusive em excecoes.
        if arduino is not None:
            from controle.direcao import steer
            _best_effort("parar os motores", steer)
            _best_effort("cortar o Futaba", arduino.parar_futaba)
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
