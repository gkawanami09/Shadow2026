"""Verifica uma parede perto da vitima sem aproximar nem fechar a garra."""

from dataclasses import dataclass
import statistics
import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


LIVRE = "livre"
PAREDE_RETA = "parede_reta"
INCONCLUSIVO = "inconclusivo"
_SEM_RESPOSTA = object()


@dataclass(frozen=True)
class WallProbeStep:
    """Resultado de um tick; uma acao fisica aparece uma unica vez."""

    state: str
    detail: str
    motor_action: str = ""
    pwm: int = 0
    terminal: bool = False
    result: object = None
    target_kind: object = None

    def motion_command(self):
        return MotionCommand(
            self.state,
            detail=self.detail,
            terminal=self.terminal,
            target_kind=self.target_kind,
        )


@dataclass(frozen=True)
class WallTargetSignature:
    """Geometria curta que impede outra esfera da mesma cor de herdar o teste."""

    kind: str
    center_y: float
    radius: float

    @classmethod
    def from_detection(cls, detection):
        return cls(
            kind=detection.kind,
            center_y=float(detection.center_y),
            radius=max(float(detection.radius), 1.0),
        )

    def matches(self, detection, frame_shape):
        if (
            detection is None
            or not detection.confirmed
            or detection.kind != self.kind
        ):
            return False
        height = frame_shape[0]
        radius_ratio = float(detection.radius) / self.radius
        return (
            cfg.BALL_WALL_PROBE_RADIUS_RATIO_MIN
            <= radius_ratio
            <= cfg.BALL_WALL_PROBE_RADIUS_RATIO_MAX
            and abs(float(detection.center_y) - self.center_y)
            <= height * cfg.BALL_WALL_PROBE_CENTER_Y_TOLERANCE_RATIO
        )


@dataclass(frozen=True)
class WallPickupAuthorization:
    """Decisao curta que so pode ser usada pela mesma vitima reaproximada."""

    target_kind: str
    wall_mode: bool
    expires_at: float
    signature: WallTargetSignature

    def matches(self, detection, frame_shape, now):
        return (
            float(now) <= self.expires_at
            and detection is not None
            and detection.kind == self.target_kind
            and self.signature.matches(detection, frame_shape)
        )


class WallProbeController:
    """Compara o eco dos dois lados depois de tirar a mesma esfera do eixo."""

    CENTER_MEASURE = "WALL_CENTER_MEASURE"
    LEFT_PENDING = "WALL_LEFT_PENDING"
    LEFT_WAIT = "WALL_LEFT"
    LEFT_BRAKE_PENDING = "WALL_LEFT_BRAKE_PENDING"
    LEFT_SETTLE = "WALL_LEFT_SETTLE"
    LEFT_VERIFY = "WALL_LEFT_VERIFY"
    LEFT_MEASURE = "WALL_LEFT_MEASURE"
    CROSS_RIGHT_PENDING = "WALL_CROSS_RIGHT_PENDING"
    CROSS_RIGHT_WAIT = "WALL_CROSS_RIGHT"
    RIGHT_BRAKE_PENDING = "WALL_RIGHT_BRAKE_PENDING"
    RIGHT_SETTLE = "WALL_RIGHT_SETTLE"
    RIGHT_VERIFY = "WALL_RIGHT_VERIFY"
    RIGHT_MEASURE = "WALL_RIGHT_MEASURE"
    RETURN_PENDING = "WALL_RETURN_PENDING"
    RETURN_WAIT = "WALL_RETURN"
    RETURN_BRAKE_PENDING = "WALL_RETURN_BRAKE_PENDING"
    RETURN_SETTLE = "WALL_RETURN_SETTLE"
    CENTER_VERIFY = "WALL_CENTER_VERIFY"
    COMPLETE = "WALL_COMPLETE"
    FAULT = "WALL_FAULT"

    def __init__(self, target_kind, target_detection, start_time=None):
        if target_kind not in ("silver", "black"):
            raise ValueError("teste de parede exige uma cor de vitima valida")
        if (
            target_detection is None
            or not target_detection.confirmed
            or target_detection.kind != target_kind
        ):
            raise ValueError(
                "teste de parede exige a deteccao confirmada da mesma vitima")

        now = time.monotonic() if start_time is None else float(start_time)
        self.target_kind = target_kind
        self.state = self.CENTER_MEASURE
        self._deadline = now + cfg.BALL_WALL_PROBE_MEASURE_TIMEOUT_S
        self._next_ultrasound = 0.0
        self._ultrasound_prepared = False
        self._samples = []
        self._offset_samples = []
        self.target_signature = WallTargetSignature.from_detection(
            target_detection)
        self._fresh_frame_after = None
        self._last_visual_timestamp = None
        self._visual_hits = 0
        self._return_duration = 0.0
        self._pending_result = None
        self._pending_detail = ""
        self._terminal_result = None
        self._terminal_detail = ""

    @property
    def terminal(self):
        return self.state in (self.COMPLETE, self.FAULT)

    def update(self, arduino, detection=None, frame_shape=(480, 640, 3), now=None):
        now = time.monotonic() if now is None else float(now)

        if self.terminal:
            return self._terminal_step()

        if self.state == self.CENTER_MEASURE:
            if not self._ultrasound_prepared:
                arduino.cancelar_ultrassom()
                self._ultrasound_prepared = True
            if not self._collect_ultrasound(arduino, now):
                return self._waiting("medindo eco central antes de mover")
            center_close = self._close_median(self._samples)
            if center_close is None:
                if self._enough_far_or_no_echo(self._samples):
                    return self._complete(
                        LIVRE,
                        "sem eco proximo no eixo; coleta normal",
                    )
                return self._complete(
                    INCONCLUSIVO,
                    "leituras centrais ausentes ou misturadas; "
                    "garra bloqueada",
                    fault=True,
                )
            self.state = self.LEFT_PENDING
            return WallProbeStep(
                self.state,
                f"eco central a {center_close} mm; testando lado esquerdo",
                motor_action="left",
                pwm=cfg.BALL_WALL_PROBE_LATERAL_PWM,
                target_kind=self.target_kind,
            )

        if self.state == self.LEFT_PENDING:
            return self._waiting("aguardando comando lateral esquerdo")
        if self.state == self.LEFT_WAIT:
            if now < self._deadline:
                return self._waiting("deslocamento curto para a esquerda")
            self.state = self.LEFT_BRAKE_PENDING
            return self._stop("lado esquerdo alcancado; freando")
        if self.state == self.LEFT_BRAKE_PENDING:
            return self._waiting("aguardando freio do lado esquerdo")
        if self.state == self.LEFT_SETTLE:
            if now < self._deadline:
                return self._waiting("esperando vibracao do lado esquerdo")
            self._begin_visual_verification(self.LEFT_VERIFY, now)
            return self._waiting("confirmando a mesma bolinha a direita do eixo")
        if self.state == self.LEFT_VERIFY:
            if self._accept_visual(detection, frame_shape, now, side="right"):
                self._begin_measurement(self.LEFT_MEASURE, now, arduino)
                return self._waiting("bolinha fora do feixe; medindo a esquerda")
            if now >= self._deadline:
                return self._begin_return(
                    INCONCLUSIVO,
                    "a mesma bolinha nao saiu para a direita; garra bloqueada",
                    action="right",
                )
            return self._waiting(
                "exigindo dois frames novos da bolinha a direita")
        if self.state == self.LEFT_MEASURE:
            if not self._collect_ultrasound(arduino, now):
                return self._waiting("medindo parede no lado esquerdo")
            self._offset_samples.append(tuple(self._samples))
            self.state = self.CROSS_RIGHT_PENDING
            return WallProbeStep(
                self.state,
                "primeira distancia pronta; cruzando para o lado direito",
                motor_action="right",
                pwm=cfg.BALL_WALL_PROBE_LATERAL_PWM,
                target_kind=self.target_kind,
            )

        if self.state == self.CROSS_RIGHT_PENDING:
            return self._waiting("aguardando travessia para a direita")
        if self.state == self.CROSS_RIGHT_WAIT:
            if now < self._deadline:
                return self._waiting("cruzando o ponto original para a direita")
            self.state = self.RIGHT_BRAKE_PENDING
            return self._stop("lado direito alcancado; freando")
        if self.state == self.RIGHT_BRAKE_PENDING:
            return self._waiting("aguardando freio do lado direito")
        if self.state == self.RIGHT_SETTLE:
            if now < self._deadline:
                return self._waiting("esperando vibracao do lado direito")
            self._begin_visual_verification(self.RIGHT_VERIFY, now)
            return self._waiting("confirmando a mesma bolinha a esquerda do eixo")
        if self.state == self.RIGHT_VERIFY:
            if self._accept_visual(detection, frame_shape, now, side="left"):
                self._begin_measurement(self.RIGHT_MEASURE, now, arduino)
                return self._waiting("bolinha fora do feixe; medindo a direita")
            if now >= self._deadline:
                return self._begin_return(
                    INCONCLUSIVO,
                    "a mesma bolinha nao saiu para a esquerda; garra bloqueada",
                    action="left",
                )
            return self._waiting(
                "exigindo dois frames novos da bolinha a esquerda")
        if self.state == self.RIGHT_MEASURE:
            if not self._collect_ultrasound(arduino, now):
                return self._waiting("medindo parede no lado direito")
            self._offset_samples.append(tuple(self._samples))
            result, detail = self._classify_offsets()
            return self._begin_return(result, detail, action="left")

        if self.state == self.RETURN_PENDING:
            return self._waiting("aguardando retorno ao ponto original")
        if self.state == self.RETURN_WAIT:
            if now < self._deadline:
                return self._waiting("voltando lateralmente ao ponto original")
            self.state = self.RETURN_BRAKE_PENDING
            return self._stop("ponto original alcancado; freando")
        if self.state == self.RETURN_BRAKE_PENDING:
            return self._waiting("aguardando freio do retorno")
        if self.state == self.RETURN_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando no ponto original")
            self._begin_visual_verification(self.CENTER_VERIFY, now)
            return self._waiting("reconfirmando a mesma bolinha no centro")
        if self.state == self.CENTER_VERIFY:
            if self._accept_visual(detection, frame_shape, now, side="center"):
                return self._complete(
                    self._pending_result,
                    self._pending_detail,
                    fault=self._pending_result == INCONCLUSIVO,
                )
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "retorno nao reconfirmou a mesma bolinha no centro; "
                    "garra bloqueada",
                    fault=True,
                )
            return self._waiting(
                "exigindo dois frames novos da bolinha centralizada")

        return self._complete(
            INCONCLUSIVO,
            f"estado desconhecido do teste de parede: {self.state}",
            fault=True,
        )

    def notify_command_written(self, state, now=None):
        now = time.monotonic() if now is None else float(now)
        if state == self.LEFT_PENDING and self.state == state:
            self.state = self.LEFT_WAIT
            self._deadline = now + cfg.BALL_WALL_PROBE_LATERAL_S
            return
        if state == self.LEFT_BRAKE_PENDING and self.state == state:
            self.state = self.LEFT_SETTLE
            self._deadline = now + cfg.BALL_WALL_PROBE_SETTLE_S
            return
        if state == self.CROSS_RIGHT_PENDING and self.state == state:
            self.state = self.CROSS_RIGHT_WAIT
            self._deadline = now + 2.0 * cfg.BALL_WALL_PROBE_LATERAL_S
            return
        if state == self.RIGHT_BRAKE_PENDING and self.state == state:
            self.state = self.RIGHT_SETTLE
            self._deadline = now + cfg.BALL_WALL_PROBE_SETTLE_S
            return
        if state == self.RETURN_PENDING and self.state == state:
            self.state = self.RETURN_WAIT
            self._deadline = now + self._return_duration
            return
        if state == self.RETURN_BRAKE_PENDING and self.state == state:
            self.state = self.RETURN_SETTLE
            self._deadline = now + cfg.BALL_WALL_PROBE_SETTLE_S
            return
        raise RuntimeError("confirmacao fora de uma acao do teste de parede")

    def fail(self, detail):
        return self._complete(INCONCLUSIVO, str(detail), fault=True)

    def _collect_ultrasound(self, arduino, now):
        completed, distance = arduino.poll_ultrassom()
        if completed:
            respondeu = bool(getattr(
                arduino,
                "ultima_leitura_ultrassom_respondeu",
                distance is not None,
            ))
            if not respondeu:
                self._samples.append(_SEM_RESPOSTA)
            elif distance is None:
                self._samples.append(None)
            else:
                distance = int(round(distance))
                self._samples.append(
                    distance if 1 <= distance <= 4000 else _SEM_RESPOSTA)
        if len(self._samples) >= cfg.BALL_WALL_PROBE_SAMPLES:
            arduino.cancelar_ultrassom()
            return True
        if now >= self._deadline:
            arduino.cancelar_ultrassom()
            while len(self._samples) < cfg.BALL_WALL_PROBE_SAMPLES:
                self._samples.append(_SEM_RESPOSTA)
            return True
        if now >= self._next_ultrasound:
            if arduino.iniciar_ultrassom(
                timeout=cfg.BALL_WALL_PROBE_READ_TIMEOUT_S
            ):
                self._next_ultrasound = (
                    now + cfg.BALL_WALL_PROBE_SAMPLE_INTERVAL_S)
        return False

    def _begin_measurement(self, state, now, arduino):
        arduino.cancelar_ultrassom()
        self.state = state
        self._samples = []
        self._next_ultrasound = 0.0
        self._deadline = now + cfg.BALL_WALL_PROBE_MEASURE_TIMEOUT_S

    def _begin_visual_verification(self, state, now):
        self.state = state
        self._fresh_frame_after = now
        self._last_visual_timestamp = None
        self._visual_hits = 0
        self._deadline = now + cfg.BALL_WALL_PROBE_FRAME_TIMEOUT_S

    def _accept_visual(self, detection, frame_shape, now, side):
        if (
            detection is None
            or not detection.confirmed
            or detection.kind != self.target_kind
            or self._fresh_frame_after is None
            or detection.timestamp <= self._fresh_frame_after
            or now - detection.timestamp > cfg.BALL_FRAME_STALE_S
        ):
            return False
        if (
            self._last_visual_timestamp is not None
            and detection.timestamp <= self._last_visual_timestamp + 1e-9
        ):
            return False
        self._last_visual_timestamp = float(detection.timestamp)

        width = frame_shape[1]
        associated = self.target_signature.matches(detection, frame_shape)
        error = detection.horizontal_error(width)
        if side == "right":
            placed = error >= cfg.BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR
        elif side == "left":
            placed = error <= -cfg.BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR
        elif side == "center":
            placed = abs(error) <= cfg.BALL_WALL_PROBE_RETURN_MAX_CENTER_ERROR
        else:
            raise ValueError(f"lado visual desconhecido: {side}")

        if associated and placed:
            self._visual_hits += 1
        else:
            self._visual_hits = 0
        return self._visual_hits >= cfg.BALL_WALL_PROBE_VISUAL_CONFIRM_FRAMES

    def _close_median(self, samples):
        close = [
            value for value in samples
            if isinstance(value, (int, float))
            and value <= cfg.BALL_WALL_PROBE_DISTANCE_MM
        ]
        if len(close) < cfg.BALL_WALL_PROBE_MIN_CLOSE_SAMPLES:
            return None
        return int(round(statistics.median(close)))

    def _enough_far_or_no_echo(self, samples):
        clear = [
            value for value in samples
            if value is None
            or (
                isinstance(value, (int, float))
                and value > cfg.BALL_WALL_PROBE_DISTANCE_MM
            )
        ]
        return len(clear) >= cfg.BALL_WALL_PROBE_MIN_CLOSE_SAMPLES

    def _classify_offsets(self):
        left, right = self._offset_samples
        left_close = self._close_median(left)
        right_close = self._close_median(right)
        if left_close is not None and right_close is not None:
            difference = abs(left_close - right_close)
            if difference <= cfg.BALL_WALL_PROBE_SIMILARITY_MM:
                return (
                    PAREDE_RETA,
                    "parede reta provavel nos dois lados "
                    f"({left_close}/{right_close} mm)",
                )
            return (
                INCONCLUSIVO,
                "ecos dos dois lados discordam; possivel quina "
                f"({left_close}/{right_close} mm)",
            )
        if (
            self._enough_far_or_no_echo(left)
            and self._enough_far_or_no_echo(right)
        ):
            return (
                LIVRE,
                "eco desapareceu nos dois lados; coleta normal",
            )
        return (
            INCONCLUSIVO,
            "um lado difere do outro; parede reta nao confirmada",
        )

    def _begin_return(self, result, detail, action):
        self._pending_result = result
        self._pending_detail = detail
        self._return_duration = cfg.BALL_WALL_PROBE_LATERAL_S
        self.state = self.RETURN_PENDING
        return WallProbeStep(
            self.state,
            "decisao guardada; retornando ao ponto original",
            motor_action=action,
            pwm=cfg.BALL_WALL_PROBE_LATERAL_PWM,
            target_kind=self.target_kind,
        )

    def _stop(self, detail):
        return WallProbeStep(
            self.state,
            detail,
            motor_action="stop",
            target_kind=self.target_kind,
        )

    def _complete(self, result, detail, fault=False):
        self.state = self.FAULT if fault else self.COMPLETE
        self._terminal_result = result
        self._terminal_detail = detail
        return self._terminal_step()

    def _terminal_step(self):
        return WallProbeStep(
            self.state,
            self._terminal_detail,
            motor_action="stop" if self.state == self.FAULT else "",
            terminal=True,
            result=self._terminal_result,
            target_kind=self.target_kind,
        )

    def _waiting(self, detail):
        return WallProbeStep(
            self.state,
            detail,
            target_kind=self.target_kind,
        )


def aplicar_acao_parede(passo, arduino, epoca_serial_esperada=None):
    """Aplica apenas a acao unica emitida pelo verificador fail-closed."""
    if epoca_serial_esperada is not None and (
        not arduino.connected
        or arduino.connection_epoch != epoca_serial_esperada
    ):
        return "serial mudou durante o teste de parede"
    try:
        pwm = int(passo.pwm)
        if passo.motor_action == "left":
            sent = arduino.rodas(-pwm, pwm, pwm, -pwm)
        elif passo.motor_action == "right":
            sent = arduino.rodas(pwm, -pwm, -pwm, pwm)
        elif passo.motor_action == "stop":
            sent = arduino.parar()
        elif passo.motor_action == "":
            return None
        else:
            return f"acao desconhecida do teste: {passo.motor_action}"
        if sent is False:
            return "comando do teste de parede nao foi enviado"
    except Exception as err:                         # noqa: BLE001
        try:
            arduino.parar()
        except Exception:
            pass
        return f"falha no teste de parede: {err}"
    if epoca_serial_esperada is not None and (
        not arduino.connected
        or arduino.connection_epoch != epoca_serial_esperada
    ):
        try:
            arduino.parar()
        except Exception:
            pass
        return "serial mudou durante o teste de parede"
    return None
