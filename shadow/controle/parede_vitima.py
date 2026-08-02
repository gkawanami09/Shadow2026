"""Verifica uma parede perto da vitima sem aproximar nem fechar a garra."""

from dataclasses import dataclass
import statistics
import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


LIVRE = "livre"
PAREDE_RETA = "parede_reta"
PAREDE_DESALINHADA = "parede_desalinhada"
VARREDURA_NECESSARIA = "varredura_necessaria"
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
    bottom_y: float

    @classmethod
    def from_detection(cls, detection):
        return cls(
            kind=detection.kind,
            center_y=float(detection.center_y),
            radius=max(float(detection.radius), 1.0),
            bottom_y=float(detection.bottom_y),
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

    def matches_pickup_depth(self, detection, frame_shape):
        """Confirma que a manobra devolveu a esfera ao NEAR de origem."""
        if not self.matches(detection, frame_shape):
            return False
        height = frame_shape[0]
        radius_ratio = float(detection.radius) / self.radius
        return (
            cfg.BALL_WALL_FINAL_RADIUS_RATIO_MIN
            <= radius_ratio
            <= cfg.BALL_WALL_FINAL_RADIUS_RATIO_MAX
            and abs(float(detection.bottom_y) - self.bottom_y)
            <= height * cfg.BALL_WALL_FINAL_BOTTOM_Y_TOLERANCE_RATIO
        )

    def pickup_depth_status(self, detection, frame_shape):
        """Retorna direcao longitudinal e distancia ate o NEAR de origem.

        ``forward`` significa que a esfera ficou menor/mais alta na imagem;
        ``reverse`` significa que ficou maior/mais baixa. Se raio e base
        pedirem sentidos opostos, a leitura e ambigua e nenhum motor e ligado.
        """
        if not self.matches(detection, frame_shape):
            return "ambiguous", float("inf")

        height = max(float(frame_shape[0]), 1.0)
        radius_ratio = float(detection.radius) / self.radius
        bottom_delta = (float(detection.bottom_y) - self.bottom_y) / height
        tolerance = cfg.BALL_WALL_FINAL_BOTTOM_Y_TOLERANCE_RATIO

        directions = set()
        if radius_ratio < cfg.BALL_WALL_FINAL_RADIUS_RATIO_MIN:
            directions.add("forward")
        elif radius_ratio > cfg.BALL_WALL_FINAL_RADIUS_RATIO_MAX:
            directions.add("reverse")
        if bottom_delta < -tolerance:
            directions.add("forward")
        elif bottom_delta > tolerance:
            directions.add("reverse")

        distance = max(
            cfg.BALL_WALL_FINAL_RADIUS_RATIO_MIN - radius_ratio,
            radius_ratio - cfg.BALL_WALL_FINAL_RADIUS_RATIO_MAX,
            abs(bottom_delta) - tolerance,
            0.0,
        )
        if not directions:
            return "ok", distance
        if len(directions) != 1:
            return "ambiguous", distance
        return directions.pop(), distance


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
    """Mede, alinha o chassi quando preciso e repete o teste da parede."""

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
    PIVOT_LEFT_PENDING = "WALL_PIVOT_LEFT_PENDING"
    PIVOT_RIGHT_PENDING = "WALL_PIVOT_RIGHT_PENDING"
    PIVOT_WAIT = "WALL_PIVOT_WAIT"
    PIVOT_BRAKE_PENDING = "WALL_PIVOT_BRAKE_PENDING"
    PIVOT_SETTLE = "WALL_PIVOT_SETTLE"
    RECENTER_VERIFY = "WALL_RECENTER_VERIFY"
    RECENTER_LEFT_PENDING = "WALL_RECENTER_LEFT_PENDING"
    RECENTER_RIGHT_PENDING = "WALL_RECENTER_RIGHT_PENDING"
    RECENTER_WAIT = "WALL_RECENTER_WAIT"
    RECENTER_BRAKE_PENDING = "WALL_RECENTER_BRAKE_PENDING"
    RECENTER_SETTLE = "WALL_RECENTER_SETTLE"
    SCAN_PIVOT_LEFT_PENDING = "WALL_SCAN_PIVOT_LEFT_PENDING"
    SCAN_PIVOT_RIGHT_PENDING = "WALL_SCAN_PIVOT_RIGHT_PENDING"
    SCAN_PIVOT_WAIT = "WALL_SCAN_PIVOT_WAIT"
    SCAN_BRAKE_PENDING = "WALL_SCAN_BRAKE_PENDING"
    SCAN_SETTLE = "WALL_SCAN_SETTLE"
    SCAN_VERIFY = "WALL_SCAN_VERIFY"
    SCAN_MEASURE = "WALL_SCAN_MEASURE"
    SCAN_CONFIRM_MEASURE = "WALL_SCAN_CONFIRM_MEASURE"
    RESTORE_PIVOT_LEFT_PENDING = "WALL_RESTORE_PIVOT_LEFT_PENDING"
    RESTORE_PIVOT_RIGHT_PENDING = "WALL_RESTORE_PIVOT_RIGHT_PENDING"
    RESTORE_WAIT = "WALL_RESTORE_WAIT"
    RESTORE_BRAKE_PENDING = "WALL_RESTORE_BRAKE_PENDING"
    RESTORE_SETTLE = "WALL_RESTORE_SETTLE"
    RESTORE_VERIFY = "WALL_RESTORE_VERIFY"
    SCAN_CROSS_LEFT_PENDING = "WALL_SCAN_CROSS_LEFT_PENDING"
    SCAN_CROSS_LEFT_WAIT = "WALL_SCAN_CROSS_LEFT_WAIT"
    SCAN_CROSS_LEFT_BRAKE_PENDING = "WALL_SCAN_CROSS_LEFT_BRAKE_PENDING"
    SCAN_CROSS_LEFT_SETTLE = "WALL_SCAN_CROSS_LEFT_SETTLE"
    SCAN_LEFT_VERIFY = "WALL_SCAN_LEFT_VERIFY"
    SCAN_ECHO_RETURN_PENDING = "WALL_SCAN_ECHO_RETURN_PENDING"
    SCAN_ECHO_RETURN_WAIT = "WALL_SCAN_ECHO_RETURN_WAIT"
    SCAN_ECHO_RETURN_BRAKE_PENDING = (
        "WALL_SCAN_ECHO_RETURN_BRAKE_PENDING")
    SCAN_ECHO_RETURN_SETTLE = "WALL_SCAN_ECHO_RETURN_SETTLE"
    DEPTH_FORWARD_PENDING = "WALL_DEPTH_FORWARD_PENDING"
    DEPTH_REVERSE_PENDING = "WALL_DEPTH_REVERSE_PENDING"
    DEPTH_WAIT = "WALL_DEPTH_WAIT"
    DEPTH_BRAKE_PENDING = "WALL_DEPTH_BRAKE_PENDING"
    DEPTH_SETTLE = "WALL_DEPTH_SETTLE"
    DEPTH_VERIFY = "WALL_DEPTH_VERIFY"
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
        # Mantido publico para o log/teste conseguir provar que o eco central
        # nao foi descartado antes dos deslocamentos laterais.
        self.center_samples = []
        self._center_context = ""
        self._offset_samples = []
        self.target_signature = WallTargetSignature.from_detection(
            target_detection)
        self._fresh_frame_after = None
        self._last_visual_timestamp = None
        self._visual_hits = 0
        self._return_duration = 0.0
        self._pending_result = None
        self._pending_detail = ""
        self._pending_direction = None
        self._terminal_result = None
        self._terminal_detail = ""
        self._correction_started = False
        self._correction_count = 0
        self._omni_pulses = 0
        self._recenter_error_before_pulse = None
        self._recenter_no_progress = 0
        self._left_offset_baseline_error = None
        self._right_offset_baseline_error = None
        self._scan_side = None
        self._scan_baseline_error = None
        self._scan_current_error = None
        self._scan_error_before_pulse = None
        self._scan_outward_pulses = 0
        self._scan_restore_pulses = 0
        self._scan_total_pulses = 0
        self._scan_no_progress = 0
        self._scan_restore_no_progress = 0
        self._scan_restore_distance_before_pulse = None
        self._scan_candidate_distance = None
        self._depth_pulses = 0
        self._depth_no_progress = 0
        self._depth_distance_before_pulse = None

    @property
    def terminal(self):
        return self.state in (self.COMPLETE, self.FAULT)

    @property
    def correction_count(self):
        return self._correction_count

    @property
    def omni_pulses(self):
        return self._omni_pulses

    def update(self, arduino, detection=None, frame_shape=(480, 640, 3), now=None):
        now = time.monotonic() if now is None else float(now)

        if self.terminal:
            return self._terminal_step()

        if self.state == self.CENTER_MEASURE:
            return self._update_center_measure(arduino, now)

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
                self._left_offset_baseline_error = detection.horizontal_error(
                    frame_shape[1])
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
                self._right_offset_baseline_error = detection.horizontal_error(
                    frame_shape[1])
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
            result, detail, direction = self._classify_offsets()
            if result == VARREDURA_NECESSARIA:
                return self._begin_scan(
                    "right", self._right_offset_baseline_error)
            return self._begin_return(
                result, detail, action="left", direction=direction)

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
            if self._pending_result == PAREDE_RETA:
                return self._update_final_center_depth(
                    detection, frame_shape, now)
            if self._accept_visual(
                detection, frame_shape, now, side="center"
            ):
                return self._finish_returned_decision()
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "retorno nao reconfirmou a mesma bolinha no centro; "
                    "garra bloqueada",
                    fault=True,
                )
            return self._waiting(
                "exigindo dois frames novos da bolinha centralizada")

        if self.state in (self.DEPTH_FORWARD_PENDING,
                          self.DEPTH_REVERSE_PENDING):
            return self._waiting("aguardando pulso longitudinal curto")
        if self.state == self.DEPTH_WAIT:
            if now < self._deadline:
                return self._waiting(
                    "ajustando a distancia sem alterar o yaw")
            self.state = self.DEPTH_BRAKE_PENDING
            return self._stop("fim do ajuste longitudinal; freando")
        if self.state == self.DEPTH_BRAKE_PENDING:
            return self._waiting("aguardando freio do ajuste longitudinal")
        if self.state == self.DEPTH_SETTLE:
            if now < self._deadline:
                return self._waiting(
                    "assentando antes de medir a profundidade")
            self._begin_visual_verification(self.DEPTH_VERIFY, now)
            return self._waiting(
                "conferindo o NEAR em uma imagem realmente nova")
        if self.state == self.DEPTH_VERIFY:
            return self._update_depth_verify(detection, frame_shape, now)

        if self.state in (self.PIVOT_LEFT_PENDING,
                          self.PIVOT_RIGHT_PENDING):
            return self._waiting("aguardando pulso de pivo traseiro")
        if self.state == self.PIVOT_WAIT:
            if now < self._deadline:
                return self._waiting("corrigindo angulo com a frente parada")
            self.state = self.PIVOT_BRAKE_PENDING
            return self._stop("fim do pivo traseiro; freando")
        if self.state == self.PIVOT_BRAKE_PENDING:
            return self._waiting("aguardando freio depois do pivo")
        if self.state == self.PIVOT_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando depois do pivo traseiro")
            self._begin_recenter_visual(now)
            return self._waiting("recentralizando a mesma bolinha")

        if self.state == self.RECENTER_VERIFY:
            return self._update_recenter(
                detection, frame_shape, now, arduino)
        if self.state in (self.RECENTER_LEFT_PENDING,
                          self.RECENTER_RIGHT_PENDING):
            return self._waiting("aguardando pulso omni de recentralizacao")
        if self.state == self.RECENTER_WAIT:
            if now < self._deadline:
                return self._waiting("recentralizando com as rodas omni")
            self.state = self.RECENTER_BRAKE_PENDING
            return self._stop("fim do pulso omni; freando")
        if self.state == self.RECENTER_BRAKE_PENDING:
            return self._waiting("aguardando freio da recentralizacao")
        if self.state == self.RECENTER_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando depois do pulso omni")
            self._begin_visual_verification(self.RECENTER_VERIFY, now)
            return self._waiting("medindo novo erro da mesma bolinha")

        if self.state in (self.SCAN_PIVOT_LEFT_PENDING,
                          self.SCAN_PIVOT_RIGHT_PENDING):
            return self._waiting("aguardando pulso angular no offset")
        if self.state == self.SCAN_PIVOT_WAIT:
            if now < self._deadline:
                return self._waiting("varrendo parede fora do eixo da esfera")
            self.state = self.SCAN_BRAKE_PENDING
            return self._stop("fim do pulso angular; freando")
        if self.state == self.SCAN_BRAKE_PENDING:
            return self._waiting("aguardando freio da varredura angular")
        if self.state == self.SCAN_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando antes de observar o offset")
            self._begin_visual_verification(self.SCAN_VERIFY, now)
            return self._waiting("confirmando frame novo depois do pivo")
        if self.state == self.SCAN_VERIFY:
            return self._update_scan_verify(
                detection, frame_shape, now, arduino)
        if self.state == self.SCAN_MEASURE:
            return self._update_scan_measure(arduino, now)
        if self.state == self.SCAN_CONFIRM_MEASURE:
            return self._update_scan_confirmation(arduino, now)

        if self.state in (self.RESTORE_PIVOT_LEFT_PENDING,
                          self.RESTORE_PIVOT_RIGHT_PENDING):
            return self._waiting("aguardando pulso de restauracao angular")
        if self.state == self.RESTORE_WAIT:
            if now < self._deadline:
                return self._waiting("restaurando o angulo visual do offset")
            self.state = self.RESTORE_BRAKE_PENDING
            return self._stop("fim do pulso de restauracao; freando")
        if self.state == self.RESTORE_BRAKE_PENDING:
            return self._waiting("aguardando freio da restauracao")
        if self.state == self.RESTORE_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando para conferir o angulo")
            self._begin_visual_verification(self.RESTORE_VERIFY, now)
            return self._waiting("comparando a esfera com o baseline do offset")
        if self.state == self.RESTORE_VERIFY:
            return self._update_restore_verify(
                detection, frame_shape, now)

        if self.state == self.SCAN_CROSS_LEFT_PENDING:
            return self._waiting("aguardando travessia ao offset esquerdo")
        if self.state == self.SCAN_CROSS_LEFT_WAIT:
            if now < self._deadline:
                return self._waiting("cruzando para varrer o offset esquerdo")
            self.state = self.SCAN_CROSS_LEFT_BRAKE_PENDING
            return self._stop("offset esquerdo alcancado; freando")
        if self.state == self.SCAN_CROSS_LEFT_BRAKE_PENDING:
            return self._waiting("aguardando freio no offset esquerdo")
        if self.state == self.SCAN_CROSS_LEFT_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando no offset esquerdo")
            self._begin_visual_verification(self.SCAN_LEFT_VERIFY, now)
            return self._waiting("confirmando a mesma esfera a direita")
        if self.state == self.SCAN_LEFT_VERIFY:
            if self._accept_visual(detection, frame_shape, now, side="right"):
                self._left_offset_baseline_error = detection.horizontal_error(
                    frame_shape[1])
                return self._begin_scan(
                    "left", self._left_offset_baseline_error)
            if now >= self._deadline:
                return self._begin_return(
                    INCONCLUSIVO,
                    "travessia nao confirmou a esfera no offset esquerdo",
                    action="right",
                )
            return self._waiting(
                "exigindo frames novos no offset esquerdo")

        if self.state == self.SCAN_ECHO_RETURN_PENDING:
            return self._waiting("aguardando retorno omni sem desfazer o yaw")
        if self.state == self.SCAN_ECHO_RETURN_WAIT:
            if now < self._deadline:
                return self._waiting("voltando ao centro somente por omni")
            self.state = self.SCAN_ECHO_RETURN_BRAKE_PENDING
            return self._stop("retorno omni concluido; freando")
        if self.state == self.SCAN_ECHO_RETURN_BRAKE_PENDING:
            return self._waiting("aguardando freio do retorno omni")
        if self.state == self.SCAN_ECHO_RETURN_SETTLE:
            if now < self._deadline:
                return self._waiting("assentando sem alterar o yaw encontrado")
            self._begin_recenter_visual(now)
            return self._waiting(
                "recentralizando a esfera sem desfazer o angulo encontrado")

        return self._complete(
            INCONCLUSIVO,
            f"estado desconhecido do teste de parede: {self.state}",
            fault=True,
        )

    def notify_command_written(self, state, now=None):
        now = time.monotonic() if now is None else float(now)
        transitions = {
            self.LEFT_PENDING: (
                self.LEFT_WAIT, cfg.BALL_WALL_PROBE_LATERAL_S),
            self.LEFT_BRAKE_PENDING: (
                self.LEFT_SETTLE, cfg.BALL_WALL_PROBE_SETTLE_S),
            self.CROSS_RIGHT_PENDING: (
                self.CROSS_RIGHT_WAIT,
                2.0 * cfg.BALL_WALL_PROBE_LATERAL_S),
            self.RIGHT_BRAKE_PENDING: (
                self.RIGHT_SETTLE, cfg.BALL_WALL_PROBE_SETTLE_S),
            self.RETURN_PENDING: (
                self.RETURN_WAIT, self._return_duration),
            self.RETURN_BRAKE_PENDING: (
                self.RETURN_SETTLE, cfg.BALL_WALL_PROBE_SETTLE_S),
            self.PIVOT_LEFT_PENDING: (
                self.PIVOT_WAIT, cfg.BALL_WALL_ALIGN_PIVOT_S),
            self.PIVOT_RIGHT_PENDING: (
                self.PIVOT_WAIT, cfg.BALL_WALL_ALIGN_PIVOT_S),
            self.PIVOT_BRAKE_PENDING: (
                self.PIVOT_SETTLE, cfg.BALL_WALL_ALIGN_SETTLE_S),
            self.RECENTER_LEFT_PENDING: (
                self.RECENTER_WAIT, cfg.BALL_WALL_ALIGN_OMNI_PULSE_S),
            self.RECENTER_RIGHT_PENDING: (
                self.RECENTER_WAIT, cfg.BALL_WALL_ALIGN_OMNI_PULSE_S),
            self.RECENTER_BRAKE_PENDING: (
                self.RECENTER_SETTLE, cfg.BALL_WALL_ALIGN_SETTLE_S),
            self.SCAN_PIVOT_LEFT_PENDING: (
                self.SCAN_PIVOT_WAIT, cfg.BALL_WALL_SCAN_PULSE_S),
            self.SCAN_PIVOT_RIGHT_PENDING: (
                self.SCAN_PIVOT_WAIT, cfg.BALL_WALL_SCAN_PULSE_S),
            self.SCAN_BRAKE_PENDING: (
                self.SCAN_SETTLE, cfg.BALL_WALL_SCAN_SETTLE_S),
            self.RESTORE_PIVOT_LEFT_PENDING: (
                self.RESTORE_WAIT, cfg.BALL_WALL_SCAN_PULSE_S),
            self.RESTORE_PIVOT_RIGHT_PENDING: (
                self.RESTORE_WAIT, cfg.BALL_WALL_SCAN_PULSE_S),
            self.RESTORE_BRAKE_PENDING: (
                self.RESTORE_SETTLE, cfg.BALL_WALL_SCAN_SETTLE_S),
            self.SCAN_CROSS_LEFT_PENDING: (
                self.SCAN_CROSS_LEFT_WAIT,
                2.0 * cfg.BALL_WALL_PROBE_LATERAL_S),
            self.SCAN_CROSS_LEFT_BRAKE_PENDING: (
                self.SCAN_CROSS_LEFT_SETTLE,
                cfg.BALL_WALL_PROBE_SETTLE_S),
            self.SCAN_ECHO_RETURN_PENDING: (
                self.SCAN_ECHO_RETURN_WAIT,
                cfg.BALL_WALL_PROBE_LATERAL_S),
            self.SCAN_ECHO_RETURN_BRAKE_PENDING: (
                self.SCAN_ECHO_RETURN_SETTLE,
                cfg.BALL_WALL_ALIGN_SETTLE_S),
            self.DEPTH_FORWARD_PENDING: (
                self.DEPTH_WAIT, cfg.BALL_WALL_DEPTH_PULSE_S),
            self.DEPTH_REVERSE_PENDING: (
                self.DEPTH_WAIT, cfg.BALL_WALL_DEPTH_PULSE_S),
            self.DEPTH_BRAKE_PENDING: (
                self.DEPTH_SETTLE, cfg.BALL_WALL_DEPTH_SETTLE_S),
        }
        if state == self.state and state in transitions:
            self.state, duration = transitions[state]
            self._deadline = now + duration
            return
        raise RuntimeError("confirmacao fora de uma acao do teste de parede")

    def fail(self, detail):
        return self._complete(INCONCLUSIVO, str(detail), fault=True)

    def _update_center_measure(self, arduino, now):
        if not self._ultrasound_prepared:
            arduino.cancelar_ultrassom()
            self._ultrasound_prepared = True
        if not self._collect_ultrasound(arduino, now):
            return self._waiting("medindo eco central antes de mover")

        samples = tuple(self._samples)
        self.center_samples.append(samples)
        if self._has_serial_timeout(samples):
            return self._complete(
                INCONCLUSIVO,
                "leituras centrais ausentes; garra bloqueada",
                fault=True,
            )

        center_close = self._close_median(samples)
        if center_close is not None:
            self._center_context = "close"
            return self._begin_bilateral_probe(
                f"eco central a {center_close} mm; testando lado esquerdo")

        far = self._stable_far_median(samples)
        if far is not None:
            self._center_context = (
                "far_after_correction"
                if self._correction_started else "far")
        else:
            self._center_context = "no_echo"
        return self._begin_bilateral_probe(
            "eco central nao basta para liberar; testando offsets e angulos")

    def _begin_bilateral_probe(self, detail):
        self._offset_samples = []
        self.state = self.LEFT_PENDING
        return WallProbeStep(
            self.state,
            detail,
            motor_action="left",
            pwm=cfg.BALL_WALL_PROBE_LATERAL_PWM,
            target_kind=self.target_kind,
        )

    def _begin_scan(self, side, baseline_error):
        if side not in ("left", "right") or baseline_error is None:
            return self._complete(
                INCONCLUSIVO,
                "baseline visual do offset indisponivel; garra bloqueada",
                fault=True,
            )
        self._scan_side = side
        self._scan_baseline_error = float(baseline_error)
        self._scan_current_error = float(baseline_error)
        self._scan_error_before_pulse = None
        self._scan_outward_pulses = 0
        self._scan_restore_pulses = 0
        self._scan_no_progress = 0
        self._scan_restore_no_progress = 0
        self._scan_restore_distance_before_pulse = None
        return self._issue_scan_pivot()

    def _issue_scan_pivot(self):
        if (
            self._scan_outward_pulses
            >= cfg.BALL_WALL_SCAN_MAX_OUTWARD_PULSES_PER_SIDE
            or self._scan_total_pulses
            >= cfg.BALL_WALL_SCAN_TOTAL_PULSE_LIMIT
        ):
            return self._complete(
                INCONCLUSIVO,
                "limite de pulsos da varredura angular; garra bloqueada",
                fault=True,
            )
        self._scan_outward_pulses += 1
        self._scan_total_pulses += 1
        self._scan_error_before_pulse = self._scan_current_error
        action = "pivot_right" if self._scan_side == "right" else "pivot_left"
        self.state = (
            self.SCAN_PIVOT_RIGHT_PENDING
            if self._scan_side == "right" else self.SCAN_PIVOT_LEFT_PENDING
        )
        return WallProbeStep(
            self.state,
            "varrendo para longe da esfera no offset " + self._scan_side,
            motor_action=action,
            pwm=cfg.BALL_WALL_SCAN_PIVOT_PWM,
            target_kind=self.target_kind,
        )

    def _update_scan_verify(self, detection, frame_shape, now, arduino):
        error = self._fresh_target_error(detection, frame_shape, now)
        if error is None:
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "sem frame novo da mesma esfera durante a varredura",
                    fault=True,
                )
            return self._waiting("aguardando frame novo da varredura")

        if self._scan_side == "right":
            outside_beam = (
                error <= -cfg.BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR)
            progressed = error <= (
                self._scan_error_before_pulse
                - cfg.BALL_WALL_SCAN_MIN_VISUAL_PROGRESS)
        else:
            outside_beam = (
                error >= cfg.BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR)
            progressed = error >= (
                self._scan_error_before_pulse
                + cfg.BALL_WALL_SCAN_MIN_VISUAL_PROGRESS)
        if not outside_beam:
            return self._complete(
                INCONCLUSIVO,
                "esfera entrou no feixe durante a varredura; garra bloqueada",
                fault=True,
            )
        self._scan_current_error = float(error)
        self._scan_no_progress = 0 if progressed else self._scan_no_progress + 1
        if (
            self._scan_no_progress
            >= cfg.BALL_WALL_SCAN_MAX_NO_PROGRESS_PULSES
        ):
            return self._complete(
                INCONCLUSIVO,
                "pivo angular nao afastou o sensor da esfera; garra bloqueada",
                fault=True,
            )
        if not progressed:
            return self._issue_scan_pivot()

        self._begin_measurement(self.SCAN_MEASURE, now, arduino)
        return self._waiting(
            "angulo confirmado por frame novo; medindo bateria de ecos")

    def _update_scan_measure(self, arduino, now):
        if not self._collect_ultrasound(arduino, now):
            return self._waiting("medindo parede no angulo varrido")
        samples = tuple(self._samples)
        if self._has_serial_timeout(samples):
            return self._complete(
                INCONCLUSIVO,
                "timeout na bateria da varredura angular; garra bloqueada",
                fault=True,
            )
        close = self._close_median(samples)
        if close is not None:
            self._scan_candidate_distance = close
            self._begin_measurement(
                self.SCAN_CONFIRM_MEASURE, now, arduino)
            return self._waiting(
                "primeira bateria com eco; exigindo confirmacao independente")
        if not self._enough_clear(samples):
            return self._complete(
                INCONCLUSIVO,
                "bateria angular ambigua; garra bloqueada",
                fault=True,
            )
        if (
            self._scan_outward_pulses
            < cfg.BALL_WALL_SCAN_MAX_OUTWARD_PULSES_PER_SIDE
        ):
            return self._issue_scan_pivot()
        return self._begin_scan_restore(now)

    def _update_scan_confirmation(self, arduino, now):
        if not self._collect_ultrasound(arduino, now):
            return self._waiting("confirmando eco angular em nova bateria")
        samples = tuple(self._samples)
        if self._has_serial_timeout(samples):
            return self._complete(
                INCONCLUSIVO,
                "timeout ao confirmar eco angular; garra bloqueada",
                fault=True,
            )
        confirmed = self._close_median(samples)
        if (
            confirmed is None
            or abs(confirmed - self._scan_candidate_distance)
            > cfg.BALL_WALL_PROBE_SIMILARITY_MM
        ):
            return self._complete(
                INCONCLUSIVO,
                "eco angular nao repetiu em bateria independente; "
                "garra bloqueada",
                fault=True,
            )
        return self._begin_scan_echo_return(int(round(statistics.median(
            (self._scan_candidate_distance, confirmed)))))

    def _begin_scan_restore(self, now):
        distance = abs(
            self._scan_current_error - self._scan_baseline_error)
        self._scan_restore_pulses = 0
        self._scan_restore_no_progress = 0
        self._scan_restore_distance_before_pulse = distance
        if distance <= cfg.BALL_WALL_SCAN_RESTORE_DEADBAND:
            self._begin_visual_verification(self.RESTORE_VERIFY, now)
            return self._waiting(
                "yaw parece restaurado; exigindo dois frames novos")
        return self._issue_restore_pivot()

    def _issue_restore_pivot(self):
        if (
            self._scan_restore_pulses
            >= cfg.BALL_WALL_SCAN_MAX_RESTORE_PULSES_PER_SIDE
            or self._scan_total_pulses
            >= cfg.BALL_WALL_SCAN_TOTAL_PULSE_LIMIT
        ):
            return self._complete(
                INCONCLUSIVO,
                "nao foi possivel restaurar o yaw do offset; garra bloqueada",
                fault=True,
            )
        # Pivot_right leva a esfera para a esquerda da imagem; pivot_left faz
        # o inverso. A escolha abaixo fecha o erro ate o baseline medido.
        action = (
            "pivot_left"
            if self._scan_current_error < self._scan_baseline_error
            else "pivot_right"
        )
        self._scan_restore_pulses += 1
        self._scan_total_pulses += 1
        self._scan_restore_distance_before_pulse = abs(
            self._scan_current_error - self._scan_baseline_error)
        self.state = (
            self.RESTORE_PIVOT_LEFT_PENDING
            if action == "pivot_left" else self.RESTORE_PIVOT_RIGHT_PENDING
        )
        return WallProbeStep(
            self.state,
            "restaurando yaw pelo erro visual do offset",
            motor_action=action,
            pwm=cfg.BALL_WALL_SCAN_PIVOT_PWM,
            target_kind=self.target_kind,
        )

    def _update_restore_verify(self, detection, frame_shape, now):
        error = self._fresh_target_error(detection, frame_shape, now)
        if error is None:
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "sem frame novo para comprovar restauracao do yaw",
                    fault=True,
                )
            return self._waiting("aguardando frame novo da restauracao")

        distance = abs(float(error) - self._scan_baseline_error)
        self._scan_current_error = float(error)
        if distance <= cfg.BALL_WALL_SCAN_RESTORE_DEADBAND:
            self._visual_hits += 1
            if (
                self._visual_hits
                >= cfg.BALL_WALL_PROBE_VISUAL_CONFIRM_FRAMES
            ):
                return self._finish_scan_restored()
            return self._waiting(
                "confirmando restauracao do yaw em outro frame novo")
        self._visual_hits = 0
        progressed = distance <= (
            self._scan_restore_distance_before_pulse
            - cfg.BALL_WALL_SCAN_MIN_VISUAL_PROGRESS)
        self._scan_restore_no_progress = (
            0 if progressed else self._scan_restore_no_progress + 1)
        if (
            self._scan_restore_no_progress
            >= cfg.BALL_WALL_SCAN_MAX_NO_PROGRESS_PULSES
        ):
            return self._complete(
                INCONCLUSIVO,
                "restauracao do yaw nao reduziu o erro; garra bloqueada",
                fault=True,
            )
        return self._issue_restore_pivot()

    def _finish_scan_restored(self):
        if self._scan_side == "right":
            self.state = self.SCAN_CROSS_LEFT_PENDING
            return WallProbeStep(
                self.state,
                "yaw direito restaurado; cruzando para o offset esquerdo",
                motor_action="left",
                pwm=cfg.BALL_WALL_PROBE_LATERAL_PWM,
                target_kind=self.target_kind,
            )

        can_release = (
            not self._correction_started
            and self._center_context in ("close", "far")
        )
        result = LIVRE if can_release else INCONCLUSIVO
        detail = (
            "duas varreduras angulares vazias e yaw restaurado; coleta normal"
            if can_release else
            "varreduras vazias nao provaram caminho livre; garra bloqueada"
        )
        return self._begin_return(result, detail, action="right")

    def _begin_scan_echo_return(self, distance):
        if self._correction_count >= cfg.BALL_WALL_ALIGN_MAX_CORRECTIONS:
            return self._complete(
                INCONCLUSIVO,
                "eco angular encontrado depois do limite de correcoes",
                fault=True,
            )
        self._correction_started = True
        self._correction_count += 1
        action = "left" if self._scan_side == "right" else "right"
        self.state = self.SCAN_ECHO_RETURN_PENDING
        return WallProbeStep(
            self.state,
            f"eco angular confirmado a {distance} mm; mantendo yaw e "
            "voltando apenas por omni",
            motor_action=action,
            pwm=cfg.BALL_WALL_PROBE_LATERAL_PWM,
            target_kind=self.target_kind,
        )

    def _begin_recenter_visual(self, now):
        self._recenter_error_before_pulse = None
        self._recenter_no_progress = 0
        self._begin_visual_verification(self.RECENTER_VERIFY, now)

    def _finish_returned_decision(self):
        if self._pending_result == PAREDE_DESALINHADA:
            if (
                self._pending_direction not in ("left", "right")
                or self._correction_count
                >= cfg.BALL_WALL_ALIGN_MAX_CORRECTIONS
            ):
                return self._complete(
                    INCONCLUSIVO,
                    "parede continuou desalinhada depois do limite seguro; "
                    "garra bloqueada",
                    fault=True,
                )
            self._correction_started = True
            self._correction_count += 1
            self.state = (
                self.PIVOT_LEFT_PENDING
                if self._pending_direction == "left"
                else self.PIVOT_RIGHT_PENDING
            )
            return WallProbeStep(
                self.state,
                "parede inclinada; corrigindo pelo lado de menor eco",
                motor_action=f"pivot_{self._pending_direction}",
                pwm=cfg.BALL_WALL_ALIGN_PIVOT_PWM,
                target_kind=self.target_kind,
            )

        if self._pending_result == LIVRE and self._correction_started:
            return self._complete(
                INCONCLUSIVO,
                "eco sumiu depois de iniciar a correcao; garra bloqueada",
                fault=True,
            )
        return self._complete(
            self._pending_result,
            self._pending_detail,
            fault=self._pending_result == INCONCLUSIVO,
        )

    def _update_final_center_depth(self, detection, frame_shape, now):
        """Exige centro e profundidade juntos em dois frames novos."""
        error = self._fresh_target_error(detection, frame_shape, now)
        if error is None:
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "retorno nao reconfirmou a mesma bolinha no centro; "
                    "garra bloqueada",
                    fault=True,
                )
            return self._waiting(
                "aguardando frame novo da bolinha no ponto de coleta")

        if abs(error) > cfg.BALL_WALL_ALIGN_CENTER_DEADBAND:
            self._visual_hits = 0
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "retorno nao reconfirmou a mesma bolinha no centro; "
                    "garra bloqueada",
                    fault=True,
                )
            return self._waiting(
                "esfera ainda fora do centro estrito do ponto de coleta")

        status, _distance = self.target_signature.pickup_depth_status(
            detection, frame_shape)
        if status == "ambiguous":
            return self._complete(
                INCONCLUSIVO,
                "raio e base da esfera discordam sobre a profundidade; "
                "garra bloqueada",
                fault=True,
            )
        if status != "ok":
            self._visual_hits = 0
            return self._issue_depth_pulse(detection, frame_shape)

        self._visual_hits += 1
        if (
            self._visual_hits
            >= cfg.BALL_WALL_PROBE_VISUAL_CONFIRM_FRAMES
        ):
            return self._finish_returned_decision()
        return self._waiting(
            "centro e NEAR confirmados; exigindo outro frame novo")

    def _issue_depth_pulse(self, detection, frame_shape):
        status, distance = self.target_signature.pickup_depth_status(
            detection, frame_shape)
        if status == "ambiguous":
            return self._complete(
                INCONCLUSIVO,
                "raio e base da esfera discordam sobre a profundidade; "
                "garra bloqueada",
                fault=True,
            )
        if status == "ok":
            return self._waiting(
                "NEAR recuperado; aguardando segunda imagem nova")
        if self._depth_pulses >= cfg.BALL_WALL_DEPTH_MAX_PULSES:
            return self._complete(
                INCONCLUSIVO,
                "limite de pulsos longitudinais sem recuperar o NEAR; "
                "garra bloqueada",
                fault=True,
            )

        self._depth_pulses += 1
        self._depth_distance_before_pulse = distance
        self.state = (
            self.DEPTH_FORWARD_PENDING
            if status == "forward" else self.DEPTH_REVERSE_PENDING
        )
        return WallProbeStep(
            self.state,
            "esfera longe do NEAR; pulso curto para frente"
            if status == "forward" else
            "esfera perto demais; pulso curto de re",
            motor_action=status,
            pwm=cfg.BALL_WALL_DEPTH_PWM,
            target_kind=self.target_kind,
        )

    def _update_depth_verify(self, detection, frame_shape, now):
        error = self._fresh_target_error(detection, frame_shape, now)
        if error is None:
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "sem frame novo da mesma esfera depois do ajuste; "
                    "garra bloqueada",
                    fault=True,
                )
            return self._waiting(
                "aguardando frame novo depois do ajuste longitudinal")

        if abs(error) > cfg.BALL_WALL_ALIGN_CENTER_DEADBAND:
            return self._complete(
                INCONCLUSIVO,
                "ajuste longitudinal tirou a esfera do centro; "
                "garra bloqueada",
                fault=True,
            )

        status, distance = self.target_signature.pickup_depth_status(
            detection, frame_shape)
        if status == "ambiguous":
            return self._complete(
                INCONCLUSIVO,
                "profundidade visual ficou ambigua depois do pulso; "
                "garra bloqueada",
                fault=True,
            )

        if self._depth_distance_before_pulse is not None:
            improved = (
                status == "ok"
                or distance <= self._depth_distance_before_pulse
                - cfg.BALL_WALL_DEPTH_MIN_PROGRESS
            )
            self._depth_no_progress = (
                0 if improved else self._depth_no_progress + 1)
            self._depth_distance_before_pulse = None
            if (
                self._depth_no_progress
                >= cfg.BALL_WALL_DEPTH_MAX_NO_PROGRESS_PULSES
            ):
                return self._complete(
                    INCONCLUSIVO,
                    "ajuste longitudinal nao aproximou a esfera do NEAR; "
                    "garra bloqueada",
                    fault=True,
                )

        if status == "ok":
            self._visual_hits += 1
            if (
                self._visual_hits
                >= cfg.BALL_WALL_PROBE_VISUAL_CONFIRM_FRAMES
            ):
                return self._finish_returned_decision()
            return self._waiting(
                "NEAR recuperado; confirmando em outro frame novo")

        self._visual_hits = 0
        return self._issue_depth_pulse(detection, frame_shape)

    def _update_recenter(self, detection, frame_shape, now, arduino):
        error = self._fresh_target_error(detection, frame_shape, now)
        if error is None:
            if now >= self._deadline:
                return self._complete(
                    INCONCLUSIVO,
                    "mesma bolinha nao reapareceu depois do pivo; "
                    "garra bloqueada",
                    fault=True,
                )
            return self._waiting("aguardando frame novo da mesma bolinha")

        absolute_error = abs(error)
        if self._recenter_error_before_pulse is not None:
            improved = absolute_error <= (
                self._recenter_error_before_pulse
                - cfg.BALL_WALL_ALIGN_MIN_PROGRESS
            )
            self._recenter_no_progress = (
                0 if improved else self._recenter_no_progress + 1)
            self._recenter_error_before_pulse = None
            if (
                self._recenter_no_progress
                >= cfg.BALL_WALL_ALIGN_MAX_NO_PROGRESS_PULSES
            ):
                return self._complete(
                    INCONCLUSIVO,
                    "recentralizacao nao reduziu o erro visual; "
                    "garra bloqueada",
                    fault=True,
                )

        if absolute_error <= cfg.BALL_WALL_ALIGN_CENTER_DEADBAND:
            self._visual_hits += 1
            if (
                self._visual_hits
                >= cfg.BALL_WALL_PROBE_VISUAL_CONFIRM_FRAMES
            ):
                self._visual_hits = 0
                self._begin_measurement(self.CENTER_MEASURE, now, arduino)
                return self._waiting(
                    "bolinha recentralizada; repetindo o teste completo")
            return self._waiting("confirmando bolinha recentralizada")

        self._visual_hits = 0
        if self._omni_pulses >= cfg.BALL_WALL_ALIGN_MAX_OMNI_PULSES:
            return self._complete(
                INCONCLUSIVO,
                "limite de pulsos omni sem centralizar; garra bloqueada",
                fault=True,
            )

        direction = "right" if error > 0 else "left"
        self._omni_pulses += 1
        self._recenter_error_before_pulse = absolute_error
        self.state = (
            self.RECENTER_RIGHT_PENDING
            if direction == "right" else self.RECENTER_LEFT_PENDING
        )
        return WallProbeStep(
            self.state,
            "bolinha fora do centro; pulso omni no mesmo lado do erro",
            motor_action=direction,
            pwm=cfg.BALL_WALL_ALIGN_OMNI_PWM,
            target_kind=self.target_kind,
        )

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
        self._ultrasound_prepared = True
        self._deadline = now + cfg.BALL_WALL_PROBE_MEASURE_TIMEOUT_S

    def _begin_visual_verification(self, state, now):
        self.state = state
        self._fresh_frame_after = now
        self._last_visual_timestamp = None
        self._visual_hits = 0
        self._deadline = now + cfg.BALL_WALL_PROBE_FRAME_TIMEOUT_S

    def _fresh_target_error(self, detection, frame_shape, now):
        if (
            detection is None
            or self._fresh_frame_after is None
            or detection.timestamp <= self._fresh_frame_after
            or now - detection.timestamp > cfg.BALL_FRAME_STALE_S
        ):
            return None
        if (
            self._last_visual_timestamp is not None
            and detection.timestamp <= self._last_visual_timestamp + 1e-9
        ):
            return None
        self._last_visual_timestamp = float(detection.timestamp)
        if (
            not detection.confirmed
            or detection.kind != self.target_kind
            or detection.truncated
            or not self.target_signature.matches(detection, frame_shape)
        ):
            # Um frame realmente novo que mostra outro alvo interrompe a
            # bateria visual. Frame ausente, repetido ou stale apenas espera.
            self._visual_hits = 0
            return None
        return detection.horizontal_error(frame_shape[1])

    def _accept_visual(self, detection, frame_shape, now, side):
        error = self._fresh_target_error(detection, frame_shape, now)
        if error is None:
            return False
        if side == "right":
            placed = error >= cfg.BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR
        elif side == "left":
            placed = error <= -cfg.BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR
        elif side == "center":
            placed = abs(error) <= cfg.BALL_WALL_PROBE_RETURN_MAX_CENTER_ERROR
        else:
            raise ValueError(f"lado visual desconhecido: {side}")

        self._visual_hits = self._visual_hits + 1 if placed else 0
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

    def _stable_far_median(self, samples):
        if any(
            isinstance(value, (int, float))
            and value <= cfg.BALL_WALL_PROBE_DISTANCE_MM
            for value in samples
        ):
            return None
        far = [
            value for value in samples
            if isinstance(value, (int, float))
            and value > cfg.BALL_WALL_PROBE_DISTANCE_MM
        ]
        if len(far) < cfg.BALL_WALL_PROBE_MIN_CLOSE_SAMPLES:
            return None
        if max(far) - min(far) > cfg.BALL_WALL_PROBE_SIMILARITY_MM:
            return None
        return int(round(statistics.median(far)))

    @staticmethod
    def _has_serial_timeout(samples):
        return any(value is _SEM_RESPOSTA for value in samples)

    def _enough_clear(self, samples):
        if self._has_serial_timeout(samples):
            return False
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
        if self._has_serial_timeout(left) or self._has_serial_timeout(right):
            return (
                INCONCLUSIVO,
                "timeout em uma leitura lateral; garra bloqueada",
                None,
            )
        left_close = self._close_median(left)
        right_close = self._close_median(right)
        left_clear = self._enough_clear(left)
        right_clear = self._enough_clear(right)

        if left_close is not None and right_close is not None:
            difference = abs(left_close - right_close)
            if difference <= cfg.BALL_WALL_PROBE_SIMILARITY_MM:
                return (
                    PAREDE_RETA,
                    "parede reta provavel nos dois lados "
                    f"({left_close}/{right_close} mm)",
                    None,
                )
            direction = "left" if left_close < right_close else "right"
            return (
                PAREDE_DESALINHADA,
                "parede inclinada pelos ecos "
                f"({left_close}/{right_close} mm)",
                direction,
            )

        if left_close is not None and right_clear:
            return (
                PAREDE_DESALINHADA,
                "eco proximo somente no lado esquerdo",
                "left",
            )
        if right_close is not None and left_clear:
            return (
                PAREDE_DESALINHADA,
                "eco proximo somente no lado direito",
                "right",
            )

        if left_clear and right_clear:
            return (
                VARREDURA_NECESSARIA,
                "offsets livres; falta varrer os angulos dos dois lados",
                None,
            )
        return (
            INCONCLUSIVO,
            "leituras laterais ambiguas; parede reta nao confirmada",
            None,
        )

    def _begin_return(self, result, detail, action, direction=None):
        self._pending_result = result
        self._pending_detail = detail
        self._pending_direction = direction
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
            motor_action=(
                "stop"
                if self.state == self.FAULT
                or self._terminal_result == PAREDE_RETA
                else ""
            ),
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
        elif passo.motor_action == "pivot_left":
            sent = arduino.rodas(0, -pwm, 0, pwm)
        elif passo.motor_action == "pivot_right":
            sent = arduino.rodas(0, pwm, 0, -pwm)
        elif passo.motor_action == "forward":
            sent = arduino.rodas(pwm, pwm, pwm, pwm)
        elif passo.motor_action == "reverse":
            sent = arduino.rodas(-pwm, -pwm, -pwm, -pwm)
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
