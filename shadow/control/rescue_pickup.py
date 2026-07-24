"""Sequencia nao bloqueante de coleta e liberacao da esfera."""

from dataclasses import dataclass
import time

import rescue_config as cfg
from control.rescue_approach import MotionCommand


@dataclass(frozen=True)
class PickupStep:
    """Saida de um tick; cada acao fisica aparece uma unica vez."""

    state: str
    detail: str
    angle: int = 190
    speed: float = 0.0
    motor_action: str = ""
    futaba_action: object = None
    stop_futaba: bool = False
    gripper_action: object = None
    terminal: bool = False

    def motion_command(self):
        return MotionCommand(
            self.state,
            angle=self.angle,
            speed=self.speed,
            detail=self.detail,
            terminal=self.terminal,
        )


class BallPickupSequencer:
    """Prende, eleva e libera a esfera conforme a cor confirmada."""

    IDLE = "PICKUP_IDLE"
    FUTABA_START = "PICKUP_FUTABA_START"
    FUTABA_PENDING = "PICKUP_FUTABA_PENDING"
    FUTABA_WAIT = "PICKUP_FUTABA"
    FORWARD_START = "PICKUP_FORWARD_START"
    FORWARD_LEAD = "PICKUP_FORWARD_LEAD"
    GRIPPERS_START = "PICKUP_GRIPPERS_START"
    GRIPPERS_WAIT = "PICKUP_GRIPPERS"
    LIFT_PENDING = "PICKUP_LIFT_PENDING"
    LIFT_WAIT = "PICKUP_LIFT"
    LOWER_PENDING = "PICKUP_LOWER_PENDING"
    LOWER_WAIT = "PICKUP_LOWER"
    RELEASE_PENDING = "PICKUP_RELEASE_PENDING"
    RELEASE_WAIT = "PICKUP_RELEASE"
    WIGGLE_PENDING = "PICKUP_WIGGLE_PENDING"
    WIGGLE_WAIT = "PICKUP_WIGGLE"
    COMPLETE = "PICKUP_COMPLETE"
    FAULT = "PICKUP_FAULT"

    def __init__(self):
        self.state = self.IDLE
        self._deadline = None
        self._kind = None
        self._wiggle_actions = ()
        self._wiggle_index = 0
        self._terminal_detail = ""

    @property
    def started(self):
        return self.state != self.IDLE

    @property
    def terminal(self):
        return self.state in (self.COMPLETE, self.FAULT)

    @property
    def target_kind(self):
        return self._kind

    def start(self, target_kind):
        """Arma a sequencia e congela a cor ate o estado terminal."""
        if self.state != self.IDLE:
            return False
        if target_kind not in ("silver", "black"):
            raise ValueError(
                "a coleta exige cor confirmada silver ou black")
        self._kind = target_kind
        self._wiggle_actions = self._build_wiggle_actions(target_kind)
        self.state = self.FUTABA_START
        return True

    def update(self, now=None):
        now = time.monotonic() if now is None else float(now)

        if self.state == self.IDLE:
            return PickupStep(
                self.IDLE,
                "coleta ainda nao iniciada",
            )

        if self.state == self.FUTABA_START:
            self.state = self.FUTABA_PENDING
            return PickupStep(
                self.FUTABA_PENDING,
                "rodas zeradas; baixando o Futaba",
                motor_action="hold",
                futaba_action=(
                    cfg.BALL_PICKUP_FUTABA_POWER,
                    cfg.BALL_PICKUP_FUTABA_MS,
                ),
            )

        if self.state == self.FUTABA_PENDING:
            return PickupStep(
                self.FUTABA_PENDING,
                "aguardando confirmacao da descida do Futaba",
            )

        if self.state == self.FUTABA_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.FUTABA_WAIT,
                    "aguardando o Futaba terminar a descida",
                )
            self.state = self.FORWARD_START
            self._deadline = None
            return PickupStep(
                self.FORWARD_START,
                "Futaba embaixo; iniciando avanco de 1,5 s",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                motor_action="forward",
                stop_futaba=True,
            )

        if self.state == self.FORWARD_START:
            return PickupStep(
                self.FORWARD_START,
                "aguardando confirmacao do comando de avanco",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
            )

        if self.state == self.FORWARD_LEAD:
            if now < self._deadline:
                return PickupStep(
                    self.FORWARD_LEAD,
                    "avancando por 1,5 s com as garras abertas",
                    angle=0,
                    speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                )
            self.state = self.GRIPPERS_START
            self._deadline = None
            return PickupStep(
                self.GRIPPERS_START,
                "reta concluida; parando e fechando as duas garras",
                motor_action="stop",
                gripper_action=(
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
            )

        if self.state == self.GRIPPERS_START:
            return PickupStep(
                self.GRIPPERS_START,
                "aguardando confirmacao do fechamento das garras",
            )

        if self.state == self.GRIPPERS_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.GRIPPERS_WAIT,
                    "rodas paradas; aguardando as garras fecharem",
                )
            self.state = self.LIFT_PENDING
            self._deadline = None
            return PickupStep(
                self.LIFT_PENDING,
                "garras fechadas; subindo o Futaba por 2,5 s",
                motor_action="hold",
                futaba_action=(
                    cfg.BALL_PICKUP_LIFT_POWER,
                    cfg.BALL_PICKUP_LIFT_MS,
                ),
            )

        if self.state == self.LIFT_PENDING:
            return PickupStep(
                self.LIFT_PENDING,
                "aguardando confirmacao da subida do Futaba",
            )

        if self.state == self.LIFT_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.LIFT_WAIT,
                    "subindo o Futaba por 2,5 s",
                )
            self.state = self.LOWER_PENDING
            self._deadline = None
            return PickupStep(
                self.LOWER_PENDING,
                "subida concluida; descendo o Futaba por 25 ms",
                stop_futaba=True,
                futaba_action=(
                    cfg.BALL_PICKUP_LOWER_POWER,
                    cfg.BALL_PICKUP_LOWER_MS,
                ),
            )

        if self.state == self.LOWER_PENDING:
            return PickupStep(
                self.LOWER_PENDING,
                "aguardando confirmacao do pulso de descida",
            )

        if self.state == self.LOWER_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.LOWER_WAIT,
                    "descendo o Futaba por 25 ms",
                )
            self.state = self.RELEASE_PENDING
            self._deadline = None
            return PickupStep(
                self.RELEASE_PENDING,
                self._release_detail(),
                stop_futaba=True,
                gripper_action=self._release_action(),
            )

        if self.state == self.RELEASE_PENDING:
            return PickupStep(
                self.RELEASE_PENDING,
                "aguardando confirmacao da primeira garra",
            )

        if self.state == self.RELEASE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.RELEASE_WAIT,
                    "aguardando a primeira garra abrir",
                )
            self._wiggle_index = 0
            self.state = self.WIGGLE_PENDING
            self._deadline = None
            return PickupStep(
                self.WIGGLE_PENDING,
                self._wiggle_detail(),
                gripper_action=self._wiggle_actions[self._wiggle_index],
            )

        if self.state == self.WIGGLE_PENDING:
            return PickupStep(
                self.WIGGLE_PENDING,
                "aguardando confirmacao do movimento de liberacao",
            )

        if self.state == self.WIGGLE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.WIGGLE_WAIT,
                    "aguardando o movimento da garra terminar",
                )
            next_index = self._wiggle_index + 1
            if next_index < len(self._wiggle_actions):
                self._wiggle_index = next_index
                self.state = self.WIGGLE_PENDING
                self._deadline = None
                return PickupStep(
                    self.WIGGLE_PENDING,
                    self._wiggle_detail(),
                    gripper_action=self._wiggle_actions[
                        self._wiggle_index],
                )
            self.state = self.COMPLETE
            self._terminal_detail = (
                f"coleta e liberacao da esfera {self._kind} concluidas")
            return PickupStep(
                self.COMPLETE,
                self._terminal_detail,
                terminal=True,
            )

        return PickupStep(
            self.state,
            self._terminal_detail,
            terminal=True,
        )

    def mark_futaba_started(self, now=None):
        """Inicia cada prazo somente depois da escrita serial correspondente."""
        now = time.monotonic() if now is None else float(now)
        if self.state == self.FUTABA_PENDING:
            self.state = self.FUTABA_WAIT
            self._deadline = (
                now
                + cfg.BALL_PICKUP_FUTABA_MS / 1000.0
                + cfg.BALL_PICKUP_FUTABA_GUARD_S
            )
            return
        if self.state == self.LIFT_PENDING:
            self.state = self.LIFT_WAIT
            self._deadline = (
                now
                + cfg.BALL_PICKUP_LIFT_MS / 1000.0
                + cfg.BALL_PICKUP_LIFT_GUARD_S
            )
            return
        if self.state == self.LOWER_PENDING:
            self.state = self.LOWER_WAIT
            self._deadline = (
                now
                + cfg.BALL_PICKUP_LOWER_MS / 1000.0
                + cfg.BALL_PICKUP_LOWER_GUARD_S
            )
            return
        raise RuntimeError(
            "confirmacao do Futaba fora de um estado de partida")

    def mark_forward_started(self, now=None):
        """Inicia os 1,5 s completos de reta antes das garras."""
        if self.state != self.FORWARD_START:
            raise RuntimeError(
                "confirmacao do avanco fora do estado de partida")
        now = time.monotonic() if now is None else float(now)
        self.state = self.FORWARD_LEAD
        self._deadline = now + cfg.BALL_PICKUP_FORWARD_S

    def mark_grippers_started(self, now=None):
        """Confirma um lote de garras e inicia seu tempo fisico."""
        now = time.monotonic() if now is None else float(now)
        if self.state == self.GRIPPERS_START:
            self.state = self.GRIPPERS_WAIT
            self._deadline = now + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            return
        if self.state == self.RELEASE_PENDING:
            self.state = self.RELEASE_WAIT
            self._deadline = now + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            return
        if self.state == self.WIGGLE_PENDING:
            self.state = self.WIGGLE_WAIT
            self._deadline = now + cfg.BALL_PICKUP_WIGGLE_STEP_S
            return
        raise RuntimeError(
            "confirmacao das garras fora de um estado de partida")

    def fail(self, detail):
        self.state = self.FAULT
        self._terminal_detail = str(detail)
        return PickupStep(
            self.FAULT,
            self._terminal_detail,
            motor_action="stop",
            stop_futaba=True,
            terminal=True,
        )

    def _release_action(self):
        delta = cfg.BALL_PICKUP_RELEASE_DELTA
        if self._kind == "silver":
            return delta, 0
        return 0, -delta

    def _release_detail(self):
        if self._kind == "silver":
            return "esfera prata; abrindo primeiro a garra esquerda"
        return "esfera preta; abrindo primeiro a garra direita"

    def _build_wiggle_actions(self, target_kind):
        delta = cfg.BALL_PICKUP_WIGGLE_DELTA
        if target_kind == "silver":
            pair = ((0, delta), (0, -delta))
        else:
            pair = ((-delta, 0), (delta, 0))
        return pair * cfg.BALL_PICKUP_WIGGLE_REPETITIONS

    def _wiggle_detail(self):
        side = "direita" if self._kind == "silver" else "esquerda"
        return (
            f"liberando esfera {self._kind}; movimento "
            f"{self._wiggle_index + 1}/{len(self._wiggle_actions)} "
            f"da garra {side}")
