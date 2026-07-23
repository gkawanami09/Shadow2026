"""Sequencia nao bloqueante de coleta depois da aproximacao da esfera."""

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
    """Re -> Futaba -> avanco -> garras, com deadlines monotonic."""

    IDLE = "PICKUP_IDLE"
    BACKUP_START = "PICKUP_BACKUP_START"
    BACKUP_PENDING = "PICKUP_BACKUP_PENDING"
    BACKUP = "PICKUP_BACKUP"
    FUTABA_START = "PICKUP_FUTABA_START"
    FUTABA_WAIT = "PICKUP_FUTABA"
    FORWARD_START = "PICKUP_FORWARD_START"
    FORWARD_LEAD = "PICKUP_FORWARD_LEAD"
    GRIPPERS_START = "PICKUP_GRIPPERS_START"
    GRIPPERS_WAIT = "PICKUP_GRIPPERS"
    COMPLETE = "PICKUP_COMPLETE"
    FAULT = "PICKUP_FAULT"

    def __init__(self):
        self.state = self.IDLE
        self._deadline = None
        self._forward_deadline = None
        self._terminal_detail = ""

    @property
    def started(self):
        return self.state != self.IDLE

    @property
    def terminal(self):
        return self.state in (self.COMPLETE, self.FAULT)

    def start(self):
        """Arma a sequencia uma unica vez; a re comeca no proximo update."""
        if self.state != self.IDLE:
            return False
        self.state = self.BACKUP_START
        return True

    def update(self, now=None):
        now = time.monotonic() if now is None else float(now)

        if self.state == self.IDLE:
            return PickupStep(
                self.IDLE,
                "coleta ainda nao iniciada",
            )

        if self.state == self.BACKUP_START:
            self.state = self.BACKUP_PENDING
            return PickupStep(
                self.BACKUP_PENDING,
                "enviando comando de re",
                angle=200,
                speed=cfg.BALL_PICKUP_REVERSE_SPEED,
                motor_action="reverse",
            )

        if self.state == self.BACKUP_PENDING:
            return PickupStep(
                self.BACKUP_PENDING,
                "aguardando confirmacao do envio da re",
                angle=200,
                speed=cfg.BALL_PICKUP_REVERSE_SPEED,
            )

        if self.state == self.BACKUP:
            if now < self._deadline:
                return PickupStep(
                    self.BACKUP,
                    "afastando da bolinha",
                    angle=200,
                    speed=cfg.BALL_PICKUP_REVERSE_SPEED,
                )

            self.state = self.FUTABA_START
            self._deadline = None
            return PickupStep(
                self.FUTABA_START,
                "rodas zeradas; baixando o Futaba",
                motor_action="hold",
                futaba_action=(
                    cfg.BALL_PICKUP_FUTABA_POWER,
                    cfg.BALL_PICKUP_FUTABA_MS,
                ),
            )

        if self.state == self.FUTABA_START:
            return PickupStep(
                self.FUTABA_START,
                "aguardando confirmacao do envio ao Futaba",
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
                "Futaba embaixo; iniciando avanco antes das garras",
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
                    "avancando antes de fechar as garras",
                    angle=0,
                    speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                )

            self.state = self.GRIPPERS_START
            self._deadline = None
            return PickupStep(
                self.GRIPPERS_START,
                "avanco iniciado; fechando as duas garras",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                gripper_action=(
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
            )

        if self.state == self.GRIPPERS_START:
            return PickupStep(
                self.GRIPPERS_START,
                "aguardando confirmacao do avanco e das garras",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
            )

        if self.state == self.GRIPPERS_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.GRIPPERS_WAIT,
                    "avancando enquanto as garras fecham",
                    angle=0,
                    speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                )
            self.state = self.COMPLETE
            self._terminal_detail = "sequencia de coleta concluida"
            return PickupStep(
                self.COMPLETE,
                self._terminal_detail,
                motor_action="stop",
                terminal=True,
            )

        return PickupStep(
            self.state,
            self._terminal_detail,
            terminal=True,
        )

    def mark_futaba_started(self, now=None):
        """Inicia o prazo somente depois que a escrita serial retornou."""
        if self.state != self.FUTABA_START:
            raise RuntimeError(
                "confirmacao do Futaba fora do estado de partida")
        now = time.monotonic() if now is None else float(now)
        self.state = self.FUTABA_WAIT
        self._deadline = (
            now
            + cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )

    def mark_reverse_started(self, now=None):
        """Conta a re depois que a escrita serial correspondente retornou."""
        if self.state != self.BACKUP_PENDING:
            raise RuntimeError(
                "confirmacao da re fora do estado de partida")
        now = time.monotonic() if now is None else float(now)
        self.state = self.BACKUP
        self._deadline = now + cfg.BALL_PICKUP_REVERSE_S

    def mark_forward_started(self, now=None):
        """Inicia os 1,5 s totais e a vantagem antes das garras."""
        if self.state != self.FORWARD_START:
            raise RuntimeError(
                "confirmacao do avanco fora do estado de partida")
        now = time.monotonic() if now is None else float(now)
        self.state = self.FORWARD_LEAD
        self._deadline = now + cfg.BALL_PICKUP_FORWARD_LEAD_S
        self._forward_deadline = now + cfg.BALL_PICKUP_FORWARD_S

    def mark_grippers_started(self, now=None):
        """Mantem o prazo total contado desde o inicio do avanco."""
        if self.state != self.GRIPPERS_START:
            raise RuntimeError(
                "confirmacao das garras fora do estado de partida")
        if self._forward_deadline is None:
            raise RuntimeError("prazo do avanco ainda nao foi iniciado")
        self.state = self.GRIPPERS_WAIT
        self._deadline = self._forward_deadline

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
