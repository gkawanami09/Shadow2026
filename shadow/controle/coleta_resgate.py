"""Controla a sequência de coleta e liberação da esfera."""

from dataclasses import dataclass
import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


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
    GRIPPERS_PREPARE_PENDING = "PICKUP_GRIPPERS_PREPARE_PENDING"
    GRIPPERS_PREPARE_WAIT = "PICKUP_GRIPPERS_PREPARE"
    PRE_FORWARD_START = "PICKUP_PRE_FORWARD_START"
    PRE_FORWARD_PENDING = "PICKUP_PRE_FORWARD_PENDING"
    PRE_FORWARD_LEAD = "PICKUP_PRE_FORWARD"
    FUTABA_START = "PICKUP_FUTABA_START"
    FUTABA_PENDING = "PICKUP_FUTABA_PENDING"
    FUTABA_WAIT = "PICKUP_FUTABA"
    FORWARD_START = "PICKUP_FORWARD_START"
    FORWARD_LEAD = "PICKUP_FORWARD_LEAD"
    FINAL_FORWARD = "PICKUP_FINAL_FORWARD"
    WALL_PAUSE_PENDING = "PICKUP_WALL_PAUSE_PENDING"
    WALL_PAUSE_WAIT = "PICKUP_WALL_PAUSE_WAIT"
    WALL_REVERSE_PENDING = "PICKUP_WALL_REVERSE_PENDING"
    WALL_REVERSE_WAIT = "PICKUP_WALL_REVERSE_WAIT"
    WALL_POST_REVERSE_PENDING = "PICKUP_WALL_POST_REVERSE_PENDING"
    WALL_POST_REVERSE_WAIT = "PICKUP_WALL_POST_REVERSE_WAIT"
    GRIPPERS_START = "PICKUP_GRIPPERS_START"
    GRIPPERS_WAIT = "PICKUP_GRIPPERS"
    LIFT_PENDING = "PICKUP_LIFT_PENDING"
    LIFT_WAIT = "PICKUP_LIFT"
    LIFT_SLOW_PENDING = "PICKUP_LIFT_SLOW_PENDING"
    LIFT_SLOW_WAIT = "PICKUP_LIFT_SLOW"
    CARRY_READY = "PICKUP_CARRY_READY"
    DEPOSIT_START = "PICKUP_DEPOSIT_START"
    LOWER_PENDING = "PICKUP_LOWER_PENDING"
    LOWER_WAIT = "PICKUP_LOWER"
    RELEASE_PENDING = "PICKUP_RELEASE_PENDING"
    RELEASE_WAIT = "PICKUP_RELEASE"
    WIGGLE_PENDING = "PICKUP_WIGGLE_PENDING"
    WIGGLE_WAIT = "PICKUP_WIGGLE"
    RESTORE_PENDING = "PICKUP_RESTORE_PENDING"
    RESTORE_WAIT = "PICKUP_RESTORE"
    COMPLETE = "PICKUP_COMPLETE"
    FAULT = "PICKUP_FAULT"

    def __init__(self, grippers_prepositioned=True):
        self.state = self.IDLE
        self._deadline = None
        self._kind = None
        self._wiggle_actions = ()
        self._wiggle_index = 0
        self._gripper_close_actions = ()
        self._gripper_close_index = 0
        self._gripper_capture_action_count = 0
        self._terminal_detail = ""
        self._release_mode = "deposit"
        self._wall_mode = False
        self._grippers_prepositioned = bool(grippers_prepositioned)

    @property
    def started(self):
        return self.state != self.IDLE

    @property
    def terminal(self):
        return self.state in (self.COMPLETE, self.FAULT)

    @property
    def target_kind(self):
        return self._kind

    @property
    def ready_for_deposit(self):
        """A esfera esta fechada e elevada, pronta para ser transportada."""
        return self.state == self.CARRY_READY

    def start(self, target_kind, wall_mode=False):
        """Arma a sequencia e congela a cor ate o estado terminal."""
        if self.state != self.IDLE:
            return False
        if target_kind not in ("silver", "black"):
            raise ValueError(
                "a coleta exige cor confirmada silver ou black")
        self._kind = target_kind
        self._wall_mode = bool(wall_mode)
        self._wiggle_actions = self._build_wiggle_actions(target_kind)
        self._gripper_close_actions = self._build_close_actions()
        self._gripper_close_index = 0
        # A garra desce antes de qualquer movimento. O avanco que antes era
        # dividido em 1 s levantada + 1 s abaixada passa inteiro para depois
        # da descida, preservando a distancia total.
        self.state = (
            self.FUTABA_START
            if self._grippers_prepositioned
            else self.GRIPPERS_PREPARE_PENDING
        )
        return True

    def update(self, now=None):
        now = time.monotonic() if now is None else float(now)

        if self.state == self.IDLE:
            return PickupStep(
                self.IDLE,
                "coleta ainda nao iniciada",
            )

        if self.state == self.GRIPPERS_PREPARE_PENDING:
            return PickupStep(
                self.GRIPPERS_PREPARE_PENDING,
                "posicionando as garras em -10/+10 graus antes da descida",
                gripper_action=(
                    cfg.BALL_PICKUP_INITIAL_LEFT_DELTA,
                    cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA,
                ),
            )

        if self.state == self.GRIPPERS_PREPARE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.GRIPPERS_PREPARE_WAIT,
                    "aguardando as garras assentarem antes de baixar",
                )
            self.state = self.FUTABA_START
            self._deadline = None
            return self.update(now=now)

        if self.state == self.PRE_FORWARD_START:
            self.state = self.PRE_FORWARD_PENDING
            return PickupStep(
                self.PRE_FORWARD_PENDING,
                "iniciando primeiro avanco de 1 s antes de baixar a garra",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                motor_action="forward",
            )

        if self.state == self.PRE_FORWARD_PENDING:
            return PickupStep(
                self.PRE_FORWARD_PENDING,
                "aguardando confirmacao do primeiro avanco",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
            )

        if self.state == self.PRE_FORWARD_LEAD:
            if now < self._deadline:
                return PickupStep(
                    self.PRE_FORWARD_LEAD,
                    "avancando por 1 s com o elevador levantado",
                    angle=0,
                    speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                )
            self.state = self.FUTABA_START
            self._deadline = None
            return PickupStep(
                self.FUTABA_START,
                "primeiro avanco concluido; parando antes de baixar",
                motor_action="stop",
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
            tempo_avanco = (
                cfg.BALL_WALL_PICKUP_FORWARD_S
                if self._wall_mode
                else cfg.BALL_PICKUP_FORWARD_LEAD_S
            )
            return PickupStep(
                self.FORWARD_START,
                (
                    "Futaba embaixo; iniciando avanco contra a parede "
                    f"por {tempo_avanco:.2f} s"
                    if self._wall_mode
                    else "Futaba embaixo; iniciando avanco total de 2 s"
                ),
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
                    (
                        "avancando com a garra aberta contra a parede"
                        if self._wall_mode
                        else "avancando por 2 s com o Futaba embaixo"
                    ),
                    angle=0,
                    speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                )
            self.state = self.FINAL_FORWARD
            self._deadline = now + cfg.BALL_PICKUP_FINAL_FORWARD_S
            return PickupStep(
                self.FINAL_FORWARD,
                "avanco final de 200 ms com o elevador baixo",
                angle=0,
                speed=cfg.BALL_PICKUP_FORWARD_SPEED,
            )

        if self.state == self.FINAL_FORWARD:
            if now < self._deadline:
                return PickupStep(
                    self.FINAL_FORWARD,
                    "completando os 200 ms finais antes de fechar",
                    angle=0,
                    speed=cfg.BALL_PICKUP_FORWARD_SPEED,
                )
            self._deadline = None
            if self._wall_mode:
                self.state = self.WALL_PAUSE_PENDING
                return PickupStep(
                    self.WALL_PAUSE_PENDING,
                    "esfera pressionada; parando antes da re curta",
                    motor_action="stop",
                )
            self.state = self.GRIPPERS_START
            return self._next_close_step(first=True)

        if self.state == self.WALL_PAUSE_PENDING:
            return PickupStep(
                self.WALL_PAUSE_PENDING,
                "aguardando confirmacao da parada antes da re",
            )

        if self.state == self.WALL_PAUSE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.WALL_PAUSE_WAIT,
                    "rodas paradas antes de inverter o movimento",
                )
            self.state = self.WALL_REVERSE_PENDING
            self._deadline = None
            return PickupStep(
                self.WALL_REVERSE_PENDING,
                "dando re curta para afastar a esfera da parede",
                angle=200,
                speed=cfg.BALL_WALL_PICKUP_REVERSE_SPEED,
                motor_action="reverse",
            )

        if self.state == self.WALL_REVERSE_PENDING:
            return PickupStep(
                self.WALL_REVERSE_PENDING,
                "aguardando confirmacao do comando de re",
                angle=200,
                speed=cfg.BALL_WALL_PICKUP_REVERSE_SPEED,
            )

        if self.state == self.WALL_REVERSE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.WALL_REVERSE_WAIT,
                    "afastando da parede antes de fechar as garras",
                    angle=200,
                    speed=cfg.BALL_WALL_PICKUP_REVERSE_SPEED,
                )
            self.state = self.WALL_POST_REVERSE_PENDING
            self._deadline = None
            return PickupStep(
                self.WALL_POST_REVERSE_PENDING,
                "re concluida; freando antes de fechar as garras",
                motor_action="stop",
            )

        if self.state == self.WALL_POST_REVERSE_PENDING:
            return PickupStep(
                self.WALL_POST_REVERSE_PENDING,
                "aguardando confirmacao da parada depois da re",
            )

        if self.state == self.WALL_POST_REVERSE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.WALL_POST_REVERSE_WAIT,
                    "aguardando o chassi assentar antes de fechar",
                )
            self.state = self.GRIPPERS_START
            self._deadline = None
            # As rodas ja foram paradas e assentaram no estado anterior.
            # Portanto o fechamento sai sozinho, sem um segundo movimento.
            return self._next_close_step(first=False)

        if self.state == self.GRIPPERS_START:
            return PickupStep(
                self.GRIPPERS_START,
                "aguardando confirmacao do fechamento das garras",
            )

        if self.state == self.GRIPPERS_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.GRIPPERS_WAIT,
                    "rodas paradas; fechando as garras gradualmente",
                )
            if self._gripper_close_index < len(
                self._gripper_close_actions
            ):
                self.state = self.GRIPPERS_START
                self._deadline = None
                return self._next_close_step()
            self.state = self.LIFT_PENDING
            self._deadline = None
            return PickupStep(
                self.LIFT_PENDING,
                "garras fechadas; iniciando subida do Futaba",
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
                    "subindo o Futaba na velocidade normal",
                )
            self.state = self.LIFT_SLOW_PENDING
            self._deadline = None
            return PickupStep(
                self.LIFT_SLOW_PENDING,
                "perto do alto; reduzindo a velocidade do Futaba",
                futaba_action=(
                    cfg.BALL_PICKUP_LIFT_SLOW_POWER,
                    cfg.BALL_PICKUP_LIFT_SLOW_MS,
                ),
            )

        if self.state == self.LIFT_SLOW_PENDING:
            return PickupStep(
                self.LIFT_SLOW_PENDING,
                "aguardando confirmacao da subida lenta do Futaba",
            )

        if self.state == self.LIFT_SLOW_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.LIFT_SLOW_WAIT,
                    "terminando a subida em velocidade reduzida",
                )
            self.state = self.CARRY_READY
            self._deadline = None
            return PickupStep(
                self.CARRY_READY,
                "esfera elevada; sustentacao curta para evitar o recuo",
                futaba_action=(
                    cfg.BALL_PICKUP_LIFT_HOLD_POWER,
                    cfg.BALL_PICKUP_LIFT_HOLD_MS,
                ),
            )

        if self.state == self.CARRY_READY:
            return PickupStep(
                self.CARRY_READY,
                "transportando a esfera; liberacao ainda bloqueada",
            )

        if self.state == self.DEPOSIT_START:
            if self._release_mode == "selection":
                # Na selecao imediatamente apos a coleta nao existe descida:
                # aquele pulso contrario de 25 ms era o pequeno recuo visivel.
                # A sustentacao curta continua enquanto a primeira garra abre.
                self.state = self.RELEASE_PENDING
                return PickupStep(
                    self.RELEASE_PENDING,
                    self._release_detail(),
                    gripper_action=self._release_action(),
                )
            self.state = self.LOWER_PENDING
            return PickupStep(
                self.LOWER_PENDING,
                "marcador correto alcancado; descendo o Futaba por 25 ms",
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
            self.state = self.RESTORE_PENDING
            self._deadline = None
            return PickupStep(
                self.RESTORE_PENDING,
                "liberacao concluida; restaurando as garras",
                gripper_action=self._restore_action(),
            )

        if self.state == self.RESTORE_PENDING:
            return PickupStep(
                self.RESTORE_PENDING,
                "aguardando confirmacao da posicao inicial das garras",
            )

        if self.state == self.RESTORE_WAIT:
            if now < self._deadline:
                return PickupStep(
                    self.RESTORE_WAIT,
                    "aguardando as garras voltarem a posicao inicial",
                )
            self.state = self.COMPLETE
            self._deadline = None
            self._terminal_detail = (
                (
                    f"coleta e selecao da esfera {self._kind} concluidas"
                    if self._release_mode == "selection"
                    else
                    f"coleta e liberacao da esfera {self._kind} concluidas"
                )
            )
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

    def resume_deposit(self):
        """Libera uma unica vez o sufixo de deposito no marcador correto."""
        if self.state != self.CARRY_READY:
            return False
        self._release_mode = "deposit"
        self.state = self.DEPOSIT_START
        return True

    def resume_selection(self):
        """Seleciona prata para a esquerda e preta para a direita."""
        if self.state != self.CARRY_READY:
            return False
        self._release_mode = "selection"
        self.state = self.DEPOSIT_START
        return True

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
            # Sem guarda entre as fases: a ordem lenta substitui a normal
            # antes que o mecanismo tenha tempo de recuar.
            self._deadline = now + cfg.BALL_PICKUP_LIFT_MS / 1000.0
            return
        if self.state == self.LIFT_SLOW_PENDING:
            self.state = self.LIFT_SLOW_WAIT
            # A sustentacao curta substitui esta ordem sem um intervalo solto.
            self._deadline = now + cfg.BALL_PICKUP_LIFT_SLOW_MS / 1000.0
            return
        if self.state == self.CARRY_READY:
            # O pulso e temporizado pelo Arduino. Na missao, a selecao comeca
            # logo depois e substitui esta ordem pela pequena descida.
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
        """Inicia o prazo do avanco feito com o Futaba embaixo."""
        now = time.monotonic() if now is None else float(now)
        if self.state == self.PRE_FORWARD_PENDING:
            self.state = self.PRE_FORWARD_LEAD
            self._deadline = now + cfg.BALL_PICKUP_PRE_FORWARD_S
            return
        if self.state != self.FORWARD_START:
            raise RuntimeError(
                "confirmacao do avanco fora do estado de partida")
        self.state = self.FORWARD_LEAD
        duracao = (
            cfg.BALL_WALL_PICKUP_FORWARD_S
            if self._wall_mode
            else cfg.BALL_PICKUP_FORWARD_LEAD_S
        )
        self._deadline = now + duracao

    def mark_wall_pause_started(self, now=None):
        """Inicia a pausa segura antes de inverter para a re."""
        if self.state != self.WALL_PAUSE_PENDING:
            raise RuntimeError(
                "confirmacao da pausa fora do modo de parede")
        now = time.monotonic() if now is None else float(now)
        self.state = self.WALL_PAUSE_WAIT
        self._deadline = (
            now + cfg.BALL_WALL_PICKUP_DIRECTION_CHANGE_PAUSE_S)

    def mark_reverse_started(self, now=None):
        """Inicia o prazo da re somente depois da escrita serial."""
        if self.state != self.WALL_REVERSE_PENDING:
            raise RuntimeError(
                "confirmacao da re fora do modo de parede")
        now = time.monotonic() if now is None else float(now)
        self.state = self.WALL_REVERSE_WAIT
        self._deadline = now + cfg.BALL_WALL_PICKUP_REVERSE_S

    def mark_post_reverse_pause_started(self, now=None):
        """Inicia o assentamento somente apos confirmar o STOP pos-re."""
        if self.state != self.WALL_POST_REVERSE_PENDING:
            raise RuntimeError(
                "confirmacao da pausa pos-re fora do modo de parede")
        now = time.monotonic() if now is None else float(now)
        self.state = self.WALL_POST_REVERSE_WAIT
        self._deadline = (
            now + cfg.BALL_WALL_PICKUP_POST_REVERSE_PAUSE_S)

    def mark_grippers_started(self, now=None):
        """Confirma um lote de garras e inicia seu tempo fisico."""
        now = time.monotonic() if now is None else float(now)
        if self.state == self.GRIPPERS_PREPARE_PENDING:
            self.state = self.GRIPPERS_PREPARE_WAIT
            self._deadline = now + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            return
        if self.state == self.GRIPPERS_START:
            self.state = self.GRIPPERS_WAIT
            if self._gripper_close_index >= len(
                self._gripper_close_actions
            ):
                intervalo = cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            elif (
                self._gripper_close_index
                <= self._gripper_capture_action_count
            ):
                intervalo = cfg.BALL_PICKUP_GRIPPER_CAPTURE_INTERVAL_S
            else:
                intervalo = cfg.BALL_PICKUP_GRIPPER_STEP_INTERVAL_S
            self._deadline = now + intervalo
            return
        if self.state == self.RELEASE_PENDING:
            self.state = self.RELEASE_WAIT
            self._deadline = now + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            return
        if self.state == self.WIGGLE_PENDING:
            self.state = self.WIGGLE_WAIT
            self._deadline = now + cfg.BALL_PICKUP_WIGGLE_STEP_S
            return
        if self.state == self.RESTORE_PENDING:
            self.state = self.RESTORE_WAIT
            self._deadline = now + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            return
        raise RuntimeError(
            "confirmacao das garras fora de um estado de partida")

    def recovery_lift_profile(self, now=None):
        """Calcula quanto ainda e seguro subir depois de um reinicio serial.

        Antes da subida normal, o Futaba pode estar em qualquer ponto baixo e
        recebe o perfil completo. Se a queda aconteceu durante a propria
        subida, usa somente o tempo que faltava. Depois da fase normal, aplica
        apenas a fase lenta: assim nao repete 1,9 s contra o batente superior.
        """
        now = time.monotonic() if now is None else float(now)
        normal_ms = int(cfg.BALL_PICKUP_LIFT_MS)
        lento_ms = int(cfg.BALL_PICKUP_LIFT_SLOW_MS)

        if self.state == self.LIFT_WAIT and self._deadline is not None:
            restante_ms = int(round(max(self._deadline - now, 0.0) * 1000.0))
            return min(restante_ms, normal_ms), lento_ms
        if self.state == self.LIFT_SLOW_WAIT and self._deadline is not None:
            restante_ms = int(round(max(self._deadline - now, 0.0) * 1000.0))
            return 0, min(restante_ms, lento_ms)
        if self.state in (
            self.LIFT_SLOW_PENDING,
            self.CARRY_READY,
            self.DEPOSIT_START,
            self.LOWER_PENDING,
            self.LOWER_WAIT,
            self.RELEASE_PENDING,
            self.RELEASE_WAIT,
            self.WIGGLE_PENDING,
            self.WIGGLE_WAIT,
            self.RESTORE_PENDING,
            self.RESTORE_WAIT,
        ):
            return 0, lento_ms
        return normal_ms, lento_ms

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

    def _restore_action(self):
        """Retorna as duas garras exatamente para a base de 0 graus."""
        if self._kind == "silver":
            return 0, -cfg.BALL_PICKUP_RIGHT_DELTA
        return (
            -cfg.BALL_PICKUP_LEFT_DELTA,
            0,
        )

    def _build_close_actions(self):
        """Captura rapido e completa o aperto sem perder nenhum grau."""
        captura = int(cfg.BALL_PICKUP_GRIPPER_CAPTURE_DEGREES)
        passo = int(cfg.BALL_PICKUP_GRIPPER_STEP_DEGREES)
        if captura <= 0 or passo <= 0:
            raise ValueError("passos das garras devem ser positivos")

        # O fechamento termina no mesmo ponto fisico de antes (-55/+55),
        # partindo agora da pre-abertura -10/+10.
        restantes = [
            int(cfg.BALL_PICKUP_LEFT_DELTA)
            - int(cfg.BALL_PICKUP_INITIAL_LEFT_DELTA),
            int(cfg.BALL_PICKUP_RIGHT_DELTA)
            - int(cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA),
        ]
        acoes = []

        # Fecha primeiro a maior parte do vao. Os comandos ainda sao separados:
        # as duas garras nunca partem juntas no mesmo pacote serial.
        for indice in (0, 1):
            restante = restantes[indice]
            if restante == 0:
                continue
            deslocamento = min(abs(restante), captura)
            deslocamento = deslocamento if restante > 0 else -deslocamento
            acao = [0, 0]
            acao[indice] = deslocamento
            acoes.append(tuple(acao))
            restantes[indice] -= deslocamento
        self._gripper_capture_action_count = len(acoes)

        while restantes[0] != 0 or restantes[1] != 0:
            for indice in (0, 1):
                restante = restantes[indice]
                if restante == 0:
                    continue
                deslocamento = min(abs(restante), passo)
                deslocamento = (
                    deslocamento if restante > 0 else -deslocamento)
                acao = [0, 0]
                acao[indice] = deslocamento
                acoes.append(tuple(acao))
                restantes[indice] -= deslocamento
        return tuple(acoes)

    def _next_close_step(self, first=False):
        if self._gripper_close_index >= len(self._gripper_close_actions):
            raise RuntimeError("fechamento gradual das garras ja terminou")
        acao = self._gripper_close_actions[self._gripper_close_index]
        self._gripper_close_index += 1
        return PickupStep(
            self.GRIPPERS_START,
            (
                "parando; " if first else ""
            )
            + "fechando uma garra por vez "
            + f"({self._gripper_close_index}/"
            + f"{len(self._gripper_close_actions)})",
            motor_action="stop" if first else "",
            gripper_action=acao,
        )

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
