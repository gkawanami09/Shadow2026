"""Coordenador explícito da missão completa do Shadow.

Este módulo NÃO substitui os controladores que já funcionam. A aproximação, a
coleta, o depósito e a busca continuam sendo responsabilidade de
``aproximacao_resgate``, ``coleta_resgate``, ``deposito_resgate`` e
``busca_resgate``. O que faltava — e o que existe aqui — é a camada de cima:

* qual é o estado da missão inteira, com um nome só e sem ``if`` soltos;
* quantas vítimas de cada cor já foram efetivamente depositadas;
* qual é a política de escolha da próxima vítima;
* quando a sala pode ser declarada concluída;
* em que ORDEM exata acontece a troca entre segue-linha e resgate.

A ordem do handoff é a parte crítica: segue-linha e resgate disputam a mesma
serial, o mesmo Arduino e os mesmos motores. Errar a ordem significa dois
donos ao mesmo tempo. Por isso ela é dado, não código espalhado — está em
``HANDOFF_TO_RESCUE``/``HANDOFF_TO_LINE`` e é executada por ``HandoffExecutor``
contra um objeto injetável, o que permite testar a sequência sem hardware.
"""

from dataclasses import dataclass, field


class MissionState:
    """Nomes canônicos dos estados da missão."""

    FOLLOW_LINE = "FOLLOW_LINE"
    ENTRY_SILVER_CANDIDATE = "ENTRY_SILVER_CANDIDATE"
    VERIFY_ENTRY_SILVER = "VERIFY_ENTRY_SILVER"
    ENTER_RESCUE_ZONE = "ENTER_RESCUE_ZONE"
    STOP_AND_HANDOFF_TO_RESCUE = "STOP_AND_HANDOFF_TO_RESCUE"
    RESCUE_SCAN = "RESCUE_SCAN"
    TARGET_BRAKE = "TARGET_BRAKE"
    TARGET_VERIFY = "TARGET_VERIFY"
    TARGET_LOCK = "TARGET_LOCK"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    PICKUP = "PICKUP"
    CARRY_READY = "CARRY_READY"
    FIND_CORRECT_TRIANGLE = "FIND_CORRECT_TRIANGLE"
    APPROACH_TRIANGLE = "APPROACH_TRIANGLE"
    DEPOSIT = "DEPOSIT"
    RESTORE_GRIPPERS = "RESTORE_GRIPPERS"
    UPDATE_INVENTORY = "UPDATE_INVENTORY"
    RESCUE_RECOVERY = "RESCUE_RECOVERY"
    VERIFY_RESCUE_COMPLETE = "VERIFY_RESCUE_COMPLETE"
    DETECT_BOTH_TRIANGLES_FINAL = "DETECT_BOTH_TRIANGLES_FINAL"
    FIND_BLACK_EXIT = "FIND_BLACK_EXIT"
    CROSS_EXIT = "CROSS_EXIT"
    STOP_AND_HANDOFF_TO_LINE = "STOP_AND_HANDOFF_TO_LINE"
    RED_FINISH = "RED_FINISH"
    ABORTED = "ABORTED"


# Quantidade fixa de vítimas na sala, definida pelo regulamento.
EXPECTED_SILVER = 2
EXPECTED_BLACK = 1
EXPECTED_TOTAL = EXPECTED_SILVER + EXPECTED_BLACK

# Destino obrigatório por cor. Duplicado propositalmente com
# ``config_resgate.DEPOSIT_MARKER_BY_BALL_KIND`` seria um risco; aqui apenas
# reexportamos a mesma verdade para quem consome a missão.
TRIANGLE_BY_VICTIM = {"silver": "green", "black": "red"}

# Políticas de escolha da próxima vítima.
POLICY_NEAREST_VALID = "nearest_valid"   # padrão OBR
POLICY_SILVER_FIRST = "silver_first"     # vantagem no mundial
VALID_POLICIES = (POLICY_NEAREST_VALID, POLICY_SILVER_FIRST)


class MissionError(RuntimeError):
    pass


@dataclass
class RescueInventory:
    """Contagem de vítimas efetivamente depositadas, por cor.

    O incremento só pode acontecer depois da liberação física confirmada — a
    chamada vem do fim do sequenciador de coleta, nunca da visão.
    """

    silver_deposited: int = 0
    black_deposited: int = 0
    #: Cores já entregues, na ordem, para diagnóstico e para o relatório.
    history: list = field(default_factory=list)

    @property
    def total_deposited(self):
        return self.silver_deposited + self.black_deposited

    @property
    def complete(self):
        return (
            self.silver_deposited >= EXPECTED_SILVER
            and self.black_deposited >= EXPECTED_BLACK
        )

    def remaining(self, kind):
        if kind == "silver":
            return max(EXPECTED_SILVER - self.silver_deposited, 0)
        if kind == "black":
            return max(EXPECTED_BLACK - self.black_deposited, 0)
        raise MissionError(f"cor de vítima desconhecida: {kind}")

    def accepts(self, kind):
        """A sala ainda tem uma vítima desta cor por resgatar?"""
        return self.remaining(kind) > 0

    def record_deposit(self, kind):
        """Registra uma entrega confirmada e devolve o total acumulado."""
        if kind not in TRIANGLE_BY_VICTIM:
            raise MissionError(f"cor de vítima desconhecida: {kind}")
        if not self.accepts(kind):
            # Contar uma quarta vítima significaria que o robô pegou algo que
            # não é vítima, ou contou duas vezes a mesma. Falhar alto é mais
            # seguro que seguir com um inventário mentiroso.
            raise MissionError(
                f"cota de vítimas {kind} já estava completa; "
                "depósito recusado pelo inventário")
        if kind == "silver":
            self.silver_deposited += 1
        else:
            self.black_deposited += 1
        self.history.append(kind)
        return self.total_deposited


class MissionCoordinator:
    """Máquina de estados da missão inteira.

    Não comanda motores e não abre câmeras. Ela recebe eventos já
    confirmados por quem tem essa autoridade e responde com o estado seguinte.
    Isso a torna testável inteira, sem Arduino e sem Raspberry.
    """

    #: Uma única varredura vazia não encerra a sala; é preciso ao menos uma
    #: tentativa de recuperação antes de desistir das vítimas que faltam.
    MAX_EMPTY_SWEEPS = 2

    #: Entradas falsas toleradas antes de desistir. Duas voltas ao percurso
    #: ainda cabem no tempo de prova; a terceira significa que o robô está
    #: preso num laço com a mesma prata falsa, e insistir só queima a prova.
    MAX_FALSE_ENTRIES = 2

    def __init__(self, policy=POLICY_NEAREST_VALID, inventory=None):
        if policy not in VALID_POLICIES:
            raise MissionError(f"política de resgate inválida: {policy}")
        self.policy = policy
        self.inventory = RescueInventory() if inventory is None else inventory
        self.state = MissionState.FOLLOW_LINE
        #: Cor da vítima presa e elevada; congelada até o depósito.
        self.carrying = None
        #: Cor do alvo travado mas ainda não capturado.
        self.pending_kind = None
        self.empty_sweeps = 0
        self.entry_count = 0
        #: Entradas que a câmera de resgate reprovou. Se o robô insistir na
        #: mesma prata falsa, é este contador que interrompe o laço.
        self.false_entries = 0
        self.abort_reason = ""
        self.history = [MissionState.FOLLOW_LINE]

    # -- utilidades ------------------------------------------------------
    def _go(self, state):
        self.state = state
        self.history.append(state)
        return state

    def _require(self, *states):
        if self.state not in states:
            raise MissionError(
                f"evento inválido no estado {self.state}; "
                f"esperado um de {states}")

    @property
    def rescue_active(self):
        return self.state in (
            MissionState.RESCUE_SCAN,
            MissionState.TARGET_BRAKE,
            MissionState.TARGET_VERIFY,
            MissionState.TARGET_LOCK,
            MissionState.ALIGN,
            MissionState.APPROACH,
            MissionState.PICKUP,
            MissionState.CARRY_READY,
            MissionState.FIND_CORRECT_TRIANGLE,
            MissionState.APPROACH_TRIANGLE,
            MissionState.DEPOSIT,
            MissionState.RESTORE_GRIPPERS,
            MissionState.UPDATE_INVENTORY,
            MissionState.RESCUE_RECOVERY,
        )

    @property
    def victim_detector_enabled(self):
        """A visão de vítimas só pode rodar enquanto ainda há o que resgatar."""
        return self.rescue_active and not self.inventory.complete

    @property
    def exit_detector_enabled(self):
        """A faixa preta só existe para o robô no estado de procurar a saída.

        Fora dele o detector nem é consultado — é isto que impede a soleira
        de saída (e a vítima preta) de interromperem a busca de vítimas.
        """
        return self.state in (
            MissionState.FIND_BLACK_EXIT, MissionState.CROSS_EXIT)

    @property
    def target_triangle(self):
        """Triângulo que pode comandar o robô agora — ou ``None``."""
        if self.carrying is None:
            return None
        return TRIANGLE_BY_VICTIM[self.carrying]

    def wants(self, kind):
        """A política atual aceita capturar uma vítima desta cor agora?"""
        if kind not in TRIANGLE_BY_VICTIM:
            return False
        if not self.inventory.accepts(kind):
            return False
        if (
            self.policy == POLICY_SILVER_FIRST
            and kind == "black"
            and self.inventory.accepts("silver")
        ):
            # No mundial as vivas valem mais e devem sair primeiro. A preta
            # continua elegível assim que as duas pratas estiverem entregues.
            return False
        return True

    def preferred_kinds(self):
        return tuple(
            kind for kind in ("silver", "black") if self.wants(kind))

    # -- percurso --------------------------------------------------------
    def on_entry_candidate(self):
        self._require(
            MissionState.FOLLOW_LINE, MissionState.ENTRY_SILVER_CANDIDATE)
        return self._go(MissionState.ENTRY_SILVER_CANDIDATE)

    def on_entry_rejected(self):
        self._require(
            MissionState.ENTRY_SILVER_CANDIDATE,
            MissionState.VERIFY_ENTRY_SILVER)
        return self._go(MissionState.FOLLOW_LINE)

    def on_entry_verifying(self):
        self._require(
            MissionState.ENTRY_SILVER_CANDIDATE,
            MissionState.VERIFY_ENTRY_SILVER)
        return self._go(MissionState.VERIFY_ENTRY_SILVER)

    def on_entry_confirmed(self):
        self._require(
            MissionState.ENTRY_SILVER_CANDIDATE,
            MissionState.VERIFY_ENTRY_SILVER)
        self.entry_count += 1
        return self._go(MissionState.ENTER_RESCUE_ZONE)

    def on_zone_entered(self):
        self._require(MissionState.ENTER_RESCUE_ZONE)
        return self._go(MissionState.STOP_AND_HANDOFF_TO_RESCUE)

    def on_rescue_started(self):
        self._require(MissionState.STOP_AND_HANDOFF_TO_RESCUE)
        return self._go(MissionState.RESCUE_SCAN)

    def on_false_entry(self):
        """A câmera de resgate abriu e não viu vítima nem triângulo.

        Diferente de ``on_entry_rejected``: lá o robô nem chegou a entrar; aqui
        ele entrou, olhou a sala e ela não existe. Volta ao percurso com o
        inventário intacto — nada foi resgatado e nada foi perdido.
        """
        self._require(MissionState.RESCUE_SCAN)
        self.false_entries += 1
        return self._go(MissionState.FOLLOW_LINE)

    # -- resgate ---------------------------------------------------------
    def on_target_locked(self, kind):
        self._require(
            MissionState.RESCUE_SCAN, MissionState.RESCUE_RECOVERY,
            MissionState.TARGET_BRAKE, MissionState.TARGET_VERIFY)
        if not self.wants(kind):
            raise MissionError(
                f"vítima {kind} recusada pela política/inventário atual")
        self.carrying = None
        self.pending_kind = kind
        return self._go(MissionState.TARGET_LOCK)

    def on_pickup_started(self, kind):
        self._require(
            MissionState.TARGET_LOCK, MissionState.ALIGN,
            MissionState.APPROACH)
        if not self.wants(kind):
            raise MissionError(
                f"coleta de vítima {kind} recusada pelo inventário")
        return self._go(MissionState.PICKUP)

    def on_victim_secured(self, kind):
        """Esfera presa e elevada: a cor congela até o depósito."""
        self._require(MissionState.PICKUP)
        self.carrying = kind
        return self._go(MissionState.CARRY_READY)

    def on_searching_triangle(self):
        self._require(
            MissionState.CARRY_READY, MissionState.FIND_CORRECT_TRIANGLE)
        return self._go(MissionState.FIND_CORRECT_TRIANGLE)

    def on_triangle_reached(self):
        self._require(
            MissionState.FIND_CORRECT_TRIANGLE,
            MissionState.APPROACH_TRIANGLE,
            MissionState.CARRY_READY)
        return self._go(MissionState.DEPOSIT)

    def on_grippers_restored(self):
        self._require(MissionState.DEPOSIT)
        return self._go(MissionState.RESTORE_GRIPPERS)

    def on_deposit_confirmed(self):
        """Só aqui o contador sobe: garras restauradas e sem falha serial."""
        self._require(MissionState.RESTORE_GRIPPERS)
        if self.carrying is None:
            raise MissionError(
                "depósito confirmado sem vítima presa registrada")
        self.inventory.record_deposit(self.carrying)
        self.carrying = None
        self.empty_sweeps = 0
        self._go(MissionState.UPDATE_INVENTORY)
        if self.inventory.complete:
            return self._go(MissionState.VERIFY_RESCUE_COMPLETE)
        return self._go(MissionState.RESCUE_SCAN)

    def on_empty_sweep(self):
        """Um giro completo sem encontrar vítima.

        Com o inventário incompleto, a primeira varredura vazia leva a uma
        recuperação (mais lenta, outro ponto de vista). Só a segunda encerra
        a sala — assim uma vítima perdida por iluminação tem segunda chance,
        e ao mesmo tempo o robô não fica preso em um laço infinito.
        """
        self._require(
            MissionState.RESCUE_SCAN, MissionState.RESCUE_RECOVERY,
            MissionState.TARGET_VERIFY)
        if self.inventory.complete:
            return self._go(MissionState.VERIFY_RESCUE_COMPLETE)
        self.empty_sweeps += 1
        if self.empty_sweeps < self.MAX_EMPTY_SWEEPS:
            return self._go(MissionState.RESCUE_RECOVERY)
        return self._go(MissionState.VERIFY_RESCUE_COMPLETE)

    def on_rescue_verified(self):
        self._require(MissionState.VERIFY_RESCUE_COMPLETE)
        return self._go(MissionState.DETECT_BOTH_TRIANGLES_FINAL)

    def on_final_triangles_mapped(self):
        self._require(MissionState.DETECT_BOTH_TRIANGLES_FINAL)
        return self._go(MissionState.FIND_BLACK_EXIT)

    def on_exit_confirmed(self):
        self._require(MissionState.FIND_BLACK_EXIT)
        return self._go(MissionState.CROSS_EXIT)

    def on_exit_crossed(self):
        self._require(MissionState.CROSS_EXIT)
        return self._go(MissionState.STOP_AND_HANDOFF_TO_LINE)

    def on_line_resumed(self):
        self._require(MissionState.STOP_AND_HANDOFF_TO_LINE)
        return self._go(MissionState.FOLLOW_LINE)

    def on_red_finish(self):
        self._require(MissionState.FOLLOW_LINE)
        return self._go(MissionState.RED_FINISH)

    def abort(self, reason=""):
        self.abort_reason = str(reason)
        return self._go(MissionState.ABORTED)


# ---------------------------------------------------------------------------
# Handoff entre segue-linha e resgate
# ---------------------------------------------------------------------------
# A ordem abaixo é o contrato de segurança da missão. Cada passo é o nome de
# um método do objeto "sistema" injetado no executor.
#
# Observação importante sobre o LED: o regulamento interno do Shadow manda
# apagá-lo antes do resgate, mas apagar o LED exige a serial. Por isso ele é
# apagado enquanto a serial do segue-linha ainda existe (único momento em que
# isso é possível) e o processo de resgate REAFIRMA o LED apagado assim que
# abre a própria serial. Os dois passos estão explícitos na lista.
HANDOFF_TO_RESCUE = (
    "stop_motors",              # PARAR antes de qualquer desmontagem
    "led_off",                  # ainda na serial do segue-linha
    "terminate_line_children",  # sinaliza visão e controle
    "join_line_children",       # espera de fato terminarem
    "assert_line_children_dead",
    "close_line_camera",        # câmera 1 liberada
    "release_serial",           # serial do segue-linha liberada
    "release_motor_lock",       # MotorOwnerLock liberado
    "acquire_rescue_motor_lock",
    "open_rescue_serial",
    "assert_led_off",           # reafirma na serial nova
    "open_rescue_camera",       # somente agora a câmera 0 abre
    "start_rescue",
)

# A ordem da volta espelha a ida. `open_line_camera` vem antes de
# `open_line_serial` porque é essa a ordem real de partida dos filhos do
# segue-linha (visão primeiro, controle depois) — e a ordem entre câmera e
# serial não é uma propriedade de segurança: as duas pertencem a donos
# diferentes. O que É propriedade de segurança, e está garantido, é a câmera
# de resgate fechar antes de a de linha abrir e a serial ter um dono só.
HANDOFF_TO_LINE = (
    "stop_motors",
    "close_rescue_camera",      # câmera 0 liberada
    "release_serial",
    "release_motor_lock",
    "acquire_line_motor_lock",
    "open_line_camera",         # câmera 1 reaberta
    "open_line_serial",
    "led_on",                   # volta ao percurso com LED aceso
    "reacquire_line",
)


class HandoffError(RuntimeError):
    pass


class HandoffExecutor:
    """Executa uma sequência de handoff contra um objeto de sistema.

    O executor é intencionalmente burro: ele apenas chama os passos na ordem
    e registra o que foi chamado. Toda a inteligência está na ordem declarada
    acima, que é o que os testes verificam. Se um passo falhar, ele para,
    tenta ``stop_motors`` e propaga — nunca continua abrindo a câmera
    seguinte com o processo anterior ainda vivo.
    """

    def __init__(self, system, steps):
        self.system = system
        self.steps = tuple(steps)
        self.log = []
        self.failed_step = None

    def run(self):
        for step in self.steps:
            action = getattr(self.system, step, None)
            if action is None:
                raise HandoffError(
                    f"o sistema não implementa o passo de handoff: {step}")
            try:
                action()
            except Exception as err:
                self.failed_step = step
                self._emergency_stop()
                raise HandoffError(
                    f"handoff abortado em '{step}': {err}") from err
            self.log.append(step)
        return tuple(self.log)

    def _emergency_stop(self):
        stop = getattr(self.system, "stop_motors", None)
        if stop is None:
            return
        try:
            stop()
            self.log.append("stop_motors")
        except Exception:
            # O watchdog de 1 s do Uno ainda corta os motores sozinho.
            pass


def index_of(log, step):
    """Posição de um passo no registro do handoff (-1 se não ocorreu)."""
    try:
        return list(log).index(step)
    except ValueError:
        return -1
