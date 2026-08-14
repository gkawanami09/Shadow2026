"""Testes da sequência de coleta das vítimas."""

import inspect
import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
import resgate as modulo_resgate  # noqa: E402
from controle.coleta_resgate import (  # noqa: E402
    BallPickupSequencer,
    PickupStep,
)
from controle.aproximacao_resgate import (  # noqa: E402
    BallApproachController,
    MotionCommand,
)
from controle.parede_vitima import (  # noqa: E402
    PAREDE_RETA,
    WallProbeStep,
    WallTargetSignature,
)
from resgate import (  # noqa: E402
    _aplicar_acoes_coleta,
    _armar_coleta_confirmada,
    _armar_coleta_parede_direta,
    _deve_reiniciar_busca_por_alvo_perdido,
    _recuperar_coleta_apos_reinicio,
)
from visao.deteccao import VictimDetection  # noqa: E402


def _ack_step(pickup, step, now):
    if step.futaba_action is not None:
        pickup.mark_futaba_started(now=now)
    if step.motor_action == "forward":
        pickup.mark_forward_started(now=now)
    if (
        step.motor_action == "stop"
        and step.state == pickup.WALL_PAUSE_PENDING
    ):
        pickup.mark_wall_pause_started(now=now)
    if (
        step.motor_action == "stop"
        and step.state == pickup.WALL_POST_REVERSE_PENDING
    ):
        pickup.mark_post_reverse_pause_started(now=now)
    if step.motor_action == "reverse":
        pickup.mark_reverse_started(now=now)
    if step.gripper_action is not None:
        pickup.mark_grippers_started(now=now)


class _ColetaParada:
    started = False


class ReaproximacaoParedeTests(unittest.TestCase):
    def test_nao_abandona_reaproximacao_com_autorizacao_ativa(self):
        comando = MotionCommand(BallApproachController.WAIT_TARGET)

        reiniciar = _deve_reiniciar_busca_por_alvo_perdido(
            busca=None,
            controlador=object(),
            coleta=_ColetaParada(),
            comando=comando,
            autorizacao_parede=object(),
        )

        self.assertFalse(reiniciar)

    def test_reinicia_busca_sem_autorizacao_de_parede(self):
        comando = MotionCommand(BallApproachController.WAIT_TARGET)

        reiniciar = _deve_reiniciar_busca_por_alvo_perdido(
            busca=None,
            controlador=object(),
            coleta=_ColetaParada(),
            comando=comando,
            autorizacao_parede=None,
        )

        self.assertTrue(reiniciar)


def _terminar_fechamento_gradual(pickup, actions, now):
    """Executa todos os pequenos passos ate iniciar a subida do Futaba."""
    ultimo = None
    while pickup.state in (pickup.GRIPPERS_START, pickup.GRIPPERS_WAIT):
        if pickup.state == pickup.GRIPPERS_WAIT:
            now = pickup._deadline
        ultimo = pickup.update(now=now)
        actions.append(ultimo)
        _ack_step(pickup, ultimo, now)
    return now, ultimo


def _terminar_subida(pickup, actions):
    """Executa a fase lenta e chega ao transporte com o Futaba parado."""
    now = pickup._deadline
    slow = pickup.update(now=now)
    actions.append(slow)
    _ack_step(pickup, slow, now)

    now = pickup._deadline
    carry = pickup.update(now=now)
    actions.append(carry)
    _ack_step(pickup, carry, now)
    return now, carry


def _run_sequence(target_kind, selection=False):
    """Executa todos os deadlines e devolve somente passos com acao."""
    pickup = BallPickupSequencer()
    pickup.start(target_kind)
    now = 0.0
    actions = []

    initial_down = pickup.update(now=now)
    actions.append(initial_down)
    _ack_step(pickup, initial_down, now)

    now += (
        cfg.BALL_PICKUP_FUTABA_MS / 1000.0
        + cfg.BALL_PICKUP_FUTABA_GUARD_S
    )
    forward = pickup.update(now=now)
    actions.append(forward)
    _ack_step(pickup, forward, now)

    now += cfg.BALL_PICKUP_FORWARD_LEAD_S
    final_forward = pickup.update(now=now)
    actions.append(final_forward)

    now += cfg.BALL_PICKUP_FINAL_FORWARD_S
    close = pickup.update(now=now)
    actions.append(close)
    _ack_step(pickup, close, now)
    now, lift = _terminar_fechamento_gradual(
        pickup, actions, now)

    now, carry = _terminar_subida(pickup, actions)
    retomou = (
        pickup.resume_selection()
        if selection else pickup.resume_deposit()
    )
    if not retomou:
        raise AssertionError("liberacao nao foi iniciada no transporte")
    first_release_step = pickup.update(now=now)
    actions.append(first_release_step)
    _ack_step(pickup, first_release_step, now)

    if selection:
        release = first_release_step
    else:
        now += (
            cfg.BALL_PICKUP_LOWER_MS / 1000.0
            + cfg.BALL_PICKUP_LOWER_GUARD_S
        )
        release = pickup.update(now=now)
        actions.append(release)
        _ack_step(pickup, release, now)

    now += cfg.BALL_PICKUP_GRIPPER_SETTLE_S
    for _ in range(cfg.BALL_PICKUP_WIGGLE_REPETITIONS * 2):
        wiggle = pickup.update(now=now)
        actions.append(wiggle)
        _ack_step(pickup, wiggle, now)
        now += cfg.BALL_PICKUP_WIGGLE_STEP_S

    restore = pickup.update(now=now)
    actions.append(restore)
    _ack_step(pickup, restore, now)
    now += cfg.BALL_PICKUP_GRIPPER_SETTLE_S

    complete = pickup.update(now=now)
    return pickup, actions, complete


class BallPickupSequencerTests(unittest.TestCase):
    def test_requested_values_are_exact(self):
        self.assertFalse(hasattr(cfg, "BALL_PICKUP_REVERSE_S"))
        self.assertEqual(cfg.BALL_PICKUP_PRE_FORWARD_S, 1.0)
        self.assertEqual(cfg.BALL_PICKUP_FORWARD_S, 1.0)
        self.assertEqual(cfg.BALL_PICKUP_FINAL_FORWARD_S, 0.2)
        self.assertEqual(
            (cfg.BALL_PICKUP_LEFT_DELTA, cfg.BALL_PICKUP_RIGHT_DELTA),
            (-55, 55),
        )
        self.assertEqual(
            (
                cfg.BALL_PICKUP_INITIAL_LEFT_DELTA,
                cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA,
            ),
            (-10, 10),
        )
        self.assertEqual(cfg.BALL_PICKUP_RELEASE_DELTA, 70)
        self.assertEqual(
            cfg.BALL_PICKUP_FORWARD_LEAD_S,
            (
                cfg.BALL_PICKUP_PRE_FORWARD_S
                + cfg.BALL_PICKUP_FORWARD_S
            ),
        )
        self.assertEqual(
            (cfg.BALL_PICKUP_LIFT_POWER, cfg.BALL_PICKUP_LIFT_MS),
            (20, 1900),
        )
        self.assertEqual(
            (
                cfg.BALL_PICKUP_LIFT_SLOW_POWER,
                cfg.BALL_PICKUP_LIFT_SLOW_MS,
            ),
            (10, 400),
        )
        self.assertEqual(
            cfg.BALL_PICKUP_LIFT_MS + cfg.BALL_PICKUP_LIFT_SLOW_MS,
            2300,
        )
        self.assertEqual(
            (
                cfg.BALL_PICKUP_LIFT_HOLD_POWER,
                cfg.BALL_PICKUP_LIFT_HOLD_MS,
            ),
            (1, 300),
        )
        self.assertEqual(
            (cfg.BALL_PICKUP_LOWER_POWER, cfg.BALL_PICKUP_LOWER_MS),
            (-20, 25),
        )
        self.assertEqual(cfg.BALL_PICKUP_WIGGLE_DELTA, 40)
        self.assertEqual(cfg.BALL_PICKUP_WIGGLE_REPETITIONS, 2)
        self.assertEqual(cfg.BALL_PICKUP_GRIPPER_CAPTURE_DEGREES, 40)

    def test_proxima_coleta_prepara_garras_antes_de_abaixar(self):
        pickup = BallPickupSequencer(grippers_prepositioned=False)
        self.assertTrue(pickup.start("silver"))

        preparo = pickup.update(now=1.0)
        self.assertEqual(preparo.state, pickup.GRIPPERS_PREPARE_PENDING)
        self.assertEqual(
            preparo.gripper_action,
            (
                cfg.BALL_PICKUP_INITIAL_LEFT_DELTA,
                cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA,
            ),
        )

        pickup.mark_grippers_started(now=1.0)
        baixando = pickup.update(
            now=1.0 + cfg.BALL_PICKUP_GRIPPER_SETTLE_S)
        self.assertEqual(baixando.state, pickup.FUTABA_PENDING)
        self.assertEqual(
            baixando.futaba_action,
            (cfg.BALL_PICKUP_FUTABA_POWER, cfg.BALL_PICKUP_FUTABA_MS),
        )
        self.assertEqual(cfg.BALL_PICKUP_GRIPPER_CAPTURE_INTERVAL_S, 0.04)
        self.assertEqual(cfg.BALL_PICKUP_GRIPPER_STEP_DEGREES, 15)
        self.assertEqual(cfg.BALL_PICKUP_GRIPPER_STEP_INTERVAL_S, 0.05)

    def test_start_requires_confirmed_kind_and_never_changes_it(self):
        pickup = BallPickupSequencer()
        for invalid in (None, "any", "green"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    pickup.start(invalid)
                self.assertEqual(pickup.state, pickup.IDLE)

        self.assertTrue(pickup.start("silver"))
        self.assertEqual(pickup.target_kind, "silver")
        self.assertFalse(pickup.start("black"))
        self.assertEqual(pickup.target_kind, "silver")

    def test_baixa_antes_de_avancar_e_preserva_o_tempo_total(self):
        pickup = BallPickupSequencer()
        pickup.start("silver")
        down = pickup.update(now=0.0)
        self.assertEqual(down.motor_action, "hold")
        self.assertIsNotNone(down.futaba_action)
        pickup.mark_futaba_started(now=0.0)

        down_wait = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=down_wait)
        self.assertEqual(
            down.futaba_action,
            (cfg.BALL_PICKUP_FUTABA_POWER, cfg.BALL_PICKUP_FUTABA_MS),
        )
        self.assertEqual(forward.motor_action, "forward")
        self.assertIsNone(forward.gripper_action)
        pickup.mark_forward_started(now=down_wait)

        before = pickup.update(
            now=down_wait + cfg.BALL_PICKUP_FORWARD_LEAD_S - 0.001)
        self.assertEqual(before.state, pickup.FORWARD_LEAD)
        self.assertIsNone(before.gripper_action)
        self.assertEqual(before.motor_action, "")

        final_forward = pickup.update(
            now=down_wait + cfg.BALL_PICKUP_FORWARD_LEAD_S)
        self.assertEqual(final_forward.state, pickup.FINAL_FORWARD)
        self.assertIsNone(final_forward.gripper_action)

        before_close = pickup.update(
            now=(
                down_wait
                + cfg.BALL_PICKUP_FORWARD_LEAD_S
                + cfg.BALL_PICKUP_FINAL_FORWARD_S
                - 0.001
            )
        )
        self.assertEqual(before_close.state, pickup.FINAL_FORWARD)
        self.assertIsNone(before_close.gripper_action)

        close = pickup.update(
            now=(
                down_wait
                + cfg.BALL_PICKUP_FORWARD_LEAD_S
                + cfg.BALL_PICKUP_FINAL_FORWARD_S
            )
        )
        self.assertEqual(close.motor_action, "stop")
        movimentos = [close.gripper_action]
        pickup.mark_grippers_started(now=(
            down_wait
            + cfg.BALL_PICKUP_FORWARD_LEAD_S
            + cfg.BALL_PICKUP_FINAL_FORWARD_S
        ))
        _agora, _lift = _terminar_fechamento_gradual(
            pickup, [], pickup._deadline)
        movimentos = list(pickup._gripper_close_actions)
        self.assertEqual(
            movimentos[:2],
            [(-40, 0), (0, 40)],
        )
        self.assertEqual(len(movimentos), 4)
        self.assertEqual(
            sum(acao[0] for acao in movimentos),
            (
                cfg.BALL_PICKUP_LEFT_DELTA
                - cfg.BALL_PICKUP_INITIAL_LEFT_DELTA
            ),
        )
        self.assertEqual(
            sum(acao[1] for acao in movimentos),
            (
                cfg.BALL_PICKUP_RIGHT_DELTA
                - cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA
            ),
        )
        self.assertTrue(all(
            (acao[0] == 0) != (acao[1] == 0)
            for acao in movimentos
        ))

    def test_modo_parede_avanca_para_re_e_so_entao_fecha(self):
        pickup = BallPickupSequencer()
        self.assertTrue(pickup.start("silver", wall_mode=True))
        now = 0.0

        down = pickup.update(now=now)
        self.assertEqual(down.motor_action, "hold")
        _ack_step(pickup, down, now)

        now += (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=now)
        self.assertEqual(forward.motor_action, "forward")
        self.assertIsNone(forward.gripper_action)
        _ack_step(pickup, forward, now)

        before_final = pickup.update(
            now=now + cfg.BALL_WALL_PICKUP_FORWARD_S - 0.001)
        self.assertEqual(before_final.state, pickup.FORWARD_LEAD)
        self.assertEqual(before_final.motor_action, "")

        now += cfg.BALL_WALL_PICKUP_FORWARD_S
        final_forward = pickup.update(now=now)
        self.assertEqual(final_forward.state, pickup.FINAL_FORWARD)
        self.assertIsNone(final_forward.gripper_action)

        before_pause = pickup.update(
            now=now + cfg.BALL_PICKUP_FINAL_FORWARD_S - 0.001)
        self.assertEqual(before_pause.state, pickup.FINAL_FORWARD)

        now += cfg.BALL_PICKUP_FINAL_FORWARD_S
        pause = pickup.update(now=now)
        self.assertEqual(pause.state, pickup.WALL_PAUSE_PENDING)
        self.assertEqual(pause.motor_action, "stop")
        self.assertIsNone(pause.gripper_action)
        _ack_step(pickup, pause, now)

        before_reverse = pickup.update(
            now=(
                now
                + cfg.BALL_WALL_PICKUP_DIRECTION_CHANGE_PAUSE_S
                - 0.001
            )
        )
        self.assertEqual(before_reverse.state, pickup.WALL_PAUSE_WAIT)

        now += cfg.BALL_WALL_PICKUP_DIRECTION_CHANGE_PAUSE_S
        reverse = pickup.update(now=now)
        self.assertEqual(reverse.state, pickup.WALL_REVERSE_PENDING)
        self.assertEqual(reverse.motor_action, "reverse")
        self.assertEqual(reverse.angle, 200)
        self.assertEqual(reverse.speed, cfg.BALL_WALL_PICKUP_REVERSE_SPEED)
        self.assertIsNone(reverse.gripper_action)
        _ack_step(pickup, reverse, now)

        before_close = pickup.update(
            now=now + cfg.BALL_WALL_PICKUP_REVERSE_S - 0.001)
        self.assertEqual(before_close.state, pickup.WALL_REVERSE_WAIT)
        self.assertIsNone(before_close.gripper_action)

        now += cfg.BALL_WALL_PICKUP_REVERSE_S
        post_reverse_stop = pickup.update(now=now)
        self.assertEqual(
            post_reverse_stop.state,
            pickup.WALL_POST_REVERSE_PENDING,
        )
        self.assertEqual(post_reverse_stop.motor_action, "stop")
        self.assertIsNone(post_reverse_stop.gripper_action)
        pending_stop = pickup.update(now=now + 10.0)
        self.assertEqual(
            pending_stop.state,
            pickup.WALL_POST_REVERSE_PENDING,
        )
        self.assertEqual(pending_stop.motor_action, "")
        self.assertIsNone(pending_stop.gripper_action)
        _ack_step(pickup, post_reverse_stop, now)

        before_close = pickup.update(
            now=(
                now
                + cfg.BALL_WALL_PICKUP_POST_REVERSE_PAUSE_S
                - 0.001
            )
        )
        self.assertEqual(
            before_close.state,
            pickup.WALL_POST_REVERSE_WAIT,
        )
        self.assertEqual(before_close.motor_action, "")
        self.assertIsNone(before_close.gripper_action)

        now += cfg.BALL_WALL_PICKUP_POST_REVERSE_PAUSE_S
        close = pickup.update(now=now)
        self.assertEqual(close.state, pickup.GRIPPERS_START)
        self.assertEqual(close.motor_action, "")
        self.assertEqual(
            close.gripper_action,
            (-cfg.BALL_PICKUP_GRIPPER_CAPTURE_DEGREES, 0),
        )
        actions = [close]
        _ack_step(pickup, close, now)
        now, _lift = _terminar_fechamento_gradual(
            pickup, actions, now)
        now, carry = _terminar_subida(pickup, actions)

        self.assertEqual(carry.state, pickup.CARRY_READY)
        self.assertEqual(
            [step.gripper_action for step in actions
             if step.gripper_action is not None],
            list(pickup._gripper_close_actions),
        )
        self.assertTrue(pickup.resume_selection())
        release = pickup.update(now=now)
        self.assertEqual(release.gripper_action, (70, 0))

    def test_modo_normal_nao_adiciona_pausa_nem_re(self):
        pickup = BallPickupSequencer()
        pickup.start("black")
        now = 0.0

        down = pickup.update(now=now)
        _ack_step(pickup, down, now)
        now += (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=now)
        _ack_step(pickup, forward, now)
        now += cfg.BALL_PICKUP_FORWARD_LEAD_S
        pickup.update(now=now)
        now += cfg.BALL_PICKUP_FINAL_FORWARD_S

        close = pickup.update(now=now)

        self.assertEqual(close.state, pickup.GRIPPERS_START)
        self.assertEqual(close.motor_action, "stop")
        self.assertEqual(
            close.gripper_action,
            (-cfg.BALL_PICKUP_GRIPPER_CAPTURE_DEGREES, 0),
        )

    def test_release_is_blocked_while_carrying_until_marker_arrival(self):
        pickup = BallPickupSequencer()
        pickup.start("silver")
        now = 0.0

        down = pickup.update(now=now)
        _ack_step(pickup, down, now)
        now += (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=now)
        _ack_step(pickup, forward, now)
        now += cfg.BALL_PICKUP_FORWARD_LEAD_S
        pickup.update(now=now)
        now += cfg.BALL_PICKUP_FINAL_FORWARD_S
        close = pickup.update(now=now)
        _ack_step(pickup, close, now)
        now, lift = _terminar_fechamento_gradual(
            pickup, [], now)
        now, carry = _terminar_subida(pickup, [])
        self.assertEqual(carry.state, pickup.CARRY_READY)
        self.assertFalse(carry.stop_futaba)
        self.assertTrue(pickup.ready_for_deposit)
        self.assertEqual(
            carry.futaba_action,
            (
                cfg.BALL_PICKUP_LIFT_HOLD_POWER,
                cfg.BALL_PICKUP_LIFT_HOLD_MS,
            ),
        )
        self.assertIsNone(carry.gripper_action)

        for later in (now + 1.0, now + 30.0):
            waiting = pickup.update(now=later)
            self.assertEqual(waiting.state, pickup.CARRY_READY)
            self.assertIsNone(waiting.futaba_action)
            self.assertIsNone(waiting.gripper_action)

        self.assertTrue(pickup.resume_deposit())
        self.assertFalse(pickup.resume_deposit())
        lower = pickup.update(now=now + 30.0)
        self.assertEqual(
            lower.futaba_action,
            (
                cfg.BALL_PICKUP_LOWER_POWER,
                cfg.BALL_PICKUP_LOWER_MS,
            ),
        )

    def test_silver_sequence_opens_left_then_wiggles_right_twice(self):
        pickup, actions, complete = _run_sequence("silver")

        self.assertEqual(
            [step.gripper_action for step in actions
            if step.gripper_action is not None],
            list(pickup._gripper_close_actions) + [
                (70, 0),
                (0, 40),
                (0, -40),
                (0, 40),
                (0, -40),
                (0, -cfg.BALL_PICKUP_RIGHT_DELTA),
            ],
        )
        self.assertEqual(
            [step.futaba_action for step in actions
             if step.futaba_action is not None],
            [
                (-20, 1500),
                (20, 1900),
                (10, 400),
                (1, 300),
                (-20, 25),
            ],
        )
        self.assertTrue(complete.terminal)
        self.assertEqual(complete.state, pickup.COMPLETE)

    def test_black_sequence_opens_right_then_wiggles_left_twice(self):
        pickup, actions, complete = _run_sequence("black")

        self.assertEqual(
            [step.gripper_action for step in actions
            if step.gripper_action is not None],
            list(pickup._gripper_close_actions) + [
                (0, -70),
                (-40, 0),
                (40, 0),
                (-40, 0),
                (40, 0),
                (-cfg.BALL_PICKUP_LEFT_DELTA, 0),
            ],
        )
        self.assertTrue(complete.terminal)
        self.assertEqual(complete.state, pickup.COMPLETE)

    def test_selecao_usa_esquerda_para_prata_e_direita_para_preta(self):
        for kind, expected_release in (
            ("silver", (70, 0)),
            ("black", (0, -70)),
        ):
            with self.subTest(kind=kind):
                pickup, actions, complete = _run_sequence(
                    kind, selection=True)
                grippers = [
                    step.gripper_action for step in actions
                    if step.gripper_action is not None
                ]
                self.assertEqual(
                    grippers[len(pickup._gripper_close_actions)],
                    expected_release,
                )
                self.assertNotIn(
                    (
                        cfg.BALL_PICKUP_LOWER_POWER,
                        cfg.BALL_PICKUP_LOWER_MS,
                    ),
                    [
                        step.futaba_action for step in actions
                        if step.futaba_action is not None
                    ],
                )
                self.assertEqual(complete.state, pickup.COMPLETE)
                self.assertIn("selecao", complete.detail)

    def test_both_colors_restore_exact_initial_gripper_positions(self):
        for kind in ("silver", "black"):
            with self.subTest(kind=kind):
                positions = [180, 0]
                for _cycle in range(2):
                    _pickup, actions, _complete = _run_sequence(kind)
                    for step in actions:
                        if step.gripper_action is None:
                            continue
                        for index, delta in enumerate(
                            step.gripper_action
                        ):
                            positions[index] = min(
                                180,
                                max(0, positions[index] + delta),
                            )
                    self.assertEqual(positions, [180, 0])

    def test_each_serial_action_is_one_shot_until_acknowledged(self):
        pickup = BallPickupSequencer()
        pickup.start("silver")

        first = pickup.update(now=0.0)
        self.assertEqual(first.motor_action, "hold")
        self.assertIsNotNone(first.futaba_action)
        pending = pickup.update(now=50.0)
        self.assertEqual(pending.motor_action, "")
        self.assertIsNone(pending.futaba_action)
        self.assertIsNone(pending.gripper_action)

        pickup.mark_futaba_started(now=50.0)
        down_done = (
            50.0
            + cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=down_done)
        pickup.mark_forward_started(now=down_done)
        final_forward = pickup.update(
            now=down_done + cfg.BALL_PICKUP_FORWARD_LEAD_S)
        self.assertEqual(final_forward.state, pickup.FINAL_FORWARD)
        close = pickup.update(
            now=(
                down_done
                + cfg.BALL_PICKUP_FORWARD_LEAD_S
                + cfg.BALL_PICKUP_FINAL_FORWARD_S
            )
        )
        self.assertIsNotNone(close.gripper_action)
        pending_close = pickup.update(now=999.0)
        self.assertIsNone(pending_close.gripper_action)

    def test_failure_is_terminal_and_stops_motors_and_futaba(self):
        pickup = BallPickupSequencer()
        pickup.start("black")
        pickup.update(now=0.0)

        fault = pickup.fail("serial ausente")
        self.assertEqual(fault.state, pickup.FAULT)
        self.assertEqual(fault.motor_action, "stop")
        self.assertTrue(fault.stop_futaba)
        self.assertIsNone(fault.gripper_action)
        self.assertTrue(fault.terminal)
        self.assertFalse(pickup.start("silver"))

    def test_perfil_de_recuperacao_completa_so_o_tempo_restante(self):
        pickup = BallPickupSequencer()
        pickup.start("silver")
        pickup.state = pickup.LIFT_WAIT
        pickup._deadline = 1.20

        self.assertEqual(
            pickup.recovery_lift_profile(now=0.70),
            (500, cfg.BALL_PICKUP_LIFT_SLOW_MS),
        )

        pickup.state = pickup.LIFT_SLOW_WAIT
        pickup._deadline = 1.00
        self.assertEqual(
            pickup.recovery_lift_profile(now=0.75),
            (0, 250),
        )


class PickupSerialRecoveryTests(unittest.TestCase):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += max(float(seconds), 0.0)

    class FakeArduino:
        def __init__(self, clock, reconnect=True, reset_at=None):
            self.clock = clock
            self.connected = False
            self.connection_epoch = 0
            self.reconnect = reconnect
            self.reset_at = reset_at
            self.reset_done = False
            self.calls = []

        def refresh(self, fail_closed=False):
            self.calls.append(("refresh", bool(fail_closed)))
            if not self.connected and self.reconnect:
                self.connected = True
                self.connection_epoch += 1
            if (
                self.connected
                and self.reset_at is not None
                and not self.reset_done
                and self.clock.now >= self.reset_at
            ):
                self.connection_epoch += 1
                self.reset_done = True

        def lado(self, esquerda, direita):
            self.calls.append(("lado", esquerda, direita))
            return self.connected

        def futaba(self, potencia, tempo_ms):
            self.calls.append(("futaba", potencia, tempo_ms))
            return self.connected

        def parar_futaba(self):
            self.calls.append(("parar_futaba",))
            return self.connected

    @staticmethod
    def coleta_em_andamento(tipo="silver"):
        coleta = BallPickupSequencer()
        coleta.start(tipo, wall_mode=True)
        return coleta

    def test_reconecta_sobe_e_reinicia_sempre_em_modo_normal(self):
        clock = self.FakeClock()
        arduino = self.FakeArduino(clock)

        nova, epoca = _recuperar_coleta_apos_reinicio(
            self.coleta_em_andamento(),
            arduino,
            tentativa=1,
            relogio=clock.monotonic,
            dormir=clock.sleep,
        )

        self.assertEqual(epoca, arduino.connection_epoch)
        self.assertTrue(nova.started)
        self.assertEqual(nova.target_kind, "silver")
        self.assertFalse(nova._wall_mode)
        self.assertEqual(nova.state, nova.GRIPPERS_PREPARE_PENDING)
        self.assertEqual(
            nova.update(now=clock.monotonic()).gripper_action,
            (
                cfg.BALL_PICKUP_INITIAL_LEFT_DELTA,
                cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA,
            ),
        )
        self.assertIn(("lado", 0, 0), arduino.calls)
        self.assertIn((
            "futaba",
            cfg.BALL_PICKUP_LIFT_POWER,
            cfg.BALL_PICKUP_LIFT_MS,
        ), arduino.calls)
        self.assertIn((
            "futaba",
            cfg.BALL_PICKUP_LIFT_SLOW_POWER,
            cfg.BALL_PICKUP_LIFT_SLOW_MS,
        ), arduino.calls)
        self.assertEqual(arduino.calls[-1], ("parar_futaba",))

    def test_reset_durante_subida_completa_apenas_o_tempo_restante(self):
        clock = self.FakeClock()
        arduino = self.FakeArduino(clock, reset_at=0.50)

        nova, _epoca = _recuperar_coleta_apos_reinicio(
            self.coleta_em_andamento("black"),
            arduino,
            tentativa=1,
            relogio=clock.monotonic,
            dormir=clock.sleep,
        )

        pulsos_normais = [
            chamada[2] for chamada in arduino.calls
            if chamada[:2] == ("futaba", cfg.BALL_PICKUP_LIFT_POWER)
        ]
        self.assertEqual(len(pulsos_normais), 2)
        self.assertEqual(pulsos_normais[0], cfg.BALL_PICKUP_LIFT_MS)
        self.assertAlmostEqual(
            pulsos_normais[1],
            cfg.BALL_PICKUP_LIFT_MS - 500,
            delta=int(cfg.BALL_PICKUP_SERIAL_RECOVERY_POLL_S * 1000) + 1,
        )
        self.assertEqual(nova.target_kind, "black")
        self.assertFalse(nova._wall_mode)

    def test_sem_reconexao_falha_depois_do_timeout_limitado(self):
        clock = self.FakeClock()
        arduino = self.FakeArduino(clock, reconnect=False)

        with self.assertRaisesRegex(RuntimeError, "nao reconectou"):
            _recuperar_coleta_apos_reinicio(
                self.coleta_em_andamento(),
                arduino,
                tentativa=1,
                relogio=clock.monotonic,
                dormir=clock.sleep,
            )

    def test_limite_de_recuperacoes_bloqueia_nova_tentativa(self):
        clock = self.FakeClock()
        arduino = self.FakeArduino(clock)

        with self.assertRaisesRegex(RuntimeError, "limite"):
            _recuperar_coleta_apos_reinicio(
                self.coleta_em_andamento(),
                arduino,
                tentativa=cfg.BALL_PICKUP_SERIAL_RECOVERY_MAX_RETRIES + 1,
                relogio=clock.monotonic,
                dormir=clock.sleep,
            )
        self.assertEqual(arduino.calls, [])

    def test_main_nao_transforma_reinicio_da_coleta_em_terminal(self):
        fonte = inspect.getsource(modulo_resgate.main)
        self.assertIn("recuperar_coleta_serial", fonte)
        self.assertNotIn(
            '"serial mudou durante a coleta; sequencia cancelada"', fonte)

    def test_modo_especial_de_parede_esta_desativado(self):
        self.assertFalse(cfg.BALL_WALL_TEST_ENABLED)


class PickupActionApplicationTests(unittest.TestCase):
    class FakeArduino:
        def __init__(self):
            self.calls = []
            self.connected = True
            self.connection_epoch = 7

        def lado(self, esquerda, direita):
            self.calls.append(("lado", esquerda, direita))
            return True

        def futaba(self, potencia, tempo_ms):
            self.calls.append(("futaba", potencia, tempo_ms))
            return True

        def parar_futaba(self):
            self.calls.append(("parar_futaba",))
            return True

        def garras(self, esquerda, direita):
            self.calls.append(("garras", esquerda, direita))
            return True

    @staticmethod
    def gravar_direcao(chamadas):
        def direcao(angulo=190, velocidade=0.8):
            chamadas.append(("direcao", angulo, velocidade))
            return True
        return direcao

    def test_descida_comeca_com_rodas_zeradas(self):
        arduino = self.FakeArduino()
        passo = PickupStep(
            "DESCER",
            "baixando",
            motor_action="hold",
            futaba_action=(-20, 1500),
        )

        erro = _aplicar_acoes_coleta(
            passo,
            arduino,
            self.gravar_direcao(arduino.calls),
            epoca_serial_esperada=7,
        )

        self.assertIsNone(erro)
        self.assertEqual(
            arduino.calls,
            [("lado", 0, 0), ("futaba", -20, 1500)],
        )

    def test_parar_acontece_antes_de_fechar_as_garras(self):
        arduino = self.FakeArduino()
        passo = PickupStep(
            "FECHAR",
            "fechando",
            motor_action="stop",
            gripper_action=(-50, 50),
        )

        erro = _aplicar_acoes_coleta(
            passo,
            arduino,
            self.gravar_direcao(arduino.calls),
            epoca_serial_esperada=7,
        )

        self.assertIsNone(erro)
        self.assertEqual(
            arduino.calls,
            [("direcao", 190, 0.8), ("garras", -50, 50)],
        )

    def test_reconexao_bloqueia_a_proxima_acao(self):
        arduino = self.FakeArduino()
        arduino.connection_epoch = 8
        passo = PickupStep(
            "AVANCAR",
            "avancando",
            angle=0,
            speed=cfg.BALL_PICKUP_FORWARD_SPEED,
            motor_action="forward",
        )

        erro = _aplicar_acoes_coleta(
            passo,
            arduino,
            self.gravar_direcao(arduino.calls),
            epoca_serial_esperada=7,
        )

        self.assertIn("serial mudou", erro)
        self.assertEqual(arduino.calls, [])

    def test_re_da_parede_usa_o_angulo_e_a_velocidade_do_passo(self):
        arduino = self.FakeArduino()
        passo = PickupStep(
            "RE_PAREDE",
            "afastando antes de fechar",
            angle=200,
            speed=cfg.BALL_WALL_PICKUP_REVERSE_SPEED,
            motor_action="reverse",
        )

        erro = _aplicar_acoes_coleta(
            passo,
            arduino,
            self.gravar_direcao(arduino.calls),
            epoca_serial_esperada=7,
        )

        self.assertIsNone(erro)
        self.assertEqual(
            arduino.calls,
            [("direcao", 200, cfg.BALL_WALL_PICKUP_REVERSE_SPEED)],
        )


class PickupHandoffTests(unittest.TestCase):
    class FakeArduino:
        connected = True
        connection_epoch = 4

    @staticmethod
    def comando_proximo(tipo="silver"):
        return MotionCommand(
            BallApproachController.NEAR,
            detail="vitima na posicao de coleta",
            terminal=True,
            pickup_in_range=True,
            pickup_confirmations=cfg.BALL_STOP_CONFIRM_FRAMES,
            target_kind=tipo,
        )

    def test_parar_estavel_arma_a_coleta_e_congela_a_cor(self):
        coleta = BallPickupSequencer()

        iniciou = _armar_coleta_confirmada(
            self.comando_proximo("black"),
            coleta,
            self.FakeArduino(),
            parada_enviada=True,
            epoca_movimento=4,
        )

        self.assertTrue(iniciou)
        self.assertTrue(coleta.started)
        self.assertEqual(coleta.target_kind, "black")

    def test_handoff_pode_armar_somente_a_coleta_especial_de_parede(self):
        coleta = BallPickupSequencer()

        iniciou = _armar_coleta_confirmada(
            self.comando_proximo("silver"),
            coleta,
            self.FakeArduino(),
            parada_enviada=True,
            epoca_movimento=4,
            modo_parede=True,
        )

        self.assertTrue(iniciou)
        self.assertTrue(coleta._wall_mode)
        primeiro = coleta.update(now=0.0)
        self.assertEqual(primeiro.state, coleta.FUTABA_PENDING)
        self.assertIsNotNone(primeiro.futaba_action)

    @staticmethod
    def deteccao_parede(
        centro_x=320,
        centro_y=300,
        raio=55,
        instante=1.0,
    ):
        return VictimDetection(
            "silver",
            center_x=centro_x,
            center_y=centro_y,
            radius=raio,
            confidence=0.95,
            confirmed=True,
            hits=5,
            timestamp=instante,
            track_locked=True,
        )

    @staticmethod
    def passo_parede():
        return WallProbeStep(
            "WALL_COMPLETE",
            "parede reta e alvo central confirmados",
            motor_action="stop",
            terminal=True,
            result=PAREDE_RETA,
            target_kind="silver",
        )

    def test_handoff_direto_preserva_yaw_e_arma_wall_mode(self):
        coleta = BallPickupSequencer()
        alvo = self.deteccao_parede()

        iniciou = _armar_coleta_parede_direta(
            self.passo_parede(),
            coleta,
            self.FakeArduino(),
            parada_enviada=True,
            epoca_movimento=4,
            epoca_parede=4,
            deteccao=alvo,
            frame_shape=(480, 640, 3),
            assinatura=WallTargetSignature.from_detection(alvo),
            agora=1.1,
        )

        self.assertTrue(iniciou)
        self.assertTrue(coleta.started)
        self.assertTrue(coleta._wall_mode)
        primeiro = coleta.update(now=0.0)
        self.assertEqual(primeiro.state, coleta.FUTABA_PENDING)
        self.assertIsNotNone(primeiro.futaba_action)
        # HOLD envia rodas zeradas sem cancelar o pulso temporizado do Futaba.
        self.assertEqual(primeiro.motor_action, "hold")

    def test_handoff_direto_recusa_epoca_antiga_ou_alvo_fora_do_centro(self):
        alvo = self.deteccao_parede()
        assinatura = WallTargetSignature.from_detection(alvo)
        argumentos = dict(
            passo_parede=self.passo_parede(),
            arduino=self.FakeArduino(),
            parada_enviada=True,
            epoca_movimento=4,
            deteccao=alvo,
            frame_shape=(480, 640, 3),
            assinatura=assinatura,
            agora=1.1,
        )

        with self.assertRaisesRegex(RuntimeError, "serial estavel"):
            _armar_coleta_parede_direta(
                coleta=BallPickupSequencer(),
                epoca_parede=3,
                **argumentos,
            )

        with self.assertRaisesRegex(RuntimeError, "reconfirmada"):
            _armar_coleta_parede_direta(
                coleta=BallPickupSequencer(),
                epoca_parede=4,
                **{
                    **argumentos,
                    "deteccao": self.deteccao_parede(centro_x=370),
                },
            )

        with self.assertRaisesRegex(RuntimeError, "reconfirmada"):
            _armar_coleta_parede_direta(
                coleta=BallPickupSequencer(),
                epoca_parede=4,
                **{
                    **argumentos,
                    "deteccao": self.deteccao_parede(
                        centro_y=235, raio=36),
                },
            )

        alvo_truncado = self.deteccao_parede()
        alvo_truncado = VictimDetection(
            alvo_truncado.kind,
            center_x=alvo_truncado.center_x,
            center_y=alvo_truncado.center_y,
            radius=alvo_truncado.radius,
            confidence=alvo_truncado.confidence,
            confirmed=True,
            hits=alvo_truncado.hits,
            timestamp=alvo_truncado.timestamp,
            track_locked=True,
            truncated=True,
        )
        with self.assertRaisesRegex(RuntimeError, "reconfirmada"):
            _armar_coleta_parede_direta(
                coleta=BallPickupSequencer(),
                epoca_parede=4,
                **{
                    **argumentos,
                    "deteccao": alvo_truncado,
                },
            )

    def test_main_nao_recria_aproximacao_depois_de_parede_reta(self):
        fonte = inspect.getsource(modulo_resgate.main)
        ramo_parede = fonte.split(
            "passo_parede.result == PAREDE_RETA", 1)[1].split(
                "elif (", 1)[0]

        self.assertIn("_armar_coleta_parede_direta", ramo_parede)
        self.assertNotIn("BallApproachController(", ramo_parede)

    def test_coleta_recusa_parar_sem_confirmacao_serial(self):
        coleta = BallPickupSequencer()

        with self.assertRaisesRegex(RuntimeError, "PARAR"):
            _armar_coleta_confirmada(
                self.comando_proximo(),
                coleta,
                self.FakeArduino(),
                parada_enviada=False,
                epoca_movimento=4,
            )

        self.assertFalse(coleta.started)

    def test_coleta_recusa_proximidade_visual_incompleta(self):
        coleta = BallPickupSequencer()
        comando = MotionCommand(
            BallApproachController.NEAR,
            terminal=True,
            pickup_in_range=True,
            pickup_confirmations=cfg.BALL_STOP_CONFIRM_FRAMES - 1,
            target_kind="silver",
        )

        with self.assertRaisesRegex(RuntimeError, "visual"):
            _armar_coleta_confirmada(
                comando,
                coleta,
                self.FakeArduino(),
                parada_enviada=True,
                epoca_movimento=4,
            )

        self.assertFalse(coleta.started)


if __name__ == "__main__":
    unittest.main()
