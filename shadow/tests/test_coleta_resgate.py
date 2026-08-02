"""Testes da sequência de coleta das vítimas."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.coleta_resgate import (  # noqa: E402
    BallPickupSequencer,
    PickupStep,
)
from controle.aproximacao_resgate import (  # noqa: E402
    BallApproachController,
    MotionCommand,
)
from resgate import (  # noqa: E402
    _aplicar_acoes_coleta,
    _armar_coleta_confirmada,
)


def _ack_step(pickup, step, now):
    if step.futaba_action is not None:
        pickup.mark_futaba_started(now=now)
    if step.motor_action == "forward":
        pickup.mark_forward_started(now=now)
    if step.gripper_action is not None:
        pickup.mark_grippers_started(now=now)


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

    now += (
        cfg.BALL_PICKUP_LIFT_MS / 1000.0
        + cfg.BALL_PICKUP_LIFT_GUARD_S
    )
    carry = pickup.update(now=now)
    actions.append(carry)
    _ack_step(pickup, carry, now)
    retomou = (
        pickup.resume_selection()
        if selection else pickup.resume_deposit()
    )
    if not retomou:
        raise AssertionError("liberacao nao foi iniciada no transporte")
    lower = pickup.update(now=now)
    actions.append(lower)
    _ack_step(pickup, lower, now)

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
            (20, 2500),
        )
        self.assertEqual(
            (cfg.BALL_PICKUP_LOWER_POWER, cfg.BALL_PICKUP_LOWER_MS),
            (-20, 25),
        )
        self.assertEqual(cfg.BALL_PICKUP_WIGGLE_DELTA, 40)
        self.assertEqual(cfg.BALL_PICKUP_WIGGLE_REPETITIONS, 2)
        self.assertEqual(cfg.BALL_PICKUP_GRIPPER_STEP_DEGREES, 10)
        self.assertEqual(cfg.BALL_PICKUP_GRIPPER_STEP_INTERVAL_S, 0.08)

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
            sum(acao[0] for acao in movimentos),
            cfg.BALL_PICKUP_LEFT_DELTA,
        )
        self.assertEqual(
            sum(acao[1] for acao in movimentos),
            cfg.BALL_PICKUP_RIGHT_DELTA,
        )
        self.assertTrue(all(
            (acao[0] == 0) != (acao[1] == 0)
            for acao in movimentos
        ))

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
        now += (
            cfg.BALL_PICKUP_LIFT_MS / 1000.0
            + cfg.BALL_PICKUP_LIFT_GUARD_S
        )

        carry = pickup.update(now=now)
        self.assertEqual(carry.state, pickup.CARRY_READY)
        self.assertTrue(carry.stop_futaba)
        self.assertTrue(pickup.ready_for_deposit)
        self.assertIsNone(carry.futaba_action)
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
            [(-20, 1500), (20, 2500), (-20, 25)],
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
