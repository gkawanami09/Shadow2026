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


def _ack_step(pickup, step, now):
    if step.futaba_action is not None:
        pickup.mark_futaba_started(now=now)
    if step.motor_action == "forward":
        pickup.mark_forward_started(now=now)
    if step.gripper_action is not None:
        pickup.mark_grippers_started(now=now)


def _run_sequence(target_kind):
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

    now += cfg.BALL_PICKUP_FORWARD_S
    close = pickup.update(now=now)
    actions.append(close)
    _ack_step(pickup, close, now)

    now += cfg.BALL_PICKUP_GRIPPER_SETTLE_S
    lift = pickup.update(now=now)
    actions.append(lift)
    _ack_step(pickup, lift, now)

    now += (
        cfg.BALL_PICKUP_LIFT_MS / 1000.0
        + cfg.BALL_PICKUP_LIFT_GUARD_S
    )
    carry = pickup.update(now=now)
    actions.append(carry)
    _ack_step(pickup, carry, now)
    if not pickup.resume_deposit():
        raise AssertionError("deposito nao foi liberado no estado de transporte")
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
        self.assertEqual(cfg.BALL_PICKUP_FORWARD_S, 1.5)
        self.assertEqual(
            cfg.BALL_PICKUP_FORWARD_LEAD_S,
            cfg.BALL_PICKUP_FORWARD_S,
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

    def test_no_gripper_action_before_full_forward_deadline(self):
        pickup = BallPickupSequencer()
        pickup.start("silver")
        first = pickup.update(now=0.0)
        pickup.mark_futaba_started(now=0.0)

        down_wait = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=down_wait)
        self.assertEqual(
            first.futaba_action,
            (cfg.BALL_PICKUP_FUTABA_POWER, cfg.BALL_PICKUP_FUTABA_MS),
        )
        self.assertEqual(forward.motor_action, "forward")
        self.assertIsNone(forward.gripper_action)
        pickup.mark_forward_started(now=down_wait)

        before = pickup.update(
            now=down_wait + cfg.BALL_PICKUP_FORWARD_S - 0.001)
        self.assertEqual(before.state, pickup.FORWARD_LEAD)
        self.assertIsNone(before.gripper_action)
        self.assertEqual(before.motor_action, "")

        close = pickup.update(
            now=down_wait + cfg.BALL_PICKUP_FORWARD_S)
        self.assertEqual(close.motor_action, "stop")
        self.assertEqual(
            close.gripper_action,
            (
                cfg.BALL_PICKUP_LEFT_DELTA,
                cfg.BALL_PICKUP_RIGHT_DELTA,
            ),
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
        now += cfg.BALL_PICKUP_FORWARD_S
        close = pickup.update(now=now)
        _ack_step(pickup, close, now)
        now += cfg.BALL_PICKUP_GRIPPER_SETTLE_S
        lift = pickup.update(now=now)
        _ack_step(pickup, lift, now)
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
            [
                (-50, 50),
                (50, 0),
                (0, 40),
                (0, -40),
                (0, 40),
                (0, -40),
                (0, -50),
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
            [
                (-50, 50),
                (0, -50),
                (-40, 0),
                (40, 0),
                (-40, 0),
                (40, 0),
                (50, 0),
            ],
        )
        self.assertTrue(complete.terminal)
        self.assertEqual(complete.state, pickup.COMPLETE)

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
        self.assertIsNotNone(first.futaba_action)
        pending = pickup.update(now=50.0)
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
        close = pickup.update(
            now=down_done + cfg.BALL_PICKUP_FORWARD_S)
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


# Testes da cola de orquestracao do resgate.py removidos: essa cola saiu do escopo atual (ver o docstring de resgate.py). Os modulos que ela orquestrava continuam no repositorio, com seus testes.


if __name__ == "__main__":
    unittest.main()
