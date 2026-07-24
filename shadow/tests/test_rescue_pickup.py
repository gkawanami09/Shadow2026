import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg  # noqa: E402
from control.rescue_pickup import (  # noqa: E402
    BallPickupSequencer,
    PickupStep,
)
from rescue_main import _apply_pickup_actions  # noqa: E402


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
            (20, 2000),
        )
        self.assertEqual(
            (cfg.BALL_PICKUP_LOWER_POWER, cfg.BALL_PICKUP_LOWER_MS),
            (-20, 25),
        )
        self.assertEqual(cfg.BALL_PICKUP_WIGGLE_DELTA, 10)
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

    def test_silver_sequence_opens_left_then_wiggles_right_twice(self):
        pickup, actions, complete = _run_sequence("silver")

        self.assertEqual(
            [step.gripper_action for step in actions
             if step.gripper_action is not None],
            [
                (-50, 50),
                (50, 0),
                (0, 10),
                (0, -10),
                (0, 10),
                (0, -10),
            ],
        )
        self.assertEqual(
            [step.futaba_action for step in actions
             if step.futaba_action is not None],
            [(-20, 1500), (20, 2000), (-20, 25)],
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
                (-10, 0),
                (10, 0),
                (-10, 0),
                (10, 0),
            ],
        )
        self.assertTrue(complete.terminal)
        self.assertEqual(complete.state, pickup.COMPLETE)

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


class PickupActionApplicationTests(unittest.TestCase):
    class FakeArduino:
        def __init__(self):
            self.calls = []
            self.connected = True
            self.connection_epoch = 7

        def lado(self, left, right):
            self.calls.append(("lado", left, right))
            return True

        def futaba(self, power, duration_ms):
            self.calls.append(("futaba", power, duration_ms))
            return True

        def parar_futaba(self):
            self.calls.append(("parar_futaba",))
            return True

        def garras(self, left, right):
            self.calls.append(("garras", left, right))
            return True

    @staticmethod
    def steer_recorder(calls):
        def steer(angle=190, speed=0.8):
            calls.append(("steer", angle, speed))
            return True
        return steer

    def test_lift_replaces_parar_keepalive_with_lado_zero(self):
        arduino = self.FakeArduino()
        step = PickupStep(
            "LIFT",
            "subindo",
            motor_action="hold",
            futaba_action=(20, 2000),
        )

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIsNone(error)
        self.assertEqual(
            arduino.calls,
            [("lado", 0, 0), ("futaba", 20, 2000)],
        )

    def test_previous_futaba_is_stopped_before_25ms_pulse(self):
        arduino = self.FakeArduino()
        step = PickupStep(
            "LOWER",
            "pulso",
            stop_futaba=True,
            futaba_action=(-20, 25),
        )

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIsNone(error)
        self.assertEqual(
            arduino.calls,
            [("parar_futaba",), ("futaba", -20, 25)],
        )

    def test_motor_stop_precedes_close(self):
        arduino = self.FakeArduino()
        step = PickupStep(
            "CLOSE",
            "fechando",
            motor_action="stop",
            gripper_action=(-50, 50),
        )

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIsNone(error)
        self.assertEqual(
            arduino.calls,
            [("steer", 190, 0.8), ("garras", -50, 50)],
        )

    def test_full_physical_command_order_for_both_colors(self):
        for kind, release in (
            (
                "silver",
                [
                    ("garras", 50, 0),
                    ("garras", 0, 10),
                    ("garras", 0, -10),
                    ("garras", 0, 10),
                    ("garras", 0, -10),
                ],
            ),
            (
                "black",
                [
                    ("garras", 0, -50),
                    ("garras", -10, 0),
                    ("garras", 10, 0),
                    ("garras", -10, 0),
                    ("garras", 10, 0),
                ],
            ),
        ):
            with self.subTest(kind=kind):
                _pickup, actions, _complete = _run_sequence(kind)
                arduino = self.FakeArduino()
                steer = self.steer_recorder(arduino.calls)
                for step in actions:
                    self.assertIsNone(
                        _apply_pickup_actions(step, arduino, steer))

                self.assertEqual(
                    arduino.calls,
                    [
                        ("lado", 0, 0),
                        ("futaba", -20, 1500),
                        ("parar_futaba",),
                        ("steer", 0, cfg.BALL_PICKUP_FORWARD_SPEED),
                        ("steer", 190, 0.8),
                        ("garras", -50, 50),
                        ("lado", 0, 0),
                        ("futaba", 20, 2000),
                        ("parar_futaba",),
                        ("futaba", -20, 25),
                        ("parar_futaba",),
                    ] + release,
                )

    def test_failed_gripper_write_remains_stopped(self):
        arduino = self.FakeArduino()
        steer = self.steer_recorder(arduino.calls)

        def fail_grippers(left, right):
            arduino.calls.append(("garras", left, right))
            return False

        arduino.garras = fail_grippers
        step = PickupStep(
            "CLOSE",
            "fechando",
            motor_action="stop",
            gripper_action=(-50, 50),
        )
        error = _apply_pickup_actions(step, arduino, steer)

        self.assertIn("garras", error)
        self.assertEqual(
            arduino.calls,
            [("steer", 190, 0.8), ("garras", -50, 50)],
        )

    def test_reconnect_after_futaba_stop_blocks_new_pulse(self):
        arduino = self.FakeArduino()

        def reconnecting_stop():
            arduino.calls.append(("parar_futaba",))
            arduino.connection_epoch += 1
            return True

        arduino.parar_futaba = reconnecting_stop
        step = PickupStep(
            "LOWER",
            "pulso",
            stop_futaba=True,
            futaba_action=(-20, 25),
        )
        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
            expected_connection_epoch=7,
        )

        self.assertIn("serial mudou", error)
        self.assertEqual(arduino.calls, [("parar_futaba",)])

    def test_failed_futaba_write_is_reported(self):
        arduino = self.FakeArduino()
        arduino.futaba = lambda *_args: False
        step = PickupStep(
            "LIFT",
            "subindo",
            motor_action="hold",
            futaba_action=(20, 2000),
        )

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIn("FUTABA", error)


if __name__ == "__main__":
    unittest.main()
