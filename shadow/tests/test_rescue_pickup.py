import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg  # noqa: E402
from control.rescue_pickup import BallPickupSequencer  # noqa: E402
from rescue_main import _apply_pickup_actions  # noqa: E402


class BallPickupSequencerTests(unittest.TestCase):
    def test_sequence_order_deadlines_and_one_shot_actions(self):
        pickup = BallPickupSequencer()
        self.assertTrue(pickup.start())
        self.assertFalse(pickup.start())

        reverse = pickup.update(now=10.0)
        self.assertEqual(reverse.motor_action, "reverse")
        self.assertEqual(reverse.angle, 200)
        self.assertEqual(reverse.speed, cfg.BALL_PICKUP_REVERSE_SPEED)
        self.assertIsNone(reverse.futaba_action)
        pickup.mark_reverse_started(now=10.0)

        still_reversing = pickup.update(
            now=10.0 + cfg.BALL_PICKUP_REVERSE_S - 0.001)
        self.assertEqual(still_reversing.state, pickup.BACKUP)
        self.assertEqual(still_reversing.motor_action, "")

        futaba_started_at = 10.0 + cfg.BALL_PICKUP_REVERSE_S
        futaba = pickup.update(now=futaba_started_at)
        self.assertEqual(futaba.motor_action, "hold")
        self.assertEqual(
            futaba.futaba_action,
            (cfg.BALL_PICKUP_FUTABA_POWER, cfg.BALL_PICKUP_FUTABA_MS),
        )
        self.assertIsNone(futaba.gripper_action)
        pickup.mark_futaba_started(now=futaba_started_at)

        wait_s = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        before_down = pickup.update(
            now=futaba_started_at + wait_s - 0.001)
        self.assertIsNone(before_down.futaba_action)
        self.assertIsNone(before_down.gripper_action)

        grippers = pickup.update(now=futaba_started_at + wait_s)
        self.assertTrue(grippers.stop_futaba)
        self.assertEqual(
            grippers.gripper_action,
            (
                cfg.BALL_PICKUP_LEFT_DELTA,
                cfg.BALL_PICKUP_RIGHT_DELTA,
            ),
        )
        pickup.mark_grippers_started(now=futaba_started_at + wait_s)

        settling = pickup.update(
            now=futaba_started_at
            + wait_s
            + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
            - 0.001
        )
        self.assertIsNone(settling.gripper_action)
        self.assertFalse(settling.terminal)

        complete = pickup.update(
            now=futaba_started_at
            + wait_s
            + cfg.BALL_PICKUP_GRIPPER_SETTLE_S
        )
        self.assertEqual(complete.motor_action, "stop")
        self.assertTrue(complete.terminal)

        latched = pickup.update(now=99.0)
        self.assertTrue(latched.terminal)
        self.assertEqual(latched.motor_action, "")
        self.assertIsNone(latched.futaba_action)
        self.assertIsNone(latched.gripper_action)

    def test_delayed_first_tick_cannot_skip_reverse(self):
        pickup = BallPickupSequencer()
        pickup.start()

        first = pickup.update(now=1000.0)
        self.assertEqual(first.motor_action, "reverse")
        self.assertEqual(pickup.state, pickup.BACKUP_PENDING)

        delivered_at = 1005.0
        pickup.mark_reverse_started(now=delivered_at)

        before_deadline = pickup.update(
            now=delivered_at + cfg.BALL_PICKUP_REVERSE_S - 0.001)
        self.assertEqual(before_deadline.state, pickup.BACKUP)
        self.assertIsNone(before_deadline.futaba_action)

    def test_failure_is_terminal_and_never_opens_grippers(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)

        fault = pickup.fail("serial ausente")
        self.assertEqual(fault.state, pickup.FAULT)
        self.assertEqual(fault.motor_action, "stop")
        self.assertTrue(fault.stop_futaba)
        self.assertIsNone(fault.gripper_action)
        self.assertTrue(fault.terminal)
        self.assertFalse(pickup.start())


class PickupActionApplicationTests(unittest.TestCase):
    class FakeArduino:
        def __init__(self):
            self.calls = []

        def lado(self, left, right):
            self.calls.append(("lado", left, right))

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
        return steer

    def test_hold_is_sent_before_futaba_without_parar(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_reverse_started(now=0.0)
        step = pickup.update(now=cfg.BALL_PICKUP_REVERSE_S)
        arduino = self.FakeArduino()

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIsNone(error)
        self.assertEqual(
            arduino.calls,
            [
                ("lado", 0, 0),
                (
                    "futaba",
                    cfg.BALL_PICKUP_FUTABA_POWER,
                    cfg.BALL_PICKUP_FUTABA_MS,
                ),
            ],
        )
        self.assertFalse(any(call[0] == "steer" for call in arduino.calls))

    def test_futaba_is_cut_then_both_grippers_are_sent_together(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_reverse_started(now=0.0)
        pickup.update(now=cfg.BALL_PICKUP_REVERSE_S)
        pickup.mark_futaba_started(now=cfg.BALL_PICKUP_REVERSE_S)
        gripper_time = (
            cfg.BALL_PICKUP_REVERSE_S
            + cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        step = pickup.update(now=gripper_time)
        arduino = self.FakeArduino()

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIsNone(error)
        self.assertEqual(
            arduino.calls,
            [
                ("parar_futaba",),
                (
                    "garras",
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
            ],
        )

    def test_failed_futaba_write_is_reported(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_reverse_started(now=0.0)
        step = pickup.update(now=cfg.BALL_PICKUP_REVERSE_S)
        arduino = self.FakeArduino()
        arduino.futaba = lambda *_args: False

        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
        )

        self.assertIn("FUTABA", error)

    def test_wait_deadline_starts_after_serial_delivery(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_reverse_started(now=0.0)
        pickup.update(now=cfg.BALL_PICKUP_REVERSE_S)

        # Simula uma reconexao lenta: o comando so retornou varios segundos
        # depois de o estado ter produzido a acao.
        delivered_at = 8.0
        pickup.mark_futaba_started(now=delivered_at)
        wait_s = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )

        before = pickup.update(now=delivered_at + wait_s - 0.001)
        self.assertIsNone(before.gripper_action)
        at_deadline = pickup.update(now=delivered_at + wait_s)
        self.assertIsNotNone(at_deadline.gripper_action)

    def test_full_handoff_command_order(self):
        pickup = BallPickupSequencer()
        arduino = self.FakeArduino()
        steer = self.steer_recorder(arduino.calls)

        # NEAR para primeiro; a coleta comeca somente no tick seguinte.
        steer()
        pickup.start()
        reverse = pickup.update(now=0.0)
        self.assertIsNone(
            _apply_pickup_actions(reverse, arduino, steer))
        pickup.mark_reverse_started(now=0.0)

        futaba = pickup.update(now=cfg.BALL_PICKUP_REVERSE_S)
        self.assertIsNone(
            _apply_pickup_actions(futaba, arduino, steer))
        pickup.mark_futaba_started(now=cfg.BALL_PICKUP_REVERSE_S)

        gripper_time = (
            cfg.BALL_PICKUP_REVERSE_S
            + cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        grippers = pickup.update(now=gripper_time)
        self.assertIsNone(
            _apply_pickup_actions(grippers, arduino, steer))
        pickup.mark_grippers_started(now=gripper_time)

        complete = pickup.update(
            now=gripper_time + cfg.BALL_PICKUP_GRIPPER_SETTLE_S)
        self.assertIsNone(
            _apply_pickup_actions(complete, arduino, steer))

        self.assertEqual(
            arduino.calls,
            [
                ("steer", 190, 0.8),
                (
                    "steer",
                    200,
                    cfg.BALL_PICKUP_REVERSE_SPEED,
                ),
                ("lado", 0, 0),
                (
                    "futaba",
                    cfg.BALL_PICKUP_FUTABA_POWER,
                    cfg.BALL_PICKUP_FUTABA_MS,
                ),
                ("parar_futaba",),
                (
                    "garras",
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
                ("steer", 190, 0.8),
            ],
        )

    def test_reconnect_between_futaba_stop_and_grippers_aborts(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_reverse_started(now=0.0)
        pickup.update(now=cfg.BALL_PICKUP_REVERSE_S)
        pickup.mark_futaba_started(now=cfg.BALL_PICKUP_REVERSE_S)
        gripper_time = (
            cfg.BALL_PICKUP_REVERSE_S
            + cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        step = pickup.update(now=gripper_time)

        arduino = self.FakeArduino()
        arduino.connected = True
        arduino.connection_epoch = 7

        def reconnecting_stop():
            arduino.calls.append(("parar_futaba",))
            arduino.connection_epoch += 1
            return True

        arduino.parar_futaba = reconnecting_stop
        error = _apply_pickup_actions(
            step,
            arduino,
            self.steer_recorder(arduino.calls),
            expected_connection_epoch=7,
        )

        self.assertIn("serial mudou", error)
        self.assertEqual(arduino.calls, [("parar_futaba",)])


if __name__ == "__main__":
    unittest.main()
