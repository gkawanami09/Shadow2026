import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg  # noqa: E402
from control.rescue_pickup import BallPickupSequencer  # noqa: E402
from rescue_main import _apply_pickup_actions  # noqa: E402


class BallPickupSequencerTests(unittest.TestCase):
    def test_collection_motion_durations_match_requested_sequence(self):
        self.assertFalse(hasattr(cfg, "BALL_PICKUP_REVERSE_S"))
        self.assertEqual(cfg.BALL_PICKUP_FORWARD_S, 2.0)
        self.assertEqual(
            cfg.BALL_PICKUP_FORWARD_LEAD_S,
            cfg.BALL_PICKUP_FORWARD_S,
        )
        self.assertEqual(
            cfg.BALL_PICKUP_FORWARD_SPEED,
            cfg.BALL_APPROACH_SPEED_NEAR,
        )

    def test_sequence_order_deadlines_and_one_shot_actions(self):
        pickup = BallPickupSequencer()
        self.assertTrue(pickup.start())
        self.assertFalse(pickup.start())

        futaba_started_at = 10.0
        futaba = pickup.update(now=futaba_started_at)
        self.assertEqual(futaba.motor_action, "hold")
        self.assertNotEqual(futaba.angle, 200)
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

        forward = pickup.update(now=futaba_started_at + wait_s)
        self.assertEqual(forward.motor_action, "forward")
        self.assertEqual(forward.angle, 0)
        self.assertEqual(
            forward.speed,
            cfg.BALL_PICKUP_FORWARD_SPEED,
        )
        self.assertTrue(forward.stop_futaba)
        self.assertIsNone(forward.gripper_action)
        forward_started_at = futaba_started_at + wait_s
        pickup.mark_forward_started(now=forward_started_at)

        before_grip = pickup.update(
            now=forward_started_at
            + cfg.BALL_PICKUP_FORWARD_LEAD_S
            - 0.001
        )
        self.assertEqual(before_grip.state, pickup.FORWARD_LEAD)
        self.assertIsNone(before_grip.gripper_action)

        grippers_started_at = (
            forward_started_at + cfg.BALL_PICKUP_FORWARD_LEAD_S)
        grippers = pickup.update(now=grippers_started_at)
        self.assertEqual(grippers.motor_action, "stop")
        self.assertFalse(grippers.stop_futaba)
        self.assertEqual(
            grippers.gripper_action,
            (
                cfg.BALL_PICKUP_LEFT_DELTA,
                cfg.BALL_PICKUP_RIGHT_DELTA,
            ),
        )
        pickup.mark_grippers_started(now=grippers_started_at)

        complete = pickup.update(
            now=forward_started_at
            + cfg.BALL_PICKUP_FORWARD_S
        )
        self.assertEqual(complete.motor_action, "")
        self.assertTrue(complete.terminal)

        latched = pickup.update(now=99.0)
        self.assertTrue(latched.terminal)
        self.assertEqual(latched.motor_action, "")
        self.assertIsNone(latched.futaba_action)
        self.assertIsNone(latched.gripper_action)

    def test_delayed_first_tick_starts_futaba_once_without_reverse(self):
        pickup = BallPickupSequencer()
        pickup.start()

        first = pickup.update(now=1000.0)
        self.assertEqual(first.motor_action, "hold")
        self.assertNotEqual(first.angle, 200)
        self.assertIsNotNone(first.futaba_action)
        self.assertEqual(pickup.state, pickup.FUTABA_PENDING)

        pending = pickup.update(now=1005.0)
        self.assertIsNone(pending.futaba_action)
        self.assertEqual(pending.state, pickup.FUTABA_PENDING)

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

    @staticmethod
    def advance_to_forward_step(pickup):
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_futaba_started(now=0.0)
        forward_time = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        return pickup.update(now=forward_time), forward_time

    @classmethod
    def advance_to_gripper_step(cls, pickup):
        forward, forward_time = cls.advance_to_forward_step(pickup)
        pickup.mark_forward_started(now=forward_time)
        gripper_time = (
            forward_time + cfg.BALL_PICKUP_FORWARD_LEAD_S)
        return pickup.update(now=gripper_time), gripper_time, forward

    def test_hold_is_sent_before_futaba_without_parar(self):
        pickup = BallPickupSequencer()
        pickup.start()
        step = pickup.update(now=0.0)
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

    def test_forward_is_sent_before_both_grippers(self):
        pickup = BallPickupSequencer()
        forward, forward_time = self.advance_to_forward_step(pickup)
        arduino = self.FakeArduino()
        steer = self.steer_recorder(arduino.calls)

        self.assertIsNone(
            _apply_pickup_actions(forward, arduino, steer))
        pickup.mark_forward_started(now=forward_time)
        before_grip = pickup.update(
            now=forward_time + cfg.BALL_PICKUP_FORWARD_LEAD_S - 0.001)
        self.assertIsNone(before_grip.gripper_action)
        grippers = pickup.update(
            now=forward_time + cfg.BALL_PICKUP_FORWARD_LEAD_S)
        self.assertIsNone(
            _apply_pickup_actions(grippers, arduino, steer))
        self.assertEqual(
            arduino.calls,
            [
                ("parar_futaba",),
                (
                    "steer",
                    0,
                    cfg.BALL_PICKUP_FORWARD_SPEED,
                ),
                ("steer", 190, 0.8),
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
        step = pickup.update(now=0.0)
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
        self.assertEqual(at_deadline.motor_action, "forward")
        self.assertIsNone(at_deadline.gripper_action)
        forward_started_at = delivered_at + wait_s
        pickup.mark_forward_started(now=forward_started_at)
        before_grip = pickup.update(
            now=forward_started_at
            + cfg.BALL_PICKUP_FORWARD_LEAD_S
            - 0.001)
        self.assertIsNone(before_grip.gripper_action)
        at_grip = pickup.update(
            now=forward_started_at + cfg.BALL_PICKUP_FORWARD_LEAD_S)
        self.assertIsNotNone(at_grip.gripper_action)
        self.assertEqual(at_grip.motor_action, "stop")

    def test_forward_deadline_starts_after_motor_delivery(self):
        pickup = BallPickupSequencer()
        forward, forward_time = self.advance_to_forward_step(pickup)
        self.assertEqual(forward.motor_action, "forward")
        self.assertIsNone(forward.gripper_action)

        # Simula demora para entregar o comando: os 2 s com garras abertas
        # comecam somente quando a escrita dos motores termina.
        delivered_at = forward_time + 0.20
        pickup.mark_forward_started(now=delivered_at)

        before = pickup.update(
            now=delivered_at + cfg.BALL_PICKUP_FORWARD_S - 0.001)
        self.assertEqual(before.state, pickup.FORWARD_LEAD)
        self.assertIsNone(before.gripper_action)

        grippers = pickup.update(
            now=delivered_at + cfg.BALL_PICKUP_FORWARD_S)
        self.assertEqual(grippers.motor_action, "stop")
        self.assertIsNotNone(grippers.gripper_action)
        pickup.mark_grippers_started(
            now=delivered_at + cfg.BALL_PICKUP_FORWARD_S + 0.20)
        complete = pickup.update(
            now=delivered_at + cfg.BALL_PICKUP_FORWARD_S + 0.20)
        self.assertEqual(complete.motor_action, "")
        self.assertTrue(complete.terminal)

    def test_failed_gripper_write_stops_forward_immediately(self):
        pickup = BallPickupSequencer()
        grippers, _gripper_time, forward = (
            self.advance_to_gripper_step(pickup))
        arduino = self.FakeArduino()
        steer = self.steer_recorder(arduino.calls)
        self.assertIsNone(
            _apply_pickup_actions(forward, arduino, steer))

        def fail_grippers(left, right):
            arduino.calls.append(("garras", left, right))
            return False

        arduino.garras = fail_grippers
        error = _apply_pickup_actions(grippers, arduino, steer)

        self.assertIsNotNone(error)
        self.assertEqual(
            arduino.calls,
            [
                ("parar_futaba",),
                ("steer", 0, cfg.BALL_PICKUP_FORWARD_SPEED),
                ("steer", 190, 0.8),
                (
                    "garras",
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
            ],
        )

    def test_failed_forward_write_never_commands_grippers(self):
        pickup = BallPickupSequencer()
        forward, _forward_time = self.advance_to_forward_step(pickup)
        arduino = self.FakeArduino()

        error = _apply_pickup_actions(
            forward,
            arduino,
            lambda *_args: False,
        )

        self.assertIsNotNone(error)
        self.assertEqual(
            arduino.calls,
            [("parar_futaba",)],
        )

    def test_full_handoff_command_order(self):
        pickup = BallPickupSequencer()
        arduino = self.FakeArduino()
        steer = self.steer_recorder(arduino.calls)

        # NEAR para primeiro; a coleta comeca somente no tick seguinte.
        steer()
        pickup.start()
        futaba = pickup.update(now=0.0)
        self.assertIsNone(
            _apply_pickup_actions(futaba, arduino, steer))
        pickup.mark_futaba_started(now=0.0)

        forward_time = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
            + cfg.BALL_PICKUP_FUTABA_GUARD_S
        )
        forward = pickup.update(now=forward_time)
        self.assertIsNone(
            _apply_pickup_actions(forward, arduino, steer))
        pickup.mark_forward_started(now=forward_time)

        gripper_time = forward_time + cfg.BALL_PICKUP_FORWARD_LEAD_S
        grippers = pickup.update(now=gripper_time)
        self.assertIsNone(
            _apply_pickup_actions(grippers, arduino, steer))
        pickup.mark_grippers_started(now=gripper_time)

        complete = pickup.update(
            now=forward_time + cfg.BALL_PICKUP_FORWARD_S)
        self.assertIsNone(
            _apply_pickup_actions(complete, arduino, steer))

        self.assertEqual(
            arduino.calls,
            [
                ("steer", 190, 0.8),
                ("lado", 0, 0),
                (
                    "futaba",
                    cfg.BALL_PICKUP_FUTABA_POWER,
                    cfg.BALL_PICKUP_FUTABA_MS,
                ),
                ("parar_futaba",),
                (
                    "steer",
                    0,
                    cfg.BALL_PICKUP_FORWARD_SPEED,
                ),
                ("steer", 190, 0.8),
                (
                    "garras",
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
            ],
        )

    def test_reconnect_between_futaba_stop_and_forward_aborts(self):
        pickup = BallPickupSequencer()
        pickup.start()
        pickup.update(now=0.0)
        pickup.mark_futaba_started(now=0.0)
        gripper_time = (
            cfg.BALL_PICKUP_FUTABA_MS / 1000.0
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

    def test_reconnect_during_grippers_stops_forward(self):
        pickup = BallPickupSequencer()
        grippers, _gripper_time, forward = (
            self.advance_to_gripper_step(pickup))

        arduino = self.FakeArduino()
        arduino.connected = True
        arduino.connection_epoch = 7
        steer = self.steer_recorder(arduino.calls)
        self.assertIsNone(_apply_pickup_actions(
            forward,
            arduino,
            steer,
            expected_connection_epoch=7,
        ))

        def reconnecting_grippers(left, right):
            arduino.calls.append(("garras", left, right))
            arduino.connection_epoch += 1
            return True

        arduino.garras = reconnecting_grippers
        error = _apply_pickup_actions(
            grippers,
            arduino,
            steer,
            expected_connection_epoch=7,
        )

        self.assertIn("serial mudou", error)
        self.assertEqual(
            arduino.calls,
            [
                ("parar_futaba",),
                ("steer", 0, cfg.BALL_PICKUP_FORWARD_SPEED),
                ("steer", 190, 0.8),
                (
                    "garras",
                    cfg.BALL_PICKUP_LEFT_DELTA,
                    cfg.BALL_PICKUP_RIGHT_DELTA,
                ),
            ],
        )

    def test_gripper_exception_stops_forward(self):
        pickup = BallPickupSequencer()
        grippers, _gripper_time, forward = (
            self.advance_to_gripper_step(pickup))
        arduino = self.FakeArduino()
        steer = self.steer_recorder(arduino.calls)
        self.assertIsNone(
            _apply_pickup_actions(forward, arduino, steer))

        def exploding_grippers(_left, _right):
            raise RuntimeError("servo indisponivel")

        arduino.garras = exploding_grippers
        error = _apply_pickup_actions(grippers, arduino, steer)

        self.assertIn("servo indisponivel", error)
        self.assertEqual(
            arduino.calls[-1],
            ("steer", 190, 0.8),
        )


if __name__ == "__main__":
    unittest.main()
