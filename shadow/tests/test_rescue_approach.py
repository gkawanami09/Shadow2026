import sys
from pathlib import Path
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg
from control.rescue_approach import BallApproachController
from control import steer as steer_module
from vision.rescue_ball import BallDetection


def detection(
    x=320, y=300, radius=25, confirmed=True, timestamp=0.0, kind="silver"
):
    return BallDetection(
        kind, x, y, radius, 0.9, confirmed, cfg.BALL_ACQUIRE_HITS, timestamp)


class BallApproachControllerTests(unittest.TestCase):
    shape = (480, 640, 3)

    def test_does_not_move_before_temporal_confirmation(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(confirmed=False), self.shape, now=0.1)
        self.assertEqual(command.state, controller.WAIT_TARGET)
        self.assertEqual(command.angle, 190)
        self.assertEqual(command.speed, 0.0)

    def test_left_and_right_alignment_signs(self):
        left = BallApproachController(start_time=0.0).update(
            detection(x=120, timestamp=0.1), self.shape, now=0.1)
        right = BallApproachController(start_time=0.0).update(
            detection(x=520, timestamp=0.1), self.shape, now=0.1)
        self.assertEqual(left.state, BallApproachController.ALIGN)
        self.assertLess(left.angle, 0)
        self.assertGreater(right.angle, 0)

    def test_centered_target_approaches_and_slows_down(self):
        controller = BallApproachController(start_time=0.0)
        far = controller.update(
            detection(radius=22, timestamp=0.1), self.shape, now=0.1)
        near = controller.update(
            detection(radius=66, y=300, timestamp=0.2), self.shape, now=0.2)
        self.assertEqual(far.state, controller.APPROACH)
        self.assertEqual(far.angle, 0)
        self.assertLess(near.speed, far.speed)

    def test_visual_near_requires_multiple_frames_and_latches(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    y=390,
                    radius=cfg.BALL_STOP_RADIUS_PX + 2,
                    timestamp=now),
                self.shape,
                now=now)
        self.assertEqual(command.state, controller.NEAR)
        self.assertTrue(command.terminal)
        latched = controller.update(None, self.shape, now=1.0)
        self.assertEqual(latched.state, controller.NEAR)
        self.assertEqual(latched.angle, 190)

    def test_visual_radius_threshold_scales_with_camera_resolution(self):
        controller = BallApproachController(start_time=0.0)
        shape = (720, 960, 3)
        scale = cfg.ball_pixel_scale(shape[1], shape[0])
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=shape[1] / 2,
                    y=585,
                    radius=(cfg.BALL_STOP_RADIUS_PX + 2) * scale,
                    timestamp=now),
                shape,
                now=now)
        self.assertEqual(command.state, controller.NEAR)
        self.assertTrue(command.terminal)

    def test_wide_fov_uses_isotropic_not_horizontal_scale(self):
        shape = (540, 960, 3)
        scale = cfg.ball_pixel_scale(shape[1], shape[0])
        self.assertAlmostEqual(scale, 1.125)

        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=shape[1] / 2,
                    y=440,
                    radius=(cfg.BALL_STOP_RADIUS_PX + 2) * scale,
                    timestamp=now),
                shape,
                now=now)
        self.assertEqual(command.state, controller.NEAR)
        self.assertTrue(command.terminal)

    def test_target_loss_stops_immediately(self):
        controller = BallApproachController(start_time=0.0)
        controller.update(
            detection(timestamp=0.1), self.shape, now=0.1)
        lost = controller.update(None, self.shape, now=0.2)
        self.assertEqual(lost.state, controller.LOST)
        self.assertEqual(lost.angle, 190)

    def test_ultrasonic_is_only_accepted_with_centered_target(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_ULTRASONIC_CONFIRM_READS):
            now = 0.1 + index * 0.1
            command = controller.update(
                detection(timestamp=now), self.shape,
                distance_mm=cfg.BALL_ULTRASONIC_STOP_MM - 5,
                ultrasonic_polled=True,
                now=now)
        self.assertEqual(command.state, controller.NEAR)

        off_center = BallApproachController(start_time=0.0)
        for index in range(cfg.BALL_ULTRASONIC_CONFIRM_READS + 1):
            now = 0.1 + index * 0.1
            command = off_center.update(
                detection(x=520, timestamp=now), self.shape,
                distance_mm=cfg.BALL_ULTRASONIC_STOP_MM - 5,
                ultrasonic_polled=True,
                now=now)
        self.assertEqual(command.state, off_center.FAULT)

    def test_first_near_ultrasonic_echo_stops_provisionally(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(timestamp=0.1),
            self.shape,
            distance_mm=cfg.BALL_ULTRASONIC_STOP_MM - 5,
            ultrasonic_polled=True,
            now=0.1)
        self.assertEqual(command.state, controller.PROXIMITY_HOLD)
        self.assertEqual(command.angle, 190)
        self.assertEqual(command.speed, 0.0)

    def test_ultrasonic_below_reliable_range_faults(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(timestamp=0.1),
            self.shape,
            distance_mm=cfg.BALL_ULTRASONIC_MIN_VALID_MM - 1,
            ultrasonic_polled=True,
            now=0.1)
        self.assertEqual(command.state, controller.FAULT)
        self.assertTrue(command.terminal)
        self.assertEqual(command.angle, 190)

    def test_stale_frame_never_moves(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(timestamp=0.0), self.shape,
            now=cfg.BALL_FRAME_STALE_S + 0.01)
        self.assertEqual(command.state, controller.LOST)
        self.assertEqual(command.angle, 190)

    def test_wait_timeout_is_a_latched_fault(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            None, self.shape, now=cfg.BALL_MAX_WAIT_S + 0.01)
        self.assertEqual(command.state, controller.FAULT)
        self.assertTrue(command.terminal)
        self.assertEqual(
            controller.update(None, self.shape, now=99).state,
            controller.FAULT)

    def test_no_radius_progress_stops_the_robot(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for now in (0.1, 1.0, 2.0, 3.0):
            command = controller.update(
                detection(radius=25, timestamp=now),
                self.shape,
                now=now)
        self.assertEqual(command.state, controller.FAULT)
        self.assertTrue(command.terminal)
        self.assertEqual(command.angle, 190)

    def test_commands_integrate_with_existing_steer_without_hardware(self):
        class FakeArduino:
            def __init__(self):
                self.calls = []

            def lado(self, left, right):
                self.calls.append(("lado", left, right))

            def rodas(self, *values):
                self.calls.append(("rodas", *values))

            def parar(self):
                self.calls.append(("parar",))

        fake = FakeArduino()
        steer_module.init_steering(fake)

        left_command = BallApproachController(start_time=0.0).update(
            detection(x=120, timestamp=0.1), self.shape, now=0.1)
        steer_module.steer(left_command.angle, left_command.speed)
        self.assertEqual(fake.calls[-1][0], "lado")
        self.assertLess(fake.calls[-1][1], 0)
        self.assertGreater(fake.calls[-1][2], 0)

        forward_command = BallApproachController(start_time=0.0).update(
            detection(timestamp=0.1), self.shape, now=0.1)
        steer_module.steer(forward_command.angle, forward_command.speed)
        self.assertGreater(fake.calls[-1][1], 0)
        self.assertGreater(fake.calls[-1][2], 0)

        stop_command = BallApproachController(start_time=0.0).update(
            None, self.shape, now=0.1)
        steer_module.steer(stop_command.angle, stop_command.speed)
        self.assertEqual(fake.calls[-1], ("parar",))


if __name__ == "__main__":
    unittest.main()
