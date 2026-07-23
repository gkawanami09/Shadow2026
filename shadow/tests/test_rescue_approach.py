import ast
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
    x=320,
    y=300,
    radius=25,
    confirmed=True,
    timestamp=0.0,
    kind="silver",
    confidence=0.9,
    hits=None,
):
    return BallDetection(
        kind,
        x,
        y,
        radius,
        confidence,
        confirmed,
        cfg.BALL_ACQUIRE_HITS if hits is None else hits,
        timestamp,
    )


def screenshot_candidate_circles(scale=1.0):
    return tuple((
        center_x * scale,
        center_y * scale,
        radius * scale,
        "silver",
        confidence,
    ) for center_x, center_y, radius, confidence in (
        (320, 330, 134, 0.91),
        (330, 330, 134, 0.89),
        (322, 340, 113, 0.87),
        (325, 350, 100, 0.84),
    )
    )


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

    def test_close_silver_range_matches_reported_reflection_frame(self):
        controller = BallApproachController(start_time=0.0)
        commands = []
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            commands.append(controller.update(
                detection(
                    x=424,
                    y=379,
                    radius=37,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                self.shape,
                candidate_count=25,
                candidate_circles=screenshot_candidate_circles(),
                now=now,
            ))

        self.assertEqual(commands[0].state, controller.NEAR_CONFIRM)
        self.assertEqual(commands[0].angle, 190)
        self.assertEqual(commands[0].speed, 0.0)
        self.assertTrue(commands[0].pickup_in_range)
        self.assertEqual(commands[0].pickup_confirmations, 1)
        self.assertEqual(commands[1].pickup_confirmations, 2)
        self.assertEqual(commands[2].state, controller.NEAR)
        self.assertTrue(commands[2].terminal)
        self.assertIn("faixa de coleta", commands[2].detail)

    def test_close_silver_range_rejects_missing_outer_sphere_evidence(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=424,
                    y=379,
                    radius=37,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                self.shape,
                candidate_count=1,
                candidate_circles=(
                    (424, 379, 37, "silver", 0.91),
                ),
                now=now,
            )

        self.assertEqual(command.state, controller.ALIGN)
        self.assertFalse(command.terminal)
        self.assertFalse(command.pickup_in_range)

    def test_close_silver_range_rejects_ball_above_marked_band(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=424,
                    y=300,
                    radius=37,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                self.shape,
                candidate_count=25,
                candidate_circles=screenshot_candidate_circles(),
                now=now,
            )

        self.assertEqual(command.state, controller.ALIGN)
        self.assertFalse(command.terminal)
        self.assertFalse(command.pickup_in_range)

    def test_close_silver_range_scales_with_preview_resolution(self):
        controller = BallApproachController(start_time=0.0)
        shape = (720, 960, 3)
        scale = cfg.ball_pixel_scale(shape[1], shape[0])
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=424 * scale,
                    y=379 * scale,
                    radius=37 * scale,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                shape,
                candidate_count=25,
                candidate_circles=screenshot_candidate_circles(scale),
                now=now,
            )

        self.assertEqual(command.state, controller.NEAR)
        self.assertTrue(command.terminal)

    def test_close_silver_range_rejects_large_circles_from_other_objects(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        unrelated_circles = (
            (136, 180, 134, "silver", 0.91),
            (502, 180, 134, "silver", 0.89),
            (120, 170, 113, "silver", 0.87),
            (520, 170, 100, "silver", 0.84),
        )
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=424,
                    y=379,
                    radius=37,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                self.shape,
                candidate_count=25,
                candidate_circles=unrelated_circles,
                now=now,
            )

        self.assertEqual(command.state, controller.ALIGN)
        self.assertFalse(command.terminal)
        self.assertFalse(command.pickup_in_range)

    def test_close_silver_range_rejects_misaligned_outer_sphere(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        right_side_circles = (
            (430, 330, 134, "silver", 0.91),
            (440, 330, 134, "silver", 0.89),
            (425, 340, 113, "silver", 0.87),
            (430, 350, 100, "silver", 0.84),
        )
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    x=424,
                    y=379,
                    radius=37,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                self.shape,
                candidate_count=25,
                candidate_circles=right_side_circles,
                now=now,
            )

        self.assertEqual(command.state, controller.ALIGN)
        self.assertFalse(command.terminal)
        self.assertFalse(command.pickup_in_range)

    def test_close_silver_range_rejects_non_silver_outer_circles(self):
        controller = BallApproachController(start_time=0.0)
        black_circles = tuple(
            (*circle[:3], "black", circle[4])
            for circle in screenshot_candidate_circles()
        )
        command = controller.update(
            detection(
                x=424,
                y=379,
                radius=37,
                confidence=0.91,
                hits=181,
                timestamp=0.1,
            ),
            self.shape,
            candidate_count=25,
            candidate_circles=black_circles,
            now=0.1,
        )

        self.assertEqual(command.state, controller.ALIGN)
        self.assertFalse(command.pickup_in_range)

    def test_rescue_runtime_never_calls_ultrasonic_sensor(self):
        source = (SHADOW_ROOT / "rescue_main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_calls = {
            "distancia_ultrassom",
            "iniciar_ultrassom",
            "poll_ultrassom",
            "cancelar_ultrassom",
        }
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_calls
        }
        self.assertEqual(called, set())

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

    def test_black_ball_also_waits_for_the_lower_pickup_line(self):
        controller = BallApproachController(start_time=0.0)
        radius = cfg.BALL_STOP_RADIUS_PX + 2

        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    y=300,
                    radius=radius,
                    kind="black",
                    timestamp=now,
                ),
                self.shape,
                now=now,
            )
        self.assertEqual(command.state, controller.APPROACH)
        self.assertFalse(command.terminal)

        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.3 + index * 0.05
            command = controller.update(
                detection(
                    y=340,
                    radius=radius,
                    kind="black",
                    timestamp=now,
                ),
                self.shape,
                now=now,
            )
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

    def test_stale_frame_never_moves(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(timestamp=0.0), self.shape,
            now=cfg.BALL_FRAME_STALE_S + 0.01)
        self.assertEqual(command.state, controller.LOST)
        self.assertEqual(command.angle, 190)

    def test_stale_frame_resets_visual_near_confirmation(self):
        controller = BallApproachController(start_time=0.0)
        near_radius = cfg.BALL_STOP_RADIUS_PX + 2
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES - 1):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(y=390, radius=near_radius, timestamp=now),
                self.shape,
                now=now)
            self.assertFalse(command.terminal)

        stale_now = cfg.BALL_FRAME_STALE_S + 1.0
        stale = controller.update(
            detection(y=390, radius=near_radius, timestamp=0.0),
            self.shape,
            now=stale_now)
        self.assertEqual(stale.state, controller.LOST)

        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES - 1):
            now = stale_now + 0.1 + index * 0.05
            command = controller.update(
                detection(y=390, radius=near_radius, timestamp=now),
                self.shape,
                now=now)
            self.assertFalse(command.terminal)
        now += 0.05
        command = controller.update(
            detection(y=390, radius=near_radius, timestamp=now),
            self.shape,
            now=now)
        self.assertEqual(command.state, controller.NEAR)

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

    def test_target_moving_down_counts_as_visual_progress(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for now, center_y in (
            (0.1, 300),
            (1.0, 304),
            (2.0, 309),
            (3.0, 314),
        ):
            command = controller.update(
                detection(
                    y=center_y,
                    radius=25,
                    timestamp=now,
                ),
                self.shape,
                now=now,
            )

        self.assertEqual(command.state, controller.APPROACH)
        self.assertFalse(command.terminal)

    def test_close_range_confirmation_precedes_progress_watchdog(self):
        controller = BallApproachController(start_time=0.0)
        for now, center_y in (
            (0.1, 330),
            (1.0, 334),
            (2.6, 339),
        ):
            command = controller.update(
                detection(
                    y=center_y,
                    radius=37,
                    timestamp=now,
                ),
                self.shape,
                now=now,
            )
            self.assertNotEqual(command.state, controller.FAULT)

        commands = []
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 2.90 + index * 0.04
            commands.append(controller.update(
                detection(
                    x=424,
                    y=379,
                    radius=37,
                    confidence=0.91,
                    hits=181 + index,
                    timestamp=now,
                ),
                self.shape,
                candidate_count=25,
                candidate_circles=screenshot_candidate_circles(),
                now=now,
            ))

        self.assertEqual(commands[0].state, controller.NEAR_CONFIRM)
        self.assertEqual(commands[-1].state, controller.NEAR)

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
