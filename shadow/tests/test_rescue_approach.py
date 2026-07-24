import ast
import sys
from pathlib import Path
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import rescue_config as cfg
from control.rescue_approach import BallApproachController
from control import steer as steer_module
from vision.rescue_ball import BallDetection, CloseCrescentEvidence


def detection(
    x=320,
    y=300,
    radius=25,
    confirmed=True,
    timestamp=0.0,
    kind="silver",
    confidence=0.9,
    hits=None,
    track_locked=False,
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
        track_locked=track_locked,
    )


def crescent_evidence(
    timestamp=0.0,
    accepted=True,
    confidence=None,
    center_x_ratio=0.50,
):
    return CloseCrescentEvidence(
        accepted=accepted,
        confidence=(
            (0.90 if accepted else 0.20)
            if confidence is None
            else confidence
        ),
        support=0.80 if accepted else 0.20,
        left_support=0.75 if accepted else 0.10,
        center_support=0.90 if accepted else 0.40,
        right_support=0.76 if accepted else 0.10,
        contrast=42.0 if accepted else 3.0,
        center_x_ratio=center_x_ratio,
        top_y_ratio=0.74,
        halfspan_ratio=0.46,
        bottom_y_ratio=0.98,
        timestamp=timestamp,
    )


def arm_crescent_history(
    controller,
    shape,
    start_time=0.05,
    kind="silver",
):
    """Simula a aproximacao crescente que autoriza a transicao ao corte."""
    height, width = shape[:2]
    last_time = start_time
    for index in range(7):
        last_time = start_time + index * 0.05
        radius_ratio = 0.045 + index * 0.0065
        bottom_ratio = 0.650 + index * 0.023
        radius = height * radius_ratio
        center_y = height * bottom_ratio - radius
        command = controller.update(
            detection(
                x=width / 2.0,
                y=center_y,
                radius=radius,
                timestamp=last_time,
                kind=kind,
            ),
            shape,
            now=last_time,
        )
        if command.terminal:
            raise AssertionError(
                f"aproximacao sintetica terminou em {command.state}")
    return last_time


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

    def test_alignment_uses_proportional_forward_arc_not_pivot(self):
        controller = BallApproachController(start_time=0.0)
        moderate = controller.update(
            detection(x=410, timestamp=0.1),
            self.shape,
            now=0.1,
        )
        strong = BallApproachController(start_time=0.0).update(
            detection(x=590, timestamp=0.1),
            self.shape,
            now=0.1,
        )

        self.assertEqual(moderate.state, controller.ALIGN)
        self.assertLessEqual(
            abs(moderate.angle),
            cfg.BALL_ALIGN_ARC_MAX_ANGLE,
        )
        self.assertLessEqual(abs(strong.angle), 90)
        self.assertGreater(abs(strong.angle), abs(moderate.angle))
        self.assertGreater(strong.speed, moderate.speed)

    def test_alignment_hysteresis_prevents_threshold_chatter(self):
        controller = BallApproachController(start_time=0.0)
        entered = controller.update(
            detection(x=430, timestamp=0.1),
            self.shape,
            now=0.1,
        )
        held = controller.update(
            detection(x=385, timestamp=0.2),
            self.shape,
            now=0.2,
        )
        exited = controller.update(
            detection(x=360, timestamp=0.3),
            self.shape,
            now=0.3,
        )

        self.assertEqual(entered.state, controller.ALIGN)
        self.assertEqual(held.state, controller.ALIGN)
        self.assertEqual(exited.state, controller.APPROACH)

    def test_approach_steering_stays_below_gentle_limit(self):
        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(x=390, timestamp=0.1),
            self.shape,
            now=0.1,
        )

        self.assertEqual(command.state, controller.APPROACH)
        self.assertLessEqual(
            abs(command.angle),
            cfg.BALL_STEER_MAX_ANGLE,
        )

    def test_centered_target_approaches_and_slows_down(self):
        controller = BallApproachController(start_time=0.0)
        far = controller.update(
            detection(radius=22, timestamp=0.1), self.shape, now=0.1)
        near = controller.update(
            detection(radius=66, y=300, timestamp=0.2), self.shape, now=0.2)
        self.assertEqual(far.state, controller.APPROACH)
        self.assertEqual(far.angle, 0)
        self.assertLess(near.speed, far.speed)

    def test_close_crescent_requires_multiple_frames_and_latches(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)

        commands = []
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = history_end + 0.05 + index * 0.05
            commands.append(controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            ))

        self.assertEqual(commands[0].state, controller.NEAR_CONFIRM)
        self.assertEqual(commands[0].angle, 190)
        self.assertEqual(commands[0].speed, 0.0)
        self.assertEqual(commands[0].pickup_confirmations, 1)
        self.assertEqual(commands[1].pickup_confirmations, 2)
        self.assertEqual(commands[2].state, controller.NEAR)
        self.assertTrue(commands[2].terminal)
        self.assertIn("meia-lua", commands[2].detail)

        latched = controller.update(None, self.shape, now=1.0)
        self.assertEqual(latched.state, controller.NEAR)
        self.assertEqual(latched.angle, 190)

    def test_locked_circle_reaching_pickup_point_triggers_in_two_frames(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)

        commands = []
        for index in range(cfg.BALL_LOCKED_CIRCLE_CONFIRM_FRAMES):
            now = history_end + 0.05 + index * 0.05
            commands.append(controller.update(
                detection(
                    x=320,
                    y=390,
                    radius=52,
                    timestamp=now,
                    track_locked=True,
                ),
                self.shape,
                now=now,
            ))

        self.assertEqual(commands[0].state, controller.NEAR_CONFIRM)
        self.assertIn("circulo travado", commands[0].detail)
        self.assertEqual(commands[-1].state, controller.NEAR)
        self.assertTrue(commands[-1].terminal)

    def test_locked_circle_cannot_cold_start_pickup(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_LOCKED_CIRCLE_CONFIRM_FRAMES + 1):
            now = 0.05 + index * 0.05
            command = controller.update(
                detection(
                    x=320,
                    y=390,
                    radius=52,
                    timestamp=now,
                    track_locked=True,
                ),
                self.shape,
                now=now,
            )

        self.assertNotEqual(command.state, controller.NEAR)
        self.assertFalse(command.terminal)

    def test_locked_circle_rejects_small_or_offcenter_reflection(self):
        for x, y, radius in (
            (320, 415, 30),
            (420, 390, 80),
        ):
            with self.subTest(x=x, radius=radius):
                controller = BallApproachController(start_time=0.0)
                history_end = arm_crescent_history(
                    controller, self.shape)
                command = None
                for index in range(
                    cfg.BALL_LOCKED_CIRCLE_CONFIRM_FRAMES
                ):
                    now = history_end + 0.05 + index * 0.05
                    command = controller.update(
                        detection(
                            x=x,
                            y=y,
                            radius=radius,
                            timestamp=now,
                            track_locked=True,
                        ),
                        self.shape,
                        now=now,
                    )

                self.assertNotEqual(command.state, controller.NEAR)
                self.assertFalse(command.terminal)

    def test_locked_circle_confirmation_survives_one_missing_frame(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        first_time = history_end + 0.05
        first = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=first_time,
                track_locked=True,
            ),
            self.shape,
            now=first_time,
        )
        held = controller.update(
            None,
            self.shape,
            now=first_time + 0.05,
        )
        second_time = first_time + 0.10
        second = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=second_time,
                track_locked=True,
            ),
            self.shape,
            now=second_time,
        )

        self.assertEqual(first.pickup_confirmations, 1)
        self.assertEqual(held.state, controller.NEAR_CONFIRM)
        self.assertEqual(held.pickup_confirmations, 1)
        self.assertEqual(second.state, controller.NEAR)

    def test_locked_circle_confirmation_resets_after_two_missing_frames(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        first_time = history_end + 0.05
        first = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=first_time,
                track_locked=True,
            ),
            self.shape,
            now=first_time,
        )
        first_miss = controller.update(
            None,
            self.shape,
            now=first_time + 0.03,
        )
        second_miss = controller.update(
            None,
            self.shape,
            now=first_time + 0.06,
        )
        reacquired_time = first_time + 0.09
        reacquired = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=reacquired_time,
                track_locked=True,
            ),
            self.shape,
            now=reacquired_time,
        )

        self.assertEqual(first.pickup_confirmations, 1)
        self.assertEqual(first_miss.state, controller.NEAR_CONFIRM)
        self.assertNotEqual(second_miss.state, controller.NEAR_CONFIRM)
        self.assertEqual(reacquired.state, controller.NEAR_CONFIRM)
        self.assertEqual(reacquired.pickup_confirmations, 1)
        self.assertFalse(reacquired.terminal)

    def test_locked_circle_confirmation_expires_before_new_measurement(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        first_time = history_end + 0.05
        first = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=first_time,
                track_locked=True,
            ),
            self.shape,
            now=first_time,
        )
        late_time = (
            first_time + cfg.BALL_NEAR_CONFIRM_GRACE_S + 0.01)
        late = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=late_time,
                track_locked=True,
            ),
            self.shape,
            now=late_time,
        )

        self.assertEqual(first.pickup_confirmations, 1)
        self.assertEqual(late.state, controller.NEAR_CONFIRM)
        self.assertEqual(late.pickup_confirmations, 1)
        self.assertFalse(late.terminal)

    def test_duplicate_or_late_locked_circle_does_not_complete(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        timestamp = history_end + 0.05
        circle = detection(
            x=320,
            y=390,
            radius=52,
            timestamp=timestamp,
            track_locked=True,
        )
        first = controller.update(
            circle, self.shape, now=timestamp)
        duplicate = controller.update(
            circle, self.shape, now=timestamp + 0.02)

        late_time = (
            timestamp + cfg.BALL_NEAR_CONFIRM_WINDOW_S + 0.01)
        late = controller.update(
            detection(
                x=320,
                y=390,
                radius=52,
                timestamp=late_time,
                track_locked=True,
            ),
            self.shape,
            now=late_time,
        )

        self.assertEqual(first.pickup_confirmations, 1)
        self.assertEqual(duplicate.pickup_confirmations, 1)
        self.assertFalse(duplicate.terminal)
        self.assertEqual(late.pickup_confirmations, 1)
        self.assertFalse(late.terminal)

    def test_large_circle_inside_old_rectangle_cannot_trigger_pickup(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES + 1):
            now = 0.1 + index * 0.05
            command = controller.update(
                detection(
                    y=390,
                    radius=cfg.BALL_STOP_RADIUS_PX + 30,
                    timestamp=now,
                ),
                self.shape,
                now=now,
            )

        self.assertEqual(command.state, controller.APPROACH)
        self.assertFalse(command.terminal)
        self.assertFalse(command.pickup_in_range)

    def test_crescent_cannot_cold_arm_without_approach_history(self):
        controller = BallApproachController(start_time=0.0)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            )

        self.assertEqual(command.state, controller.WAIT_TARGET)
        self.assertFalse(command.terminal)
        self.assertFalse(command.pickup_in_range)

    def test_distant_target_plus_strong_crescent_is_rejected(self):
        controller = BallApproachController(start_time=0.0)
        controller.update(
            detection(
                y=220,
                radius=18,
                timestamp=0.05,
            ),
            self.shape,
            now=0.05,
        )
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = 0.1 + index * 0.05
            command = controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            )

        self.assertNotEqual(command.state, controller.NEAR)
        self.assertFalse(command.terminal)

    def test_stale_crescent_is_rejected(self):
        controller = BallApproachController(start_time=0.0)
        controller.update(
            detection(timestamp=0.1), self.shape, now=0.1)
        now = 0.1 + cfg.BALL_FRAME_STALE_S + 0.01
        command = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=0.1),
            now=now,
        )

        self.assertEqual(command.state, controller.LOST)
        self.assertFalse(command.pickup_in_range)

    def test_brief_crescent_interruption_keeps_confirmation_stopped(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        first = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(
                timestamp=history_end + 0.05),
            now=history_end + 0.05,
        )
        self.assertEqual(first.pickup_confirmations, 1)

        interrupted = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(
                timestamp=history_end + 0.10,
                accepted=False,
            ),
            now=history_end + 0.10,
        )
        self.assertEqual(interrupted.state, controller.NEAR_CONFIRM)
        self.assertEqual(interrupted.pickup_confirmations, 1)

        commands = []
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES - 1):
            now = history_end + 0.15 + index * 0.05
            commands.append(controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            ))
        self.assertEqual(commands[0].pickup_confirmations, 2)
        self.assertEqual(commands[-1].state, controller.NEAR)

    def test_long_crescent_interruption_restarts_confirmation(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        first_time = history_end + 0.05
        first = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(
                timestamp=first_time),
            now=first_time,
        )
        self.assertEqual(first.pickup_confirmations, 1)

        interrupted_at = (
            first_time + cfg.BALL_NEAR_CONFIRM_GRACE_S + 0.01)
        interrupted = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(
                timestamp=interrupted_at,
                accepted=False,
            ),
            now=interrupted_at,
        )
        self.assertEqual(interrupted.state, controller.LOST)

        restarted = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(
                timestamp=interrupted_at + 0.05),
            now=interrupted_at + 0.05,
        )
        self.assertEqual(restarted.pickup_confirmations, 1)

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

    def test_crescent_gate_is_resolution_independent(self):
        controller = BallApproachController(start_time=0.0)
        shape = (720, 960, 3)
        history_end = arm_crescent_history(controller, shape)
        command = None
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = history_end + 0.05 + index * 0.05
            command = controller.update(
                None,
                shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            )
        self.assertEqual(command.state, controller.NEAR)
        self.assertTrue(command.terminal)

    def test_black_ball_can_finish_with_the_same_crescent_gate(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(
            controller, self.shape, kind="black")

        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = history_end + 0.05 + index * 0.05
            command = controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            )
        self.assertEqual(command.state, controller.NEAR)
        self.assertTrue(command.terminal)

    def test_wide_fov_uses_isotropic_not_horizontal_scale(self):
        shape = (540, 960, 3)
        scale = cfg.ball_pixel_scale(shape[1], shape[0])
        self.assertAlmostEqual(scale, 1.125)

        controller = BallApproachController(start_time=0.0)
        command = controller.update(
            detection(
                x=shape[1] / 2,
                y=400,
                radius=cfg.BALL_SLOW_RADIUS_PX * scale,
                timestamp=0.1,
            ),
            shape,
            now=0.1,
        )
        self.assertEqual(command.state, controller.APPROACH)
        self.assertFalse(command.terminal)

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
        history_end = arm_crescent_history(controller, self.shape)
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES - 1):
            now = history_end + 0.05 + index * 0.05
            command = controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            )
            self.assertFalse(command.terminal)

        stale_now = history_end + cfg.BALL_FRAME_STALE_S + 1.0
        stale = controller.update(
            detection(timestamp=0.0),
            self.shape,
            now=stale_now)
        self.assertEqual(stale.state, controller.LOST)

        history_end = arm_crescent_history(
            controller,
            self.shape,
            start_time=stale_now + 0.05,
        )
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES - 1):
            now = history_end + 0.05 + index * 0.05
            command = controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            )
            self.assertFalse(command.terminal)
        now += 0.05
        command = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=now),
            now=now,
        )
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
        history_end = arm_crescent_history(controller, self.shape)
        controller.progress.clear()
        controller.progress.extend((
            (history_end - cfg.BALL_PROGRESS_WINDOW_S, 50.0, 380.0),
            (history_end, 50.0, 380.0),
        ))

        commands = []
        for index in range(cfg.BALL_STOP_CONFIRM_FRAMES):
            now = history_end + 0.04 + index * 0.04
            commands.append(controller.update(
                None,
                self.shape,
                crescent_evidence=crescent_evidence(timestamp=now),
                now=now,
            ))

        self.assertEqual(commands[0].state, controller.NEAR_CONFIRM)
        self.assertEqual(commands[-1].state, controller.NEAR)

    def test_duplicate_crescent_timestamp_does_not_add_confirmation(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        timestamp = history_end + 0.05
        first = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=timestamp),
            now=timestamp,
        )
        repeated = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=timestamp),
            now=timestamp + 0.01,
        )

        self.assertEqual(first.pickup_confirmations, 1)
        self.assertEqual(repeated.pickup_confirmations, 1)
        self.assertFalse(repeated.terminal)

    def test_expired_approach_token_cannot_arm_crescent(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        now = history_end + cfg.BALL_CRESCENT_TOKEN_TTL_S + 0.01
        command = controller.update(
            None,
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=now),
            now=now,
        )

        self.assertEqual(command.state, controller.LOST)
        self.assertFalse(command.pickup_in_range)

    def test_current_distant_circle_vetoes_crescent(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        now = history_end + 0.05
        command = controller.update(
            detection(
                y=220,
                radius=18,
                timestamp=now,
            ),
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=now),
            now=now,
        )

        self.assertEqual(command.state, controller.APPROACH)
        self.assertEqual(command.pickup_confirmations, 0)
        self.assertFalse(command.pickup_in_range)

    def test_close_offcenter_inner_reflection_does_not_veto_dome(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        now = history_end + 0.05
        command = controller.update(
            detection(
                x=424,
                y=379,
                radius=37,
                timestamp=now,
            ),
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=now),
            now=now,
        )

        self.assertEqual(command.state, controller.NEAR_CONFIRM)
        self.assertEqual(command.pickup_confirmations, 1)

    def test_small_but_deep_inner_reflection_does_not_veto_dome(self):
        controller = BallApproachController(start_time=0.0)
        history_end = arm_crescent_history(controller, self.shape)
        now = history_end + 0.05
        command = controller.update(
            detection(
                x=390,
                y=390,
                radius=24,
                timestamp=now,
            ),
            self.shape,
            crescent_evidence=crescent_evidence(timestamp=now),
            now=now,
        )

        self.assertEqual(command.state, controller.NEAR_CONFIRM)
        self.assertEqual(command.pickup_confirmations, 1)

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
        self.assertLessEqual(abs(left_command.angle), 110)
        self.assertGreater(fake.calls[-1][1], 0)
        self.assertGreater(fake.calls[-1][2], 0)
        self.assertLess(fake.calls[-1][1], fake.calls[-1][2])
        self.assertGreaterEqual(
            fake.calls[-1][1],
            0.20 * fake.calls[-1][2],
        )

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
