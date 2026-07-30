"""Testes da busca ciclica por giro tanque."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.busca_resgate import BallSearchController  # noqa: E402
from visao.deteccao import VictimDetection as BallDetection  # noqa: E402


def _detection(
    timestamp,
    kind="silver",
    confirmed=True,
    track_locked=True,
    confidence=0.90,
    hits=3,
):
    return BallDetection(
        kind=kind,
        center_x=320.0,
        center_y=260.0,
        radius=40.0,
        confidence=confidence,
        confirmed=confirmed,
        hits=hits,
        timestamp=float(timestamp),
        track_locked=track_locked,
    )


class BallSearchControllerTests(unittest.TestCase):
    def test_requested_search_uses_tank_turn(self):
        search = BallSearchController(start_time=10.0)

        command = search.update(None, now=10.0)

        self.assertEqual(command.state, search.START)
        self.assertEqual(command.angle, 180)
        self.assertEqual(
            round(command.speed * 120 * 1.2),
            cfg.BALL_SEARCH_TANK_PWM,
        )
        self.assertEqual(cfg.BALL_SEARCH_TANK_PWM, 80)
        self.assertEqual(cfg.BALL_SEARCH_FULL_TURN_S, 3.54)
        self.assertFalse(command.terminal)

    def test_full_turn_timer_only_starts_after_serial_ack(self):
        search = BallSearchController(start_time=0.0)

        self.assertEqual(
            search.update(None, now=100.0).state,
            search.START,
        )
        search.mark_rotation_started(now=100.0)

        before = search.update(
            None,
            now=100.0 + cfg.BALL_SEARCH_FULL_TURN_S - 0.001,
        )
        complete = search.update(
            None,
            now=100.0 + cfg.BALL_SEARCH_FULL_TURN_S,
        )

        self.assertEqual(before.state, search.ROTATING)
        self.assertFalse(before.terminal)
        self.assertEqual(complete.state, search.TURN_STOP)
        self.assertEqual(complete.angle, 190)
        self.assertFalse(complete.terminal)

        stopped_at = 100.0 + cfg.BALL_SEARCH_FULL_TURN_S
        search.mark_full_turn_stopped(now=stopped_at)
        checking = search.update(
            None,
            now=(
                stopped_at
                + cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
                - 0.001
            ),
        )
        terminal = search.update(
            None,
            now=stopped_at + cfg.BALL_SEARCH_VERIFY_TIMEOUT_S,
        )
        self.assertEqual(checking.state, search.FINAL_VERIFY)
        self.assertFalse(checking.terminal)
        self.assertEqual(terminal.state, search.COMPLETE)
        self.assertTrue(terminal.terminal)

    def test_valid_target_wins_at_exact_full_turn_deadline(self):
        search = BallSearchController(start_time=0.0)
        search.update(None, now=0.0)
        search.mark_rotation_started(now=0.0)
        deadline = cfg.BALL_SEARCH_FULL_TURN_S

        command = search.update(
            _detection(deadline),
            now=deadline,
        )

        self.assertEqual(command.state, search.TARGET_STOP)
        self.assertFalse(command.terminal)
        self.assertEqual(command.angle, 190)

    def test_final_stationary_check_can_still_acquire_target(self):
        search = BallSearchController(start_time=0.0)
        search.update(None, now=0.0)
        search.mark_rotation_started(now=0.0)
        deadline = cfg.BALL_SEARCH_FULL_TURN_S
        stop = search.update(None, now=deadline)
        self.assertEqual(stop.state, search.TURN_STOP)
        search.mark_full_turn_stopped(now=deadline)

        acquired = search.update(
            _detection(deadline + 0.10),
            now=deadline + 0.10,
        )

        self.assertEqual(acquired.state, search.ACQUIRED)
        self.assertEqual(acquired.target_kind, "silver")

    def test_target_is_reconfirmed_only_with_post_stop_frame(self):
        search = BallSearchController(start_time=0.0)
        stop = search.update(_detection(0.0), now=0.0)
        self.assertEqual(stop.state, search.TARGET_STOP)
        search.mark_target_stopped(now=0.10)

        self.assertFalse(search.frame_allowed(0.10))
        self.assertTrue(search.frame_allowed(0.11))
        old = search.update(_detection(0.09), now=0.20)
        acquired = search.update(_detection(0.21), now=0.21)

        self.assertEqual(old.state, search.VERIFY)
        self.assertEqual(acquired.state, search.ACQUIRED)
        self.assertTrue(search.target_acquired)
        self.assertEqual(search.target_kind, "silver")

    def test_first_plausible_candidate_brakes_during_rotation(self):
        search = BallSearchController(start_time=0.0)
        search.update(None, now=0.0)
        search.mark_rotation_started(now=0.0)

        stop = search.update(
            _detection(
                0.10,
                confirmed=False,
                track_locked=False,
                hits=1,
            ),
            now=0.10,
        )

        self.assertEqual(stop.state, search.TARGET_STOP)
        self.assertEqual(stop.angle, 190)
        self.assertIn("freando", stop.detail)

    def test_tentative_candidate_still_requires_stationary_confirmation(self):
        search = BallSearchController(start_time=0.0)
        stop = search.update(
            _detection(
                0.0,
                kind="silver",
                confirmed=False,
                track_locked=False,
                hits=1,
            ),
            now=0.0,
        )
        self.assertEqual(stop.state, search.TARGET_STOP)
        search.mark_target_stopped(now=0.10)

        still_waiting = search.update(
            _detection(
                0.20,
                kind="black",
                confirmed=False,
                track_locked=False,
                hits=2,
            ),
            now=0.20,
        )
        acquired = search.update(
            _detection(0.30, kind="black"),
            now=0.30,
        )

        self.assertEqual(still_waiting.state, search.VERIFY)
        self.assertEqual(acquired.state, search.ACQUIRED)
        self.assertEqual(acquired.target_kind, "black")

    def test_weak_candidate_does_not_interrupt_turn(self):
        search = BallSearchController(start_time=0.0)
        search.update(None, now=0.0)
        search.mark_rotation_started(now=0.0)

        rotating = search.update(
            _detection(
                0.10,
                confirmed=False,
                track_locked=False,
                confidence=(
                    cfg.BALL_SEARCH_BRAKE_MIN_CONFIDENCE - 0.01
                ),
                hits=1,
            ),
            now=0.10,
        )

        self.assertEqual(rotating.state, search.ROTATING)
        self.assertEqual(rotating.angle, 180)

    def test_verification_timeout_discards_track_and_restarts_turn(self):
        search = BallSearchController(start_time=0.0)
        search.update(_detection(0.0), now=0.0)
        search.mark_target_stopped(now=0.10)

        restart = search.update(
            None,
            now=0.10 + cfg.BALL_SEARCH_VERIFY_TIMEOUT_S,
        )

        self.assertEqual(restart.state, search.START)
        self.assertEqual(restart.angle, 180)
        self.assertTrue(search.consume_tracking_reset())
        self.assertFalse(search.consume_tracking_reset())
        search.mark_rotation_started(now=1.10)
        self.assertEqual(search.state, search.ROTATING)

    def test_false_candidate_resumes_only_remaining_rotation_time(self):
        search = BallSearchController(start_time=0.0)
        search.update(None, now=0.0)
        search.mark_rotation_started(now=0.0)
        stop = search.update(
            _detection(
                2.0,
                confirmed=False,
                track_locked=False,
                hits=1,
            ),
            now=2.0,
        )
        self.assertEqual(stop.state, search.TARGET_STOP)
        search.mark_target_stopped(now=2.0)

        restart = search.update(
            None,
            now=2.0 + cfg.BALL_SEARCH_VERIFY_TIMEOUT_S,
        )
        self.assertEqual(restart.state, search.START)
        search.mark_rotation_started(now=3.0)
        remaining = cfg.BALL_SEARCH_FULL_TURN_S - 2.0

        before = search.update(
            None,
            now=3.0 + remaining - 0.001,
        )
        finished = search.update(
            None,
            now=3.0 + remaining,
        )

        self.assertEqual(before.state, search.ROTATING)
        self.assertEqual(finished.state, search.TURN_STOP)

    def test_other_color_cannot_replace_target_during_stationary_check(self):
        search = BallSearchController(start_time=0.0)
        search.update(_detection(0.0, kind="silver"), now=0.0)
        search.mark_target_stopped(now=0.10)

        command = search.update(
            _detection(0.20, kind="black"),
            now=0.20,
        )

        self.assertEqual(command.state, search.VERIFY)
        self.assertEqual(search.target_kind, "silver")


# Testes da cola de orquestracao do resgate.py removidos: essa cola saiu do escopo atual (ver o docstring de resgate.py). Os modulos que ela orquestrava continuam no repositorio, com seus testes.


if __name__ == "__main__":
    unittest.main()
