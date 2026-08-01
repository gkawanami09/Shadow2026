"""Testes da navegacao visual ate os triangulos de evacuacao."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.coleta_resgate import BallPickupSequencer  # noqa: E402
from controle.deposito_resgate import DepositMarkerController  # noqa: E402


FRAME_SHAPE = (480, 640, 3)


def pickup_ready(target_kind):
    """Avanca pela coleta real ate a esfera estar presa e elevada."""
    pickup = BallPickupSequencer()
    pickup.start(target_kind)
    now = 0.0

    pickup.update(now=now)
    pickup.mark_futaba_started(now=now)
    now += (
        cfg.BALL_PICKUP_FUTABA_MS / 1000.0
        + cfg.BALL_PICKUP_FUTABA_GUARD_S
    )
    pickup.update(now=now)
    pickup.mark_forward_started(now=now)
    now += cfg.BALL_PICKUP_FORWARD_LEAD_S
    pickup.update(now=now)
    now += cfg.BALL_PICKUP_FINAL_FORWARD_S
    pickup.update(now=now)
    pickup.mark_grippers_started(now=now)
    now += cfg.BALL_PICKUP_GRIPPER_SETTLE_S
    pickup.update(now=now)
    pickup.mark_futaba_started(now=now)
    now += (
        cfg.BALL_PICKUP_LIFT_MS / 1000.0
        + cfg.BALL_PICKUP_LIFT_GUARD_S
    )
    pickup.update(now=now)

    if not pickup.ready_for_deposit:
        raise AssertionError("fixture nao chegou ao estado de transporte")
    return pickup


def marker(
    timestamp,
    kind="green",
    center_x=320.0,
    center_y=330.0,
    width=120.0,
    height=100.0,
    bottom_y=380.0,
    confidence=0.90,
    confirmed=True,
    track_locked=True,
):
    return SimpleNamespace(
        kind=kind,
        center_x=float(center_x),
        center_y=float(center_y),
        width=float(width),
        height=float(height),
        bottom_y=float(bottom_y),
        area=float(width * height * 0.5),
        confidence=float(confidence),
        confirmed=bool(confirmed),
        hits=3 if confirmed else 1,
        timestamp=float(timestamp),
        track_locked=bool(track_locked),
    )


class DepositMarkerControllerTests(unittest.TestCase):
    def test_ball_kind_mapping_is_immutable(self):
        self.assertEqual(
            cfg.DEPOSIT_MARKER_BY_BALL_KIND,
            {"silver": "green", "black": "red"},
        )

    def test_search_is_slow_tank_and_timer_starts_only_after_ack(self):
        controller = DepositMarkerController("green", start_time=0.0)

        start = controller.update(None, FRAME_SHAPE, now=100.0)
        self.assertEqual(start.state, controller.START)
        self.assertEqual(start.angle, 180)
        self.assertEqual(start.speed, cfg.DEPOSIT_SEARCH_TANK_SPEED)

        controller.mark_rotation_started(now=100.0)
        before = controller.update(
            None,
            FRAME_SHAPE,
            now=100.0 + cfg.DEPOSIT_SEARCH_FULL_TURN_S - 0.001,
        )
        stop = controller.update(
            None,
            FRAME_SHAPE,
            now=100.0 + cfg.DEPOSIT_SEARCH_FULL_TURN_S,
        )
        self.assertEqual(before.state, controller.ROTATING)
        self.assertEqual(stop.state, controller.TURN_STOP)
        self.assertEqual(stop.angle, 190)

    def test_busca_pulsada_gira_para_observa_e_so_entao_aceita_vermelho(self):
        controller = DepositMarkerController(
            "red",
            start_time=0.0,
            pulsed_search=True,
        )

        start = controller.update(None, FRAME_SHAPE, now=0.0)
        self.assertEqual(start.state, controller.PULSE_BRAKE)
        self.assertEqual(start.angle, 190)
        controller.mark_pulse_stopped(now=0.0)

        assentou_inicial = cfg.DEPOSIT_SEARCH_SETTLE_S
        controller.update(None, FRAME_SHAPE, now=assentou_inicial)
        iniciou_em = (
            assentou_inicial + cfg.DEPOSIT_SEARCH_OBSERVE_TIMEOUT_S)
        liberou = controller.update(None, FRAME_SHAPE, now=iniciou_em)
        self.assertEqual(liberou.state, controller.START)
        controller.mark_rotation_started(now=iniciou_em)

        girando = controller.update(
            None,
            FRAME_SHAPE,
            now=iniciou_em + cfg.DEPOSIT_SEARCH_PULSE_S - 0.001,
        )
        freando = controller.update(
            None,
            FRAME_SHAPE,
            now=iniciou_em + cfg.DEPOSIT_SEARCH_PULSE_S,
        )
        self.assertEqual(girando.state, controller.ROTATING)
        self.assertEqual(freando.state, controller.PULSE_BRAKE)
        self.assertEqual(freando.angle, 190)

        fim_pulso = iniciou_em + cfg.DEPOSIT_SEARCH_PULSE_S
        controller.mark_pulse_stopped(now=fim_pulso)
        assentando = controller.update(
            None,
            FRAME_SHAPE,
            now=fim_pulso + cfg.DEPOSIT_SEARCH_SETTLE_S - 0.001,
        )
        observando_em = fim_pulso + cfg.DEPOSIT_SEARCH_SETTLE_S
        observando = controller.update(
            None,
            FRAME_SHAPE,
            now=observando_em,
        )
        self.assertEqual(assentando.state, controller.PULSE_SETTLE)
        self.assertEqual(observando.state, controller.PULSE_OBSERVE)
        self.assertEqual(observando.angle, 190)

        antigo = controller.update(
            marker(observando_em - 0.01, kind="red"),
            FRAME_SHAPE,
            now=observando_em + 0.01,
        )
        encontrado = controller.update(
            marker(observando_em + 0.02, kind="red"),
            FRAME_SHAPE,
            now=observando_em + 0.02,
        )
        self.assertEqual(antigo.state, controller.PULSE_OBSERVE)
        self.assertEqual(encontrado.state, controller.TARGET_STOP)

        controller.mark_target_stopped(now=observando_em + 0.03)
        centralizando = controller.update(
            marker(
                observando_em + 0.04,
                kind="red",
                center_x=520.0,
            ),
            FRAME_SHAPE,
            now=observando_em + 0.04,
        )
        self.assertEqual(centralizando.state, controller.ALIGN)

    def test_vermelho_visivel_antes_do_primeiro_giro_e_preservado(self):
        controller = DepositMarkerController(
            "red",
            start_time=0.0,
            pulsed_search=True,
        )
        parar = controller.update(None, FRAME_SHAPE, now=0.0)
        controller.mark_pulse_stopped(now=0.0)
        assentou = cfg.DEPOSIT_SEARCH_SETTLE_S
        controller.update(None, FRAME_SHAPE, now=assentou)

        encontrado = controller.update(
            marker(assentou + 0.01, kind="red"),
            FRAME_SHAPE,
            now=assentou + 0.02,
        )

        self.assertEqual(parar.state, controller.PULSE_BRAKE)
        self.assertEqual(encontrado.state, controller.TARGET_STOP)
        self.assertAlmostEqual(controller._rotation_elapsed_s, 0.0)

    def test_wrong_color_is_ignored(self):
        controller = DepositMarkerController("green", start_time=0.0)

        command = controller.update(
            marker(0.0, kind="red"),
            FRAME_SHAPE,
            now=0.0,
        )

        self.assertEqual(command.state, controller.START)
        self.assertEqual(command.angle, 180)

    def test_tentative_marker_brakes_but_pre_stop_frame_cannot_confirm(self):
        controller = DepositMarkerController("green", start_time=0.0)
        tentative = marker(
            0.0,
            confirmed=False,
            track_locked=False,
        )

        stop = controller.update(
            tentative, FRAME_SHAPE, now=0.0)
        self.assertEqual(stop.state, controller.TARGET_STOP)
        controller.mark_target_stopped(now=0.10)
        self.assertFalse(controller.frame_allowed(0.10))
        self.assertTrue(controller.frame_allowed(0.11))

        old = controller.update(
            marker(0.09), FRAME_SHAPE, now=0.20)
        acquired = controller.update(
            marker(0.21), FRAME_SHAPE, now=0.21)
        self.assertEqual(old.state, controller.VERIFY)
        self.assertEqual(acquired.state, controller.APPROACH)

    def test_align_direction_and_speed_are_gentle(self):
        for center_x, sign in ((520.0, 1), (120.0, -1)):
            with self.subTest(center_x=center_x):
                controller = DepositMarkerController(
                    "green", start_time=0.0)
                controller.state = controller.APPROACH

                command = controller.update(
                    marker(0.0, center_x=center_x),
                    FRAME_SHAPE,
                    now=0.0,
                )

                self.assertEqual(command.state, controller.ALIGN)
                self.assertEqual(
                    1 if command.angle > 0 else -1,
                    sign,
                )
                self.assertLessEqual(
                    command.speed,
                    cfg.DEPOSIT_ALIGN_SPEED_MAX,
                )

    def test_arrival_needs_three_distinct_frames_and_stop_ack(self):
        controller = DepositMarkerController("red", start_time=0.0)
        controller.state = controller.APPROACH
        near_kwargs = {
            "kind": "red",
            "width": 220.0,
            "bottom_y": 440.0,
        }

        first = controller.update(
            marker(0.00, **near_kwargs),
            FRAME_SHAPE,
            now=0.00,
        )
        repeated = controller.update(
            marker(0.00, **near_kwargs),
            FRAME_SHAPE,
            now=0.05,
        )
        second = controller.update(
            marker(0.10, **near_kwargs),
            FRAME_SHAPE,
            now=0.10,
        )
        third = controller.update(
            marker(0.20, **near_kwargs),
            FRAME_SHAPE,
            now=0.20,
        )

        self.assertEqual(first.angle, 190)
        self.assertEqual(repeated.state, controller.APPROACH)
        self.assertEqual(second.state, controller.APPROACH)
        self.assertEqual(third.state, controller.ARRIVAL_STOP)
        self.assertFalse(third.terminal)
        controller.mark_arrival_stopped(now=0.21)
        arrived = controller.update(
            None, FRAME_SHAPE, now=0.21)
        self.assertEqual(arrived.state, controller.ARRIVED)
        self.assertTrue(arrived.terminal)

    def test_confirmacao_visual_pode_ser_configurada_sem_mudar_o_padrao(self):
        controller = DepositMarkerController(
            "green",
            start_time=0.0,
            near_confirm_frames=1,
        )
        controller.state = controller.APPROACH

        command = controller.update(
            marker(0.10, width=220.0, bottom_y=440.0),
            FRAME_SHAPE,
            now=0.10,
        )

        self.assertEqual(command.state, controller.ARRIVAL_STOP)
        self.assertEqual(command.angle, 190)
        self.assertFalse(command.terminal)

    def test_loss_stops_before_search_can_resume(self):
        controller = DepositMarkerController("green", start_time=0.0)
        controller.state = controller.APPROACH
        controller._last_seen_at = 0.0

        stop = controller.update(
            None,
            FRAME_SHAPE,
            now=cfg.DEPOSIT_REACQUIRE_TIMEOUT_S,
        )

        self.assertEqual(stop.state, controller.LOST_STOP)
        self.assertEqual(stop.angle, 190)
        controller.mark_lost_stopped(now=1.0)
        self.assertTrue(controller.consume_tracking_reset())
        restart = controller.update(
            None, FRAME_SHAPE, now=1.0)
        self.assertEqual(restart.state, controller.START)
        self.assertEqual(restart.angle, 180)

    def test_persistent_marker_without_visual_progress_faults(self):
        controller = DepositMarkerController("green", start_time=0.0)
        controller.state = controller.APPROACH
        controller._active_started_at = 0.0
        stuck = marker(
            0.0,
            width=120.0,
            bottom_y=380.0,
        )

        moving = controller.update(
            stuck, FRAME_SHAPE, now=0.0)
        fault = controller.update(
            marker(
                cfg.DEPOSIT_PROGRESS_TIMEOUT_S,
                width=120.0,
                bottom_y=380.0,
            ),
            FRAME_SHAPE,
            now=cfg.DEPOSIT_PROGRESS_TIMEOUT_S,
        )

        self.assertEqual(moving.state, controller.APPROACH)
        self.assertEqual(fault.state, controller.FAULT)
        self.assertTrue(fault.terminal)
        self.assertIn("mantida", fault.detail)

    def test_progress_resets_short_watchdog_but_global_timeout_still_stops(self):
        controller = DepositMarkerController("red", start_time=0.0)
        controller.state = controller.APPROACH
        controller._active_started_at = 0.0

        controller.update(
            marker(0.0, kind="red", width=100.0, bottom_y=350.0),
            FRAME_SHAPE,
            now=0.0,
        )
        progressing = controller.update(
            marker(
                5.0,
                kind="red",
                width=120.0,
                bottom_y=370.0,
            ),
            FRAME_SHAPE,
            now=5.0,
        )
        global_fault = controller.update(
            None,
            FRAME_SHAPE,
            now=cfg.DEPOSIT_MAX_ACTIVE_S,
        )

        self.assertNotEqual(progressing.state, controller.FAULT)
        self.assertEqual(global_fault.state, controller.FAULT)
        self.assertTrue(global_fault.terminal)

    def test_no_marker_after_full_turn_faults_without_authorizing_deposit(self):
        controller = DepositMarkerController("red", start_time=0.0)
        controller.update(None, FRAME_SHAPE, now=0.0)
        controller.mark_rotation_started(now=0.0)
        turn_stop = controller.update(
            None,
            FRAME_SHAPE,
            now=cfg.DEPOSIT_SEARCH_FULL_TURN_S,
        )
        self.assertEqual(turn_stop.state, controller.TURN_STOP)
        controller.mark_full_turn_stopped(
            now=cfg.DEPOSIT_SEARCH_FULL_TURN_S)

        fault = controller.update(
            None,
            FRAME_SHAPE,
            now=(
                cfg.DEPOSIT_SEARCH_FULL_TURN_S
                + cfg.DEPOSIT_SEARCH_VERIFY_TIMEOUT_S
            ),
        )

        self.assertEqual(fault.state, controller.FAULT)
        self.assertTrue(fault.terminal)
        self.assertFalse(controller.arrived)
        self.assertIn("mantida", fault.detail)

    # Testes da cola de orquestracao removidos junto com ela: a
    # coleta, o deposito e os codigos de saida da missao sairam do
    # escopo atual do resgate.py. Os modulos continuam no repo.


if __name__ == "__main__":
    unittest.main()
