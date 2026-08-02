"""Testes do verificador nao bloqueante de parede junto da vitima."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.parede_vitima import (  # noqa: E402
    INCONCLUSIVO,
    LIVRE,
    PAREDE_DESALINHADA,
    PAREDE_RETA,
    WallPickupAuthorization,
    WallProbeController,
    WallTargetSignature,
    aplicar_acao_parede,
)
from visao.deteccao import VictimDetection  # noqa: E402


TIMEOUT_ULTRASSOM = object()


class ArduinoFalso:
    connected = True
    connection_epoch = 7

    def __init__(self, distances):
        self.distances = list(distances)
        self.pending = False
        self.calls = []
        self.ultima_leitura_ultrassom_respondeu = False

    def iniciar_ultrassom(self, timeout):
        if self.pending:
            return False
        self.pending = True
        self.calls.append(("start_ultra", timeout))
        return True

    def poll_ultrassom(self):
        if not self.pending:
            return False, None
        self.pending = False
        value = self.distances.pop(0)
        self.ultima_leitura_ultrassom_respondeu = (
            value is not TIMEOUT_ULTRASSOM)
        return True, None if value is TIMEOUT_ULTRASSOM else value

    def cancelar_ultrassom(self):
        self.pending = False
        self.calls.append(("cancel_ultra",))

    def rodas(self, *values):
        self.calls.append(("rodas",) + tuple(values))
        return True

    def parar(self):
        self.calls.append(("parar",))
        return True


def detection(
    now,
    center_x=320,
    center_y=300,
    kind="silver",
    radius=55,
    truncated=False,
):
    return VictimDetection(
        kind,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        confidence=0.95,
        confirmed=True,
        hits=5,
        timestamp=now,
        track_locked=True,
        truncated=truncated,
    )


def run_probe(
    distances,
    visual_mode="normal",
    recenter_errors=None,
    scan_errors=None,
    restore_errors=None,
    depth_frames=None,
):
    target = detection(0.0)
    controller = WallProbeController(
        "silver", target_detection=target, start_time=0.0)
    arduino = ArduinoFalso(distances)
    recenter_errors = list(recenter_errors or [])
    scan_errors = list(scan_errors or [])
    restore_errors = list(restore_errors or [])
    depth_frames = list(depth_frames or [])
    now = 0.0
    steps = []
    for _ in range(1200):
        seen = None
        if controller.state == controller.LEFT_VERIFY:
            center_x = 320 if visual_mode == "no_shift" else 500
            seen = detection(now, center_x=center_x)
        elif controller.state == controller.RIGHT_VERIFY:
            center_x = 320 if visual_mode == "no_shift" else 140
            seen = detection(now, center_x=center_x)
        elif controller.state == controller.CENTER_VERIFY:
            if visual_mode == "no_return":
                center_x = 500
            elif visual_mode == "wall_loose_center":
                center_x = 368  # erro 0,15: passava no limite antigo 0,22
            else:
                center_x = 320
            if depth_frames:
                center_y, radius = depth_frames.pop(0)
            else:
                center_y, radius = 300, 55
            seen = detection(
                now,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
            )
        elif controller.state == controller.DEPTH_VERIFY:
            if depth_frames:
                center_y, radius = depth_frames.pop(0)
            else:
                center_y, radius = 300, 55
            seen = detection(now, center_y=center_y, radius=radius)
        elif controller.state == controller.RECENTER_VERIFY:
            if recenter_errors:
                error = recenter_errors.pop(0)
                seen = detection(now, center_x=320 + error * 320)
        elif controller.state == controller.SCAN_VERIFY:
            if scan_errors:
                error = scan_errors.pop(0)
            else:
                sign = -1 if controller._scan_side == "right" else 1
                error = (
                    controller._scan_baseline_error
                    + sign * 0.10 * controller._scan_outward_pulses
                )
            seen = detection(now, center_x=320 + error * 320)
        elif controller.state == controller.RESTORE_VERIFY:
            error = (
                restore_errors.pop(0)
                if restore_errors else controller._scan_baseline_error
            )
            seen = detection(now, center_x=320 + error * 320)
        elif controller.state == controller.SCAN_LEFT_VERIFY:
            seen = detection(now, center_x=500)
        step = controller.update(
            arduino,
            detection=seen,
            frame_shape=(480, 640, 3),
            now=now,
        )
        steps.append(step)
        if step.motor_action:
            error = aplicar_acao_parede(
                step, arduino, epoca_serial_esperada=7)
            if error is not None:
                raise AssertionError(error)
            if not step.terminal:
                controller.notify_command_written(step.state, now=now)
        if step.terminal:
            return controller, arduino, steps, step
        now += 0.07
    raise AssertionError("verificador nao chegou a um estado terminal")


class WallProbeControllerTests(unittest.TestCase):
    def test_requires_original_confirmed_target(self):
        with self.assertRaisesRegex(ValueError, "deteccao confirmada"):
            WallProbeController("silver", None, start_time=0.0)
        with self.assertRaisesRegex(ValueError, "mesma vitima"):
            WallProbeController(
                "silver", detection(0.0, kind="black"), start_time=0.0)

    def test_center_far_releases_only_after_bilateral_and_angular_scans(self):
        _controller, arduino, _steps, final = run_probe(
            [400, 410, 390] + [None] * 24)

        self.assertEqual(final.result, LIVRE)
        self.assertTrue(any(call[0] == "rodas" for call in arduino.calls))

    def test_center_without_echo_runs_bilateral_and_never_releases_pickup(self):
        controller, arduino, _steps, final = run_probe(
            [None] * 27)

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao provaram caminho livre", final.detail)
        self.assertEqual(controller.center_samples, [(None, None, None)])
        self.assertTrue(any(call[0] == "rodas" for call in arduino.calls))

    def test_center_serial_timeout_is_inconclusive(self):
        _controller, arduino, _steps, final = run_probe(
            [TIMEOUT_ULTRASSOM] * 3)

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("ausentes", final.detail)
        self.assertFalse(any(call[0] == "rodas" for call in arduino.calls))

    def test_one_center_timeout_cannot_be_hidden_by_valid_echoes(self):
        _controller, arduino, _steps, final = run_probe(
            [100, 101, TIMEOUT_ULTRASSOM])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("ausentes", final.detail)
        self.assertFalse(any(call[0] == "rodas" for call in arduino.calls))

    def test_one_near_center_echo_blocks_far_numeric_shortcut(self):
        _controller, arduino, _steps, final = run_probe([
            400, 410, 100,
            *([None] * 24),
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertTrue(any(call[0] == "rodas" for call in arduino.calls))

    def test_wall_authorization_only_matches_same_target_before_expiry(self):
        signature = WallTargetSignature.from_detection(
            detection(1.0, radius=55))
        authorization = WallPickupAuthorization(
            target_kind="silver",
            wall_mode=True,
            expires_at=5.0,
            signature=signature,
        )

        self.assertTrue(authorization.matches(
            detection(2.0, radius=58), (480, 640, 3), now=2.0))
        self.assertFalse(authorization.matches(
            detection(6.0, radius=58), (480, 640, 3), now=6.0))
        self.assertFalse(authorization.matches(
            detection(2.0, radius=12), (480, 640, 3), now=2.0))

    def test_both_sides_confirm_probable_straight_wall_and_return(self):
        _controller, arduino, _steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertEqual(final.motor_action, "stop")
        wheels = [call for call in arduino.calls if call[0] == "rodas"]
        self.assertEqual(len(wheels), 3)
        self.assertEqual(wheels[0], wheels[2])
        self.assertEqual(
            wheels[1][1:],
            tuple(-value for value in wheels[0][1:]),
        )

    def test_straight_wall_requires_final_center_inside_point_zero_eight(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], visual_mode="wall_loose_center")

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("mesma bolinha no centro", final.detail)

    def test_final_depth_advances_in_short_pulses_and_reconfirms_near(self):
        controller, arduino, steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], depth_frames=[
            (235, 36),             # retorno longe do NEAR original
            (270, 45),             # primeiro pulso realmente aproxima
            (300, 55), (300, 55),  # dois frames finais independentes
        ])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertEqual(controller._depth_pulses, 2)
        self.assertEqual(sum(
            step.motor_action == "forward" for step in steps), 2)
        actions = [call for call in arduino.calls
                   if call[0] in ("rodas", "parar")]
        for index, call in enumerate(actions[:-1]):
            if call == ("rodas", 40, 40, 40, 40):
                self.assertEqual(actions[index + 1], ("parar",))

    def test_one_near_frame_followed_by_far_frame_cannot_finish(self):
        _controller, _arduino, steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], depth_frames=[
            (300, 55),            # uma confirmacao ainda nao autoriza
            (235, 36),            # o segundo frame invalida a bateria
            (300, 55), (300, 55),
        ])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertTrue(any(
            step.motor_action == "forward" for step in steps))

    def test_final_depth_reverses_when_ball_is_too_close(self):
        controller, arduino, steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], depth_frames=[
            (315, 70),
            (300, 55), (300, 55),
        ])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertEqual(controller._depth_pulses, 1)
        self.assertEqual(sum(
            step.motor_action == "reverse" for step in steps), 1)
        reverse_index = arduino.calls.index(
            ("rodas", -40, -40, -40, -40))
        self.assertEqual(arduino.calls[reverse_index + 1], ("parar",))

    def test_final_depth_without_progress_fails_closed(self):
        controller, _arduino, _steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], depth_frames=[
            (235, 36), (235, 36), (235, 36),
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao aproximou", final.detail)
        self.assertEqual(controller._depth_pulses, 2)

    def test_final_depth_has_six_pulse_hard_limit_even_with_progress(self):
        controller, _arduino, _steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], depth_frames=[
            (230, 36),
            (238, 38), (246, 40), (254, 42),
            (262, 44), (270, 45), (276, 46),
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("limite de pulsos longitudinais", final.detail)
        self.assertEqual(controller._depth_pulses, 6)

    def test_conflicting_radius_and_bottom_never_move_longitudinally(self):
        _controller, arduino, steps, final = run_probe([
            100, 104, 102,
            112, 110, 114,
            118, 115, 116,
        ], depth_frames=[
            # Raio pede frente, mas a base mais baixa pede re.
            (365, 40),
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("discordam", final.detail)
        self.assertFalse(any(
            step.motor_action in ("forward", "reverse") for step in steps))
        self.assertNotIn(("rodas", 40, 40, 40, 40), arduino.calls)

    def test_echo_disappears_on_both_sides(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 101, 99,
            *([None] * 24),
        ])

        self.assertEqual(final.result, LIVRE)
        self.assertIn("duas varreduras", final.detail)

    def test_empty_scans_are_mirrored_and_each_pulse_stops(self):
        _controller, arduino, steps, final = run_probe(
            [100, 101, 99] + [None] * 24)

        self.assertEqual(final.result, LIVRE)
        scan_actions = [
            step.motor_action for step in steps
            if step.state in (
                WallProbeController.SCAN_PIVOT_LEFT_PENDING,
                WallProbeController.SCAN_PIVOT_RIGHT_PENDING,
            )
        ]
        self.assertEqual(
            scan_actions,
            ["pivot_right"] * 3 + ["pivot_left"] * 3,
        )
        self.assertGreaterEqual(sum(
            step.state == WallProbeController.RESTORE_VERIFY
            for step in steps
        ), 4)
        actions = [call for call in arduino.calls
                   if call[0] in ("rodas", "parar")]
        for index, call in enumerate(actions[:-1]):
            if call in (
                ("rodas", 0, 50, 0, -50),
                ("rodas", 0, -50, 0, 50),
            ):
                self.assertEqual(actions[index + 1], ("parar",))

    def test_angular_echo_keeps_yaw_recenters_by_omni_and_reprobes(self):
        controller, arduino, steps, final = run_probe([
            # Centro e offsets comuns nao enxergam a parede inclinada.
            100, 102, 101,
            None, None, None,
            None, None, None,
            # Duas baterias independentes confirmam no scan direito.
            135, 137, 136,
            139, 138, 140,
            # Depois do retorno omni, o probe repetido fica simetrico.
            105, 106, 104,
            115, 116, 114,
            119, 118, 120,
        ], recenter_errors=[0.04, 0.03])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertEqual(controller.correction_count, 1)
        self.assertEqual(controller.omni_pulses, 0)
        self.assertTrue(any(
            step.motor_action == "pivot_right" for step in steps))
        # Nao existe pivot_left de restauracao antes do novo probe: o yaw que
        # encontrou a parede foi preservado e o offset voltou por omni left.
        first_scan = next(
            index for index, step in enumerate(steps)
            if step.state == controller.SCAN_PIVOT_RIGHT_PENDING)
        reprobe = next(
            index for index, step in enumerate(steps[first_scan + 1:],
                                               first_scan + 1)
            if step.state == controller.LEFT_PENDING)
        self.assertFalse(any(
            step.motor_action == "pivot_left"
            for step in steps[first_scan + 1:reprobe]
        ))
        self.assertIn(("rodas", -50, 50, 50, -50), arduino.calls)

    def test_left_angular_echo_is_the_mirrored_yaw_preserving_flow(self):
        controller, arduino, steps, final = run_probe([
            100, 102, 101,
            None, None, None,
            None, None, None,
            # Scan direito inteiro vazio.
            *([None] * 9),
            # Scan esquerdo confirma em duas baterias.
            132, 134, 133,
            136, 135, 137,
            # Reprobe simetrico depois do retorno omni direito.
            105, 106, 104,
            115, 116, 114,
            119, 118, 120,
        ], recenter_errors=[-0.04, -0.03])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertEqual(controller.correction_count, 1)
        left_scan = next(
            index for index, step in enumerate(steps)
            if (
                step.state == controller.SCAN_PIVOT_LEFT_PENDING
                and "offset left" in step.detail
            ))
        reprobe = next(
            index for index, step in enumerate(steps[left_scan + 1:],
                                               left_scan + 1)
            if step.state == controller.LEFT_PENDING)
        self.assertFalse(any(
            step.motor_action == "pivot_right"
            for step in steps[left_scan + 1:reprobe]
        ))
        # Do offset esquerdo, o retorno ao centro e omni para a direita.
        self.assertIn(("rodas", 50, -50, -50, 50), arduino.calls)

    def test_angular_candidate_requires_a_second_close_battery(self):
        controller, _arduino, _steps, final = run_probe([
            100, 102, 101,
            None, None, None,
            None, None, None,
            135, 137, 136,
            None, None, None,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao repetiu", final.detail)
        self.assertEqual(controller.correction_count, 0)

    def test_scan_outward_without_visual_progress_fails_closed(self):
        controller, _arduino, _steps, final = run_probe([
            100, 102, 101,
            None, None, None,
            None, None, None,
        ], scan_errors=[-0.56, -0.56])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao afastou", final.detail)
        self.assertEqual(controller._scan_outward_pulses, 2)

    def test_scan_never_measures_if_ball_enters_ultrasonic_beam(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 102, 101,
            None, None, None,
            None, None, None,
        ], scan_errors=[0.0])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("entrou no feixe", final.detail)

    def test_scan_restore_requires_progress_and_has_a_hard_limit(self):
        controller, _arduino, _steps, final = run_probe([
            100, 102, 101,
            None, None, None,
            None, None, None,
            *([None] * 9),
        ], restore_errors=[-0.80, -0.80, -0.80])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao reduziu", final.detail)
        self.assertLessEqual(
            controller._scan_restore_pulses,
            4,
        )

    def test_restore_inside_deadband_still_requires_two_new_frames(self):
        controller = WallProbeController(
            "silver", detection(0.0), start_time=0.0)
        arduino = ArduinoFalso([])
        controller._scan_side = "right"
        controller._scan_baseline_error = -0.50
        controller._scan_current_error = -0.48

        first_step = controller._begin_scan_restore(now=1.0)
        self.assertEqual(first_step.state, controller.RESTORE_VERIFY)
        self.assertFalse(first_step.motor_action)

        first = controller.update(
            arduino,
            detection=detection(1.1, center_x=160),
            frame_shape=(480, 640, 3),
            now=1.1,
        )
        self.assertEqual(first.state, controller.RESTORE_VERIFY)
        self.assertFalse(first.motor_action)

        second = controller.update(
            arduino,
            detection=detection(1.2, center_x=160),
            frame_shape=(480, 640, 3),
            now=1.2,
        )
        self.assertEqual(second.state, controller.SCAN_CROSS_LEFT_PENDING)
        self.assertEqual(second.motor_action, "left")

    def test_offset_serial_timeouts_do_not_release_pickup(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 101, 99,
            TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM,
            TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("timeout", final.detail)

    def test_one_offset_timeout_cannot_be_hidden_by_close_echoes(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 101, 99,
            110, 111, TIMEOUT_ULTRASSOM,
            115, 116, 114,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("timeout", final.detail)

    def test_asymmetric_ranges_trigger_left_rear_pivot(self):
        controller, arduino, steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao reapareceu", final.detail)
        self.assertEqual(controller.correction_count, 1)
        pivot = next(step for step in steps
                     if step.motor_action == "pivot_left")
        self.assertEqual(pivot.pwm, 50)
        self.assertIn(("rodas", 0, -50, 0, 50), arduino.calls)
        pivot_index = arduino.calls.index(("rodas", 0, -50, 0, 50))
        self.assertIn(("parar",), arduino.calls[pivot_index + 1:])

    def test_asymmetric_ranges_trigger_mirrored_right_rear_pivot(self):
        controller, arduino, steps, final = run_probe([
            90, 92, 88,
            175, 172, 178,
            80, 82, 79,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertEqual(controller.correction_count, 1)
        self.assertTrue(any(
            step.motor_action == "pivot_right" for step in steps))
        self.assertIn(("rodas", 0, 50, 0, -50), arduino.calls)

    def test_internal_asymmetry_result_points_to_smaller_or_only_echo(self):
        controller = WallProbeController(
            "silver", detection(0.0), start_time=0.0)
        controller._center_context = "close"

        controller._offset_samples = [
            (80, 81, 79), (170, 171, 169)]
        result, _detail, direction = controller._classify_offsets()
        self.assertEqual((result, direction),
                         (PAREDE_DESALINHADA, "left"))

        controller._offset_samples = [
            (None, None, None), (90, 91, 89)]
        result, _detail, direction = controller._classify_offsets()
        self.assertEqual((result, direction),
                         (PAREDE_DESALINHADA, "right"))

    def test_only_one_side_with_wall_is_inconclusive(self):
        controller, arduino, steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            None, None, None,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertEqual(controller.correction_count, 1)
        self.assertTrue(any(
            step.motor_action == "pivot_left" for step in steps))
        self.assertIn(("rodas", 0, -50, 0, 50), arduino.calls)

    def test_pivot_omni_recenter_and_reprobe_converge_to_wall(self):
        controller, arduino, steps, final = run_probe([
            # Primeira leitura: robo inclinado para a esquerda.
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
            # Depois do pivo/recentralizacao: parede simetrica.
            100, 102, 101,
            110, 112, 111,
            115, 116, 114,
        ], recenter_errors=[0.30, 0.04, 0.03])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertEqual(controller.correction_count, 1)
        self.assertEqual(controller.omni_pulses, 1)
        self.assertEqual(len(controller.center_samples), 2)
        self.assertTrue(any(
            step.motor_action == "pivot_left" for step in steps))
        # Erro visual positivo: pulso omni para o mesmo lado, a direita.
        self.assertIn(("rodas", 45, -45, -45, 45), arduino.calls)
        # Tanto o pivo quanto o pulso omni terminam em STOP antes de observar.
        actions = [call for call in arduino.calls
                   if call[0] in ("rodas", "parar")]
        pivot_index = actions.index(("rodas", 0, -50, 0, 50))
        omni_index = actions.index(("rodas", 45, -45, -45, 45))
        self.assertEqual(actions[pivot_index + 1], ("parar",))
        self.assertEqual(actions[omni_index + 1], ("parar",))

    def test_recenter_omni_direction_is_mirrored(self):
        _controller, arduino, _steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
            100, 102, 101,
            110, 112, 111,
            115, 116, 114,
        ], recenter_errors=[-0.30, -0.04, -0.03])

        self.assertEqual(final.result, PAREDE_RETA)
        self.assertIn(("rodas", -45, 45, 45, -45), arduino.calls)

    def test_three_corrections_are_the_hard_limit(self):
        one_probe = [
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
        ]
        controller, arduino, steps, final = run_probe(
            one_probe * 4,
            recenter_errors=[0.0, 0.0] * 3,
        )

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("limite seguro", final.detail)
        self.assertEqual(controller.correction_count, 3)
        self.assertEqual(sum(
            step.motor_action == "pivot_left" for step in steps), 3)
        self.assertEqual(sum(
            call == ("rodas", 0, -50, 0, 50)
            for call in arduino.calls), 3)

    def test_six_omni_pulses_are_the_hard_limit(self):
        controller, _arduino, _steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
        ], recenter_errors=[0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("limite de pulsos omni", final.detail)
        self.assertEqual(controller.omni_pulses, 6)

    def test_recenter_without_progress_fails_closed(self):
        controller, _arduino, _steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
        ], recenter_errors=[0.30, 0.30, 0.30])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("nao reduziu", final.detail)
        self.assertEqual(controller.omni_pulses, 2)

    def test_ball_must_move_to_expected_side_and_return_to_center(self):
        _controller, arduino, _steps, final = run_probe(
            [100, 102, 101], visual_mode="no_shift")

        self.assertEqual(final.result, INCONCLUSIVO)
        wheels = [call for call in arduino.calls if call[0] == "rodas"]
        self.assertEqual(len(wheels), 2)  # esquerda e retorno para a direita

    def test_return_must_reconfirm_same_centered_ball(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 102, 101,
            110, 112, 111,
            115, 116, 114,
        ], visual_mode="no_return")

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("mesma bolinha no centro", final.detail)

    def test_two_distinct_new_frames_are_required(self):
        target = detection(0.0)
        controller = WallProbeController(
            "silver", target_detection=target, start_time=0.0)
        controller._begin_visual_verification(
            controller.LEFT_VERIFY, now=1.0)
        first = detection(1.1, center_x=500)

        self.assertFalse(controller._accept_visual(
            first, (480, 640, 3), now=1.1, side="right"))
        self.assertFalse(controller._accept_visual(
            first, (480, 640, 3), now=1.2, side="right"))
        second = detection(1.2, center_x=500)
        self.assertTrue(controller._accept_visual(
            second, (480, 640, 3), now=1.2, side="right"))

    def test_fresh_incompatible_target_resets_visual_hits(self):
        controller = WallProbeController(
            "silver", detection(0.0), start_time=0.0)
        controller._begin_visual_verification(
            controller.LEFT_VERIFY, now=1.0)

        self.assertFalse(controller._accept_visual(
            detection(1.1, center_x=500),
            (480, 640, 3), now=1.1, side="right"))
        self.assertFalse(controller._accept_visual(
            detection(1.2, center_x=500, kind="black"),
            (480, 640, 3), now=1.2, side="right"))
        self.assertFalse(controller._accept_visual(
            detection(1.3, center_x=500),
            (480, 640, 3), now=1.3, side="right"))
        self.assertTrue(controller._accept_visual(
            detection(1.4, center_x=500),
            (480, 640, 3), now=1.4, side="right"))

    def test_truncated_target_cannot_complete_visual_battery(self):
        controller = WallProbeController(
            "silver", detection(0.0), start_time=0.0)
        controller._begin_visual_verification(
            controller.LEFT_VERIFY, now=1.0)

        self.assertFalse(controller._accept_visual(
            detection(1.1, center_x=500),
            (480, 640, 3), now=1.1, side="right"))
        self.assertFalse(controller._accept_visual(
            detection(1.2, center_x=500, truncated=True),
            (480, 640, 3), now=1.2, side="right"))
        self.assertFalse(controller._accept_visual(
            detection(1.3, center_x=500),
            (480, 640, 3), now=1.3, side="right"))

    def test_different_geometry_cannot_replace_same_color_target(self):
        target = detection(0.0, radius=55)
        controller = WallProbeController(
            "silver", target_detection=target, start_time=0.0)
        controller._begin_visual_verification(
            controller.LEFT_VERIFY, now=1.0)

        for timestamp in (1.1, 1.2, 1.3):
            accepted = controller._accept_visual(
                detection(timestamp, center_x=500, radius=12),
                (480, 640, 3),
                now=timestamp,
                side="right",
            )
            self.assertFalse(accepted)

    def test_measurement_timeout_cancels_late_ultrasound(self):
        target = detection(0.0)
        controller = WallProbeController(
            "silver", target_detection=target, start_time=0.0)
        arduino = ArduinoFalso([100])
        arduino.pending = True

        finished = controller._collect_ultrasound(arduino, now=1.0)

        self.assertTrue(finished)
        self.assertFalse(arduino.pending)
        self.assertIn(("cancel_ultra",), arduino.calls)

    def test_serial_epoch_change_blocks_lateral_command(self):
        controller = WallProbeController(
            "black",
            target_detection=detection(0.0, kind="black"),
            start_time=0.0,
        )
        arduino = ArduinoFalso([100, 101, 99])
        now = 0.0
        step = None
        for _ in range(10):
            step = controller.update(arduino, now=now)
            if step.motor_action == "left":
                break
            now += 0.07
        arduino.connection_epoch = 8

        error = aplicar_acao_parede(
            step, arduino, epoca_serial_esperada=7)

        self.assertIn("serial mudou", error)
        self.assertFalse(any(call[0] == "rodas" for call in arduino.calls))


if __name__ == "__main__":
    unittest.main()
