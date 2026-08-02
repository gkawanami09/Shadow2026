"""Testes do verificador nao bloqueante de parede junto da vitima."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.parede_vitima import (  # noqa: E402
    INCONCLUSIVO,
    LIVRE,
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


def detection(now, center_x=320, kind="silver", radius=55):
    return VictimDetection(
        kind,
        center_x=center_x,
        center_y=300,
        radius=radius,
        confidence=0.95,
        confirmed=True,
        hits=5,
        timestamp=now,
        track_locked=True,
    )


def run_probe(distances, visual_mode="normal"):
    target = detection(0.0)
    controller = WallProbeController(
        "silver", target_detection=target, start_time=0.0)
    arduino = ArduinoFalso(distances)
    now = 0.0
    steps = []
    for _ in range(400):
        seen = None
        if controller.state == controller.LEFT_VERIFY:
            center_x = 320 if visual_mode == "no_shift" else 500
            seen = detection(now, center_x=center_x)
        elif controller.state == controller.RIGHT_VERIFY:
            center_x = 320 if visual_mode == "no_shift" else 140
            seen = detection(now, center_x=center_x)
        elif controller.state == controller.CENTER_VERIFY:
            center_x = 500 if visual_mode == "no_return" else 320
            seen = detection(now, center_x=center_x)
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

    def test_center_far_releases_normal_pickup_without_strafe(self):
        _controller, arduino, _steps, final = run_probe([400, 410, 390])

        self.assertEqual(final.result, LIVRE)
        self.assertFalse(any(call[0] == "rodas" for call in arduino.calls))

    def test_center_without_physical_echo_releases_normal_pickup(self):
        _controller, arduino, _steps, final = run_probe(
            [None, None, None])

        self.assertEqual(final.result, LIVRE)
        self.assertIn("sem eco", final.detail)
        self.assertFalse(any(call[0] == "rodas" for call in arduino.calls))

    def test_center_serial_timeout_is_inconclusive(self):
        _controller, arduino, _steps, final = run_probe(
            [TIMEOUT_ULTRASSOM] * 3)

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("ausentes", final.detail)
        self.assertFalse(any(call[0] == "rodas" for call in arduino.calls))

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
        wheels = [call for call in arduino.calls if call[0] == "rodas"]
        self.assertEqual(len(wheels), 3)
        self.assertEqual(wheels[0], wheels[2])
        self.assertEqual(
            wheels[1][1:],
            tuple(-value for value in wheels[0][1:]),
        )

    def test_echo_disappears_on_both_sides(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 101, 99,
            None, None, None,
            None, None, None,
        ])

        self.assertEqual(final.result, LIVRE)
        self.assertIn("dois lados", final.detail)

    def test_offset_serial_timeouts_do_not_release_pickup(self):
        _controller, _arduino, _steps, final = run_probe([
            100, 101, 99,
            TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM,
            TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM, TIMEOUT_ULTRASSOM,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("um lado difere", final.detail)

    def test_different_side_ranges_are_treated_as_possible_corner(self):
        _controller, _arduino, _steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            170, 172, 168,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("possivel quina", final.detail)

    def test_only_one_side_with_wall_is_inconclusive(self):
        _controller, _arduino, _steps, final = run_probe([
            90, 92, 88,
            80, 82, 79,
            None, None, None,
        ])

        self.assertEqual(final.result, INCONCLUSIVO)
        self.assertIn("um lado difere", final.detail)

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
