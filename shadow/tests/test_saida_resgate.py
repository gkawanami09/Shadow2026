"""Testes da fase de saída e do mapeamento final dos dois triângulos."""

from dataclasses import dataclass
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
import resgate as resgate_runtime  # noqa: E402
from controle.saida_resgate import ExitPhaseController  # noqa: E402
from resgate import (  # noqa: E402
    CORREDOR_BLOQUEADO,
    CORREDOR_INCONCLUSIVO,
    CORREDOR_LIVRE,
    _recuperar_bloqueio_saida,
    _validar_corredor_saida,
)
from tests.test_confirmacao_saida_linha import cena_prata  # noqa: E402
from visao.triangulos_finais import (  # noqa: E402
    annotate_final_triangles,
    dominant_channel,
    overlay_color_report,
)


FRAME_SHAPE = (480, 640, 3)


@dataclass(frozen=True)
class FakeExit:
    center_x: float = 320.0
    timestamp: float = 0.0


@dataclass(frozen=True)
class FakeMarker:
    bbox: tuple
    confidence: float = 0.80
    hits: int = 3
    track_locked: bool = True
    confirmed: bool = True


class FakeMapper:
    def __init__(self, both=False, frames=0):
        self.confirmed = {"green": both, "red": both}
        self.frames = frames

    @property
    def both_found(self):
        return all(self.confirmed.values())


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(float(seconds), 0.001)


class FakeUltrasonicArduino:
    def __init__(self, readings):
        self.readings = list(readings)
        self.connected = True
        self.connection_epoch = 1
        self.active = False

    def cancelar_ultrassom(self):
        self.active = False

    def iniciar_ultrassom(self, timeout):
        if self.active:
            return False
        self.active = True
        return True

    def poll_ultrassom(self):
        if not self.active:
            return False, None
        self.active = False
        if not self.readings:
            return True, None
        return True, self.readings.pop(0)

    def refresh(self, fail_closed=True):
        return True


class ExitPhaseTests(unittest.TestCase):
    def setUp(self):
        self.exit = ExitPhaseController(start_time=0.0)

    def _ate_observar(self, inicio=0.0):
        """Mapeia, gira um pulso, freia, assenta e chega em OBSERVE."""
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(both=True), now=inicio)
        self.assertEqual(command.state, self.exit.SEARCH_START)
        self.exit.notify_command_written(command.state, now=inicio)

        girando = self.exit.update(None, FRAME_SHAPE, now=inicio + 0.05)
        self.assertEqual(girando.state, self.exit.SEARCH_ROTATE)

        fim = inicio + cfg.EXIT_SEARCH_PULSE_S
        freio = self.exit.update(None, FRAME_SHAPE, now=fim)
        self.assertEqual(freio.state, self.exit.SEARCH_BRAKE)
        self.exit.notify_command_written(freio.state, now=fim)

        assentou = fim + cfg.EXIT_SEARCH_SETTLE_S
        observando = self.exit.update(None, FRAME_SHAPE, now=assentou)
        self.assertEqual(observando.state, self.exit.SEARCH_OBSERVE)
        return assentou

    def test_comeca_mapeando_os_dois_triangulos(self):
        self.assertEqual(self.exit.state, self.exit.MAP_TRIANGLES)
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(frames=0), now=0.0)
        self.assertEqual(command.state, self.exit.MAP_TRIANGLES)
        self.assertEqual(command.angle, 190)

    def test_mapeamento_nao_prende_a_missao(self):
        """Não achar os dois triângulos não pode impedir a saída."""
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(frames=0),
            now=cfg.FINAL_TRIANGLE_MAP_TIMEOUT_S)
        self.assertEqual(command.state, self.exit.SEARCH_START)

    def test_procura_pulsada_para_para_observar(self):
        self._ate_observar()
        self.assertTrue(self.exit.stopped)
        command = self.exit.update(None, FRAME_SHAPE, now=0.45)
        self.assertEqual(command.angle, 190)

    def test_frames_do_giro_nao_confirmam_a_saida(self):
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(both=True), now=0.0)
        self.exit.notify_command_written(command.state, now=0.0)
        self.exit.update(None, FRAME_SHAPE, now=0.05)
        self.assertFalse(self.exit.frame_allowed(0.10))

    def test_candidato_durante_giro_freia_antes_do_fim_do_pulso(self):
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(both=True), now=0.0)
        self.exit.notify_command_written(command.state, now=0.0)

        instante = min(0.05, cfg.EXIT_SEARCH_PULSE_S / 2.0)
        candidato = FakeExit(timestamp=instante)
        freio = self.exit.update(
            candidato, FRAME_SHAPE, now=instante)

        self.assertEqual(freio.state, self.exit.SEARCH_BRAKE)
        self.assertEqual(freio.angle, 190)
        self.assertEqual(freio.speed, 0.0)
        self.assertFalse(self.exit.terminal)

    def test_frame_anterior_ao_assentamento_nao_alinha(self):
        assentou = self._ate_observar()
        antigo = FakeExit(timestamp=assentou - 0.05)
        command = self.exit.update(
            antigo, FRAME_SHAPE, now=assentou + 0.02)
        self.assertEqual(command.state, self.exit.SEARCH_OBSERVE)

    def test_soleira_confirmada_inicia_avanco_reto_imediato(self):
        assentou = self._ate_observar()
        soleira = FakeExit(center_x=320.0, timestamp=assentou + 0.01)
        command = self.exit.update(
            soleira, FRAME_SHAPE, now=assentou + 0.02)
        self.assertEqual(command.state, self.exit.CROSS)
        self.assertEqual(command.angle, 0)
        self.assertEqual(command.speed, cfg.EXIT_ADVANCE_SPEED)
        self.assertEqual(cfg.EXIT_ADVANCE_PWM, 80)

    def test_rejeicao_final_da_re_de_um_segundo(self):
        self.assertEqual(cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_S, 1.0)

    def test_soleira_deslocada_tambem_inicia_reto_sem_alinhamento_fraco(self):
        assentou = self._ate_observar()
        soleira = FakeExit(center_x=600.0, timestamp=assentou + 0.01)
        command = self.exit.update(
            soleira, FRAME_SHAPE, now=assentou + 0.02)
        self.assertEqual(command.state, self.exit.CROSS)
        self.assertEqual(command.angle, 0)

    def test_perda_curta_apos_confirmar_nao_interrompe_o_avanco(self):
        assentou = self._ate_observar()
        soleira = FakeExit(timestamp=assentou + 0.01)
        command = self.exit.update(
            soleira, FRAME_SHAPE, now=assentou + 0.02)
        self.exit.notify_command_written(
            command.state, now=assentou + 0.03)
        command = self.exit.update(
            None,
            FRAME_SHAPE,
            now=assentou + 0.03 + cfg.EXIT_ADVANCE_MIN_S / 2,
        )
        self.assertEqual(command.state, self.exit.CROSS)
        self.assertEqual(command.angle, 0)

    def test_travessia_termina_quando_a_faixa_passa_para_tras(self):
        assentou = self._ate_observar()
        soleira = FakeExit(timestamp=assentou + 0.01)
        command = self.exit.update(
            soleira, FRAME_SHAPE, now=assentou + 0.02)
        self.assertEqual(command.state, self.exit.CROSS)
        self.assertEqual(command.angle, 0)
        self.exit.notify_command_written(command.state, now=assentou + 0.03)

        cedo = assentou + 0.03 + cfg.EXIT_ADVANCE_MIN_S / 2
        andando = self.exit.update(None, FRAME_SHAPE, now=cedo)
        self.assertEqual(andando.state, self.exit.CROSS)

        tarde = assentou + 0.03 + cfg.EXIT_ADVANCE_MIN_S + 0.01
        final = self.exit.update(None, FRAME_SHAPE, now=tarde)
        self.assertEqual(final.state, self.exit.DONE)
        self.assertTrue(final.terminal)
        self.assertEqual(final.angle, 190)
        self.assertTrue(self.exit.succeeded)
        self.assertAlmostEqual(
            self.exit.cross_elapsed_s,
            tarde - (assentou + 0.03),
        )

    def test_travessia_tem_timeout_de_seguranca(self):
        assentou = self._ate_observar()
        soleira = FakeExit(timestamp=assentou + 0.01)
        command = self.exit.update(
            soleira, FRAME_SHAPE, now=assentou + 0.02)
        inicio = assentou + 0.03
        self.exit.notify_command_written(command.state, now=inicio)
        # A faixa continua sendo vista: só o timeout encerra.
        fim = inicio + cfg.EXIT_ADVANCE_TIMEOUT_S
        presente = FakeExit(timestamp=fim)
        final = self.exit.update(
            presente, FRAME_SHAPE, now=fim)
        self.assertEqual(final.state, self.exit.DONE)
        self.assertEqual(final.angle, 190)

    def test_procura_sem_sucesso_termina_parada(self):
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(both=True),
            now=cfg.EXIT_SEARCH_TIMEOUT_S + 1.0)
        self.assertEqual(command.state, self.exit.FAILED)
        self.assertTrue(command.terminal)
        self.assertEqual(command.angle, 190)
        self.assertFalse(self.exit.succeeded)


class ExitClearanceTests(unittest.TestCase):
    def _validate(self, readings):
        clock = FakeClock()
        arduino = FakeUltrasonicArduino(readings)
        return _validar_corredor_saida(
            arduino,
            relogio=clock,
            dormir=clock.sleep,
        )

    def test_cinco_medidas_acima_de_15_cm_liberam_camera_de_linha(self):
        state, distance, readings = self._validate(
            [210, 190, 180, 205, 195])
        self.assertEqual(state, CORREDOR_LIVRE)
        self.assertEqual(distance, 180)
        self.assertEqual(readings, (210, 190, 180, 205, 195))

    def test_duas_medidas_em_15_cm_bloqueiam_a_tentativa(self):
        state, distance, readings = self._validate([143, 260, 148])
        self.assertEqual(state, CORREDOR_BLOQUEADO)
        self.assertEqual(distance, 146)
        self.assertEqual(readings, (143, 260, 148))

    def test_uma_medida_proxima_isolada_nao_libera_a_tentativa(self):
        state, distance, readings = self._validate([145, 230, 240])
        self.assertEqual(state, CORREDOR_INCONCLUSIVO)
        self.assertEqual(distance, 145)
        self.assertEqual(readings, (145, 230, 240))

    def test_sensor_sem_eco_nao_e_tratado_como_corredor_livre(self):
        state, distance, readings = self._validate([None] * 20)
        self.assertEqual(state, CORREDOR_INCONCLUSIVO)
        self.assertIsNone(distance)
        self.assertEqual(readings, ())

    def test_bloqueio_recua_300ms_e_gira_500ms_antes_de_recomecar(self):
        movimentos = []
        paradas = []

        def direcao(*args):
            if args:
                movimentos.append(args)
            else:
                paradas.append(True)
            return True

        with patch.object(
            resgate_runtime,
            "_mover_saida_por_tempo",
            side_effect=lambda _a, _d, angulo, velocidade, duracao, epoca: (
                movimentos.append((angulo, velocidade, duracao, epoca))
            ),
        ):
            _recuperar_bloqueio_saida(
                FakeUltrasonicArduino([]),
                direcao,
                7,
            )

        self.assertEqual(
            movimentos,
            [
                (
                    200,
                    cfg.EXIT_CLEARANCE_REVERSE_SPEED,
                    0.30,
                    7,
                ),
                (
                    cfg.DEPOSIT_SEARCH_TANK_ANGLE,
                    cfg.RED_DEPOSIT_SEARCH_TANK_SPEED,
                    cfg.EXIT_CLEARANCE_ESCAPE_TURN_S,
                    7,
                ),
            ],
        )
        self.assertEqual(cfg.EXIT_CLEARANCE_ESCAPE_TURN_S, 0.80)
        self.assertEqual(cfg.EXIT_SEARCH_PULSE_S, 0.80)
        self.assertEqual(len(paradas), 3)


class SilverStripeRuntimeTests(unittest.TestCase):
    class Arduino:
        def __init__(self, readings=None):
            self.connected = True
            self.connection_epoch = 1
            self.readings = list(
                [300] * 20 if readings is None else readings)
            self.ultrasonic_active = False

        def refresh(self, fail_closed=True):
            return True

        def cancelar_ultrassom(self):
            self.ultrasonic_active = False

        def iniciar_ultrassom(self, timeout):
            if self.ultrasonic_active:
                return False
            self.ultrasonic_active = True
            return True

        def poll_ultrassom(self):
            if not self.ultrasonic_active:
                return False, None
            self.ultrasonic_active = False
            if not self.readings:
                return True, None
            return True, self.readings.pop(0)

    class Camera:
        def __init__(self, clock):
            self.clock = clock
            self.closed = False

        def get_frame(self):
            self.clock.sleep(0.10)
            frame = cena_prata()
            # Primeiro a faixa aparece alta. No quadro seguinte, o avanco a
            # trouxe ao centro e so entao a votacao pode comecar.
            if self.clock.now <= 0.10:
                return frame
            deslocado = np.full_like(frame, 205)
            deslocado[50:] = frame[:-50]
            return deslocado

        def close(self):
            self.closed = True

    def test_cinza_para_estabiliza_e_da_re_sem_entrar_no_segue_linha(self):
        clock = FakeClock()
        camera = self.Camera(clock)
        commands = []

        def fake_steer(angle=190.0, speed=0.8, **_kwargs):
            commands.append((angle, speed))
            return True

        with (
            patch.object(
                resgate_runtime.time, "monotonic", side_effect=clock),
            patch.object(
                resgate_runtime.time, "sleep", side_effect=clock.sleep),
            patch("controle.direcao.steer", side_effect=fake_steer),
            patch("visao.captura.LineCamera", return_value=camera),
        ):
            resultado = resgate_runtime._confirmar_saida_com_camera_linha(
                self.Arduino(),
                debug=False,
            )

        self.assertEqual(resultado, resgate_runtime.NAO_PRETA)
        self.assertTrue(camera.closed)
        self.assertIn(
            (200, cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_SPEED),
            commands,
        )
        self.assertNotIn(
            (0, cfg.EXIT_LINE_VERIFY_BLACK_FORWARD_SPEED),
            commands,
        )

    def test_parede_durante_camera_de_linha_desfaz_todo_avanco(self):
        clock = FakeClock()
        camera = self.Camera(clock)
        arduino = self.Arduino([120, 110, 105])
        commands = []
        reverse_durations = []

        def fake_steer(angle=190.0, speed=0.8, **_kwargs):
            commands.append((angle, speed))
            return True

        def fake_move(
            _arduino,
            _acao,
            angle,
            _speed,
            duration,
            _epoch,
        ):
            self.assertEqual(angle, 200)
            reverse_durations.append(duration)

        with (
            patch.object(
                resgate_runtime.time, "monotonic", side_effect=clock),
            patch.object(
                resgate_runtime.time, "sleep", side_effect=clock.sleep),
            patch("controle.direcao.steer", side_effect=fake_steer),
            patch("visao.captura.LineCamera", return_value=camera),
            patch.object(
                resgate_runtime,
                "_mover_saida_por_tempo",
                side_effect=fake_move,
            ),
        ):
            resultado = resgate_runtime._confirmar_saida_com_camera_linha(
                arduino,
                debug=False,
                recuo_base_s=0.60,
            )

        self.assertEqual(resultado, resgate_runtime.CORREDOR_BLOQUEADO)
        self.assertEqual(len(reverse_durations), 1)
        self.assertGreater(reverse_durations[0], 0.60)
        self.assertNotIn(
            (0, cfg.EXIT_LINE_VERIFY_BLACK_FORWARD_SPEED),
            commands,
        )


class OverlayColorTests(unittest.TestCase):
    """As cores do overlay não podem estar invertidas.

    Um verde desenhado em vermelho faz a equipe recalibrar a cor errada em
    campo; por isso este teste lê os pixels realmente desenhados.
    """

    def test_constantes_bgr_do_overlay(self):
        report = overlay_color_report()
        self.assertEqual(report["green"], (0, 255, 0))
        self.assertEqual(report["red"], (0, 0, 255))

    def test_canal_dominante_por_cor(self):
        report = overlay_color_report()
        self.assertEqual(dominant_channel(report["green"]), 1)  # G
        self.assertEqual(dominant_channel(report["red"]), 2)    # R

    def test_pixels_desenhados_nao_estao_invertidos(self):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        detections = {
            "green": FakeMarker(bbox=(40, 300, 120, 90)),
            "red": FakeMarker(bbox=(420, 300, 120, 90)),
        }
        canvas = annotate_final_triangles(frame, detections)

        verde = canvas[300, 40:160]
        vermelho = canvas[300, 420:540]
        verde_pixels = verde[verde.any(axis=1)]
        vermelho_pixels = vermelho[vermelho.any(axis=1)]
        self.assertTrue(len(verde_pixels) > 0)
        self.assertTrue(len(vermelho_pixels) > 0)

        # O retângulo do triângulo verde é desenhado em verde puro…
        self.assertTrue(
            np.all(verde_pixels[:, 1] == 255),
            "o retângulo do triângulo verde não saiu verde")
        self.assertTrue(np.all(verde_pixels[:, 2] == 0))
        # …e o do vermelho, em vermelho puro.
        self.assertTrue(
            np.all(vermelho_pixels[:, 2] == 255),
            "o retângulo do triângulo vermelho não saiu vermelho")
        self.assertTrue(np.all(vermelho_pixels[:, 1] == 0))

    def test_deteccao_ausente_nao_desenha_nada(self):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        canvas = annotate_final_triangles(frame, {"green": None, "red": None})
        self.assertEqual(int(canvas.sum()), 0)


if __name__ == "__main__":
    unittest.main()
