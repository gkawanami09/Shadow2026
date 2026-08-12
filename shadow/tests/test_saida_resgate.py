"""Regressoes da busca, centralizacao e travessia da saida do resgate."""

from dataclasses import dataclass
from pathlib import Path
import sys
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
    angle_deg: float = 0.0
    confidence: float = 0.0


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
        """Completa um pulso e chega parado a SEARCH_OBSERVE."""
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

    def _ate_cross(self):
        assentou = self._ate_observar()
        primeiro_t = assentou + 0.01
        primeiro = self.exit.update(
            FakeExit(timestamp=primeiro_t),
            FRAME_SHAPE,
            now=primeiro_t + 0.001,
        )
        self.assertEqual(primeiro.state, self.exit.ALIGN)
        self.assertEqual(primeiro.angle, 190)

        segundo_t = primeiro_t + 0.02
        cross = self.exit.update(
            FakeExit(timestamp=segundo_t),
            FRAME_SHAPE,
            now=segundo_t + 0.001,
        )
        self.assertEqual(cross.state, self.exit.CROSS)
        return segundo_t + 0.001, cross

    def test_mapeia_antes_de_procurar_e_timeout_do_mapa_nao_prende(self):
        mapeando = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(), now=0.0)
        self.assertEqual(mapeando.state, self.exit.MAP_TRIANGLES)

        procurando = self.exit.update(
            None,
            FRAME_SHAPE,
            mapper=FakeMapper(),
            now=cfg.FINAL_TRIANGLE_MAP_TIMEOUT_S,
        )
        self.assertEqual(procurando.state, self.exit.SEARCH_START)

    def test_frames_capturados_durante_o_giro_nao_confirmam(self):
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(both=True), now=0.0)
        self.exit.notify_command_written(command.state, now=0.0)

        candidato = FakeExit(timestamp=0.05)
        girando = self.exit.update(candidato, FRAME_SHAPE, now=0.05)

        self.assertEqual(girando.state, self.exit.SEARCH_ROTATE)
        self.assertFalse(self.exit.frame_allowed(candidato.timestamp))

    def test_previa_forte_em_movimento_so_freia_e_nao_autoriza_cross(self):
        command = self.exit.update(
            None, FRAME_SHAPE, mapper=FakeMapper(both=True), now=0.0)
        self.exit.notify_command_written(command.state, now=0.0)
        preview = FakeExit(
            center_x=320.0,
            timestamp=0.05,
            confidence=cfg.EXIT_MODEL_FAST_LOCK_CONFIDENCE,
        )

        freio = self.exit.update(preview, FRAME_SHAPE, now=0.05)
        self.assertEqual(freio.state, self.exit.SEARCH_BRAKE)
        self.exit.notify_command_written(freio.state, now=0.05)

        assentou = 0.05 + cfg.EXIT_SEARCH_SETTLE_S
        observando = self.exit.update(None, FRAME_SHAPE, now=assentou)
        self.assertEqual(observando.state, self.exit.SEARCH_OBSERVE)
        ainda_observando = self.exit.update(
            None, FRAME_SHAPE, now=assentou + 0.01)
        self.assertEqual(ainda_observando.state, self.exit.SEARCH_OBSERVE)
        self.assertFalse(self.exit.terminal)

    def test_cross_exige_dois_timestamps_distintos_e_centralizados(self):
        assentou = self._ate_observar()
        timestamp = assentou + 0.01
        deteccao = FakeExit(timestamp=timestamp)

        primeiro = self.exit.update(
            deteccao, FRAME_SHAPE, now=timestamp + 0.001)
        repetido = self.exit.update(
            deteccao, FRAME_SHAPE, now=timestamp + 0.002)

        self.assertEqual(primeiro.state, self.exit.ALIGN)
        self.assertEqual(repetido.state, self.exit.ALIGN)
        self.assertEqual(repetido.angle, 190)

        novo = self.exit.update(
            FakeExit(timestamp=timestamp + 0.02),
            FRAME_SHAPE,
            now=timestamp + 0.021,
        )
        self.assertEqual(novo.state, self.exit.CROSS)

    def test_centro_encontrado_durante_curva_freia_assenta_e_reconfirma(self):
        assentou = self._ate_observar()
        inicio = assentou + 0.01
        curva = self.exit.update(
            FakeExit(center_x=600.0, timestamp=inicio),
            FRAME_SHAPE,
            now=inicio + 0.001,
        )
        self.assertEqual(curva.state, self.exit.ALIGN_ARC)

        freio = self.exit.update(
            FakeExit(center_x=320.0, timestamp=inicio + 0.02),
            FRAME_SHAPE,
            now=inicio + 0.021,
        )
        self.assertEqual(freio.state, self.exit.ALIGN_BRAKE)
        self.assertEqual(freio.angle, 190)
        self.exit.notify_command_written(freio.state, now=inicio + 0.021)

        fim_settle = inicio + 0.021 + cfg.EXIT_ALIGN_SETTLE_S
        parado = self.exit.update(None, FRAME_SHAPE, now=fim_settle)
        self.assertEqual(parado.state, self.exit.ALIGN)

        frame_1_t = fim_settle + 0.01
        frame_1 = self.exit.update(
            FakeExit(timestamp=frame_1_t),
            FRAME_SHAPE,
            now=frame_1_t + 0.001,
        )
        self.assertEqual(frame_1.state, self.exit.ALIGN)
        frame_2_t = frame_1_t + 0.02
        frame_2 = self.exit.update(
            FakeExit(timestamp=frame_2_t),
            FRAME_SHAPE,
            now=frame_2_t + 0.001,
        )
        self.assertEqual(frame_2.state, self.exit.CROSS)

    def test_perda_persistente_no_alinhamento_reabre_busca(self):
        assentou = self._ate_observar()
        visto_em = assentou + 0.01
        curva = self.exit.update(
            FakeExit(center_x=600.0, timestamp=visto_em),
            FRAME_SHAPE,
            now=visto_em,
        )
        self.assertEqual(curva.state, self.exit.ALIGN_ARC)

        comando = self.exit.update(
            None,
            FRAME_SHAPE,
            now=visto_em + cfg.EXIT_ALIGN_LOST_TIMEOUT_S + 0.01,
        )
        self.assertEqual(comando.state, self.exit.SEARCH_BRAKE)
        self.assertEqual(comando.angle, 190)

    def test_cross_so_termina_apos_perda_persistente(self):
        alinhado_em, cross = self._ate_cross()
        inicio = alinhado_em + 0.01
        self.exit.notify_command_written(cross.state, now=inicio)

        perda_curta = inicio + 0.02
        ainda = self.exit.update(None, FRAME_SHAPE, now=perda_curta)
        self.assertEqual(ainda.state, self.exit.CROSS)
        ainda_curta = self.exit.update(
            None,
            FRAME_SHAPE,
            now=perda_curta + cfg.EXIT_ADVANCE_LOST_CONFIRM_S / 2,
        )
        self.assertEqual(ainda_curta.state, self.exit.CROSS)

        # Reaparecer zera qualquer perda curta acumulada.
        reapareceu = (
            perda_curta + cfg.EXIT_ADVANCE_LOST_CONFIRM_S / 2 + 0.01)
        presente = self.exit.update(
            FakeExit(timestamp=reapareceu),
            FRAME_SHAPE,
            now=reapareceu,
        )
        self.assertEqual(presente.state, self.exit.CROSS)

        perda = reapareceu + 0.01
        curta = self.exit.update(None, FRAME_SHAPE, now=perda)
        self.assertEqual(curta.state, self.exit.CROSS)
        ainda_curta = self.exit.update(
            None,
            FRAME_SHAPE,
            now=perda + cfg.EXIT_ADVANCE_LOST_CONFIRM_S / 2,
        )
        self.assertEqual(ainda_curta.state, self.exit.CROSS)

        confirmada = self.exit.update(
            None,
            FRAME_SHAPE,
            now=perda + cfg.EXIT_ADVANCE_LOST_CONFIRM_S,
        )
        self.assertEqual(confirmada.state, self.exit.DONE)
        self.assertTrue(confirmada.terminal)
        self.assertTrue(self.exit.succeeded)

    def test_timeout_da_travessia_falha_em_vez_de_liberar_segue_linha(self):
        alinhado_em, cross = self._ate_cross()
        inicio = alinhado_em + 0.01
        self.exit.notify_command_written(cross.state, now=inicio)
        fim = inicio + cfg.EXIT_ADVANCE_TIMEOUT_S

        final = self.exit.update(
            FakeExit(timestamp=fim), FRAME_SHAPE, now=fim)

        self.assertEqual(final.state, self.exit.FAILED)
        self.assertTrue(final.terminal)
        self.assertFalse(self.exit.succeeded)
        self.assertEqual(final.angle, 190)

    def test_resultado_envelhecido_nao_e_confundido_com_perda_real(self):
        alinhado_em, cross = self._ate_cross()
        inicio = alinhado_em + 0.01
        self.exit.notify_command_written(cross.state, now=inicio)
        antiga = FakeExit(timestamp=inicio)

        depois_do_minimo = (
            inicio + cfg.EXIT_ADVANCE_LOST_CONFIRM_S + 0.05)
        ainda = self.exit.update(
            antiga, FRAME_SHAPE, now=depois_do_minimo)

        self.assertEqual(ainda.state, self.exit.CROSS)
        self.assertFalse(ainda.terminal)

    def test_procura_sem_saida_termina_parada(self):
        command = self.exit.update(
            None,
            FRAME_SHAPE,
            mapper=FakeMapper(both=True),
            now=cfg.EXIT_SEARCH_TIMEOUT_S + 1.0,
        )
        self.assertEqual(command.state, self.exit.FAILED)
        self.assertTrue(command.terminal)
        self.assertEqual(command.angle, 190)


class ExitClearanceTests(unittest.TestCase):
    def _validate(self, readings):
        clock = FakeClock()
        arduino = FakeUltrasonicArduino(readings)
        return _validar_corredor_saida(
            arduino,
            relogio=clock,
            dormir=clock.sleep,
        )

    def test_cinco_medidas_livres_liberam_camera_de_linha(self):
        state, distance, readings = self._validate(
            [210, 190, 180, 205, 195])
        self.assertEqual(state, CORREDOR_LIVRE)
        self.assertEqual(distance, 180)
        self.assertEqual(readings, (210, 190, 180, 205, 195))

    def test_duas_medidas_proximas_bloqueiam(self):
        state, distance, readings = self._validate([143, 260, 148])
        self.assertEqual(state, CORREDOR_BLOQUEADO)
        self.assertEqual(distance, 146)
        self.assertEqual(readings, (143, 260, 148))

    def test_sensor_sem_eco_fica_inconclusivo(self):
        state, distance, readings = self._validate([None] * 20)
        self.assertEqual(state, CORREDOR_INCONCLUSIVO)
        self.assertIsNone(distance)
        self.assertEqual(readings, ())

    def test_bloqueio_recua_e_gira_um_pulso(self):
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
                FakeUltrasonicArduino([]), direcao, 7)

        self.assertEqual(
            movimentos,
            [
                (
                    200,
                    cfg.EXIT_CLEARANCE_REVERSE_SPEED,
                    cfg.EXIT_CLEARANCE_BLOCKED_REVERSE_S,
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
        self.assertEqual(len(paradas), 3)


class OverlayColorTests(unittest.TestCase):
    def test_constantes_bgr_do_overlay(self):
        report = overlay_color_report()
        self.assertEqual(report["green"], (0, 255, 0))
        self.assertEqual(report["red"], (0, 0, 255))

    def test_canal_dominante_por_cor(self):
        report = overlay_color_report()
        self.assertEqual(dominant_channel(report["green"]), 1)
        self.assertEqual(dominant_channel(report["red"]), 2)

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
        self.assertGreater(len(verde_pixels), 0)
        self.assertGreater(len(vermelho_pixels), 0)
        self.assertTrue(np.all(verde_pixels[:, 1] == 255))
        self.assertTrue(np.all(verde_pixels[:, 2] == 0))
        self.assertTrue(np.all(vermelho_pixels[:, 2] == 255))
        self.assertTrue(np.all(vermelho_pixels[:, 1] == 0))

    def test_deteccao_ausente_nao_desenha(self):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        canvas = annotate_final_triangles(
            frame, {"green": None, "red": None})
        self.assertEqual(int(canvas.sum()), 0)


if __name__ == "__main__":
    unittest.main()
