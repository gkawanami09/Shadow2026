"""Regressoes do runtime preto/prata e do handoff para o segue-linha."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
import config_resgate as cfg  # noqa: E402
from controle import saida_linha  # noqa: E402
from controle.retomada_saida import ErroRetomadaSaida  # noqa: E402
from visao.confirmacao_saida_linha import (  # noqa: E402
    INCONCLUSIVA,
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
)


def cena_preta_centralizada():
    frame = np.full(
        (config.camera_y, config.camera_x, 3), 205, dtype=np.uint8)
    frame[:95, 200:248] = 35
    frame[95:165, :] = 35
    return frame


def cena_prata_centralizada():
    frame = np.full(
        (config.camera_y, config.camera_x, 3), 205, dtype=np.uint8)
    topo, base = 85, 165
    yy, xx = np.indices((base - topo, config.camera_x))
    textura = 95 + ((xx // 4 + yy // 3) % 2) * 65
    frame[topo:base, :, 0] = textura
    frame[topo:base, :, 1] = textura
    frame[topo:base, :, 2] = textura
    frame[:topo, 200:248] = 35
    return frame


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(float(seconds), 0.001)


class FakeCamera:
    def __init__(self, clock, frame):
        self.clock = clock
        self.frame = frame
        self.closed = False
        self.frames = 0

    def get_frame(self):
        self.clock.sleep(0.05)
        self.frames += 1
        return self.frame.copy()

    def close(self):
        self.closed = True


class FakeArduino:
    def __init__(self):
        self.connected = True
        self.connection_epoch = 7
        self.led_modes = []
        self.paradas = 0

    def refresh(self, fail_closed=True):
        return True

    def led(self, mode):
        self.led_modes.append(str(mode).upper())

    def parar(self):
        self.paradas += 1
        return True


class FakeMonitor:
    def __init__(self):
        self.parada_confirmada = False
        self.distancia_confirmada_mm = None
        self.distancias_validas = ()
        self.cancelamentos = 0

    def atualizar(self, arduino, agora=None):
        return False

    def cancelar(self, arduino):
        self.cancelamentos += 1


class FakeMonitorBloqueio(FakeMonitor):
    def __init__(self, bloquear_na_atualizacao):
        super().__init__()
        self.bloquear_na_atualizacao = int(bloquear_na_atualizacao)
        self.atualizacoes = 0

    def atualizar(self, arduino, agora=None):
        self.atualizacoes += 1
        if self.atualizacoes >= self.bloquear_na_atualizacao:
            self.parada_confirmada = True
            self.distancia_confirmada_mm = 120.0
            self.distancias_validas = (122, 118)
        return self.parada_confirmada


class RecordingSteer:
    def __init__(self, clock):
        self.clock = clock
        self.events = []

    def __call__(self, *args, **kwargs):
        self.events.append((self.clock.now, args, kwargs))
        return True


class ScriptedConfirmador:
    """Fecha uma votacao por instancia, deixando a orquestracao observavel."""

    def __init__(self, decisao):
        self.decisao_script = decisao
        self.votos_pretos = int(decisao == PRETA)
        self.votos_nao_pretos = int(decisao == NAO_PRETA)

    def update(self, frame, timestamp=None, now=None):
        resultado = ClassificadorFaixaSaidaLinha().classificar(
            frame, timestamp=timestamp)
        resultado = replace(
            resultado, classificacao=self.decisao_script)
        return self.decisao_script, resultado


class ConfirmadorFactory:
    def __init__(self, decisoes):
        self.decisoes = list(decisoes)
        self.calls = []
        self.instances = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self.decisoes:
            raise AssertionError("confirmador extra nao esperado")
        instance = ScriptedConfirmador(self.decisoes.pop(0))
        self.instances.append(instance)
        return instance


class RetomadaFactory:
    def __init__(self, falhar=False):
        self.falhar = falhar
        self.calls = []
        self.executou = 0

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        owner = self

        class Retomada:
            def executar(self):
                owner.executou += 1
                if owner.falhar:
                    raise ErroRetomadaSaida("terceira linha nao confirmada")
                return SimpleNamespace(
                    orientacao_soleira="nivel",
                    fase_encontro="avanco_0_3s",
                )

        return Retomada()


class SaidaLinhaRuntimeTests(unittest.TestCase):
    def _executar(self, decisoes, *, falhar_retomada=False):
        clock = FakeClock()
        frame = (
            cena_prata_centralizada()
            if decisoes[0] == NAO_PRETA
            else cena_preta_centralizada()
        )
        camera = FakeCamera(clock, frame)
        arduino = FakeArduino()
        monitor = FakeMonitor()
        steer = RecordingSteer(clock)
        confirmadores = ConfirmadorFactory(decisoes)
        retomada = RetomadaFactory(falhar=falhar_retomada)

        with patch.object(
            saida_linha,
            "ConfirmadorFaixaSaidaLinha",
            new=confirmadores,
        ):
            resultado = saida_linha.confirmar_saida_com_camera_linha(
                arduino,
                steer,
                lambda: camera,
                relogio=clock,
                dormir=clock.sleep,
                monitor_factory=lambda: monitor,
                retomada_factory=retomada,
            )
        return SimpleNamespace(
            resultado=resultado,
            clock=clock,
            camera=camera,
            arduino=arduino,
            monitor=monitor,
            steer=steer,
            confirmadores=confirmadores,
            retomada=retomada,
        )

    def test_preto_so_e_liberado_depois_da_retomada_confirmada(self):
        run = self._executar([PRETA, PRETA])

        self.assertEqual(run.resultado, PRETA)
        self.assertEqual(run.retomada.executou, 1)
        self.assertEqual(len(run.retomada.calls), 1)
        self.assertEqual(len(run.confirmadores.instances), 2)
        self.assertTrue(run.camera.closed)
        self.assertEqual(run.arduino.led_modes, ["ACESO"])
        self.assertEqual(run.steer.events[-1][1], ())

    def test_prata_e_reconfirmada_e_da_re_exata_de_um_segundo(self):
        run = self._executar([NAO_PRETA, NAO_PRETA])

        self.assertEqual(run.resultado, NAO_PRETA)
        self.assertEqual(run.retomada.executou, 0)
        self.assertEqual(len(run.confirmadores.instances), 2)
        self.assertEqual(run.arduino.led_modes, ["ACESO", "APAGADO"])

        eventos = run.steer.events
        indice_re = next(
            i for i, (_t, args, _kwargs) in enumerate(eventos)
            if args == (200, cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_SPEED)
        )
        inicio_re = eventos[indice_re][0]
        fim_re = next(
            t for t, args, _kwargs in eventos[indice_re + 1:]
            if args == ()
        )
        self.assertAlmostEqual(
            fim_re - inicio_re,
            cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_S,
            places=9,
        )
        self.assertEqual(cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_S, 1.0)

    def test_rechecagem_e_independente_e_simetrica_para_as_duas_cores(self):
        for primaria, secundaria in (
            (PRETA, NAO_PRETA),
            (NAO_PRETA, PRETA),
        ):
            with self.subTest(primaria=primaria, secundaria=secundaria):
                run = self._executar([primaria, secundaria])

                self.assertEqual(run.resultado, INCONCLUSIVA)
                self.assertEqual(len(run.confirmadores.instances), 2)
                self.assertEqual(run.retomada.executou, 0)
                primeira_kwargs = run.confirmadores.calls[0][1]
                segunda_kwargs = run.confirmadores.calls[1][1]
                self.assertEqual(primeira_kwargs, {})
                self.assertEqual(
                    segunda_kwargs["tamanho_janela"],
                    cfg.EXIT_LINE_VERIFY_RECHECK_WINDOW,
                )
                self.assertEqual(
                    segunda_kwargs["votos_pretos"],
                    cfg.EXIT_LINE_VERIFY_RECHECK_BLACK_VOTES,
                )
                self.assertEqual(
                    segunda_kwargs["votos_nao_pretos"],
                    cfg.EXIT_LINE_VERIFY_RECHECK_SILVER_VOTES,
                )

    def test_falha_da_retomada_nao_libera_preto_e_termina_parada(self):
        run = self._executar(
            [PRETA, PRETA], falhar_retomada=True)

        self.assertEqual(run.resultado, saida_linha.RETOMADA_FALHOU)
        self.assertEqual(run.retomada.executou, 1)
        self.assertEqual(run.arduino.led_modes, ["ACESO", "APAGADO"])
        self.assertTrue(run.camera.closed)
        self.assertEqual(run.steer.events[-1][1], ())
        self.assertFalse(any(
            args and args[0] in (0, 200)
            for _t, args, _kwargs in run.steer.events
        ))

    def test_bloqueio_desfaz_todo_avanco_acumulado_mais_margem(self):
        clock = FakeClock()
        frame = np.full(
            (config.camera_y, config.camera_x, 3), 205, dtype=np.uint8
        )
        frame[18:68, :] = 35
        frame[:18, 200:248] = 35
        camera = FakeCamera(clock, frame)
        arduino = FakeArduino()
        monitor = FakeMonitorBloqueio(bloquear_na_atualizacao=14)
        steer = RecordingSteer(clock)

        resultado = saida_linha.confirmar_saida_com_camera_linha(
            arduino,
            steer,
            lambda: camera,
            relogio=clock,
            dormir=clock.sleep,
            monitor_factory=lambda: monitor,
        )

        self.assertEqual(resultado, saida_linha.CORREDOR_BLOQUEADO)
        avancos = [
            args for _t, args, _kwargs in steer.events
            if args == (0, cfg.EXIT_LINE_VERIFY_SPEED)
        ]
        self.assertGreaterEqual(len(avancos), 4)
        indice_re = next(
            i for i, (_t, args, _kwargs) in enumerate(steer.events)
            if args == (200, cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_SPEED)
        )
        inicio_re = steer.events[indice_re][0]
        fim_re = next(
            t for t, args, _kwargs in steer.events[indice_re + 1:]
            if args == ()
        )
        esperado = max(
            cfg.EXIT_CLEARANCE_BLOCKED_REVERSE_S,
            len(avancos) * cfg.EXIT_LINE_VERIFY_STEP_S
            + cfg.EXIT_LINE_VERIFY_MISSED_REVERSE_MARGIN_S,
        )
        self.assertAlmostEqual(fim_re - inicio_re, esperado, places=9)


if __name__ == "__main__":
    unittest.main()
