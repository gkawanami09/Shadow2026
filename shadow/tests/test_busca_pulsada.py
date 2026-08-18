"""Testes da busca pulsada em tanque (gira → freia → assenta → observa)."""

from dataclasses import dataclass
import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.busca_pulsada import (  # noqa: E402
    PulsedBallSearchController,
    make_search_controller,
)
from controle.busca_resgate import BallSearchController  # noqa: E402


@dataclass(frozen=True)
class FakeDetection:
    kind: str = "silver"
    confidence: float = 0.90
    confirmed: bool = True
    track_locked: bool = True
    timestamp: float = 0.0


class PulsedCycleTests(unittest.TestCase):
    def setUp(self):
        self.search = PulsedBallSearchController(start_time=0.0)

    def _pulso_completo(self, inicio):
        """Executa um pulso: gira, freia, assenta e chega em OBSERVE."""
        command = self.search.update(None, now=inicio)
        self.assertEqual(command.state, self.search.START)
        self.search.notify_command_written(command.state, now=inicio)

        girando = self.search.update(None, now=inicio + 0.05)
        self.assertEqual(girando.state, self.search.ROTATING)
        self.assertEqual(girando.speed, cfg.BALL_SEARCH_TANK_SPEED)

        fim_pulso = inicio + cfg.BALL_SEARCH_PULSE_S
        freio = self.search.update(None, now=fim_pulso)
        self.assertEqual(freio.state, self.search.BRAKE)
        self.assertEqual(freio.angle, 190)
        self.search.notify_command_written(freio.state, now=fim_pulso)

        assentou = fim_pulso + cfg.BALL_SEARCH_SETTLE_S
        observando = self.search.update(None, now=assentou)
        self.assertEqual(observando.state, self.search.OBSERVE)
        return assentou

    def test_ciclo_gira_para_e_observa(self):
        assentou = self._pulso_completo(0.0)
        self.assertTrue(self.search.stopped)
        self.assertEqual(self.search.pulses, 1)
        self.assertIsNotNone(assentou)

    def test_primeiro_comando_ja_inicia_o_giro_de_busca(self):
        inicio = 0.0
        giro = self.search.update(None, now=inicio)

        self.assertEqual(giro.state, self.search.START)
        self.assertEqual(giro.angle, cfg.BALL_SEARCH_TANK_ANGLE)
        self.assertEqual(giro.speed, cfg.BALL_SEARCH_TANK_SPEED)

    def test_o_giro_realmente_para_para_observar(self):
        assentou = self._pulso_completo(0.0)
        command = self.search.update(None, now=assentou + 0.01)
        self.assertEqual(command.angle, 190)
        self.assertEqual(command.speed, 0.0)

    def test_frames_do_giro_nao_sao_aceitos(self):
        inicio = 0.0
        command = self.search.update(None, now=inicio)
        self.search.notify_command_written(command.state, now=inicio)
        self.search.update(None, now=inicio + 0.05)
        self.assertEqual(self.search.state, self.search.ROTATING)
        self.assertFalse(self.search.frame_allowed(inicio + 0.10))

    def test_frame_anterior_a_parada_nao_confirma(self):
        assentou = self._pulso_completo(0.0)
        # Frame capturado durante o pulso, entregue depois do assentamento.
        antigo = FakeDetection(timestamp=assentou - 0.10)
        self.assertFalse(self.search.frame_allowed(antigo.timestamp))
        command = self.search.update(antigo, now=assentou + 0.02)
        self.assertNotEqual(command.state, self.search.TARGET_STOP)
        self.assertNotEqual(command.state, self.search.ACQUIRED)

    def test_frame_posterior_ao_assentamento_trava_o_alvo(self):
        assentou = self._pulso_completo(0.0)
        novo = FakeDetection(timestamp=assentou + 0.01)
        command = self.search.update(novo, now=assentou + 0.02)
        self.assertEqual(command.state, self.search.TARGET_STOP)
        self.assertEqual(command.angle, 190)
        self.assertEqual(command.target_kind, "silver")

    def test_candidato_durante_o_giro_freia_mas_nao_confirma(self):
        inicio = 0.0
        command = self.search.update(None, now=inicio)
        self.search.notify_command_written(command.state, now=inicio)
        candidato = FakeDetection(
            confirmed=False, track_locked=False, timestamp=inicio + 0.05)
        freio = self.search.update(candidato, now=inicio + 0.06)
        self.assertEqual(freio.state, self.search.TARGET_STOP)
        self.assertEqual(freio.angle, 190)
        # O alvo ainda é tentativo: precisa ser reconfirmado já parado.
        self.assertEqual(self.search.state, self.search.TARGET_STOP)

        parou_em = inicio + 0.07
        self.search.notify_command_written(freio.state, now=parou_em)
        self.assertEqual(self.search.state, self.search.VERIFY)
        # O mesmo frame borrado, reentregue, não pode fechar a confirmação.
        ainda = self.search.update(candidato, now=parou_em + 0.01)
        self.assertEqual(ainda.state, self.search.VERIFY)

    def test_falso_candidato_e_descartado_e_a_busca_retoma(self):
        inicio = 0.0
        command = self.search.update(None, now=inicio)
        self.search.notify_command_written(command.state, now=inicio)
        candidato = FakeDetection(
            confirmed=False, track_locked=False, timestamp=inicio + 0.05)
        freio = self.search.update(candidato, now=inicio + 0.06)
        self.search.notify_command_written(freio.state, now=inicio + 0.07)

        expirou = inicio + 0.07 + cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
        retomada = self.search.update(None, now=expirou)
        self.assertEqual(retomada.state, self.search.START)
        self.assertTrue(self.search.consume_tracking_reset())

    def test_pausa_nao_conta_como_angulo_percorrido(self):
        """Tempo ativo e tempo total divergem: só o ativo estima o ângulo."""
        self._pulso_completo(0.0)
        ativo = self.search._rotation_elapsed_s
        self.assertAlmostEqual(ativo, cfg.BALL_SEARCH_PULSE_S, places=6)
        # Ficar parado observando por muito tempo não avança a cobertura.
        self.search.update(None, now=5.0)
        self.assertAlmostEqual(
            self.search._rotation_elapsed_s, ativo, places=6)

    def test_retomada_preserva_a_cobertura_restante(self):
        agora = 0.0
        for _ in range(3):
            agora = self._pulso_completo(agora) + 0.01
            # Sem candidato e sem frames novos, o timeout do OBSERVE avança.
            agora += cfg.BALL_SEARCH_OBSERVE_TIMEOUT_S
            self.search.update(None, now=agora)
        esperado = 3 * cfg.BALL_SEARCH_PULSE_S
        self.assertAlmostEqual(
            self.search._rotation_elapsed_s, esperado, places=6)
        self.assertEqual(self.search.pulses, 3)

    def test_cobertura_completa_encerra_sem_laco_infinito(self):
        agora = 0.0
        # Força o acumulado de giro a ultrapassar o 360 calibrado.
        self.search._rotation_elapsed_s = cfg.BALL_SEARCH_FULL_TURN_S
        assentou = self._pulso_completo(agora)
        agora = assentou + cfg.BALL_SEARCH_OBSERVE_TIMEOUT_S
        final = self.search.update(None, now=agora)
        self.assertEqual(final.state, self.search.FINAL_VERIFY)
        expirou = agora + cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
        concluido = self.search.update(None, now=expirou)
        self.assertEqual(concluido.state, self.search.COMPLETE)
        self.assertTrue(concluido.terminal)

    def test_timeout_global_encerra_a_busca(self):
        command = self.search.update(
            None, now=cfg.BALL_SEARCH_TOTAL_TIMEOUT_S + 1.0)
        self.assertEqual(command.state, self.search.COMPLETE)
        self.assertTrue(command.terminal)
        self.assertEqual(command.angle, 190)

    def test_observa_entre_dois_e_quatro_frames(self):
        self.assertGreaterEqual(cfg.BALL_SEARCH_OBSERVE_FRAMES, 2)
        self.assertLessEqual(cfg.BALL_SEARCH_OBSERVE_FRAMES, 4)


class PolicyFilterTests(unittest.TestCase):
    def test_politica_pode_recusar_uma_cor(self):
        """Um filtro de estratégia pode recusar uma cor sem travar nela."""
        search = PulsedBallSearchController(
            start_time=0.0, accepts_kind=lambda kind: kind == "silver")
        command = search.update(None, now=0.0)
        search.notify_command_written(command.state, now=0.0)
        assentou = cfg.BALL_SEARCH_SETTLE_S
        search.update(None, now=assentou)

        preta = FakeDetection(kind="black", timestamp=assentou + 0.01)
        command = search.update(preta, now=assentou + 0.02)
        self.assertNotEqual(command.state, search.TARGET_STOP)

        prata = FakeDetection(kind="silver", timestamp=assentou + 0.03)
        command = search.update(prata, now=assentou + 0.04)
        self.assertEqual(command.state, search.TARGET_STOP)


class FactoryAndCompatibilityTests(unittest.TestCase):
    def test_fabrica_entrega_a_busca_pulsada_por_padrao(self):
        self.assertTrue(cfg.BALL_SEARCH_PULSED)
        self.assertIsInstance(
            make_search_controller(start_time=0.0),
            PulsedBallSearchController)

    def test_controlador_continuo_aceita_a_interface_unificada(self):
        """O contínuo continua existindo e responde ao mesmo método."""
        search = BallSearchController(start_time=0.0)
        command = search.update(None, now=0.0)
        self.assertEqual(command.state, search.START)
        self.assertFalse(search.notify_command_written(search.START, now=0.0))
        self.assertEqual(search.state, search.ROTATING)

    def test_as_duas_buscas_expõem_a_mesma_interface(self):
        metodos = (
            "update", "frame_allowed", "consume_tracking_reset",
            "notify_command_written", "mark_rotation_started",
            "mark_target_stopped", "mark_full_turn_stopped",
        )
        propriedades = ("target_acquired", "terminal", "target_kind")
        for classe in (PulsedBallSearchController, BallSearchController):
            instancia = classe(start_time=0.0)
            for nome in metodos:
                self.assertTrue(
                    callable(getattr(instancia, nome, None)),
                    f"{classe.__name__} não expõe {nome}")
            for nome in propriedades:
                self.assertTrue(
                    hasattr(instancia, nome),
                    f"{classe.__name__} não expõe {nome}")


if __name__ == "__main__":
    unittest.main()
