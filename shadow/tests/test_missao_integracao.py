"""Testes dos pontos de integração da missão dentro do resgate.

Aqui não há câmera, serial nem motores: apenas as decisões que ``resgate.py``
e ``mission.py`` tomam com base no inventário e na fase de saída.
"""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
import config_resgate as cfg  # noqa: E402
from controle.missao import (  # noqa: E402
    MissionCoordinator,
    RescueInventory,
    POLICY_SILVER_FIRST,
)
from controle.saida_resgate import ExitPhaseController  # noqa: E402
import resgate  # noqa: E402


class ExitCodeTests(unittest.TestCase):
    """O supervisor precisa distinguir 'saiu da sala' de 'só resgatou'."""

    @staticmethod
    def _inventario_completo():
        inventory = RescueInventory()
        inventory.record_deposit("silver")
        inventory.record_deposit("silver")
        inventory.record_deposit("black")
        return inventory

    def test_tres_vitimas_e_saida_confirmada_e_sucesso(self):
        saida = ExitPhaseController(start_time=0.0)
        saida.state = ExitPhaseController.DONE
        self.assertEqual(
            resgate.mission_exit_code(self._inventario_completo(), saida),
            resgate.EXIT_OK)

    def test_tres_vitimas_sem_achar_a_saida_nao_e_sucesso(self):
        saida = ExitPhaseController(start_time=0.0)
        saida.state = ExitPhaseController.FAILED
        self.assertEqual(
            resgate.mission_exit_code(self._inventario_completo(), saida),
            resgate.EXIT_INCOMPLETE)

    def test_saida_encontrada_mas_vitimas_faltando_nao_e_sucesso(self):
        inventory = RescueInventory()
        inventory.record_deposit("silver")
        saida = ExitPhaseController(start_time=0.0)
        saida.state = ExitPhaseController.DONE
        self.assertEqual(
            resgate.mission_exit_code(inventory, saida),
            resgate.EXIT_INCOMPLETE)

    def test_modo_por_etapas_nao_exige_a_saida(self):
        self.assertEqual(
            resgate.mission_exit_code(
                self._inventario_completo(), None, no_exit_phase=True),
            resgate.EXIT_OK)

    def test_codigos_batem_com_os_lidos_pelo_supervisor(self):
        import mission
        self.assertEqual(resgate.EXIT_OK, mission.RESCUE_EXIT_OK)
        self.assertEqual(
            resgate.EXIT_INCOMPLETE, mission.RESCUE_EXIT_INCOMPLETE)


class SearchPolicyWiringTests(unittest.TestCase):
    """A busca recebe o filtro de cor da política ativa."""

    def test_reset_para_a_proxima_esfera_usa_a_politica(self):
        class FakeWorker:
            def __init__(self):
                self.reset_calls = 0

            def reset_tracking(self):
                self.reset_calls += 1

        class FakeGate:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

        coordinator = MissionCoordinator(policy=POLICY_SILVER_FIRST)
        worker, gate = FakeWorker(), FakeGate()
        search, pickup = resgate._reset_for_next_search(
            worker, gate, 0.0, coordinator)

        self.assertEqual(worker.reset_calls, 1)
        self.assertEqual(gate.reset_calls, 1)
        self.assertFalse(pickup.started)
        # silver_first: a preta é recusada enquanto faltar uma prata.
        self.assertFalse(search._kind_allowed("black"))
        self.assertTrue(search._kind_allowed("silver"))

        coordinator.inventory.record_deposit("silver")
        coordinator.inventory.record_deposit("silver")
        self.assertTrue(search._kind_allowed("black"))

    def test_cor_ja_completa_nao_volta_a_ser_buscada(self):
        coordinator = MissionCoordinator()
        coordinator.inventory.record_deposit("black")
        search, _ = resgate._reset_for_next_search(
            type("W", (), {"reset_tracking": lambda self: None})(),
            type("G", (), {"reset": lambda self: None})(),
            0.0, coordinator)
        self.assertFalse(search._kind_allowed("black"))
        self.assertTrue(search._kind_allowed("silver"))


class ConfigProfileSeparationTests(unittest.TestCase):
    """Perfis de câmera diferentes não podem se sobrepor."""

    def test_faixa_prata_pertence_a_camera_de_linha(self):
        self.assertTrue(hasattr(config, "ENTRY_SILVER_MIN_DEFAULT"))
        self.assertTrue(hasattr(config, "ENTRY_SILVER_MAX_DEFAULT"))
        # E não vaza para o módulo do resgate.
        self.assertFalse(hasattr(cfg, "ENTRY_SILVER_MIN_DEFAULT"))

    def test_faixa_preta_pertence_a_camera_de_resgate(self):
        self.assertTrue(hasattr(cfg, "EXIT_BLACK_HSV_MIN"))
        self.assertFalse(hasattr(config, "EXIT_BLACK_HSV_MIN"))

    def test_vitima_e_entrada_tem_limiares_independentes(self):
        """Prata da vítima e prata da entrada não compartilham constantes."""
        self.assertNotEqual(
            id(config.ENTRY_SILVER_MAX_SATURATION),
            id(cfg.BALL_SILVER_S_MAX))
        # A entrada é calibrada em HSV; a vítima usa uma bateria própria de
        # assinaturas (lisa, amassada, clara, tingida). São modelos distintos.
        self.assertTrue(hasattr(cfg, "BALL_SILVER_SMOOTH_INNER_V_MIN"))
        self.assertFalse(hasattr(config, "BALL_SILVER_SMOOTH_INNER_V_MIN"))


class LineFollowerUnchangedTests(unittest.TestCase):
    """Rodar `main.py` sozinho não pode mudar de comportamento."""

    def test_sem_mission_mode_o_detector_de_entrada_nao_e_construido(self):
        from shared.dados_compartilhados import mission_mode
        from visao import entrada_missao

        anterior = mission_mode.value
        try:
            mission_mode.value = False
            self.assertIsNone(entrada_missao.build_entry_gate())
        finally:
            mission_mode.value = anterior

    def test_atualizacao_sem_portao_e_inofensiva(self):
        from visao import entrada_missao
        # Sem portão (modo main.py) a função retorna sem tocar em nada.
        self.assertIsNone(
            entrada_missao.update_entry_silver(None, None, 0.0))

    def test_valores_da_missao_comecam_desligados(self):
        import shared.dados_compartilhados as shared
        self.assertFalse(shared.rescue_requested.value)
        self.assertFalse(shared.red_finished.value)
        self.assertFalse(shared.entry_silver_confirmed.value)

    def test_entrada_desarmada_nao_reprocessa(self):
        """Depois de entrar na sala, a faixa prata deixa de ser avaliada."""
        from shared.dados_compartilhados import (entry_armed,
                                                 entry_silver_detected)
        from visao import entrada_missao

        class GatePlaceholder:
            def update(self, *args, **kwargs):
                raise AssertionError(
                    "o portão não pode ser consultado com a entrada desarmada")

        anterior = entry_armed.value
        try:
            entry_armed.value = False
            entry_silver_detected.value = True
            entrada_missao.update_entry_silver(GatePlaceholder(), None, 0.0)
            self.assertFalse(entry_silver_detected.value)
        finally:
            entry_armed.value = anterior


class PulsedSearchConfigTests(unittest.TestCase):
    def test_pulso_e_setores_batem_com_o_360_calibrado(self):
        """setores × pulso deve ficar próximo do 360 temporizado."""
        cobertura = cfg.BALL_SEARCH_SECTORS * cfg.BALL_SEARCH_PULSE_S
        self.assertAlmostEqual(
            cobertura, cfg.BALL_SEARCH_FULL_TURN_S, delta=1.0)

    def test_timeout_total_cobre_a_varredura_com_pausas(self):
        pausas = cfg.BALL_SEARCH_SECTORS * (
            cfg.BALL_SEARCH_SETTLE_S + cfg.BALL_SEARCH_OBSERVE_TIMEOUT_S)
        minimo = cfg.BALL_SEARCH_FULL_TURN_S + pausas
        self.assertGreaterEqual(cfg.BALL_SEARCH_TOTAL_TIMEOUT_S, minimo)


if __name__ == "__main__":
    unittest.main()
