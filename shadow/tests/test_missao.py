"""Testes do coordenador da missão, do inventário e da ordem do handoff."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.missao import (  # noqa: E402
    EXPECTED_TOTAL,
    HANDOFF_TO_LINE,
    HANDOFF_TO_RESCUE,
    HandoffError,
    HandoffExecutor,
    MissionCoordinator,
    MissionError,
    MissionState,
    POLICY_NEAREST_VALID,
    POLICY_SILVER_FIRST,
    RescueInventory,
    TRIANGLE_BY_VICTIM,
    index_of,
)


def _resgatar(coordinator, kind):
    """Percorre um ciclo completo de uma vítima até o depósito confirmado."""
    coordinator.on_target_locked(kind)
    coordinator.on_pickup_started(kind)
    coordinator.on_victim_secured(kind)
    coordinator.on_searching_triangle()
    coordinator.on_triangle_reached()
    coordinator.on_grippers_restored()
    return coordinator.on_deposit_confirmed()


class InventoryTests(unittest.TestCase):
    def test_duas_pratas_e_uma_preta_completam_a_sala(self):
        inventory = RescueInventory()
        inventory.record_deposit("silver")
        inventory.record_deposit("black")
        self.assertFalse(inventory.complete)
        inventory.record_deposit("silver")
        self.assertTrue(inventory.complete)
        self.assertEqual(inventory.total_deposited, EXPECTED_TOTAL)
        self.assertEqual(inventory.silver_deposited, 2)
        self.assertEqual(inventory.black_deposited, 1)

    def test_terceira_prata_e_recusada(self):
        inventory = RescueInventory()
        inventory.record_deposit("silver")
        inventory.record_deposit("silver")
        with self.assertRaises(MissionError):
            inventory.record_deposit("silver")

    def test_segunda_preta_e_recusada(self):
        inventory = RescueInventory()
        inventory.record_deposit("black")
        with self.assertRaises(MissionError):
            inventory.record_deposit("black")

    def test_cor_desconhecida_falha(self):
        with self.assertRaises(MissionError):
            RescueInventory().record_deposit("azul")


class DestinationTests(unittest.TestCase):
    def test_prata_vai_para_o_verde_e_preta_para_o_vermelho(self):
        self.assertEqual(TRIANGLE_BY_VICTIM["silver"], "green")
        self.assertEqual(TRIANGLE_BY_VICTIM["black"], "red")

    def test_o_destino_acompanha_a_vitima_presa(self):
        coordinator = MissionCoordinator()
        self._ate_o_resgate(coordinator)
        coordinator.on_target_locked("silver")
        coordinator.on_pickup_started("silver")
        coordinator.on_victim_secured("silver")
        self.assertEqual(coordinator.target_triangle, "green")

        coordinator.on_searching_triangle()
        coordinator.on_triangle_reached()
        coordinator.on_grippers_restored()
        coordinator.on_deposit_confirmed()
        # Sem vítima presa, nenhum triângulo pode comandar o robô.
        self.assertIsNone(coordinator.target_triangle)

        coordinator.on_target_locked("black")
        coordinator.on_pickup_started("black")
        coordinator.on_victim_secured("black")
        self.assertEqual(coordinator.target_triangle, "red")

    @staticmethod
    def _ate_o_resgate(coordinator):
        coordinator.on_entry_candidate()
        coordinator.on_entry_confirmed()
        coordinator.on_zone_entered()
        coordinator.on_rescue_started()


class MissionFlowTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = MissionCoordinator()
        self.coordinator.on_entry_candidate()
        self.coordinator.on_entry_confirmed()
        self.coordinator.on_zone_entered()
        self.coordinator.on_rescue_started()

    def test_percurso_completo_de_tres_vitimas(self):
        _resgatar(self.coordinator, "silver")
        self.assertEqual(self.coordinator.state, MissionState.RESCUE_SCAN)
        _resgatar(self.coordinator, "black")
        self.assertEqual(self.coordinator.state, MissionState.RESCUE_SCAN)
        _resgatar(self.coordinator, "silver")
        self.assertEqual(
            self.coordinator.state, MissionState.VERIFY_RESCUE_COMPLETE)
        self.assertTrue(self.coordinator.inventory.complete)

    def test_uma_vitima_por_vez_ate_o_deposito(self):
        """Não existe acumular três: prender uma bloqueia buscar a próxima."""
        self.coordinator.on_target_locked("silver")
        self.coordinator.on_pickup_started("silver")
        self.coordinator.on_victim_secured("silver")
        self.assertEqual(self.coordinator.state, MissionState.CARRY_READY)
        with self.assertRaises(MissionError):
            self.coordinator.on_target_locked("black")

    def test_contagem_so_apos_restaurar_as_garras(self):
        self.coordinator.on_target_locked("silver")
        self.coordinator.on_pickup_started("silver")
        self.coordinator.on_victim_secured("silver")
        self.coordinator.on_searching_triangle()
        self.coordinator.on_triangle_reached()
        self.assertEqual(self.coordinator.inventory.total_deposited, 0)
        # Chegar ao triângulo não conta: falta restaurar as garras.
        with self.assertRaises(MissionError):
            self.coordinator.on_deposit_confirmed()
        self.coordinator.on_grippers_restored()
        self.coordinator.on_deposit_confirmed()
        self.assertEqual(self.coordinator.inventory.total_deposited, 1)

    def test_uma_varredura_vazia_gera_recuperacao_e_nao_encerra(self):
        _resgatar(self.coordinator, "silver")
        self.coordinator.on_empty_sweep()
        self.assertEqual(
            self.coordinator.state, MissionState.RESCUE_RECOVERY)
        self.assertFalse(self.coordinator.inventory.complete)

    def test_segunda_varredura_vazia_encerra_sem_laco_infinito(self):
        self.coordinator.on_empty_sweep()
        self.coordinator.on_empty_sweep()
        self.assertEqual(
            self.coordinator.state, MissionState.VERIFY_RESCUE_COMPLETE)

    def test_deposito_zera_a_contagem_de_varreduras_vazias(self):
        self.coordinator.on_empty_sweep()
        self.assertEqual(self.coordinator.empty_sweeps, 1)
        self.coordinator.on_target_locked("silver")
        self.coordinator.on_pickup_started("silver")
        self.coordinator.on_victim_secured("silver")
        self.coordinator.on_searching_triangle()
        self.coordinator.on_triangle_reached()
        self.coordinator.on_grippers_restored()
        self.coordinator.on_deposit_confirmed()
        self.assertEqual(self.coordinator.empty_sweeps, 0)

    def test_detector_de_vitimas_desliga_apos_as_tres(self):
        self.assertTrue(self.coordinator.victim_detector_enabled)
        _resgatar(self.coordinator, "silver")
        _resgatar(self.coordinator, "silver")
        _resgatar(self.coordinator, "black")
        self.assertFalse(self.coordinator.victim_detector_enabled)

    def test_faixa_preta_so_existe_no_estado_de_saida(self):
        self.assertFalse(self.coordinator.exit_detector_enabled)
        _resgatar(self.coordinator, "silver")
        # Durante a busca da próxima vítima a saída continua invisível.
        self.assertFalse(self.coordinator.exit_detector_enabled)
        _resgatar(self.coordinator, "silver")
        _resgatar(self.coordinator, "black")
        self.coordinator.on_rescue_verified()
        self.coordinator.on_final_triangles_mapped()
        self.assertEqual(self.coordinator.state, MissionState.FIND_BLACK_EXIT)
        self.assertTrue(self.coordinator.exit_detector_enabled)

    def test_saida_retorno_ao_segue_linha_e_vermelho_final(self):
        _resgatar(self.coordinator, "silver")
        _resgatar(self.coordinator, "silver")
        _resgatar(self.coordinator, "black")
        self.coordinator.on_rescue_verified()
        self.coordinator.on_final_triangles_mapped()
        self.coordinator.on_exit_confirmed()
        self.coordinator.on_exit_crossed()
        self.assertEqual(
            self.coordinator.state, MissionState.STOP_AND_HANDOFF_TO_LINE)
        self.coordinator.on_line_resumed()
        self.assertEqual(self.coordinator.state, MissionState.FOLLOW_LINE)
        self.coordinator.on_red_finish()
        self.assertEqual(self.coordinator.state, MissionState.RED_FINISH)

    def test_evento_fora_de_ordem_e_recusado(self):
        with self.assertRaises(MissionError):
            self.coordinator.on_exit_confirmed()


class PolicyTests(unittest.TestCase):
    def test_politica_padrao_aceita_qualquer_cor(self):
        coordinator = MissionCoordinator(policy=POLICY_NEAREST_VALID)
        self.assertEqual(
            coordinator.preferred_kinds(), ("silver", "black"))

    def test_silver_first_adia_a_preta(self):
        coordinator = MissionCoordinator(policy=POLICY_SILVER_FIRST)
        self.assertEqual(coordinator.preferred_kinds(), ("silver",))
        self.assertFalse(coordinator.wants("black"))
        coordinator.inventory.record_deposit("silver")
        self.assertFalse(coordinator.wants("black"))
        coordinator.inventory.record_deposit("silver")
        # Com as duas vivas entregues, a morta volta a ser elegível.
        self.assertTrue(coordinator.wants("black"))

    def test_politica_invalida_falha(self):
        with self.assertRaises(MissionError):
            MissionCoordinator(policy="qualquer_uma")

    def test_cor_ja_completa_nao_e_mais_desejada(self):
        coordinator = MissionCoordinator()
        coordinator.inventory.record_deposit("black")
        self.assertFalse(coordinator.wants("black"))
        self.assertTrue(coordinator.wants("silver"))


class FakeSystem:
    """Sistema instrumentado: registra chamadas e nunca toca em hardware."""

    def __init__(self, falha_em=None):
        self.calls = []
        self.falha_em = falha_em

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def step():
            self.calls.append(name)
            if name == self.falha_em:
                raise RuntimeError(f"falha simulada em {name}")
        return step


class HandoffToRescueTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem()
        self.log = HandoffExecutor(self.system, HANDOFF_TO_RESCUE).run()

    def test_parar_e_o_primeiro_passo(self):
        self.assertEqual(self.log[0], "stop_motors")

    def test_led_apagado_antes_de_liberar_a_serial(self):
        self.assertLess(
            index_of(self.log, "led_off"),
            index_of(self.log, "release_serial"))

    def test_filhos_encerrados_antes_de_fechar_a_camera_de_linha(self):
        self.assertLess(
            index_of(self.log, "join_line_children"),
            index_of(self.log, "close_line_camera"))
        self.assertLess(
            index_of(self.log, "assert_line_children_dead"),
            index_of(self.log, "close_line_camera"))

    def test_camera_de_linha_fecha_antes_de_abrir_a_de_resgate(self):
        """Requisito absoluto: nunca as duas câmeras abertas ao mesmo tempo."""
        self.assertLess(
            index_of(self.log, "close_line_camera"),
            index_of(self.log, "open_rescue_camera"))

    def test_serial_nunca_tem_dois_donos(self):
        self.assertLess(
            index_of(self.log, "release_serial"),
            index_of(self.log, "open_rescue_serial"))

    def test_lock_liberado_antes_de_ser_readquirido(self):
        self.assertLess(
            index_of(self.log, "release_motor_lock"),
            index_of(self.log, "acquire_rescue_motor_lock"))

    def test_led_reafirmado_apagado_na_serial_nova(self):
        self.assertLess(
            index_of(self.log, "open_rescue_serial"),
            index_of(self.log, "assert_led_off"))
        self.assertLess(
            index_of(self.log, "assert_led_off"),
            index_of(self.log, "open_rescue_camera"))

    def test_resgate_inicia_por_ultimo(self):
        self.assertEqual(self.log[-1], "start_rescue")


class HandoffToLineTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem()
        self.log = HandoffExecutor(self.system, HANDOFF_TO_LINE).run()

    def test_parar_antes_de_fechar_a_camera_de_resgate(self):
        self.assertLess(
            index_of(self.log, "stop_motors"),
            index_of(self.log, "close_rescue_camera"))

    def test_camera_de_resgate_fecha_antes_de_abrir_a_de_linha(self):
        self.assertLess(
            index_of(self.log, "close_rescue_camera"),
            index_of(self.log, "open_line_camera"))

    def test_led_aceso_ao_voltar_ao_percurso(self):
        self.assertIn("led_on", self.log)
        self.assertLess(
            index_of(self.log, "open_line_serial"),
            index_of(self.log, "led_on"))

    def test_linha_e_reacquirida_no_fim(self):
        self.assertEqual(self.log[-1], "reacquire_line")


class HandoffFailureTests(unittest.TestCase):
    def test_falha_interrompe_e_para_os_motores(self):
        system = FakeSystem(falha_em="close_line_camera")
        executor = HandoffExecutor(system, HANDOFF_TO_RESCUE)
        with self.assertRaises(HandoffError):
            executor.run()
        self.assertEqual(executor.failed_step, "close_line_camera")
        # A câmera de resgate nunca chegou a abrir.
        self.assertNotIn("open_rescue_camera", system.calls)
        # E os motores foram parados de novo depois da falha.
        self.assertEqual(system.calls[-1], "stop_motors")

    def test_falha_ao_encerrar_filho_nao_abre_a_camera_de_resgate(self):
        system = FakeSystem(falha_em="join_line_children")
        with self.assertRaises(HandoffError):
            HandoffExecutor(system, HANDOFF_TO_RESCUE).run()
        self.assertNotIn("open_rescue_camera", system.calls)
        self.assertNotIn("start_rescue", system.calls)

    def test_passo_ausente_no_sistema_e_erro_explicito(self):
        class Incompleto:
            def stop_motors(self):
                pass

        with self.assertRaises(HandoffError):
            HandoffExecutor(Incompleto(), HANDOFF_TO_RESCUE).run()


if __name__ == "__main__":
    unittest.main()
