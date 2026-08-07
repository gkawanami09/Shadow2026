"""Confirmação da sala de resgate pela segunda câmera.

A faixa prata prova que há prata à frente. Estes testes cobrem a pergunta
seguinte, que é a que decide a prova: há uma SALA ali?
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.confirmacao_entrada import (  # noqa: E402
    ConfirmacaoEntradaResgate,
)
from controle.missao import (  # noqa: E402
    MissionCoordinator,
    MissionError,
    MissionState,
)


@dataclass(frozen=True)
class FakeDeteccao:
    confirmed: bool
    kind: str = "silver"


class ConfirmacaoEntradaTests(unittest.TestCase):
    def _confirmacao(self, janela=4.0):
        return ConfirmacaoEntradaResgate(janela_s=janela, inicio=100.0)

    def test_sem_nada_no_quadro_nao_confirma(self):
        c = self._confirmacao()
        self.assertFalse(c.observar(agora=100.5))
        self.assertFalse(c.observar(vitima=None, marcadores={}, agora=101.0))
        self.assertFalse(c.confirmado)

    def test_vitima_confirmada_libera_o_resgate(self):
        c = self._confirmacao()
        self.assertTrue(
            c.observar(vitima=FakeDeteccao(True, "black"), agora=100.5))
        self.assertEqual(c.motivo, "vitima:black")

    def test_triangulo_verde_libera_o_resgate(self):
        c = self._confirmacao()
        self.assertTrue(c.observar(
            marcadores={"green": FakeDeteccao(True), "red": None},
            agora=100.5))
        self.assertEqual(c.motivo, "marcador:green")

    def test_triangulo_vermelho_libera_o_resgate(self):
        c = self._confirmacao()
        self.assertTrue(c.observar(
            marcadores={"green": None, "red": FakeDeteccao(True)},
            agora=100.5))
        self.assertEqual(c.motivo, "marcador:red")

    def test_candidato_nao_confirmado_nao_basta(self):
        """A evidência é a MESMA que o resgate exige para agir.

        Aceitar um candidato solto trocaria o filtro fraco da faixa por outro
        filtro fraco, e a etapa inteira perderia o sentido.
        """
        c = self._confirmacao()
        self.assertFalse(c.observar(
            vitima=FakeDeteccao(False),
            marcadores={"green": FakeDeteccao(False),
                        "red": FakeDeteccao(False)},
            agora=100.5))
        self.assertFalse(c.confirmado)

    def test_expira_no_prazo_e_nao_antes(self):
        c = self._confirmacao(janela=4.0)
        self.assertFalse(c.expirou(103.9))
        self.assertTrue(c.expirou(104.0))

    def test_confirmacao_e_definitiva_e_nunca_expira(self):
        """Quem confirmou a sala não pode ser desarmado pelo relógio."""
        c = self._confirmacao(janela=4.0)
        c.observar(vitima=FakeDeteccao(True), agora=100.5)
        self.assertFalse(c.expirou(999.0))
        self.assertTrue(c.observar(agora=999.0))

    def test_restante_nunca_fica_negativo(self):
        c = self._confirmacao(janela=4.0)
        self.assertAlmostEqual(c.restante(102.0), 2.0)
        self.assertEqual(c.restante(200.0), 0.0)

    def test_marcador_desconhecido_e_ignorado(self):
        c = self._confirmacao()
        self.assertFalse(
            c.observar(marcadores={"azul": FakeDeteccao(True)}, agora=100.5))


class EntradaFalsaNaMissaoTests(unittest.TestCase):
    def _ate_o_resgate(self):
        coordinator = MissionCoordinator()
        coordinator.on_entry_candidate()
        coordinator.on_entry_confirmed()
        coordinator.on_zone_entered()
        coordinator.on_rescue_started()
        return coordinator

    def test_entrada_falsa_devolve_ao_percurso(self):
        coordinator = self._ate_o_resgate()
        self.assertEqual(
            coordinator.on_false_entry(), MissionState.FOLLOW_LINE)
        self.assertEqual(coordinator.false_entries, 1)

    def test_entrada_falsa_nao_mexe_no_inventario(self):
        """Nada foi resgatado e nada foi perdido: o placar não pode mudar."""
        coordinator = self._ate_o_resgate()
        antes = (
            coordinator.inventory.silver_deposited,
            coordinator.inventory.black_deposited,
        )
        coordinator.on_false_entry()
        depois = (
            coordinator.inventory.silver_deposited,
            coordinator.inventory.black_deposited,
        )
        self.assertEqual(antes, depois)
        self.assertIsNone(coordinator.carrying)

    def test_pode_voltar_a_procurar_a_entrada_depois_da_falsa(self):
        """O robô precisa achar a sala VERDADEIRA mais adiante."""
        coordinator = self._ate_o_resgate()
        coordinator.on_false_entry()
        self.assertEqual(
            coordinator.on_entry_candidate(),
            MissionState.ENTRY_SILVER_CANDIDATE)
        self.assertEqual(
            coordinator.on_entry_confirmed(), MissionState.ENTER_RESCUE_ZONE)

    def test_entrada_falsa_so_vale_vindo_do_resgate(self):
        coordinator = MissionCoordinator()
        with self.assertRaises(MissionError):
            coordinator.on_false_entry()

    def test_contador_de_entradas_falsas_acumula(self):
        coordinator = self._ate_o_resgate()
        coordinator.on_false_entry()
        coordinator.on_entry_candidate()
        coordinator.on_entry_confirmed()
        coordinator.on_zone_entered()
        coordinator.on_rescue_started()
        coordinator.on_false_entry()
        self.assertEqual(coordinator.false_entries, 2)
        self.assertGreaterEqual(coordinator.MAX_FALSE_ENTRIES, 1)


if __name__ == "__main__":
    unittest.main()
