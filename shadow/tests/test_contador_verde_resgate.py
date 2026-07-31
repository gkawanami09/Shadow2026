"""Testes da contagem de passagens verdes durante a busca pulsada."""

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.contador_verde_resgate import (  # noqa: E402
    BUSCA_CONCLUIR,
    BUSCA_FALHAR,
    BUSCA_REINICIAR,
    ContadorVerdeBusca,
    decidir_apos_varredura,
)


@dataclass(frozen=True)
class VerdeFalso:
    timestamp: float
    confirmed: bool = True


class ContadorVerdeBuscaTests(unittest.TestCase):
    def test_frames_seguidos_do_mesmo_verde_contam_uma_vez(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=3)

        self.assertTrue(contador.observar(VerdeFalso(1.0)))
        self.assertFalse(contador.observar(VerdeFalso(2.0)))
        self.assertFalse(contador.observar(VerdeFalso(3.0)))
        self.assertEqual(contador.quantidade, 1)
        self.assertFalse(contador.completo)

    def test_verde_precisa_sumir_antes_de_contar_de_novo(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=3)

        contador.observar(VerdeFalso(1.0))
        contador.observar(None)
        contador.observar(None)
        self.assertFalse(contador.observar(VerdeFalso(2.0)))
        self.assertEqual(contador.quantidade, 1)

        contador.observar(None)
        contador.observar(None)
        contador.observar(None)
        self.assertTrue(contador.observar(VerdeFalso(3.0)))
        self.assertTrue(contador.completo)

    def test_mesmo_giro_com_oscilacao_nao_conta_o_verde_duas_vezes(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=3)

        self.assertTrue(contador.observar(VerdeFalso(1.0), varredura=0))
        for _ in range(6):
            contador.observar(None, varredura=0)
        self.assertFalse(
            contador.observar(VerdeFalso(2.0), varredura=0))
        self.assertEqual(contador.quantidade, 1)
        self.assertFalse(contador.completo)

    def test_segunda_varredura_pode_confirmar_a_segunda_passagem(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=3)

        contador.observar(VerdeFalso(1.0), varredura=0)
        for _ in range(3):
            contador.observar(None, varredura=0)

        self.assertTrue(
            contador.observar(VerdeFalso(2.0), varredura=1))
        self.assertEqual(contador.quantidade, 2)
        self.assertTrue(contador.completo)

    def test_frame_durante_giro_nao_conta_nem_rearma(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=1)

        contador.observar(VerdeFalso(1.0))
        contador.observar(None, permitido=False)
        self.assertFalse(
            contador.observar(VerdeFalso(2.0), permitido=True))
        self.assertEqual(contador.quantidade, 1)

    def test_deteccao_ainda_nao_confirmada_nao_conta(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=1)

        self.assertFalse(
            contador.observar(VerdeFalso(1.0, confirmed=False)))
        self.assertTrue(contador.observar(VerdeFalso(2.0, confirmed=True)))
        self.assertEqual(contador.quantidade, 1)

    def test_coleta_concluida_pode_zerar_a_contagem(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=1)
        contador.observar(VerdeFalso(1.0))
        contador.observar(None)
        contador.observar(VerdeFalso(2.0))
        self.assertTrue(contador.completo)

        contador.reset()

        self.assertEqual(contador.quantidade, 0)
        self.assertFalse(contador.completo)
        self.assertTrue(contador.observar(VerdeFalso(3.0)))

    def test_mesmo_timestamp_nao_e_contado_duas_vezes(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=1)

        contador.observar(VerdeFalso(1.0))
        contador.observar(None)
        self.assertFalse(contador.observar(VerdeFalso(1.0)))
        self.assertEqual(contador.quantidade, 1)

    def test_fim_da_varredura_reinicia_conclui_ou_falha(self):
        contador = ContadorVerdeBusca(necessario=2, frames_para_rearmar=1)
        self.assertEqual(
            decidir_apos_varredura(contador, 1),
            BUSCA_REINICIAR,
        )

        contador.observar(VerdeFalso(1.0))
        contador.observar(None)
        contador.observar(VerdeFalso(2.0))
        self.assertEqual(
            decidir_apos_varredura(contador, 1),
            BUSCA_CONCLUIR,
        )

        contador.reset()
        self.assertEqual(
            decidir_apos_varredura(
                contador, cfg.RESCUE_SEARCH_MAX_EMPTY_SWEEPS),
            BUSCA_FALHAR,
        )


if __name__ == "__main__":
    unittest.main()
