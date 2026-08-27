"""Testes da transicao geometrica para o giro verde."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.manobra_verde import (  # noqa: E402
    correcao_aproximacao,
    ramo_pronto_para_giro,
)


class ManobraVerdeTests(unittest.TestCase):
    def test_aproximacao_reta_nao_antecipa_ramo_travado(self):
        self.assertEqual(correcao_aproximacao(config.camera_x / 2), 0.)

    def test_aproximacao_corrige_apenas_deslocamento_da_base(self):
        direita = correcao_aproximacao(config.camera_x * .75)
        esquerda = correcao_aproximacao(config.camera_x * .25)

        self.assertGreater(direita, 0.)
        self.assertAlmostEqual(direita, -esquerda)

    def test_aproximacao_limita_correcao_para_continuar_avancando(self):
        self.assertEqual(
            correcao_aproximacao(config.camera_x),
            config.GREEN_APPROACH_MAX_CORRECTION,
        )

    def test_ramo_distante_ainda_nao_inicia_tanque(self):
        self.assertFalse(ramo_pronto_para_giro(
            "right",
            faixa_transversal_y=config.camera_y * .30,
        ))

    def test_ramo_direito_proximo_inicia_tanque(self):
        self.assertTrue(ramo_pronto_para_giro(
            "right",
            faixa_transversal_y=config.camera_y * .70,
        ))

    def test_ramo_esquerdo_proximo_inicia_tanque(self):
        self.assertTrue(ramo_pronto_para_giro(
            "left",
            faixa_transversal_y=config.camera_y * .70,
        ))

    def test_faixa_ausente_nao_inicia_tanque(self):
        self.assertFalse(ramo_pronto_para_giro(
            "right",
            faixa_transversal_y=-1,
        ))


if __name__ == "__main__":
    unittest.main()
