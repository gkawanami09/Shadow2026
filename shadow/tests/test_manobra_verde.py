"""Testes da transicao geometrica para o giro verde."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.manobra_verde import ramo_pronto_para_giro  # noqa: E402


class ManobraVerdeTests(unittest.TestCase):
    def test_ramo_distante_ainda_nao_inicia_tanque(self):
        self.assertFalse(ramo_pronto_para_giro(
            "right",
            ponto_alvo_x=config.camera_x - 1,
            ponto_alvo_y=config.camera_y * .30,
        ))

    def test_ramo_direito_proximo_inicia_tanque(self):
        self.assertTrue(ramo_pronto_para_giro(
            "right",
            ponto_alvo_x=config.camera_x - 1,
            ponto_alvo_y=config.camera_y * .60,
        ))

    def test_ramo_esquerdo_proximo_inicia_tanque(self):
        self.assertTrue(ramo_pronto_para_giro(
            "left",
            ponto_alvo_x=1,
            ponto_alvo_y=config.camera_y * .60,
        ))

    def test_alvo_do_lado_oposto_nao_desfaz_direcao_travada(self):
        self.assertFalse(ramo_pronto_para_giro(
            "right",
            ponto_alvo_x=1,
            ponto_alvo_y=config.camera_y * .80,
        ))


if __name__ == "__main__":
    unittest.main()
