"""Testes da faixa transversal que posiciona o giro verde."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.faixa_verde import (altura_faixa_transversal,
                               tem_ramo_lateral)  # noqa: E402


class FaixaVerdeTests(unittest.TestCase):
    def mascara(self):
        mascara = np.zeros((config.camera_y, config.camera_x), dtype=np.uint8)
        meio = config.camera_x // 2
        mascara[:, meio - 8:meio + 9] = 255
        return mascara

    def test_linha_vertical_sozinha_nao_libera_giro(self):
        self.assertEqual(altura_faixa_transversal(
            self.mascara(), "left"), -1.)

    def test_faixa_esquerda_publica_altura_central(self):
        mascara = self.mascara()
        mascara[120:133, 50:config.camera_x // 2 + 8] = 255

        self.assertAlmostEqual(
            altura_faixa_transversal(mascara, "left"), 126.)

    def test_faixa_direita_publica_altura_central(self):
        mascara = self.mascara()
        mascara[150:163, config.camera_x // 2 - 8:400] = 255

        self.assertAlmostEqual(
            altura_faixa_transversal(mascara, "right"), 156.)

    def test_dois_riscos_separados_nao_imitam_faixa_continua(self):
        mascara = self.mascara()
        mascara[120:133, 30:90] = 255

        self.assertEqual(altura_faixa_transversal(
            mascara, "left"), -1.)

    def test_t_com_ramo_apenas_a_direita_e_intersecao(self):
        mascara = self.mascara()
        mascara[120:133, config.camera_x // 2 - 8:400] = 255
        self.assertTrue(tem_ramo_lateral(mascara))

    def test_linha_vertical_nao_e_intersecao_lateral(self):
        self.assertFalse(tem_ramo_lateral(self.mascara()))

    def test_risco_desligado_do_eixo_nao_e_intersecao(self):
        mascara = self.mascara()
        mascara[120:133, 330:440] = 255
        self.assertFalse(tem_ramo_lateral(mascara))


if __name__ == "__main__":
    unittest.main()
