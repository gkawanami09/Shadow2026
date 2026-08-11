"""Geometrias da continuacao do percurso depois da area de resgate."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.continuacao_saida import detectar_continuacao_saida  # noqa: E402


class ContinuacaoSaidaTests(unittest.TestCase):
    def setUp(self):
        self.mascara = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8)
        self.centro = config.camera_x // 2

    def test_formato_t_escolhe_a_ponta_distante_em_frente(self):
        cv2.line(
            self.mascara,
            (45, config.camera_y - 30),
            (config.camera_x - 45, config.camera_y - 30),
            255,
            18,
        )
        cv2.line(
            self.mascara,
            (self.centro, config.camera_y - 30),
            (self.centro, 25),
            255,
            18,
        )

        deteccao = detectar_continuacao_saida(self.mascara)

        self.assertIsNotNone(deteccao)
        self.assertAlmostEqual(deteccao.alvo_x, self.centro, delta=20)
        self.assertLess(deteccao.alvo_y, config.camera_y * .25)

    def test_formato_l_rotacionado_escolhe_a_ponta_mais_distante(self):
        cv2.line(
            self.mascara,
            (self.centro - 20, config.camera_y - 15),
            (self.centro - 85, 105),
            255,
            20,
        )
        cv2.line(
            self.mascara,
            (self.centro - 85, 105),
            (config.camera_x - 45, 75),
            255,
            20,
        )

        deteccao = detectar_continuacao_saida(self.mascara)

        self.assertIsNotNone(deteccao)
        self.assertGreater(deteccao.alvo_x, config.camera_x * .75)
        self.assertLess(deteccao.alvo_y, config.camera_y * .50)

    def test_faixa_horizontal_isolada_nao_e_continuacao(self):
        cv2.line(
            self.mascara,
            (35, config.camera_y - 45),
            (config.camera_x - 35, config.camera_y - 45),
            255,
            22,
        )

        self.assertIsNone(detectar_continuacao_saida(self.mascara))

    def test_linha_reta_apontada_para_frente_e_aceita(self):
        cv2.line(
            self.mascara,
            (self.centro, config.camera_y - 1),
            (self.centro + 15, 25),
            255,
            18,
        )

        deteccao = detectar_continuacao_saida(self.mascara)

        self.assertIsNotNone(deteccao)
        self.assertAlmostEqual(deteccao.alvo_x, self.centro + 15, delta=20)

    def test_reta_diagonal_lateral_ainda_nao_e_entregue(self):
        cv2.line(
            self.mascara,
            (self.centro, config.camera_y - 1),
            (config.camera_x - 20, 110),
            255,
            18,
        )

        self.assertIsNone(detectar_continuacao_saida(self.mascara))


if __name__ == "__main__":
    unittest.main()
