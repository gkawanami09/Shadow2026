"""Testes da geometria multiponto do seguidor de linha V2."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.trajetoria_linha import estimar_trajetoria  # noqa: E402


ALTURA = 252
LARGURA = 448


def _contorno(pontos):
    return np.asarray(pontos, dtype=np.int32).reshape((-1, 1, 2))


class EstimadorTrajetoriaTests(unittest.TestCase):
    def test_reta_central_tem_tres_erros_proximos_de_zero(self):
        resultado = estimar_trajetoria(_contorno([
            (210, 251), (210, 60), (238, 60), (238, 251),
        ]), (ALTURA, LARGURA))

        self.assertTrue(resultado.valida)
        self.assertGreater(resultado.confianca, .9)
        self.assertAlmostEqual(resultado.lateral, 0., places=2)
        self.assertAlmostEqual(resultado.orientacao, 0., places=2)
        self.assertAlmostEqual(resultado.curvatura, 0., places=2)

    def test_reta_deslocada_publica_erro_lateral(self):
        resultado = estimar_trajetoria(_contorno([
            (260, 251), (260, 60), (288, 60), (288, 251),
        ]), (ALTURA, LARGURA))

        self.assertTrue(resultado.valida)
        self.assertGreater(resultado.lateral, .20)
        self.assertAlmostEqual(resultado.orientacao, 0., places=2)

    def test_linha_inclinada_publica_orientacao_para_direita(self):
        resultado = estimar_trajetoria(_contorno([
            (210, 251), (290, 60), (318, 60), (238, 251),
        ]), (ALTURA, LARGURA))

        self.assertTrue(resultado.valida)
        self.assertGreater(resultado.orientacao, .25)

    def test_perspectiva_baixa_aceita_linha_larga_no_rodape(self):
        resultado = estimar_trajetoria(_contorno([
            (105, 55), (145, 55), (390, 251), (155, 251),
        ]), (ALTURA, LARGURA))

        self.assertTrue(resultado.valida)
        self.assertGreater(resultado.confianca, .52)
        self.assertGreater(resultado.largura_normalizada, .20)

    def test_barra_transversal_nao_vira_trajetoria(self):
        resultado = estimar_trajetoria(_contorno([
            (20, 195), (428, 195), (428, 230), (20, 230),
        ]), (ALTURA, LARGURA))

        self.assertFalse(resultado.valida)


if __name__ == "__main__":
    unittest.main()
