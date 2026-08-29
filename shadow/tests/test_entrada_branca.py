"""Contrato visual da entrada: preto lateral, branco central e sem linha."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.entrada_branca import detectar_entrada_branca  # noqa: E402


class EntradaBrancaTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((252, 448), dtype=np.uint8)
        self.mask[106:237, 8:143] = 255
        self.mask[106:237, 305:439] = 255

    def test_preto_dos_lados_centro_branco_sem_linha_confirma(self):
        resultado = detectar_entrada_branca(
            self.mask, linha_a_frente=False)

        self.assertTrue(resultado.candidata)
        self.assertEqual(resultado.preto_centro, 0.)

    def test_continuacao_da_linha_bloqueia(self):
        resultado = detectar_entrada_branca(
            self.mask, linha_a_frente=True)

        self.assertFalse(resultado.candidata)

    def test_preto_no_meio_bloqueia(self):
        self.mask[106:237, 170:278] = 255

        self.assertFalse(
            detectar_entrada_branca(self.mask).candidata)

    def test_apenas_um_lado_preto_bloqueia(self):
        self.mask[:, 305:439] = 0

        self.assertFalse(
            detectar_entrada_branca(self.mask).candidata)


if __name__ == "__main__":
    unittest.main()
