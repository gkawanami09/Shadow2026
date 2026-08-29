"""Regressoes para reflexos que formam uma parede na borda da imagem."""

import sys
from pathlib import Path
import types
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

try:
    import numba  # noqa: F401
except ImportError:
    def _njit(function=None, **_kwargs):
        return function if function is not None else lambda decorated: decorated
    sys.modules["numba"] = types.SimpleNamespace(njit=_njit)

from visao.linha import contorno_grudado_na_borda  # noqa: E402


def contorno(pontos):
    return np.asarray(pontos, dtype=np.int32).reshape((-1, 1, 2))


class LinhaBordaTests(unittest.TestCase):
    def test_parede_vertical_na_borda_direita_e_rejeitada(self):
        falso = contorno([(158, 8), (159, 8), (159, 86), (158, 86)])

        self.assertTrue(contorno_grudado_na_borda(falso, 160, 100))

    def test_faixa_transversal_que_so_toca_a_borda_e_preservada(self):
        faixa = contorno([(0, 45), (159, 45), (159, 55), (0, 55)])

        self.assertFalse(contorno_grudado_na_borda(faixa, 160, 100))


if __name__ == "__main__":
    unittest.main()
