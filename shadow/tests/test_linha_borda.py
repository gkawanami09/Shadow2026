"""Regressoes para reflexos do prata nas extremidades da camera de linha."""

import sys
from pathlib import Path
import types
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

# O filtro testado e' OpenCV puro. Em maquinas de desenvolvimento sem Numba,
# permita importar o restante de ``visao.linha`` sem compilar o rastreador.
try:
    import numba  # noqa: F401
except ImportError:
    def _njit(function=None, **_kwargs):
        return function if function is not None else lambda decorated: decorated
    sys.modules["numba"] = types.SimpleNamespace(njit=_njit)

from visao.linha import (  # noqa: E402
    mascarar_extremidades_com_linha_central,
    remover_componentes_isolados_da_borda,
)


class LinhaBordaTests(unittest.TestCase):
    def test_ilhas_nas_duas_extremidades_sao_ignoradas(self):
        mascara = np.zeros((100, 160), dtype=np.uint8)
        # Linha valida no corredor central.
        mascara[20:100, 72:88] = 255
        # Reflexos vistos apenas nos dois cantos do prata.
        mascara[55:95, :12] = 255
        mascara[50:95, 148:] = 255

        filtrada = remover_componentes_isolados_da_borda(mascara)

        self.assertTrue(np.all(filtrada[55:95, :12] == 0))
        self.assertTrue(np.all(filtrada[50:95, 148:] == 0))
        self.assertTrue(np.all(filtrada[20:100, 72:88] == 255))

    def test_curva_que_alcanca_a_borda_continua_valida(self):
        mascara = np.zeros((100, 160), dtype=np.uint8)
        # O ramo toca a borda, mas chega ate o corredor central: e' uma
        # trajetoria possivel, nao uma ilha de reflexo.
        mascara[45:65, :90] = 255

        filtrada = remover_componentes_isolados_da_borda(mascara)

        self.assertTrue(np.all(filtrada[45:65, :90] == 255))

    def test_escuro_lateral_nao_puxa_com_linha_central_visivel(self):
        mascara = np.zeros((100, 160), dtype=np.uint8)
        mascara[55:100, 72:88] = 255
        # Mancha escura ligada artificialmente a linha e a borda direita.
        mascara[70:100, 80:160] = 255

        filtrada = mascarar_extremidades_com_linha_central(mascara)

        self.assertTrue(np.all(filtrada[:, 120:] == 0))
        self.assertTrue(np.all(filtrada[55:100, 72:80] == 255))

    def test_borda_volta_a_valer_depois_que_linha_sai_do_centro(self):
        mascara = np.zeros((100, 160), dtype=np.uint8)
        mascara[55:100, 125:150] = 255

        filtrada = mascarar_extremidades_com_linha_central(mascara)

        self.assertTrue(np.all(filtrada[55:100, 125:150] == 255))


if __name__ == "__main__":
    unittest.main()
