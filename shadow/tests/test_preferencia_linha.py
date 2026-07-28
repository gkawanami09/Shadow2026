"""Testes do peso visual para retomar a linha pela esquerda."""

import sys
from pathlib import Path
import types
import unittest

import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

# O ambiente de testes no Windows não instala Numba. A função decorada é
# Python puro e pode ser validada sem JIT; na Raspberry o pacote real continua
# sendo importado normalmente.
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    numba_falso = types.ModuleType("numba")

    def njit_falso(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda funcao: funcao

    numba_falso.njit = njit_falso
    sys.modules["numba"] = numba_falso

from shared.dados_compartilhados import timer  # noqa: E402
from visao.linha import calculate_angle, init_tracker  # noqa: E402


class PreferenciaLinhaTests(unittest.TestCase):
    def setUp(self):
        init_tracker()
        for nome in (
            "multiple_bottom",
            "multiple_side_l",
            "multiple_side_r",
        ):
            timer.set_timer(nome, 0)

    def test_linha_transversal_recebe_peso_para_esquerda(self):
        contorno = np.array(
            [[[0, 110]], [[447, 110]], [[447, 150]], [[0, 150]]],
            dtype=np.int32,
        )
        recorte_vazio = np.empty((0, 1, 2), dtype=np.int32)

        angulo_normal, ponto_normal, _ = calculate_angle(
            contorno,
            recorte_vazio,
            average_line_angle=30,
            turn_direction="straight",
            last_bottom_point=224,
            average_line_point=224,
            preferir_esquerda=False,
        )
        angulo_esquerda, ponto_esquerda, _ = calculate_angle(
            contorno,
            recorte_vazio,
            average_line_angle=30,
            turn_direction="straight",
            last_bottom_point=224,
            average_line_point=224,
            preferir_esquerda=True,
        )

        self.assertGreater(angulo_normal, 0)
        self.assertGreater(ponto_normal[0], 224)
        self.assertLess(angulo_esquerda, 0)
        self.assertLess(ponto_esquerda[0], 224)

    def test_preferencia_nao_desvia_linha_que_continua_em_frente(self):
        contorno = np.array(
            [[[200, 0]], [[248, 0]], [[248, 251]], [[200, 251]]],
            dtype=np.int32,
        )
        recorte = contorno[contorno[:, 0, 1] > 151].reshape(-1, 1, 2)

        angulo, ponto, _ = calculate_angle(
            contorno,
            recorte,
            average_line_angle=0,
            turn_direction="straight",
            last_bottom_point=224,
            average_line_point=224,
            preferir_esquerda=True,
        )

        self.assertAlmostEqual(ponto[0], 224, delta=1)
        self.assertAlmostEqual(angulo, 0, delta=1)

    def test_linha_transversal_torta_tambem_recebe_peso_esquerdo(self):
        contorno = np.array(
            [[[80, 110]], [[380, 110]], [[360, 150]], [[100, 150]]],
            dtype=np.int32,
        )
        recorte_vazio = np.empty((0, 1, 2), dtype=np.int32)

        angulo_normal, _, _ = calculate_angle(
            contorno,
            recorte_vazio,
            average_line_angle=0,
            turn_direction="straight",
            last_bottom_point=224,
            average_line_point=224,
            preferir_esquerda=False,
        )
        angulo_esquerda, ponto_esquerda, _ = calculate_angle(
            contorno,
            recorte_vazio,
            average_line_angle=0,
            turn_direction="straight",
            last_bottom_point=224,
            average_line_point=224,
            preferir_esquerda=True,
        )

        self.assertAlmostEqual(angulo_normal, 0, delta=10)
        self.assertLess(angulo_esquerda, -40)
        self.assertLess(ponto_esquerda[0], 224)


if __name__ == "__main__":
    unittest.main()
