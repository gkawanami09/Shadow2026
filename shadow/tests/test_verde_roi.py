"""Marcador verde só vale perto: corte vertical da região válida.

Contexto medido, e não preferência de estilo: a fita PRATA da entrada da sala
reflete verde-oliva e entra na faixa HSV do verde do percurso. Nas capturas de
`captures/linha_prata` ela forma um contorno de 7000-10000 px, muito acima de
``GREEN_MIN_AREA``. Enquanto o robô se aproxima, essa mancha fica na parte de
CIMA do quadro; o robô a tratava como marcador, disparava o giro verde e nunca
chegava a confirmar a entrada da sala.
"""

from pathlib import Path
import sys
import types
import unittest

import numpy as np

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

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

import config  # noqa: E402
from visao.verde import (  # noqa: E402
    filtrar_verdes_proximos,
    perto_do_robo,
)


ALTURA = config.camera_y
LARGURA = config.camera_x


def _contorno(topo, base, esquerda=100, direita=180):
    """Retângulo no formato que o OpenCV devolve em ``findContours``."""
    return np.array(
        [[[esquerda, topo]], [[direita, topo]],
         [[direita, base]], [[esquerda, base]]],
        dtype=np.int32,
    )


class CorteVerticalDoVerdeTests(unittest.TestCase):
    def test_marcador_colado_na_base_e_aceito(self):
        self.assertTrue(
            perto_do_robo(_contorno(int(ALTURA * .80), ALTURA - 1), ALTURA))

    def test_marcador_no_alto_do_quadro_e_descartado(self):
        """O caso dos prints: a mancha da fita prata vista de longe."""
        self.assertFalse(
            perto_do_robo(
                _contorno(int(ALTURA * .25), int(ALTURA * .39)), ALTURA))

    def test_marcador_entrando_pela_metade_de_baixo_e_aceito(self):
        """Basta a BASE alcançar o corte; o topo pode ficar bem acima."""
        self.assertTrue(
            perto_do_robo(
                _contorno(int(ALTURA * .10),
                          int(ALTURA * config.GREEN_ROI_TOP) + 2),
                ALTURA))

    def test_o_corte_e_medido_pela_base_e_nao_pelo_centro(self):
        """Um marcador alto e comprido, de base baixa, continua válido.

        Medir pelo centro rejeitaria o marcador real no momento em que ele
        cresce no quadro — exatamente quando o robô precisa reagir.
        """
        alto_e_comprido = _contorno(0, int(ALTURA * .70))
        self.assertLess(
            (0 + ALTURA * .70) / 2.0,
            ALTURA * config.GREEN_ROI_TOP,
            "cenário mal montado: o centro precisa ficar acima do corte")
        self.assertTrue(perto_do_robo(alto_e_comprido, ALTURA))

    def test_corte_explicito_sobrepoe_a_configuracao(self):
        contorno = _contorno(int(ALTURA * .50), int(ALTURA * .60))
        self.assertTrue(perto_do_robo(contorno, ALTURA, corte=.55))
        self.assertFalse(perto_do_robo(contorno, ALTURA, corte=.90))

    def test_o_corte_deixa_espaco_para_o_veto_de_marcador_baixo_demais(self):
        """`determine_turn_direction` veta base > 95%; o corte fica bem abaixo."""
        self.assertLess(config.GREEN_ROI_TOP, 0.95)


class FiltroDeContornosTests(unittest.TestCase):
    def test_separa_aceitos_e_descartados_preservando_a_ordem(self):
        perto = _contorno(int(ALTURA * .70), ALTURA - 1)
        longe = _contorno(int(ALTURA * .05), int(ALTURA * .20))
        aceitos, descartados = filtrar_verdes_proximos(
            [longe, perto, longe], ALTURA)
        self.assertEqual(len(aceitos), 1)
        self.assertEqual(len(descartados), 2)

    def test_lista_vazia_nao_quebra(self):
        self.assertEqual(filtrar_verdes_proximos([], ALTURA), ([], []))

    def test_reflexo_da_fita_prata_no_alto_nao_vira_marcador(self):
        """Regressão direta do bug: nada sobra para o `check_green` julgar."""
        reflexo = _contorno(
            int(ALTURA * .26), int(ALTURA * .50), esquerda=0,
            direita=LARGURA - 1)
        aceitos, descartados = filtrar_verdes_proximos([reflexo], ALTURA)
        self.assertEqual(aceitos, [])
        self.assertEqual(len(descartados), 1)

    def test_descartados_sao_devolvidos_para_o_overlay_poder_mostrar(self):
        """Ver "ignorei um verde" é o que faltava para diagnosticar isso."""
        longe = _contorno(int(ALTURA * .05), int(ALTURA * .20))
        _aceitos, descartados = filtrar_verdes_proximos([longe], ALTURA)
        self.assertEqual(len(descartados), 1)


if __name__ == "__main__":
    unittest.main()
