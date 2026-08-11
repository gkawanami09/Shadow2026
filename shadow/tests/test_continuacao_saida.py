"""Geometrias da continuacao do percurso depois da area de resgate."""

import sys
from pathlib import Path
import types
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

# O Python de testes no Windows nao tem Numba. O algoritmo testado aqui nao
# depende do JIT; na Raspberry o modulo real continua sendo usado normalmente.
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
from visao import linha as linha_module  # noqa: E402
from visao.continuacao_saida import detectar_continuacao_saida  # noqa: E402
from visao.linha import calculate_angle, determine_correct_line  # noqa: E402


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

    def test_reta_diagonal_lateral_e_entregue_ao_segue_linha(self):
        cv2.line(
            self.mascara,
            (self.centro, config.camera_y - 1),
            (config.camera_x - 20, 110),
            255,
            18,
        )

        deteccao = detectar_continuacao_saida(self.mascara)

        self.assertIsNotNone(deteccao)
        self.assertGreater(deteccao.alvo_x, config.camera_x * .75)

    def test_alvo_da_saida_da_peso_ao_contorno_da_ramificacao(self):
        mascara = np.zeros_like(self.mascara)
        cv2.rectangle(
            mascara,
            (25, config.camera_y - 42),
            (config.camera_x - 25, config.camera_y - 10),
            255,
            -1,
        )
        cv2.line(
            mascara,
            (config.camera_x - 100, config.camera_y - 55),
            (config.camera_x - 55, 35),
            255,
            18,
        )
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        linha_module.init_tracker()

        selecionado, _ = determine_correct_line(
            contornos,
            turn_direction="right",
            alvo_saida=(config.camera_x - 55, 35),
        )

        x, y, largura, altura = cv2.boundingRect(selecionado)
        self.assertGreater(x, config.camera_x / 2)
        self.assertLess(y, config.camera_y * .25)
        self.assertGreater(altura, largura)

    def test_alvo_da_ramificacao_tem_peso_maior_no_angulo(self):
        contorno = np.array(
            [
                [[25, config.camera_y - 42]],
                [[config.camera_x - 25, config.camera_y - 42]],
                [[config.camera_x - 25, config.camera_y - 10]],
                [[25, config.camera_y - 10]],
            ],
            dtype=np.int32,
        )
        alvo = (config.camera_x - 55, 35)

        angulo, ponto, _ = calculate_angle(
            contorno,
            contorno,
            average_line_angle=0,
            turn_direction="right",
            last_bottom_point=config.camera_x / 2,
            average_line_point=config.camera_x / 2,
            alvo_saida=alvo,
        )

        self.assertEqual(tuple(int(valor) for valor in ponto), alvo)
        self.assertGreater(angulo, 80)


if __name__ == "__main__":
    unittest.main()
