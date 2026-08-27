"""Regressoes da geometria obrigatoria dos marcadores verdes."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.verde import ConfirmadorVerde, check_green  # noqa: E402


def _quadrado(x=160, y=130, lado=60):
    return np.array(
        [[[x, y]], [[x + lado, y]], [[x + lado, y + lado]],
         [[x, y + lado]]],
        dtype=np.int32,
    )


def _mascara_ao_redor(*, topo=True, esquerda=False, direita=False,
                       baixo=False, x=160, y=130, lado=60):
    mascara = np.zeros(
        (config.camera_y, config.camera_x), dtype=np.uint8)
    if topo:
        cv2.rectangle(mascara, (x + 3, y - 14),
                      (x + lado - 3, y - 1), 255, -1)
    if baixo:
        cv2.rectangle(mascara, (x + 3, y + lado + 1),
                      (x + lado - 3, y + lado + 14), 255, -1)
    if esquerda:
        cv2.rectangle(mascara, (x - 14, y + 3),
                      (x - 1, y + lado - 3), 255, -1)
    if direita:
        cv2.rectangle(mascara, (x + lado + 1, y + 3),
                      (x + lado + 14, y + lado - 3), 255, -1)
    return mascara


class ValidacaoVerdeTests(unittest.TestCase):
    def test_verde_para_esquerda_exige_preto_acima_e_a_direita(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(topo=True, direita=True))
        self.assertEqual(direcao, "left")

    def test_verde_para_direita_exige_preto_acima_e_a_esquerda(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(topo=True, esquerda=True))
        self.assertEqual(direcao, "right")

    def test_sem_preto_acima_nao_autoriza_curva(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=False, direita=True))
        self.assertEqual(direcao, "straight")

    def test_preto_dos_dois_lados_e_ambiguo(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=True, esquerda=True, direita=True))
        self.assertEqual(direcao, "straight")

    def test_preto_em_baixo_tambem_invalida_o_marcador(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=True, direita=True, baixo=True))
        self.assertEqual(direcao, "straight")

    def test_ruido_preto_descontinuo_nao_passa(self):
        mascara = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8)
        for x in range(163, 218, 8):
            cv2.rectangle(mascara, (x, 118), (x + 2, 128), 255, -1)
        for y in range(133, 188, 8):
            cv2.rectangle(mascara, (222, y), (232, y + 2), 255, -1)
        self.assertEqual(check_green([_quadrado()], mascara), "straight")

    def test_mancha_verde_alongada_nao_e_quadrado(self):
        mancha = np.array(
            [[[120, 130]], [[240, 130]], [[240, 160]], [[120, 160]]],
            dtype=np.int32,
        )
        self.assertEqual(
            check_green(
                [mancha], _mascara_ao_redor(topo=True, direita=True)),
            "straight",
        )

    def test_direcao_precisa_de_tres_quadros_consecutivos(self):
        confirmador = ConfirmadorVerde(frames=3)
        self.assertEqual(confirmador.atualizar("left"), "straight")
        self.assertEqual(confirmador.atualizar("left"), "straight")
        self.assertEqual(confirmador.atualizar("left"), "left")

    def test_quadro_invalido_zerar_confirmacao(self):
        confirmador = ConfirmadorVerde(frames=3)
        confirmador.atualizar("right")
        confirmador.atualizar("right")
        self.assertEqual(confirmador.atualizar("straight"), "straight")
        self.assertEqual(confirmador.atualizar("right"), "straight")


if __name__ == "__main__":
    unittest.main()
