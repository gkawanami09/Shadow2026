"""Testes da confirmação temporal da faixa vermelha."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.vermelho import (ConfirmadorVermelho,  # noqa: E402
                            check_contour_size)


class ConfirmadorVermelhoTests(unittest.TestCase):
    def test_um_frame_isolado_nao_confirma(self):
        confirmador = ConfirmadorVermelho()

        resultados = [
            confirmador.atualizar(valor)
            for valor in (False, True, False)
        ]

        self.assertEqual(resultados, [False, False, False])

    def test_dois_de_tres_confirmam(self):
        confirmador = ConfirmadorVermelho()

        resultados = [
            confirmador.atualizar(valor)
            for valor in (True, False, True)
        ]

        self.assertEqual(resultados, [False, False, True])

    def test_dois_frames_seguidos_confirmam_no_segundo(self):
        confirmador = ConfirmadorVermelho()

        self.assertFalse(confirmador.atualizar(True))
        self.assertTrue(confirmador.atualizar(True))

    def test_confirmacao_sai_quando_os_votos_saem_da_janela(self):
        confirmador = ConfirmadorVermelho()

        resultados = [
            confirmador.atualizar(valor)
            for valor in (True, True, False, False)
        ]

        self.assertEqual(resultados, [False, True, True, False])

    def test_reiniciar_apaga_os_votos(self):
        confirmador = ConfirmadorVermelho()
        confirmador.atualizar(True)
        confirmador.reiniciar()

        self.assertFalse(confirmador.atualizar(True))

    def test_configuracao_invalida_e_rejeitada(self):
        for confirmacoes, tamanho in ((0, 3), (4, 3), (1, 0)):
            with self.subTest(confirmacoes=confirmacoes, tamanho=tamanho):
                with self.assertRaises(ValueError):
                    ConfirmadorVermelho(confirmacoes, tamanho)


class GeometriaFaixaVermelhaTests(unittest.TestCase):
    @staticmethod
    def _contornos(x1, y1, x2, y2):
        mascara = np.zeros((252, 448), dtype=np.uint8)
        cv2.rectangle(mascara, (x1, y1), (x2, y2), 255, -1)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contornos

    def test_faixa_horizontal_distante_e_candidata(self):
        contornos = self._contornos(80, 100, 368, 107)
        self.assertTrue(check_contour_size(
            contornos, "red", frame_shape=(252, 448)))

    def test_mancha_pequena_vermelha_nao_para_o_robo(self):
        contornos = self._contornos(190, 95, 230, 135)
        self.assertFalse(check_contour_size(
            contornos, "red", frame_shape=(252, 448)))

    def test_risco_vertical_vermelho_nao_e_faixa_final(self):
        contornos = self._contornos(215, 30, 223, 220)
        self.assertFalse(check_contour_size(
            contornos, "red", frame_shape=(252, 448)))


if __name__ == "__main__":
    unittest.main()
