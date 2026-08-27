"""Testes da geometria de zigue-zague vista pela camera wide."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.zigzag import detectar_zigzag  # noqa: E402


def contorno_de_linha(pontos, espessura=18):
    imagem = np.zeros((252, 448), dtype=np.uint8)
    cv2.polylines(
        imagem,
        [np.asarray(pontos, dtype=np.int32)],
        False,
        255,
        espessura,
    )
    contornos, _ = cv2.findContours(
        imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contornos, key=cv2.contourArea)


class DetectorZigzagTests(unittest.TestCase):
    def test_reconhece_alternancias_que_retornam_ao_corredor(self):
        contorno = contorno_de_linha([
            (224, 251), (130, 200), (318, 150),
            (130, 100), (224, 5),
        ])

        self.assertTrue(detectar_zigzag(contorno))

    def test_reconhecimento_e_espelhado(self):
        contorno = contorno_de_linha([
            (224, 251), (318, 200), (130, 150),
            (318, 100), (224, 5),
        ])

        self.assertTrue(detectar_zigzag(contorno))

    def test_reta_curva_e_canto_nao_sao_zigzag(self):
        geometrias = (
            [(224, 251), (224, 5)],
            [(224, 251), (230, 210), (260, 170),
             (310, 120), (390, 40)],
            [(224, 251), (224, 125), (420, 125)],
        )

        for pontos in geometrias:
            with self.subTest(pontos=pontos):
                self.assertFalse(detectar_zigzag(
                    contorno_de_linha(pontos)))

    def test_degraus_horizontais_nao_sao_confundidos_com_zigzag(self):
        contorno = contorno_de_linha([
            (224, 251), (224, 210), (130, 210), (130, 160),
            (310, 160), (310, 110), (130, 110), (130, 60),
            (224, 60), (224, 10),
        ])

        self.assertFalse(detectar_zigzag(contorno))

    def test_alternancia_que_termina_em_outro_corredor_nao_corta_caminho(self):
        contorno = contorno_de_linha([
            (90, 251), (190, 200), (40, 150),
            (210, 100), (390, 5),
        ])

        self.assertFalse(detectar_zigzag(contorno))

    def test_desenho_curto_nao_e_confirmado(self):
        contorno = contorno_de_linha([
            (224, 251), (150, 230), (300, 210),
            (150, 190), (224, 170),
        ])

        self.assertFalse(detectar_zigzag(contorno))


if __name__ == "__main__":
    unittest.main()
