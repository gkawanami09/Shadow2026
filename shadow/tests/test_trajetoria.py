"""Testes do ponto futuro usado pelo pure pursuit visual."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.trajetoria import (  # noqa: E402
    caminho_confiavel_ate_origem,
    extrair_ponto_futuro,
)


def linha(pontos, espessura=18):
    mascara = np.zeros((252, 448), dtype=np.uint8)
    cv2.polylines(
        mascara,
        [np.asarray(pontos, dtype=np.int32)],
        False,
        255,
        espessura,
    )
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contorno = max(contornos, key=cv2.contourArea)
    return mascara, contorno


class PontoFuturoTests(unittest.TestCase):
    def futuro(self, pontos):
        mascara, contorno = linha(pontos)
        return extrair_ponto_futuro(
            contorno,
            mascara_linha=mascara,
            origem_x=pontos[0][0],
        )

    def test_reta_central_mira_o_eixo_do_robo(self):
        futuro = self.futuro([(224, 251), (224, 5)])

        self.assertTrue(futuro.valido)
        self.assertAlmostEqual(futuro.x, 224., delta=3.)
        self.assertLess(futuro.y, 30.)

    def test_reta_deslocada_mira_o_corredor_deslocado(self):
        futuro = self.futuro([(300, 251), (300, 5)])

        self.assertTrue(futuro.valido)
        self.assertAlmostEqual(futuro.x, 300., delta=3.)

    def test_desvio_intermediario_que_volta_mira_o_final(self):
        futuro = self.futuro([
            (224, 251), (130, 200), (318, 150),
            (130, 100), (224, 5),
        ])

        self.assertTrue(futuro.valido)
        self.assertAlmostEqual(futuro.x, 224., delta=15.)

    def test_reflexos_no_zigzag_nao_encurtam_o_ponto_futuro(self):
        pontos = [
            (224, 251), (75, 200), (385, 150),
            (65, 90), (224, 5),
        ]
        mascara, contorno = linha(pontos, espessura=40)
        # Buracos claros como os observados no video continuam dentro de um
        # unico contorno, mas antes eram confundidos com ramos de um circulo.
        for x, y, raio in (
            (180, 184, 12),
            (285, 164, 14),
            (180, 112, 12),
        ):
            cv2.circle(mascara, (x, y), raio, 0, -1)

        futuro = extrair_ponto_futuro(
            contorno,
            mascara_linha=mascara,
            origem_x=224,
        )

        self.assertTrue(futuro.valido)
        self.assertLess(futuro.y, 30.)
        self.assertAlmostEqual(futuro.x, 224., delta=20.)

    def test_desvio_que_termina_a_direita_mira_a_direita(self):
        futuro = self.futuro([
            (224, 251), (130, 200), (318, 150),
            (180, 100), (360, 5),
        ])

        self.assertTrue(futuro.valido)
        self.assertGreater(futuro.x, 320.)

    def test_canto_em_l_mira_o_trecho_horizontal(self):
        futuro = self.futuro([
            (224, 251), (224, 125), (420, 125),
        ])

        self.assertTrue(futuro.valido)
        self.assertGreater(futuro.x, 290.)
        self.assertLess(futuro.y, 140.)

    def test_linha_curta_falha_fechada(self):
        futuro = self.futuro([(224, 251), (300, 220)])

        self.assertFalse(futuro.valido)

    def test_fragmentos_escuros_do_prata_nao_formam_trajetoria(self):
        mascara = np.zeros((252, 448), dtype=np.uint8)
        # Cada fragmento e largo, mas nenhum deles forma um caminho continuo
        # de baixo para cima que possa orientar o segue-linha.
        for y in (230, 185, 140, 95):
            cv2.rectangle(mascara, (120, y), (350, y + 16), 255, -1)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contorno in contornos:
            futuro = extrair_ponto_futuro(
                contorno, mascara_linha=mascara, origem_x=224)
            self.assertFalse(futuro.valido)

    def test_caminho_precisa_chegar_ate_a_base_do_robo(self):
        mascara = np.zeros((252, 448), dtype=np.uint8)
        # Mancha longa sobre o prata, mas iniciada longe do rodape.
        cv2.rectangle(mascara, (205, 45), (243, 190), 255, -1)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        anterior = config.LINE_PATH_REQUIRE_BASE_ANCHOR
        config.LINE_PATH_REQUIRE_BASE_ANCHOR = True
        try:
            self.assertFalse(caminho_confiavel_ate_origem(
                contornos[0], mascara_linha=mascara, origem_x=224))
        finally:
            config.LINE_PATH_REQUIRE_BASE_ANCHOR = anterior

    def test_caminho_central_continuo_e_confiavel(self):
        mascara, contorno = linha([(224, 251), (224, 5)])

        self.assertTrue(caminho_confiavel_ate_origem(
            contorno, mascara_linha=mascara, origem_x=224))

    def test_circulo_nao_e_reduzido_a_media_dos_dois_ramos(self):
        mascara = np.zeros((252, 448), dtype=np.uint8)
        cv2.line(mascara, (224, 251), (224, 220), 255, 18)
        cv2.ellipse(mascara, (224, 150), (75, 75), 0, 90, 450, 255, 18)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contorno = max(contornos, key=cv2.contourArea)

        futuro = extrair_ponto_futuro(
            contorno, mascara_linha=mascara, origem_x=224)

        self.assertTrue(futuro.valido)
        self.assertGreater(abs(futuro.x - 224.), 35.)


if __name__ == "__main__":
    unittest.main()
