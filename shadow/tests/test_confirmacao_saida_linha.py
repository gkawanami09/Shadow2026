"""Testes da confirmacao preta/prata feita pela camera do segue-linha."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.confirmacao_saida_linha import (  # noqa: E402
    INCONCLUSIVA,
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
    ConfirmadorFaixaSaidaLinha,
    faixa_centralizada,
    posicao_vertical_faixa,
)


def cena_preta():
    frame = np.full((252, 448, 3), 205, dtype=np.uint8)
    frame[:95, 200:248] = 35
    frame[95:165, :] = 35
    return frame


def cena_prata():
    frame = np.full((252, 448, 3), 205, dtype=np.uint8)
    frame[:80, 200:248] = 35
    yy, xx = np.indices((90, 448))
    textura = 85 + ((xx // 4 + yy // 3) % 2) * 75
    frame[80:170, :, 0] = textura
    frame[80:170, :, 1] = textura
    frame[80:170, :, 2] = textura
    return frame


class ClassificadorTests(unittest.TestCase):
    def test_faixa_preta_lisa_e_aceita(self):
        resultado = ClassificadorFaixaSaidaLinha().classificar(cena_preta())
        self.assertEqual(resultado.classificacao, PRETA)
        self.assertTrue(resultado.faixa_presente)

    def test_faixa_prata_reflexiva_e_rejeitada(self):
        resultado = ClassificadorFaixaSaidaLinha().classificar(cena_prata())
        self.assertEqual(resultado.classificacao, NAO_PRETA)
        self.assertTrue(resultado.faixa_presente)

    def test_piso_vazio_nao_decide(self):
        vazio = np.full((252, 448, 3), 205, dtype=np.uint8)
        resultado = ClassificadorFaixaSaidaLinha().classificar(vazio)
        self.assertEqual(resultado.classificacao, INCONCLUSIVA)
        self.assertFalse(resultado.faixa_presente)

    def test_preto_no_meio_da_tela_esta_centralizado(self):
        resultado = ClassificadorFaixaSaidaLinha().classificar(cena_preta())
        self.assertAlmostEqual(
            posicao_vertical_faixa(resultado), 0.52, delta=0.03)
        self.assertTrue(faixa_centralizada(resultado))

    def test_prata_alta_ainda_nao_esta_centralizada(self):
        resultado = ClassificadorFaixaSaidaLinha().classificar(cena_prata())
        self.assertLess(posicao_vertical_faixa(resultado), 0.40)
        self.assertFalse(faixa_centralizada(resultado))

    def test_as_quatro_fotos_reais_ficam_separadas(self):
        raiz = SHADOW_ROOT / "captures"
        pretas = sorted((raiz / "linha_preta").glob("*140*.png"))
        pratas = sorted((raiz / "linha_prata").glob("*140*.png"))
        if len(pretas) < 2 or len(pratas) < 2:
            self.skipTest("as quatro fotos reais nao estao neste checkout")

        classificador = ClassificadorFaixaSaidaLinha()
        for caminho in pretas:
            with self.subTest(caminho=caminho.name):
                frame = cv2.imread(str(caminho))
                self.assertEqual(
                    classificador.classificar(frame).classificacao,
                    PRETA,
                )
        for caminho in pratas:
            with self.subTest(caminho=caminho.name):
                frame = cv2.imread(str(caminho))
                self.assertEqual(
                    classificador.classificar(frame).classificacao,
                    NAO_PRETA,
                )


class ConfirmadorTests(unittest.TestCase):
    def test_um_frame_preto_nao_libera(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        decisao, _ = confirmador.update(
            cena_preta(), timestamp=1.0, now=1.0)
        self.assertIsNone(decisao)

    def test_preto_exige_quatro_frames_parados(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        decisao = None
        for indice in range(3):
            instante = 1.0 + indice * 0.03
            decisao, _ = confirmador.update(
                cena_preta(), timestamp=instante, now=instante)
        self.assertIsNone(decisao)

        instante = 1.09
        decisao, _ = confirmador.update(
            cena_preta(), timestamp=instante, now=instante)
        self.assertEqual(decisao, PRETA)

    def test_dois_frames_prata_rejeitam(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        decisao = None
        for indice in range(2):
            instante = 1.0 + indice * 0.03
            decisao, _ = confirmador.update(
                cena_prata(), timestamp=instante, now=instante)
        self.assertEqual(decisao, NAO_PRETA)

    def test_dois_cinzas_bloqueiam_mesmo_apos_tres_votos_pretos(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        instante = 1.0
        for _ in range(3):
            decisao, _ = confirmador.update(
                cena_preta(), timestamp=instante, now=instante)
            instante += 0.03
        self.assertIsNone(decisao)

        for _ in range(2):
            decisao, _ = confirmador.update(
                cena_prata(), timestamp=instante, now=instante)
            instante += 0.03
        self.assertEqual(decisao, NAO_PRETA)

    def test_timestamp_repetido_nao_da_tres_votos(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        for _ in range(5):
            decisao, _ = confirmador.update(
                cena_preta(), timestamp=1.0, now=1.0)
        self.assertIsNone(decisao)
        self.assertEqual(confirmador.votos_pretos, 1)


if __name__ == "__main__":
    unittest.main()
