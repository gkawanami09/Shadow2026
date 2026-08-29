"""Contrato do score fotometrico da faixa prata na camera de linha."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.faixa_prata_entrada import detectar_faixa_prata  # noqa: E402
from visao.entrada_missao import ScoreEntryPipeline  # noqa: E402


SHAPE = (252, 448)


def mascara_linha_ate(y):
    mascara = np.zeros(SHAPE, dtype=np.uint8)
    mascara[y:, 210:238] = 255
    return mascara


def faixa_prata_reflexiva():
    frame = np.full((*SHAPE, 3), 170, dtype=np.uint8)
    y0, y1 = 145, 175
    frame[y0:y1] = 175
    # Metal com highlights e sombras vizinhos: ambos neutros em HSV.
    for x in range(0, SHAPE[1], 32):
        frame[y0:y1, x:x + 8] = 255
        frame[y0:y1, x + 12:x + 20] = 45
    return frame, (y0, y1)


class FaixaPrataEntradaTests(unittest.TestCase):
    def test_piso_branco_uniforme_nao_e_prata(self):
        frame = np.full((*SHAPE, 3), 240, dtype=np.uint8)
        resultado = detectar_faixa_prata(
            frame, mascara_linha_ate(180), line_aligned=True)

        self.assertFalse(resultado.candidata)
        self.assertLess(resultado.score, 9)

    def test_reflexo_pequeno_isolado_nao_e_prata(self):
        frame = np.full((*SHAPE, 3), 170, dtype=np.uint8)
        frame[155:170, 205:240] = 255
        resultado = detectar_faixa_prata(
            frame, mascara_linha_ate(180), line_aligned=True)

        self.assertFalse(resultado.candidata)
        self.assertLess(resultado.largura_ratio, .62)

    def test_sombra_ampla_sem_highlight_nao_e_prata(self):
        frame = np.full((*SHAPE, 3), 80, dtype=np.uint8)
        resultado = detectar_faixa_prata(
            frame, mascara_linha_ate(180), line_aligned=True)

        self.assertFalse(resultado.candidata)

    def test_faixa_reflexiva_larga_com_linha_terminando_confirma_candidata(self):
        frame, (y0, y1) = faixa_prata_reflexiva()
        resultado = detectar_faixa_prata(
            frame, mascara_linha_ate(y1), line_aligned=True)

        self.assertTrue(resultado.linha_fim)
        self.assertGreaterEqual(resultado.largura_ratio, .62)
        self.assertGreaterEqual(resultado.score, 9)
        self.assertTrue(resultado.candidata)

    def test_linha_continuando_depois_da_faixa_rejeita_entrada(self):
        frame, (y0, y1) = faixa_prata_reflexiva()
        mascara = mascara_linha_ate(y1)
        mascara[:y0 - 4, 210:238] = 255
        resultado = detectar_faixa_prata(
            frame, mascara, line_aligned=True)

        self.assertFalse(resultado.linha_fim)
        self.assertFalse(resultado.candidata)

    def test_confirma_somente_apos_quatro_frames_fortes(self):
        frame, (_y0, y1) = faixa_prata_reflexiva()
        pipeline = ScoreEntryPipeline()
        pipeline.set_armed(True)
        mascara = mascara_linha_ate(y1)

        confirmed = False
        for index in range(4):
            pipeline.submit(
                frame, index / 40., True, black_mask=mascara)
            confirmed, _detection = pipeline.poll()
            if index < 3:
                self.assertFalse(confirmed)

        self.assertTrue(confirmed)
        self.assertEqual(pipeline.votes, 4)


if __name__ == "__main__":
    unittest.main()
