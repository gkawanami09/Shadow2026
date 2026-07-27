"""Testes do overlay do resgate.

Overlay é diagnóstico, mas diagnóstico errado custa caro: um verde desenhado
em vermelho faz a equipe recalibrar a cor errada em campo. Por isso as cores
são lidas dos pixels realmente desenhados, não das constantes.
"""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from visao import overlay_resgate  # noqa: E402
from visao.deteccao import VictimDetection  # noqa: E402


FORMA = (480, 640, 3)


class MarcadorFalso:
    def __init__(self, bbox):
        self.bbox = bbox
        self.confidence = 0.80
        self.hits = 3
        self.track_locked = True
        self.confirmed = True


def _quadro():
    return np.zeros(FORMA, dtype=np.uint8)


class CoresDoMarcadorTests(unittest.TestCase):
    def test_verde_sai_verde_e_vermelho_sai_vermelho(self):
        tela = overlay_resgate.desenhar_marcadores(
            _quadro(),
            {
                "green": MarcadorFalso((40, 300, 120, 90)),
                "red": MarcadorFalso((420, 300, 120, 90)),
            },
        )
        verde = tela[300, 40:160]
        vermelho = tela[300, 420:540]
        pix_verde = verde[verde.any(axis=1)]
        pix_vermelho = vermelho[vermelho.any(axis=1)]

        self.assertTrue(len(pix_verde) > 0)
        self.assertTrue(len(pix_vermelho) > 0)
        # BGR: verde puro tem canal 1 saturado e canal 2 zerado.
        self.assertTrue(np.all(pix_verde[:, 1] == 255))
        self.assertTrue(np.all(pix_verde[:, 2] == 0))
        self.assertTrue(np.all(pix_vermelho[:, 2] == 255))
        self.assertTrue(np.all(pix_vermelho[:, 1] == 0))

    def test_constantes_do_overlay(self):
        self.assertEqual(cfg.FINAL_TRIANGLE_OVERLAY_BGR["green"], (0, 255, 0))
        self.assertEqual(cfg.FINAL_TRIANGLE_OVERLAY_BGR["red"], (0, 0, 255))

    def test_sem_marcador_nao_desenha_nada(self):
        tela = overlay_resgate.desenhar_marcadores(
            _quadro(), {"green": None, "red": None})
        self.assertEqual(int(tela.sum()), 0)


class VitimaTests(unittest.TestCase):
    @staticmethod
    def _vitima(kind="silver", truncated=False):
        return VictimDetection(
            kind, 320.0, 400.0, 40.0, 0.9, True, 3, 0.0,
            track_locked=True, truncated=truncated)

    def test_prata_e_preta_usam_cores_distintas(self):
        prata = overlay_resgate.desenhar_vitima(
            _quadro(), self._vitima("silver"))
        preta = overlay_resgate.desenhar_vitima(
            _quadro(), self._vitima("black"))
        self.assertFalse(np.array_equal(prata, preta))

    def test_vitima_cortada_e_sinalizada_no_preview(self):
        """A equipe precisa ver por que a coleta não disparou."""
        normal = overlay_resgate.desenhar_vitima(
            _quadro(), self._vitima(truncated=False))
        cortada = overlay_resgate.desenhar_vitima(
            _quadro(), self._vitima(truncated=True))
        # A versão cortada desenha o aviso extra, logo tem mais pixels.
        self.assertGreater(int(cortada.sum()), int(normal.sum()))

    def test_sem_vitima_nao_desenha(self):
        tela = overlay_resgate.desenhar_vitima(_quadro(), None)
        self.assertEqual(int(tela.sum()), 0)


class MontagemTests(unittest.TestCase):
    def test_anotar_nao_altera_o_frame_original(self):
        original = _quadro()
        copia = original.copy()
        overlay_resgate.anotar(
            original, estado="TESTE", detalhe="nao deve alterar")
        self.assertTrue(np.array_equal(original, copia))

    def test_modo_sem_motores_e_anunciado(self):
        tela = overlay_resgate.anotar(
            _quadro(), estado="WAIT", motores_ativos=False)
        self.assertGreater(int(tela.sum()), 0)


if __name__ == "__main__":
    unittest.main()
