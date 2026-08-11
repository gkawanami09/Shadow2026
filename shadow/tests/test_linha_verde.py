"""Prioridade do ramo indicado por um marcador verde confirmado."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao import linha as linha_module  # noqa: E402
from visao.linha import determine_correct_line  # noqa: E402


def _contorno_vertical(x_centro):
    """Ramo largo o suficiente para tocar a base da imagem."""
    y_topo = int(config.camera_y * .45)
    return np.array(
        [
            [[x_centro - 12, y_topo]],
            [[x_centro + 12, y_topo]],
            [[x_centro + 12, config.camera_y - 1]],
            [[x_centro - 12, config.camera_y - 1]],
        ],
        dtype=np.int32,
    )


class LinhaVerdeTests(unittest.TestCase):
    def setUp(self):
        linha_module.init_tracker()
        linha_module.x_last = config.camera_x / 2
        linha_module.y_last = config.camera_y / 2

    def test_verde_confirmado_supera_historico_da_correcao_no_mesmo_frame(self):
        """Em uma bifurcacao, verde à direita não pode manter o ramo esquerdo."""
        ramo_esquerdo = _contorno_vertical(int(config.camera_x * .35))
        ramo_direito = _contorno_vertical(int(config.camera_x * .76))

        selecionado, _ = determine_correct_line(
            [ramo_esquerdo, ramo_direito], turn_direction="right")

        x, _, largura, _ = cv2.boundingRect(selecionado)
        self.assertGreater(x + largura / 2, config.camera_x / 2)


if __name__ == "__main__":
    unittest.main()
