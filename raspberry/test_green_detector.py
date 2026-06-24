"""Testes sinteticos do detector visual de verde."""

import unittest

import cv2
import numpy as np

from green_detector import detectar_verde


class GreenDetectorTests(unittest.TestCase):
    def criar_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def retangulo_verde(self, frame, canto_superior, canto_inferior):
        cv2.rectangle(frame, canto_superior, canto_inferior, (0, 255, 0), -1)

    def test_nenhum_verde(self):
        resultado = detectar_verde(self.criar_frame())
        self.assertEqual(resultado["tipo"], "NENHUM")
        self.assertEqual(resultado["qtd_contornos"], 0)

    def test_verde_a_esquerda(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (80, 200), (180, 300))
        resultado = detectar_verde(frame)
        self.assertEqual(resultado["tipo"], "ESQUERDA")
        self.assertGreater(resultado["area_esquerda"], 0)
        self.assertEqual(resultado["area_direita"], 0)

    def test_verde_a_direita(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (460, 200), (560, 300))
        resultado = detectar_verde(frame)
        self.assertEqual(resultado["tipo"], "DIREITA")
        self.assertGreater(resultado["area_direita"], 0)
        self.assertEqual(resultado["area_esquerda"], 0)

    def test_verde_duplo_balanceado(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (80, 200), (180, 300))
        self.retangulo_verde(frame, (460, 200), (560, 300))
        resultado = detectar_verde(frame)
        self.assertEqual(resultado["tipo"], "DUPLO")
        self.assertGreater(resultado["area_esquerda"], 0)
        self.assertGreater(resultado["area_direita"], 0)

    def test_ruido_pequeno_nao_e_verde(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (100, 220), (104, 224))
        self.retangulo_verde(frame, (150, 230), (154, 234))
        resultado = detectar_verde(frame)
        self.assertEqual(resultado["tipo"], "NENHUM")
        self.assertEqual(resultado["qtd_contornos"], 0)

    def test_verde_central_e_ambiguo(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (290, 200), (350, 300))
        resultado = detectar_verde(frame)
        self.assertEqual(resultado["tipo"], "AMBIGUO")
        self.assertGreater(resultado["area_centro"], 0)

    def test_x_referencia_deslocado_e_respeitado(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (340, 200), (400, 300))
        resultado = detectar_verde(frame, x_referencia=500)
        self.assertEqual(resultado["tipo"], "ESQUERDA")
        self.assertGreater(resultado["area_esquerda"], 0)


if __name__ == "__main__":
    unittest.main()
