"""Testes sinteticos do detector de verde com confirmacao pela linha."""

import unittest

import cv2
import numpy as np

from raspberry.backup.verde.green_detector import detectar_verde


class GreenDetectorTests(unittest.TestCase):
    def criar_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def retangulo_verde(self, frame, canto_superior, canto_inferior):
        cv2.rectangle(frame, canto_superior, canto_inferior, (0, 255, 0), -1)

    def mascara_linha_vertical(self, x=320, largura_linha=60):
        mascara = np.zeros((480, 640), dtype=np.uint8)
        metade = largura_linha // 2
        cv2.rectangle(mascara, (x - metade, 0), (x + metade, 479), 255, -1)
        return mascara

    def detectar_confirmado(self, frame, x_referencia=None, mascara_linha=None):
        if mascara_linha is None:
            mascara_linha = self.mascara_linha_vertical()
        return detectar_verde(frame, x_referencia, mascara_linha)

    def test_nenhum_verde(self):
        resultado = self.detectar_confirmado(self.criar_frame())
        self.assertEqual(resultado["tipo"], "NENHUM")
        self.assertEqual(resultado["qtd_contornos"], 0)

    def test_verde_a_esquerda_confirmado(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (180, 300), (240, 400))
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo"], "ESQUERDA")
        self.assertTrue(resultado["confirmado"])
        self.assertGreater(resultado["area_esquerda"], 0)

    def test_verde_a_direita_confirmado(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (400, 300), (460, 400))
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo"], "DIREITA")
        self.assertTrue(resultado["confirmado"])
        self.assertGreater(resultado["area_direita"], 0)

    def test_verde_duplo_confirmado(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (180, 300), (240, 400))
        self.retangulo_verde(frame, (400, 300), (460, 400))
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo"], "DUPLO")
        self.assertEqual(resultado["qtd_contornos_confirmados"], 2)

    def test_ruido_pequeno_nao_e_verde(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (100, 300), (104, 304))
        self.retangulo_verde(frame, (150, 310), (154, 314))
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo_detectado"], "NENHUM")
        self.assertEqual(resultado["tipo"], "NENHUM")

    def test_verde_central_confirmado_e_ambiguo(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (290, 300), (350, 400))
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo"], "AMBIGUO")
        self.assertGreater(resultado["area_centro"], 0)

    def test_x_referencia_deslocado_e_respeitado(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (340, 300), (400, 400))
        resultado = self.detectar_confirmado(
            frame, x_referencia=500, mascara_linha=self.mascara_linha_vertical(x=400)
        )
        self.assertEqual(resultado["tipo"], "ESQUERDA")

    def test_verde_claro_fraco_e_rejeitado(self):
        frame = self.criar_frame()
        cv2.rectangle(frame, (180, 300), (240, 400), (180, 210, 180), -1)
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo_detectado"], "NENHUM")

    def test_verde_saturado_expoe_metricas(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (180, 300), (240, 400))
        resultado = self.detectar_confirmado(frame)
        contorno = resultado["contornos"][0]
        for chave in (
            "mean_h", "mean_s", "mean_v", "mean_b", "mean_g", "mean_r",
            "g_minus_r", "g_minus_b", "green_ratio", "aspect_ratio", "confirmado",
            "motivo_confirmacao", "black_near_pixels", "area_in_confirm_zone_ratio",
        ):
            self.assertIn(chave, contorno)

    def test_verde_direita_ignora_falso_verde_esquerda(self):
        frame = self.criar_frame()
        cv2.rectangle(frame, (180, 300), (240, 400), (180, 210, 180), -1)
        self.retangulo_verde(frame, (400, 300), (460, 400))
        self.assertEqual(self.detectar_confirmado(frame)["tipo"], "DIREITA")

    def test_verde_esquerda_ignora_falso_verde_direita(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (180, 300), (240, 400))
        cv2.rectangle(frame, (400, 300), (460, 400), (180, 210, 180), -1)
        self.assertEqual(self.detectar_confirmado(frame)["tipo"], "ESQUERDA")

    def test_verde_sem_linha_proxima_nao_confirma(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (400, 300), (460, 400))
        resultado = self.detectar_confirmado(
            frame, mascara_linha=np.zeros((480, 640), dtype=np.uint8)
        )
        self.assertEqual(resultado["tipo_detectado"], "DIREITA")
        self.assertEqual(resultado["tipo"], "NENHUM")
        self.assertFalse(resultado["confirmado"])
        self.assertEqual(resultado["observacao"], "sem_linha_preta_proxima")

    def test_verde_fora_da_zona_baixa_nao_confirma(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (400, 140), (460, 220))
        resultado = self.detectar_confirmado(frame)
        self.assertEqual(resultado["tipo_detectado"], "DIREITA")
        self.assertEqual(resultado["tipo"], "NENHUM")
        self.assertIn("fora_zona_confirmacao", resultado["observacao"])

    def test_chamada_sem_mascara_preserva_detectado_sem_confirmar(self):
        frame = self.criar_frame()
        self.retangulo_verde(frame, (400, 300), (460, 400))
        resultado = detectar_verde(frame)
        self.assertEqual(resultado["tipo_detectado"], "DIREITA")
        self.assertEqual(resultado["tipo_confirmado"], "NENHUM")
        self.assertEqual(resultado["tipo"], "NENHUM")
        self.assertFalse(resultado["confirmado"])
        self.assertEqual(resultado["observacao"], "sem_mascara_linha")
        self.assertEqual(resultado["qtd_contornos_confirmados"], 0)


if __name__ == "__main__":
    unittest.main()
