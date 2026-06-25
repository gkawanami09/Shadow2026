"""Testes sinteticos da camada visual de verde acionavel."""

import unittest

import cv2
import numpy as np

from raspberry.backup.verde.green_action import analisar_intersecao_preta, decidir_verde_acionavel, dividir_zonas_intersecao


class GreenActionTests(unittest.TestCase):
    def mascara(self, partes):
        mascara = np.zeros((480, 640), dtype=np.uint8)
        zonas = dividir_zonas_intersecao(640, 480, 320)
        for parte in partes:
            x1, y1, x2, y2 = zonas[f"zona_{parte}"]
            cv2.rectangle(mascara, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
        return mascara

    def verde(self, lados=()):
        contornos = []
        for lado in lados:
            x = {"ESQUERDA": 200, "DIREITA": 400, "CENTRO": 300}[lado]
            contornos.append({"lado": lado, "area": 1200.0, "bbox": (x, 400, 60, 60), "confirmado": True})
        return {"contornos_confirmados": contornos}

    def decidir(self, partes, lados=()):
        analise = analisar_intersecao_preta(self.mascara(partes), 320)
        return decidir_verde_acionavel(self.verde(lados), analise)

    def test_verde_direita_lateral_direita(self):
        resultado = self.decidir(("centro", "direita"), ("DIREITA",))
        self.assertEqual(resultado["verde_acionavel"], "DIREITA")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_DIREITA")

    def test_verde_esquerda_lateral_esquerda(self):
        resultado = self.decidir(("centro", "esquerda"), ("ESQUERDA",))
        self.assertEqual(resultado["verde_acionavel"], "ESQUERDA")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_ESQUERDA")

    def test_dois_verdes_cruz_retorno(self):
        resultado = self.decidir(("centro", "esquerda", "direita"), ("ESQUERDA", "DIREITA"))
        self.assertEqual(resultado["verde_acionavel"], "DUPLO")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_RETORNO")

    def test_cruz_sem_verde(self):
        resultado = self.decidir(("centro", "esquerda", "direita"))
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")
        self.assertEqual(resultado["motivo_acao"], "intersecao_sem_verde")

    def test_laterais_sem_verde(self):
        for lado in ("esquerda", "direita"):
            with self.subTest(lado=lado):
                resultado = self.decidir(("centro", lado))
                self.assertEqual(resultado["verde_acionavel"], "NENHUM")
                self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")

    def test_reta_com_verde_confirmado_nao_aciona(self):
        resultado = self.decidir(("centro",), ("DIREITA",))
        self.assertEqual(resultado["tipo_intersecao"], "RETA")
        self.assertEqual(resultado["verde_acionavel"], "NENHUM")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")

    def test_reta_sem_verde_segue_reto(self):
        resultado = self.decidir(("centro",))
        self.assertEqual(resultado["tipo_intersecao"], "RETA")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")

    def test_linha_grossa_nao_vira_cruz(self):
        mascara = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mascara, (260, 384), (380, 470), 255, -1)
        analise = analisar_intersecao_preta(mascara, 320)
        self.assertNotEqual(analise["tipo_intersecao"], "CRUZ")

    def test_intersecao_ambigua_com_verde(self):
        resultado = self.decidir(("esquerda",), ("ESQUERDA",))
        self.assertEqual(resultado["tipo_intersecao"], "AMBIGUA")
        self.assertEqual(resultado["verde_acionavel"], "NENHUM")
        self.assertEqual(resultado["motivo_acao"], "intersecao_ambigua")

    def test_verde_acima_e_diagnostico_nao_bloqueante(self):
        analise = analisar_intersecao_preta(self.mascara(("centro", "direita")), 320)
        verde = self.verde(("DIREITA",))
        verde["contornos_confirmados"][0]["bbox"] = (400, 280, 60, 60)
        resultado = decidir_verde_acionavel(verde, analise)
        contorno = verde["contornos_confirmados"][0]
        self.assertTrue(contorno["possivel_verde_depois_intersecao"])
        self.assertEqual(resultado["acao_visual"], "PREPARAR_DIREITA")

    def test_verde_lado_incompativel(self):
        verde = self.verde(("DIREITA",))
        resultado = decidir_verde_acionavel(
            verde, analisar_intersecao_preta(self.mascara(("centro", "esquerda")), 320)
        )
        self.assertEqual(resultado["verde_acionavel"], "NENHUM")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")
        self.assertEqual(resultado["analise_intersecao"]["tipo_intersecao"], "LATERAL_ESQ")
        self.assertEqual(verde["contornos_confirmados"][0]["motivo_acionavel"], "verde_lado_incompativel")

    def test_centro_so_e_ambiguo_em_intersecao_confiavel(self):
        resultado = self.decidir(("centro", "esquerda", "direita"), ("CENTRO",))
        self.assertEqual(resultado["verde_acionavel"], "AMBIGUO")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")
        self.assertEqual(resultado["motivo_acao"], "verde_ambiguo")


if __name__ == "__main__":
    unittest.main()
