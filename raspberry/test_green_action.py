"""Testes sinteticos da decisao de verde por topologia da intersecao."""

import unittest

import cv2
import numpy as np

from green_action import (
    analisar_intersecao_preta,
    decidir_verde_acionavel,
    dividir_zonas_intersecao,
)


class GreenActionTests(unittest.TestCase):
    def analise(self, tipo="CRUZ", node_y=320, confiavel=True):
        return {
            "intersecao_detectada": tipo in {"CRUZ", "LATERAL_ESQ", "LATERAL_DIR"},
            "tipo_intersecao": tipo,
            "node_y_near": node_y if confiavel else None,
            "node_y_far": node_y - 8 if confiavel else None,
            "node_y_center": node_y - 4 if confiavel else None,
            "node_confiavel": confiavel,
            "node_rows_count": 5 if confiavel else 0,
            "zonas": {"zona_intersecao": (0, 280, 640, 400)},
        }

    def verde(self, lados=("ESQUERDA",), y=330, h=60):
        contornos = []
        for lado in lados:
            x = {"ESQUERDA": 200, "DIREITA": 400, "CENTRO": 300}[lado]
            contornos.append({"lado": lado, "area": 1200.0, "bbox": (x, y, 60, h), "confirmado": True})
        return {"contornos_confirmados": contornos}

    def test_estima_no_por_ramos_laterais(self):
        mascara = np.zeros((480, 640), dtype=np.uint8)
        zonas = dividir_zonas_intersecao(640, 480, 320)
        x1, y1, x2, _ = zonas["zona_esquerda"]
        cv2.rectangle(mascara, (x1, y1 + 10), (x2 - 1, y1 + 14), 255, -1)
        analise = analisar_intersecao_preta(mascara, 320)
        self.assertTrue(analise["node_confiavel"])
        self.assertEqual(analise["node_rows_count"], 5)
        self.assertEqual(analise["node_y_near"], y1 + 14)

    def test_verde_esquerda_antes_aciona(self):
        verde = self.verde(("ESQUERDA",), y=330)
        resultado = decidir_verde_acionavel(verde, self.analise("LATERAL_ESQ"))
        self.assertEqual(verde["contornos_confirmados"][0]["verde_posicao_intersecao"], "ANTES_INTERSECAO")
        self.assertEqual(resultado["verde_acionavel"], "ESQUERDA")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_ESQUERDA")

    def test_verde_direita_antes_aciona(self):
        resultado = decidir_verde_acionavel(self.verde(("DIREITA",), y=330), self.analise("LATERAL_DIR"))
        self.assertEqual(resultado["verde_acionavel"], "DIREITA")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_DIREITA")

    def test_verde_depois_e_ignorado(self):
        verde = self.verde(("ESQUERDA",), y=240, h=50)
        resultado = decidir_verde_acionavel(verde, self.analise())
        contorno = verde["contornos_confirmados"][0]
        self.assertEqual(contorno["verde_posicao_intersecao"], "DEPOIS_INTERSECAO")
        self.assertFalse(contorno["acionavel"])
        self.assertEqual(contorno["motivo_acionavel"], "verde_depois_intersecao")
        self.assertEqual(resultado["motivo_acao"], "verde_depois_intersecao")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")

    def test_verde_perto_do_no_e_ambiguo(self):
        verde = self.verde(("ESQUERDA",), y=280, h=70)
        resultado = decidir_verde_acionavel(verde, self.analise())
        self.assertEqual(verde["contornos_confirmados"][0]["verde_posicao_intersecao"], "AMBIGUO")
        self.assertEqual(resultado["motivo_acao"], "verde_posicao_ambigua")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")

    def test_dois_verdes_antes_geram_retorno(self):
        resultado = decidir_verde_acionavel(self.verde(("ESQUERDA", "DIREITA"), y=330), self.analise())
        self.assertEqual(resultado["verde_acionavel"], "DUPLO")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_RETORNO")

    def test_dois_verdes_depois_nao_geram_retorno(self):
        resultado = decidir_verde_acionavel(self.verde(("ESQUERDA", "DIREITA"), y=240, h=50), self.analise())
        self.assertEqual(resultado["verde_acionavel"], "NENHUM")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")
        self.assertEqual(resultado["motivo_acao"], "verde_depois_intersecao")
        self.assertEqual(resultado["qtd_contornos_acionaveis"], 0)

    def test_um_verde_antes_e_outro_depois_usa_so_o_antes(self):
        verde = self.verde(("ESQUERDA", "DIREITA"), y=330)
        verde["contornos_confirmados"][1]["bbox"] = (400, 240, 60, 50)
        resultado = decidir_verde_acionavel(verde, self.analise())
        self.assertEqual(resultado["verde_acionavel"], "ESQUERDA")
        self.assertEqual(resultado["acao_visual"], "PREPARAR_ESQUERDA")
        self.assertEqual(resultado["qtd_contornos_acionaveis"], 1)

    def test_no_nao_confiavel_nao_aciona(self):
        resultado = decidir_verde_acionavel(self.verde(), self.analise(confiavel=False))
        self.assertEqual(resultado["verde_acionavel"], "NENHUM")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")
        self.assertEqual(resultado["motivo_acao"], "no_intersecao_nao_confiavel")

    def test_reta_e_nenhuma_seguem_reto(self):
        for tipo in ("RETA", "NENHUMA"):
            with self.subTest(tipo=tipo):
                resultado = decidir_verde_acionavel(self.verde(), self.analise(tipo))
                self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")

    def test_verde_central_antes_permanece_ambiguo(self):
        resultado = decidir_verde_acionavel(self.verde(("CENTRO",), y=330), self.analise())
        self.assertEqual(resultado["verde_acionavel"], "AMBIGUO")
        self.assertEqual(resultado["acao_visual"], "SEGUIR_RETO")
        self.assertEqual(resultado["motivo_acao"], "verde_ambiguo")

    def test_lado_incompativel_nao_aciona(self):
        verde = self.verde(("DIREITA",), y=330)
        resultado = decidir_verde_acionavel(verde, self.analise("LATERAL_ESQ"))
        self.assertEqual(resultado["verde_acionavel"], "NENHUM")
        self.assertEqual(verde["contornos_confirmados"][0]["motivo_acionavel"], "verde_lado_incompativel")


if __name__ == "__main__":
    unittest.main()
