"""Testes sinteticos do Green OBR Analyzer por adjacencia."""

import unittest

import cv2
import numpy as np

from green_obr_analyzer import (
    LogCompacto,
    analisar_adjacencia_preta_verde,
    classificar_marcador_verde_por_adjacencia,
    decidir_verde_obr_por_adjacencia,
    formatar_log_detalhe,
    formatar_log_compacto,
)


class GreenObrAnalyzerTests(unittest.TestCase):
    def mascara(self):
        return np.zeros((240, 320), dtype=np.uint8)

    def contorno(self, lado="ESQUERDA", bbox=(120, 120, 30, 30), confirmado=True):
        return {
            "lado": lado,
            "bbox": bbox,
            "area": 900.0,
            "confirmado": confirmado,
            "mean_s": 120,
            "g_minus_r": 45,
            "g_minus_b": 35,
        }

    def resultado_verde(self, contornos, detectados=None):
        return {
            "tipo_confirmado": "NENHUM" if not contornos else "ESQUERDA",
            "contornos": contornos,
            "contornos_confirmados": contornos,
            "qtd_contornos_detectados": len(contornos) if detectados is None else detectados,
            "qtd_contornos_confirmados": len(contornos),
        }

    def analise(self, tipo="CRUZ"):
        return {"tipo_intersecao": tipo}

    def pintar_adj(self, mascara, bbox, top=False, bottom=False, left=False, right=False):
        x, y, w, h = bbox
        mx = max(8, int(w * 0.8))
        my = max(8, int(h * 0.8))
        if top:
            cv2.rectangle(mascara, (x - mx, y - my), (x + w + mx - 1, y - 1), 255, -1)
        if bottom:
            cv2.rectangle(mascara, (x - mx, y + h), (x + w + mx - 1, y + h + my - 1), 255, -1)
        if left:
            cv2.rectangle(mascara, (x - mx, y), (x - 1, y + h - 1), 255, -1)
        if right:
            cv2.rectangle(mascara, (x + w, y), (x + w + mx - 1, y + h - 1), 255, -1)

    def decidir_com_adj(self, itens, tipo_intersecao="CRUZ"):
        mascara = self.mascara()
        contornos = []
        for item in itens:
            contorno = self.contorno(lado=item.get("lado", "ESQUERDA"), bbox=item["bbox"])
            self.pintar_adj(
                mascara,
                item["bbox"],
                top=item.get("top", False),
                bottom=item.get("bottom", False),
                left=item.get("left", False),
                right=item.get("right", False),
            )
            contornos.append(contorno)
        return decidir_verde_obr_por_adjacencia(
            self.resultado_verde(contornos), mascara, self.analise(tipo_intersecao)
        )

    def test_formatacao_compacta(self):
        linha = formatar_log_compacto(
            {
                "decisao": "ESQ",
                "verde": "ESQUERDA",
                "qtd_confirmados": 1,
                "qtd_detectados": 1,
                "adj": "T1B0L0R1",
                "intersecao": "CRUZ",
                "confianca": "OK",
                "motivo": "marcador_esq",
            }
        )
        self.assertIn("[GOBR]", linha)
        self.assertIn("dec=ESQ", linha)
        self.assertIn("v=ESQUERDA 1/1", linha)
        self.assertIn("adj=T1B0L0R1", linha)
        self.assertIn("int=CRUZ", linha)
        self.assertIn("conf=OK", linha)
        self.assertIn("motivo=marcador_esq", linha)

    def test_adjacencia_detecta_preto_acima_e_direita(self):
        mascara = self.mascara()
        contorno = self.contorno()
        self.pintar_adj(mascara, contorno["bbox"], top=True, right=True)
        analisar_adjacencia_preta_verde(contorno, mascara, 320, 240)
        self.assertTrue(contorno["black_top"])
        self.assertTrue(contorno["black_right"])
        self.assertFalse(contorno["black_left"])
        self.assertFalse(contorno["black_bottom"])

    def test_marcador_esquerdo_valido(self):
        resultado = self.decidir_com_adj(
            [{"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "top": True, "right": True}]
        )
        self.assertEqual(resultado["decisao"], "ESQ")
        self.assertEqual(resultado["motivo"], "marcador_esq")

    def test_marcador_direito_valido(self):
        resultado = self.decidir_com_adj(
            [{"bbox": (190, 120, 30, 30), "lado": "DIREITA", "top": True, "left": True}]
        )
        self.assertEqual(resultado["decisao"], "DIR")
        self.assertEqual(resultado["motivo"], "marcador_dir")

    def test_verde_depois_ou_falso(self):
        resultado = self.decidir_com_adj(
            [{"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "bottom": True, "right": True}]
        )
        self.assertEqual(resultado["decisao"], "RETO")
        self.assertEqual(resultado["motivo"], "verde_depois_ou_falso")

    def test_dois_verdes_validos_retorno(self):
        resultado = self.decidir_com_adj(
            [
                {"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "top": True, "right": True},
                {"bbox": (190, 120, 30, 30), "lado": "DIREITA", "top": True, "left": True},
            ]
        )
        self.assertEqual(resultado["decisao"], "RETORNO")
        self.assertEqual(resultado["verde"], "DUPLO")

    def test_um_valido_e_um_falso_nao_gera_retorno(self):
        resultado = self.decidir_com_adj(
            [
                {"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "top": True, "right": True},
                {"bbox": (190, 120, 30, 30), "lado": "DIREITA", "bottom": True, "left": True},
            ]
        )
        self.assertEqual(resultado["decisao"], "ESQ")
        self.assertEqual(resultado["verde"], "ESQUERDA")

    def test_padrao_ambiguo(self):
        resultado = self.decidir_com_adj(
            [
                {
                    "bbox": (90, 120, 30, 30),
                    "lado": "ESQUERDA",
                    "top": True,
                    "bottom": True,
                    "left": True,
                    "right": True,
                }
            ]
        )
        self.assertEqual(resultado["decisao"], "RETO")
        self.assertEqual(resultado["motivo"], "verde_ambiguo")

    def test_log_compacto_nao_repete_antes_do_intervalo(self):
        log = LogCompacto(intervalo=0.25)
        self.assertTrue(log.deve_logar("linha", 10.0))
        self.assertFalse(log.deve_logar("linha", 10.1))

    def test_mudanca_de_decisao_loga_imediatamente(self):
        log = LogCompacto(intervalo=0.25)
        self.assertTrue(log.deve_logar("dec=RETO", 10.0))
        self.assertTrue(log.deve_logar("dec=ESQ", 10.1))

    def test_classificador_rejeita_nao_confirmado(self):
        contorno = self.contorno(confirmado=False)
        classe, motivo = classificar_marcador_verde_por_adjacencia(contorno)
        self.assertEqual(classe, "MARCADOR_INVALIDO")
        self.assertEqual(motivo, "verde_nao_confirmado")

    def test_intersecao_nenhuma_nao_bloqueia_marcador_esquerdo(self):
        resultado = self.decidir_com_adj(
            [{"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "top": True, "right": True}],
            tipo_intersecao="NENHUMA",
        )
        self.assertEqual(resultado["decisao"], "ESQ")
        self.assertEqual(resultado["confianca"], "BAIXA")
        self.assertEqual(resultado["motivo"], "marcador_esq_sem_intersecao")

    def test_intersecao_reta_nao_bloqueia_marcador_direito(self):
        resultado = self.decidir_com_adj(
            [{"bbox": (190, 120, 30, 30), "lado": "DIREITA", "top": True, "left": True}],
            tipo_intersecao="RETA",
        )
        self.assertEqual(resultado["decisao"], "DIR")
        self.assertEqual(resultado["confianca"], "BAIXA")
        self.assertEqual(resultado["motivo"], "marcador_dir_sem_intersecao")

    def test_intersecao_ambigua_nao_bloqueia_retorno(self):
        resultado = self.decidir_com_adj(
            [
                {"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "top": True, "right": True},
                {"bbox": (190, 120, 30, 30), "lado": "DIREITA", "top": True, "left": True},
            ],
            tipo_intersecao="AMBIGUA",
        )
        self.assertEqual(resultado["decisao"], "RETORNO")
        self.assertEqual(resultado["confianca"], "BAIXA")
        self.assertEqual(resultado["motivo"], "duplo_sem_intersecao")

    def test_log_compacto_inclui_confianca_baixa(self):
        linha = formatar_log_compacto(
            {
                "decisao": "ESQ",
                "verde": "ESQUERDA",
                "qtd_confirmados": 1,
                "qtd_detectados": 1,
                "adj": "T1B0L0R1",
                "intersecao": "NENHUMA",
                "confianca": "BAIXA",
                "motivo": "marcador_esq_sem_intersecao",
            }
        )
        self.assertIn("conf=BAIXA", linha)
        self.assertIn("motivo=marcador_esq_sem_intersecao", linha)

    def test_log_detalhe_mostra_confirmados_e_rejeitados(self):
        mascara = self.mascara()
        confirmado = self.contorno(bbox=(90, 120, 30, 30))
        rejeitado = {
            **self.contorno(bbox=(190, 120, 30, 30), confirmado=False),
            "motivo_confirmacao": "sem_linha_preta_proxima",
            "black_near_pixels": 12,
            "area_in_confirm_zone_ratio": 0.42,
        }
        self.pintar_adj(mascara, confirmado["bbox"], top=True, right=True)
        resultado = decidir_verde_obr_por_adjacencia(
            {
                "tipo_confirmado": "ESQUERDA",
                "contornos": [confirmado, rejeitado],
                "contornos_confirmados": [confirmado],
                "qtd_contornos_detectados": 2,
                "qtd_contornos_confirmados": 1,
            },
            mascara,
            self.analise("CRUZ"),
        )
        detalhe = formatar_log_detalhe(resultado)
        self.assertIn("confirmados:", detalhe)
        self.assertIn("rejeitados:", detalhe)
        self.assertIn("motivo_confirmacao=sem_linha_preta_proxima", detalhe)

    def test_intersecao_reta_nao_bloqueia_esquerda_legado(self):
        resultado = self.decidir_com_adj(
            [{"bbox": (90, 120, 30, 30), "lado": "ESQUERDA", "top": True, "right": True}],
            tipo_intersecao="RETA",
        )
        self.assertEqual(resultado["decisao"], "ESQ")
        self.assertEqual(resultado["motivo"], "marcador_esq_sem_intersecao")


if __name__ == "__main__":
    unittest.main()
