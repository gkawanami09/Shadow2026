"""Testes determinísticos do hotfix de recuperação por destino."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

try:
    import serial  # noqa: F401
except ModuleNotFoundError:
    serial = types.ModuleType("serial")
    serial.SerialException = Exception
    serial_tools = types.ModuleType("serial.tools")
    serial_list_ports = types.ModuleType("serial.tools.list_ports")
    serial_list_ports.comports = lambda: []
    serial_tools.list_ports = serial_list_ports
    serial.tools = serial_tools
    sys.modules.update({
        "serial": serial,
        "serial.tools": serial_tools,
        "serial.tools.list_ports": serial_list_ports,
    })

import follow_destinos as follow


def memoria():
    return {
        "ultimo_lado_recuperacao": "ESQUERDA",
        "lado_recuperacao_pendente": "CENTRO",
        "frames_confirmacao_lado_recuperacao": 0,
        "ultimo_tipo_confiavel": "PERDIDO",
        "ultimo_score_confiavel": 0,
        "frames_mesmo_tipo": 0,
        "tipo_pendente": "PERDIDO",
        "frames_confirmacao_tipo": 0,
        "em_recuperacao": False,
    }


def destino(tipo, angulo, score=100):
    return {
        "ok": True,
        "tipo": tipo,
        "angulo_destino": angulo,
        "score": score,
        "validos_por_tipo": {"FRENTE": [], "CURVA": [], "RETORNO": []},
    }


class MemoriaRecuperacaoTests(unittest.TestCase):
    def test_frente_direita_nao_altera_lado_de_recuperacao(self):
        estado = memoria()
        follow.atualizar_memoria_recuperacao(destino("FRENTE", 20), estado)

        self.assertEqual(estado["ultimo_lado_recuperacao"], "ESQUERDA")
        self.assertEqual(estado["lado_recuperacao_pendente"], "CENTRO")

    def test_curva_forte_exige_dois_frames_iguais(self):
        estado = memoria()
        curva = destino("CURVA", 40)
        follow.atualizar_memoria_recuperacao(curva, estado)
        self.assertEqual(estado["ultimo_lado_recuperacao"], "ESQUERDA")
        self.assertEqual(estado["lado_recuperacao_pendente"], "DIREITA")

        follow.atualizar_memoria_recuperacao(curva, estado)
        self.assertEqual(estado["ultimo_lado_recuperacao"], "DIREITA")
        self.assertEqual(estado["lado_recuperacao_pendente"], "CENTRO")

    def test_curvas_fracas_ou_alternantes_nao_contaminam_memoria(self):
        estado = memoria()
        follow.atualizar_memoria_recuperacao(destino("CURVA", 30), estado)
        follow.atualizar_memoria_recuperacao(destino("CURVA", 40), estado)
        follow.atualizar_memoria_recuperacao(destino("CURVA", -40), estado)

        self.assertEqual(estado["ultimo_lado_recuperacao"], "ESQUERDA")
        self.assertEqual(estado["lado_recuperacao_pendente"], "ESQUERDA")
        self.assertEqual(estado["frames_confirmacao_lado_recuperacao"], 1)

    def test_retorno_confirmado_atualiza_lado_de_recuperacao(self):
        estado = memoria()
        retorno = destino("RETORNO", 80)

        follow.atualizar_memoria_recuperacao(destino("CURVA", 40), estado)
        self.assertTrue(follow.decidir_confirmacao_destino(retorno, estado))
        self.assertEqual(estado["frames_confirmacao_lado_recuperacao"], 1)
        self.assertFalse(follow.decidir_confirmacao_destino(retorno, estado))
        self.assertEqual(estado["ultimo_lado_recuperacao"], "DIREITA")


class RecuperacaoESelecaoTests(unittest.TestCase):
    def test_destino_valido_sem_linha_confirma_sem_recuperar(self):
        resultado = {"encontrou_linha": False}
        destino_valido = destino("RETORNO", 80)
        deve_recuperar, motivo = follow.avaliar_recuperacao(resultado, destino_valido, memoria())

        self.assertFalse(deve_recuperar)
        self.assertEqual(motivo, "LINHA_FALSE_DESTINO_OK")
        self.assertTrue(follow.deve_confirmar_destino(resultado, destino_valido))

    def test_recuperacao_latente_com_destino_valido_nao_gira(self):
        resultado = {"encontrou_linha": False}
        destino_valido = destino("RETORNO", 80)
        estado = memoria()
        estado["em_recuperacao"] = True

        deve_recuperar, motivo = follow.avaliar_recuperacao(resultado, destino_valido, estado)

        self.assertFalse(deve_recuperar)
        self.assertEqual(motivo, "LINHA_FALSE_DESTINO_OK")
        self.assertTrue(follow.deve_confirmar_destino(resultado, destino_valido))

    def test_tipo_usa_vetor_final_e_nao_angulo_do_raio(self):
        resultado = {
            "mascara_limpa": np.zeros((100, 100), dtype=np.uint8),
            "centro_imagem_x": 50,
            "x_inicio_roi": 0,
            "y_inicio_roi": 0,
        }

        def amostra_frontal(_, origem, angulo):
            if angulo != 0:
                return {"ok": False, "angulo": angulo, "origem_local": origem, "ponto_raio": origem, "hits": []}
            return {
                "ok": True, "angulo": angulo, "origem_local": origem, "ponto_raio": (50, 20),
                "pixels": 100, "continuidade": 1.0, "distancia": 160, "segmento_hits": 6,
                "destino_local": (80, 60), "hits": [],
            }

        with patch("follow_destinos.amostrar_raio", side_effect=amostra_frontal):
            escolhido = follow.escolher_destino(resultado)

        self.assertTrue(escolhido["ok"])
        self.assertEqual(escolhido["angulo_raio"], 0)
        self.assertEqual(escolhido["tipo"], "CURVA")
        self.assertEqual(escolhido["lado"], "DIREITA")


if __name__ == "__main__":
    unittest.main()
