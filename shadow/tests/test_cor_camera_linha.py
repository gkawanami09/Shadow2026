"""Ordem de canais da câmera de linha e as faixas de cor que dependem dela.

Contexto medido na arena: a câmera de linha convertia RGB→BGR, o que TROCAVA
os canais R e B. A câmera de resgate nunca fez isso. O sintoma foi vermelho
aparecendo com matiz 120 (azul) e a faixa vermelha final nunca sendo
detectada; o verde sobrevivia porque tinha sido calibrado já em cima da
imagem trocada.

Estes testes existem para que a correção não seja desfeita sem querer e para
documentar a relação exata entre os dois espaços de matiz.
"""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402


def matiz(bgr):
    return int(cv2.cvtColor(
        np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0][0])


def trocar_rb(bgr):
    return [bgr[2], bgr[1], bgr[0]]


#: Amostras BGR reais de marcadores e faixas, na ordem CORRETA de canais.
VERMELHOS = ([40, 40, 200], [30, 30, 120], [55, 60, 210])
VERDES = ([40, 200, 60], [30, 120, 50], [50, 190, 120])


class RelacaoEntreOsEspacosTests(unittest.TestCase):
    def test_a_troca_leva_vermelho_para_o_azul(self):
        """É por isso que a faixa vermelha nunca era detectada."""
        for bgr in VERMELHOS:
            self.assertLessEqual(matiz(bgr), 10)
            self.assertGreater(matiz(trocar_rb(bgr)), 100)

    def test_conversao_de_matiz_e_exata(self):
        """H_correto = 120 − H_trocado (módulo 180), com folga de 1 unidade."""
        for bgr in VERMELHOS + VERDES:
            esperado = (120 - matiz(trocar_rb(bgr))) % 180
            self.assertLessEqual(
                abs(matiz(bgr) - esperado), 1,
                f"conversao falhou para {bgr}")

    def test_saturacao_e_brilho_nao_mudam_com_a_troca(self):
        """Só o matiz muda — por isso a migração mexeu apenas em H."""
        for bgr in VERMELHOS + VERDES:
            direto = cv2.cvtColor(
                np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            trocado = cv2.cvtColor(
                np.uint8([[trocar_rb(bgr)]]), cv2.COLOR_BGR2HSV)[0][0]
            self.assertEqual(int(direto[1]), int(trocado[1]))
            self.assertEqual(int(direto[2]), int(trocado[2]))


class FaixasCalibradasTests(unittest.TestCase):
    """As faixas salvas precisam casar com a imagem de canais CORRETOS."""

    def test_vermelho_casa_agora(self):
        baixo = (config.RED_MIN_1_DEFAULT[0], config.RED_MAX_1_DEFAULT[0])
        alto = (config.RED_MIN_2_DEFAULT[0], config.RED_MAX_2_DEFAULT[0])
        for bgr in VERMELHOS:
            h = matiz(bgr)
            self.assertTrue(
                baixo[0] <= h <= baixo[1] or alto[0] <= h <= alto[1],
                f"vermelho {bgr} (H={h}) fora das duas bandas")

    def test_verde_migrado_continua_casando(self):
        """A migração precisa preservar o comportamento do segue-linha."""
        minimo = config.GREEN_MIN_DEFAULT[0]
        maximo = config.GREEN_MAX_DEFAULT[0]
        for bgr in VERDES:
            h = matiz(bgr)
            self.assertTrue(
                minimo <= h <= maximo,
                f"verde {bgr} (H={h}) fora da faixa {minimo}..{maximo}")

    def test_verde_escuro_saturado_e_aceito_sem_aceitar_cinza_escuro(self):
        minimo = np.array(config.GREEN_MIN_DEFAULT, dtype=np.uint8)
        maximo = np.array(config.GREEN_MAX_DEFAULT, dtype=np.uint8)
        verde_escuro = np.uint8([[[35, 180, 24]]])
        cinza_escuro = np.uint8([[[35, 50, 24]]])

        self.assertEqual(int(cv2.inRange(verde_escuro, minimo, maximo)[0, 0]),
                         255)
        self.assertEqual(int(cv2.inRange(cinza_escuro, minimo, maximo)[0, 0]),
                         0)

    def test_verde_e_vermelho_nao_se_sobrepoem(self):
        self.assertGreater(
            config.GREEN_MIN_DEFAULT[0], config.RED_MAX_1_DEFAULT[0],
            "faixa verde invadiu a banda baixa do vermelho")
        self.assertLess(
            config.GREEN_MAX_DEFAULT[0], config.RED_MIN_2_DEFAULT[0],
            "faixa verde invadiu a banda alta do vermelho")

    def test_preto_e_praticamente_simetrico_entre_canais(self):
        """O preto quase não sente a troca; por isso não foi migrado."""
        for teto in (config.BLACK_MAX_NORMAL_TOP_DEFAULT,
                     config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT):
            self.assertLessEqual(
                abs(teto[0] - teto[2]), 4,
                f"teto de preto {teto} nao e simetrico entre B e R")


class OrdemDeCanaisDaCameraTests(unittest.TestCase):
    def test_camera_de_linha_nao_converte_mais_rgb_para_bgr(self):
        """Guarda contra reintroduzir a troca sem perceber."""
        fonte = (SHADOW_ROOT / "visao" / "captura.py").read_text(
            encoding="utf-8")
        corpo = fonte.split("def get_frame")[1]
        self.assertNotIn(
            "COLOR_RGB2BGR", corpo,
            "a troca R<->B voltou para a camera de linha")

    def test_as_duas_cameras_tratam_a_cor_do_mesmo_jeito(self):
        linha = (SHADOW_ROOT / "visao" / "captura.py").read_text(
            encoding="utf-8").split("def get_frame")[1]
        resgate = (SHADOW_ROOT / "visao" / "captura_resgate.py").read_text(
            encoding="utf-8").split("def get_frame")[1]
        for corpo, nome in ((linha, "linha"), (resgate, "resgate")):
            self.assertIn(
                "COLOR_BGRA2BGR", corpo,
                f"camera de {nome} nao trata o buffer como BGR nativo")


if __name__ == "__main__":
    unittest.main()
