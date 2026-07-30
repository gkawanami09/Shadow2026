"""Testes da ida final ao retangulo verde."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.retangulo_verde_resgate import (  # noqa: E402
    ConfirmadorTelaVerde,
    ControladorRetanguloVerde,
    medir_verde,
)
from visao.marcador_resgate import color_masks  # noqa: E402


FORMATO = (480, 640, 3)


def mascara_com_faixa(x1, x2):
    mascara = np.zeros(FORMATO[:2], dtype=np.uint8)
    mascara[:, x1:x2] = 255
    return mascara


class ConfirmadorTelaVerdeTests(unittest.TestCase):
    def test_tela_quase_toda_verde_precisa_de_tres_frames_novos(self):
        confirmador = ConfirmadorTelaVerde()
        mascara = np.full(FORMATO[:2], 255, dtype=np.uint8)

        self.assertFalse(confirmador.observar(mascara, 0.00))
        self.assertFalse(confirmador.observar(mascara, 0.10))
        self.assertTrue(confirmador.observar(mascara, 0.20))
        self.assertTrue(confirmador.confirmado)

    def test_timestamp_repetido_nao_aumenta_confirmacao(self):
        confirmador = ConfirmadorTelaVerde()
        mascara = np.full(FORMATO[:2], 255, dtype=np.uint8)

        confirmador.observar(mascara, 1.0)
        confirmador.observar(mascara, 1.0)

        self.assertEqual(confirmador.quantidade, 1)
        self.assertFalse(confirmador.confirmado)

    def test_retangulo_visivel_nao_e_confundido_com_tela_inteira(self):
        confirmador = ConfirmadorTelaVerde()
        mascara = mascara_com_faixa(160, 480)

        for timestamp in (0.0, 0.1, 0.2, 0.3):
            self.assertFalse(
                confirmador.observar(mascara, timestamp))

        proporcao, erro = medir_verde(mascara)
        self.assertLess(proporcao, cfg.RESCUE_GREEN_FULL_FRAME_MIN_RATIO)
        self.assertAlmostEqual(erro, 0.0, delta=0.02)

    def test_centroide_indica_o_lado_do_verde(self):
        _proporcao, erro_esquerda = medir_verde(
            mascara_com_faixa(0, 180))
        _proporcao, erro_direita = medir_verde(
            mascara_com_faixa(460, 640))

        self.assertLess(erro_esquerda, 0.0)
        self.assertGreater(erro_direita, 0.0)

    def test_confirmacao_recebe_a_mascara_hsv_usada_no_programa(self):
        quadro_verde = np.zeros(FORMATO, dtype=np.uint8)
        quadro_verde[:, :] = (0, 255, 0)
        quadro_branco = np.full(FORMATO, 255, dtype=np.uint8)

        proporcao_verde, _erro = medir_verde(
            color_masks(quadro_verde)["green"])
        proporcao_branco, _erro = medir_verde(
            color_masks(quadro_branco)["green"])

        self.assertGreaterEqual(
            proporcao_verde, cfg.RESCUE_GREEN_FULL_FRAME_MIN_RATIO)
        self.assertEqual(proporcao_branco, 0.0)


class ControladorRetanguloVerdeTests(unittest.TestCase):
    def setUp(self):
        self.controlador = ControladorRetanguloVerde(start_time=0.0)
        self.controlador.navegacao.state = (
            self.controlador.navegacao.ARRIVAL_STOP)
        mudou = self.controlador.notify_command_written(
            self.controlador.navegacao.ARRIVAL_STOP, now=0.0)
        self.assertTrue(mudou)
        self.assertTrue(self.controlador.aproximacao_final)

    def test_avanca_corrigindo_para_o_lado_do_verde(self):
        comando = self.controlador.update(
            None,
            FORMATO,
            mascara_verde=mascara_com_faixa(400, 640),
            timestamp_frame=0.01,
            now=0.01,
        )

        self.assertEqual(
            comando.state, self.controlador.APROXIMACAO_FINAL)
        self.assertGreater(comando.angle, 0)
        self.assertEqual(
            comando.speed, cfg.RESCUE_GREEN_FINAL_FORWARD_SPEED)

    def test_para_para_confirmar_e_encerra_no_terceiro_frame(self):
        cheia = np.full(FORMATO[:2], 255, dtype=np.uint8)

        primeiro = self.controlador.update(
            None, FORMATO, cheia, 0.01, now=0.01)
        segundo = self.controlador.update(
            None, FORMATO, cheia, 0.11, now=0.11)
        terceiro = self.controlador.update(
            None, FORMATO, cheia, 0.21, now=0.21)

        self.assertEqual(
            primeiro.state, self.controlador.CONFIRMANDO_TELA)
        self.assertEqual(primeiro.angle, 190)
        self.assertFalse(primeiro.terminal)
        self.assertFalse(segundo.terminal)
        self.assertEqual(terceiro.state, self.controlador.CONCLUIDO)
        self.assertTrue(terceiro.terminal)
        self.assertEqual(terceiro.angle, 190)

    def test_nao_avanca_sem_um_frame_novo(self):
        comando = self.controlador.update(
            None, FORMATO, now=0.10)

        self.assertEqual(comando.angle, 190)
        self.assertEqual(comando.speed, 0.0)
        self.assertFalse(comando.terminal)

    def test_para_em_falha_se_o_verde_sumir(self):
        vazio = np.zeros(FORMATO[:2], dtype=np.uint8)
        primeiro = self.controlador.update(
            None, FORMATO, vazio, 0.01, now=0.01)
        falha = self.controlador.update(
            None,
            FORMATO,
            vazio,
            0.02 + cfg.RESCUE_GREEN_FINAL_LOST_TIMEOUT_S,
            now=0.01 + cfg.RESCUE_GREEN_FINAL_LOST_TIMEOUT_S,
        )

        self.assertEqual(primeiro.angle, 190)
        self.assertFalse(primeiro.terminal)
        self.assertEqual(falha.state, self.controlador.FALHA)
        self.assertTrue(falha.terminal)

    def test_timeout_final_tambem_funciona_com_frames_chegando(self):
        parcial = mascara_com_faixa(200, 440)
        self.controlador.update(
            None, FORMATO, parcial, 0.01, now=0.01)
        falha = self.controlador.update(
            None,
            FORMATO,
            parcial,
            cfg.RESCUE_GREEN_FINAL_MAX_ACTIVE_S + 0.02,
            now=cfg.RESCUE_GREEN_FINAL_MAX_ACTIVE_S + 0.02,
        )

        self.assertEqual(falha.state, self.controlador.FALHA)
        self.assertTrue(falha.terminal)


if __name__ == "__main__":
    unittest.main()
