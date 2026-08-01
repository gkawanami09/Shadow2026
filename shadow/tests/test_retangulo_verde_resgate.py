"""Testes da ida final ao retangulo verde."""

import sys
from pathlib import Path
from types import SimpleNamespace
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


def marcador_verde(
    timestamp,
    center_x=320.0,
    width=120.0,
    bottom_y=380.0,
):
    return SimpleNamespace(
        kind="green",
        center_x=float(center_x),
        center_y=330.0,
        width=float(width),
        height=100.0,
        bottom_y=float(bottom_y),
        area=float(width * 50.0),
        confidence=0.95,
        confirmed=True,
        hits=3,
        timestamp=float(timestamp),
        track_locked=True,
    )


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

    def test_mesma_logica_pode_procurar_e_confirmar_o_vermelho(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            avanco_direto=True,
            target_kind="red",
        )

        self.assertEqual(controlador.target_kind, "red")
        self.assertEqual(controlador.navegacao.target_kind, "red")
        self.assertTrue(controlador.navegacao.pulsed_search)
        self.assertEqual(
            controlador.navegacao.search_tank_speed,
            cfg.RED_DEPOSIT_SEARCH_TANK_SPEED,
        )
        self.assertEqual(
            controlador.navegacao.search_full_turn_s,
            cfg.RED_DEPOSIT_SEARCH_FULL_TURN_S,
        )
        chegada = controlador.update(
            None,
            FORMATO,
            now=0.1,
            distancia_chegada_mm=cfg.RESCUE_GREEN_ARRIVAL_DISTANCE_MM,
            distancia_atual_mm=cfg.RESCUE_GREEN_ARRIVAL_DISTANCE_MM,
        )
        self.assertEqual(chegada.state, "RED_ARRIVAL_7CM")
        self.assertTrue(chegada.terminal)

    def test_controlador_vermelho_observa_parado_antes_do_primeiro_pulso(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            target_kind="red",
        )

        inicio = controlador.update(None, FORMATO, now=0.0)
        self.assertEqual(
            inicio.state,
            controlador.navegacao.PULSE_BRAKE,
        )
        self.assertEqual(inicio.angle, 190)
        self.assertTrue(controlador.notify_command_written(
            inicio.state, now=0.0))

        assentando = controlador.update(
            None,
            FORMATO,
            now=0.01,
        )
        observando = controlador.update(
            None,
            FORMATO,
            now=cfg.DEPOSIT_SEARCH_SETTLE_S,
        )
        self.assertEqual(
            assentando.state,
            controlador.navegacao.PULSE_SETTLE,
        )
        self.assertEqual(assentando.angle, 190)
        self.assertEqual(
            observando.state,
            controlador.navegacao.PULSE_OBSERVE,
        )
        self.assertEqual(observando.angle, 190)

    def test_avanca_reto_em_pwm_80_mesmo_com_verde_de_um_lado(self):
        comando = self.controlador.update(
            None,
            FORMATO,
            mascara_verde=mascara_com_faixa(400, 640),
            timestamp_frame=0.01,
            now=0.01,
            distancia_atual_mm=300,
        )

        self.assertEqual(
            comando.state, self.controlador.APROXIMACAO_FINAL)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed, cfg.RESCUE_GREEN_FINAL_FORWARD_SPEED)
        self.assertEqual(
            round(comando.speed * 120), cfg.RESCUE_GREEN_FINAL_PWM)

    def test_camera_alinha_e_aproxima_antes_de_habilitar_ultrassom(self):
        controlador = ControladorRetanguloVerde(start_time=0.0)

        encontrou = controlador.update(
            marcador_verde(0.01, center_x=500),
            FORMATO,
            now=0.01,
            distancia_chegada_mm=30,
        )
        self.assertEqual(encontrou.state, controlador.navegacao.TARGET_STOP)
        self.assertFalse(controlador.ultrassom_habilitado)
        controlador.notify_command_written(encontrou.state, now=0.02)

        alinhando = controlador.update(
            marcador_verde(0.03, center_x=500),
            FORMATO,
            now=0.03,
            distancia_chegada_mm=30,
        )
        self.assertEqual(alinhando.state, controlador.navegacao.ALIGN)
        self.assertEqual(alinhando.angle, 180)
        self.assertEqual(
            round(alinhando.speed * 120 * 1.2),
            cfg.RESCUE_GREEN_FINAL_PWM,
        )
        self.assertFalse(controlador.ultrassom_habilitado)

        aproximando = controlador.update(
            marcador_verde(
                0.06,
                center_x=320,
                width=120,
                bottom_y=380,
            ),
            FORMATO,
            now=0.06,
            distancia_chegada_mm=30,
        )
        self.assertEqual(
            aproximando.state, controlador.navegacao.APPROACH)
        self.assertEqual(aproximando.angle, 0)
        self.assertEqual(
            round(aproximando.speed * 120),
            cfg.RESCUE_GREEN_FINAL_PWM,
        )
        self.assertFalse(controlador.ultrassom_habilitado)

        chegada_visual = controlador.update(
            marcador_verde(
                0.10,
                center_x=320,
                width=220,
                bottom_y=430,
            ),
            FORMATO,
            now=0.10,
            distancia_chegada_mm=30,
        )

        self.assertEqual(
            chegada_visual.state, controlador.navegacao.ARRIVAL_STOP)
        self.assertFalse(controlador.ultrassom_habilitado)
        controlador.notify_command_written(chegada_visual.state, now=0.11)
        self.assertTrue(controlador.ultrassom_habilitado)

        avanco = controlador.update(
            marcador_verde(0.12),
            FORMATO,
            now=0.12,
            distancia_atual_mm=300,
        )
        self.assertEqual(avanco.state, controlador.APROXIMACAO_FINAL)
        self.assertEqual(avanco.angle, 0)
        self.assertEqual(
            round(avanco.speed * 120), cfg.RESCUE_GREEN_FINAL_PWM)

    def test_programa_principal_nao_pula_alinhamento_visual(self):
        fonte = (SHADOW_ROOT / "resgate.py").read_text(encoding="utf-8")

        self.assertNotIn("avanco_direto=True", fonte)
        self.assertIn(
            "and controlador_verde.ultrassom_habilitado",
            fonte,
        )
        self.assertIn(
            "monitor_chegada_verde = None",
            fonte,
        )

    def test_avanco_direto_nao_exige_tela_inteira_verde(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            avanco_direto=True,
        )
        cheia = np.full(FORMATO[:2], 255, dtype=np.uint8)

        comando = controlador.update(
            None,
            FORMATO,
            cheia,
            0.10,
            now=0.10,
            distancia_atual_mm=300,
        )

        self.assertEqual(comando.state, controlador.APROXIMACAO_FINAL)
        self.assertFalse(comando.terminal)
        self.assertEqual(comando.angle, 0)
        self.assertIn("ultrassonico", comando.detail)

    def test_avanco_direto_so_encerra_em_sete_centimetros(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            avanco_direto=True,
        )

        longe = controlador.update(
            None,
            FORMATO,
            now=0.10,
            distancia_atual_mm=71,
        )
        primeira_leitura = controlador.update(
            None,
            FORMATO,
            now=0.15,
            distancia_atual_mm=70,
        )
        chegada = controlador.update(
            None,
            FORMATO,
            now=0.20,
            distancia_chegada_mm=70,
            distancia_atual_mm=70,
        )
        travado = controlador.update(None, FORMATO, now=0.30)

        self.assertEqual(longe.state, controlador.APROXIMACAO_FINAL)
        self.assertFalse(longe.terminal)
        self.assertEqual(
            primeira_leitura.state, controlador.CONFIRMANDO_DISTANCIA)
        self.assertEqual(primeira_leitura.angle, 190)
        self.assertFalse(primeira_leitura.terminal)
        self.assertEqual(chegada.state, controlador.CONCLUIDO)
        self.assertTrue(chegada.terminal)
        self.assertEqual(chegada.angle, 190)
        self.assertEqual(chegada.speed, 0.0)
        self.assertEqual(travado.state, controlador.CONCLUIDO)
        self.assertTrue(travado.terminal)

    def test_avanco_direto_espera_primeira_leitura_valida(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            avanco_direto=True,
        )

        comando = controlador.update(None, FORMATO, now=0.10)

        self.assertEqual(comando.state, controlador.APROXIMACAO_FINAL)
        self.assertEqual(comando.angle, 190)
        self.assertEqual(comando.speed, 0.0)
        self.assertIn("validando ultrassonico", comando.detail)

    def test_avanco_direto_para_no_primeiro_sem_eco_e_falha_no_terceiro(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            avanco_direto=True,
        )

        aguardando = controlador.update(
            None,
            FORMATO,
            now=0.10,
            distancia_atual_mm=300,
            ultrassonico_sem_eco=True,
        )
        falha = controlador.update(
            None,
            FORMATO,
            now=0.20,
            distancia_atual_mm=300,
            ultrassonico_sem_eco=True,
            ultrassonico_falhou=True,
        )

        self.assertEqual(aguardando.angle, 190)
        self.assertFalse(aguardando.terminal)
        self.assertEqual(falha.state, controlador.FALHA)
        self.assertTrue(falha.terminal)

    def test_avanco_direto_para_se_ultrassonico_nao_confirmar(self):
        controlador = ControladorRetanguloVerde(
            start_time=0.0,
            avanco_direto=True,
        )

        falha = controlador.update(
            None,
            FORMATO,
            now=cfg.RESCUE_GREEN_FINAL_MAX_ACTIVE_S,
        )

        self.assertEqual(falha.state, controlador.FALHA)
        self.assertTrue(falha.terminal)
        self.assertEqual(falha.angle, 190)

    def test_tela_inteira_verde_nao_substitui_o_ultrassonico(self):
        cheia = np.full(FORMATO[:2], 255, dtype=np.uint8)

        comando = self.controlador.update(
            None, FORMATO, cheia, 0.01, now=0.01)

        self.assertEqual(comando.state, self.controlador.APROXIMACAO_FINAL)
        self.assertEqual(comando.angle, 190)
        self.assertFalse(comando.terminal)
        self.assertIn("validando ultrassonico", comando.detail)

    def test_nao_avanca_sem_um_frame_novo(self):
        comando = self.controlador.update(
            None, FORMATO, now=0.10)

        self.assertEqual(comando.angle, 190)
        self.assertEqual(comando.speed, 0.0)
        self.assertFalse(comando.terminal)

    def test_leitura_sem_eco_para_sem_depender_da_mascara_verde(self):
        comando = self.controlador.update(
            None,
            FORMATO,
            now=0.10,
            distancia_atual_mm=300,
            ultrassonico_sem_eco=True,
        )

        self.assertEqual(comando.angle, 190)
        self.assertEqual(comando.speed, 0.0)
        self.assertFalse(comando.terminal)

    def test_timeout_final_funciona_mesmo_com_distancia_valida(self):
        self.controlador.update(
            None, FORMATO, now=0.01, distancia_atual_mm=300)
        falha = self.controlador.update(
            None,
            FORMATO,
            now=cfg.RESCUE_GREEN_FINAL_MAX_ACTIVE_S + 0.02,
            distancia_atual_mm=300,
        )

        self.assertEqual(falha.state, self.controlador.FALHA)
        self.assertTrue(falha.terminal)


if __name__ == "__main__":
    unittest.main()
