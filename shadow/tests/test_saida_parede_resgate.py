"""Testes: girar, alinhar e parar a 118 mm da parede frontal."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.saida_parede_resgate import ControladorSaidaParede  # noqa: E402


class SaidaParedeResgateTests(unittest.TestCase):
    @staticmethod
    def _passo(
        controlador,
        instante,
        *,
        yaw=0,
        lateral=200,
        frente=400,
        enviar_yaw=True,
        enviar_lateral=True,
        enviar_frente=True,
        respondeu_lateral=True,
        respondeu_frente=True,
    ):
        if enviar_yaw:
            controlador.observar_mpu(yaw, instante)
        if enviar_lateral:
            controlador.observar_ultrassom(
                "LATERAL", lateral, respondeu_lateral, instante)
        if enviar_frente:
            controlador.observar_ultrassom(
                "FRENTE", frente, respondeu_frente, instante)
        comando = controlador.atualizar(instante)
        if not comando.terminal:
            controlador.notificar_comando_escrito(comando.state, instante)
        return comando

    def _chegar_ao_alinhamento(self, yaw_final=90):
        controlador = ControladorSaidaParede(start_time=0.0)
        self.assertTrue(controlador.solicita_zerar_mpu)
        self.assertTrue(controlador.confirmar_mpu_zerado(True, 0.0))

        comando = self._passo(controlador, 0.0, yaw=0)
        self.assertEqual(controlador.state, controlador.AFASTAR_VERMELHO)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_PWM / 120.0,
        )

        instante = cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_S + 0.01
        comando = self._passo(controlador, instante, yaw=0)
        self.assertEqual(controlador.state, controlador.ASSENTAR_INICIAL)
        self.assertEqual(comando.speed, 0.0)

        instante += cfg.SAIDA_PAREDE_ASSENTAMENTO_S + 0.01
        comando = self._passo(controlador, instante, yaw=0)
        self.assertEqual(controlador.state, controlador.GIRO_INICIAL_DIREITA)
        self.assertEqual(comando.angle, 180)

        sinal = 1 if yaw_final == 90 else -1
        comando = self._passo(controlador, instante + 0.03, yaw=6 * sinal)
        self.assertEqual(comando.angle, 180)
        comando = self._passo(controlador, instante + 0.40, yaw=yaw_final)
        self.assertEqual(controlador.state, controlador.ALINHAR_DIREITA)
        self.assertEqual(comando.speed, 0.0)
        self.assertEqual(controlador.heading_parede, float(yaw_final))
        return controlador, instante + 0.40

    def test_gira_direita_mesmo_com_yaw_invertido(self):
        controlador, _ = self._chegar_ao_alinhamento(yaw_final=270)
        self.assertEqual(controlador.heading_parede, 270.0)

    def _chegar_ao_avanco_frente(self):
        """Isola os testes do trecho posterior a verificacao inicial."""
        controlador, instante = self._chegar_ao_alinhamento()
        controlador._entrar_avanco_frente(instante)
        return controlador, instante

    def test_apoia_a_frente_e_translada_so_apos_um_segundo_lateral_estavel(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        comando = self._passo(
            controlador, instante + 0.01, yaw=90, lateral=130, frente=300)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_AVANCO_ATE_FRENTE_PWM / 120.0,
        )

        comando = self._passo(
            controlador, instante + 0.08, yaw=90, lateral=130, frente=118)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.PIVO_TRASEIRO_ESTABILIZAR)
        self.assertEqual(controlador.lado_ultrassom_atual, "LATERAL")
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, instante + 0.22, yaw=90, lateral=130, frente=116)
        self.assertFalse(comando.terminal)
        self.assertEqual(
            comando.angle,
            cfg.SAIDA_PAREDE_ANGULO_CURVA_TRASEIRA,
        )
        self.assertGreater(comando.angle, -110)
        self.assertLess(comando.angle, 0)
        self.assertTrue(comando.pivo_traseiro)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_PWM_PIVO_TRASEIRO / 120.0,
        )
        self.assertEqual(comando.toque_frente_direita_pwm, 0)

        comando = self._passo(
            controlador, instante + 0.50, yaw=90, lateral=130, frente=116)
        self.assertEqual(
            comando.toque_frente_direita_pwm,
            cfg.SAIDA_PAREDE_PWM_TOQUE_FRENTE_DIREITA,
        )
        self.assertIn("toque na dianteira direita", comando.detail)

        comando = self._passo(
            controlador, instante + 0.74, yaw=90, lateral=130, frente=120)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.TRANSLADAR_DIREITA_FINAL)
        comando = self._passo(
            controlador, instante + 1.26, yaw=90, lateral=130, frente=118)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.VERIFICAR_TRIANGULO_VERDE)
        self.assertTrue(controlador.usa_camera_triangulo_verde)
        self.assertEqual(comando.speed, 0.0)

    def test_reinicia_o_segundo_estavel_quando_o_lateral_variar_demais(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=118)
        comando = self._passo(
            controlador, instante + 0.13, yaw=90, lateral=125, frente=116)
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador,
            instante + 0.48,
            yaw=90,
            lateral=125 + cfg.SAIDA_PAREDE_TOLERANCIA_ESTABILIDADE_LATERAL_MM + 1,
        )
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador, instante + 0.88, yaw=90, lateral=131, frente=122)
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador, instante + 1.00, yaw=90, lateral=131, frente=123)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.TRANSLADAR_DIREITA_FINAL)
        comando = self._passo(
            controlador,
            instante + 1.00 + cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.01,
            yaw=90,
            lateral=131,
            frente=123,
        )
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.VERIFICAR_TRIANGULO_VERDE)

    def test_afasta_a_esquerda_ate_120_mm_antes_de_abrir_a_camera(self):
        controlador, instante = self._chegar_ao_alinhamento()

        # 110 mm ainda esta dentro da tolerancia do alinhamento (125 +/- 15),
        # mas nao deixa espaco suficiente para a camera procurar o triangulo.
        comando = self._passo(
            controlador, instante + 0.01, yaw=90, lateral=110, frente=400)
        self.assertEqual(controlador.state, controlador.AFASTAR_ESQUERDA_INICIAL)
        self.assertEqual(comando.speed, 0.0)
        self.assertFalse(controlador.usa_camera_triangulo_verde)

        comando = self._passo(
            controlador, instante + 0.02, yaw=90, lateral=115, frente=400)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))
        self.assertIn("yaw monitorado", comando.detail)
        self.assertFalse(controlador.usa_camera_triangulo_verde)

        comando = self._passo(
            controlador, instante + 0.09, yaw=90, lateral=120, frente=400)
        self.assertEqual(controlador.state, controlador.VERIFICAR_TRIANGULO_VERDE)
        self.assertTrue(controlador.usa_camera_triangulo_verde)
        self.assertEqual(comando.speed, 0.0)

    def test_corrige_yaw_antes_de_continuar_afastamento_inicial(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=110, frente=400)

        comando = self._passo(
            controlador, instante + 0.02, yaw=100, lateral=115, frente=400)
        self.assertEqual(
            controlador.state,
            controlador.CORRIGIR_YAW_AFASTAMENTO_INICIAL,
        )
        self.assertEqual(comando.angle, -180)

        comando = self._passo(
            controlador, instante + 0.08, yaw=90, lateral=115, frente=400)
        self.assertEqual(controlador.state, controlador.AFASTAR_ESQUERDA_INICIAL)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, instante + 0.09, yaw=90, lateral=115, frente=400)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))

    def test_verde_confirmado_gira_45_esquerda_e_reinicia_avanco_frontal(self):
        controlador = ControladorSaidaParede(start_time=0.0)
        controlador._sinal_yaw_por_giro_direita = 1.0
        controlador.observar_mpu(90, 0.0)
        controlador._entrar(controlador.VERIFICAR_TRIANGULO_VERDE, 0.0)
        controlador.notificar_comando_escrito(
            controlador.VERIFICAR_TRIANGULO_VERDE,
            0.0,
        )

        self.assertTrue(controlador.observar_triangulo_verde(True, 0.01))
        comando = self._passo(
            controlador, 0.01, yaw=90, lateral=125, frente=400)
        self.assertEqual(
            controlador.state,
            controlador.AGUARDAR_MPU_TRIANGULO_VERDE,
        )
        self.assertTrue(controlador.prioriza_mpu)
        self.assertEqual(comando.speed, 0.0)

        # Fechar a camera pode levar mais que o timeout normal do MPU. Nesse
        # intervalo o robo continua parado, sem transformar um yaw antigo em
        # falha do giro.
        comando = controlador.atualizar(0.45)
        self.assertFalse(comando.terminal)
        self.assertEqual(
            controlador.state,
            controlador.AGUARDAR_MPU_TRIANGULO_VERDE,
        )

        comando = self._passo(
            controlador, 0.50, yaw=90, lateral=125, frente=400)
        self.assertEqual(controlador.state, controlador.GIRO_TRIANGULO_VERDE)
        self.assertEqual(comando.angle, -180)
        self.assertIn("45 graus para a esquerda", comando.detail)

        comando = self._passo(
            controlador, 0.58, yaw=45, lateral=125, frente=400)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)
        self.assertEqual(controlador.heading_parede, 45.0)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, 0.59, yaw=45, lateral=125, frente=300)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_AVANCO_ATE_FRENTE_PWM / 120.0,
        )

    def test_para_se_camera_nao_confirmar_triangulo_verde_no_prazo(self):
        controlador = ControladorSaidaParede(start_time=0.0)
        controlador._entrar(controlador.VERIFICAR_TRIANGULO_VERDE, 0.0)
        controlador.notificar_comando_escrito(
            controlador.VERIFICAR_TRIANGULO_VERDE,
            0.0,
        )
        comando = controlador.atualizar(
            cfg.SAIDA_PAREDE_TIMEOUT_TRIANGULO_VERDE_S + 0.01)
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("camera frontal", comando.detail)

    def test_falha_se_o_lateral_nao_der_leitura_nova_para_o_pivo(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=118)
        comando = self._passo(
            controlador,
            instante + 0.01 + cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S + 0.01,
            yaw=90,
            enviar_lateral=False,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("pivo traseiro", comando.detail)

    def test_translada_se_lateral_oscilar_ate_timeout_do_pivo(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=118)
        self._passo(
            controlador, instante + 0.13, yaw=90, lateral=118, frente=118)
        comando = self._passo(
            controlador,
            instante + 0.01 + cfg.SAIDA_PAREDE_TIMEOUT_PIVO_TRASEIRO_S + 0.01,
            yaw=90,
            lateral=140,
            frente=118,
        )
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.TRANSLADAR_DIREITA_FINAL)
        self.assertEqual(
            comando.wheel_speeds,
            (
                cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                -cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                -cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
            ),
        )
        self.assertIn("mesmo sem estabilizar", comando.detail)

    def test_alinha_para_esquerda_se_a_parede_ja_estiver_proxima(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(controlador, instante + 0.01, yaw=90, lateral=90)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))

        comando = self._passo(
            controlador, instante + 0.08, yaw=90, lateral=120, frente=400)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.VERIFICAR_TRIANGULO_VERDE)
        self.assertTrue(controlador.usa_camera_triangulo_verde)

    def test_corrige_yaw_antes_de_continuar_a_translacao(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(controlador, instante + 0.01, yaw=100, lateral=200)
        self.assertEqual(controlador.state, controlador.CORRIGIR_YAW_ALINHAMENTO)
        self.assertEqual(comando.angle, -180)

        comando = self._passo(controlador, instante + 0.08, yaw=90, lateral=200)
        self.assertEqual(controlador.state, controlador.ALINHAR_DIREITA)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(controlador, instante + 0.09, yaw=90, lateral=200)
        self.assertEqual(comando.wheel_speeds, (45, -45, -45, 45))

    def test_para_em_falha_se_nao_receber_ultrassom_novo_apos_giro(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(
            controlador,
            instante + cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S + 0.01,
            yaw=90,
            enviar_lateral=False,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("ultrassom lateral", comando.detail)

    def test_para_em_falha_se_nao_encontrar_a_distancia_alvo_no_prazo(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(controlador, instante + 0.01, yaw=90, lateral=200)
        comando = self._passo(
            controlador,
            instante + cfg.SAIDA_PAREDE_TIMEOUT_ALINHAMENTO_S + 0.01,
            yaw=90,
            lateral=200,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("timeout", comando.detail)

    def test_para_em_falha_se_o_frontal_nao_responder_apos_alinhamento(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        self._passo(
            controlador,
            instante + 0.01,
            yaw=90,
            lateral=125,
            enviar_frente=False,
        )
        comando = self._passo(
            controlador,
            instante + 0.01 + cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S + 0.01,
            yaw=90,
            lateral=125,
            enviar_frente=False,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("ultrassom frontal", comando.detail)

    def test_para_imediatamente_se_o_frontal_responder_sem_eco(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        comando = self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=None)

        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("sem eco", comando.detail)

    def test_para_em_falha_se_nao_chegar_na_parede_frontal_no_prazo(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=300)
        comando = self._passo(
            controlador,
            instante + cfg.SAIDA_PAREDE_TIMEOUT_AVANCO_FRENTE_S + 0.02,
            yaw=90,
            lateral=125,
            frente=300,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("timeout", comando.detail)

    def test_corrige_yaw_durante_o_avanco_reto(self):
        controlador, instante = self._chegar_ao_avanco_frente()
        comando = self._passo(
            controlador, instante + 0.01, yaw=100, lateral=125, frente=300)
        self.assertEqual(controlador.state, controlador.CORRIGIR_YAW_AVANCO_FRENTE)
        self.assertEqual(comando.angle, -180)

        comando = self._passo(
            controlador, instante + 0.08, yaw=90, lateral=125, frente=300)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, instante + 0.09, yaw=90, lateral=125, frente=300)
        self.assertEqual(comando.angle, 0)


if __name__ == "__main__":
    unittest.main()
