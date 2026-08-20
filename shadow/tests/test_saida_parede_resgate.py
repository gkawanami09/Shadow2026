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

    def test_apoia_a_frente_e_para_so_apos_um_segundo_estavel(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(controlador, instante + 0.01, yaw=90, lateral=200)
        self.assertEqual(comando.wheel_speeds, (45, -45, -45, 45))
        self.assertEqual(controlador.state, controlador.ALINHAR_DIREITA)

        comando = self._passo(
            controlador, instante + 0.08, yaw=90, lateral=130, frente=400)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)
        self.assertEqual(comando.speed, 0.0)
        self.assertIsNone(comando.wheel_speeds)

        comando = self._passo(
            controlador, instante + 0.09, yaw=90, lateral=130, frente=300)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_AVANCO_ATE_FRENTE_PWM / 120.0,
        )

        comando = self._passo(
            controlador, instante + 0.16, yaw=90, lateral=130, frente=118)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.PIVO_TRASEIRO_ESTABILIZAR)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, instante + 0.30, yaw=90, lateral=130, frente=116)
        self.assertFalse(comando.terminal)
        self.assertEqual(comando.angle, -180)
        self.assertTrue(comando.pivo_traseiro)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_PWM_PIVO_TRASEIRO / 120.0,
        )

        comando = self._passo(
            controlador, instante + 0.82, yaw=90, lateral=130, frente=120)
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador, instante + 1.34, yaw=90, lateral=130, frente=118)
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.PAREDE_FRENTE_ESTAVEL)
        self.assertEqual(comando.speed, 0.0)

    def test_reinicia_o_segundo_estavel_quando_o_frontal_variar_demais(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=400)
        self._passo(
            controlador, instante + 0.08, yaw=90, lateral=125, frente=118)
        comando = self._passo(
            controlador, instante + 0.20, yaw=90, lateral=125, frente=116)
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador,
            instante + 0.55,
            yaw=90,
            lateral=125,
            frente=116 + cfg.SAIDA_PAREDE_TOLERANCIA_ESTABILIDADE_FRENTE_MM + 1,
        )
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador, instante + 1.15, yaw=90, lateral=125, frente=122)
        self.assertFalse(comando.terminal)
        comando = self._passo(
            controlador, instante + 1.61, yaw=90, lateral=125, frente=123)
        self.assertTrue(comando.terminal)

    def test_falha_se_o_frontal_nao_der_leitura_nova_para_o_pivo(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=400)
        self._passo(
            controlador, instante + 0.08, yaw=90, lateral=125, frente=118)
        comando = self._passo(
            controlador,
            instante + 0.08 + cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S + 0.01,
            yaw=90,
            lateral=125,
            enviar_frente=False,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("pivo traseiro", comando.detail)

    def test_alinha_para_esquerda_se_a_parede_ja_estiver_proxima(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(controlador, instante + 0.01, yaw=90, lateral=90)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))

        comando = self._passo(
            controlador, instante + 0.08, yaw=90, lateral=120, frente=400)
        self.assertFalse(comando.terminal)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)

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
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=400)
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
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=400)
        comando = self._passo(
            controlador, instante + 0.02, yaw=90, lateral=125, frente=None)

        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("sem eco", comando.detail)

    def test_para_em_falha_se_nao_chegar_na_parede_frontal_no_prazo(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=400)
        self._passo(
            controlador, instante + 0.02, yaw=90, lateral=125, frente=300)
        comando = self._passo(
            controlador,
            instante + 0.01 + cfg.SAIDA_PAREDE_TIMEOUT_AVANCO_FRENTE_S + 0.01,
            yaw=90,
            lateral=125,
            frente=300,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("timeout", comando.detail)

    def test_corrige_yaw_durante_o_avanco_reto(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(
            controlador, instante + 0.01, yaw=90, lateral=125, frente=400)
        comando = self._passo(
            controlador, instante + 0.02, yaw=100, lateral=125, frente=300)
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
