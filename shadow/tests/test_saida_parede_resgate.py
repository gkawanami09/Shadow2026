"""Testes da manobra curta pos-vermelho: girar e alinhar na parede."""

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
        enviar_yaw=True,
        enviar_lateral=True,
        respondeu_lateral=True,
    ):
        if enviar_yaw:
            controlador.observar_mpu(yaw, instante)
        if enviar_lateral:
            controlador.observar_ultrassom(
                "LATERAL", lateral, respondeu_lateral, instante)
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

    def test_alinha_para_direita_e_para_ao_chegar_na_faixa(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(controlador, instante + 0.01, yaw=90, lateral=200)
        self.assertEqual(comando.wheel_speeds, (45, -45, -45, 45))
        self.assertEqual(controlador.state, controlador.ALINHAR_DIREITA)

        comando = self._passo(controlador, instante + 0.08, yaw=90, lateral=130)
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.ALINHADO)
        self.assertEqual(comando.speed, 0.0)
        self.assertIsNone(comando.wheel_speeds)

    def test_alinha_para_esquerda_se_a_parede_ja_estiver_proxima(self):
        controlador, instante = self._chegar_ao_alinhamento()
        comando = self._passo(controlador, instante + 0.01, yaw=90, lateral=90)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))

        comando = self._passo(controlador, instante + 0.08, yaw=90, lateral=120)
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.ALINHADO)

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

    def test_estado_alinhado_permanece_parado(self):
        controlador, instante = self._chegar_ao_alinhamento()
        self._passo(controlador, instante + 0.01, yaw=90, lateral=125)
        comando = self._passo(controlador, instante + 1.0, yaw=90, lateral=300)
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.ALINHADO)
        self.assertEqual(comando.speed, 0.0)
        self.assertIsNone(comando.wheel_speeds)


if __name__ == "__main__":
    unittest.main()
