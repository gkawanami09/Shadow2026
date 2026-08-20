"""Testes da rota de parede executada depois do deposito vermelho."""

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
        yaw=90,
        lateral=110,
        frente=400,
        enviar_yaw=True,
        enviar_lateral=True,
        enviar_frente=True,
    ):
        if enviar_yaw:
            controlador.observar_mpu(yaw, instante)
        if enviar_lateral:
            controlador.observar_ultrassom("LATERAL", lateral, True, instante)
        if enviar_frente:
            controlador.observar_ultrassom("FRENTE", frente, True, instante)
        comando = controlador.atualizar(instante)
        if not comando.terminal:
            controlador.notificar_comando_escrito(comando.state, instante)
        return comando

    def _chegar_ao_primeiro_avanco(self, yaw_final=90):
        """Executa apenas vermelho -> 90 graus -> espera o frontal novo."""
        controlador = ControladorSaidaParede(start_time=0.0)
        self.assertTrue(controlador.confirmar_mpu_zerado(True, 0.0))

        comando = self._passo(controlador, 0.0, yaw=0)
        self.assertEqual(controlador.state, controlador.AFASTAR_VERMELHO)
        self.assertEqual(comando.angle, 0)

        instante = cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_S + 0.01
        comando = self._passo(controlador, instante, yaw=0)
        self.assertEqual(controlador.state, controlador.ASSENTAR_INICIAL)
        self.assertEqual(comando.speed, 0.0)

        instante += cfg.SAIDA_PAREDE_ASSENTAMENTO_S + 0.01
        comando = self._passo(controlador, instante, yaw=0)
        self.assertEqual(controlador.state, controlador.GIRO_INICIAL_DIREITA)
        self.assertEqual(comando.angle, 180)

        sinal = 1 if yaw_final == 90 else -1
        self._passo(controlador, instante + 0.03, yaw=6 * sinal)
        comando = self._passo(controlador, instante + 0.40, yaw=yaw_final)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)
        self.assertEqual(comando.speed, 0.0)
        self.assertEqual(controlador.heading_parede, float(yaw_final))
        return controlador, instante + 0.40

    def _executar_passagem_ate_afastamento(self, controlador, inicio, destino):
        """Frente 118 -> pivo -> direita -> esquerda 120."""
        comando = self._passo(
            controlador, inicio + 0.01, yaw=90, lateral=110, frente=118)
        self.assertEqual(controlador.state, controlador.PIVO_TRASEIRO_ESTABILIZAR)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, inicio + 0.02, yaw=90, lateral=110, frente=118)
        self.assertTrue(comando.pivo_traseiro)
        self.assertEqual(comando.angle, cfg.SAIDA_PAREDE_ANGULO_CURVA_TRASEIRA)

        comando = self._passo(
            controlador, inicio + 0.60, yaw=90, lateral=110, frente=118)
        self.assertEqual(controlador.state, controlador.TRANSLADAR_DIREITA)
        self.assertEqual(
            comando.wheel_speeds,
            (
                cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                -cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                -cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
            ),
        )

        comando = self._passo(
            controlador,
            inicio + 0.60 + cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.01,
            yaw=90,
            lateral=110,
            frente=118,
        )
        self.assertEqual(controlador.state, controlador.AFASTAR_ESQUERDA_120)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador,
            inicio + 0.60 + cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.02,
            yaw=90,
            lateral=119,
            frente=118,
        )
        self.assertEqual(
            comando.wheel_speeds,
            (
                -cfg.SAIDA_PAREDE_PWM_TRANSLACAO_ESQUERDA,
                cfg.SAIDA_PAREDE_PWM_TRANSLACAO_ESQUERDA,
                cfg.SAIDA_PAREDE_PWM_TRANSLACAO_ESQUERDA,
                -cfg.SAIDA_PAREDE_PWM_TRANSLACAO_ESQUERDA,
            ),
        )
        self.assertIn("yaw monitorado", comando.detail)

        comando = self._passo(
            controlador,
            inicio + 0.60 + cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.09,
            yaw=90,
            lateral=120,
            frente=118,
        )
        self.assertEqual(controlador.state, destino)
        return inicio + 0.60 + cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.09, comando

    def test_apos_90_graus_avanca_sem_alinhamento_lateral(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        self.assertEqual(controlador.lado_ultrassom_atual, "FRENTE")

        comando = self._passo(
            controlador, instante + 0.01, yaw=90, lateral=40, frente=300)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_AVANCO_ATE_FRENTE_PWM / 120.0,
        )

    def test_giro_inicial_funciona_com_yaw_invertido(self):
        controlador, _ = self._chegar_ao_primeiro_avanco(yaw_final=270)
        self.assertEqual(controlador.heading_parede, 270.0)

    def test_camera_so_abre_depois_da_primeira_passagem_com_lateral_120(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        _, comando = self._executar_passagem_ate_afastamento(
            controlador,
            instante,
            controlador.VERIFICAR_TRIANGULO_VERDE,
        )
        self.assertTrue(controlador.usa_camera_triangulo_verde)
        self.assertEqual(comando.speed, 0.0)

    def test_verde_retomar_reto_sem_giro_de_45_graus(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        instante, _ = self._executar_passagem_ate_afastamento(
            controlador,
            instante,
            controlador.VERIFICAR_TRIANGULO_VERDE,
        )

        self.assertTrue(controlador.observar_triangulo_verde(True, instante + 0.01))
        comando = self._passo(
            controlador, instante + 0.01, yaw=90, lateral=120, frente=118)
        self.assertEqual(controlador.state, controlador.AGUARDAR_MPU_TRIANGULO_VERDE)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, instante + 0.02, yaw=90, lateral=120, frente=300)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, instante + 0.03, yaw=90, lateral=120, frente=300)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(
            comando.speed,
            cfg.SAIDA_PAREDE_AVANCO_ATE_FRENTE_PWM / 120.0,
        )
        self.assertEqual(controlador.heading_parede, 90.0)

    def test_verde_executa_mais_duas_passagens_e_para_apos_120_mm(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        instante, _ = self._executar_passagem_ate_afastamento(
            controlador,
            instante,
            controlador.VERIFICAR_TRIANGULO_VERDE,
        )

        controlador.observar_triangulo_verde(True, instante + 0.01)
        self._passo(controlador, instante + 0.01, yaw=90, lateral=120, frente=118)
        self._passo(controlador, instante + 0.02, yaw=90, lateral=120, frente=300)
        self.assertEqual(controlador.state, controlador.AVANCAR_ATE_PAREDE_FRENTE)

        instante, _ = self._executar_passagem_ate_afastamento(
            controlador,
            instante + 0.02,
            controlador.AVANCAR_ATE_PAREDE_FRENTE,
        )
        instante, comando = self._executar_passagem_ate_afastamento(
            controlador,
            instante,
            controlador.SAIDA_CONCLUIDA,
        )
        self.assertTrue(comando.terminal)
        self.assertEqual(comando.speed, 0.0)
        self.assertTrue(controlador.terminal)

    def test_sem_verde_para_em_falha_sem_mover(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        instante, _ = self._executar_passagem_ate_afastamento(
            controlador,
            instante,
            controlador.VERIFICAR_TRIANGULO_VERDE,
        )
        comando = controlador.atualizar(
            instante + cfg.SAIDA_PAREDE_TIMEOUT_TRIANGULO_VERDE_S + 0.01)
        self.assertTrue(comando.terminal)
        self.assertEqual(controlador.state, controlador.FALHA)
        self.assertIn("camera frontal", comando.detail)

    def test_afastamento_corrige_yaw_antes_de_continuar_a_esquerda(self):
        controlador = ControladorSaidaParede(start_time=0.0)
        controlador._heading_parede = 90.0
        controlador._sinal_yaw_por_giro_direita = 1.0
        controlador.observar_ultrassom("LATERAL", 110, True, 0.0)
        controlador._entrar_afastamento_esquerda(
            0.0,
            controlador._DESTINO_CAMERA,
        )
        controlador.notificar_comando_escrito(controlador.state, 0.0)

        comando = self._passo(
            controlador, 0.01, yaw=100, lateral=115, frente=118)
        self.assertEqual(
            controlador.state,
            controlador.CORRIGIR_YAW_AFASTAMENTO_ESQUERDA,
        )
        self.assertEqual(comando.angle, -180)

        comando = self._passo(
            controlador, 0.08, yaw=90, lateral=115, frente=118)
        self.assertEqual(controlador.state, controlador.AFASTAR_ESQUERDA_120)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador, 0.09, yaw=90, lateral=115, frente=118)
        self.assertEqual(
            comando.wheel_speeds,
            (-45, 45, 45, -45),
        )

    def test_pivo_atualiza_heading_para_nao_girar_antes_da_camera(self):
        controlador = ControladorSaidaParede(start_time=0.0)
        controlador._heading_parede = 271.7
        controlador._sinal_yaw_por_giro_direita = 1.0
        controlador.observar_mpu(315.6, 0.0)
        controlador.observar_ultrassom("LATERAL", 110, True, 0.0)
        controlador._entrar_transladar_direita(0.0)
        self.assertEqual(controlador.heading_parede, 315.6)
        controlador.notificar_comando_escrito(controlador.state, 0.0)

        comando = self._passo(
            controlador,
            cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.01,
            yaw=315.6,
            lateral=110,
            frente=118,
        )
        self.assertEqual(controlador.state, controlador.AFASTAR_ESQUERDA_120)
        self.assertEqual(comando.speed, 0.0)

        comando = self._passo(
            controlador,
            cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S + 0.02,
            yaw=315.6,
            lateral=119,
            frente=118,
        )
        self.assertEqual(controlador.state, controlador.AFASTAR_ESQUERDA_120)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))

    def test_pivo_usa_timeout_de_dois_segundos_e_ainda_translada_direita(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        self._passo(controlador, instante + 0.01, yaw=90, lateral=110, frente=118)
        self._passo(controlador, instante + 0.02, yaw=90, lateral=110, frente=118)
        comando = self._passo(
            controlador,
            instante + 0.02 + cfg.SAIDA_PAREDE_TIMEOUT_PIVO_TRASEIRO_S + 0.01,
            yaw=90,
            lateral=130,
            frente=118,
        )
        self.assertEqual(controlador.state, controlador.TRANSLADAR_DIREITA)
        self.assertEqual(
            comando.wheel_speeds,
            (100, -100, -100, 100),
        )
        self.assertIn("mesmo sem estabilizar", comando.detail)

    def test_para_se_ultrassom_frontal_responder_sem_eco(self):
        controlador, instante = self._chegar_ao_primeiro_avanco()
        comando = self._passo(
            controlador,
            instante + 0.01,
            yaw=90,
            lateral=110,
            frente=None,
        )
        self.assertTrue(comando.terminal)
        self.assertIn("sem eco", comando.detail)


if __name__ == "__main__":
    unittest.main()
