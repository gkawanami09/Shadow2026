"""Testes da maquina de estados da saida pela parede direita."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.saida_parede_resgate import (  # noqa: E402
    ControladorSaidaParede,
)


NAO_PRETA = "nao_preta"


class SaidaParedeResgateTests(unittest.TestCase):
    @staticmethod
    def _passo(
        controlador,
        instante,
        *,
        yaw,
        frente=450,
        lateral=120,
        aceitar=True,
    ):
        controlador.observar_mpu(yaw, instante)
        controlador.observar_ultrassom("FRENTE", frente, True, instante)
        controlador.observar_ultrassom("LATERAL", lateral, True, instante)
        comando = controlador.atualizar(instante)
        if aceitar and not comando.terminal:
            controlador.notificar_comando_escrito(comando.state, instante)
        return comando

    def _chegar_a_parede_direita(self, yaw_direita=90, lateral_inicial=120):
        controlador = ControladorSaidaParede(start_time=0.0)
        self.assertTrue(controlador.solicita_zerar_mpu)
        self.assertTrue(controlador.confirmar_mpu_zerado(True, 0.0))

        self._passo(controlador, 0.0, yaw=0, lateral=lateral_inicial)
        self._passo(
            controlador,
            cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_S + 0.01,
            yaw=0,
            lateral=lateral_inicial,
        )
        instante_giro = (
            cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_S
            + cfg.SAIDA_PAREDE_ASSENTAMENTO_S
            + 0.02
        )
        comando = self._passo(
            controlador,
            instante_giro,
            yaw=0,
            lateral=lateral_inicial,
        )
        self.assertEqual(controlador.state, controlador.GIRO_INICIAL_DIREITA)
        self.assertEqual(comando.angle, 180)

        # Uma pequena mudanca revela o sinal real do gyro no giro fisico a
        # direita. O teste tambem cobre a montagem mais comum: yaw crescente.
        sinal = 1 if yaw_direita == 90 else -1
        comando = self._passo(
            controlador,
            instante_giro + 0.03,
            yaw=6 * sinal,
            lateral=lateral_inicial,
        )
        self.assertEqual(comando.angle, 180)
        comando = self._passo(
            controlador,
            instante_giro + 0.40,
            yaw=yaw_direita,
            lateral=lateral_inicial,
        )
        self.assertEqual(controlador.state, controlador.SEGUIR_PAREDE)
        self.assertEqual(comando.angle, 0)
        self.assertEqual(controlador.heading_parede, float(yaw_direita))
        return controlador, instante_giro + 0.40

    def test_giro_inicial_calibra_yaw_invertido_sem_inverter_direita_fisica(self):
        controlador, _ = self._chegar_a_parede_direita(yaw_direita=270)

        self.assertEqual(controlador.heading_parede, 270.0)

    def test_triangulo_desvia_45_e_retorna_ao_heading_da_parede(self):
        controlador, instante = self._chegar_a_parede_direita()
        instante += cfg.SAIDA_PAREDE_COOLDOWN_TRIANGULO_S + 0.01
        controlador.observar_triangulo(True, instante)
        comando = self._passo(controlador, instante, yaw=90)
        self.assertEqual(controlador.state, controlador.PARAR_TRIANGULO)
        self.assertEqual(comando.angle, 190)

        instante += cfg.SAIDA_PAREDE_ASSENTAMENTO_S + 0.01
        comando = self._passo(controlador, instante, yaw=90)
        self.assertEqual(controlador.state, controlador.DESVIAR_TRIANGULO)
        self.assertEqual(comando.angle, -180)

        instante += 0.10
        controlador.observar_triangulo(False, instante)
        comando = self._passo(controlador, instante, yaw=45)
        self.assertEqual(controlador.state, controlador.PASSAR_TRIANGULO)
        self.assertEqual(comando.angle, 0)

        instante += cfg.SAIDA_PAREDE_AVANCO_MIN_TRIANGULO_S + 0.01
        comando = self._passo(controlador, instante, yaw=45)
        self.assertEqual(controlador.state, controlador.RETORNAR_TRIANGULO)
        self.assertEqual(comando.angle, 180)

        instante += 0.10
        comando = self._passo(controlador, instante, yaw=90)
        self.assertEqual(controlador.state, controlador.SEGUIR_PAREDE)
        self.assertEqual(comando.angle, 0)

    def test_parede_frontal_vira_a_esquerda_e_mantem_parede_a_direita(self):
        controlador, instante = self._chegar_a_parede_direita()
        self._passo(controlador, instante + 0.02, yaw=90, frente=100)
        instante += 0.04
        comando = self._passo(controlador, instante, yaw=90, frente=100)
        self.assertEqual(controlador.state, controlador.PARAR_PAREDE)
        self.assertEqual(comando.angle, 190)

        instante += cfg.SAIDA_PAREDE_ASSENTAMENTO_S + 0.01
        comando = self._passo(controlador, instante, yaw=90, frente=100)
        self.assertEqual(controlador.state, controlador.GIRO_PAREDE_ESQUERDA)
        self.assertEqual(comando.angle, -180)

        instante += 0.10
        comando = self._passo(controlador, instante, yaw=0, frente=450)
        self.assertEqual(controlador.state, controlador.SEGUIR_PAREDE)
        self.assertEqual(comando.angle, 0)

    def test_espaco_aberto_inicial_nao_e_saida_antes_de_ver_parede(self):
        controlador, instante = self._chegar_a_parede_direita(
            lateral_inicial=300)

        for _ in range(cfg.SAIDA_PAREDE_CONFIRMACOES_ABERTURA + 1):
            instante += 0.02
            comando = self._passo(controlador, instante, yaw=90, lateral=300)

        self.assertEqual(controlador.state, controlador.SEGUIR_PAREDE)
        self.assertEqual(comando.angle, 0)

    def test_abertura_sem_preto_desfaz_manobra_e_nao_retesta_o_mesmo_vao(self):
        controlador, instante = self._chegar_a_parede_direita()

        for _ in range(cfg.SAIDA_PAREDE_CONFIRMACOES_ABERTURA):
            instante += 0.02
            comando = self._passo(controlador, instante, yaw=90, lateral=300)
        self.assertEqual(controlador.state, controlador.PARAR_ABERTURA)
        self.assertEqual(comando.angle, 190)

        instante += cfg.SAIDA_PAREDE_ASSENTAMENTO_S + 0.01
        comando = self._passo(controlador, instante, yaw=90, lateral=300)
        self.assertEqual(controlador.state, controlador.AVANCAR_ENTRADA)
        self.assertEqual(comando.angle, 0)

        instante += cfg.SAIDA_PAREDE_AVANCO_ENTRADA_S + 0.01
        comando = self._passo(controlador, instante, yaw=90, lateral=300)
        self.assertEqual(controlador.state, controlador.TRANSLADAR_ESQUERDA)
        self.assertEqual(comando.wheel_speeds, (-45, 45, 45, -45))

        instante += cfg.SAIDA_PAREDE_TRANSLACAO_ESQUERDA_S + 0.01
        comando = self._passo(controlador, instante, yaw=90, lateral=300)
        self.assertEqual(controlador.state, controlador.GIRO_ENTRADA_DIREITA)
        self.assertEqual(comando.angle, 180)

        instante += 0.10
        comando = self._passo(controlador, instante, yaw=180, lateral=300)
        self.assertEqual(controlador.state, controlador.PRONTO_SONDA_LINHA)
        self.assertEqual(comando.angle, 190)
        self.assertTrue(controlador.solicita_sonda_linha)
        self.assertTrue(controlador.iniciar_sonda_linha())
        self.assertTrue(
            controlador.registrar_resultado_sonda(NAO_PRETA, 0.20, instante))

        # O primeiro comando de re apos a troca de camera inicia o relogio
        # do movimento; o recuo medido so passa a contar depois dele.
        instante += 0.01
        self._passo(controlador, instante, yaw=180, lateral=300)
        instante += 0.21
        comando = self._passo(controlador, instante, yaw=180, lateral=300)
        self.assertEqual(controlador.state, controlador.GIRO_RETORNO_ESQUERDA)
        self.assertEqual(comando.angle, -180)

        instante += 0.10
        comando = self._passo(controlador, instante, yaw=90, lateral=300)
        self.assertEqual(controlador.state, controlador.TRANSLADAR_DIREITA)
        self.assertEqual(comando.wheel_speeds, (45, -45, -45, 45))

        instante += cfg.SAIDA_PAREDE_TRANSLACAO_ESQUERDA_S + 0.01
        comando = self._passo(controlador, instante, yaw=90, lateral=300)
        self.assertEqual(controlador.state, controlador.ABRIR_CAMERA_FRONTAL)
        self.assertEqual(comando.angle, 190)
        self.assertTrue(controlador.solicita_camera_frontal)
        self.assertTrue(controlador.confirmar_camera_frontal_aberta(True, instante))

        for _ in range(cfg.SAIDA_PAREDE_CONFIRMACOES_PAREDE):
            instante += 0.02
            comando = self._passo(controlador, instante, yaw=90, lateral=120)
        self.assertEqual(controlador.state, controlador.SEGUIR_PAREDE)
        self.assertEqual(comando.angle, 0)


if __name__ == "__main__":
    unittest.main()
