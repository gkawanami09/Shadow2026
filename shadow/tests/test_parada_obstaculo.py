"""Testes da confirmação ultrassônica de obstáculo."""

import sys
from pathlib import Path
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.parada_obstaculo import MonitorObstaculo  # noqa: E402


class ArduinoFalso:
    def __init__(self, respostas=()):
        self.respostas = list(respostas)
        self.solicitacoes = []

    def poll_ultrassom(self):
        if self.respostas:
            return self.respostas.pop(0)
        return False, None

    def iniciar_ultrassom(self, timeout):
        self.solicitacoes.append(timeout)
        return True


def criar_monitor():
    return MonitorObstaculo(
        distancia_parada_mm=100,
        intervalo_s=.06,
        timeout_s=.08,
        confirmacoes=2,
        tamanho_historico=3,
        janela_s=.20,
        distancia_minima_mm=1,
        distancia_maxima_mm=4000,
    )


class MonitorObstaculoTests(unittest.TestCase):
    def test_uma_leitura_proxima_isolada_nao_para(self):
        arduino = ArduinoFalso(((True, 90), (True, 300), (True, 250)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertFalse(monitor.atualizar(arduino, agora=0.12))

    def test_duas_leituras_proximas_param_rapidamente(self):
        arduino = ArduinoFalso(((True, 98), (True, 94)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))
        self.assertEqual(monitor.distancia_confirmada_mm, 96)

    def test_dez_centimetros_entram_no_limite(self):
        arduino = ArduinoFalso(((True, 100), (True, 100)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))

    def test_acima_de_dez_centimetros_nao_para(self):
        arduino = ArduinoFalso(((True, 101), (True, 101), (True, 101)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertFalse(monitor.atualizar(arduino, agora=0.12))

    def test_duas_de_tres_toleram_um_eco_ruidoso(self):
        arduino = ArduinoFalso(
            ((True, 95), (True, 180), (True, 92)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertTrue(monitor.atualizar(arduino, agora=0.12))

    def test_leitura_antiga_nao_confirma_obstaculo_novo(self):
        arduino = ArduinoFalso(((True, 90), (True, 95)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.25))

    def test_sem_eco_nao_conta_como_obstaculo(self):
        arduino = ArduinoFalso(
            ((True, 90), (True, None), (True, 250)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertFalse(monitor.atualizar(arduino, agora=0.12))

    def test_nao_solicita_mais_rapido_que_sessenta_ms(self):
        arduino = ArduinoFalso()
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        monitor.atualizar(arduino, agora=0.03)
        monitor.atualizar(arduino, agora=0.06)

        self.assertEqual(arduino.solicitacoes, [.08, .08])

    def test_parada_permanece_travada_sem_nova_leitura(self):
        arduino = ArduinoFalso(((True, 80), (True, 85)))
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))
        solicitacoes_antes = len(arduino.solicitacoes)

        self.assertTrue(monitor.atualizar(arduino, agora=2.00))
        self.assertEqual(len(arduino.solicitacoes), solicitacoes_antes)


if __name__ == "__main__":
    unittest.main()
