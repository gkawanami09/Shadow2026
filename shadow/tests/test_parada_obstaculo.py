"""Testes da confirmação ultrassônica de obstáculo."""

import sys
from pathlib import Path
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.parada_obstaculo import (  # noqa: E402
    MonitorObstaculo,
    deslizar_para_esquerda,
)


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


class ArduinoMovimentoFalso:
    def __init__(self, rodas_enviadas=True):
        self.comandos = []
        self.rodas_enviadas = rodas_enviadas
        self.connected = True
        self.connection_epoch = 1

    def parar(self):
        self.comandos.append(("parar",))
        return True

    def rodas(self, fe, te, fd, td):
        self.comandos.append(("rodas", fe, te, fd, td))
        return self.rodas_enviadas

    def refresh(self, fail_closed=False):
        self.comandos.append(("refresh", fail_closed))


class RelogioFalso:
    def __init__(self):
        self.tempo = 0.0

    def monotonic(self):
        return self.tempo

    def sleep(self, duracao):
        self.tempo += duracao


def criar_monitor():
    return MonitorObstaculo(
        distancia_parada_mm=50,
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
        arduino = ArduinoFalso(((True, 40), (True, 300), (True, 250)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertFalse(monitor.atualizar(arduino, agora=0.12))

    def test_duas_leituras_proximas_param_rapidamente(self):
        arduino = ArduinoFalso(((True, 48), (True, 44)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))
        self.assertEqual(monitor.distancia_confirmada_mm, 46)

    def test_cinco_centimetros_entram_no_limite(self):
        arduino = ArduinoFalso(((True, 50), (True, 50)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))

    def test_acima_de_cinco_centimetros_nao_para(self):
        arduino = ArduinoFalso(((True, 51), (True, 51), (True, 51)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertFalse(monitor.atualizar(arduino, agora=0.12))

    def test_duas_de_tres_toleram_um_eco_ruidoso(self):
        arduino = ArduinoFalso(
            ((True, 45), (True, 180), (True, 42)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertTrue(monitor.atualizar(arduino, agora=0.12))

    def test_leitura_antiga_nao_confirma_obstaculo_novo(self):
        arduino = ArduinoFalso(((True, 40), (True, 45)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.25))

    def test_sem_eco_nao_conta_como_obstaculo(self):
        arduino = ArduinoFalso(
            ((True, 40), (True, None), (True, 250)))
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
        arduino = ArduinoFalso(((True, 30), (True, 35)))
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))
        solicitacoes_antes = len(arduino.solicitacoes)

        self.assertTrue(monitor.atualizar(arduino, agora=2.00))
        self.assertEqual(len(arduino.solicitacoes), solicitacoes_antes)


class DesvioLateralTests(unittest.TestCase):
    def test_desliza_para_esquerda_por_dois_e_meio_segundos(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        deslizar_para_esquerda(
            arduino,
            pwm=60,
            duracao_s=2.5,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(arduino.comandos[0], ("parar",))
        self.assertEqual(
            arduino.comandos[1],
            ("rodas", -60, 60, 60, -60),
        )
        self.assertEqual(arduino.comandos[-1], ("parar",))
        self.assertAlmostEqual(relogio.tempo, 2.5)
        self.assertTrue(
            all(
                comando[0] in ("parar", "rodas", "refresh")
                for comando in arduino.comandos
            )
        )

    def test_falha_ao_enviar_rodas_ainda_termina_parado(self):
        arduino = ArduinoMovimentoFalso(rodas_enviadas=False)

        with self.assertRaises(RuntimeError):
            deslizar_para_esquerda(arduino)

        self.assertEqual(arduino.comandos[-1], ("parar",))

    def test_interrupcao_termina_parado_sem_esperar_duracao(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        deslizar_para_esquerda(
            arduino,
            deve_encerrar=lambda: True,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(arduino.comandos[-1], ("parar",))
        self.assertEqual(relogio.tempo, 0.0)

    def test_pwm_acima_do_limite_e_rejeitado_sem_mover(self):
        arduino = ArduinoMovimentoFalso()

        with self.assertRaises(ValueError):
            deslizar_para_esquerda(arduino, pwm=121)

        self.assertEqual(arduino.comandos, [])


if __name__ == "__main__":
    unittest.main()
