"""Testes da confirmação ultrassônica de obstáculo."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.parada_obstaculo import (  # noqa: E402
    MonitorObstaculo,
    avancar_ate_linha,
    desviar_obstaculo,
    orientacao_continuacao_saida,
    procurar_continuacao_saida_pulsada,
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

    def lado(self, esq, dir_):
        self.comandos.append(("lado", esq, dir_))
        return True

    def refresh(self, fail_closed=False):
        self.comandos.append(("refresh", fail_closed))


class RelogioFalso:
    def __init__(self):
        self.tempo = 0.0

    def monotonic(self):
        return self.tempo

    def sleep(self, duracao):
        self.tempo += duracao


def criar_monitor(confirmacoes=2):
    return MonitorObstaculo(
        distancia_parada_mm=50,
        intervalo_s=.06,
        timeout_s=.08,
        confirmacoes=confirmacoes,
        tamanho_historico=3,
        janela_s=.20,
        distancia_minima_mm=1,
        distancia_maxima_mm=4000,
    )


class MonitorObstaculoTests(unittest.TestCase):
    def test_uma_leitura_a_oito_cm_so_bloqueia_velocidade_rapida(self):
        arduino = ArduinoFalso(((True, 80),))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.bloqueia_velocidade_rapida)
        self.assertFalse(monitor.parada_confirmada)
        self.assertEqual(monitor.leituras_concluidas, 1)
        self.assertEqual(monitor.leituras_invalidas_consecutivas, 0)
        self.assertEqual(monitor.ultima_distancia_valida_mm, 80)

    def test_acima_de_dez_cm_nao_bloqueia_velocidade_rapida(self):
        arduino = ArduinoFalso(((True, 101),))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.bloqueia_velocidade_rapida)

    def test_primeira_leitura_a_quatro_cm_desacelera_sem_parar(self):
        arduino = ArduinoFalso(((True, 40), (True, 40)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.bloqueia_velocidade_rapida)
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))

    def test_bloqueio_rapido_expira_com_a_janela(self):
        arduino = ArduinoFalso(((True, 80),))
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        self.assertTrue(monitor.bloqueia_velocidade_rapida)
        monitor.atualizar(arduino, agora=0.21)
        self.assertFalse(monitor.bloqueia_velocidade_rapida)

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

    def test_confirmacao_nao_abre_uma_medicao_extra(self):
        arduino = ArduinoFalso(((True, 48), (True, 44)))
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))

        self.assertEqual(arduino.solicitacoes, [.08])

    def test_cinco_centimetros_entram_no_limite(self):
        arduino = ArduinoFalso(((True, 50), (True, 50)))
        monitor = criar_monitor()

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))

    def test_deposito_pode_exigir_tres_leituras_proximas(self):
        arduino = ArduinoFalso(
            ((True, 49), (True, 45), (True, 48)))
        monitor = criar_monitor(confirmacoes=3)

        self.assertFalse(monitor.atualizar(arduino, agora=0.00))
        self.assertFalse(monitor.atualizar(arduino, agora=0.06))
        self.assertTrue(monitor.atualizar(arduino, agora=0.12))
        self.assertEqual(monitor.distancia_confirmada_mm, 48)

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
        self.assertEqual(monitor.leituras_invalidas_consecutivas, 1)
        self.assertFalse(monitor.atualizar(arduino, agora=0.12))
        self.assertEqual(monitor.leituras_invalidas_consecutivas, 0)
        self.assertEqual(monitor.ultima_distancia_valida_mm, 250)

    def test_tres_leituras_sem_eco_ficam_registradas(self):
        arduino = ArduinoFalso(
            ((True, None), (True, None), (True, None)))
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        monitor.atualizar(arduino, agora=0.06)
        monitor.atualizar(arduino, agora=0.12)

        self.assertEqual(monitor.leituras_concluidas, 3)
        self.assertEqual(monitor.leituras_invalidas_consecutivas, 3)
        self.assertIsNone(monitor.ultima_distancia_valida_mm)

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

    def test_reiniciar_libera_nova_deteccao(self):
        arduino = ArduinoFalso(
            ((True, 30), (True, 35), (True, 200)))
        monitor = criar_monitor()

        monitor.atualizar(arduino, agora=0.00)
        self.assertTrue(monitor.atualizar(arduino, agora=0.06))

        monitor.reiniciar()

        self.assertFalse(monitor.parada_confirmada)
        self.assertIsNone(monitor.distancia_confirmada_mm)
        self.assertEqual(monitor.leituras_concluidas, 0)
        self.assertEqual(monitor.leituras_invalidas_consecutivas, 0)
        self.assertIsNone(monitor.ultima_distancia_valida_mm)
        self.assertFalse(monitor.bloqueia_velocidade_rapida)
        self.assertFalse(monitor.atualizar(arduino, agora=1.00))


class DesvioObstaculoTests(unittest.TestCase):
    def test_lateral_avanco_e_retorno_lateral_direita(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        desviar_obstaculo(
            arduino,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(arduino.comandos[0], ("parar",))
        self.assertEqual(
            [
                comando
                for comando in arduino.comandos
                if comando[0] == "rodas"
            ],
            [
                ("rodas", -60, 60, 60, -60),
                ("rodas", 60, 60, 60, 60),
                ("rodas", 60, -60, -60, 60),
            ],
        )
        self.assertEqual([], [comando for comando in arduino.comandos
                              if comando[0] == "lado"])
        self.assertEqual(arduino.comandos[-1], ("parar",))
        self.assertAlmostEqual(relogio.tempo, 5.0)
        self.assertTrue(
            all(
                comando[0] in ("parar", "rodas", "lado", "refresh")
                for comando in arduino.comandos
            )
        )

    def test_falha_ao_enviar_rodas_ainda_termina_parado(self):
        arduino = ArduinoMovimentoFalso(rodas_enviadas=False)

        with self.assertRaises(RuntimeError):
            desviar_obstaculo(arduino)

        self.assertEqual(arduino.comandos[-1], ("parar",))

    def test_interrupcao_termina_parado_sem_esperar_duracao(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        desviar_obstaculo(
            arduino,
            deve_encerrar=lambda: True,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(arduino.comandos[-1], ("parar",))
        self.assertEqual(relogio.tempo, 0.0)
        self.assertEqual(
            [
                comando
                for comando in arduino.comandos
                if comando[0] == "rodas"
            ],
            [("rodas", -60, 60, 60, -60)],
        )

    def test_pwm_acima_do_limite_e_rejeitado_sem_mover(self):
        arduino = ArduinoMovimentoFalso()

        with self.assertRaises(ValueError):
            desviar_obstaculo(arduino, pwm_lateral=121)

        self.assertEqual(arduino.comandos, [])

    def test_avanca_ate_confirmar_linha(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        encontrou = avancar_ate_linha(
            arduino,
            linha_proxima=lambda: relogio.tempo >= .20,
            timeout_s=1.0,
            confirmacao_s=.10,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertTrue(encontrou)
        self.assertEqual(arduino.comandos[0], ("parar",))
        self.assertEqual(arduino.comandos[1], ("lado", 60, 60))
        self.assertEqual(arduino.comandos[-1], ("parar",))
        self.assertGreaterEqual(relogio.tempo, .30)

    @staticmethod
    def _resultado_continuacao(**alteracoes):
        dados = dict(
            sequencia=1,
            publicado_em=10.0,
            continuacao_saida_detectada=True,
            continuacao_saida_x=config.camera_x / 2,
            continuacao_saida_y=config.camera_y * .20,
            continuacao_saida_distancia=.70,
        )
        dados.update(alteracoes)
        return SimpleNamespace(**dados)

    def test_ponta_distante_central_confirma_continuacao_real(self):
        resultado = self._resultado_continuacao()

        self.assertEqual(
            orientacao_continuacao_saida(resultado, agora=10.05),
            "centro",
        )

    def test_ponta_distante_lateral_libera_segue_linha(self):
        resultado = self._resultado_continuacao(
            continuacao_saida_x=config.camera_x * .85,
        )

        self.assertEqual(
            orientacao_continuacao_saida(resultado, agora=10.05),
            "direita",
        )

    def test_ponta_distante_a_esquerda_libera_segue_linha(self):
        resultado = self._resultado_continuacao(
            continuacao_saida_x=config.camera_x * .15,
        )

        self.assertEqual(
            orientacao_continuacao_saida(resultado, agora=10.05),
            "esquerda",
        )

    def test_ausencia_de_alvo_nao_libera_segue_linha(self):
        resultado = self._resultado_continuacao(
            continuacao_saida_detectada=False,
        )

        self.assertIsNone(
            orientacao_continuacao_saida(resultado, agora=10.05))

    def test_resultado_antigo_nao_inicia_segue_linha(self):
        resultado = self._resultado_continuacao(publicado_em=9.0)

        self.assertIsNone(
            orientacao_continuacao_saida(resultado, agora=10.0))

    def test_configura_busca_pulsada_pos_resgate(self):
        self.assertEqual(config.EXIT_LINE_PULSES_LEFT, 2)
        self.assertEqual(config.EXIT_LINE_PULSES_RIGHT, 4)
        self.assertEqual(config.EXIT_LINE_PULSE_S, .40)

    def test_saida_com_continuacao_no_centro_mapeia_e_retorna_ao_ramo(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        lado = procurar_continuacao_saida_pulsada(
            arduino,
            orientacao_ramificacao=lambda: "centro",
            pwm=60,
            duracao_pulso_s=.05,
            pausa_assentamento_s=.02,
            observacao_s=.05,
            confirmacao_s=.025,
            pulsos_esquerda=2,
            pulsos_direita=4,
            re_inicial_s=.05,
            avanco_tentativa_s=.05,
            re_final_s=.05,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(lado, "centro")
        comandos_tanque = [
            comando for comando in arduino.comandos
            if comando == ("rodas", -60, -60, 60, 60)
            or comando == ("rodas", 60, 60, -60, -60)
        ]
        self.assertGreaterEqual(len(comandos_tanque), 6)
        self.assertEqual(arduino.comandos[-1], ("parar",))

    def test_segundo_ciclo_aceita_ramificacao_valida_vista_de_lado(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        lado = procurar_continuacao_saida_pulsada(
            arduino,
            orientacao_ramificacao=lambda: "direita",
            pwm=60,
            duracao_pulso_s=.05,
            pausa_assentamento_s=.02,
            observacao_s=.05,
            confirmacao_s=.025,
            pulsos_esquerda=2,
            pulsos_direita=4,
            re_inicial_s=.05,
            avanco_tentativa_s=.05,
            re_final_s=.05,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(lado, "centro")
        self.assertIn(("rodas", -60, -60, 60, 60), arduino.comandos)
        self.assertIn(("rodas", 60, 60, -60, -60), arduino.comandos)
        self.assertIn(("rodas", -60, -60, -60, -60), arduino.comandos)
        self.assertEqual(arduino.comandos[-1], ("parar",))

    def test_dois_ciclos_sem_ramificacao_executam_re_maior(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        lado = procurar_continuacao_saida_pulsada(
            arduino,
            orientacao_ramificacao=lambda: None,
            pwm=60,
            duracao_pulso_s=.05,
            pausa_assentamento_s=.02,
            observacao_s=.05,
            confirmacao_s=.025,
            pulsos_esquerda=2,
            pulsos_direita=4,
            re_inicial_s=.05,
            avanco_tentativa_s=.05,
            re_final_s=.10,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertIsNone(lado)
        self.assertGreaterEqual(
            arduino.comandos.count(("rodas", -60, -60, -60, -60)), 2)
        self.assertEqual(arduino.comandos[-1], ("parar",))

    def test_saida_mapeia_direita_esquerda_e_volta_para_o_meio(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()
        leituras = 0

        def orientacao_mapeada():
            nonlocal leituras
            leituras += 1
            # Tres leituras por parada em cada uma das duas varreduras: a
            # ponta passa pela direita e depois pela esquerda. So a leitura
            # final, ja no rumo calculado, aparece no centro.
            if leituras > 36:
                return "centro"
            indice_no_ciclo = (leituras - 1) % 18
            if indice_no_ciclo < 9:
                return "direita"
            return "esquerda"

        lado = procurar_continuacao_saida_pulsada(
            arduino,
            orientacao_ramificacao=orientacao_mapeada,
            pwm=60,
            duracao_pulso_s=.05,
            pausa_assentamento_s=.02,
            observacao_s=.05,
            confirmacao_s=.025,
            pulsos_esquerda=2,
            pulsos_direita=4,
            re_inicial_s=.05,
            avanco_tentativa_s=.05,
            re_final_s=.05,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertEqual(lado, "centro")
        # Alem dos dois pulsos iniciais a esquerda, o retorno de p2 para o
        # meio entre p-1 e p0 requer dois pulsos inteiros e meio a esquerda.
        self.assertGreaterEqual(
            arduino.comandos.count(("rodas", -60, -60, 60, 60)), 5)

    def test_busca_para_no_timeout_sem_linha(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        encontrou = avancar_ate_linha(
            arduino,
            linha_proxima=lambda: False,
            timeout_s=.20,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertFalse(encontrou)
        self.assertEqual(arduino.comandos[-1], ("parar",))
        self.assertAlmostEqual(relogio.tempo, .20)

    def test_um_pulso_visual_nao_confirma_a_linha(self):
        arduino = ArduinoMovimentoFalso()
        relogio = RelogioFalso()

        encontrou = avancar_ate_linha(
            arduino,
            linha_proxima=lambda: .10 <= relogio.tempo < .15,
            timeout_s=.30,
            confirmacao_s=.10,
            relogio=relogio.monotonic,
            dormir=relogio.sleep,
        )

        self.assertFalse(encontrou)
        self.assertEqual(arduino.comandos[-1], ("parar",))

if __name__ == "__main__":
    unittest.main()
