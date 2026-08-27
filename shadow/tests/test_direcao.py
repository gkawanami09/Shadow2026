"""Testes unitarios do mapeamento de angulo para os motores."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle import direcao  # noqa: E402
from controle.estado_verde import (GreenDecision, GreenManeuverFSM,
                                    GreenObservation)  # noqa: E402
from config import LINE_FOLLOW_SPEED, LINE_FOLLOW_PWM  # noqa: E402


class ArduinoFalso:
    def __init__(self):
        self.chamadas = []

    def lado(self, esquerda, direita):
        self.chamadas.append(("lado", esquerda, direita))
        return True

    def rodas(self, fe, te, fd, td):
        self.chamadas.append(("rodas", fe, te, fd, td))
        return True

    def parar(self):
        self.chamadas.append(("parar",))
        return True


class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.arduino = ArduinoFalso()
        direcao.init_steering(self.arduino)

    def tearDown(self):
        # O modulo e global por processo; nenhum teste posterior pode herdar
        # um ArduinoFalso sem a API completa de keepalive.
        direcao.init_steering(None)

    def test_reta_tem_o_mesmo_pwm_nos_dois_lados(self):
        direcao.steer(0, LINE_FOLLOW_SPEED)

        self.assertEqual(
            self.arduino.chamadas,
            [("lado", LINE_FOLLOW_PWM, LINE_FOLLOW_PWM)],
        )

    def test_mixer_continuo_cruza_o_zero_sem_salto(self):
        self.assertEqual(
            direcao.mix_line_pwm(.25, LINE_FOLLOW_SPEED),
            (80, 40),
        )
        self.assertEqual(
            direcao.mix_line_pwm(.50, LINE_FOLLOW_SPEED),
            (80, 0),
        )
        self.assertEqual(
            direcao.mix_line_pwm(.75, LINE_FOLLOW_SPEED),
            (80, -40),
        )

    def test_mixer_e_simetrico_para_esquerda_e_direita(self):
        direita = direcao.mix_line_pwm(.75, LINE_FOLLOW_SPEED)
        esquerda = direcao.mix_line_pwm(-.75, LINE_FOLLOW_SPEED)

        self.assertEqual(direita, tuple(reversed(esquerda)))
        self.assertEqual(direita, (80, -40))

    def test_pivo_total_usa_os_dois_pares_sem_separar_eixos(self):
        direcao.steer_line(1., LINE_FOLLOW_SPEED)

        self.assertEqual(self.arduino.chamadas, [("lado", 80, -80)])

    def test_canto_fechado_mantem_os_eixos_sincronizados(self):
        direcao.steer_line(-.72, LINE_FOLLOW_SPEED)

        self.assertEqual(self.arduino.chamadas, [("lado", -35, 80)])

    def test_evento_verde_travado_chega_aos_motores_com_sinal_correto(self):
        for decisao, sinais in (
            (GreenDecision.RIGHT, (1, -1)),
            (GreenDecision.LEFT, (-1, 1)),
        ):
            with self.subTest(decisao=decisao):
                self.arduino.chamadas.clear()
                fsm = GreenManeuverFSM()
                fsm.observe(GreenObservation(
                    sequence=1,
                    junction_id=1,
                    decision_id=1,
                    timestamp=0.,
                    decision=decisao,
                    confidence=1.,
                    marker_ids=(1,),
                ))

                direcao.steer(fsm.locked_turn_angle(), .5)

                _, esquerda, direita = self.arduino.chamadas[-1]
                self.assertEqual(
                    (1 if esquerda > 0 else -1,
                     1 if direita > 0 else -1),
                    sinais,
                )

    def test_observador_publica_pwm_real_de_giro_re_e_parada(self):
        publicados = []
        direcao.set_motion_observer(
            lambda esquerda, direita: publicados.append(
                (esquerda, direita)))

        direcao.steer(180, .5)
        direcao.steer(-180, .5)
        direcao.steer(200, .5)
        direcao.steer()

        self.assertEqual(publicados, [
            (72, -72),
            (-72, 72),
            (-60, -60),
            (0, 0),
        ])

    def test_motion_guard_transforma_movimento_em_parada(self):
        direcao.set_motion_guard(lambda: False)

        self.assertFalse(direcao.steer(180, .5))
        self.assertFalse(direcao.steer_line(1., .5))

        self.assertEqual(
            self.arduino.chamadas,
            [("parar",), ("parar",)],
        )

    def test_motion_guard_interrompe_espera_bloqueante(self):
        direcao.set_motion_guard(lambda: False)

        self.assertFalse(direcao.sleep_steering(.5))

        self.assertEqual(self.arduino.chamadas, [("parar",)])


if __name__ == "__main__":
    unittest.main()
