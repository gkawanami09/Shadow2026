"""Testes unitarios do mapeamento de angulo para os motores."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle import direcao  # noqa: E402
from config import LINE_FOLLOW_SPEED, LINE_FOLLOW_PWM  # noqa: E402


class ArduinoFalso:
    def __init__(self):
        self.chamadas = []

    def lado(self, esquerda, direita):
        self.chamadas.append(("lado", esquerda, direita))
        return True

    def parar(self):
        self.chamadas.append(("parar",))
        return True


class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.arduino = ArduinoFalso()
        direcao.init_steering(self.arduino)

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


if __name__ == "__main__":
    unittest.main()
