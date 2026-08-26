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

    def rodas(self, frente_esquerda, tras_esquerda,
              frente_direita, tras_direita):
        self.chamadas.append((
            "rodas", frente_esquerda, tras_esquerda,
            frente_direita, tras_direita,
        ))
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

    def test_curva_normal_eh_arco_continuo_sem_re_na_roda_interna(self):
        direcao.steer(180, LINE_FOLLOW_SPEED, center_pivot=False)

        comando, esquerda, direita = self.arduino.chamadas[-1]
        self.assertEqual(comando, "lado")
        self.assertGreater(esquerda, 0)
        self.assertEqual(direita, 0)
        self.assertLess(esquerda, LINE_FOLLOW_PWM)

    def test_pivo_central_so_acontece_quando_pedido(self):
        direcao.steer(-180, LINE_FOLLOW_SPEED, center_pivot=True)

        self.assertEqual(self.arduino.chamadas[-1][0], "lado")
        _, esquerda, direita = self.arduino.chamadas[-1]
        self.assertLess(esquerda, 0)
        self.assertGreater(direita, 0)


if __name__ == "__main__":
    unittest.main()
