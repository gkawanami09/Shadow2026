"""Testes da entrada direta da manobra pos-vermelho."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import saida  # noqa: E402


class SaidaDiretaTests(unittest.TestCase):
    class TravaFalsa:
        def __init__(self):
            self.adquirida = False
            self.liberada = False

        def acquire(self):
            self.adquirida = True
            return self

        def release(self):
            self.liberada = True

    class ArduinoFalso:
        def __init__(self):
            self.sessao_travada = False
            self.leds = []
            self.fechado = False

        def travar_sessao(self):
            self.sessao_travada = True

        def led(self, modo):
            self.leds.append(modo)
            return True

        def close(self):
            self.fechado = True

    def test_roda_apenas_a_manobra_e_fecha_a_sessao(self):
        trava = self.TravaFalsa()
        arduino = self.ArduinoFalso()
        with (
            patch("saida.MotorOwnerLock", return_value=trava),
            patch("saida.Arduino", return_value=arduino),
            patch("saida.init_steering") as iniciar_direcao,
            patch("saida.steer", return_value=True) as parar,
            patch(
                "saida.executar_alinhamento_parede",
                return_value="saida_concluida",
            ) as executar,
        ):
            codigo = saida.main([])

        self.assertEqual(codigo, saida.EXIT_OK)
        self.assertTrue(trava.adquirida)
        self.assertTrue(trava.liberada)
        iniciar_direcao.assert_called_once_with(arduino)
        executar.assert_called_once_with(arduino, camera_index=None)
        self.assertTrue(arduino.sessao_travada)
        self.assertTrue(arduino.fechado)
        self.assertEqual(arduino.leds, ["APAGADO"])
        self.assertGreaterEqual(parar.call_count, 2)

    def test_falha_na_manobra_fecha_arduino_e_nao_retorna_sucesso(self):
        trava = self.TravaFalsa()
        arduino = self.ArduinoFalso()
        with (
            patch("saida.MotorOwnerLock", return_value=trava),
            patch("saida.Arduino", return_value=arduino),
            patch("saida.init_steering"),
            patch("saida.steer", return_value=True),
            patch("saida.executar_alinhamento_parede", return_value=None),
        ):
            codigo = saida.main([])

        self.assertEqual(codigo, saida.EXIT_FALHA_MANOBRA)
        self.assertTrue(arduino.fechado)
        self.assertTrue(trava.liberada)


if __name__ == "__main__":
    unittest.main()
