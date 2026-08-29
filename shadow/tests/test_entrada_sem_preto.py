"""Testes do handoff ao resgate disparado pela ausencia de preto."""

import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

# O ambiente minimo de testes locais nao instala pyserial. O caminho coberto
# abaixo nao abre porta serial; precisa apenas importar o modulo de controle.
try:
    import serial  # noqa: F401
except ModuleNotFoundError:
    serial_stub = types.ModuleType("serial")
    serial_stub.SerialException = OSError
    serial_tools_stub = types.ModuleType("serial.tools")
    serial_list_ports_stub = types.ModuleType("serial.tools.list_ports")
    serial_list_ports_stub.comports = lambda: ()
    serial_tools_stub.list_ports = serial_list_ports_stub
    serial_stub.tools = serial_tools_stub
    sys.modules["serial"] = serial_stub
    sys.modules["serial.tools"] = serial_tools_stub
    sys.modules["serial.tools.list_ports"] = serial_list_ports_stub

from controle import ciclo  # noqa: E402


class ArduinoFalso:
    def __init__(self, *, led_result=True, connected=True):
        self.led_result = led_result
        self.connected = connected
        self.led_calls = []

    def led(self, mode):
        self.led_calls.append(mode)
        return self.led_result


class EntryAfterNoBlackTests(unittest.TestCase):
    def setUp(self):
        self.entry_armed_original = ciclo.entry_armed.value
        self.terminate_original = ciclo.terminate.value
        self.status_original = ciclo.status.value
        ciclo.entry_armed.value = True
        ciclo.terminate.value = False

    def tearDown(self):
        ciclo.entry_armed.value = self.entry_armed_original
        ciclo.terminate.value = self.terminate_original
        ciclo.status.value = self.status_original

    def test_handoff_confirmado_desarma_entrada(self):
        arduino = ArduinoFalso(led_result=True)

        with patch.object(ciclo, "steer", return_value=True):
            entrou = ciclo._enter_rescue_after_no_black(arduino)

        self.assertTrue(entrou)
        self.assertFalse(ciclo.entry_armed.value)


class StartupTurnSequenceTests(unittest.TestCase):
    def setUp(self):
        self.terminate_original = ciclo.terminate.value
        self.status_original = ciclo.status.value
        ciclo.terminate.value = False

    def tearDown(self):
        ciclo.terminate.value = self.terminate_original
        ciclo.status.value = self.status_original

    def test_sequencia_gira_d_e_d_e_d_com_tempos_configurados(self):
        arduino = ArduinoFalso(connected=True)
        movimentos = []
        esperas = []

        with (
            patch.object(
                ciclo, "steer_line",
                side_effect=lambda sentido, velocidade, tank=False: (
                    movimentos.append((sentido, velocidade, tank)) or True)),
            patch.object(ciclo, "steer", return_value=True) as parar,
            patch.object(
                ciclo, "sleep_steering",
                side_effect=lambda duracao: esperas.append(duracao)),
        ):
            self.assertTrue(ciclo._executar_sequencia_partida(arduino))

        self.assertEqual(
            [movimento[0] for movimento in movimentos],
            [1., -1., 1., -1., 1.])
        self.assertEqual(
            [duracao for duracao in esperas if duracao >= .5],
            [.5, 1., 1., 1., .5])
        self.assertTrue(all(movimento[2] for movimento in movimentos))
        self.assertEqual(parar.call_count, 5)


if __name__ == "__main__":
    unittest.main()
