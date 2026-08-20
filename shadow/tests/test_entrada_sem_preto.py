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

    def test_falha_ao_apagar_led_mantem_entrada_armada_para_retry(self):
        arduino = ArduinoFalso(led_result=False)

        with patch.object(ciclo, "steer", return_value=True):
            entrou = ciclo._enter_rescue_after_no_black(arduino)

        self.assertFalse(entrou)
        self.assertTrue(ciclo.entry_armed.value)
        self.assertEqual(arduino.led_calls, ["APAGADO"])
        self.assertEqual(
            ciclo.status.value,
            'Falha ao apagar LED na entrada do resgate - tentando novamente',
        )

    def test_handoff_confirmado_desarma_entrada(self):
        arduino = ArduinoFalso(led_result=True)

        with patch.object(ciclo, "steer", return_value=True):
            entrou = ciclo._enter_rescue_after_no_black(arduino)

        self.assertTrue(entrou)
        self.assertFalse(ciclo.entry_armed.value)


if __name__ == "__main__":
    unittest.main()
