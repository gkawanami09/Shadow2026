import sys
from pathlib import Path
import types
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

try:
    import serial  # noqa: F401
except ModuleNotFoundError:
    serial_module = types.ModuleType("serial")
    serial_module.SerialException = OSError
    serial_module.Serial = object
    serial_tools = types.ModuleType("serial.tools")
    list_ports = types.ModuleType("serial.tools.list_ports")
    list_ports.comports = lambda: []
    serial_tools.list_ports = list_ports
    serial_module.tools = serial_tools
    sys.modules["serial"] = serial_module
    sys.modules["serial.tools"] = serial_tools
    sys.modules["serial.tools.list_ports"] = list_ports

from serial_link.arduino import Arduino  # noqa: E402


class RescueSerialLedTests(unittest.TestCase):
    @staticmethod
    def _bare_arduino():
        arduino = Arduino.__new__(Arduino)
        arduino._connected = False
        arduino._last_reconnect_t = -1e9
        arduino._desired_led_mode = None
        return arduino

    def test_led_command_is_remembered(self):
        arduino = self._bare_arduino()
        sent = []
        arduino._send_aux_cmd = sent.append

        arduino.led("apagado")

        self.assertEqual(arduino._desired_led_mode, "APAGADO")
        self.assertEqual(sent, ["LED APAGADO"])

    def test_reconnect_restores_led_before_returning(self):
        arduino = self._bare_arduino()
        arduino._desired_led_mode = "APAGADO"
        arduino._candidate_ports = lambda: ["/dev/ttyACM0"]
        sent = []
        arduino._send_aux_cmd = sent.append

        def connect(_device):
            arduino._connected = True
            return True

        arduino._try_port = connect
        arduino._try_reconnect()

        self.assertTrue(arduino._connected)
        self.assertEqual(sent, ["LED APAGADO"])


if __name__ == "__main__":
    unittest.main()
