import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

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


class FakeSerial:
    def __init__(self):
        self.incoming = bytearray()
        self.writes = []
        self.timeout = 0
        self.reply_on_write = None

    @property
    def in_waiting(self):
        return len(self.incoming)

    def feed(self, data):
        self.incoming.extend(data)

    def read(self, size):
        result = bytes(self.incoming[:size])
        del self.incoming[:size]
        return result

    def write(self, data):
        self.writes.append(data)
        if self.reply_on_write is not None:
            self.feed(self.reply_on_write)


class RescueSerialLedTests(unittest.TestCase):
    @staticmethod
    def _bare_arduino():
        arduino = Arduino.__new__(Arduino)
        arduino._connected = False
        arduino._connection_epoch = 0
        arduino._last_reconnect_t = -1e9
        arduino._desired_led_mode = None
        arduino._ser = None
        arduino._last_cmd = None
        arduino._last_send_t = 0.0
        arduino._rx_buffer = bytearray()
        arduino._ultra_pending = False
        arduino._ultra_deadline = 0.0
        arduino._ultra_ready = False
        arduino._ultra_value = None
        arduino._manual_pending = False
        arduino._manual_response = None
        return arduino

    def _connected_arduino(self):
        arduino = self._bare_arduino()
        arduino._connected = True
        arduino._ser = FakeSerial()
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

    def test_connection_epoch_changes_when_new_serial_is_adopted(self):
        arduino = self._bare_arduino()

        arduino._adopt(FakeSerial())
        first_epoch = arduino.connection_epoch
        arduino._adopt(FakeSerial())

        self.assertEqual(first_epoch, 1)
        self.assertEqual(arduino.connection_epoch, 2)

    def test_refresh_after_reconnect_discards_old_motion(self):
        arduino = self._bare_arduino()
        arduino._last_cmd = "LADO 50 50"
        arduino._last_send_t = 0.0
        sent = []

        def reconnect():
            arduino._connected = True

        arduino._try_reconnect = reconnect
        arduino._write_line = sent.append
        arduino._drain = lambda: None

        arduino.refresh(fail_closed=True)

        self.assertEqual(sent, ["PARAR"])
        self.assertEqual(arduino._last_cmd, "PARAR")

    def test_default_refresh_preserves_existing_mode_behavior(self):
        arduino = self._bare_arduino()
        arduino._last_cmd = "LADO 40 40"
        arduino._last_send_t = 0.0
        sent = []
        arduino._write_line = sent.append
        arduino._drain = lambda: None

        arduino.refresh()

        self.assertEqual(sent, ["LADO 40 40"])

    def test_ultrasonic_response_survives_refresh_and_split_line(self):
        arduino = self._connected_arduino()
        self.assertTrue(arduino.iniciar_ultrassom(timeout=0.05))
        self.assertEqual(arduino._ser.writes, [b"ULTRASSOM\n"])

        arduino._ser.feed(b"OK ULTRA")
        arduino.refresh()
        self.assertEqual(arduino.poll_ultrassom(), (False, None))

        arduino._ser.feed(b"SSOM 143\n")
        arduino.refresh()
        self.assertEqual(arduino.poll_ultrassom(), (True, 143))
        self.assertEqual(arduino.poll_ultrassom(), (False, None))

    def test_ultrasonic_start_rejects_overlap_and_unconsumed_result(self):
        arduino = self._connected_arduino()
        self.assertTrue(arduino.iniciar_ultrassom(timeout=0.05))
        self.assertFalse(arduino.iniciar_ultrassom(timeout=0.05))

        arduino._ser.feed(b"OK ULTRASSOM 200\n")
        arduino.refresh()
        self.assertFalse(arduino.iniciar_ultrassom(timeout=0.05))
        self.assertEqual(arduino.poll_ultrassom(), (True, 200))
        self.assertTrue(arduino.iniciar_ultrassom(timeout=0.05))

    def test_ultrasonic_no_echo_and_late_cancelled_reply(self):
        arduino = self._connected_arduino()
        self.assertTrue(arduino.iniciar_ultrassom(timeout=0.05))
        arduino._ser.feed(b"OK ULTRASSOM -1\n")
        self.assertEqual(arduino.poll_ultrassom(), (True, None))

        self.assertTrue(arduino.iniciar_ultrassom(timeout=0.05))
        arduino.cancelar_ultrassom()
        arduino._ser.feed(b"OK ULTRASSOM 99\n")
        arduino.refresh()
        self.assertEqual(arduino.poll_ultrassom(), (False, None))

    def test_ultrasonic_poll_times_out_without_blocking(self):
        arduino = self._connected_arduino()
        with patch(
            "serial_link.arduino.time.monotonic",
            side_effect=(10.0, 10.1),
        ):
            self.assertTrue(arduino.iniciar_ultrassom(timeout=0.05))
            self.assertEqual(arduino.poll_ultrassom(), (True, None))

    def test_manual_serial_command_still_receives_routed_reply(self):
        arduino = self._connected_arduino()
        arduino._ser.reply_on_write = b"PONG\n"

        self.assertEqual(
            arduino.comando_serial("PING", timeout=0.05),
            "PONG",
        )

    def test_grippers_are_sent_in_one_usb_write(self):
        arduino = self._connected_arduino()

        self.assertTrue(arduino.garras(-50, 50))

        self.assertEqual(
            arduino._ser.writes,
            [b"SERVO GARRA_ESQ -50\nSERVO GARRA_DIR 50\n"],
        )
        self.assertIsNone(arduino._last_cmd)

    def test_gripper_batch_validates_both_deltas_before_writing(self):
        arduino = self._connected_arduino()

        with self.assertRaisesRegex(ValueError, "fora"):
            arduino.garras(-50, 181)

        self.assertEqual(arduino._ser.writes, [])

    def test_futaba_write_reports_serial_delivery(self):
        arduino = self._connected_arduino()

        self.assertTrue(arduino.futaba(-20, 1500))

        self.assertEqual(
            arduino._ser.writes,
            [b"FUTABA -20 1500\n"],
        )

    def test_futaba_wait_refreshes_lado_zero_not_parar(self):
        arduino = self._connected_arduino()
        arduino.lado(0, 0)
        arduino.futaba(-20, 1500)

        with patch(
            "serial_link.arduino.time.monotonic",
            return_value=arduino._last_send_t + 0.30,
        ):
            arduino.refresh(fail_closed=True)

        self.assertEqual(
            arduino._ser.writes,
            [
                b"LADO 0 0\n",
                b"FUTABA -20 1500\n",
                b"LADO 0 0\n",
            ],
        )
        self.assertNotIn(b"PARAR\n", arduino._ser.writes)


if __name__ == "__main__":
    unittest.main()
