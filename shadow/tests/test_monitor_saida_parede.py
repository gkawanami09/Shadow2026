"""Testes do agendamento serial da saida pela parede."""

from types import SimpleNamespace
import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.monitor_saida_parede import MonitorSensoresSaida  # noqa: E402


class ArduinoFalso:
    def __init__(self):
        self.comandos = []
        self.consultas_sensores_pendentes = False
        self.ultima_leitura_ultrassom_respondeu = True
        self._mpu_pronto = None
        self._ultrassom_pronto = None

    def iniciar_mpu(self, timeout):
        self.comandos.append("MPU")
        self.consultas_sensores_pendentes = True
        return True

    def poll_mpu(self):
        if self._mpu_pronto is None:
            return False, None
        leitura = self._mpu_pronto
        self._mpu_pronto = None
        self.consultas_sensores_pendentes = False
        return True, leitura

    def iniciar_ultrassom(self, timeout, lado):
        self.comandos.append(f"ULTRASSOM {lado}")
        self.consultas_sensores_pendentes = True
        return True

    def poll_ultrassom(self):
        if self._ultrassom_pronto is None:
            return False, None
        leitura = self._ultrassom_pronto
        self._ultrassom_pronto = None
        self.consultas_sensores_pendentes = False
        return True, leitura

    def cancelar_ultrassom(self):
        self.consultas_sensores_pendentes = False

    def cancelar_mpu(self):
        self.consultas_sensores_pendentes = False


class ControladorFalso:
    def __init__(self):
        self.yaws = []
        self.ultrassons = []

    def observar_mpu(self, yaw, instante):
        self.yaws.append((yaw, instante))

    def observar_ultrassom(self, lado, distancia, respondeu, instante):
        self.ultrassons.append((lado, distancia, respondeu, instante))


class MonitorSaidaParedeTests(unittest.TestCase):
    def test_alterna_mpu_e_ultrassom_lateral_sem_consultar_frente(self):
        arduino = ArduinoFalso()
        controlador = ControladorFalso()
        monitor = MonitorSensoresSaida(arduino)

        self.assertTrue(monitor.agendar_proxima(0.00))
        arduino._mpu_pronto = SimpleNamespace(yaw_graus=4.0)
        monitor.atualizar_controlador(controlador, 0.01)

        self.assertTrue(monitor.agendar_proxima(0.08))
        arduino._ultrassom_pronto = 410
        monitor.atualizar_controlador(controlador, 0.09)

        self.assertTrue(monitor.agendar_proxima(0.10))
        arduino._mpu_pronto = SimpleNamespace(yaw_graus=5.0)
        monitor.atualizar_controlador(controlador, 0.11)

        self.assertTrue(monitor.agendar_proxima(0.17))
        arduino._ultrassom_pronto = 145
        monitor.atualizar_controlador(controlador, 0.18)

        self.assertEqual(
            arduino.comandos,
            ["MPU", "ULTRASSOM LATERAL", "MPU", "ULTRASSOM LATERAL"],
        )
        self.assertEqual([leitura[0] for leitura in controlador.ultrassons], [
            "LATERAL", "LATERAL",
        ])
        self.assertEqual([leitura[0] for leitura in controlador.yaws], [4.0, 5.0])

    def test_giro_prioriza_mpu_sem_intercalar_ultrassom(self):
        arduino = ArduinoFalso()
        controlador = ControladorFalso()
        monitor = MonitorSensoresSaida(arduino)

        self.assertTrue(monitor.agendar_proxima(0.00, priorizar_mpu=True))
        arduino._mpu_pronto = SimpleNamespace(yaw_graus=4.0)
        monitor.atualizar_controlador(controlador, 0.01)

        self.assertTrue(monitor.agendar_proxima(0.06, priorizar_mpu=True))

        self.assertEqual(arduino.comandos, ["MPU", "MPU"])


if __name__ == "__main__":
    unittest.main()
