"""Regressoes da troca exclusiva para a camera de segue-linha."""

from types import SimpleNamespace
import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

# O teste exercita apenas a orquestracao. No ambiente de CI sem OpenCV, um
# modulo vazio basta para importar os detectores que serao substituidos abaixo.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.SimpleNamespace()

from controle import sonda_linha_saida as sonda  # noqa: E402


class RelogioFalso:
    def __init__(self):
        self.tempo = 0.0

    def __call__(self):
        return self.tempo

    def dormir(self, segundos):
        self.tempo += max(float(segundos), 0.001)


class CameraFalsa:
    def __init__(self, relogio):
        self.relogio = relogio
        self.fechada = False
        self.frames = 0

    def get_frame(self):
        self.frames += 1
        self.relogio.dormir(0.02)
        return object()

    def close(self):
        self.fechada = True


class ArduinoFalso:
    def __init__(self):
        self.connected = True
        self.connection_epoch = 3
        self.leds = []

    def refresh(self, fail_closed=True):
        return True

    def led(self, modo):
        self.leds.append(modo)
        return True


class DirecaoGravada:
    def __init__(self):
        self.comandos = []

    def __call__(self, *argumentos):
        self.comandos.append(argumentos)
        return True


class ClassificadorFalso:
    def __init__(self, classificacao):
        self.classificacao = classificacao

    def classificar(self, _frame, timestamp=None):
        return SimpleNamespace(classificacao=self.classificacao)


class SondaLinhaSaidaTests(unittest.TestCase):
    def _executar(self, classificacao):
        relogio = RelogioFalso()
        arduino = ArduinoFalso()
        camera = CameraFalsa(relogio)
        direcao = DirecaoGravada()
        with (
            patch.object(
                sonda,
                "ClassificadorFaixaSaidaLinha",
                new=lambda: ClassificadorFalso(classificacao),
            ),
            patch.object(sonda, "faixa_pronta_para_confirmacao", new=lambda _: True),
            patch.object(sonda.cfg, "EXIT_LINE_CAMERA_WARMUP_S", new=0.0),
        ):
            resultado = sonda.testar_abertura_com_camera_linha(
                arduino,
                direcao,
                relogio=relogio,
                dormir=relogio.dormir,
                camera_factory=lambda: camera,
            )
        return resultado, arduino, camera, direcao

    def test_faixa_nao_preta_rejeita_abertura_e_mede_avanco_parcial(self):
        resultado, arduino, camera, direcao = self._executar("nao_preta")

        self.assertEqual(resultado.resultado, "nao_preta")
        self.assertGreater(resultado.avanco_s, 0.0)
        self.assertTrue(camera.fechada)
        self.assertEqual(arduino.leds, ["ACESO", "APAGADO"])
        self.assertIn((0, sonda.cfg.SAIDA_PAREDE_PWM_SONDA_LINHA / 120.0), direcao.comandos)

    def test_preto_so_e_liberado_depois_da_retomada_e_mantem_led_aceso(self):
        relogio = RelogioFalso()
        arduino = ArduinoFalso()
        camera = CameraFalsa(relogio)
        direcao = DirecaoGravada()
        retomadas = []

        class RetomadaFalsa:
            def __init__(self, *argumentos):
                retomadas.append(argumentos)

            def executar(self):
                return object()

        with (
            patch.object(
                sonda,
                "ClassificadorFaixaSaidaLinha",
                new=lambda: ClassificadorFalso("preta"),
            ),
            patch.object(sonda, "faixa_pronta_para_confirmacao", new=lambda _: True),
            patch.object(sonda, "ControladorRetomadaSaida", new=RetomadaFalsa),
            patch.object(sonda.cfg, "EXIT_LINE_CAMERA_WARMUP_S", new=0.0),
        ):
            resultado = sonda.testar_abertura_com_camera_linha(
                arduino,
                direcao,
                relogio=relogio,
                dormir=relogio.dormir,
                camera_factory=lambda: camera,
            )

        self.assertEqual(resultado.resultado, "preta")
        self.assertEqual(len(retomadas), 1)
        self.assertTrue(camera.fechada)
        self.assertEqual(arduino.leds, ["ACESO"])


if __name__ == "__main__":
    unittest.main()
