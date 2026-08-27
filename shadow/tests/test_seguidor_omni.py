"""Testes do controle visual omni sem MPU."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle import direcao  # noqa: E402
from controle.seguidor_omni import ControladorSeguidorOmni  # noqa: E402


class ArduinoFalso:
    def __init__(self):
        self.chamadas = []

    def rodas(self, fe, te, fd, td):
        self.chamadas.append((fe, te, fd, td))
        return True


def _resultado(agora, **mudancas):
    dados = dict(
        publicado_em=agora,
        valida=True,
        lateral=0.,
        orientacao=0.,
        curvatura=0.,
        confianca=.95,
    )
    dados.update(mudancas)
    return SimpleNamespace(**dados)


class SeguidorOmniTests(unittest.TestCase):
    def setUp(self):
        self.arduino = ArduinoFalso()
        direcao.init_steering(self.arduino)
        self.controle = ControladorSeguidorOmni()

    def test_reta_envia_pwm_80_nas_quatro_rodas(self):
        enviado = self.controle.atualizar(
            _resultado(10.), config.LINE_FOLLOW_SPEED, agora=10.)

        self.assertTrue(enviado)
        self.assertEqual(self.arduino.chamadas[-1], (80, 80, 80, 80))

    def test_orientacao_direita_gira_sem_ultrapassar_pwm_80(self):
        self.controle.atualizar(
            _resultado(20., orientacao=.45),
            config.LINE_FOLLOW_SPEED,
            agora=20.,
        )
        fe, te, fd, td = self.arduino.chamadas[-1]

        self.assertGreater(fe, fd)
        self.assertGreater(te, td)
        self.assertLessEqual(max(abs(v) for v in (fe, te, fd, td)), 80)

    def test_erro_lateral_usa_pares_diagonais(self):
        self.controle.atualizar(
            _resultado(30., lateral=.5),
            config.LINE_FOLLOW_SPEED,
            agora=30.,
        )
        fe, te, fd, td = self.arduino.chamadas[-1]

        self.assertGreater(fe, te)
        self.assertGreater(td, fd)
        self.assertEqual(self.controle.modo, "recentrando")

    def test_erro_fora_do_corredor_reduz_avanco(self):
        self.controle.atualizar(
            _resultado(40., lateral=.20),
            config.LINE_FOLLOW_SPEED,
            agora=40.,
        )
        rodas_deslocadas = self.arduino.chamadas[-1]

        self.assertEqual(self.controle.modo, "recentrando")
        self.assertLess(sum(rodas_deslocadas) / 4., 50.)

    def test_pequeno_erro_permanece_no_modo_normal(self):
        self.controle.atualizar(
            _resultado(50., lateral=.06),
            config.LINE_FOLLOW_SPEED,
            agora=50.,
        )

        self.assertEqual(self.controle.modo, "normal")

    def test_resultado_antigo_pede_fallback_sem_mover(self):
        enviado = self.controle.atualizar(
            _resultado(1.), config.LINE_FOLLOW_SPEED, agora=2.)

        self.assertFalse(enviado)
        self.assertEqual(self.arduino.chamadas, [])


if __name__ == "__main__":
    unittest.main()
