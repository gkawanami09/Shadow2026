"""Contrato da manobra de 180° acionada por dois verdes."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle import retorno  # noqa: E402


class RetornoTests(unittest.TestCase):
    def test_retorno_sempre_gira_direita_com_trecho_cego_e_re_curta(self):
        previous_detected = retorno.line_detected.value
        previous_bottom = retorno.last_bottom_point.value
        previous_size = retorno.line_size.value
        previous_timeout = retorno.T_180_SEARCH_TIMEOUT
        try:
            retorno.line_detected.value = True
            retorno.last_bottom_point.value = config.camera_x // 2
            retorno.line_size.value = config.TURN_AROUND_SMALL_LINE
            # Não é necessário esperar a busca para validar os comandos
            # temporizados; zera o timeout somente neste teste.
            retorno.T_180_SEARCH_TIMEOUT = 0
            with patch.object(retorno, "steer") as steer, \
                    patch.object(retorno, "sleep_steering") as sleep:
                next_direction = retorno.turn_around("l")

            pivots = [call.args for call in steer.call_args_list
                      if call.args and abs(call.args[0]) == 180]
            self.assertTrue(pivots)
            self.assertTrue(all(args[0] == 180 for args in pivots))
            self.assertEqual(next_direction, "r")
            durations = [call.args[0] for call in sleep.call_args_list]
            self.assertIn(config.T_180_BLIND_EXTRA, durations)
            self.assertEqual(config.T_180_BLIND_EXTRA, .10)
            self.assertIn(config.TURN_AROUND_REVERSE, durations)
            self.assertEqual(config.TURN_AROUND_REVERSE, .15)
        finally:
            retorno.line_detected.value = previous_detected
            retorno.last_bottom_point.value = previous_bottom
            retorno.line_size.value = previous_size
            retorno.T_180_SEARCH_TIMEOUT = previous_timeout
