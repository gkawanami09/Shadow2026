"""Garante que uma falha de resgate nunca seja confundida com a volta normal."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from mission import (  # noqa: E402
    RESCUE_EXIT_ARDUINO_DESCONECTADO,
    RESCUE_EXIT_OK,
    RESCUE_RETURN_COMPLETED,
    RESCUE_RETURN_RESTART_AFTER_ARDUINO,
    RESCUE_RETURN_STOPPED,
    rescue_return_action,
)


class RescueReturnSafetyTests(unittest.TestCase):
    def test_apenas_resgate_concluido_pode_voltar_normalmente_ao_percurso(self):
        self.assertEqual(
            rescue_return_action(RESCUE_EXIT_OK),
            RESCUE_RETURN_COMPLETED,
        )

    def test_so_desconexao_do_arduino_autoriza_reinicio_do_percurso(self):
        self.assertEqual(
            rescue_return_action(RESCUE_EXIT_ARDUINO_DESCONECTADO),
            RESCUE_RETURN_RESTART_AFTER_ARDUINO,
        )
        self.assertEqual(rescue_return_action(3), RESCUE_RETURN_STOPPED)
        self.assertEqual(rescue_return_action(4), RESCUE_RETURN_STOPPED)
        self.assertEqual(rescue_return_action(99), RESCUE_RETURN_STOPPED)


if __name__ == "__main__":
    unittest.main()
