"""Testes da trava de segurança dos motores."""

import sys
from pathlib import Path
import tempfile
import unittest

SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.trava_motores import MotorLockError, MotorOwnerLock


class MotorOwnerLockTests(unittest.TestCase):
    def test_second_motor_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motors.lock"
            first = MotorOwnerLock("first", path)
            second = MotorOwnerLock("second", path)
            first.acquire()
            try:
                with self.assertRaises(MotorLockError):
                    second.acquire()
            finally:
                first.release()

            second.acquire()
            second.release()


if __name__ == "__main__":
    unittest.main()
