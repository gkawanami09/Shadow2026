"""Regressões da gravação sincronizada de visão."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import cv2


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.gravacao_debug import GravadorVisao  # noqa: E402


class GravadorVisaoTests(unittest.TestCase):
    def test_jsonl_preserva_sequence_e_remove_nan(self):
        with TemporaryDirectory() as directory:
            recorder = GravadorVisao(
                directory, largura=16, altura=12, fps=10.)
            try:
                recorder.gravar(
                    np.zeros((12, 16, 3), dtype=np.uint8),
                    {
                        "sequence": 42,
                        "decision_id": 7,
                        "yaw": float("nan"),
                        "target": (8., 3.),
                    },
                )
                log_path = recorder.log_path
            finally:
                recorder.close()

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["sequence"], 42)
            self.assertEqual(records[0]["decision_id"], 7)
            self.assertIsNone(records[0]["yaw"])
            self.assertEqual(records[0]["target"], [8., 3.])
            self.assertEqual(records[0]["frame_index"], 1)
            raw_path = Path(directory) / records[0]["raw_frame"]
            self.assertTrue(raw_path.is_file())
            manifest = json.loads(
                recorder.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["session_id"], recorder.session_id)
            self.assertEqual(manifest["width"], 16)

    def test_png_cru_preserva_todos_os_pixels(self):
        with TemporaryDirectory() as directory:
            frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
            recorder = GravadorVisao(
                directory, largura=16, altura=12, fps=10.)
            try:
                recorder.gravar(frame, {"sequence": 9, "decision_id": 3})
                registro = json.loads(
                    recorder.log_path.read_text(encoding="utf-8").strip()
                )
                restored = cv2.imread(
                    str(Path(directory) / registro["raw_frame"]),
                    cv2.IMREAD_COLOR,
                )
            finally:
                recorder.close()

            self.assertTrue(np.array_equal(restored, frame))

    def test_duas_sessoes_no_mesmo_diretorio_nunca_se_sobrescrevem(self):
        with TemporaryDirectory() as directory:
            first = GravadorVisao(directory, largura=16, altura=12, fps=10.)
            second = GravadorVisao(directory, largura=16, altura=12, fps=10.)
            try:
                self.assertNotEqual(first.log_path, second.log_path)
                self.assertNotEqual(first.frames_path, second.frames_path)
                frame = np.zeros((12, 16, 3), dtype=np.uint8)
                first.gravar(frame, {"sequence": 1})
                second.gravar(frame, {"sequence": 2})
            finally:
                first.close()
                second.close()

            self.assertIn('"sequence":1', first.log_path.read_text("utf-8"))
            self.assertIn('"sequence":2', second.log_path.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
