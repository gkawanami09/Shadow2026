"""Contrato entre o YOLO da saída e o alinhamento geométrico existente."""

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.faixa_saida import ExitStripeDetection  # noqa: E402
from visao.saida_yolo import (  # noqa: E402
    ExitModel,
    ExitModelDetection,
    LatestFrameExitDetector,
    ModelGuidedExitDetector,
)


class ModeloFalso:
    def __init__(self, detection):
        self.detection = detection

    def detect(self, _frame):
        return self.detection


class GeometriaFalsa:
    def __init__(self, detection):
        self.detection = detection
        self.calls = 0

    def detect(self, _frame, timestamp=None):
        self.calls += 1
        return self.detection


def faixa(centro_x=320.0, centro_y=240.0, angulo=0.0):
    return ExitStripeDetection(
        center_x=centro_x,
        center_y=centro_y,
        width=300,
        height=24,
        top_y=228,
        bottom_y=252,
        span_ratio=.47,
        thickness_ratio=.05,
        aspect=12.5,
        value=25.0,
        surround_contrast=40.0,
        confidence=.50,
        timestamp=0.0,
        bbox=(170, 228, 300, 24),
        angle_deg=angulo,
    )


class SaidaYoloTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_modelo_puro_nao_exige_angulo_do_opencv(self):
        modelo = ModeloFalso(ExitModelDetection((140, 190, 360, 100), .91))
        geometria = GeometriaFalsa(faixa(320, 240, -7.0))
        detector = ModelGuidedExitDetector(modelo, geometria)

        detection = detector.detect(self.frame, timestamp=1.0)

        self.assertEqual(detection.angle_deg, 0.0)
        self.assertEqual(detection.confidence, .91)
        self.assertIsNone(detector.last_geometry_detection)
        self.assertEqual(geometria.calls, 0)

    def test_geometria_fora_da_caixa_nao_rouba_o_alvo_do_modelo(self):
        modelo = ModeloFalso(ExitModelDetection((60, 180, 160, 100), .88))
        detector = ModelGuidedExitDetector(
            modelo, GeometriaFalsa(faixa(520, 240, 12.0)))

        detection = detector.detect(self.frame, timestamp=2.0)

        self.assertAlmostEqual(detection.center_x, 140.0)
        self.assertEqual(detection.angle_deg, 0.0)
        self.assertIsNone(detector.last_geometry_detection)

    def test_sem_modelo_nao_consulta_geometria(self):
        geometria = GeometriaFalsa(faixa())
        detector = ModelGuidedExitDetector(ModeloFalso(None), geometria)

        self.assertIsNone(detector.detect(self.frame, timestamp=3.0))
        self.assertEqual(geometria.calls, 0)

    def test_previa_do_giro_nao_executa_geometria(self):
        geometria = GeometriaFalsa(faixa())
        detector = ModelGuidedExitDetector(
            ModeloFalso(ExitModelDetection((140, 190, 360, 100), .85)),
            geometria,
        )

        detection = detector.preview(self.frame, timestamp=4.0)

        self.assertIsNotNone(detection)
        self.assertEqual(geometria.calls, 0)

    def test_decodifica_saida_yolov8_de_uma_classe(self):
        model = ExitModel(input_size=416, min_confidence=.60)
        # Formato real do export: (4 + classes, N). N é muito maior que 5
        # no modelo real (3549); dez candidatos já exercitam a transposição.
        output = np.zeros((5, 10), dtype=np.float32)
        output[:, 0] = (208.0, 208.0, 208.0, 40.0, .90)
        output[:, 1] = (100.0, 100.0, 50.0, 40.0, .30)

        detection = model._decode(output, 1.0, 0, 0, 416, 416)

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.center_x, 208.0)
        self.assertAlmostEqual(detection.confidence, .90)


class SaidaAssincronaTests(unittest.TestCase):
    def test_worker_nao_bloqueia_e_descarta_backlog_antigo(self):
        iniciou = threading.Event()
        liberar = threading.Event()

        class DetectorLento:
            def detect(self, frame, timestamp=None):
                iniciou.set()
                liberar.wait(timeout=1.0)
                return faixa(
                    centro_x=float(frame[0, 0, 0]),
                    centro_y=240.0,
                )

            def preview(self, frame, timestamp=None):
                return self.detect(frame, timestamp=timestamp)

        worker = LatestFrameExitDetector(DetectorLento())
        try:
            primeiro = np.full((4, 4, 3), 1, dtype=np.uint8)
            segundo = np.full((4, 4, 3), 2, dtype=np.uint8)
            ultimo = np.full((4, 4, 3), 3, dtype=np.uint8)
            worker.submit(primeiro, 1.0, source_sequence=1)
            self.assertTrue(iniciou.wait(timeout=1.0))

            comecou = time.perf_counter()
            worker.submit(segundo, 2.0, source_sequence=2)
            worker.submit(ultimo, 3.0, source_sequence=3)
            self.assertLess(time.perf_counter() - comecou, 0.05)
            liberar.set()

            limite = time.monotonic() + 1.0
            resultado = None
            while time.monotonic() < limite:
                atual = worker.poll()
                if atual is not None and atual.source_sequence == 3:
                    resultado = atual
                    break
                time.sleep(0.005)

            self.assertIsNotNone(resultado)
            self.assertEqual(resultado.detection.center_x, 3.0)
            self.assertGreaterEqual(resultado.dropped_frames, 1)
        finally:
            liberar.set()
            worker.close()


if __name__ == "__main__":
    unittest.main()
