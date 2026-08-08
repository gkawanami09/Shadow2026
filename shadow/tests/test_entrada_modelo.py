"""Entrada da sala: modelo ONNX + alinhamento obrigatório da linha."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.entrada_missao import (  # noqa: E402
    EntryDetection, EntryGate, EntryModel, update_entry_silver)


class _Net:
    def __init__(self, output):
        self.output = output

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.output


class _Model:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detect(self, frame):
        return next(self.detections)


class EntryModelTests(unittest.TestCase):
    def test_modelo_de_entrada_usa_o_caminho_configurado(self):
        self.assertEqual(
            EntryModel().path,
            SHADOW_ROOT / config.ENTRY_MODEL_PATH,
        )

    def test_alinhamento_recente_tem_janela_curta(self):
        self.assertGreater(config.ENTRY_ALIGNMENT_HOLD_S, 0)
        self.assertLess(config.ENTRY_ALIGNMENT_HOLD_S, 1)

    def test_saida_yolo_uma_classe_vira_caixa_no_frame(self):
        model = EntryModel(input_size=640, min_confidence=.6)
        output = np.zeros((1, 5, 6), dtype=np.float32)
        output[0, :, 0] = (320, 320, 200, 100, .91)
        output[0, :, 1] = (10, 10, 10, 10, .2)
        model.net = _Net(output)
        detection = model.detect(np.zeros((252, 448, 3), dtype=np.uint8))
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.confidence, .91, places=5)
        self.assertGreater(detection.bbox[2], 0)
        self.assertGreater(detection.bbox[3], 0)


class EntryGateTests(unittest.TestCase):
    def test_inicio_da_inferencia_da_prioridade_ao_modelo(self):
        from shared.dados_compartilhados import (
            entry_armed, entry_model_priority)
        previous_armed = entry_armed.value
        previous_priority = entry_model_priority.value
        try:
            entry_armed.value = True
            gate = EntryGate(model=_Model([EntryDetection((1, 2, 3, 4), .7)]))
            update_entry_silver(gate, None, 1., line_aligned=True)
            self.assertTrue(entry_model_priority.value)
        finally:
            entry_armed.value = previous_armed
            entry_model_priority.value = previous_priority

    def test_uma_deteccao_muito_confiavel_confirma_na_hora(self):
        detection = EntryDetection((1, 2, 3, 4), .95)
        gate = EntryGate(model=_Model([detection]))
        confirmed, _ = gate.update(None, 0., line_aligned=True)
        self.assertTrue(confirmed)
        self.assertEqual(gate.last_reason, "confirmada_rapida")

    def test_dois_frames_alinhados_confirmam(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate(model=_Model([detection, detection]))
        self.assertFalse(gate.update(None, 0., line_aligned=True)[0])
        self.assertTrue(gate.update(None, .03, line_aligned=True)[0])

    def test_modelo_sem_alinhamento_nao_aciona_resgate(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate(model=_Model([detection, detection, detection]))
        self.assertFalse(gate.update(None, 0., line_aligned=False)[0])
        self.assertFalse(gate.update(None, .03, line_aligned=False)[0])
        self.assertFalse(gate.update(None, .06, line_aligned=True)[0])
        self.assertEqual(gate.last_reason, "votando")
