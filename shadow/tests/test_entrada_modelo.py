"""Entrada da sala: modelo ONNX + alinhamento obrigatório da linha."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.entrada_missao import (  # noqa: E402
    EntryDetection, EntryGate, EntryInference, EntryModel,
    update_entry_silver)


class _Session:
    def __init__(self, output):
        self.output = output

    def run(self, _outputs, inputs):
        self.inputs = inputs
        return [self.output]


class _Pipeline:
    votes = 0
    last_reason = "modelo_sem_faixa"

    def submit(self, frame, timestamp, line_aligned):
        self.submitted = (frame, timestamp, line_aligned)

    def poll(self):
        return False, None


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
        model._session = _Session(output)
        model._input_name = "images"
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
            update_entry_silver(
                _Pipeline(), np.zeros((2, 2, 3), dtype=np.uint8), 1.,
                line_aligned=True, wait_for_result=True)
            self.assertTrue(entry_model_priority.value)
        finally:
            entry_armed.value = previous_armed
            entry_model_priority.value = previous_priority

    def test_uma_deteccao_muito_confiavel_confirma_na_hora(self):
        detection = EntryDetection((1, 2, 3, 4), .95)
        gate = EntryGate()
        confirmed, _ = gate.update(EntryInference(0., True, detection))
        self.assertTrue(confirmed)
        self.assertEqual(gate.last_reason, "confirmada_rapida")

    def test_dois_frames_alinhados_confirmam(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(0., True, detection))[0])
        self.assertTrue(gate.update(EntryInference(.03, True, detection))[0])

    def test_modelo_sem_alinhamento_nao_aciona_resgate(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(0., False, detection))[0])
        self.assertFalse(gate.update(EntryInference(.03, False, detection))[0])
        self.assertFalse(gate.update(EntryInference(.06, True, detection))[0])
        self.assertEqual(gate.last_reason, "votando")
