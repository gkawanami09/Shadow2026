"""Entrada da sala: modelo de prata + alinhamento obrigatório da linha."""

import sys
from pathlib import Path
import threading
import time
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.entrada_missao import (  # noqa: E402
    EntryDetection, EntryGate, EntryInference, EntryModel, EntryModelWorker,
    EntryPipeline,
    has_black_after_entry_detection, has_ramp_black_near_entry_detection)


class _Session:
    def __init__(self, output):
        self.output = output

    def run(self, _outputs, inputs):
        self.inputs = inputs
        return [self.output]


class EntryModelTests(unittest.TestCase):
    def test_modelo_de_entrada_usa_o_caminho_configurado(self):
        self.assertEqual(
            EntryModel().path,
            SHADOW_ROOT / config.ENTRY_MODEL_PATH,
        )

    def test_alinhamento_recente_tem_janela_curta(self):
        self.assertGreater(config.ENTRY_ALIGNMENT_HOLD_S, 0)
        self.assertLessEqual(config.ENTRY_ALIGNMENT_HOLD_S, 1.0)

    def test_saida_yolo_uma_classe_vira_caixa_no_frame(self):
        model = EntryModel(
            backend="onnx", input_size=640, min_confidence=.6)
        output = np.zeros((1, 5, 6), dtype=np.float32)
        output[0, :, 0] = (320, 320, 300, 80, .91)
        output[0, :, 1] = (10, 10, 10, 10, .2)
        model._session = _Session(output)
        model._input_name = "images"
        detection = model.detect(np.zeros((252, 448, 3), dtype=np.uint8))
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.confidence, .91, places=5)
        self.assertGreater(detection.bbox[2], 0)
        self.assertGreater(detection.bbox[3], 0)

    def test_confianca_abaixo_do_limiar_fica_visivel_para_calibracao(self):
        model = EntryModel(
            backend="onnx", input_size=640, min_confidence=.45)
        output = np.zeros((1, 5, 6), dtype=np.float32)
        output[0, :, 0] = (320, 320, 200, 100, .44)
        model._session = _Session(output)
        model._input_name = "images"

        detection = model.detect(np.zeros((252, 448, 3), dtype=np.uint8))

        self.assertIsNone(detection)
        self.assertAlmostEqual(model.last_confidence, .44, places=5)

    def test_limiar_padrao_da_prata_exige_confianca_maior(self):
        self.assertEqual(config.ENTRY_MODEL_MIN_CONFIDENCE, .55)

    def test_caixa_pequena_ou_quadrada_nao_vira_prata(self):
        model = EntryModel(
            backend="onnx", input_size=640, min_confidence=.5)
        output = np.zeros((1, 5, 6), dtype=np.float32)
        output[0, :, 0] = (320, 320, 80, 80, .91)
        model._session = _Session(output)
        model._input_name = "images"

        detection = model.detect(np.zeros((252, 448, 3), dtype=np.uint8))

        self.assertIsNone(detection)
        self.assertAlmostEqual(model.last_confidence, .91, places=5)


class EntryModelWorkerTests(unittest.TestCase):
    def test_janela_curta_preserva_frame_prata_enquanto_modelo_ocupado(self):
        started = threading.Event()
        release = threading.Event()

        class ModelFalso:
            def detect(self, frame):
                marker = int(frame[0, 0, 0])
                if marker == 1:
                    started.set()
                    release.wait(timeout=1.0)
                else:
                    # Da tempo para o processo de visao consumir cada resultado.
                    time.sleep(.01)
                return (
                    EntryDetection((20, 20, 40, 20), .80)
                    if marker == 2 else None
                )

        worker = EntryModelWorker(ModelFalso()).start()
        try:
            worker.submit(np.full((24, 24, 3), 1, dtype=np.uint8), 1., True)
            self.assertTrue(started.wait(timeout=.5))
            # A prata aparece enquanto o primeiro frame ainda esta no modelo.
            worker.submit(np.full((24, 24, 3), 2, dtype=np.uint8), 2., True)
            for marker in range(3, 9):
                worker.submit(
                    np.full((24, 24, 3), marker, dtype=np.uint8),
                    float(marker),
                    True,
                )
            release.set()

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                inference = worker.poll()
                if inference is not None and inference.timestamp == 2.0:
                    self.assertIsNotNone(inference.detection)
                    self.assertFalse(inference.black_ahead)
                    self.assertFalse(inference.ramp_black_ahead)
                    break
                time.sleep(.005)
            else:
                self.fail("o frame de prata foi perdido pela fila do modelo")
        finally:
            release.set()
            worker.close()


class EntryGateTests(unittest.TestCase):
    def test_preto_que_leva_ate_a_prata_nao_veta_o_resgate(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Black below the detection is the line approaching the silver strip.
        mask[70:95, 40:60] = 255
        detection = EntryDetection((40, 55, 20, 10), .8)

        self.assertFalse(has_black_after_entry_detection(mask, detection))

    def test_preto_alem_da_prata_veta_o_resgate(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # At the top is the track beyond the silver strip in driving direction.
        mask[10:45, 40:60] = 255
        detection = EntryDetection((40, 55, 20, 10), .8)

        self.assertTrue(has_black_after_entry_detection(mask, detection))

    def test_preto_do_limiar_da_rampa_antes_da_prata_nao_veta(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[70:95, 40:60] = 255
        detection = EntryDetection((40, 55, 20, 10), .8)

        self.assertFalse(has_black_after_entry_detection(mask, detection))

    def test_barra_preta_larga_da_rampa_perto_da_prata_veta(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[42:55, 15:85] = 255
        detection = EntryDetection((40, 55, 20, 12), .8)

        self.assertTrue(has_ramp_black_near_entry_detection(mask, detection))

    def test_linha_preta_estreita_na_entrada_nao_vira_barra_da_rampa(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:70, 43:57] = 255
        detection = EntryDetection((40, 55, 20, 12), .8)

        self.assertFalse(has_ramp_black_near_entry_detection(mask, detection))

    def test_prata_alinhada_sem_preto_confirma_em_dois_frames(self):
        detection = EntryDetection((1, 2, 3, 4), .95)
        gate = EntryGate()
        confirmed, _ = gate.update(EntryInference(0., True, detection, 0.))
        self.assertFalse(confirmed)
        self.assertTrue(gate.is_validating)
        self.assertEqual(gate.votes, 1)
        confirmed, _ = gate.update(EntryInference(.01, True, detection, 0.))
        self.assertTrue(confirmed)
        self.assertEqual(gate.last_reason, "confirmada")

    def test_confirmacao_rapida_usa_chegada_do_resultado(self):
        detection = EntryDetection((1, 2, 3, 4), .95)
        gate = EntryGate()
        self.assertFalse(gate.update(
            EntryInference(10., True, detection, 0.), now=100.)[0])
        self.assertTrue(gate.update(
            EntryInference(10.01, True, detection, 0.), now=100.01)[0])

    def test_modelo_sem_faixa_nao_confirma(self):
        gate = EntryGate()
        self.assertFalse(gate.update(
            EntryInference(0., True, None, 0.), now=10.)[0])
        self.assertEqual(gate.state, EntryGate.IDLE)
        self.assertEqual(gate.last_reason, "modelo_sem_faixa")

    def test_confirmacao_em_dois_frames_e_configuravel(self):
        self.assertEqual(config.ENTRY_SILVER_VOTES_NEEDED, 2)
        self.assertEqual(config.ENTRY_SILVER_VALIDATION_S, 0.0)

    def test_um_frame_sem_preto_nao_basta_para_confirmar(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(0., True, detection, 0.))[0])
        self.assertTrue(gate.update(EntryInference(.01, True, detection, 0.))[0])

    def test_modelo_sem_alinhamento_nao_aciona_resgate(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(0., False, detection, 0.))[0])
        self.assertFalse(gate.update(EntryInference(.03, False, detection, 0.))[0])
        self.assertFalse(gate.update(EntryInference(.06, True, detection, 0.))[0])
        self.assertTrue(gate.update(EntryInference(.07, True, detection, 0.))[0])
        self.assertEqual(gate.last_reason, "confirmada")

    def test_prata_com_linha_preta_renova_o_timeout(self):
        detection = EntryDetection((1, 2, 3, 4), .99)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(
            0., True, detection, 0., black_ahead=True,
        ))[0])
        self.assertEqual(
            gate.last_reason, "linha_preta_depois_da_prata_seguindo_linha")

        self.assertFalse(gate.update(EntryInference(
            .3, True, detection, 0., black_ahead=True,
        ))[0])
        self.assertFalse(gate.update(EntryInference(
            .79, True, detection, 0., black_ahead=False,
        ))[0])
        self.assertEqual(gate.votes, 0)
        self.assertEqual(gate.last_reason, "preto_apos_prata_seguindo_linha")
        self.assertEqual(gate.state, EntryGate.BLACK_FOLLOW)

    def test_prata_sem_preto_depois_confirma_o_resgate_em_dois_frames(self):
        detection = EntryDetection((1, 2, 3, 4), .70)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(
            0., True, detection, 0.,
            black_ahead=False, ramp_black_ahead=False))[0])
        self.assertTrue(gate.update(EntryInference(
            .01, True, detection, 0.,
            black_ahead=False, ramp_black_ahead=False))[0])

    def test_prata_sem_alinhamento_nao_entra(self):
        detection = EntryDetection((1, 2, 3, 4), .70)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(
            0., False, detection, 0., black_ahead=False))[0])
        self.assertFalse(gate.is_validating)
        self.assertEqual(gate.last_reason, "faixa_sem_linha_alinhada")

    def test_preto_entre_dois_positivos_zera_a_candidatura_anterior(self):
        detection = EntryDetection((1, 2, 3, 4), .70)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(
            0., True, detection, 0., black_ahead=True))[0])
        # O positivo posterior é um voto novo; não soma com o de antes do
        # trecho preto da rampa.
        self.assertFalse(gate.update(EntryInference(
            1.02, True, detection, 0., black_ahead=False))[0])
        self.assertEqual(gate.votes, 1)
        self.assertTrue(gate.update(EntryInference(
            1.03, True, detection, 0., black_ahead=False))[0])

    def test_prata_com_preto_da_rampa_bloqueia_nova_prata_por_um_segundo(self):
        detection = EntryDetection((1, 2, 3, 4), .99)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(
            0., True, detection, 0.,
            black_ahead=False, ramp_black_ahead=True,
        ))[0])
        self.assertEqual(
            gate.last_reason, "preto_rampa_depois_da_prata_seguindo_linha")

        timeout = config.ENTRY_BLACK_FOLLOW_TIMEOUT_S
        self.assertEqual(timeout, 1.0)
        self.assertFalse(gate.update(EntryInference(
            timeout - .01, True, detection, 0., black_ahead=False,
        ))[0])
        self.assertEqual(gate.last_reason, "preto_apos_prata_seguindo_linha")

        # Sem preto pelo limiar da rampa, uma prata real volta a votar depois
        # do timeout, para por um segundo e so entao confirma.
        self.assertFalse(gate.update(EntryInference(
            timeout + .01, True, detection, 0., black_ahead=False,
        ))[0])
        self.assertTrue(gate.update(EntryInference(
            timeout + .02, True, detection, 0., black_ahead=False,
        ))[0])
        self.assertEqual(gate.votes, config.ENTRY_SILVER_VOTES_NEEDED)

    def test_bloqueio_preto_expira_sem_inferencia_nova(self):
        detection = EntryDetection((1, 2, 3, 4), .99)
        gate = EntryGate()
        self.assertFalse(gate.update(EntryInference(
            0., True, detection, 0., black_ahead=True,
        ), now=10.)[0])
        self.assertEqual(gate.state, EntryGate.BLACK_FOLLOW)

        self.assertFalse(gate.update(
            None,
            now=10. + config.ENTRY_BLACK_FOLLOW_TIMEOUT_S - .01,
        )[0])
        self.assertEqual(gate.state, EntryGate.BLACK_FOLLOW)

        self.assertFalse(gate.update(
            None,
            now=10. + config.ENTRY_BLACK_FOLLOW_TIMEOUT_S,
        )[0])
        self.assertEqual(gate.state, EntryGate.IDLE)
        self.assertEqual(gate.last_reason, "preto_apos_prata_liberado")

    def test_limiar_de_preto_da_rampa_e_calibravel(self):
        self.assertEqual(
            config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT,
            [27, 27, 26],
        )
        calibrador = (SHADOW_ROOT / "tools" / "calibrar_cores.py").read_text(
            encoding="utf-8")
        self.assertIn('"3": ("black_max_ramp_down_top", "bgr_ceiling")',
                      calibrador)

    def test_reset_descarta_votos_e_o_timestamp_da_fase_anterior(self):
        detection = EntryDetection((1, 2, 3, 4), .7)
        gate = EntryGate()
        gate.update(EntryInference(1., True, detection, 0.))
        self.assertEqual(gate.votes, 1)

        gate.reset()

        self.assertEqual(gate.votes, 0)
        self.assertIsNone(gate.last_detection)
        self.assertEqual(gate.state, EntryGate.IDLE)
        # O novo trecho da pista pode comecar com timestamp menor no replay
        # ou em uma nova sessao de camera: ele nao herda o frame anterior.
        self.assertFalse(gate.update(EntryInference(.1, True, detection, 0.))[0])
        self.assertTrue(gate.update(EntryInference(.11, True, detection, 0.))[0])

    def test_rearme_invalida_o_worker_e_zera_o_portao_uma_unica_vez(self):
        class WorkerFalso:
            def __init__(self):
                self.resets = 0

            def reset(self):
                self.resets += 1

        detection = EntryDetection((1, 2, 3, 4), .7)
        pipeline = object.__new__(EntryPipeline)
        pipeline.worker = WorkerFalso()
        pipeline.gate = EntryGate()
        pipeline.gate.update(EntryInference(1., True, detection, 0.))
        pipeline.last_inference = object()
        pipeline._armed = True

        pipeline.set_armed(False)
        pipeline.set_armed(False)

        self.assertEqual(pipeline.worker.resets, 1)
        self.assertEqual(pipeline.gate.votes, 0)
        self.assertIsNone(pipeline.gate.last_detection)
        self.assertIsNone(pipeline.last_inference)
