"""Entrada no resgate por ``entrada.onnx``, isolada do segue-linha.

O YOLO roda em uma única thread própria e sempre recebe somente o frame mais
recente. Portanto uma inferência lenta não reduz o FPS da linha, não cria fila
de imagens velhas e não deixa a lógica verde/preto decidir antes da entrada.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np

import config
from shared.dados_compartilhados import (
    entry_armed, entry_model_priority, entry_silver_confirmed,
    entry_silver_detected, entry_silver_reason, entry_silver_votes,
    mission_mode)


SHADOW_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EntryDetection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class EntryInference:
    timestamp: float
    line_aligned: bool
    detection: EntryDetection | None
    inference_ms: float


class EntryModel:
    """YOLO de uma classe, pelo mesmo ONNX Runtime do modelo de vítimas."""

    def __init__(self, path=None, input_size=None, min_confidence=None):
        configured = config.ENTRY_MODEL_PATH if path is None else path
        self.path = Path(configured)
        if not self.path.is_absolute():
            self.path = SHADOW_ROOT / self.path
        self.input_size = int(
            config.ENTRY_MODEL_INPUT if input_size is None else input_size)
        self.min_confidence = float(
            config.ENTRY_MODEL_MIN_CONFIDENCE
            if min_confidence is None else min_confidence)
        self._session = None
        self._input_name = None

    def load(self):
        if not self.path.is_file():
            raise RuntimeError(f"modelo de entrada não encontrado: {self.path}")
        try:
            import onnxruntime
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime não está instalado no Pi; ele é necessário para "
                "entrada.onnx e para o modelo de vítimas.") from error
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = config.ENTRY_MODEL_THREADS
        self._session = onnxruntime.InferenceSession(
            str(self.path), sess_options=options,
            providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        return self

    def detect(self, frame_bgr):
        if self._session is None:
            raise RuntimeError("modelo de entrada não foi carregado")
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame BGR inválido")
        import cv2
        height, width = frame_bgr.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized = cv2.resize(
            frame_bgr, (round(width * scale), round(height * scale)))
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        offset_x = (self.input_size - resized.shape[1]) // 2
        offset_y = (self.input_size - resized.shape[0]) // 2
        canvas[offset_y:offset_y + resized.shape[0],
               offset_x:offset_x + resized.shape[1]] = resized
        # A câmera e o OpenCV usam BGR; o export YOLO recebe RGB normalizado.
        input_tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]
        output = np.squeeze(
            self._session.run(None, {self._input_name: input_tensor})[0])
        if output.ndim != 2:
            return None
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] < 5:
            raise RuntimeError("saída inesperada do modelo entrada.onnx")
        index = int(np.argmax(output[:, 4]))
        confidence = float(output[index, 4])
        if confidence < self.min_confidence:
            return None
        center_x, center_y, box_width, box_height = output[index, :4]
        x = (float(center_x) - float(box_width) / 2 - offset_x) / scale
        y = (float(center_y) - float(box_height) / 2 - offset_y) / scale
        return EntryDetection(
            bbox=(x, y, float(box_width) / scale, float(box_height) / scale),
            confidence=confidence)


class EntryModelWorker:
    """Inferência assíncrona sem backlog: só o quadro mais novo sobrevive."""

    def __init__(self, model):
        self.model = model
        self._condition = threading.Condition()
        self._pending = None
        self._latest = None
        self._delivered_timestamp = None
        self._error = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="shadow-entrada-onnx", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame, timestamp, line_aligned):
        # A cópia impede a câmera de reutilizar o buffer durante a inferência.
        with self._condition:
            self._pending = (frame.copy(), float(timestamp), bool(line_aligned))
            self._condition.notify()

    def poll(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("falha ao executar entrada.onnx") from self._error
            if self._latest is None:
                return None
            if self._latest.timestamp == self._delivered_timestamp:
                return None
            self._delivered_timestamp = self._latest.timestamp
            return self._latest

    def close(self):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                frame, timestamp, line_aligned = self._pending
                self._pending = None
            try:
                started = time.perf_counter()
                detection = self.model.detect(frame)
                inference_ms = (time.perf_counter() - started) * 1000.
            except Exception as error:  # surfaced in the vision process
                with self._condition:
                    self._error = error
                return
            with self._condition:
                self._latest = EntryInference(
                    timestamp, line_aligned, detection, inference_ms)


class EntryGate:
    """Confirma o modelo apenas em capturas alinhadas à linha preta."""

    def __init__(self):
        self._hits = deque(maxlen=config.ENTRY_SILVER_VOTE_WINDOW)
        self.last_detection = None
        self.last_reason = "início"
        self._last_timestamp = None

    @property
    def votes(self):
        return sum(self._hits)

    def update(self, inference):
        if inference is None:
            return False, self.last_detection
        if (self._last_timestamp is not None
                and inference.timestamp <= self._last_timestamp):
            self.last_reason = "frame_repetido"
            return False, self.last_detection
        self._last_timestamp = inference.timestamp
        self.last_detection = inference.detection
        if inference.detection is None:
            self._hits.append(False)
            self.last_reason = "modelo_sem_faixa"
            return False, None
        if not inference.line_aligned:
            self._hits.append(False)
            self.last_reason = "faixa_sem_linha_alinhada"
            return False, inference.detection
        self._hits.append(True)
        fast = inference.detection.confidence >= config.ENTRY_MODEL_FAST_CONFIDENCE
        confirmed = fast or self.votes >= config.ENTRY_SILVER_VOTES_NEEDED
        self.last_reason = (
            "confirmada_rapida" if fast
            else "confirmada" if confirmed else "votando")
        return confirmed, inference.detection


class EntryPipeline:
    """Modelo assíncrono + confirmação temporal, dono do ciclo de vida."""

    def __init__(self):
        self.model = EntryModel().load()
        self.worker = EntryModelWorker(self.model).start()
        self.gate = EntryGate()
        self.last_inference = None

    @property
    def last_detection(self):
        return self.gate.last_detection

    @property
    def last_reason(self):
        return self.gate.last_reason

    @property
    def votes(self):
        return self.gate.votes

    def submit(self, frame, timestamp, line_aligned):
        self.worker.submit(frame, timestamp, line_aligned)

    def poll(self):
        inference = self.worker.poll()
        if inference is not None:
            self.last_inference = inference
        return self.gate.update(inference)

    def close(self):
        self.worker.close()


def build_entry_gate():
    """Só cria o worker durante a fase de linha da missão completa."""
    if not mission_mode.value or not config.ENTRY_SILVER_ENABLED:
        return None
    pipeline = EntryPipeline()
    print(f"[visão] modelo de entrada armado: {pipeline.model.path.name}")
    return pipeline


def update_entry_silver(
    entry_gate, frame, captured_at, *, line_aligned=False,
    wait_for_result=False,
):
    """Entrega o frame ao YOLO e publica somente resultados prontos."""
    if entry_gate is None:
        return
    if not entry_armed.value:
        entry_silver_detected.value = False
        entry_model_priority.value = False
        return
    # A confirmação pertence ao processo de controle. Mantenha-a publicada
    # até ele parar, apagar o LED e solicitar o handoff; não deixe um poll sem
    # resultado apagar o único frame que confirmou a entrada.
    if entry_silver_confirmed.value:
        entry_model_priority.value = True
        return
    entry_gate.submit(frame, captured_at, line_aligned)
    confirmed, detection = entry_gate.poll()
    entry_silver_detected.value = detection is not None
    entry_silver_votes.value = entry_gate.votes
    entry_silver_reason.value = entry_gate.last_reason
    # Ao perder uma linha previamente alinhada, espera a inferência pendente.
    # Se o modelo já encontrou prata, ele também interrompe verde/preto.
    entry_model_priority.value = bool(
        wait_for_result
        or (detection is not None
            and entry_gate.last_reason != "faixa_sem_linha_alinhada")
    )
    if confirmed and not entry_silver_confirmed.value:
        print("[visão] faixa PRATA confirmada pelo modelo "
              f"({entry_gate.votes}/{config.ENTRY_SILVER_VOTE_WINDOW} votos)")
    entry_silver_confirmed.value = confirmed
