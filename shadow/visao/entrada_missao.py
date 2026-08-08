"""Entrada no resgate pelo modelo de prata, isolada do segue-linha.

O worker mantém somente o frame mais recente, sem criar uma fila de imagens
velhas. Ele não altera comandos de motor: segue-linha, verde e lacuna continuam
sob responsabilidade exclusiva do controle. Quando instalado, NCNN é usado no
CPU ARM; o ONNX Runtime permanece como contingência segura.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np

import config
from shared.dados_compartilhados import (
    entry_armed, entry_silver_confirmed, entry_silver_detected,
    entry_silver_reason, entry_silver_votes, mission_mode)


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
    """YOLO de uma classe via NCNN, com contingência para ONNX Runtime."""

    def __init__(
        self, path=None, input_size=None, min_confidence=None, *,
        backend=None, ncnn_path=None,
    ):
        configured = config.ENTRY_MODEL_PATH if path is None else path
        self.path = self._resolve_path(configured)
        configured_ncnn = (
            config.ENTRY_NCNN_MODEL_PATH if ncnn_path is None else ncnn_path)
        self.ncnn_path = self._resolve_path(configured_ncnn)
        self.backend = (
            config.ENTRY_MODEL_BACKEND if backend is None else backend).lower()
        if self.backend not in {"auto", "ncnn", "onnx"}:
            raise ValueError(
                "backend da entrada deve ser 'auto', 'ncnn' ou 'onnx'")
        self._input_size_is_explicit = input_size is not None
        self.input_size = int(
            config.ENTRY_MODEL_INPUT if input_size is None else input_size)
        self.min_confidence = float(
            config.ENTRY_MODEL_MIN_CONFIDENCE
            if min_confidence is None else min_confidence)
        self._session = None
        self._input_name = None
        self._net = None
        self._ncnn = None
        self.active_backend = None

    @staticmethod
    def _resolve_path(configured):
        path = Path(configured)
        return path if path.is_absolute() else SHADOW_ROOT / path

    @property
    def active_path(self):
        return self.ncnn_path if self.active_backend == "ncnn" else self.path

    def load(self):
        ncnn_error = None
        if self.backend in {"auto", "ncnn"}:
            try:
                return self._load_ncnn()
            except RuntimeError as error:
                ncnn_error = error
                if self.backend == "ncnn":
                    raise
                print(f"[visão] NCNN indisponível ({error}); usando ONNX")
        if self.backend in {"auto", "onnx"}:
            try:
                return self._load_onnx()
            except RuntimeError as onnx_error:
                if ncnn_error is not None:
                    raise RuntimeError(
                        "não consegui abrir NCNN nem ONNX para a entrada") \
                        from onnx_error
                raise
        raise RuntimeError("backend de entrada não suportado")

    def _load_ncnn(self):
        self._select_input_size("ncnn")
        param = self.ncnn_path / "model.ncnn.param"
        weights = self.ncnn_path / "model.ncnn.bin"
        if not param.is_file() or not weights.is_file():
            raise RuntimeError(
                "modelo NCNN não encontrado: "
                f"{self.ncnn_path / 'model.ncnn.param'} / "
                f"{self.ncnn_path / 'model.ncnn.bin'}")
        try:
            import ncnn
        except ImportError as error:
            raise RuntimeError("pacote Python 'ncnn' não está instalado") from error
        net = ncnn.Net()
        net.opt.num_threads = config.ENTRY_MODEL_THREADS
        net.opt.use_vulkan_compute = False
        if net.load_param(str(param)) != 0:
            raise RuntimeError(f"não consegui abrir o parâmetro NCNN: {param}")
        if net.load_model(str(weights)) != 0:
            raise RuntimeError(f"não consegui abrir os pesos NCNN: {weights}")
        self._net = net
        self._ncnn = ncnn
        self.active_backend = "ncnn"
        return self

    def _load_onnx(self):
        self._select_input_size("onnx")
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
        self.active_backend = "onnx"
        return self

    def _select_input_size(self, backend):
        if self._input_size_is_explicit:
            return
        configured = (
            config.ENTRY_NCNN_MODEL_INPUT
            if backend == "ncnn" else config.ENTRY_MODEL_INPUT)
        self.input_size = int(configured)

    def detect(self, frame_bgr):
        if self.active_backend == "ncnn":
            if self._net is None or self._ncnn is None:
                raise RuntimeError("modelo NCNN de entrada não foi carregado")
        elif self._session is None:
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
        if self.active_backend == "ncnn":
            output = self._run_ncnn(input_tensor)
        else:
            output = self._session.run(
                None, {self._input_name: input_tensor})[0]
        output = np.squeeze(output)
        if output.ndim != 2:
            return None
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] < 5:
            raise RuntimeError("saída inesperada do modelo de entrada")
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

    def _run_ncnn(self, input_tensor):
        # O PNNX exportado pelo Ultralytics define NCHW `in0` → `out0`.
        # `clone()` impede que o NCNN referencie memória reutilizada pelo NumPy.
        image = np.ascontiguousarray(input_tensor[0])
        with self._net.create_extractor() as extractor:
            extractor.set_light_mode(True)
            if extractor.input("in0", self._ncnn.Mat(image).clone()) != 0:
                raise RuntimeError("não consegui enviar imagem para o NCNN")
            result, output = extractor.extract("out0")
        if result != 0:
            raise RuntimeError("não consegui obter resultado do NCNN")
        return np.array(output)


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
            target=self._run, name="shadow-entrada-modelo", daemon=True)

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
    print("[visão] modelo de entrada armado: "
          f"{pipeline.model.active_path.name} ({pipeline.model.active_backend})")
    return pipeline


def update_entry_silver(entry_gate, frame, captured_at, *, line_aligned=False):
    """Entrega o frame ao YOLO e publica somente resultados prontos."""
    if entry_gate is None:
        return
    if not entry_armed.value:
        entry_silver_detected.value = False
        return
    # A confirmação pertence ao processo de controle. Mantenha-a publicada
    # até ele parar, apagar o LED e solicitar o handoff; não deixe um poll sem
    # resultado apagar o único frame que confirmou a entrada.
    if entry_silver_confirmed.value:
        return
    entry_gate.submit(frame, captured_at, line_aligned)
    confirmed, detection = entry_gate.poll()
    entry_silver_detected.value = detection is not None
    entry_silver_votes.value = entry_gate.votes
    entry_silver_reason.value = entry_gate.last_reason
    if confirmed and not entry_silver_confirmed.value:
        print("[visão] faixa PRATA confirmada pelo modelo "
              f"({entry_gate.votes}/{config.ENTRY_SILVER_VOTE_WINDOW} votos)")
    entry_silver_confirmed.value = confirmed
