"""Entrada no resgate por ``entrada.onnx``, somente durante a missão.

O modelo é aberto pelo processo que já possui a câmera de linha. Ele nunca é
criado por ``main.py`` e para de inferir assim que a entrada é confirmada.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

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


class EntryModel:
    """YOLO de uma classe para a faixa prata, usando OpenCV DNN."""

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
        self.net = None

    def load(self):
        if not self.path.is_file():
            raise RuntimeError(f"modelo de entrada não encontrado: {self.path}")
        import cv2
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.path))
        except cv2.error as error:
            raise RuntimeError(
                f"não consegui abrir o modelo de entrada {self.path}: {error}") from error
        return self

    def detect(self, frame):
        if self.net is None:
            raise RuntimeError("modelo de entrada não foi carregado")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame BGR inválido")
        import cv2
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        offset_x = (self.input_size - resized.shape[1]) // 2
        offset_y = (self.input_size - resized.shape[0]) // 2
        canvas[offset_y:offset_y + resized.shape[0],
               offset_x:offset_x + resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, size=(self.input_size, self.input_size),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        output = np.squeeze(self.net.forward())
        if output.ndim != 2:
            return None
        if output.shape[0] < output.shape[1]:
            output = output.T
        # Modelo confirmado: YOLO de UMA classe: cx, cy, w, h, confiança.
        if output.shape[1] < 5:
            raise RuntimeError("saída inesperada do modelo entrada.onnx")
        index = int(np.argmax(output[:, 4]))
        confidence = float(output[index, 4])
        if confidence < self.min_confidence:
            return None
        cx, cy, box_w, box_h = output[index, :4]
        x = (float(cx) - float(box_w) / 2 - offset_x) / scale
        y = (float(cy) - float(box_h) / 2 - offset_y) / scale
        return EntryDetection(
            bbox=(x, y, float(box_w) / scale, float(box_h) / scale),
            confidence=confidence)


class EntryGate:
    """Confirma dois frames do modelo, ambos com o robô alinhado à linha."""

    def __init__(self, model=None):
        self.model = EntryModel() if model is None else model
        self._hits = deque(maxlen=config.ENTRY_SILVER_VOTE_WINDOW)
        self.last_detection = None
        self.last_reason = "início"
        self._last_timestamp = None

    @property
    def votes(self):
        return sum(self._hits)

    def update(self, frame, timestamp, line_aligned):
        timestamp = float(timestamp)
        if (self._last_timestamp is not None
                and timestamp <= self._last_timestamp):
            self.last_reason = "frame_repetido"
            return False, self.last_detection
        self._last_timestamp = timestamp
        # A inferência de 640×640 só é necessária quando o robô está reto na
        # ponta da linha. No seguimento normal ela não consome CPU nem reduz
        # a cadência do pipeline de linha.
        if not line_aligned:
            self.last_detection = None
            self._hits.append(False)
            self.last_reason = "linha_nao_alinhada"
            return False, None
        detection = self.model.detect(frame)
        self.last_detection = detection
        if detection is None:
            self._hits.append(False)
            self.last_reason = "modelo_sem_faixa"
            return False, None
        self._hits.append(True)
        self.last_reason = "confirmada" if self.votes >= config.ENTRY_SILVER_VOTES_NEEDED else "votando"
        return self.votes >= config.ENTRY_SILVER_VOTES_NEEDED, detection


def build_entry_gate():
    """Cria e carrega o modelo apenas na fase de segue-linha da missão."""
    if not mission_mode.value or not config.ENTRY_SILVER_ENABLED:
        return None
    gate = EntryGate()
    gate.model.load()
    print(f"[visão] modelo de entrada armado: {gate.model.path.name}")
    return gate


def update_entry_silver(entry_gate, frame, captured_at, *, line_aligned=False):
    """Executa o modelo e publica a decisão para o processo de controle."""
    if entry_gate is None:
        return
    if not entry_armed.value:
        entry_silver_detected.value = False
        return
    confirmed, detection = entry_gate.update(frame, captured_at, line_aligned)
    entry_silver_detected.value = detection is not None
    entry_silver_votes.value = entry_gate.votes
    entry_silver_reason.value = entry_gate.last_reason
    if confirmed and not entry_silver_confirmed.value:
        print(f"[visão] faixa PRATA confirmada pelo modelo ({entry_gate.votes}/{config.ENTRY_SILVER_VOTE_WINDOW} votos)")
    entry_silver_confirmed.value = confirmed
