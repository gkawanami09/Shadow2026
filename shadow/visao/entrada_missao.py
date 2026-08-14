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
    ENTRY_SILVER_BLACK_FOLLOW, ENTRY_SILVER_IDLE, ENTRY_SILVER_VALIDATING,
    entry_armed, entry_silver_confirmed, entry_silver_detected,
    entry_silver_reason, entry_silver_state, entry_silver_votes, mission_mode)


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
    # Preto no corredor central alem da candidata, no mesmo frame capturado.
    # ``None`` preserva os consumidores antigos; no percurso real sempre vem
    # ``True`` ou ``False`` da visao da linha.
    black_ahead: bool | None = None
    # Mesmo contexto, mas medido pelo limiar de preto exclusivo da rampa.
    ramp_black_ahead: bool | None = None


def has_black_after_entry_detection(mask, detection):
    """Returns black-line evidence beyond the silver detection.

    The robot moves from the bottom toward the top of the camera image. Only
    black above the YOLO box can be a continuation after the silver strip;
    black below it is the approach line and must not reject a real rescue.
    """
    if detection is None or mask is None or mask.ndim != 2:
        return False

    height, width = mask.shape
    if height == 0 or width == 0:
        return False
    try:
        _x, silver_top, _box_width, _box_height = detection.bbox
        silver_top = float(silver_top)
    except (TypeError, ValueError):
        return False

    guard_px = max(
        1, int(round(height * config.ENTRY_BLACK_AFTER_SILVER_GUARD_RATIO)))
    after_end = min(height, max(0, int(np.floor(silver_top)) - guard_px))
    x_start = int(width * config.GAP_AHEAD_X_MIN)
    x_end = int(width * config.GAP_AHEAD_X_MAX)
    after_silver = mask[:after_end, x_start:x_end]
    if not after_silver.size:
        return False

    row_fill = np.count_nonzero(after_silver, axis=1) / after_silver.shape[1]
    return bool(
        np.mean(row_fill >= config.GAP_AHEAD_ROW_FILL)
        >= config.GAP_AHEAD_ROW_PERSISTENCE
    )


def has_ramp_black_near_entry_detection(mask, detection):
    """Detecta a barra preta larga da rampa próxima à falsa faixa prata.

    A linha regular que chega à sala é estreita e central. Já a saída da
    rampa aparece como uma barra transversal escura. Medir a largura em uma
    janela grande separa as duas sem exigir outra inferência de prata.
    """
    if detection is None or mask is None or mask.ndim != 2:
        return False
    height, width = mask.shape
    if height == 0 or width == 0:
        return False
    try:
        _x, silver_top, _box_width, box_height = detection.bbox
        silver_top = float(silver_top)
        silver_bottom = silver_top + float(box_height)
    except (TypeError, ValueError):
        return False

    margin = max(1, int(round(
        height * config.ENTRY_RAMP_BLACK_NEAR_BOX_MARGIN_RATIO)))
    y_start = max(0, int(np.floor(silver_top)) - margin)
    y_end = min(height, int(np.ceil(silver_bottom)) + margin)
    x_start = int(width * config.ENTRY_RAMP_BLACK_X_MIN)
    x_end = int(width * config.ENTRY_RAMP_BLACK_X_MAX)
    nearby = mask[y_start:y_end, x_start:x_end]
    if not nearby.size:
        return False
    row_fill = np.count_nonzero(nearby, axis=1) / nearby.shape[1]
    rows_required = max(1, int(round(
        height * config.ENTRY_RAMP_BLACK_MIN_ROWS_RATIO)))
    return bool(np.count_nonzero(
        row_fill >= config.ENTRY_RAMP_BLACK_ROW_FILL) >= rows_required)


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
        # Confianca bruta do melhor candidato, inclusive quando ainda fica
        # abaixo do limiar e por isso nao vira uma deteccao.
        self.last_confidence = None

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
        self.last_confidence = None
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
        self.last_confidence = confidence
        if confidence < self.min_confidence:
            return None
        center_x, center_y, box_width, box_height = output[index, :4]
        x = (float(center_x) - float(box_width) / 2 - offset_x) / scale
        y = (float(center_y) - float(box_height) / 2 - offset_y) / scale
        box_width = float(box_width) / scale
        box_height = float(box_height) / scale
        # O modelo pode dar confianca alta para reflexos e objetos claros da
        # pista. A entrada real sempre e uma faixa larga, horizontal.
        if (box_width < width * config.ENTRY_SILVER_MIN_WIDTH_RATIO
                or box_width / max(1., box_height)
                < config.ENTRY_SILVER_MIN_ASPECT_RATIO):
            return None
        return EntryDetection(
            bbox=(x, y, box_width, box_height),
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
        # Janela limitada: preserva a faixa curta vista em alta velocidade sem
        # deixar a inferencia acumular uma fila longa de imagens antigas.
        self._pending = deque(maxlen=config.ENTRY_MODEL_PENDING_FRAMES)
        self._latest = None
        self._delivered_timestamp = None
        self._generation = 0
        self._error = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="shadow-entrada-modelo", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def submit(
        self,
        frame,
        timestamp,
        line_aligned,
        black_mask=None,
        ramp_black_mask=None,
    ):
        # A cópia impede a câmera de reutilizar o buffer durante a inferência.
        with self._condition:
            self._pending.append((
                self._generation,
                frame.copy(),
                float(timestamp),
                bool(line_aligned),
                None if black_mask is None else black_mask.copy(),
                None if ramp_black_mask is None else ramp_black_mask.copy(),
            ))
            self._condition.notify()

    def reset(self):
        """Invalida trabalhos e resultados de uma fase anterior da pista."""
        with self._condition:
            self._generation += 1
            self._pending.clear()
            self._latest = None
            self._delivered_timestamp = None
            self._condition.notify_all()

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
            self._pending.clear()
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                (generation, frame, timestamp, line_aligned,
                 black_mask, ramp_black_mask) = self._pending.popleft()
            try:
                started = time.perf_counter()
                detection = self.model.detect(frame)
                inference_ms = (time.perf_counter() - started) * 1000.
                black_ahead = has_black_after_entry_detection(
                    black_mask, detection)
                # A barra da rampa pode ficar um pouco mais clara que o teto
                # exclusivo calibrado. A geometria transversal larga abaixo
                # e' o discriminante; por isso aceita a evidencia de qualquer
                # uma das duas mascaras, sem confundir a linha estreita real.
                ramp_black_ahead = (
                    has_ramp_black_near_entry_detection(
                        black_mask, detection)
                    or has_ramp_black_near_entry_detection(
                        ramp_black_mask, detection)
                )
            except Exception as error:  # surfaced in the vision process
                with self._condition:
                    self._error = error
                return
            with self._condition:
                if generation != self._generation:
                    # O 180 (ou outro rearme) invalidou este frame enquanto a
                    # inferencia estava em andamento. Nunca o entregue na
                    # fase seguinte.
                    continue
                self._latest = EntryInference(
                    timestamp,
                    line_aligned,
                    detection,
                    inference_ms,
                    black_ahead,
                    ramp_black_ahead,
                )


class EntryGate:
    """Confirma a prata ou bloqueia quando a pista preta continua depois."""

    IDLE = ENTRY_SILVER_IDLE
    VALIDATING = ENTRY_SILVER_VALIDATING
    BLACK_FOLLOW = ENTRY_SILVER_BLACK_FOLLOW

    def __init__(self):
        self._hits = deque(maxlen=config.ENTRY_SILVER_VOTE_WINDOW)
        self.last_detection = None
        self.last_reason = "início"
        self._last_timestamp = None
        self._state = self.IDLE
        self._validation_started_at = None
        self._black_follow_until = float("-inf")

    @property
    def votes(self):
        return sum(self._hits)

    @property
    def state(self):
        return self._state

    @property
    def is_validating(self):
        return self._state == self.VALIDATING

    def _clear_validation(self):
        self._hits.clear()
        self._validation_started_at = None

    def _start_black_follow(self, timestamp, source):
        self._clear_validation()
        self._state = self.BLACK_FOLLOW
        self._black_follow_until = (
            timestamp + config.ENTRY_BLACK_FOLLOW_TIMEOUT_S)
        self.last_reason = f"{source}_seguindo_linha"

    @staticmethod
    def _black_source(inference):
        if inference.ramp_black_ahead is True:
            return "preto_rampa_depois_da_prata"
        if inference.black_ahead is True:
            return "linha_preta_depois_da_prata"
        return None

    def reset(self):
        """Esquece votos, prazos e caixas da fase de percurso anterior."""
        self._clear_validation()
        self.last_detection = None
        self.last_reason = "reiniciado"
        self._last_timestamp = None
        self._state = self.IDLE
        self._black_follow_until = float("-inf")

    def tick(self, now):
        """Expira estados mesmo se o worker ainda nao tiver outro resultado."""
        now = float(now)
        if self._state == self.VALIDATING:
            if now - self._validation_started_at >= config.ENTRY_SILVER_VALIDATION_S:
                # Sem uma nova leitura de prata nao existe prova suficiente
                # para entrar. Libera o controle, mas nunca confirma.
                self._clear_validation()
                self._state = self.IDLE
                self.last_reason = "validacao_prata_sem_resultado"
        elif (self._state == self.BLACK_FOLLOW
              and now >= self._black_follow_until):
            self._state = self.IDLE
            self._black_follow_until = float("-inf")
            self.last_reason = "preto_apos_prata_liberado"
        return self._state

    def update(self, inference, now=None):
        """Recebe uma inferencia pronta e usa sua chegada para os prazos.

        O timestamp de captura só identifica o frame. Quando a configuração
        pede observação parada, o prazo começa na chegada da inferência.
        """
        if inference is None:
            if now is not None:
                self.tick(now)
            return False, self.last_detection
        if (self._last_timestamp is not None
                and inference.timestamp <= self._last_timestamp):
            self.last_reason = "frame_repetido"
            return False, self.last_detection
        self._last_timestamp = inference.timestamp
        observed_at = (
            inference.timestamp if now is None else float(now))
        if inference.detection is not None:
            self.last_detection = inference.detection

        black_source = self._black_source(inference)
        if config.ENTRY_REJECT_SILVER_WITH_BLACK_AHEAD and black_source:
            # Preto alem da caixa, pelos limiares normal ou de rampa, prova
            # que ainda existe pista. Ele sempre vence uma candidata prata.
            self._start_black_follow(observed_at, black_source)
            return False, inference.detection

        if self._state == self.BLACK_FOLLOW:
            if observed_at < self._black_follow_until:
                self.last_reason = "preto_apos_prata_seguindo_linha"
                return False, inference.detection
            self._state = self.IDLE
            self._black_follow_until = float("-inf")

        if self._state == self.VALIDATING:
            if inference.detection is None:
                # O robo esta parado: se a prata some da camera durante a
                # observacao, falha fechado e nao entra no resgate.
                self._clear_validation()
                self._state = self.IDLE
                self.last_reason = "prata_sumiu_na_validacao"
                return False, None

            self._hits.append(True)
            elapsed = observed_at - self._validation_started_at
            if elapsed + 1e-9 < config.ENTRY_SILVER_VALIDATION_S:
                self.last_reason = "validando_prata_parado"
                return False, inference.detection

            confirmed = self.votes >= config.ENTRY_SILVER_VOTES_NEEDED
            self._validation_started_at = None
            self._state = self.IDLE
            if not confirmed:
                self._hits.clear()
            self.last_reason = (
                "confirmada" if confirmed else "prata_nao_confirmada")
            return confirmed, inference.detection

        if inference.detection is None:
            self._hits.append(False)
            self.last_reason = "modelo_sem_faixa"
            return False, None
        if not inference.line_aligned:
            self._hits.append(False)
            self.last_reason = "faixa_sem_linha_alinhada"
            return False, inference.detection

        # O alinhamento é exigido neste primeiro positivo para não entregar a
        # sala com o robô atravessado na faixa.
        self._clear_validation()
        self._hits.append(True)
        # Com a proteção de preto depois da caixa ativa, um único prata
        # alinhado é suficiente. Isso evita o robô cruzar toda a faixa antes
        # de obter uma segunda inferência. Se a configuração pedir mais de um
        # voto, conserva o fluxo antigo de parar e observar.
        if self.votes >= config.ENTRY_SILVER_VOTES_NEEDED:
            self._state = self.IDLE
            self.last_reason = "confirmada"
            return True, inference.detection
        self._state = self.VALIDATING
        self._validation_started_at = observed_at
        self.last_reason = "validando_prata_parado"
        return False, inference.detection


class EntryPipeline:
    """Modelo assíncrono + confirmação temporal, dono do ciclo de vida."""

    def __init__(self):
        self.model = EntryModel().load()
        self.worker = EntryModelWorker(self.model).start()
        self.gate = EntryGate()
        self.last_inference = None
        self._armed = False

    @property
    def last_detection(self):
        return self.gate.last_detection

    @property
    def last_reason(self):
        return self.gate.last_reason

    @property
    def votes(self):
        return self.gate.votes

    @property
    def state(self):
        return self.gate.state

    def submit(
        self,
        frame,
        timestamp,
        line_aligned,
        *,
        black_mask=None,
        ramp_black_mask=None,
    ):
        self.worker.submit(
            frame,
            timestamp,
            line_aligned,
            black_mask,
            ramp_black_mask,
        )

    def set_armed(self, armed):
        """Separa o rearme atual de inferencias/votos anteriores."""
        armed = bool(armed)
        if armed == self._armed:
            return
        self.worker.reset()
        self.gate.reset()
        self.last_inference = None
        self._armed = armed

    def poll(self):
        inference = self.worker.poll()
        if inference is not None:
            self.last_inference = inference
        # O Gate e dono dos prazos; ele nao compara o relogio de captura da
        # visao com o relogio do processo de controle.
        return self.gate.update(inference, now=time.perf_counter())

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


def update_entry_silver(
    entry_gate,
    frame,
    captured_at,
    *,
    line_aligned=False,
    black_mask=None,
    ramp_black_mask=None,
):
    """Entrega o frame ao YOLO e publica somente resultados prontos."""
    if entry_gate is None:
        return
    if not entry_armed.value:
        entry_gate.set_armed(False)
        entry_silver_detected.value = False
        entry_silver_confirmed.value = False
        entry_silver_votes.value = 0
        entry_silver_reason.value = "entrada desarmada"
        entry_silver_state.value = ENTRY_SILVER_IDLE
        return
    entry_gate.set_armed(True)
    # A confirmação pertence ao processo de controle. Mantenha-a publicada
    # até ele parar, apagar o LED e solicitar o handoff; não deixe um poll sem
    # resultado apagar o único frame que confirmou a entrada.
    if entry_silver_confirmed.value:
        return
    entry_gate.submit(
        frame,
        captured_at,
        line_aligned,
        black_mask=black_mask,
        ramp_black_mask=ramp_black_mask,
    )
    confirmed, detection = entry_gate.poll()
    state = getattr(entry_gate, "state", ENTRY_SILVER_IDLE)
    entry_silver_detected.value = (
        detection is not None or state == ENTRY_SILVER_VALIDATING)
    entry_silver_votes.value = entry_gate.votes
    entry_silver_reason.value = entry_gate.last_reason
    entry_silver_state.value = state
    if confirmed and not entry_silver_confirmed.value:
        print("[visão] faixa PRATA confirmada pelo modelo "
              f"({entry_gate.votes}/{config.ENTRY_SILVER_VOTE_WINDOW} votos)")
    entry_silver_confirmed.value = confirmed
