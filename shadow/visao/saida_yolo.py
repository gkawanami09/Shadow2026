"""Detector treinado da saída, com geometria opcional para o alinhamento.

O YOLO diz *qual* objeto é a faixa de saída. A caixa de um YOLO normal não
informa com segurança a inclinação da faixa, então o detector geométrico
existente é usado apenas quando encontra uma faixa dentro da caixa prevista.
Isso preserva a correção de yaw sem permitir que uma mancha preta fora da
detecção do modelo commande o robô.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time

import numpy as np

import config_resgate as cfg
from visao.faixa_saida import BlackExitDetector, ExitStripeDetection


SHADOW_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExitModelDetection:
    """Caixa da única classe do modelo treinado: ``Saida``."""

    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center_x(self):
        return self.bbox[0] + self.bbox[2] / 2.0

    @property
    def center_y(self):
        return self.bbox[1] + self.bbox[3] / 2.0


class ExitModelError(RuntimeError):
    """O modelo de saída não pôde ser aberto em nenhum backend."""


class ExitModel:
    """YOLO de uma classe via NCNN, com fallback explícito para ONNX."""

    def __init__(self, path=None, input_size=None, *, backend=None,
                 ncnn_path=None, min_confidence=None):
        configured_path = cfg.EXIT_MODEL_PATH if path is None else path
        configured_ncnn = (
            cfg.EXIT_NCNN_MODEL_PATH if ncnn_path is None else ncnn_path)
        self.path = self._resolve(configured_path)
        self.ncnn_path = self._resolve(configured_ncnn)
        self.backend = (
            cfg.EXIT_MODEL_BACKEND if backend is None else backend).lower()
        if self.backend not in {"auto", "ncnn", "onnx"}:
            raise ValueError("backend da saída deve ser auto, ncnn ou onnx")
        self._input_size_is_explicit = input_size is not None
        self.input_size = int(
            cfg.EXIT_MODEL_INPUT if input_size is None else input_size)
        self.min_confidence = float(
            cfg.EXIT_MODEL_MIN_CONFIDENCE
            if min_confidence is None else min_confidence)
        self._net = None
        self._ncnn = None
        self._session = None
        self._input_name = None
        self.active_backend = None
        self.last_raw_confidence = 0.0
        self.last_raw_detection = None
        self.last_inference_ms = 0.0

    @staticmethod
    def _resolve(configured):
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
            except ExitModelError as error:
                ncnn_error = error
                if self.backend == "ncnn":
                    raise
                print(f"[saida] NCNN indisponível ({error}); usando ONNX")
        if self.backend in {"auto", "onnx"}:
            try:
                return self._load_onnx()
            except ExitModelError as error:
                if ncnn_error is not None:
                    raise ExitModelError(
                        "não consegui abrir NCNN nem ONNX para a saída") \
                        from error
                raise
        raise ExitModelError("backend de saída não suportado")

    def _load_ncnn(self):
        self._select_input_size("ncnn")
        param = self.ncnn_path / "model.ncnn.param"
        weights = self.ncnn_path / "model.ncnn.bin"
        if not param.is_file() or not weights.is_file():
            raise ExitModelError(
                "modelo NCNN de saída não encontrado: "
                f"{param} / {weights}")
        try:
            import ncnn
        except ImportError as error:
            raise ExitModelError("pacote Python 'ncnn' não está instalado") \
                from error
        net = ncnn.Net()
        net.opt.num_threads = int(cfg.EXIT_MODEL_THREADS)
        net.opt.use_vulkan_compute = False
        if cfg.EXIT_MODEL_USE_FP16:
            # O Cortex-A76 do Pi 5 possui FP16 nativo. O NCNN mantem o mesmo
            # modelo e usa os kernels ARM de meia precisao durante a inferencia.
            for option_name in (
                "use_fp16_packed",
                "use_fp16_storage",
                "use_fp16_arithmetic",
            ):
                if hasattr(net.opt, option_name):
                    setattr(net.opt, option_name, True)
        if net.load_param(str(param)) != 0 or net.load_model(str(weights)) != 0:
            raise ExitModelError("não consegui abrir o modelo NCNN de saída")
        self._net = net
        self._ncnn = ncnn
        self.active_backend = "ncnn"
        return self

    def _load_onnx(self):
        self._select_input_size("onnx")
        if not self.path.is_file():
            raise ExitModelError(f"modelo ONNX de saída não encontrado: {self.path}")
        try:
            import onnxruntime
        except ImportError as error:
            raise ExitModelError("onnxruntime não está instalado") from error
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = int(cfg.EXIT_MODEL_THREADS)
        self._session = onnxruntime.InferenceSession(
            str(self.path), sess_options=options,
            providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.active_backend = "onnx"
        return self

    def _select_input_size(self, backend):
        if not self._input_size_is_explicit:
            self.input_size = int(
                cfg.EXIT_NCNN_MODEL_INPUT
                if backend == "ncnn" else cfg.EXIT_MODEL_INPUT)

    def detect(self, frame_bgr):
        if self.active_backend == "ncnn":
            if self._net is None or self._ncnn is None:
                raise ExitModelError("modelo NCNN de saída não foi carregado")
        elif self._session is None:
            raise ExitModelError("modelo de saída não foi carregado")
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame BGR inválido para o modelo de saída")
        started = time.perf_counter()
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
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(
            tensor.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        output = (
            self._run_ncnn(tensor)
            if self.active_backend == "ncnn"
            else self._session.run(None, {self._input_name: tensor})[0]
        )
        detection = self._decode(
            output, scale, offset_x, offset_y, width, height)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return detection

    def _decode(self, output, scale, offset_x, offset_y, width, height):
        predictions = np.squeeze(output)
        self.last_raw_confidence = 0.0
        self.last_raw_detection = None
        if predictions.ndim != 2:
            return None
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise ExitModelError("saída inesperada do modelo de saída")
        index = int(np.argmax(predictions[:, 4]))
        confidence = float(predictions[index, 4])
        self.last_raw_confidence = confidence
        center_x, center_y, box_width, box_height = predictions[index, :4]
        x = (float(center_x) - float(box_width) / 2.0 - offset_x) / scale
        y = (float(center_y) - float(box_height) / 2.0 - offset_y) / scale
        x = min(max(x, 0.0), float(width))
        y = min(max(y, 0.0), float(height))
        box_width = min(max(float(box_width) / scale, 0.0), float(width) - x)
        box_height = min(max(float(box_height) / scale, 0.0), float(height) - y)
        if box_width <= 0.0 or box_height <= 0.0:
            return None
        detection = ExitModelDetection(
            (x, y, box_width, box_height), confidence)
        self.last_raw_detection = detection
        return detection if confidence >= self.min_confidence else None

    def _run_ncnn(self, tensor):
        image = np.ascontiguousarray(tensor[0])
        with self._net.create_extractor() as extractor:
            extractor.set_light_mode(True)
            if extractor.input("in0", self._ncnn.Mat(image).clone()) != 0:
                raise ExitModelError("não consegui enviar a imagem para NCNN")
            result, output = extractor.extract("out0")
        if result != 0:
            raise ExitModelError("não consegui obter a saída do NCNN")
        return np.array(output)


class ModelGuidedExitDetector:
    """Modelo para aparência; geometria limitada à região detectada."""

    def __init__(self, model, geometry_detector=None):
        self.model = model
        self.geometry_detector = (
            BlackExitDetector() if geometry_detector is None
            else geometry_detector)
        self.last_model_detection = None
        self.last_geometry_detection = None
        self.fast_lock_confidence = float(cfg.EXIT_MODEL_FAST_LOCK_CONFIDENCE)

    def detect(self, frame_bgr, timestamp=None):
        model_detection = self.model.detect(frame_bgr)
        self.last_model_detection = model_detection
        self.last_geometry_detection = None
        if model_detection is None:
            return None

        if cfg.EXIT_MODEL_USE_GEOMETRY:
            geometry = self.geometry_detector.detect(
                frame_bgr, timestamp=timestamp)
            if geometry is not None and self._geometry_matches_model(
                geometry, model_detection, frame_bgr.shape
            ):
                self.last_geometry_detection = geometry
                return replace(
                    geometry, confidence=model_detection.confidence)
        return self._from_model(model_detection, frame_bgr.shape, timestamp)

    def preview(self, frame_bgr, timestamp=None):
        """Detecção barata durante o giro: modelo, sem Hough/geometria.

        A prévia só serve para frear no primeiro frame em que a saída entra
        no quadro. O ângulo será medido em ``detect`` depois do chassi parar.
        """
        model_detection = self.model.detect(frame_bgr)
        self.last_model_detection = model_detection
        self.last_geometry_detection = None
        if model_detection is None:
            return None
        return self._from_model(model_detection, frame_bgr.shape, timestamp)

    @staticmethod
    def _geometry_matches_model(geometry, model_detection, frame_shape):
        height, width = frame_shape[:2]
        x, y, box_width, box_height = model_detection.bbox
        margin = float(cfg.EXIT_MODEL_GEOMETRY_MARGIN_RATIO)
        left = max(0.0, x - box_width * margin)
        right = min(float(width), x + box_width * (1.0 + margin))
        top = max(0.0, y - box_height * margin)
        bottom = min(float(height), y + box_height * (1.0 + margin))
        return (
            left <= float(geometry.center_x) <= right
            and top <= float(geometry.center_y) <= bottom
        )

    @staticmethod
    def _from_model(detection, frame_shape, timestamp):
        height, width = frame_shape[:2]
        x, y, box_width, box_height = detection.bbox
        narrow = max(min(box_width, box_height), 1.0)
        return ExitStripeDetection(
            center_x=detection.center_x,
            center_y=detection.center_y,
            width=max(int(round(box_width)), 1),
            height=max(int(round(box_height)), 1),
            top_y=max(int(round(y)), 0),
            bottom_y=min(int(round(y + box_height)), max(height - 1, 0)),
            span_ratio=float(box_width) / max(float(width), 1.0),
            thickness_ratio=float(box_height) / max(float(height), 1.0),
            aspect=max(box_width, box_height) / narrow,
            value=0.0,
            surround_contrast=0.0,
            confidence=detection.confidence,
            timestamp=0.0 if timestamp is None else float(timestamp),
            bbox=(x, y, box_width, box_height),
            angle_deg=0.0,
        )


@dataclass(frozen=True)
class AsyncExitResult:
    sequence: int
    source_sequence: object
    detection: object
    frame_shape: tuple
    captured_at: float
    completed_at: float
    processing_ms: float
    preview: bool
    dropped_frames: int


class LatestFrameExitDetector:
    """Executa o modelo de saida fora do controle e nunca acumula backlog."""

    def __init__(self, detector, clock=time.monotonic):
        self.detector = detector
        self._clock = clock
        self._condition = threading.Condition()
        self._pending = None
        self._result = None
        self._error = None
        self._stopping = False
        self._next_sequence = 0
        self._dropped_frames = 0
        self._thread = threading.Thread(
            target=self._run,
            name="shadow-saida-modelo",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        frame,
        captured_at,
        source_sequence=None,
        *,
        preview=False,
    ):
        if frame is None:
            raise ValueError("frame ausente para o modelo de saida")
        with self._condition:
            if self._stopping:
                return None
            self._next_sequence += 1
            sequence = self._next_sequence
            if self._pending is not None:
                self._dropped_frames += 1
            # A fonte de captura publica uma matriz propria. Manter somente a
            # referencia evita copiar 640x480 a cada frame da camera.
            self._pending = (
                sequence,
                frame,
                float(captured_at),
                source_sequence,
                bool(preview),
            )
            self._condition.notify()
            return sequence

    def poll(self, after_sequence=0):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    f"detector assincrono da saida falhou: {self._error}"
                ) from self._error
            if (
                self._result is not None
                and self._result.sequence > after_sequence
            ):
                return self._result
            return None

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def close(self, timeout=2.0):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._result = None
            self._condition.notify_all()
        self._thread.join(timeout=max(float(timeout), 0.0))
        return not self._thread.is_alive()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                (
                    sequence,
                    frame,
                    captured_at,
                    source_sequence,
                    preview,
                ) = self._pending
                self._pending = None
                dropped_frames = self._dropped_frames

            started = self._clock()
            try:
                detection = (
                    self.detector.preview(frame, timestamp=captured_at)
                    if preview
                    else self.detector.detect(frame, timestamp=captured_at)
                )
                completed = self._clock()
                result = AsyncExitResult(
                    sequence=sequence,
                    source_sequence=source_sequence,
                    detection=detection,
                    frame_shape=tuple(frame.shape),
                    captured_at=captured_at,
                    completed_at=completed,
                    processing_ms=(completed - started) * 1000.0,
                    preview=preview,
                    dropped_frames=dropped_frames,
                )
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return

            with self._condition:
                if self._stopping:
                    return
                self._result = result
                self._condition.notify_all()
