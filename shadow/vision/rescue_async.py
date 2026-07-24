"""Worker de visao com politica latest-frame para manter o preview fluido."""

from dataclasses import dataclass, replace
import threading
import time

import cv2

from vision.rescue_ball import BallDetection


def _fit_detector_size(frame_shape, max_width, max_height):
    height, width = frame_shape[:2]
    scale = min(
        float(max_width) / max(width, 1),
        float(max_height) / max(height, 1),
        1.0,
    )
    output_width = max(2, int(round(width * scale)))
    output_height = max(2, int(round(height * scale)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    return output_width, output_height


def _scale_detection(detection, from_shape, to_shape):
    if detection is None:
        return None
    from_height, from_width = from_shape[:2]
    to_height, to_width = to_shape[:2]
    scale_x = float(to_width) / max(from_width, 1)
    scale_y = float(to_height) / max(from_height, 1)
    radius_scale = min(scale_x, scale_y)
    return BallDetection(
        detection.kind,
        detection.center_x * scale_x,
        detection.center_y * scale_y,
        detection.radius * radius_scale,
        detection.confidence,
        detection.confirmed,
        detection.hits,
        detection.timestamp,
        track_locked=detection.track_locked,
    )


@dataclass(frozen=True)
class AsyncDetectionResult:
    sequence: int
    source_sequence: object
    detection: object
    frame_shape: tuple
    detector_shape: tuple
    captured_at: float
    completed_at: float
    processing_s: float
    dropped_frames: int
    generation: int
    hough_used: bool
    contour_proposals: int
    hough_proposals: int
    candidate_count: int
    candidate_radii: tuple
    diagnostic: str
    candidate_circles: tuple = ()
    crescent_evidence: object = None
    locked_detection: object = None


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    frame: object
    captured_at: float


class FreshDetectionGate:
    """Exige resultados frescos e distintos, mesmo se o tracker veio de stale."""

    def __init__(self, required_hits, max_misses=0):
        self.required_hits = max(int(required_hits), 1)
        self.max_misses = max(int(max_misses), 0)
        self._fresh_hits = 0
        self._last_tracker_hits = None
        self._misses = 0

    def reset(self):
        self._fresh_hits = 0
        self._last_tracker_hits = None
        self._misses = 0

    def accept(self, detection):
        if detection is None:
            self._misses += 1
            if self._misses > self.max_misses:
                self.reset()
            return None
        self._misses = 0

        tracker_hits = int(detection.hits)
        if (
            self._fresh_hits == 0
            or tracker_hits <= 1
            or (
                self._last_tracker_hits is not None
                and tracker_hits <= self._last_tracker_hits
            )
        ):
            self._fresh_hits = 1
        else:
            self._fresh_hits += 1
        self._last_tracker_hits = tracker_hits

        confirmed = (
            detection.confirmed
            and self._fresh_hits >= self.required_hits
        )
        return replace(
            detection,
            confirmed=confirmed,
            hits=self._fresh_hits,
        )


class LatestFrameSource:
    """Captura em thread propria para o controle nunca bloquear na camera."""

    def __init__(self, source, clock=time.monotonic):
        self.source = source
        self._clock = clock
        self._condition = threading.Condition()
        self._latest = None
        self._error = None
        self._ended = False
        self._stopping = False
        self._sequence = 0
        self._close_thread = None
        self._close_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="shadow-rescue-capture",
            daemon=True,
        )
        self._thread.start()

    def poll(self, after_sequence=0):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    f"captura assincrona falhou: {self._error}") \
                    from self._error
            if (
                self._latest is not None
                and self._latest.sequence > after_sequence
            ):
                return self._latest
            return None

    @property
    def ended(self):
        with self._condition:
            return self._ended

    def close(self, timeout=2.0):
        timeout = max(float(timeout), 0.0)
        deadline = time.monotonic() + timeout
        with self._condition:
            self._stopping = True
            self._ended = True
            self._latest = None
            self._condition.notify_all()
            if self._close_thread is None:
                # stop()/release() costuma desbloquear capture_array/read. Ele
                # tambem fica fora do caller para um driver defeituoso nao
                # congelar indefinidamente o encerramento de seguranca.
                self._close_thread = threading.Thread(
                    target=self._close_source,
                    name="shadow-rescue-source-close",
                    daemon=True,
                )
                self._close_thread.start()

        self._close_thread.join(timeout=max(deadline - time.monotonic(), 0.0))
        self._thread.join(timeout=max(deadline - time.monotonic(), 0.0))
        if not self._close_thread.is_alive() and self._close_error is not None:
            raise RuntimeError(
                f"falha ao fechar fonte de imagem: {self._close_error}"
            ) from self._close_error
        return (
            not self._thread.is_alive()
            and not self._close_thread.is_alive()
        )

    def _close_source(self):
        try:
            self.source.close()
        except Exception as err:
            self._close_error = err

    def _run(self):
        while True:
            with self._condition:
                if self._stopping:
                    return
            try:
                frame = self.source.get_frame()
            except Exception as err:
                with self._condition:
                    if not self._stopping:
                        self._error = err
                    self._ended = True
                    self._condition.notify_all()
                return
            captured_at = self._clock()
            with self._condition:
                if self._stopping:
                    return
                if frame is None:
                    self._ended = True
                    self._condition.notify_all()
                    return
                self._sequence += 1
                self._latest = CapturedFrame(
                    self._sequence, frame, captured_at)
                self._condition.notify_all()


class LatestFrameBallDetector:
    """Processa no maximo um frame pendente e descarta backlog antigo."""

    def __init__(
        self,
        detector,
        max_width,
        max_height,
        clock=time.monotonic,
    ):
        self.detector = detector
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self._clock = clock
        self._condition = threading.Condition()
        self._pending = None
        self._result = None
        self._error = None
        self._stopping = False
        self._next_sequence = 0
        self._dropped_frames = 0
        self._generation = 0
        self._reset_requested = False
        self._thread = threading.Thread(
            target=self._run,
            name="shadow-rescue-detector",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame, captured_at=None, source_sequence=None):
        if frame is None:
            raise ValueError("frame ausente")
        captured_at = (
            self._clock() if captured_at is None else float(captured_at))
        with self._condition:
            if self._stopping:
                return None
            self._next_sequence += 1
            sequence = self._next_sequence
            if self._pending is not None:
                self._dropped_frames += 1
            # capture_array/read entregam uma matriz propria. O worker mantem
            # esta referencia ate terminar; nao existe fila crescente.
            self._pending = (
                sequence,
                frame,
                captured_at,
                self._generation,
                source_sequence,
            )
            self._condition.notify()
            return sequence

    def reset_tracking(self):
        """Invalida trabalho antigo e exige nova confirmacao temporal."""
        with self._condition:
            self._generation += 1
            self._reset_requested = True
            self._pending = None
            self._result = None
            self._condition.notify_all()

    def poll(self, after_sequence=0):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    f"detector assincrono falhou: {self._error}") \
                    from self._error
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
                    generation,
                    source_sequence,
                ) = self._pending
                self._pending = None
                dropped_frames = self._dropped_frames
                should_reset = (
                    self._reset_requested
                    and generation == self._generation
                )
                if should_reset:
                    self._reset_requested = False

            started_at = self._clock()
            try:
                if should_reset:
                    self.detector.reset()
                detector_size = _fit_detector_size(
                    frame.shape, self.max_width, self.max_height)
                if detector_size == (frame.shape[1], frame.shape[0]):
                    detector_frame = frame
                else:
                    detector_frame = cv2.resize(
                        frame, detector_size, interpolation=cv2.INTER_AREA)
                detection = self.detector.detect(
                    detector_frame, timestamp=captured_at)
                hough_used = bool(
                    getattr(self.detector, "last_hough_used", False))
                candidates = tuple(
                    getattr(self.detector, "last_candidates", ()))
                candidate_count = len(candidates)
                contour_proposals = int(
                    getattr(
                        self.detector,
                        "last_contour_proposals",
                        0,
                    ))
                hough_proposals = int(
                    getattr(
                        self.detector,
                        "last_hough_proposals",
                        0,
                    ))
                diagnostic = str(
                    getattr(self.detector, "last_diagnostic", ""))
                crescent_evidence = getattr(
                    self.detector,
                    "last_crescent_evidence",
                    None,
                )
                locked_detection = getattr(
                    self.detector,
                    "last_locked_detection",
                    None,
                )
                detection = _scale_detection(
                    detection, detector_frame.shape, frame.shape)
                locked_detection = _scale_detection(
                    locked_detection,
                    detector_frame.shape,
                    frame.shape,
                )
                detector_height, detector_width = detector_frame.shape[:2]
                frame_height, frame_width = frame.shape[:2]
                radius_scale = min(
                    float(frame_width) / max(detector_width, 1),
                    float(frame_height) / max(detector_height, 1),
                )
                scale_x = float(frame_width) / max(detector_width, 1)
                scale_y = float(frame_height) / max(detector_height, 1)
                candidate_circles = tuple(sorted(
                    (
                        (
                            round(float(candidate.center_x) * scale_x, 1),
                            round(float(candidate.center_y) * scale_y, 1),
                            round(float(candidate.radius) * radius_scale, 1),
                            str(candidate.kind),
                            round(float(candidate.confidence), 3),
                        )
                        for candidate in candidates
                    ),
                    key=lambda circle: circle[2],
                    reverse=True,
                )[:8])
                candidate_radii = tuple(
                    circle[2] for circle in candidate_circles[:4])
                completed_at = self._clock()
                result = AsyncDetectionResult(
                    sequence=sequence,
                    source_sequence=source_sequence,
                    detection=detection,
                    frame_shape=tuple(frame.shape),
                    detector_shape=tuple(detector_frame.shape),
                    captured_at=captured_at,
                    completed_at=completed_at,
                    processing_s=completed_at - started_at,
                    dropped_frames=dropped_frames,
                    generation=generation,
                    hough_used=hough_used,
                    contour_proposals=contour_proposals,
                    hough_proposals=hough_proposals,
                    candidate_count=candidate_count,
                    candidate_radii=candidate_radii,
                    diagnostic=diagnostic,
                    candidate_circles=candidate_circles,
                    crescent_evidence=crescent_evidence,
                    locked_detection=locked_detection,
                )
            except Exception as err:
                with self._condition:
                    self._error = err
                    self._condition.notify_all()
                return

            with self._condition:
                if self._stopping:
                    return
                if generation != self._generation:
                    # O controle declarou a imagem stale enquanto este frame
                    # era processado. Nunca publicar nem confirmar esse epoch.
                    continue
                self._result = result
                self._condition.notify_all()
