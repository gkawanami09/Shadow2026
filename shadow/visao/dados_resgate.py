"""Gravação assíncrona de imagens usadas para calibrar o resgate.

O loop de controle faz somente uma copia do frame e publica em uma mailbox de
capacidade um. Codificacao PNG e escrita em disco acontecem exclusivamente no
worker, sem fila crescente e sem qualquer acesso ao Arduino ou ao detector.
"""

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import cv2
import numpy as np


SCHEMA_VERSION = 1
SHADOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = SHADOW_ROOT / "captures" / "dados_resgate"
_SAFE_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


@dataclass(frozen=True)
class DatasetSubmitResult:
    """Resposta imediata de :meth:`RescueDatasetWriter.submit`."""

    status: str
    capture_id: object = None

    @property
    def accepted(self):
        return self.status == "accepted"


@dataclass(frozen=True)
class _Snapshot:
    capture_id: str
    frame: object
    metadata: object
    submitted_utc: str


def _inside_shadow(path):
    shadow_root = SHADOW_ROOT.resolve()
    resolved = Path(path).resolve(strict=False)
    try:
        resolved.relative_to(shadow_root)
    except ValueError:
        return False
    return True


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(
        f"tipo nao serializavel no metadata: {type(value).__name__}")


def _metadata_snapshot(metadata):
    if metadata is None:
        metadata = {}
    if not hasattr(metadata, "items"):
        raise ValueError("metadata precisa ser um mapeamento JSON")
    try:
        serialized = json.dumps(
            dict(metadata),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        return json.loads(serialized)
    except (TypeError, ValueError) as err:
        raise ValueError(f"metadata invalido: {err}") from err


def _color_space(frame):
    if frame.ndim == 2:
        return "GRAY8"
    channels = frame.shape[2]
    if channels == 1:
        return "GRAY8"
    if channels == 3:
        return "BGR8_opencv"
    if channels == 4:
        return "BGRA8_opencv"
    return f"UNKNOWN_{channels}_CHANNELS"


class RescueDatasetWriter:
    """Worker de PNG+JSON com uma unica posicao pendente.

    ``submit`` nunca codifica nem escreve em disco. Se a posicao pendente ja
    estiver ocupada, retorna ``mailbox_full`` imediatamente. Pode existir uma
    escrita ativa e, no maximo, uma proxima amostra pendente.
    """

    def __init__(
        self,
        output_dir=None,
        session_id=None,
        png_compression=2,
    ):
        base_dir = (
            DEFAULT_DATASET_ROOT
            if output_dir is None
            else Path(output_dir)
        )
        if not base_dir.is_absolute():
            base_dir = SHADOW_ROOT / base_dir
        if not _inside_shadow(base_dir):
            raise ValueError(
                "o dataset precisa permanecer dentro da pasta shadow")

        if session_id is None:
            utc_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            session_id = (
                f"session_{utc_id}_p{os.getpid()}_{uuid.uuid4().hex[:6]}")
        session_id = str(session_id)
        if not _SAFE_SESSION.fullmatch(session_id):
            raise ValueError(
                "session_id deve usar apenas letras, numeros, ponto, _ ou -")

        base_dir.mkdir(parents=True, exist_ok=True)
        session_dir = base_dir / session_id
        session_dir.mkdir(parents=False, exist_ok=False)
        session_dir = session_dir.resolve()
        if not _inside_shadow(session_dir):
            raise ValueError(
                "a sessao do dataset resolveu para fora da pasta shadow")

        self.session_dir = session_dir
        self.png_compression = int(np.clip(png_compression, 0, 9))
        self._condition = threading.Condition()
        self._pending = None
        self._closing = False
        self._active = False
        self._next_id = 0
        self._completed_count = 0
        self._failed_count = 0
        self._last_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="shadow-rescue-dataset",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame, metadata=None):
        """Publica uma copia bruta e retorna status/identificador imediatamente."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame precisa ser um ndarray nao vazio")
        if frame.dtype != np.uint8:
            raise ValueError("frame do dataset precisa usar dtype uint8")
        if frame.ndim not in (2, 3):
            raise ValueError("frame precisa ter duas ou tres dimensoes")
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
            raise ValueError("frame precisa ter 1, 3 ou 4 canais")

        with self._condition:
            if self._closing:
                return DatasetSubmitResult("closed")
            if self._pending is not None:
                return DatasetSubmitResult("mailbox_full")

            # A copia e a normalizacao pequena de metadata sao o unico trabalho
            # sincrono. O lock impede dois submitters de ocuparem o mesmo slot.
            frame_snapshot = np.array(frame, copy=True, order="C")
            metadata_copy = _metadata_snapshot(metadata)
            self._next_id += 1
            submitted_utc = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z")
            capture_id = (
                f"frame_{self._next_id:06d}_"
                f"{time.time_ns()}")
            self._pending = _Snapshot(
                capture_id,
                frame_snapshot,
                metadata_copy,
                submitted_utc,
            )
            self._condition.notify()
            return DatasetSubmitResult("accepted", capture_id)

    def close(self, timeout=2.0):
        """Fecha para novos submits e tenta drenar a unica amostra pendente."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=max(float(timeout), 0.0))
        return not self._thread.is_alive()

    @property
    def completed_count(self):
        with self._condition:
            return self._completed_count

    @property
    def failed_count(self):
        with self._condition:
            return self._failed_count

    @property
    def last_error(self):
        with self._condition:
            return self._last_error

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._pending is None and self._closing:
                    return
                snapshot = self._pending
                self._pending = None
                self._active = True

            error = None
            try:
                self._write_snapshot(snapshot)
            except Exception as err:
                error = err

            with self._condition:
                self._active = False
                if error is None:
                    self._completed_count += 1
                    self._last_error = None
                else:
                    self._failed_count += 1
                    self._last_error = (
                        f"{snapshot.capture_id}: {error}")
                self._condition.notify_all()

    def _write_snapshot(self, snapshot):
        png_path = self.session_dir / f"{snapshot.capture_id}.png"
        json_path = self.session_dir / f"{snapshot.capture_id}.json"
        temp_tag = uuid.uuid4().hex
        png_temp = self.session_dir / (
            f".{snapshot.capture_id}.{temp_tag}.png.tmp")
        json_temp = self.session_dir / (
            f".{snapshot.capture_id}.{temp_tag}.json.tmp")

        encoded_ok, encoded = cv2.imencode(
            ".png",
            snapshot.frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
        )
        if not encoded_ok:
            raise RuntimeError("OpenCV nao conseguiu codificar o PNG")

        height, width = snapshot.frame.shape[:2]
        channels = (
            1 if snapshot.frame.ndim == 2 else snapshot.frame.shape[2])
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": snapshot.capture_id,
            "submitted_utc": snapshot.submitted_utc,
            "image": {
                "filename": png_path.name,
                "format": "png",
                "lossless": True,
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "dtype": str(snapshot.frame.dtype),
                "color_space": _color_space(snapshot.frame),
            },
            "metadata": snapshot.metadata,
        }
        json_bytes = (
            json.dumps(
                sidecar,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        png_committed = False
        try:
            png_temp.write_bytes(encoded.tobytes())
            json_temp.write_bytes(json_bytes)
            os.replace(png_temp, png_path)
            png_committed = True
            os.replace(json_temp, json_path)
        except Exception:
            self._unlink_if_present(png_temp)
            self._unlink_if_present(json_temp)
            if png_committed:
                self._unlink_if_present(png_path)
            raise

    @staticmethod
    def _unlink_if_present(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
