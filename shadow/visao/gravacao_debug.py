"""Gravacao opcional e sincronizada para reproduzir falhas de percurso."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


class GravadorVisao:
    """Salva PNG lossless e JSONL ligados pela mesma sequencia.

    MJPG nao serve para replay de segmentacao: a compressao altera justamente
    os pixels proximos aos limiares de preto/verde. Cada frame, portanto, vira
    um PNG independente. O identificador aleatorio impede que duas sessoes
    iniciadas no mesmo segundo apaguem o diagnostico anterior.
    """

    def __init__(self, diretorio, *, largura, altura, fps, manifest=None):
        raiz = Path(diretorio).expanduser().resolve()
        raiz.mkdir(parents=True, exist_ok=True)
        self.largura = int(largura)
        self.altura = int(altura)
        self.fps = float(fps)
        if self.largura <= 0 or self.altura <= 0 or self.fps <= 0.:
            raise ValueError("geometria/FPS da gravacao precisa ser positiva")
        prefixo = (
            datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            + "-" + uuid4().hex[:8]
        )
        self.session_id = prefixo
        self.frames_path = raiz / f"visao-crua-{prefixo}"
        self.frames_path.mkdir(parents=False, exist_ok=False)
        self.log_path = raiz / f"visao-{prefixo}.jsonl"
        self.manifest_path = raiz / f"visao-{prefixo}.manifest.json"
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._frame_index = 0
        manifesto = {
            "session_id": self.session_id,
            "width": self.largura,
            "height": self.altura,
            "fps": self.fps,
        }
        if manifest is not None:
            if not isinstance(manifest, dict):
                raise ValueError("manifest precisa ser dict")
            manifesto.update(manifest)
        self.manifest_path.write_text(
            json.dumps(
                self._serializavel(manifesto),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            "[visao] diagnostico lossless ativo: "
            f"{self.frames_path}; {self.log_path}"
        )

    @staticmethod
    def _serializavel(valor):
        if is_dataclass(valor):
            return asdict(valor)
        if isinstance(valor, np.generic):
            valor = valor.item()
        if isinstance(valor, float) and not math.isfinite(valor):
            return None
        if hasattr(valor, "value") and not isinstance(valor, (str, bytes)):
            return valor.value
        if isinstance(valor, tuple):
            return [GravadorVisao._serializavel(item) for item in valor]
        if isinstance(valor, dict):
            return {
                str(chave): GravadorVisao._serializavel(item)
                for chave, item in valor.items()
            }
        return valor

    def gravar(self, frame_cru, registro):
        if self._log is None:
            raise ValueError("gravador ja foi encerrado")
        frame = np.asarray(frame_cru)
        if (
            frame.dtype != np.uint8
            or frame.shape != (self.altura, self.largura, 3)
        ):
            raise ValueError(
                "frame cru deve ser uint8 BGR com shape "
                f"({self.altura}, {self.largura}, 3)"
            )
        if not isinstance(registro, dict) or "sequence" not in registro:
            raise ValueError("registro precisa conter sequence")
        sequence = int(registro["sequence"])
        if sequence < 0:
            raise ValueError("sequence nao pode ser negativa")
        self._frame_index += 1
        nome_frame = (
            f"frame-{self._frame_index:09d}-seq-{sequence:012d}.png"
        )
        caminho_frame = self.frames_path / nome_frame
        if not cv2.imwrite(
            str(caminho_frame),
            frame,
            (cv2.IMWRITE_PNG_COMPRESSION, 1),
        ):
            raise OSError(f"nao foi possivel salvar frame cru {caminho_frame}")

        registro_completo = dict(registro)
        registro_completo["frame_index"] = self._frame_index
        registro_completo["raw_frame"] = str(
            caminho_frame.relative_to(self.log_path.parent)
        )
        linha = self._serializavel(registro_completo)
        self._log.write(json.dumps(
            linha, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ) + "\n")

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None
