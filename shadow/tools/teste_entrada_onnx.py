#!/usr/bin/env python3
"""Mostra o resultado bruto do modelo de entrada na câmera de linha.

Não abre serial nem movimenta motores. Use antes do teste da missão para
verificar o modelo com a iluminação e a montagem atuais do robô.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from visao.captura import LineCamera  # noqa: E402
from visao.entrada_missao import EntryModel  # noqa: E402


WINDOW = "Shadow2026 - modelo de entrada (q ou Esc fecha)"


def main():
    model = EntryModel().load()
    print(f"[entrada] usando {model.active_backend}: {model.active_path}")
    camera = LineCamera()
    try:
        while True:
            frame = camera.get_frame()
            started = time.perf_counter()
            detection = model.detect(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000
            preview = frame.copy()
            text = f"{model.active_backend.upper()}: sem faixa ({elapsed_ms:.0f} ms)"
            color = (0, 0, 255)
            if detection is not None:
                x, y, width, height = detection.bbox
                cv2.rectangle(
                    preview, (round(x), round(y)),
                    (round(x + width), round(y + height)), (0, 255, 255), 2)
                text = (f"{model.active_backend.upper()} PRATA "
                        f"conf={detection.confidence:.2f} ({elapsed_ms:.0f} ms)")
                color = (0, 255, 0)
            cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, color, 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
