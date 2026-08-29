#!/usr/bin/env python3
"""Calibrador visual da faixa prata da entrada.

Use com a missao parada (a camera de linha so pode ser aberta por um processo):
    cd shadow
    python3 tools/calibrar_prata.py

Teclas: ``s`` salva em ``config.ini``; ``q`` sai. Reinicie a missao apos salvar.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from shared.gerenciadores import ConfigManager  # noqa: E402
from visao.captura import LineCamera  # noqa: E402
from visao.faixa_prata_entrada import (  # noqa: E402
    SILVER_CONFIG_KEYS, SILVER_CONFIG_SECTION, carregar_parametros_prata,
    detectar_faixa_prata,
)

WINDOW = "Shadow2026 - calibracao prata"
MANAGER = ConfigManager(str(config.CONFIG_INI_PATH))


def _black_mask(frame):
    """Replica a mascara normal de preto, so para validar fim da linha."""
    top = MANAGER.read_variable("color_values_line", "black_max_normal_top")
    bottom = MANAGER.read_variable("color_values_line", "black_max_normal_bottom")
    top = np.array(top or config.BLACK_MAX_NORMAL_TOP_DEFAULT, dtype=np.uint8)
    bottom = np.array(bottom or config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT, dtype=np.uint8)
    split = frame.shape[0] // 2
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[:split] = cv2.inRange(frame[:split], np.zeros(3, np.uint8), top)
    mask[split:] = cv2.inRange(frame[split:], np.zeros(3, np.uint8), bottom)
    return mask


def _trackbar(name, value, maximum):
    cv2.createTrackbar(name, WINDOW, int(value), maximum, lambda _value: None)


def _params_from_bars(base):
    values = vars(base).copy()
    values.update({
        "ENTRY_SILVER_MAX_SATURATION": float(cv2.getTrackbarPos("S max", WINDOW)),
        "ENTRY_SILVER_MIN_STD_VALUE": float(cv2.getTrackbarPos("Std V min", WINDOW)),
        "ENTRY_SILVER_BRIGHT_VALUE": cv2.getTrackbarPos("V claro min", WINDOW),
        "ENTRY_SILVER_DARK_VALUE": cv2.getTrackbarPos("V escuro max", WINDOW),
        "ENTRY_SILVER_MIN_DARK_ROW_RATIO": cv2.getTrackbarPos("Escuro faixa %", WINDOW) / 100.,
        "ENTRY_SILVER_MIN_WIDE_RATIO": cv2.getTrackbarPos("Largura min %", WINDOW) / 100.,
        "ENTRY_SILVER_SCORE_MIN": cv2.getTrackbarPos("Score min", WINDOW),
        "ENTRY_SILVER_CONFIRM_FRAMES": max(1, cv2.getTrackbarPos("Frames", WINDOW)),
    })
    return SimpleNamespace(**values)


def _save(params):
    for attr, (ini_name, _cast) in SILVER_CONFIG_KEYS.items():
        MANAGER.write_variable(SILVER_CONFIG_SECTION, ini_name, getattr(params, attr))
    print("[PRATA] calibracao salva em config.ini; reinicie a missao para aplicar.")


def main():
    base = carregar_parametros_prata()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, config.camera_x * 2, config.camera_y * 2 + 180)
    _trackbar("S max", base.ENTRY_SILVER_MAX_SATURATION, 255)
    _trackbar("Std V min", base.ENTRY_SILVER_MIN_STD_VALUE, 128)
    _trackbar("V claro min", base.ENTRY_SILVER_BRIGHT_VALUE, 255)
    _trackbar("V escuro max", base.ENTRY_SILVER_DARK_VALUE, 255)
    _trackbar("Escuro faixa %", round(base.ENTRY_SILVER_MIN_DARK_ROW_RATIO * 100), 100)
    _trackbar("Largura min %", round(base.ENTRY_SILVER_MIN_WIDE_RATIO * 100), 100)
    _trackbar("Score min", base.ENTRY_SILVER_SCORE_MIN, 10)
    _trackbar("Frames", base.ENTRY_SILVER_CONFIRM_FRAMES, 12)
    print(__doc__)
    camera = LineCamera()
    try:
        while True:
            frame = camera.get_frame()
            params = _params_from_bars(base)
            measurement = detectar_faixa_prata(
                frame, _black_mask(frame), line_aligned=True, params=params)
            view = frame.copy()
            if measurement.bbox:
                x, y, w, h = measurement.bbox
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 255), 2)
            text = (
                f"score={measurement.score}/{params.ENTRY_SILVER_SCORE_MIN} "
                f"sat={measurement.saturacao_media:.0f} "
                f"stdV={measurement.desvio_brilho:.0f} "
                f"claro={measurement.pct_claro:.0%} escuro={measurement.pct_escuro:.0%} "
                f"largura={measurement.largura_ratio:.0%} fimLinha={int(measurement.linha_fim)}"
            )
            candidata = bool(
                measurement.bbox is not None and measurement.linha_fim
                and measurement.score >= params.ENTRY_SILVER_SCORE_MIN)
            cv2.putText(view, "PRATA: " + ("CANDIDATA" if candidata else "nao"),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 255, 255), 2)
            cv2.putText(view, text, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .38,
                        (0, 255, 255), 1)
            mask_view = cv2.cvtColor(_black_mask(frame), cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_view, "mascara preta: fimLinha precisa ser 1", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            stacked = np.vstack((view, mask_view))
            cv2.imshow(WINDOW, stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                _save(params)
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
