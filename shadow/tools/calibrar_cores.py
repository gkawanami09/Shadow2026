#!/usr/bin/env python3
"""Calibra as cores da CÂMERA DE LINHA usando barras deslizantes.

Grupos disponíveis (tecle o número para trocar):

    1  preto da linha, teto superior
    2  preto da linha, teto inferior
    3  preto da rampa
    4  verde dos marcadores
    5  vermelho, banda 1 (o vermelho cruza a origem do H)
    6  vermelho, banda 2
    7  PRATA da faixa de entrada da sala de resgate

O grupo 7 é um perfil INDEPENDENTE. Ele não compartilha limites com a esfera
prateada da vítima: aquela pertence à câmera de resgate e vive em
``config_resgate.py``. Nada aqui é gravado sobre os limites do resgate, e
nada do resgate é lido aqui.

No grupo 7 a janela mostra, além da máscara, o resultado do detector real:
a ROI, a faixa aceita e o motivo exato da rejeição quando ela não passa.
Isso permite calibrar olhando para a decisão, não só para os pixels.

Teclas: `s` salva o grupo atual, `q` sai.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from shared.gerenciadores import ConfigManager  # noqa: E402
from visao.captura import LineCamera  # noqa: E402
from visao.faixa_entrada import EntrySilverDetector  # noqa: E402

WINDOW = "Shadow2026 - calibracao de cores"

config_manager = ConfigManager(str(config.CONFIG_INI_PATH))


def read_ini(name, fallback):
    value = config_manager.read_variable('color_values_line', name)
    return list(value) if value is not None else list(fallback)


# Cada grupo: (titulo, [(nome_ini, valores, rotulos_trackbar, maximos)], modo)
GROUPS = {
    "1": ("black_max_normal_top", "bgr_ceiling"),
    "2": ("black_max_normal_bottom", "bgr_ceiling"),
    "3": ("black_max_ramp_down_top", "bgr_ceiling"),
    "4": ("green", "hsv_range"),
    "5": ("red_1", "hsv_range"),
    "6": ("red_2", "hsv_range"),
    # Perfil próprio da faixa de entrada; ver o docstring do módulo.
    "7": ("entry_silver", "hsv_range"),
}

HSV_KEYS = {"green": ("green_min", "green_max"),
            "red_1": ("red_min_1", "red_max_1"),
            "red_2": ("red_min_2", "red_max_2"),
            "entry_silver": ("entry_silver_min", "entry_silver_max")}

HSV_DEFAULTS = {"green_min": config.GREEN_MIN_DEFAULT, "green_max": config.GREEN_MAX_DEFAULT,
                "red_min_1": config.RED_MIN_1_DEFAULT, "red_max_1": config.RED_MAX_1_DEFAULT,
                "red_min_2": config.RED_MIN_2_DEFAULT, "red_max_2": config.RED_MAX_2_DEFAULT,
                "entry_silver_min": config.ENTRY_SILVER_MIN_DEFAULT,
                "entry_silver_max": config.ENTRY_SILVER_MAX_DEFAULT}

BGR_DEFAULTS = {"black_max_normal_top": config.BLACK_MAX_NORMAL_TOP_DEFAULT,
                "black_max_normal_bottom": config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT,
                "black_max_ramp_down_top": config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT}


def build_trackbars(group_name, mode):
    cv2.destroyAllWindows()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, config.camera_x * 2, config.camera_y * 2 + 80)

    if mode == "bgr_ceiling":
        values = read_ini(group_name, BGR_DEFAULTS[group_name])
        for label, val, vmax in zip(("B max", "G max", "R max"), values, (255, 255, 255)):
            cv2.createTrackbar(label, WINDOW, int(val), vmax, lambda _v: None)
    else:
        key_min, key_max = HSV_KEYS[group_name]
        vmin = read_ini(key_min, HSV_DEFAULTS[key_min])
        vmax = read_ini(key_max, HSV_DEFAULTS[key_max])
        for label, val, top in (("H min", vmin[0], 180), ("S min", vmin[1], 255), ("V min", vmin[2], 255),
                                ("H max", vmax[0], 180), ("S max", vmax[1], 255), ("V max", vmax[2], 255)):
            cv2.createTrackbar(label, WINDOW, int(val), top, lambda _v: None)


def get_bgr_values():
    return [cv2.getTrackbarPos("B max", WINDOW),
            cv2.getTrackbarPos("G max", WINDOW),
            cv2.getTrackbarPos("R max", WINDOW)]


def get_hsv_values():
    vmin = [cv2.getTrackbarPos("H min", WINDOW),
            cv2.getTrackbarPos("S min", WINDOW),
            cv2.getTrackbarPos("V min", WINDOW)]
    vmax = [cv2.getTrackbarPos("H max", WINDOW),
            cv2.getTrackbarPos("S max", WINDOW),
            cv2.getTrackbarPos("V max", WINDOW)]
    return vmin, vmax


def annotate_entry_silver(frame, detector, vmin, vmax):
    """Mostra ROI, faixa aceita e o motivo exato da rejeição.

    Calibrar a faixa prata olhando só para a máscara engana: a máscara pode
    parecer perfeita e o candidato ainda cair na geometria, no reflexo ou no
    contraste. Aqui a decisão do detector real fica visível.
    """
    detector.hsv_min = list(vmin)
    detector.hsv_max = list(vmax)
    height, width = frame.shape[:2]

    roi_top = int(round(height * config.ENTRY_SILVER_ROI_TOP))
    cv2.line(frame, (0, roi_top), (width, roi_top), (0, 200, 255), 1)
    cv2.putText(frame, "ROI", (4, roi_top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 200, 255), 1)

    detection = detector.detect(frame, line_ahead=False, timestamp=0.0)
    if detection is not None:
        x, y, box_width, box_height = detection.bbox
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                      (0, 255, 0), 2)
        cv2.putText(
            frame,
            (f"ACEITA conf={detection.confidence:.2f} "
             f"larg={detection.span_ratio:.2f} "
             f"esp={detection.thickness_ratio:.2f} "
             f"prop={detection.aspect:.1f}"),
            (4, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, .42,
            (0, 255, 0), 1)
        cv2.putText(
            frame,
            (f"S={detection.saturation:.0f} "
             f"dyn={detection.dynamic_range:.0f} "
             f"brilho={detection.highlight_fraction:.3f} "
             f"contraste={detection.surround_contrast:.0f}"),
            (4, min(y + box_height + 14, height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 0), 1)
    else:
        banda = detector.last_band
        if banda is not None:
            cv2.rectangle(frame, (banda.left_x, banda.top_y),
                          (banda.right_x, banda.bottom_y), (0, 140, 255), 1)
        cv2.putText(frame, f"REJEITADA: {detector.last_reason}",
                    (4, height - 24), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 80, 255), 2)
    return frame


def main():
    print("Abrindo câmera…")
    camera = LineCamera()
    entry_detector = EntrySilverDetector()

    group_key = "1"
    group_name, mode = GROUPS[group_key]
    build_trackbars(group_name, mode)
    print(__doc__)

    try:
        while True:
            frame = camera.get_frame()

            if mode == "bgr_ceiling":
                ceiling = get_bgr_values()
                mask = cv2.inRange(frame, np.array(config.BLACK_MIN_DEFAULT), np.array(ceiling))
            else:
                vmin, vmax = get_hsv_values()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array(vmin), np.array(vmax))
                if group_name == "entry_silver":
                    frame = annotate_entry_silver(
                        frame, entry_detector, vmin, vmax)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            stacked = np.vstack((frame, mask_bgr))
            legenda = (
                f"[{group_key}] {group_name}  (s=salvar, q=sair)"
                + ("  — CAMERA DE LINHA, perfil proprio da entrada"
                   if group_name == "entry_silver" else ""))
            cv2.putText(stacked, legenda,
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            cv2.imshow(WINDOW, stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if mode == "bgr_ceiling":
                    values = get_bgr_values()
                    config_manager.write_variable('color_values_line', group_name, values)
                    print(f"salvo: {group_name} = {values}")
                else:
                    key_min, key_max = HSV_KEYS[group_name]
                    vmin, vmax = get_hsv_values()
                    config_manager.write_variable('color_values_line', key_min, vmin)
                    config_manager.write_variable('color_values_line', key_max, vmax)
                    print(f"salvo: {key_min} = {vmin} | {key_max} = {vmax}")
            elif key != 255 and chr(key) in GROUPS:
                group_key = chr(key)
                group_name, mode = GROUPS[group_key]
                build_trackbars(group_name, mode)
                print(f"grupo: {group_name}")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
