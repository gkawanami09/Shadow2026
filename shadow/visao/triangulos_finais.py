"""Mapeamento final dos DOIS triângulos, depois das três vítimas resgatadas.

Durante o transporte, apenas o triângulo da cor correta pode comandar o robô.
Esta fase é diferente: ela roda os dois detectores ao mesmo tempo, só para
registrar onde a sala colocou cada triângulo. Nenhuma detecção daqui comanda
motores — é diagnóstico e prova de que o robô entendeu a arena.

O overlay é a parte que mais engana em campo: um verde desenhado em vermelho
faz a equipe recalibrar a cor errada. Por isso as cores vêm de
``config_resgate.FINAL_TRIANGLE_OVERLAY_BGR`` e existe um teste dedicado que
lê os pixels desenhados e confirma que verde saiu verde e vermelho saiu
vermelho.
"""

import cv2
import numpy as np

import config_resgate as cfg
from visao.marcador_resgate import MarkerDetector, color_masks


class FinalTriangleMapper:
    """Roda os detectores verde e vermelho sobre o mesmo frame."""

    def __init__(self):
        # Duas instâncias independentes: cada uma mantém seu próprio track e
        # seu próprio lock, sem disputar identidade com a outra.
        self.detectors = {
            "green": MarkerDetector("green"),
            "red": MarkerDetector("red"),
        }
        self.detections = {"green": None, "red": None}
        self.confirmed = {"green": False, "red": False}
        self.frames = 0

    @property
    def both_found(self):
        return self.confirmed["green"] and self.confirmed["red"]

    def reset(self):
        for detector in self.detectors.values():
            detector.reset()
        self.detections = {"green": None, "red": None}
        self.confirmed = {"green": False, "red": False}
        self.frames = 0

    def update(self, frame_bgr, timestamp=None):
        """Detecta os dois triângulos reaproveitando uma única conversão HSV."""
        masks = color_masks(frame_bgr)
        for kind, detector in self.detectors.items():
            detection = detector.detect(
                frame_bgr, timestamp=timestamp, masks=masks)
            self.detections[kind] = detection
            if detection is not None and detection.confirmed:
                # Uma vez confirmado, o mapeamento permanece: o triângulo não
                # se move, e girar o robô tira-o do campo de visão.
                self.confirmed[kind] = True
        self.frames += 1
        return dict(self.detections)


def annotate_final_triangles(frame_bgr, detections, copy=True):
    """Desenha os dois triângulos com rótulo, confiança e confirmações."""
    canvas = frame_bgr.copy() if copy else frame_bgr
    for kind in ("green", "red"):
        detection = detections.get(kind)
        if detection is None:
            continue
        color = cfg.FINAL_TRIANGLE_OVERLAY_BGR[kind]
        x, y, width, height = detection.bbox
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + width)), int(round(y + height))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            (
                f"{kind} {detection.confidence:.2f} "
                f"hits={detection.hits}"
                f"{' LOCK' if detection.track_locked else ''}"
            ),
            (max(x0, 4), max(y0 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def overlay_color_report():
    """Resumo textual das cores do overlay, usado no log e nos testes."""
    return {
        kind: tuple(int(channel) for channel in color)
        for kind, color in cfg.FINAL_TRIANGLE_OVERLAY_BGR.items()
    }


def dominant_channel(color_bgr):
    """Índice do canal dominante de uma cor BGR (0=B, 1=G, 2=R)."""
    return int(np.argmax(np.asarray(color_bgr, dtype=np.int32)))
