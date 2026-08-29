"""Vigia YOLO da câmera frontal enquanto a missão segue a linha.

Não controla motores: somente pede ao controle, que possui a serial, que
execute a parada segura e entregue o robô ao resgate.
"""

import time

import config_resgate as cfg
from visao.captura_resgate import RescueCamera
from visao.vitima_yolo import VictimDetector, VictimModel


def vigiar_vitimas(camera_index, terminate, confirmed, status=None):
    """Publica uma vítima rastreada, plausível e de alta confiança."""
    model = VictimModel().carregar()
    detector = VictimDetector(model=model, target_kind="any")
    camera = None
    try:
        camera = RescueCamera(camera_index)
        while not terminate.value:
            detection = detector.detect(
                camera.get_frame(), timestamp=time.monotonic())
            if (
                detection is not None
                and detection.confirmed
                and detection.confidence >= cfg.MISSION_YOLO_RESCUE_MIN_CONFIDENCE
            ):
                confirmed.value = True
                if status is not None:
                    status.value = "Vitima YOLO confirmada - parando para resgate"
                print("[vigia-yolo] vítima %s confirmada: %.2f" % (
                    detection.kind, detection.confidence))
                while not terminate.value:
                    time.sleep(0.02)
                return
    finally:
        if camera is not None:
            camera.close()
