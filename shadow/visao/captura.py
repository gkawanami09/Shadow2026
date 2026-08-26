"""Abre a câmera de linha e entrega imagens no tamanho usado pelo detector."""

import math
import time

import cv2

from config import (CAPTURE_FPS, CAPTURE_FPS_FALLBACK, CAPTURE_HEIGHT,
                    CAPTURE_WIDTH, LENS_POSITION, LINE_CAMERA_INDEX,
                    camera_x, camera_y)


def escolher_fps_captura(modos_sensor):
    """Escolhe um FPS que o sensor realmente anunciou para pelo menos VGA."""
    fps_compativeis = []
    for modo in modos_sensor or ():
        try:
            largura, altura = modo["size"]
            fps = float(modo["fps"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            largura >= CAPTURE_WIDTH
            and altura >= CAPTURE_HEIGHT
            and math.isfinite(fps)
            and fps > 0
        ):
            fps_compativeis.append(fps)

    # Em Picamera2 antigo (ou num mock) a lista pode não existir. Nesse caso
    # preservamos exatamente a configuração que já funcionava no robô.
    if not fps_compativeis:
        return float(CAPTURE_FPS_FALLBACK)

    maior_fps = max(fps_compativeis)
    if maior_fps < CAPTURE_FPS_FALLBACK:
        return maior_fps
    return min(float(CAPTURE_FPS), maior_fps)


def obter_recorte_maximo(camera_controls):
    """Retorna o maior ScalerCrop anunciado pelo driver, se disponível."""
    try:
        recorte_bruto = camera_controls["ScalerCrop"][1]
        recorte = tuple(recorte_bruto)
    except (AttributeError, KeyError, TypeError, IndexError):
        try:
            recorte = (
                recorte_bruto.x,
                recorte_bruto.y,
                recorte_bruto.width,
                recorte_bruto.height,
            )
        except (AttributeError, UnboundLocalError):
            return None

    if (
        len(recorte) != 4
        or not all(isinstance(valor, int) for valor in recorte)
        or recorte[2] <= 0
        or recorte[3] <= 0
    ):
        return None
    return recorte


class LineCamera:
    def __init__(self):
        from picamera2 import Picamera2  # import local: so existe no Pi

        camera_info = Picamera2.global_camera_info()
        if not 0 <= LINE_CAMERA_INDEX < len(camera_info):
            raise RuntimeError(
                "camera de segue-linha no indice "
                f"{LINE_CAMERA_INDEX} indisponivel; detectadas: "
                f"{camera_info}"
            )
        print(
            "[camera] abrindo camera de segue-linha explicita "
            f"{LINE_CAMERA_INDEX} (flat 2)"
        )
        self.picam2 = Picamera2(camera_num=LINE_CAMERA_INDEX)

        try:
            modos_sensor = self.picam2.sensor_modes
        except (AttributeError, RuntimeError, TypeError):
            modos_sensor = ()
        fps_escolhido = escolher_fps_captura(modos_sensor)
        try:
            self._configurar_e_iniciar(fps_escolhido)
        except Exception as erro_fps:
            if fps_escolhido <= CAPTURE_FPS_FALLBACK:
                raise
            # Alguns drivers aceitam criar a configuração rápida, mas só
            # recusam em configure/start. Reabrir a câmera limpa esse estado
            # parcial antes de voltar para os 40 FPS já usados no robô.
            print(
                "[camera] modo rápido recusado pelo driver "
                f"({erro_fps}); voltando para "
                f"{CAPTURE_FPS_FALLBACK} FPS"
            )
            self.close()
            time.sleep(.05)
            self.picam2 = Picamera2(camera_num=LINE_CAMERA_INDEX)
            self._configurar_e_iniciar(float(CAPTURE_FPS_FALLBACK))

        self._abrir_campo_de_visao()
        self._configurar_foco()

        time.sleep(0.1)

    def _abrir_campo_de_visao(self):
        """Pede ao libcamera o sensor inteiro, sem zoom/crop digital."""
        recorte = obter_recorte_maximo(
            getattr(self.picam2, "camera_controls", None)
        )
        if recorte is None or not hasattr(self.picam2, "set_controls"):
            # A proporção 16:9 da configuração já evita recorte lateral em
            # drivers antigos que não expõem ScalerCrop.
            return

        try:
            self.picam2.set_controls({"ScalerCrop": recorte})
            print(f"[camera] campo de visão máximo: ScalerCrop={recorte}")
        except Exception as err:
            print(f"[camera] ScalerCrop máximo ignorado pelo driver: {err}")

    def _configurar_foco(self):
        """Mantém objetos próximos focados sem restringir o alcance da lente."""
        if not hasattr(self.picam2, "set_controls"):
            return

        try:
            from libcamera import controls

            if LENS_POSITION is None:
                foco = {"AfMode": controls.AfModeEnum.Continuous}
                # A faixa completa inclui o foco próximo e evita travá-lo só
                # no intervalo normal. Há versões antigas sem AfRange.
                if hasattr(controls, "AfRangeEnum"):
                    foco["AfRange"] = controls.AfRangeEnum.Full
                self.picam2.set_controls(foco)
                print("[camera] autofocus contínuo ativado (faixa completa)")
            else:
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": LENS_POSITION,
                })
                print(f"[camera] foco manual: LensPosition={LENS_POSITION}")
        except Exception as err:
            print(f"[camera] controle de foco ignorado (módulo sem AF?): {err}")

    def _configurar_e_iniciar(self, fps):
        self.capture_fps = float(fps)
        frame_us = int(round(1_000_000 / self.capture_fps))
        print(
            "[camera] captura solicitada em "
            f"{self.capture_fps:.1f} FPS "
            f"({frame_us} us por frame)"
        )
        try:
            video_config = self.picam2.create_video_configuration(
                main={
                    "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                    "format": "RGB888",
                },
                controls={"FrameDurationLimits": (frame_us, frame_us)},
                buffer_count=4,
                # O frame devolvido precisa ser posterior ao pedido. Isso
                # remove até um período de atraso escondido da fila interna.
                queue=False,
            )
        except TypeError:
            try:
                # Versões intermediárias podem aceitar o controle de FPS, mas
                # ainda não conhecer o argumento ``queue``.
                video_config = self.picam2.create_video_configuration(
                    main={
                        "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                        "format": "RGB888",
                    },
                    controls={
                        "FrameDurationLimits": (frame_us, frame_us),
                    },
                    buffer_count=4,
                )
            except TypeError:
                # Compatibilidade final com Picamera2 antigo. Sem controle
                # explícito, o FPS medido impede a aceleração se ele ficar lento.
                video_config = self.picam2.create_video_configuration(
                    main={
                        "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                        "format": "RGB888",
                    },
                )

        self.picam2.configure(video_config)
        self.picam2.start()

    def sensor_modes(self):
        return self.picam2.sensor_modes

    def get_frame(self):
        """Retorna uma imagem BGR 448×252 da câmera de linha.

        O Picamera2 rotula este formato como "RGB888", mas entrega os bytes
        na ordem B,G,R — que já é a ordem nativa do OpenCV. A conversão
        RGB→BGR que existia aqui TROCAVA os canais R e B, e a câmera de
        resgate (`captura_resgate.py`) nunca fez isso.

        Consequência medida na arena: vermelho aparecia com matiz 120 (azul)
        e nunca casava com as faixas 0–10 / 170–180 — a faixa vermelha final
        simplesmente não era detectável. O verde sobrevivia porque foi
        calibrado já em cima da imagem trocada.

        Ao remover a troca, as faixas de matiz do verde precisaram ser
        migradas por `H_correto = 120 − H_trocado` (S e V não mudam, pois
        trocar dois canais não altera máximo nem mínimo). Isso foi feito em
        `config.py` e em `config.ini`.
        """
        raw = self.picam2.capture_array("main")
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        return cv2.resize(raw, (camera_x, camera_y))

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass
