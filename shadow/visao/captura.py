"""Abre a câmera de linha e entrega imagens no tamanho usado pelo detector."""

import math
import time

from config import (CAPTURE_FPS, CAPTURE_FPS_FALLBACK, CAPTURE_HEIGHT,
                    CAPTURE_WIDTH, LENS_POSITION, LINE_CAMERA_INDEX,
                    LINE_CAMERA_SENSOR_ID,
                    LINE_CAMERA_EXPOSURE_VALUE,
                    LINE_CAMERA_LOCK_AUTO_CONTROLS,
                    LINE_CAMERA_WARMUP_S, camera_x, camera_y)


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


def escolher_modo_sensor_campo_aberto(modos_sensor, fps_alvo):
    """Escolhe um modo que mantenha o sensor inteiro dentro do FPS pedido.

    A IMX708 da Camera Module 3 Wide anuncia 1536×864 a 120 FPS, mas esse
    modo já chega recortado ao ISP. O modo 2304×1296 usa o sensor completo
    com binning. Fixar esse modo impede que o seletor automático priorize FPS
    e elimine as laterais antes de ``ScalerCrop`` poder atuar.
    """
    candidatos = []
    for modo in modos_sensor or ():
        try:
            largura, altura = modo["size"]
            bit_depth = int(modo["bit_depth"])
            fps = float(modo["fps"])
            x, y, largura_crop, altura_crop = modo["crop_limits"]
        except (KeyError, TypeError, ValueError):
            continue

        if (
            largura < CAPTURE_WIDTH
            or altura < CAPTURE_HEIGHT
            or not math.isfinite(fps)
            or fps < fps_alvo
            or x != 0
            or y != 0
            or largura_crop <= 0
            or altura_crop <= 0
        ):
            continue
        candidatos.append((fps, largura * altura, (largura, altura), bit_depth))

    if not candidatos:
        return None

    # Entre os modos sem crop que suportam o FPS, preferimos o mais rápido;
    # em empate, o de menor resolução reduz custo no ISP sem reduzir o FoV.
    _, _, tamanho, bit_depth = max(
        candidatos,
        key=lambda candidato: (candidato[0], -candidato[1]),
    )
    return {"output_size": tamanho, "bit_depth": bit_depth}


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


def normalizar_recorte_metadata(recorte_bruto):
    """Converte tuple/libcamera.Rectangle em ``(x, y, w, h)`` validado."""

    try:
        recorte = tuple(int(valor) for valor in recorte_bruto)
    except (TypeError, ValueError):
        try:
            recorte = (
                int(recorte_bruto.x),
                int(recorte_bruto.y),
                int(recorte_bruto.width),
                int(recorte_bruto.height),
            )
        except (AttributeError, TypeError, ValueError):
            return None
    if len(recorte) != 4 or recorte[2] <= 0 or recorte[3] <= 0:
        return None
    return recorte


def normalizar_identidade_sensor(camera_info):
    """Extrai o modelo realmente anunciado por ``global_camera_info``."""
    if not isinstance(camera_info, dict):
        return "unknown"
    value = None
    for key in ("Model", "model", "SensorModel", "sensor_model"):
        candidate = camera_info.get(key)
        if candidate is not None and str(candidate).strip():
            value = str(candidate).strip().casefold()
            break
    if value is None:
        return "unknown"
    return "_".join(value.replace("-", " ").split())


def _normalizar_modo_sensor(modo):
    if not isinstance(modo, dict):
        return None
    try:
        largura, altura = modo["output_size"]
        bit_depth = int(modo["bit_depth"])
        largura = int(largura)
        altura = int(altura)
    except (KeyError, TypeError, ValueError):
        return None
    if largura <= 0 or altura <= 0 or bit_depth <= 0:
        return None
    return {
        "output_size": (largura, altura),
        "bit_depth": bit_depth,
    }


def _normalizar_stream_principal(stream):
    if not isinstance(stream, dict):
        return None
    try:
        tamanho = tuple(int(valor) for valor in stream["size"])
        formato = str(stream["format"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(tamanho) != 2 or tamanho[0] <= 0 or tamanho[1] <= 0:
        return None
    return {"size": tamanho, "format": formato}


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
        selected_info = camera_info[LINE_CAMERA_INDEX]
        self.camera_index = int(LINE_CAMERA_INDEX)
        self.camera_info = (
            dict(selected_info)
            if isinstance(selected_info, dict)
            else {"raw": str(selected_info)}
        )
        self.sensor_id = normalizar_identidade_sensor(self.camera_info)
        self._sensor_mode_applied = None
        self._main_stream_applied = None
        self._scaler_crop_requested = None
        self._scaler_crop_applied = None
        self._capture_fps_confirmed = None
        self._lens_position_confirmed = None
        print(
            "[camera] abrindo camera de segue-linha explicita "
            f"{LINE_CAMERA_INDEX} (flat 2, sensor={self.sensor_id})"
        )
        self.picam2 = Picamera2(camera_num=LINE_CAMERA_INDEX)

        try:
            modos_sensor = self.picam2.sensor_modes
        except (AttributeError, RuntimeError, TypeError):
            modos_sensor = ()
        fps_escolhido = escolher_fps_captura(modos_sensor)
        self._sensor_config = escolher_modo_sensor_campo_aberto(
            modos_sensor,
            fps_escolhido,
        )
        if self._sensor_config is not None:
            print(
                "[camera] modo sem crop selecionado: "
                f"{self._sensor_config['output_size']} "
                f"({fps_escolhido:.1f} FPS solicitados)"
            )
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
            fps_fallback = float(CAPTURE_FPS_FALLBACK)
            self._sensor_config = escolher_modo_sensor_campo_aberto(
                modos_sensor,
                fps_fallback,
            )
            self._configurar_e_iniciar(fps_fallback)

        self._abrir_campo_de_visao()
        # AE/AWB precisam enxergar alguns frames antes do único ciclo de AF.
        self._estabilizar_imagem()
        self._confirmar_geometria_e_fps()
        self._configurar_foco()

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
            self._scaler_crop_requested = tuple(recorte)
            print(
                "[camera] campo de visão máximo solicitado: "
                f"ScalerCrop={recorte}"
            )
        except Exception as err:
            print(f"[camera] ScalerCrop máximo ignorado pelo driver: {err}")

    def _confirmar_geometria_e_fps(self, *, tentativas=6):
        """Confirma por metadata o crop e o FrameDuration realmente aplicados."""

        self._scaler_crop_applied = None
        self._capture_fps_confirmed = None
        metadata_reader = getattr(self.picam2, "capture_metadata", None)
        if metadata_reader is None:
            print(
                "[camera] metadata de crop/FPS indisponível; "
                "calibração competitiva permanecerá bloqueada"
            )
            return

        alvo_crop = self._scaler_crop_requested
        alvo_frame_us = int(round(1_000_000 / float(self.capture_fps)))
        ultimo_crop = None
        ultimo_frame_us = None
        for _ in range(max(int(tentativas), 1)):
            try:
                metadata = metadata_reader()
            except Exception:
                time.sleep(.02)
                continue
            if isinstance(metadata, dict):
                ultimo_crop = normalizar_recorte_metadata(
                    metadata.get("ScalerCrop"))
                try:
                    ultimo_frame_us = float(metadata.get("FrameDuration"))
                except (TypeError, ValueError):
                    ultimo_frame_us = None
            crop_ok = bool(
                alvo_crop is not None and ultimo_crop == alvo_crop)
            fps_ok = bool(
                ultimo_frame_us is not None
                and math.isfinite(ultimo_frame_us)
                and ultimo_frame_us > 0.
                and math.isclose(
                    ultimo_frame_us,
                    alvo_frame_us,
                    rel_tol=.02,
                    abs_tol=50.,
                )
            )
            if crop_ok and fps_ok:
                self._scaler_crop_applied = ultimo_crop
                self._capture_fps_confirmed = 1_000_000. / ultimo_frame_us
                self.capture_fps = self._capture_fps_confirmed
                print(
                    "[camera] crop/FPS confirmados pelo metadata: "
                    f"ScalerCrop={ultimo_crop}, "
                    f"FrameDuration={ultimo_frame_us:.0f} us"
                )
                return
            time.sleep(.02)

        print(
            "[camera] driver não confirmou crop/FPS solicitados "
            f"(crop={ultimo_crop}, FrameDuration={ultimo_frame_us}); "
            "calibração competitiva permanecerá bloqueada"
        )

    def _configurar_foco(self):
        """Foca uma vez na partida e depois mantém a lente imóvel.

        O foco contínuo caçava a pista a poucos centímetros da Camera Module
        3 Wide. Um ciclo automático conserva a adaptação a cada módulo, mas
        travar a posição encontrada remove os movimentos durante a prova.
        """
        if not hasattr(self.picam2, "set_controls"):
            return

        try:
            from libcamera import controls

            if LENS_POSITION is None:
                foco = {}
                # A faixa completa inclui o foco próximo. Há versões
                # antigas do libcamera sem AfRange.
                if hasattr(controls, "AfRangeEnum"):
                    foco["AfRange"] = controls.AfRangeEnum.Full
                if foco:
                    self.picam2.set_controls(foco)

                autofocus = getattr(self.picam2, "autofocus_cycle", None)
                metadata = getattr(self.picam2, "capture_metadata", None)
                if autofocus is None or metadata is None:
                    # Compatibilidade com Picamera2 antigo: nesse caso não é
                    # seguro inventar uma posição de lente.
                    self.picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Continuous,
                    })
                    print("[camera] Picamera2 antigo; autofocus contínuo mantido")
                    return

                try:
                    sucesso = bool(autofocus())
                    posicao_bruta = metadata().get("LensPosition")
                    posicao = float(posicao_bruta)
                except Exception as err:
                    # Mesmo sem metadata, sair do AF impede a lente de caçar
                    # foco durante o percurso. O driver conserva sua posição.
                    self.picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Manual,
                    })
                    print(
                        "[camera] autofocus de partida incompleto "
                        f"({err}); foco atual mantido em modo manual"
                    )
                    return
                if not math.isfinite(posicao):
                    self.picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Manual,
                    })
                    print("[camera] LensPosition inválida; foco atual mantido")
                    return
                posicao = self.aplicar_posicao_lente(posicao)
                resultado = "confirmado" if sucesso else "melhor tentativa"
                print(
                    "[camera] autofocus de partida "
                    f"{resultado}; foco travado em LensPosition={posicao:.3f}"
                )
            else:
                posicao = self.aplicar_posicao_lente(LENS_POSITION)
                print(f"[camera] foco manual: LensPosition={posicao:.3f}")
        except Exception as err:
            print(f"[camera] controle de foco ignorado (módulo sem AF?): {err}")

    def aplicar_posicao_lente(self, lens_position, *, tentativas=6):
        """Aplica foco manual e confirma pelo metadata antes da competição."""
        from visao.calibracao_wide import (LENS_POSITION_TOLERANCE,
                                           validate_lens_position)

        alvo = validate_lens_position(lens_position)
        if (
            not hasattr(self.picam2, "set_controls")
            or not hasattr(self.picam2, "capture_metadata")
        ):
            self._lens_position_confirmed = None
            raise RuntimeError(
                "Picamera2 não permite aplicar e confirmar LensPosition")
        try:
            from libcamera import controls
        except ImportError as error:
            self._lens_position_confirmed = None
            raise RuntimeError(
                "libcamera indisponível para aplicar LensPosition") from error
        try:
            af_manual = controls.AfModeEnum.Manual
        except AttributeError as error:
            self._lens_position_confirmed = None
            raise RuntimeError(
                "libcamera não expõe o modo de foco manual") from error

        try:
            quantidade = int(tentativas)
        except (TypeError, ValueError) as error:
            raise RuntimeError("quantidade de confirmações de foco inválida") from error
        if quantidade <= 0:
            raise RuntimeError("quantidade de confirmações de foco inválida")

        self._lens_position_confirmed = None
        try:
            self.picam2.set_controls({
                "AfMode": af_manual,
                "LensPosition": alvo,
            })
        except Exception as error:
            raise RuntimeError(
                "Picamera2 recusou a LensPosition da calibração") from error
        ultima_posicao = None
        ultimo_erro = None
        for _ in range(quantidade):
            try:
                metadata = self.picam2.capture_metadata()
            except Exception as error:
                ultimo_erro = error
                time.sleep(.02)
                continue
            try:
                confirmada = float(metadata.get("LensPosition"))
            except (AttributeError, TypeError, ValueError):
                confirmada = float("nan")
            if math.isfinite(confirmada):
                ultima_posicao = confirmada
                if math.isclose(
                    confirmada,
                    alvo,
                    rel_tol=0.,
                    abs_tol=LENS_POSITION_TOLERANCE,
                ):
                    self._lens_position_confirmed = confirmada
                    return confirmada
            time.sleep(.02)

        detalhe = (
            f"metadata indisponível: {ultimo_erro}"
            if ultimo_erro is not None and ultima_posicao is None
            else "metadata ausente"
            if ultima_posicao is None
            else f"metadata={ultima_posicao:.4f}, alvo={alvo:.4f}"
        )
        raise RuntimeError(f"LensPosition não foi confirmada ({detalhe})")

    def _estabilizar_imagem(self):
        """Evita que muito preto no quadro altere brilho e cor da pista."""
        if not hasattr(self.picam2, "set_controls"):
            time.sleep(.1)
            return

        try:
            self.picam2.set_controls({
                "AeEnable": True,
                "AwbEnable": True,
                "ExposureValue": float(LINE_CAMERA_EXPOSURE_VALUE),
            })
        except Exception as err:
            print(f"[camera] compensacao de exposicao ignorada: {err}")

        time.sleep(float(LINE_CAMERA_WARMUP_S))
        if not LINE_CAMERA_LOCK_AUTO_CONTROLS:
            print("[camera] exposicao e balanco de branco automaticos ativos")
            return
        if not hasattr(self.picam2, "capture_metadata"):
            print("[camera] metadata indisponivel; AE/AWB permanecem automaticos")
            return

        try:
            metadata = self.picam2.capture_metadata()
            controles = {"AeEnable": False, "AwbEnable": False}
            for nome in ("ExposureTime", "AnalogueGain", "ColourGains"):
                valor = metadata.get(nome)
                if valor is not None:
                    controles[nome] = valor
            self.picam2.set_controls(controles)
            print(
                "[camera] exposicao e balanco de branco travados: "
                f"ExposureTime={metadata.get('ExposureTime', '?')} "
                f"AnalogueGain={metadata.get('AnalogueGain', '?')}"
            )
        except Exception as err:
            print(f"[camera] trava de exposicao/AWB ignorada: {err}")

    def _configurar_e_iniciar(self, fps):
        self._sensor_mode_applied = None
        self._main_stream_applied = None
        self.capture_fps = float(fps)
        frame_us = int(round(1_000_000 / self.capture_fps))
        print(
            "[camera] captura solicitada em "
            f"{self.capture_fps:.1f} FPS "
            f"({frame_us} us por frame)"
        )
        opcoes = {
            "main": {
                "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                "format": "RGB888",
            },
            "controls": {"FrameDurationLimits": (frame_us, frame_us)},
            "buffer_count": 4,
        }
        if self._sensor_config is not None:
            # Disponível no Picamera2 do Raspberry Pi OS Bookworm. Informar o
            # modo exato evita que a escolha automática caia no 1536×864
            # recortado só por ele ser mais rápido.
            opcoes["sensor"] = self._sensor_config

        sensor_mode_sent = False
        try:
            video_config = self.picam2.create_video_configuration(
                **opcoes,
                # O frame devolvido precisa ser posterior ao pedido. Isso
                # remove até um período de atraso escondido da fila interna.
                queue=False,
            )
            sensor_mode_sent = self._sensor_config is not None
        except TypeError:
            try:
                # Versões intermediárias podem aceitar o controle de FPS, mas
                # ainda não conhecer o argumento ``queue``.
                video_config = self.picam2.create_video_configuration(**opcoes)
                sensor_mode_sent = self._sensor_config is not None
            except TypeError:
                # Picamera2 antigo não aceita ``sensor``. Continua funcional,
                # mas não há como garantir o modo do sensor por essa API.
                video_config = self.picam2.create_video_configuration(
                    main=opcoes["main"],
                )

        self.picam2.configure(video_config)
        try:
            configured = self.picam2.camera_configuration()
            actual_main = _normalizar_stream_principal(
                configured.get("main"))
            if actual_main == {
                "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                "format": "RGB888",
            }:
                self._main_stream_applied = actual_main
            else:
                print(
                    "[camera] stream principal não confirmado "
                    f"(recebido={actual_main}); calibração competitiva "
                    "permanecerá bloqueada"
                )
            if sensor_mode_sent:
                actual_mode = _normalizar_modo_sensor(
                    configured.get("sensor"))
                if actual_mode is not None:
                    self._sensor_mode_applied = actual_mode
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
        if self._main_stream_applied is None:
            print(
                "[camera] driver não confirmou resolução/formato do stream; "
                "calibração competitiva permanecerá bloqueada"
            )
        if sensor_mode_sent and self._sensor_mode_applied is None:
            print(
                "[camera] driver não confirmou o modo bruto; "
                "calibração competitiva permanecerá bloqueada"
            )
        self.picam2.start()

    @property
    def sensor_mode(self):
        """Modo bruto que o driver confirmou, ou ``None`` se incerto."""
        if self._sensor_mode_applied is None:
            return None
        return dict(self._sensor_mode_applied)

    @property
    def scaler_crop(self):
        """Maior crop confirmado pelo metadata, ou ``None`` se incerto."""
        return self._scaler_crop_applied

    @property
    def lens_position(self):
        """LensPosition manual confirmada pelo metadata, ou ``None``."""
        return self._lens_position_confirmed

    @property
    def capture_mode_id(self):
        """Assinatura canônica do modo geométrico realmente configurado."""
        from visao.calibracao_wide import build_capture_mode_id

        sensor_esperado = str(LINE_CAMERA_SENSOR_ID).strip().casefold()
        if self.sensor_id.casefold() != sensor_esperado:
            raise RuntimeError(
                "sensor da câmera de linha difere do hardware calibrado: "
                f"{self.sensor_id} != {sensor_esperado}"
            )
        if self._sensor_mode_applied is None:
            raise RuntimeError(
                "modo bruto do sensor não foi confirmado pelo Picamera2")
        if self._main_stream_applied != {
            "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
            "format": "RGB888",
        }:
            raise RuntimeError(
                "resolução/formato do stream principal não foi confirmado")
        if self._scaler_crop_applied is None:
            raise RuntimeError(
                "ScalerCrop máximo não foi confirmado pelo metadata")
        if self._capture_fps_confirmed is None:
            raise RuntimeError(
                "FrameDuration/FPS não foi confirmado pelo metadata")

        return build_capture_mode_id(
            (camera_x, camera_y),
            self._capture_fps_confirmed,
            full_fov=self._sensor_mode_applied is not None,
            sensor_mode=self._sensor_mode_applied,
            scaler_crop=self._scaler_crop_applied,
        )

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
        if raw.ndim != 3 or raw.shape != (camera_y, camera_x, 3):
            raise RuntimeError(
                "stream principal mudou de geometria/formato em runtime: "
                f"shape={getattr(raw, 'shape', None)}, esperado="
                f"({camera_y}, {camera_x}, 3)"
            )
        return raw

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.close()
        except Exception:
            pass
