#!/usr/bin/env python3
"""Resgate — encontra, coleta, seleciona e procura todas as vítimas.

Depois de cada coleta, a vítima prata é selecionada pela garra esquerda e a
preta pela direita. O robô volta à busca pulsada. Duas passagens separadas
pelo marcador verde sem uma coleta no meio encerram a procura. Então o robô
avança até o ultrassônico confirmar 7 cm do retângulo verde. Se houver vítima
prata armazenada, gira 180 graus, alinha de ré e esvazia o lado esquerdo da
caçamba antes de encerrar.

Arquitetura da visão
--------------------
    modelo treinado  -> aparência   (é vítima? prata ou preta?)
    plausibilidade   -> geometria   (cabe fisicamente ali?)
    rastreamento     -> tempo       (aparece de forma consistente?)

O detector clássico anterior misturava as três coisas em dez portões
encadeados de aparência. Medido: recall de 45% em imagens novas, e queda para
20% com uma piora modesta em cada portão — porque dez portões em série
multiplicam a fragilidade. A separação acima existe para que trocar de arena
afete só o modelo, que é a única peça retreinável.

Os marcadores continuam clássicos, de propósito: a cromaticidade separa
marcador de cadeira vermelha com folga medida (124-148 contra 63-79), e isso
não precisa de treino.

Exemplos::

    python3 shadow/resgate.py --debug                 # só visão, sem motores
    python3 shadow/resgate.py --sem-vitimas --debug   # só marcadores
    python3 shadow/resgate.py --drive --camera-index 0 --debug
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

import config_resgate as cfg  # noqa: E402
from controle.aproximacao_resgate import (  # noqa: E402
    BallApproachController,
    MotionCommand,
)
from controle.busca_pulsada import make_search_controller  # noqa: E402
from controle.coleta_resgate import BallPickupSequencer  # noqa: E402
from controle.contador_verde_resgate import (  # noqa: E402
    BUSCA_CONCLUIR,
    BUSCA_REINICIAR,
    ContadorVerdeBusca,
    decidir_apos_varredura,
)
from controle.deposito_cinza_resgate import (  # noqa: E402
    SequenciadorDepositoCinza,
)
from controle.parada_obstaculo import MonitorObstaculo  # noqa: E402
from controle.retangulo_verde_resgate import (  # noqa: E402
    ControladorRetanguloVerde,
)
from visao import overlay_resgate  # noqa: E402
from visao.marcador_resgate import (  # noqa: E402
    GreenRectangleDetector,
    MarkerDetector,
    color_masks,
)
from visao.resgate_assincrono import (  # noqa: E402
    FreshDetectionGate,
    LatestFrameBallDetector,
    LatestFrameSource,
)
from visao.vitima_yolo import (  # noqa: E402
    ModeloAusenteError,
    VictimDetector,
    VictimModel,
    modelo_disponivel,
)


JANELA = "Shadow2026 - resgate (visao)"
INTERVALO_CONTROLE_OCIOSO_S = 0.25
INTERVALO_LOG_S = 0.50
TICK_S = 0.005

EXIT_OK = 0
EXIT_SEM_MODELO = 4


class VideoSource:
    """Reprodução de vídeo gravado. Nunca aciona motores."""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"nao foi possivel abrir o video: {path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_period = 1.0 / fps if 1.0 <= fps <= 120.0 else 1.0 / 30.0
        self.next_frame_at = time.monotonic()

    def get_frame(self):
        restante = self.next_frame_at - time.monotonic()
        if restante > 0:
            time.sleep(restante)
        ok, frame = self.capture.read()
        self.next_frame_at = max(
            self.next_frame_at + self.frame_period, time.monotonic())
        return frame if ok else None

    def close(self):
        self.capture.release()


def _taxa(amostras):
    if len(amostras) < 2:
        return 0.0
    decorrido = amostras[-1][1] - amostras[0][1]
    return (
        (amostras[-1][0] - amostras[0][0]) / decorrido
        if decorrido > 0 else 0.0
    )


def _melhor_esforco(rotulo, acao):
    try:
        return acao()
    except Exception as err:                        # noqa: BLE001
        print(f"[resgate] falha ao {rotulo}: {err}")
        return None


def _aplicar_acoes_coleta(
    passo,
    arduino,
    acao_direcao,
    epoca_serial_esperada=None,
):
    """Aplica uma única vez os eventos físicos emitidos pelo sequenciador."""
    avanco_iniciado = False
    motor_parado = False

    def serial_mudou():
        return (
            epoca_serial_esperada is not None
            and (
                not arduino.connected
                or arduino.connection_epoch != epoca_serial_esperada
            )
        )

    def abortar(detalhe):
        # Se o avanço começou ou as garras seriam acionadas, uma falha precisa
        # parar as rodas neste mesmo tick.
        if (
            not motor_parado
            and (avanco_iniciado or passo.gripper_action is not None)
        ):
            try:
                acao_direcao()
            except Exception:
                pass
        return detalhe

    try:
        if serial_mudou():
            return abortar("serial mudou durante a coleta")

        if passo.motor_action == "hold":
            # PARAR também corta o Futaba no firmware. LADO 0 0 mantém as
            # rodas zeradas sem interromper o pulso temporizado do elevador.
            if arduino.lado(0, 0) is False:
                return "LADO 0 0 nao foi enviado pela serial"
        elif passo.motor_action == "stop":
            if acao_direcao() is False:
                return "PARAR nao foi enviado pela serial"
            motor_parado = True
        elif passo.motor_action not in ("", "forward"):
            return f"acao de motor desconhecida: {passo.motor_action}"

        if serial_mudou():
            return abortar("serial mudou durante a coleta")

        if passo.stop_futaba:
            if arduino.parar_futaba() is False:
                return "FUTABA PARAR nao foi enviado pela serial"
            if serial_mudou():
                return abortar("serial mudou durante a coleta")

        if passo.futaba_action is not None:
            potencia, tempo_ms = passo.futaba_action
            if arduino.futaba(potencia, tempo_ms) is False:
                return "FUTABA nao foi enviado pela serial"
            if serial_mudou():
                return abortar("serial mudou durante a coleta")

        if passo.motor_action == "forward":
            if acao_direcao(passo.angle, passo.speed) is False:
                return "comando de avanco nao foi enviado pela serial"
            avanco_iniciado = True
            if serial_mudou():
                return abortar("serial mudou durante a coleta")

        if passo.gripper_action is not None:
            esquerda, direita = passo.gripper_action
            if arduino.garras(esquerda, direita) is False:
                return abortar(
                    "comando simultaneo das garras nao foi enviado")
            if serial_mudou():
                return abortar("serial mudou durante a coleta")
    except Exception as err:                         # noqa: BLE001
        return abortar(f"falha ao comandar coleta: {err}")
    return None


def _aplicar_acoes_deposito_cinza(
    passo,
    arduino,
    acao_direcao,
    epoca_serial_esperada=None,
):
    """Aplica motor antes da caçamba e bloqueia reconexão no meio da etapa."""

    def serial_mudou():
        return (
            epoca_serial_esperada is not None
            and (
                not arduino.connected
                or arduino.connection_epoch != epoca_serial_esperada
            )
        )

    def abortar(detalhe):
        try:
            acao_direcao()
        except Exception:
            pass
        return detalhe

    try:
        if serial_mudou():
            return "serial mudou durante o deposito da vitima prata"
        if acao_direcao(passo.angle, passo.speed) is False:
            return abortar(
                "comando dos motores do deposito nao foi enviado")
        if serial_mudou():
            return abortar(
                "serial mudou durante o deposito da vitima prata")

        if passo.bucket_delta is not None:
            if arduino.servo("CACAMBA", passo.bucket_delta) is False:
                return abortar(
                    "comando do servo da cacamba nao foi enviado")
            if serial_mudou():
                return abortar(
                    "serial mudou durante o deposito da vitima prata")
    except Exception as err:                         # noqa: BLE001
        return abortar(f"falha ao comandar deposito: {err}")
    return None


def _preparar_deposito_cinza(vitimas_prata_resgatadas):
    """Cria a sequencia fisica final sem depender do contador da coleta."""
    quantidade = max(int(vitimas_prata_resgatadas), 0)
    return (
        SequenciadorDepositoCinza(),
        MotionCommand(
            SequenciadorDepositoCinza.INICIO,
            detail=(
                f"{quantidade} vitima(s) prata registrada(s); "
                "iniciando giro e deposito esquerdo"
            ),
        ),
    )


def _armar_coleta_confirmada(
    comando,
    coleta,
    arduino,
    parada_enviada,
    epoca_movimento,
):
    """Só arma a coleta após confirmação visual e PARAR serial estável."""
    if (
        comando.state != BallApproachController.NEAR
        or coleta.started
    ):
        return False
    if (
        not comando.terminal
        or not comando.pickup_in_range
        or comando.pickup_confirmations < cfg.BALL_STOP_CONFIRM_FRAMES
    ):
        raise RuntimeError(
            "coleta recusada: proximidade visual nao foi confirmada")
    if (
        parada_enviada is not True
        or epoca_movimento is None
        or arduino is None
        or not arduino.connected
        or arduino.connection_epoch != epoca_movimento
    ):
        raise RuntimeError(
            "coleta recusada: PARAR da aproximacao nao teve "
            "escrita serial estavel")
    if comando.target_kind not in ("silver", "black"):
        raise RuntimeError(
            "coleta recusada: cor da vitima nao foi confirmada")
    if not coleta.start(comando.target_kind):
        raise RuntimeError("coleta recusada: sequencia ja iniciada")
    return True


class MarkerPair:
    """Verde e vermelho detectados no mesmo frame, para identificação.

    Durante a busca eles apenas contam passagens. Depois que a busca confirma
    que não há mais vítimas, somente o verde pode comandar a rota final.
    """

    def __init__(self):
        self.detectors = {
            "green": MarkerDetector("green"),
            "red": MarkerDetector("red"),
        }
        self.detector_retangulo_verde = GreenRectangleDetector()
        self.detections = {"green": None, "red": None}
        self.confirmados = {"green": False, "red": False}
        self.mascaras = {"green": None, "red": None}

    def update(self, frame, timestamp):
        # Uma conversão HSV só, reaproveitada pelos dois detectores.
        mascaras = color_masks(frame)
        self.mascaras = mascaras
        verde_padrao = self.detectors["green"].detect(
            frame, timestamp=timestamp, masks=mascaras)
        verde_retangulo = self.detector_retangulo_verde.detect(
            frame, timestamp=timestamp, masks=mascaras)
        verdes = tuple(
            deteccao
            for deteccao in (verde_padrao, verde_retangulo)
            if deteccao is not None
        )
        self.detections["green"] = (
            max(
                verdes,
                key=lambda item: (
                    bool(item.confirmed),
                    bool(item.track_locked),
                    item.confidence,
                    item.area,
                ),
            )
            if verdes else None
        )
        self.detections["red"] = self.detectors["red"].detect(
            frame, timestamp=timestamp, masks=mascaras)

        for tipo, deteccao in self.detections.items():
            if deteccao is not None and deteccao.confirmed:
                self.confirmados[tipo] = True
        return dict(self.detections)

    def resumo(self):
        partes = []
        for tipo in ("green", "red"):
            deteccao = self.detections[tipo]
            if deteccao is None:
                if tipo == "green":
                    partes.append(
                        "green:- "
                        f"mask={self.detector_retangulo_verde.last_mask_ratio:.1%}"
                    )
                else:
                    partes.append(f"{tipo}:-")
            else:
                partes.append(
                    f"{tipo}:{deteccao.confidence:.2f}"
                    f"{'*' if deteccao.confirmed else ''}")
        return " ".join(partes)

    def reset(self):
        for detector in self.detectors.values():
            detector.reset()
        self.detector_retangulo_verde.reset()
        self.detections = {"green": None, "red": None}
        self.mascaras = {"green": None, "red": None}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resgate: procura, coleta, seleciona por cor e busca a proxima "
            "vitima"))
    parser.add_argument(
        "--camera-index", type=int,
        help=(
            "indice da camera de resgate; sem --drive usa "
            f"{cfg.RESCUE_CAMERA_INDEX}"))
    parser.add_argument(
        "--target", choices=("any", "black", "silver"), default="any",
        help="tipo de vitima aceito (padrao: any)")
    parser.add_argument(
        "--drive", action="store_true",
        help=(
            "AUTORIZA movimento; sem isto o Arduino fica em PARAR "
            "(o LED ainda e apagado no uso da camera real)"))
    parser.add_argument(
        "--debug", action="store_true",
        help="mostra a camera anotada; q ou Esc encerra")
    parser.add_argument(
        "--video", type=Path,
        help="processa um video gravado; sempre sem motores")
    parser.add_argument(
        "--sem-vitimas", action="store_true",
        help=(
            "desliga o detector de vitimas e exercita so os marcadores; "
            "util antes de o modelo existir"))
    parser.add_argument(
        "--sem-marcadores", action="store_true",
        help="desliga os marcadores e exercita so as vitimas")
    args = parser.parse_args()
    if args.video is not None and args.drive:
        parser.error("--drive nao pode ser usado junto com --video")
    if args.drive and args.camera_index is None:
        parser.error(
            "--drive exige --camera-index explicito; confirme antes com "
            "--debug qual imagem e a da camera frontal de resgate")
    if args.camera_index is None:
        args.camera_index = cfg.RESCUE_CAMERA_INDEX
    return args


def preparar_detector_de_vitimas(args):
    """Carrega o modelo ou explica exatamente o que falta."""
    if args.sem_vitimas:
        print("[resgate] detector de vitimas DESLIGADO (--sem-vitimas)")
        return None
    if not modelo_disponivel():
        modelo = VictimModel()
        try:
            modelo.carregar()
        except ModeloAusenteError as err:
            print(f"\n[resgate] {err}\n")
            return False
    modelo = VictimModel().carregar()
    print(f"[resgate] modelo de vitimas carregado: {modelo.caminho}")
    return VictimDetector(model=modelo, target_kind=args.target)


def main():
    args = parse_args()
    fonte = None
    arduino = None
    trava = None
    captura = None
    trabalhador = None
    marcadores = None
    controlador = None
    controlador_verde = None
    monitor_chegada_verde = None
    deposito_cinza = None
    busca = None
    coleta = None
    contador_verde = ContadorVerdeBusca()
    sessao_hardware = args.video is None

    ultimo_estado = None
    ultimo_detalhe = None
    ultimo_log = 0.0
    ultimo_controle_ocioso = 0.0
    proxima_atualizacao_ultrassom_verde = 0.0
    epoca_busca = None
    epoca_coleta = None
    epoca_verde = None
    epoca_deposito_cinza = None
    varreduras_sem_vitima = 0
    vitimas_resgatadas = 0
    vitimas_prata_resgatadas = 0
    amostras_captura = deque(maxlen=60)
    instantes_deteccao = deque(maxlen=30)

    detector = preparar_detector_de_vitimas(args)
    if detector is False:
        return EXIT_SEM_MODELO

    try:
        if sessao_hardware:
            from controle.trava_motores import MotorLockError, MotorOwnerLock
            trava = MotorOwnerLock(
                "aproximacao-resgate" if args.drive else "visao-resgate")
            try:
                trava.acquire()
            except MotorLockError as err:
                raise RuntimeError(
                    f"modo de resgate recusado: {err}") from err

            from comunicacao_serial.arduino import Arduino
            from controle.direcao import init_steering, steer

            arduino = Arduino()
            init_steering(arduino)
            steer()
            arduino.led("APAGADO")
            print(
                "[resgate] LED APAGADO antes de abrir a camera; "
                "motores em PARAR")

        if args.video is not None:
            fonte = VideoSource(args.video)
            print(f"[resgate] replay sem motores: {args.video}")
        else:
            from visao.captura_resgate import RescueCamera
            fonte = RescueCamera(args.camera_index)

        captura = LatestFrameSource(fonte)
        if detector is not None:
            trabalhador = LatestFrameBallDetector(
                detector,
                max_width=cfg.RESCUE_DETECTOR_MAX_WIDTH,
                max_height=cfg.RESCUE_DETECTOR_MAX_HEIGHT,
            )
        portao = FreshDetectionGate(
            cfg.VICTIM_ACQUIRE_HITS,
            max_misses=cfg.BALL_FRESH_GATE_MAX_MISSES,
        )
        if not args.sem_marcadores:
            marcadores = MarkerPair()

        inicio = time.monotonic()
        armado_em = (
            inicio + cfg.RESCUE_ARM_DELAY_S if args.drive else inicio)
        controlador = (
            None if args.drive
            else BallApproachController(start_time=armado_em))
        busca = (
            make_search_controller(start_time=armado_em)
            if args.drive else None)
        coleta = BallPickupSequencer()
        comando = MotionCommand(
            "ARMING" if args.drive else BallApproachController.WAIT_TARGET,
            detail=(
                "camera ativa; mantendo PARAR durante a contagem"
                if args.drive
                else "parado; aguardando confirmacao temporal"),
        )

        sequencia_frame = 0
        sequencia_resultado = 0
        frame_atual = None
        resultado_atual = None
        deteccao_atual = None
        marcadores_atuais = {}
        metricas = None

        if args.drive:
            print(
                "[resgate] MOVIMENTO AUTORIZADO. Mantendo PARAR por "
                f"{cfg.RESCUE_ARM_DELAY_S:.0f} s.")
        else:
            print(
                "[resgate] modo de visao: motores desativados. "
                "Use --drive so depois de validar a visao.")

        while True:
            pacote = captura.poll(sequencia_frame)
            frame_novo = pacote is not None
            if frame_novo:
                sequencia_frame = pacote.sequence
                frame_atual = pacote.frame
                amostras_captura.append(
                    (pacote.sequence, pacote.captured_at))
            elif captura.ended:
                print("[resgate] fim da fonte de imagem")
                break

            agora = time.monotonic()
            armado = agora >= armado_em

            if (
                trabalhador is not None
                and frame_novo
                and armado
                and not coleta.started
                and controlador_verde is None
                and deposito_cinza is None
            ):
                trabalhador.submit(
                    frame_atual,
                    captured_at=pacote.captured_at,
                    source_sequence=pacote.sequence,
                )

            if (
                marcadores is not None
                and frame_novo
                and not coleta.started
                and deposito_cinza is None
            ):
                marcadores_atuais = marcadores.update(
                    frame_atual, pacote.captured_at)
                if busca is not None:
                    somou_verde = contador_verde.observar(
                        marcadores_atuais.get("green"),
                        permitido=busca.frame_allowed(pacote.captured_at),
                    )
                    if somou_verde:
                        print(
                            "[resgate] passagem verde "
                            f"{contador_verde.quantidade}/"
                            f"{contador_verde.necessario}")
                        if args.drive and contador_verde.completo:
                            # A segunda passagem foi confirmada com o robo
                            # parado durante SEARCH_OBSERVE. O painel ja esta
                            # no quadro: primeiro centralizar e aproximar pela
                            # camera; o ultrassonico continua desabilitado.
                            if trabalhador is not None:
                                trabalhador.reset_tracking()
                            portao.reset()
                            busca = None
                            controlador = None
                            controlador_verde = ControladorRetanguloVerde(
                                start_time=agora,
                            )
                            monitor_chegada_verde = None
                            proxima_atualizacao_ultrassom_verde = 0.0
                            epoca_busca = None
                            epoca_verde = (
                                arduino.connection_epoch
                                if arduino is not None else None
                            )
                            resultado_atual = None
                            deteccao_atual = None
                            ultimo_controle_ocioso = 0.0
                            print(
                                "[resgate] GREEN_ROUTE_START: segundo verde "
                                "confirmado; alinhando primeiro pela camera")

            resultado = None
            if trabalhador is not None:
                resultado = trabalhador.poll(sequencia_resultado)
                if not trabalhador.is_alive:
                    trabalhador.poll(sequencia_resultado)
                    raise RuntimeError(
                        "detector assincrono encerrou inesperadamente")

            agora = time.monotonic()
            comando_atualizado = False
            passo_coleta = None
            passo_deposito_cinza = None
            coleta_concluida = None
            distancia_chegada_verde_mm = None
            distancia_atual_verde_mm = None
            medicao_ultrassom_verde_atualizada = False
            ultrassonico_verde_sem_eco = False
            ultrassonico_verde_falhou = False

            if (
                args.drive
                and controlador_verde is not None
                and controlador_verde.ultrassom_habilitado
                and monitor_chegada_verde is None
            ):
                monitor_chegada_verde = MonitorObstaculo(
                    distancia_parada_mm=(
                        cfg.RESCUE_GREEN_ARRIVAL_DISTANCE_MM
                    ),
                    confirmacoes=(
                        cfg.RESCUE_GREEN_ULTRASONIC_CONFIRM_READINGS
                    ),
                    tamanho_historico=(
                        cfg.RESCUE_GREEN_ULTRASONIC_CONFIRM_READINGS
                    ),
                )
                proxima_atualizacao_ultrassom_verde = 0.0
                print(
                    "[resgate] camera alinhada; ultrassonico habilitado")

            if (
                args.drive
                and controlador_verde is not None
                and monitor_chegada_verde is not None
                and arduino is not None
            ):
                if agora >= proxima_atualizacao_ultrassom_verde:
                    leituras_antes = monitor_chegada_verde.leituras_concluidas
                    monitor_chegada_verde.atualizar(arduino, agora=agora)
                    medicao_ultrassom_verde_atualizada = (
                        monitor_chegada_verde.leituras_concluidas
                        != leituras_antes
                    )
                    proxima_atualizacao_ultrassom_verde = (
                        agora
                        + cfg.RESCUE_GREEN_ULTRASONIC_POLL_INTERVAL_S
                    )

                distancia_chegada_verde_mm = (
                    monitor_chegada_verde.distancia_confirmada_mm)
                distancia_atual_verde_mm = (
                    monitor_chegada_verde.ultima_distancia_valida_mm)
                ultrassonico_verde_sem_eco = (
                    monitor_chegada_verde.leituras_invalidas_consecutivas > 0
                )
                ultrassonico_verde_falhou = (
                    monitor_chegada_verde.leituras_invalidas_consecutivas
                    >= cfg.RESCUE_GREEN_ULTRASONIC_MAX_NO_ECHO
                )

            if not armado:
                restante = max(armado_em - agora, 0.0)
                comando = MotionCommand(
                    "ARMING",
                    detail=f"camera fluida; PARAR por mais {restante:.1f} s")
            elif coleta.started:
                # Um resultado que já estava em processamento pode chegar
                # depois do início da coleta. Ele serve apenas para telemetria:
                # a visão nunca volta a comandar as rodas nesta sequência.
                if resultado is not None:
                    sequencia_resultado = resultado.sequence
                    metricas = resultado
                    instantes_deteccao.append(resultado.completed_at)
                resultado_atual = None
                deteccao_atual = None

                if (
                    arduino is not None
                    and epoca_coleta is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_coleta
                    )
                ):
                    passo_coleta = coleta.fail(
                        "serial mudou durante a coleta; sequencia cancelada")
                else:
                    passo_coleta = coleta.update(now=agora)

                comando = passo_coleta.motion_command()
                comando_atualizado = True
            elif deposito_cinza is not None:
                if resultado is not None:
                    sequencia_resultado = resultado.sequence
                    metricas = resultado
                    instantes_deteccao.append(resultado.completed_at)
                resultado_atual = None
                deteccao_atual = None
                passo_deposito_cinza = deposito_cinza.update(now=agora)
                comando = passo_deposito_cinza.motion_command()
                comando_atualizado = True
            elif controlador_verde is not None:
                # A camera primeiro confirma, centraliza e se aproxima do
                # verde. Somente depois da parada visual o ultrassonico e
                # habilitado para os centimetros finais.
                if resultado is not None:
                    sequencia_resultado = resultado.sequence
                    metricas = resultado
                    instantes_deteccao.append(resultado.completed_at)
                resultado_atual = None
                deteccao_atual = None

                forma = (
                    frame_atual.shape if frame_atual is not None
                    else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                          cfg.RESCUE_CAMERA_MAX_WIDTH, 3))
                dados_ultrassom = {
                    "distancia_chegada_mm": distancia_chegada_verde_mm,
                    "distancia_atual_mm": distancia_atual_verde_mm,
                    "ultrassonico_sem_eco": ultrassonico_verde_sem_eco,
                    "ultrassonico_falhou": ultrassonico_verde_falhou,
                }
                if (
                    distancia_chegada_verde_mm is not None
                    or medicao_ultrassom_verde_atualizada
                ):
                    comando = controlador_verde.update(
                        marcadores_atuais.get("green"),
                        forma,
                        now=agora,
                        **dados_ultrassom,
                    )
                    comando_atualizado = True
                    ultimo_controle_ocioso = agora
                elif frame_novo:
                    comando = controlador_verde.update(
                        marcadores_atuais.get("green"),
                        forma,
                        mascara_verde=(
                            marcadores.mascaras.get("green")
                            if marcadores is not None else None
                        ),
                        timestamp_frame=pacote.captured_at,
                        now=agora,
                        **dados_ultrassom,
                    )
                    comando_atualizado = True
                    ultimo_controle_ocioso = agora
                elif (
                    agora - ultimo_controle_ocioso
                    >= INTERVALO_CONTROLE_OCIOSO_S
                ):
                    comando = controlador_verde.update(
                        None, forma, now=agora, **dados_ultrassom)
                    comando_atualizado = True
                    ultimo_controle_ocioso = agora
            elif detector is None:
                comando = MotionCommand(
                    "MARCADORES",
                    detail=(
                        "somente marcadores: "
                        f"{marcadores.resumo() if marcadores else '-'}"),
                )
            elif resultado is not None:
                sequencia_resultado = resultado.sequence
                metricas = resultado
                instantes_deteccao.append(resultado.completed_at)
                idade = agora - resultado.captured_at

                if idade > cfg.BALL_FRAME_STALE_S:
                    # Nunca mover com imagem que venceu durante o processamento.
                    comando = (
                        busca.update(None, now=agora) if busca is not None
                        else controlador.update(
                            resultado.detection, resultado.frame_shape,
                            now=agora))
                    comando_atualizado = True
                    resultado_atual = None
                    deteccao_atual = None
                    portao.reset()
                    ultimo_controle_ocioso = agora
                else:
                    resultado_atual = resultado
                    if (
                        busca is not None
                        and not busca.frame_allowed(resultado.captured_at)
                    ):
                        # Frame capturado com o chassi girando nao confirma.
                        portao.reset()
                        confirmada = None
                    else:
                        confirmada = portao.accept(resultado.detection)
                    deteccao_atual = confirmada

                    if busca is not None:
                        comando = busca.update(confirmada, now=agora)
                        if busca.target_acquired:
                            controlador = BallApproachController(
                                start_time=agora)
                            comando = controlador.update(
                                confirmada, resultado.frame_shape, now=agora)
                            busca = None
                            epoca_busca = None
                    else:
                        comando = controlador.update(
                            confirmada, resultado.frame_shape, now=agora)
                    comando_atualizado = True
            elif (
                agora - ultimo_controle_ocioso
                >= INTERVALO_CONTROLE_OCIOSO_S
            ):
                forma = (
                    frame_atual.shape if frame_atual is not None
                    else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                          cfg.RESCUE_CAMERA_MAX_WIDTH, 3))
                comando = (
                    busca.update(None, now=agora) if busca is not None
                    else controlador.update(None, forma, now=agora))
                comando_atualizado = True
                ultimo_controle_ocioso = agora

            # Alvo perdido durante a aproximacao: volta a procurar em vez de
            # esperar parado ate o timeout longo.
            if (
                args.drive
                and busca is None
                and controlador is not None
                and not coleta.started
                and comando.state == BallApproachController.WAIT_TARGET
            ):
                if trabalhador is not None:
                    trabalhador.reset_tracking()
                portao.reset()
                busca = make_search_controller(start_time=agora)
                controlador = None
                resultado_atual = None
                deteccao_atual = None
                epoca_busca = None
                ultimo_controle_ocioso = 0.0
                comando = busca.update(None, now=agora)
                comando_atualizado = True

            epoca_movimento = None
            movimento_enviado = None
            if args.drive and arduino is not None and comando_atualizado:
                from controle.direcao import steer
                if passo_coleta is not None:
                    erro_coleta = _aplicar_acoes_coleta(
                        passo_coleta,
                        arduino,
                        steer,
                        epoca_serial_esperada=epoca_coleta,
                    )
                    if erro_coleta is None:
                        concluido_em = time.monotonic()
                        if passo_coleta.futaba_action is not None:
                            coleta.mark_futaba_started(now=concluido_em)
                        if passo_coleta.motor_action == "forward":
                            coleta.mark_forward_started(now=concluido_em)
                        if passo_coleta.gripper_action is not None:
                            coleta.mark_grippers_started(now=concluido_em)
                        if (
                            passo_coleta.state
                            == BallPickupSequencer.CARRY_READY
                        ):
                            if not coleta.resume_selection():
                                raise RuntimeError(
                                    "nao foi possivel iniciar a selecao "
                                    "da vitima")
                        elif (
                            passo_coleta.state
                            == BallPickupSequencer.COMPLETE
                        ):
                            coleta_concluida = coleta.target_kind
                    else:
                        passo_coleta = coleta.fail(erro_coleta)
                        comando = passo_coleta.motion_command()
                        _aplicar_acoes_coleta(
                            passo_coleta,
                            arduino,
                            steer,
                        )
                elif passo_deposito_cinza is not None:
                    epoca_movimento = arduino.connection_epoch
                    erro_deposito = _aplicar_acoes_deposito_cinza(
                        passo_deposito_cinza,
                        arduino,
                        steer,
                        epoca_serial_esperada=epoca_deposito_cinza,
                    )
                    movimento_enviado = erro_deposito is None

                    if erro_deposito is not None:
                        passo_deposito_cinza = deposito_cinza.fail(
                            erro_deposito)
                        comando = passo_deposito_cinza.motion_command()
                        steer()
                    else:
                        deposito_cinza.notify_command_written(
                            passo_deposito_cinza.state,
                            now=time.monotonic(),
                        )
                else:
                    epoca_movimento = arduino.connection_epoch
                    movimento_enviado = steer(
                        comando.angle, comando.speed)
                    if movimento_enviado is False:
                        raise RuntimeError(
                            "comando de movimento nao foi enviado pela serial")
                    concluido_em = time.monotonic()
                    if busca is not None:
                        if busca.consume_tracking_reset():
                            if trabalhador is not None:
                                trabalhador.reset_tracking()
                            portao.reset()
                            resultado_atual = None
                            deteccao_atual = None
                        if comando.state == busca.START:
                            epoca_busca = arduino.connection_epoch
                        if busca.notify_command_written(
                            comando.state, concluido_em
                        ):
                            portao.reset()
                            resultado_atual = None
                            deteccao_atual = None
                            ultimo_controle_ocioso = concluido_em
                    elif controlador_verde is not None:
                        if controlador_verde.consume_tracking_reset():
                            if marcadores is not None:
                                marcadores.reset()
                            marcadores_atuais = {}
                        controlador_verde.notify_command_written(
                            comando.state, concluido_em)

            if arduino is not None:
                arduino.refresh(fail_closed=True)
                if (
                    epoca_movimento is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_movimento
                    )
                ):
                    raise RuntimeError(
                        "serial mudou depois do comando visual; "
                        "alvo invalidado e motores parados")
                if (
                    busca is not None
                    and epoca_busca is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_busca
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante o giro de busca; "
                        "cobertura invalidada e motores parados")
                if (
                    coleta.started
                    and epoca_coleta is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_coleta
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante a coleta; "
                        "motores e Futaba parados")
                if (
                    controlador_verde is not None
                    and epoca_verde is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_verde
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante a ida ao retangulo verde; "
                        "motores parados")
                if (
                    deposito_cinza is not None
                    and epoca_deposito_cinza is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch
                        != epoca_deposito_cinza
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante o deposito da vitima prata; "
                        "motores parados e cacamba nao comandada")

            if (
                args.drive
                and controlador_verde is not None
                and comando.state == ControladorRetanguloVerde.CONCLUIDO
                and comando.terminal
            ):
                print(
                    "[resgate] chegada a 7 cm confirmada; vitimas prata "
                    f"armazenadas={vitimas_prata_resgatadas}")
                controlador_verde = None
                monitor_chegada_verde = None
                deposito_cinza, comando = _preparar_deposito_cinza(
                    vitimas_prata_resgatadas)
                epoca_verde = None
                epoca_deposito_cinza = (
                    arduino.connection_epoch
                    if arduino is not None else None
                )
                ultimo_controle_ocioso = 0.0
                comando_atualizado = False

            if coleta_concluida is not None:
                vitimas_resgatadas += 1
                if coleta_concluida == "silver":
                    vitimas_prata_resgatadas += 1
                contador_verde.reset()
                varreduras_sem_vitima = 0
                epoca_coleta = None
                coleta = BallPickupSequencer()
                if trabalhador is not None:
                    trabalhador.reset_tracking()
                portao.reset()
                if marcadores is not None:
                    marcadores.reset()
                    marcadores_atuais = {}
                busca = make_search_controller(start_time=agora)
                controlador = None
                resultado_atual = None
                deteccao_atual = None
                epoca_busca = None
                ultimo_controle_ocioso = 0.0
                comando = MotionCommand(
                    "RESCUE_SELECTED",
                    detail=(
                        f"vitima {coleta_concluida} selecionada; "
                        f"total {vitimas_resgatadas}; iniciando nova busca"),
                )
                comando_atualizado = False

            if (
                args.drive
                and busca is not None
                and comando.state == busca.COMPLETE
                and comando.terminal
            ):
                varreduras_sem_vitima += 1
                decisao_busca = decidir_apos_varredura(
                    contador_verde, varreduras_sem_vitima)
                if decisao_busca == BUSCA_CONCLUIR:
                    if marcadores is None:
                        comando = MotionCommand(
                            "GREEN_FINAL_FAULT",
                            detail=(
                                "busca terminou, mas os marcadores foram "
                                "desativados; robo parado"),
                            terminal=True,
                        )
                    else:
                        if trabalhador is not None:
                            trabalhador.reset_tracking()
                        portao.reset()
                        busca = None
                        controlador = None
                        controlador_verde = ControladorRetanguloVerde(
                            start_time=agora,
                        )
                        monitor_chegada_verde = None
                        proxima_atualizacao_ultrassom_verde = 0.0
                        epoca_busca = None
                        epoca_verde = (
                            arduino.connection_epoch
                            if arduino is not None else None
                        )
                        resultado_atual = None
                        deteccao_atual = None
                        ultimo_controle_ocioso = 0.0
                        comando = MotionCommand(
                            "GREEN_ROUTE_START",
                            detail=(
                                "verde visto em duas passagens separadas e "
                                "nenhuma vitima encontrada; "
                                "alinhando ao retangulo verde pela camera"),
                        )
                        comando_atualizado = False
                elif decisao_busca == BUSCA_REINICIAR:
                    busca = make_search_controller(start_time=agora)
                    epoca_busca = None
                    ultimo_controle_ocioso = 0.0
                    comando = MotionCommand(
                        "SEARCH_RESTART",
                        detail=(
                            f"volta {varreduras_sem_vitima} sem vitima; "
                            f"verde {contador_verde.quantidade}/"
                            f"{contador_verde.necessario}; "
                            "iniciando outra busca pulsada"),
                    )
                    comando_atualizado = False
                else:
                    comando = MotionCommand(
                        "RESCUE_SEARCH_FAULT",
                        detail=(
                            "limite seguro de varreduras atingido sem duas "
                            "passagens verdes; robo parado"),
                        terminal=True,
                    )

            if (
                args.drive
                and not coleta.started
                and _armar_coleta_confirmada(
                    comando,
                    coleta,
                    arduino,
                    movimento_enviado,
                    epoca_movimento,
                )
            ):
                epoca_coleta = arduino.connection_epoch
                if trabalhador is not None:
                    trabalhador.reset_tracking()
                portao.reset()
                busca = None
                controlador = None
                resultado_atual = None
                deteccao_atual = None
                ultimo_controle_ocioso = 0.0
                print(
                    f"[coleta] vitima {coleta.target_kind} confirmada; "
                    "avancando, baixando, avancando, fechando e selecionando")

            log_agora = time.monotonic()
            if (
                comando.state != ultimo_estado
                or (
                    comando.detail != ultimo_detalhe
                    and log_agora - ultimo_log >= INTERVALO_LOG_S
                )
            ):
                print(f"[resgate] {comando.state}: {comando.detail}")
                ultimo_estado = comando.state
                ultimo_detalhe = comando.detail
                ultimo_log = log_agora

            if args.debug and frame_atual is not None:
                idade_resultado = (
                    time.monotonic() - resultado_atual.captured_at
                    if resultado_atual is not None else None)
                mostrar = (
                    deteccao_atual
                    if (
                        resultado_atual is not None
                        and deteccao_atual is not None
                        and idade_resultado <= cfg.BALL_FRAME_STALE_S
                    )
                    else None
                )
                desempenho = (
                    f"cam {_taxa(amostras_captura):.1f} | "
                    f"vis {len(instantes_deteccao)} | "
                    f"{(metricas.processing_s * 1000.0) if metricas else 0:.0f}ms"
                    f" | {marcadores.resumo() if marcadores else ''}"
                )
                anotado = overlay_resgate.anotar(
                    frame_atual,
                    detection=mostrar,
                    marcadores=marcadores_atuais,
                    estado=comando.state,
                    detalhe=comando.detail,
                    motores_ativos=args.drive,
                    desempenho=desempenho,
                    guard=getattr(detector, "guard", None),
                )
                cv2.imshow(JANELA, anotado)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break

            if comando.terminal:
                transicao_para_coleta = (
                    args.drive
                    and comando.state == BallApproachController.NEAR
                    and coleta.started
                )
                if transicao_para_coleta:
                    time.sleep(TICK_S)
                    continue
                print(
                    f"[resgate] estado terminal {comando.state}; "
                    "motores parados")
                if not args.debug or args.drive:
                    break

            time.sleep(TICK_S)

    except RuntimeError as err:
        print(f"[resgate] ERRO: {err}")
    except KeyboardInterrupt:
        print("\n[resgate] Ctrl-C")
    finally:
        # PARAR vem antes de encerrar worker e camera, inclusive em excecao.
        if arduino is not None:
            from controle.direcao import steer
            _melhor_esforco("parar os motores", steer)
            _melhor_esforco("cortar o Futaba", arduino.parar_futaba)
        if trabalhador is not None:
            _melhor_esforco(
                "encerrar o detector",
                lambda: trabalhador.close(
                    timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S))
        if captura is not None:
            _melhor_esforco(
                "encerrar a captura",
                lambda: captura.close(
                    timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S))
        elif fonte is not None:
            _melhor_esforco("fechar a fonte", fonte.close)
        if arduino is not None:
            _melhor_esforco("fechar o Arduino", arduino.close)
        if trava is not None:
            _melhor_esforco("liberar a trava", trava.release)
        _melhor_esforco("fechar a janela", cv2.destroyAllWindows)
        if marcadores is not None:
            print(
                "[resgate] marcadores confirmados: "
                f"verde={marcadores.confirmados['green']} "
                f"vermelho={marcadores.confirmados['red']}")
        print(
            "[resgate] encerrado com PARAR" if args.drive
            else "[resgate] encerrado; motores nunca foram habilitados")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
