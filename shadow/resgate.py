#!/usr/bin/env python3
"""Resgate — ESCOPO ATUAL: ver a vítima, chegar perto dela e ver os marcadores.

A coleta, o depósito e a saída da sala NÃO estão neste programa ainda. Os
módulos deles continuam no repositório (``controle/coleta_resgate.py``,
``controle/deposito_resgate.py``, ``controle/saida_resgate.py``) com a
sequência de garra e Futaba já calibrada preservada — eles voltam quando a
visão estiver confiável. Fazer o contrário seria empilhar lógica em cima de
uma percepção que ainda erra.

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
from visao import overlay_resgate  # noqa: E402
from visao.marcador_resgate import MarkerDetector, color_masks  # noqa: E402
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


class MarkerPair:
    """Verde e vermelho detectados no mesmo frame, para identificação.

    Nesta fase nenhum marcador comanda o robô — servem para você confirmar
    que os dois são reconhecidos. Quando o depósito voltar ao escopo, só o
    triângulo da cor da vítima presa poderá comandar.
    """

    def __init__(self):
        self.detectors = {
            "green": MarkerDetector("green"),
            "red": MarkerDetector("red"),
        }
        self.detections = {"green": None, "red": None}
        self.confirmados = {"green": False, "red": False}

    def update(self, frame, timestamp):
        # Uma conversão HSV só, reaproveitada pelos dois detectores.
        mascaras = color_masks(frame)
        for tipo, detector in self.detectors.items():
            deteccao = detector.detect(
                frame, timestamp=timestamp, masks=mascaras)
            self.detections[tipo] = deteccao
            if deteccao is not None and deteccao.confirmed:
                self.confirmados[tipo] = True
        return dict(self.detections)

    def resumo(self):
        partes = []
        for tipo in ("green", "red"):
            deteccao = self.detections[tipo]
            if deteccao is None:
                partes.append(f"{tipo}:-")
            else:
                partes.append(
                    f"{tipo}:{deteccao.confidence:.2f}"
                    f"{'*' if deteccao.confirmed else ''}")
        return " ".join(partes)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visao do resgate: encontra a vitima, aproxima e identifica os "
            "marcadores verde e vermelho"))
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
    busca = None
    sessao_hardware = args.video is None

    ultimo_estado = None
    ultimo_detalhe = None
    ultimo_log = 0.0
    ultimo_controle_ocioso = 0.0
    epoca_busca = None
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

            if trabalhador is not None and frame_novo and armado:
                trabalhador.submit(
                    frame_atual,
                    captured_at=pacote.captured_at,
                    source_sequence=pacote.sequence,
                )

            if marcadores is not None and frame_novo:
                marcadores_atuais = marcadores.update(
                    frame_atual, pacote.captured_at)

            resultado = None
            if trabalhador is not None:
                resultado = trabalhador.poll(sequencia_resultado)
                if not trabalhador.is_alive:
                    trabalhador.poll(sequencia_resultado)
                    raise RuntimeError(
                        "detector assincrono encerrou inesperadamente")

            agora = time.monotonic()
            comando_atualizado = False

            if not armado:
                restante = max(armado_em - agora, 0.0)
                comando = MotionCommand(
                    "ARMING",
                    detail=f"camera fluida; PARAR por mais {restante:.1f} s")
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
            if args.drive and arduino is not None and comando_atualizado:
                from controle.direcao import steer
                epoca_movimento = arduino.connection_epoch
                if steer(comando.angle, comando.speed) is False:
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
