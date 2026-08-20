#!/usr/bin/env python3
"""Resgate — encontra, coleta, seleciona e procura todas as vítimas.

Depois de cada coleta, a vítima prata é selecionada pela garra esquerda e a
preta pela direita. O robô volta à busca pulsada. Duas passagens separadas
pelo marcador verde sem uma coleta no meio encerram a procura. Então o robô
avança até o ultrassônico confirmar 7 cm do retângulo verde, executa o
depósito desse lado e então procura o vermelho. No vermelho repete a mesma
sequência física, abrindo a caçamba para o lado oposto antes de encerrar.

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
import math
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
    ContadorVerdeBusca,
    tempo_busca_resgate_esgotado,
)
from controle.deposito_cinza_resgate import (  # noqa: E402
    SequenciadorDepositoCinza,
)
from controle.parada_obstaculo import MonitorObstaculo  # noqa: E402
from controle.parede_vitima import (  # noqa: E402
    LIVRE,
    PAREDE_RETA,
    WallPickupAuthorization,
    WallProbeController,
    aplicar_acao_parede,
)
from controle.retangulo_verde_resgate import (  # noqa: E402
    ControladorRetanguloVerde,
)
from controle.saida_resgate import ExitPhaseController  # noqa: E402
from visao import overlay_resgate  # noqa: E402
from visao.confirmacao_saida_linha import (  # noqa: E402
    NAO_PRETA,
    PRETA,
)
from visao.faixa_saida import BlackExitGate  # noqa: E402
from visao.saida_yolo import (  # noqa: E402
    ExitModel,
    LatestFrameExitDetector,
    ModelGuidedExitDetector,
)
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
EXIT_INCOMPLETE = 3
EXIT_SEM_MODELO = 4
EXIT_ARDUINO_DESCONECTADO = 5

RETOMADA_FALHOU = "retomada_falhou"
LINHA_NAO_ENCONTRADA = "linha_nao_encontrada"


class ArduinoDisconnectedDuringRescue(RuntimeError):
    """Sinaliza ao supervisor que a unica retomada permitida e por reconexao."""


def runtime_error_exit_code(args, arduino, current_exit_code):
    """Converte uma falha serial tardia em retorno recuperavel da missao.

    Uma escrita pode descobrir que o cabo saiu antes do laco principal notar
    ``connected=False``. No modo gerenciado, esse caso precisa voltar ao
    percurso em vez de parecer uma falha generica do resgate.
    """
    if (
        getattr(args, "gerenciado_pela_missao", False)
        and arduino is not None
        and not arduino.connected
    ):
        return EXIT_ARDUINO_DESCONECTADO
    return current_exit_code


def _aguardar_desligamento_do_arduino(arduino, motivo):
    """Mantém o resgate parado até a placa ser desligada fisicamente.

    Uma exceção durante uma fase física não pode transformar-se em retorno ao
    segue-linha: ainda pode haver vítima presa ou depósito pendente. No modo
    gerenciado a própria queda da placa é a única autorização para o
    supervisor reiniciar a missão desde o percurso.
    """
    from controle.direcao import steer

    try:
        steer()
    except (OSError, RuntimeError):
        pass
    print(
        "[resgate] falha retida dentro do resgate; motores em PARAR. "
        f"{motivo}. Desligue e religue o Arduino para reiniciar a missao.")
    while arduino.connected:
        # PING é inofensivo para os motores e permite notar a retirada do USB
        # mesmo quando não há outro comando físico em andamento.
        arduino.ping()
        arduino.refresh(fail_closed=True)
        time.sleep(0.25)


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
    movimento_iniciado = False
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
            and (movimento_iniciado or passo.gripper_action is not None)
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
        elif passo.motor_action not in ("", "forward", "reverse"):
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

        if passo.motor_action in ("forward", "reverse"):
            if acao_direcao(passo.angle, passo.speed) is False:
                nome = "avanco" if passo.motor_action == "forward" else "re"
                return f"comando de {nome} nao foi enviado pela serial"
            movimento_iniciado = True
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


def _aguardar_reconexao_coleta(
    arduino,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Espera o Arduino voltar e deixa as quatro rodas realmente zeradas."""
    limite = relogio() + cfg.BALL_PICKUP_SERIAL_RECOVERY_CONNECT_TIMEOUT_S
    while True:
        arduino.refresh(fail_closed=True)
        if arduino.connected:
            epoca = arduino.connection_epoch
            if arduino.lado(0, 0) is not False and (
                arduino.connected and arduino.connection_epoch == epoca
            ):
                return epoca
        restante = limite - relogio()
        if restante <= 0:
            raise RuntimeError(
                "Arduino nao reconectou durante a recuperacao da coleta")
        dormir(min(cfg.BALL_PICKUP_SERIAL_RECOVERY_POLL_S, restante))


def _executar_pulso_futaba_recuperacao(
    arduino,
    epoca,
    potencia,
    tempo_ms,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Executa parte da subida e devolve o tempo que faltou se houver reset."""
    tempo_ms = max(int(round(tempo_ms)), 0)
    if tempo_ms == 0:
        return True, 0
    inicio = relogio()
    if arduino.futaba(potencia, tempo_ms) is False or (
        not arduino.connected or arduino.connection_epoch != epoca
    ):
        return False, tempo_ms

    fim = inicio + tempo_ms / 1000.0
    while True:
        restante_s = fim - relogio()
        if restante_s <= 0:
            return True, 0
        dormir(min(cfg.BALL_PICKUP_SERIAL_RECOVERY_POLL_S, restante_s))
        arduino.refresh(fail_closed=True)
        if not arduino.connected or arduino.connection_epoch != epoca:
            executado_ms = max(int(round((relogio() - inicio) * 1000.0)), 0)
            return False, max(tempo_ms - executado_ms, 0)


def _recuperar_coleta_apos_reinicio(
    coleta,
    arduino,
    tentativa,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Sobe o Futaba apos o reboot e rearma a mesma coleta em modo normal."""
    if coleta is None or coleta.target_kind not in ("silver", "black"):
        raise RuntimeError(
            "recuperacao recusada: cor da coleta anterior desconhecida")
    tentativa = int(tentativa)
    if not 1 <= tentativa <= cfg.BALL_PICKUP_SERIAL_RECOVERY_MAX_RETRIES:
        raise RuntimeError(
            "limite de recuperacoes seriais da coleta atingido")

    tipo = coleta.target_kind
    normal_ms, lento_ms = coleta.recovery_lift_profile(now=relogio())
    reinicios_internos = 0

    while normal_ms > 0 or lento_ms > 0:
        epoca = _aguardar_reconexao_coleta(
            arduino, relogio=relogio, dormir=dormir)

        if normal_ms > 0:
            terminou, normal_ms = _executar_pulso_futaba_recuperacao(
                arduino,
                epoca,
                cfg.BALL_PICKUP_LIFT_POWER,
                normal_ms,
                relogio=relogio,
                dormir=dormir,
            )
            if not terminou:
                reinicios_internos += 1
                if (
                    reinicios_internos
                    > cfg.BALL_PICKUP_SERIAL_RECOVERY_MAX_RETRIES
                ):
                    raise RuntimeError(
                        "Arduino reiniciou repetidamente durante a subida "
                        "de recuperacao")
                continue

        if lento_ms > 0:
            terminou, lento_ms = _executar_pulso_futaba_recuperacao(
                arduino,
                epoca,
                cfg.BALL_PICKUP_LIFT_SLOW_POWER,
                lento_ms,
                relogio=relogio,
                dormir=dormir,
            )
            if not terminou:
                reinicios_internos += 1
                if (
                    reinicios_internos
                    > cfg.BALL_PICKUP_SERIAL_RECOVERY_MAX_RETRIES
                ):
                    raise RuntimeError(
                        "Arduino reiniciou repetidamente durante a subida "
                        "lenta de recuperacao")
                continue

    epoca = _aguardar_reconexao_coleta(
        arduino, relogio=relogio, dormir=dormir)
    if arduino.parar_futaba() is False or (
        not arduino.connected or arduino.connection_epoch != epoca
    ):
        raise RuntimeError(
            "nao foi possivel parar o Futaba depois da recuperacao")

    nova_coleta = BallPickupSequencer(grippers_prepositioned=False)
    if not nova_coleta.start(tipo, wall_mode=False):
        raise RuntimeError("nao foi possivel reiniciar a coleta normal")
    return nova_coleta, epoca


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
            return "serial mudou durante o deposito final"
        if acao_direcao(passo.angle, passo.speed) is False:
            return abortar(
                "comando dos motores do deposito nao foi enviado")
        if serial_mudou():
            return abortar(
                "serial mudou durante o deposito final")

        if passo.bucket_delta is not None:
            if arduino.servo("CACAMBA", passo.bucket_delta) is False:
                return abortar(
                    "comando do servo da cacamba nao foi enviado")
            if serial_mudou():
                return abortar(
                    "serial mudou durante o deposito final")
    except Exception as err:                         # noqa: BLE001
        return abortar(f"falha ao comandar deposito: {err}")
    return None


def _preparar_deposito_final(marcador_destino, vitimas_resgatadas):
    """Cria a sequencia fisica do marcador e do lado correspondente."""
    if marcador_destino not in ("green", "red"):
        raise ValueError("marcador_destino deve ser green ou red")
    quantidade = max(int(vitimas_resgatadas), 0)
    cor_vitima = "prata" if marcador_destino == "green" else "preta"
    lado = "verde" if marcador_destino == "green" else "vermelho"
    sequenciador = SequenciadorDepositoCinza(marcador_destino)
    return (
        sequenciador,
        MotionCommand(
            sequenciador.INICIO,
            detail=(
                f"{quantidade} vitima(s) {cor_vitima} registrada(s); "
                f"iniciando giro e deposito {lado}"
            ),
        ),
    )


def _preparar_deposito_cinza(vitimas_prata_resgatadas):
    """Mantem a entrada antiga para o deposito prata no marcador verde."""
    return _preparar_deposito_final("green", vitimas_prata_resgatadas)


def _recuperar_rota_deposito(controlador, agora):
    """Recomeça a busca do marcador sem abandonar a sala de resgate.

    Perder o retangulo durante a aproximação não significa que o depósito
    acabou. A nova instância descarta votos, frames e timeout do marcador
    antigo, mas a contagem de vítimas continua no laço principal do resgate.
    """
    marcador = controlador.target_kind
    nome = "verde" if marcador == "green" else "vermelho"
    novo_controlador = ControladorRetanguloVerde(
        start_time=agora,
        target_kind=marcador,
    )
    return novo_controlador, MotionCommand(
        f"{marcador.upper()}_ROUTE_RECOVERY",
        detail=(
            f"rota do marcador {nome} perdeu a confirmacao; "
            "permanecendo no resgate e procurando-o novamente"
        ),
    )


def _proximo_marcador_deposito(marcador_atual):
    """Depois do verde vem o vermelho; depois do vermelho a rota termina."""
    if marcador_atual == "green":
        return "red"
    if marcador_atual == "red":
        return None
    raise ValueError("marcador_atual deve ser green ou red")


def _deposito_vermelho_libera_saida(deposito, passo):
    """A saída só pode iniciar após o ciclo físico completo no vermelho."""
    return bool(
        deposito is not None
        and passo is not None
        and deposito.marcador_destino == "red"
        and passo.state == deposito.CONCLUIDO
        and passo.terminal
        and deposito.deposito_fisico_comandado
    )


def _saida_concluida_libera_missao(args, codigo_saida, comando):
    """Distingue o único terminal normal dos terminais recuperáveis/fatais."""
    return bool(
        args.drive
        and codigo_saida == EXIT_OK
        and comando.state == "EXIT_BLACK_CONFIRMED"
        and comando.terminal
    )


def _iniciar_busca_segura(arduino, start_time=None, accepts_kind=None):
    """Cria uma busca somente depois de substituir movimento antigo por PARAR."""
    if arduino is not None and arduino.parar() is False:
        raise RuntimeError(
            "nao foi possivel parar antes de iniciar a busca de bolinhas")
    return make_search_controller(
        start_time=start_time,
        accepts_kind=accepts_kind,
    )


def _validar_comando_da_busca(comando):
    """Impede que um estado de busca envie frente, ré ou rodas livres."""
    angulos_permitidos = (190, cfg.BALL_SEARCH_TANK_ANGLE)
    if (
        comando.wheel_speeds is not None
        or comando.angle not in angulos_permitidos
    ):
        raise RuntimeError(
            "comando proibido durante a busca de bolinhas: "
            f"estado={comando.state}, angulo={comando.angle}, "
            f"rodas={comando.wheel_speeds}")


def _validar_inicio_coleta(
    comando,
    coleta,
    arduino,
    parada_enviada,
    epoca_movimento,
):
    """Valida o ponto de coleta sem mexer no Futaba ou nas garras."""
    if (
        comando.state != BallApproachController.NEAR
        or coleta.started
    ):
        return None
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
    return comando.target_kind


def _armar_coleta_confirmada(
    comando,
    coleta,
    arduino,
    parada_enviada,
    epoca_movimento,
    modo_parede=False,
):
    """Só arma a coleta após confirmação visual e PARAR serial estável."""
    target_kind = _validar_inicio_coleta(
        comando,
        coleta,
        arduino,
        parada_enviada,
        epoca_movimento,
    )
    if target_kind is None:
        return False
    if not coleta.start(target_kind, wall_mode=modo_parede):
        raise RuntimeError("coleta recusada: sequencia ja iniciada")
    return True


def _armar_coleta_parede_direta(
    passo_parede,
    coleta,
    arduino,
    parada_enviada,
    epoca_movimento,
    epoca_parede,
    deteccao,
    frame_shape,
    assinatura,
    agora,
):
    """Preserva o yaw confirmado e inicia a coleta especial sem reaproximar."""
    if (
        passo_parede is None
        or not passo_parede.terminal
        or passo_parede.result != PAREDE_RETA
        or passo_parede.motor_action != "stop"
    ):
        raise RuntimeError(
            "coleta de parede recusada: teste nao terminou em PARAR")
    if (
        parada_enviada is not True
        or epoca_movimento is None
        or arduino is None
        or not arduino.connected
        or arduino.connection_epoch != epoca_movimento
        or epoca_parede != epoca_movimento
    ):
        raise RuntimeError(
            "coleta de parede recusada: PARAR nao teve serial estavel")
    if passo_parede.target_kind not in ("silver", "black"):
        raise RuntimeError(
            "coleta de parede recusada: cor da vitima nao confirmada")
    if (
        deteccao is None
        or not deteccao.confirmed
        or deteccao.kind != passo_parede.target_kind
        or deteccao.truncated
        or agora - deteccao.timestamp > cfg.BALL_FRAME_STALE_S
        or assinatura is None
        or not assinatura.matches(deteccao, frame_shape)
        or not assinatura.matches_pickup_depth(deteccao, frame_shape)
        or abs(deteccao.horizontal_error(frame_shape[1]))
        > cfg.BALL_WALL_ALIGN_CENTER_DEADBAND
    ):
        raise RuntimeError(
            "coleta de parede recusada: mesma vitima central nao foi "
            "reconfirmada")
    if coleta.started:
        raise RuntimeError("coleta de parede recusada: sequencia ja iniciada")
    if not coleta.start(passo_parede.target_kind, wall_mode=True):
        raise RuntimeError("coleta de parede recusada pelo sequenciador")
    return True


def _deve_reiniciar_busca_por_alvo_perdido(
    busca,
    controlador,
    coleta,
    comando,
    autorizacao_parede=None,
):
    """Nao abandona a reaproximacao enquanto seus frames sao confirmados."""
    return (
        busca is None
        and controlador is not None
        and not coleta.started
        and autorizacao_parede is None
        and comando.state == BallApproachController.WAIT_TARGET
    )


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


def _mover_saida_por_tempo(
    arduino,
    acao_direcao,
    angulo,
    velocidade,
    duracao,
    epoca_serial,
):
    """Executa um trecho curto sem deixar o watchdog ou a serial vencerem."""
    enviado = acao_direcao(angulo, velocidade)
    if enviado is False:
        raise RuntimeError("comando da verificacao de saida nao foi enviado")
    prazo = time.monotonic() + max(float(duracao), 0.0)
    while time.monotonic() < prazo:
        arduino.refresh(fail_closed=True)
        if (
            not arduino.connected
            or arduino.connection_epoch != epoca_serial
        ):
            acao_direcao()
            raise RuntimeError(
                "serial mudou durante a verificacao da faixa de saida")
        time.sleep(min(0.02, max(prazo - time.monotonic(), 0.0)))


def _avancar_entrada_da_missao(args, arduino, acao_direcao):
    """Atravessa a faixa prata antes da primeira busca por vitimas."""
    if not (
        args.drive
        and args.gerenciado_pela_missao
        and arduino is not None
    ):
        return False

    epoca_serial = arduino.connection_epoch
    print(
        "[resgate] entrada da missao: avancando reto por "
        f"{cfg.MISSION_ENTRY_FORWARD_S:.1f} s antes de procurar vitimas")
    try:
        _mover_saida_por_tempo(
            arduino,
            acao_direcao,
            0,
            cfg.MISSION_ENTRY_FORWARD_SPEED,
            cfg.MISSION_ENTRY_FORWARD_S,
            epoca_serial,
        )
    finally:
        acao_direcao()
    print(
        "[resgate] entrada concluida; iniciando a busca giratoria "
        "das vitimas")
    return True


def _posicionar_garras_inicio_resgate(args, arduino):
    """Posiciona as garras antes da primeira busca por vitimas."""
    if not args.drive or arduino is None:
        return False
    epoca_serial = arduino.connection_epoch
    if arduino.garras(
        cfg.BALL_PICKUP_INITIAL_LEFT_DELTA,
        cfg.BALL_PICKUP_INITIAL_RIGHT_DELTA,
    ) is False or (
        not arduino.connected
        or arduino.connection_epoch != epoca_serial
    ):
        raise RuntimeError(
            "nao foi possivel posicionar as garras no inicio do resgate")
    print("[resgate] garras iniciais: esquerda=-20, direita=+20 graus")
    return True


def _confirmar_saida_com_camera_linha(
    arduino,
    debug=False,
):
    """Confirma cor e terceira linha mantendo camera/serial na mesma sessao."""
    from controle.direcao import steer
    from controle.saida_linha import confirmar_saida_com_camera_linha
    from visao.captura import LineCamera

    return confirmar_saida_com_camera_linha(
        arduino,
        steer,
        LineCamera,
        debug=debug,
        janela_debug=JANELA,
    )


def _executar_saida_parede(arduino, camera_index, debug=False):
    """Compatibilidade: delega a manobra direta pos-vermelho."""
    del camera_index, debug
    from controle.saida_parede_resgate import executar_alinhamento_parede

    return executar_alinhamento_parede(arduino, intervalo_s=TICK_S)


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
    parser.add_argument(
        "--gerenciado-pela-missao", action="store_true",
        help=argparse.SUPPRESS)
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


def novo_portao_saida(modelo_saida=None):
    """Cria o gate da saída somente quando essa fase realmente começa."""
    if not cfg.EXIT_MODEL_ENABLED:
        return BlackExitGate(), modelo_saida
    if modelo_saida is None:
        modelo_saida = ExitModel().load()
        print(
            "[saida] modelo carregado: "
            f"{modelo_saida.active_path.name} "
            f"({modelo_saida.active_backend}, "
            f"{modelo_saida.input_size}x{modelo_saida.input_size})")
    detector = ModelGuidedExitDetector(modelo_saida)
    return BlackExitGate(detector=detector), modelo_saida


def main(args=None):
    if args is None:
        args = parse_args()
    fonte = None
    arduino = None
    trava = None
    captura = None
    trabalhador = None
    trabalhador_saida = None
    marcadores = None
    controlador = None
    controlador_verde = None
    monitor_chegada_verde = None
    deposito_cinza = None
    controlador_saida = None
    portao_saida = None
    modelo_saida = None
    deteccao_saida = None
    faltas_saida = 0
    verificador_parede = None
    busca = None
    coleta = None
    contador_verde = ContadorVerdeBusca()
    sessao_hardware = args.video is None

    ultimo_estado = None
    ultimo_detalhe = None
    ultimo_log = 0.0
    ultimo_log_modelo_saida = 0.0
    ultimo_controle_ocioso = 0.0
    proxima_atualizacao_ultrassom_verde = 0.0
    epoca_busca = None
    epoca_coleta = None
    epoca_parede = None
    epoca_verde = None
    epoca_deposito_cinza = None
    epoca_saida = None
    inicio_saida = None
    varreduras_sem_vitima = 0
    vitimas_resgatadas = 0
    vitimas_prata_resgatadas = 0
    vitimas_pretas_resgatadas = 0
    tentativas_recuperacao_coleta = 0
    coleta_apos_teste_parede = None
    amostras_captura = deque(maxlen=60)
    instantes_deteccao = deque(maxlen=30)
    codigo_saida = EXIT_INCOMPLETE if args.drive else EXIT_OK

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
            if args.gerenciado_pela_missao:
                # Ao cair a placa, a tentativa inteira volta ao percurso; nao
                # e seguro retomar coleta, deposito ou saida numa serial nova.
                arduino.travar_sessao()
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

        _posicionar_garras_inicio_resgate(args, arduino)
        entrada_da_missao_concluida = _avancar_entrada_da_missao(
            args,
            arduino,
            steer if sessao_hardware else None,
        )
        inicio = time.monotonic()
        armado_em = (
            inicio
            if entrada_da_missao_concluida or not args.drive
            else inicio + cfg.RESCUE_ARM_DELAY_S
        )
        fim_busca_resgate_em = (
            armado_em + cfg.RESCUE_SEARCH_DURATION_S
        )
        controlador = (
            None if args.drive
            else BallApproachController(start_time=armado_em))
        busca = (
            _iniciar_busca_segura(arduino, start_time=armado_em)
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
        sequencia_resultado_saida = 0
        frame_atual = None
        resultado_atual = None
        deteccao_atual = None
        marcadores_atuais = {}
        metricas = None

        def recuperar_coleta_serial(motivo):
            nonlocal coleta
            nonlocal epoca_coleta
            nonlocal tentativas_recuperacao_coleta

            tentativas_recuperacao_coleta += 1
            print(
                "[coleta] Arduino reiniciou; recuperacao "
                f"{tentativas_recuperacao_coleta}/"
                f"{cfg.BALL_PICKUP_SERIAL_RECOVERY_MAX_RETRIES}: {motivo}")
            coleta, epoca_coleta = _recuperar_coleta_apos_reinicio(
                coleta,
                arduino,
                tentativas_recuperacao_coleta,
            )
            if trabalhador is not None:
                trabalhador.reset_tracking()
            portao.reset()
            print(
                "[coleta] Futaba levantado e serial estavel; "
                f"reiniciando coleta normal da vitima {coleta.target_kind}")
            return MotionCommand(
                "PICKUP_SERIAL_RECOVERED",
                detail=(
                    "Arduino reconectado; Futaba levantado; repetindo a "
                    "coleta normal"
                ),
                target_kind=coleta.target_kind,
            )

        if args.drive:
            print(
                "[resgate] MOVIMENTO AUTORIZADO. Mantendo PARAR por "
                f"{cfg.RESCUE_ARM_DELAY_S:.0f} s.")
        else:
            print(
                "[resgate] modo de visao: motores desativados. "
                "Use --drive so depois de validar a visao.")

        while True:
            if (
                args.drive
                and sessao_hardware
                and arduino is not None
                and not arduino.connected
            ):
                raise ArduinoDisconnectedDuringRescue(
                    "Arduino desconectado durante o resgate; "
                    "reinicie a missao pelo percurso")
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
                coleta_apos_teste_parede is not None
                and agora > coleta_apos_teste_parede.expires_at
            ):
                print(
                    "[resgate] autorizacao do teste de parede expirou; "
                    "voltando a procurar a vitima com seguranca")
                coleta_apos_teste_parede = None

            if (
                trabalhador is not None
                and frame_novo
                and armado
                and not coleta.started
                and controlador_verde is None
                and deposito_cinza is None
                and controlador_saida is None
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
                and controlador_saida is None
            ):
                marcadores_atuais = marcadores.update(
                    frame_atual, pacote.captured_at)
                if busca is not None:
                    somou_verde = contador_verde.observar(
                        marcadores_atuais.get("green"),
                        permitido=busca.frame_allowed(pacote.captured_at),
                        varredura=varreduras_sem_vitima,
                    )
                    if somou_verde:
                        print(
                            "[resgate] passagem verde "
                            f"{contador_verde.quantidade}/"
                            f"{contador_verde.necessario}")
                        if args.drive and contador_verde.completo:
                            # A segunda passagem foi confirmada com o robo
                            # parado em SEARCH_OBSERVE e nenhuma bolinha foi
                            # travada pela busca. E seguro encerrar antes dos
                            # 90 s e comecar a rota verde imediatamente.
                            if trabalhador is not None:
                                trabalhador.reset_tracking()
                            portao.reset()
                            busca = None
                            controlador = None
                            verificador_parede = None
                            coleta_apos_teste_parede = None
                            controlador_verde = ControladorRetanguloVerde(
                                start_time=agora,
                            )
                            monitor_chegada_verde = None
                            proxima_atualizacao_ultrassom_verde = 0.0
                            epoca_busca = None
                            epoca_parede = None
                            epoca_verde = (
                                arduino.connection_epoch
                                if arduino is not None else None
                            )
                            resultado_atual = None
                            deteccao_atual = None
                            ultimo_controle_ocioso = 0.0
                            print(
                                "[resgate] GREEN_ROUTE_START: segundo verde "
                                "confirmado sem bolinha; alinhando pela "
                                "camera")

            if (
                controlador_saida is not None
                and portao_saida is not None
                and frame_novo
            ):
                previa_saida = (
                    controlador_saida.state
                    == controlador_saida.SEARCH_ROTATE
                )
                permitido_saida = (
                    previa_saida
                    or controlador_saida.frame_allowed(pacote.captured_at)
                )
                if trabalhador_saida is not None and permitido_saida:
                    # O worker substitui qualquer pendencia pelo frame mais
                    # novo. A camera/debug nunca espera os ~200 ms do NCNN.
                    trabalhador_saida.submit(
                        frame_atual,
                        captured_at=pacote.captured_at,
                        source_sequence=pacote.sequence,
                        preview=previa_saida,
                    )
                elif trabalhador_saida is None:
                    # Fallback classico quando EXIT_MODEL_ENABLED=False.
                    if previa_saida:
                        deteccao_saida = portao_saida.preview(
                            frame_atual,
                            timestamp=pacote.captured_at,
                        )
                    elif permitido_saida:
                        confirmada_saida, candidata_saida = portao_saida.update(
                            frame_atual,
                            timestamp=pacote.captured_at,
                            now=agora,
                        )
                        if confirmada_saida:
                            deteccao_saida = candidata_saida
                            faltas_saida = 0
                        else:
                            deteccao_saida = None

            resultado_saida = None
            if trabalhador_saida is not None:
                resultado_saida = trabalhador_saida.poll(
                    sequencia_resultado_saida)
                if not trabalhador_saida.is_alive:
                    trabalhador_saida.poll(sequencia_resultado_saida)
                    raise RuntimeError(
                        "detector assincrono da saida encerrou inesperadamente")

            if (
                resultado_saida is not None
                and controlador_saida is not None
                and portao_saida is not None
            ):
                sequencia_resultado_saida = resultado_saida.sequence
                if resultado_saida.preview:
                    if (
                        controlador_saida.state
                        == controlador_saida.SEARCH_ROTATE
                    ):
                        deteccao_saida = portao_saida.preview_detection(
                            resultado_saida.detection)
                elif controlador_saida.frame_allowed(
                    resultado_saida.captured_at
                ):
                    confirmada_saida, candidata_saida = (
                        portao_saida.update_detection(
                            resultado_saida.detection,
                            resultado_saida.frame_shape,
                            timestamp=resultado_saida.captured_at,
                            now=agora,
                        )
                    )
                    if portao_saida.just_locked and candidata_saida is not None:
                        print(
                            "[saida] alvo TRAVADO pelo modelo: "
                            f"confianca={candidata_saida.confidence:.0%}; "
                            "direcionando para o centro")
                    if confirmada_saida:
                        deteccao_saida = candidata_saida
                        faltas_saida = 0
                    else:
                        deteccao_saida = None

            if (
                args.debug
                and modelo_saida is not None
                and controlador_saida is not None
                and agora - ultimo_log_modelo_saida >= 0.50
            ):
                aceito = (
                    modelo_saida.last_raw_confidence
                    >= modelo_saida.min_confidence)
                print(
                    "[saida/modelo] "
                    f"raw={modelo_saida.last_raw_confidence:.0%} "
                    f"limiar={modelo_saida.min_confidence:.0%} "
                    f"aceito={'sim' if aceito else 'nao'} "
                    f"lock={'sim' if portao_saida.track_locked else 'nao'} "
                    f"{modelo_saida.last_inference_ms:.0f}ms async")
                ultimo_log_modelo_saida = agora

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
            passo_parede = None
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
            elif (
                args.drive
                and not coleta.started
                and (
                    busca is not None
                    or controlador is not None
                    or verificador_parede is not None
                )
                and tempo_busca_resgate_esgotado(
                    armado_em, agora
                )
            ):
                # O giro pode estar ativo neste instante. Primeiro escreve
                # PARAR; a troca para a rota verde acontece somente depois do
                # acknowledge da serial, mais abaixo.
                comando = MotionCommand(
                    "RESCUE_SEARCH_TIME_STOP",
                    detail=(
                        f"{cfg.RESCUE_SEARCH_DURATION_S:.0f} s de busca "
                        "concluidos; parando para iniciar os depositos"
                    ),
                )
                comando_atualizado = True
            elif verificador_parede is not None:
                # Durante este teste as rodas obedecem somente ao pequeno
                # deslocamento lateral do verificador. A deteccao continua
                # rodando para comprovar que a esfera saiu do eixo central do
                # ultrassonico antes de interpretar o eco como parede.
                if resultado is not None:
                    sequencia_resultado = resultado.sequence
                    metricas = resultado
                    instantes_deteccao.append(resultado.completed_at)
                    if agora - resultado.captured_at <= cfg.BALL_FRAME_STALE_S:
                        resultado_atual = resultado
                        deteccao_atual = resultado.detection
                    else:
                        resultado_atual = None
                        deteccao_atual = None

                forma = (
                    frame_atual.shape if frame_atual is not None
                    else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                          cfg.RESCUE_CAMERA_MAX_WIDTH, 3))
                passo_parede = verificador_parede.update(
                    arduino,
                    detection=deteccao_atual,
                    frame_shape=forma,
                    now=agora,
                )
                comando = passo_parede.motion_command()
                comando_atualizado = True
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

                serial_coleta_mudou = False
                if arduino is not None and epoca_coleta is not None:
                    arduino.refresh(fail_closed=True)
                    serial_coleta_mudou = (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_coleta
                    )

                if serial_coleta_mudou:
                    comando = recuperar_coleta_serial(
                        "conexao mudou antes do proximo passo")
                    passo_coleta = None
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
            elif controlador_saida is not None:
                # Esta fase so existe depois do deposito vermelho. A camera
                # de resgate procura e centraliza a soleira; a confirmacao de
                # preto contra prata fica para a camera de linha, mais perto.
                if resultado is not None:
                    sequencia_resultado = resultado.sequence
                    metricas = resultado
                    instantes_deteccao.append(resultado.completed_at)
                resultado_atual = None
                deteccao_atual = None

                if (
                    frame_novo
                    or agora - ultimo_controle_ocioso
                    >= INTERVALO_CONTROLE_OCIOSO_S
                ):
                    forma = (
                        frame_atual.shape if frame_atual is not None
                        else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                              cfg.RESCUE_CAMERA_MAX_WIDTH, 3)
                    )
                    comando = controlador_saida.update(
                        deteccao_saida,
                        forma,
                        mapper=None,
                        now=agora,
                    )
                    comando_atualizado = True
                    ultimo_controle_ocioso = agora
            elif controlador_verde is not None:
                # A camera primeiro confirma, centraliza e se aproxima do
                # marcador escolhido. Somente depois da parada visual o
                # ultrassonico e habilitado para os centimetros finais.
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
                marcador_destino = controlador_verde.target_kind
                deteccao_destino = marcadores_atuais.get(marcador_destino)
                if (
                    distancia_chegada_verde_mm is not None
                    or medicao_ultrassom_verde_atualizada
                ):
                    comando = controlador_verde.update(
                        deteccao_destino,
                        forma,
                        now=agora,
                        **dados_ultrassom,
                    )
                    comando_atualizado = True
                    ultimo_controle_ocioso = agora
                elif frame_novo:
                    comando = controlador_verde.update(
                        deteccao_destino,
                        forma,
                        mascara_verde=(
                            marcadores.mascaras.get(marcador_destino)
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
            if args.drive and _deve_reiniciar_busca_por_alvo_perdido(
                busca,
                controlador,
                coleta,
                comando,
                autorizacao_parede=coleta_apos_teste_parede,
            ):
                # O resultado do teste vale somente para a mesma aproximacao.
                # Se a vitima sumiu, outra esfera da mesma cor nao pode herdar
                # por engano a autorizacao especial de parede.
                coleta_apos_teste_parede = None
                if trabalhador is not None:
                    trabalhador.reset_tracking()
                portao.reset()
                busca = _iniciar_busca_segura(arduino, start_time=agora)
                controlador = None
                resultado_atual = None
                deteccao_atual = None
                epoca_busca = None
                ultimo_controle_ocioso = agora
                comando = busca.update(None, now=agora)
                comando_atualizado = True

            epoca_movimento = None
            movimento_enviado = None
            if args.drive and arduino is not None and comando_atualizado:
                from controle.direcao import steer
                if busca is not None:
                    try:
                        _validar_comando_da_busca(comando)
                    except RuntimeError:
                        steer()
                        raise
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
                        if (
                            passo_coleta.state
                            == BallPickupSequencer.WALL_PAUSE_PENDING
                            and passo_coleta.motor_action == "stop"
                        ):
                            coleta.mark_wall_pause_started(now=concluido_em)
                        if (
                            passo_coleta.state
                            == BallPickupSequencer.WALL_POST_REVERSE_PENDING
                            and passo_coleta.motor_action == "stop"
                        ):
                            coleta.mark_post_reverse_pause_started(
                                now=concluido_em)
                        if passo_coleta.motor_action == "forward":
                            coleta.mark_forward_started(now=concluido_em)
                        if passo_coleta.motor_action == "reverse":
                            coleta.mark_reverse_started(now=concluido_em)
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
                        serial_reiniciou = (
                            not arduino.connected
                            or epoca_coleta is None
                            or arduino.connection_epoch != epoca_coleta
                        )
                        if serial_reiniciou:
                            comando = recuperar_coleta_serial(erro_coleta)
                            passo_coleta = None
                        else:
                            passo_coleta = coleta.fail(erro_coleta)
                            comando = passo_coleta.motion_command()
                            _aplicar_acoes_coleta(
                                passo_coleta,
                                arduino,
                                steer,
                            )
                elif passo_parede is not None:
                    epoca_movimento = arduino.connection_epoch
                    erro_parede = aplicar_acao_parede(
                        passo_parede,
                        arduino,
                        epoca_serial_esperada=epoca_parede,
                    )
                    movimento_enviado = erro_parede is None
                    if erro_parede is not None:
                        passo_parede = verificador_parede.fail(
                            erro_parede)
                        comando = passo_parede.motion_command()
                        aplicar_acao_parede(passo_parede, arduino)
                    elif (
                        passo_parede.motor_action
                        and not passo_parede.terminal
                    ):
                        verificador_parede.notify_command_written(
                            passo_parede.state,
                            now=time.monotonic(),
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
                    if comando.wheel_speeds is not None:
                        movimento_enviado = arduino.rodas(
                            *comando.wheel_speeds)
                    else:
                        movimento_enviado = steer(
                            comando.angle, comando.speed)
                    if movimento_enviado is False:
                        raise RuntimeError(
                            "comando de movimento nao foi enviado pela serial")
                    concluido_em = time.monotonic()
                    if comando.state == "RESCUE_SEARCH_TIME_STOP":
                        # O limite venceu antes de iniciar uma coleta, ou
                        # durante a aproximacao. A parada ja foi escrita; a
                        # partir daqui nenhum alvo de vitima pode retomar as
                        # rodas antes da rota de deposito.
                        if trabalhador is not None:
                            trabalhador.reset_tracking()
                        portao.reset()
                        if marcadores is not None:
                            marcadores.reset()
                        marcadores_atuais = {}
                        busca = None
                        controlador = None
                        verificador_parede = None
                        coleta_apos_teste_parede = None
                        controlador_verde = ControladorRetanguloVerde(
                            start_time=concluido_em,
                        )
                        monitor_chegada_verde = None
                        proxima_atualizacao_ultrassom_verde = 0.0
                        epoca_busca = None
                        epoca_parede = None
                        epoca_verde = arduino.connection_epoch
                        resultado_atual = None
                        deteccao_atual = None
                        ultimo_controle_ocioso = 0.0
                        comando = MotionCommand(
                            "GREEN_ROUTE_START",
                            detail=(
                                f"{cfg.RESCUE_SEARCH_DURATION_S:.0f} s de "
                                "busca concluidos; procurando e alinhando "
                                "ao retangulo verde",
                            ),
                        )
                        comando_atualizado = False
                    elif busca is not None:
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
                    elif controlador_saida is not None:
                        if controlador_saida.consume_tracking_reset():
                            if portao_saida is not None:
                                portao_saida.reset(now=concluido_em)
                            deteccao_saida = None
                            faltas_saida = 0
                        if comando.state == controlador_saida.SEARCH_START:
                            epoca_saida = arduino.connection_epoch
                        if controlador_saida.notify_command_written(
                            comando.state, concluido_em
                        ):
                            if portao_saida is not None:
                                portao_saida.reset(now=concluido_em)
                            deteccao_saida = None
                            faltas_saida = 0

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
                    verificador_parede is not None
                    and epoca_parede is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_parede
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante o teste de parede; "
                        "motores e garras parados")
                if (
                    coleta.started
                    and epoca_coleta is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_coleta
                    )
                ):
                    comando = recuperar_coleta_serial(
                        "conexao mudou depois do comando da coleta")
                    passo_coleta = None
                    epoca_movimento = None
                    movimento_enviado = None
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
                        "serial mudou durante o deposito final; "
                        "motores parados e cacamba nao comandada")
                if (
                    controlador_saida is not None
                    and epoca_saida is not None
                    and (
                        not arduino.connected
                        or arduino.connection_epoch != epoca_saida
                    )
                ):
                    raise RuntimeError(
                        "serial mudou durante a busca da faixa de saida; "
                        "motores parados")

            if (
                args.drive
                and verificador_parede is not None
                and passo_parede is not None
                and passo_parede.terminal
                and passo_parede.result == PAREDE_RETA
            ):
                _armar_coleta_parede_direta(
                    passo_parede,
                    coleta,
                    arduino,
                    movimento_enviado,
                    epoca_movimento,
                    epoca_parede,
                    deteccao_atual,
                    (
                        frame_atual.shape if frame_atual is not None
                        else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                              cfg.RESCUE_CAMERA_MAX_WIDTH, 3)
                    ),
                    verificador_parede.target_signature,
                    agora,
                )
                epoca_coleta = arduino.connection_epoch
                print(
                    "[resgate] parede reta e mesma vitima confirmadas; "
                    "iniciando coleta especial sem desfazer o yaw")
                verificador_parede = None
                passo_parede = None
                epoca_parede = None
                coleta_apos_teste_parede = None
                busca = None
                controlador = None
                resultado_atual = None
                deteccao_atual = None
                portao.reset()
                if trabalhador is not None:
                    trabalhador.reset_tracking()
                ultimo_controle_ocioso = 0.0
                comando = MotionCommand(
                    "WALL_PICKUP_ARMED",
                    detail=(
                        "PARAR confirmado; coleta de parede iniciada no "
                        "mesmo yaw"),
                    target_kind=coleta.target_kind,
                )

            elif (
                args.drive
                and verificador_parede is not None
                and passo_parede is not None
                and passo_parede.terminal
                and passo_parede.result == LIVRE
            ):
                coleta_apos_teste_parede = WallPickupAuthorization(
                    target_kind=passo_parede.target_kind,
                    wall_mode=False,
                    expires_at=(
                        agora + cfg.BALL_WALL_REAPPROACH_AUTH_S),
                    signature=verificador_parede.target_signature,
                )
                print(
                    "[resgate] caminho livre confirmado; "
                    "retomando a aproximacao visual antes da coleta")
                verificador_parede = None
                passo_parede = None
                epoca_parede = None
                # Nao reinicia o tracker aqui: ele e a identidade temporal da
                # esfera testada. O portao abaixo ainda exige frames novos.
                portao.reset()
                busca = None
                controlador = BallApproachController(start_time=agora)
                resultado_atual = None
                deteccao_atual = None
                ultimo_controle_ocioso = agora
                comando = MotionCommand(
                    "WALL_REAPPROACH",
                    detail=(
                        "caminho livre; procurando novamente a vitima "
                        "no ponto original"
                    ),
                )
                comando_atualizado = False

            if (
                args.drive
                and deposito_cinza is not None
                and passo_deposito_cinza is not None
                and passo_deposito_cinza.state == deposito_cinza.CONCLUIDO
                and passo_deposito_cinza.terminal
            ):
                marcador_concluido = deposito_cinza.marcador_destino
                if not deposito_cinza.deposito_fisico_comandado:
                    # Falha fechada: a sequencia nunca pode liberar a rota
                    # seguinte (principalmente a saida) sem a abertura e a
                    # restauracao da caçamba terem sido aceitas pela serial.
                    raise RuntimeError(
                        "sequencia de deposito terminou sem confirmar os "
                        "comandos da cacamba; busca da saida bloqueada")
                proximo_marcador = _proximo_marcador_deposito(
                    marcador_concluido)
                if proximo_marcador is not None:
                    print(
                        "[resgate] deposito verde concluido; "
                        "iniciando procura pelo marcador vermelho")
                    deposito_cinza = None
                    epoca_deposito_cinza = None
                    # A rota de deposito e exclusiva: depois de entregar no
                    # verde, uma bolinha que apareca no quadro nao pode
                    # reabrir busca/coleta. Ate o vermelho ser depositado, o
                    # unico alvo que comanda as rodas e o marcador vermelho.
                    busca = None
                    controlador = None
                    verificador_parede = None
                    coleta_apos_teste_parede = None
                    resultado_atual = None
                    deteccao_atual = None
                    if trabalhador is not None:
                        trabalhador.reset_tracking()
                    portao.reset()
                    if marcadores is None:
                        comando = MotionCommand(
                            "RED_FINAL_FAULT",
                            detail=(
                                "deposito verde terminou, mas os marcadores "
                                "estao desativados; robo parado"),
                            terminal=True,
                        )
                    else:
                        marcadores.reset()
                        marcadores_atuais = {}
                        controlador_verde = ControladorRetanguloVerde(
                            start_time=agora,
                            target_kind=proximo_marcador,
                        )
                        monitor_chegada_verde = None
                        proxima_atualizacao_ultrassom_verde = 0.0
                        epoca_verde = (
                            arduino.connection_epoch
                            if arduino is not None else None
                        )
                        ultimo_controle_ocioso = 0.0
                        comando = MotionCommand(
                            "RED_ROUTE_START",
                            detail=(
                                "reto final do verde concluido; procurando "
                                "e alinhando ao retangulo vermelho"),
                        )
                        comando_atualizado = False
                else:
                    # O reto final do deposito vermelho terminou. Somente
                    # agora a faixa de saida passa a existir para a visao.
                    if not _deposito_vermelho_libera_saida(
                        deposito_cinza, passo_deposito_cinza
                    ):
                        raise RuntimeError(
                            "tentativa de iniciar a saida sem deposito "
                            "vermelho fisicamente concluido")
                    print(
                        "[resgate] deposito vermelho concluido; "
                        "procurando e alinhando com a faixa de saida")
                    deposito_cinza = None
                    epoca_deposito_cinza = None
                    busca = None
                    controlador = None
                    controlador_verde = None
                    monitor_chegada_verde = None
                    resultado_atual = None
                    deteccao_atual = None
                    # Não haverá mais vítimas nesta execução. Fechar o worker
                    # agora libera CPU antes de armar o modelo NCNN da saída.
                    if trabalhador is not None:
                        trabalhador.close(
                            timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
                        trabalhador = None
                    portao.reset()
                    if cfg.SAIDA_PAREDE_ATIVADA:
                        # A nova rota possui seu proprio ciclo de cameras.
                        # Encerrar a frontal assincrona antes de entrar nela
                        # impede duas Picameras abertas em paralelo.
                        if trabalhador_saida is not None:
                            trabalhador_saida.close(
                                timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
                            trabalhador_saida = None
                        if captura is not None:
                            fechou = captura.close(
                                timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
                            if not fechou:
                                raise RuntimeError(
                                    "camera frontal nao fechou antes da "
                                    "saida pela parede")
                            captura = None
                            fonte = None
                        print(
                            "[resgate] deposito vermelho concluido; "
                            "iniciando saida pela parede direita")
                        resultado_saida_parede = _executar_saida_parede(
                            arduino,
                            args.camera_index,
                            debug=args.debug,
                        )
                        controlador_saida = None
                        portao_saida = None
                        epoca_saida = None
                        inicio_saida = None
                        deteccao_saida = None
                        faltas_saida = 0
                        if resultado_saida_parede == "parede_frente_estavel":
                            codigo_saida = EXIT_INCOMPLETE
                            comando = MotionCommand(
                                "EXIT_WALL_FRONT_STABLE",
                                detail=(
                                    "giro, alinhamento, parede frontal e "
                                    "pivo traseiro concluidos; robo parado"),
                                terminal=True,
                            )
                        else:
                            codigo_saida = EXIT_INCOMPLETE
                            comando = MotionCommand(
                                "EXIT_WALL_FAILED",
                                detail=(
                                    "giro, alinhamento ou avanco ate a parede "
                                    "nao foi concluido com seguranca"),
                                terminal=True,
                            )
                        comando_atualizado = False
                    else:
                        inicio_saida = agora
                        controlador_saida = ExitPhaseController(
                            start_time=inicio_saida)
                        portao_saida, modelo_saida = novo_portao_saida(
                            modelo_saida)
                        trabalhador_saida = (
                            LatestFrameExitDetector(portao_saida.detector)
                            if modelo_saida is not None else None
                        )
                        sequencia_resultado_saida = 0
                        deteccao_saida = None
                        faltas_saida = 0
                        epoca_saida = arduino.connection_epoch
                        forma = (
                            frame_atual.shape if frame_atual is not None
                            else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                                  cfg.RESCUE_CAMERA_MAX_WIDTH, 3)
                        )
                        comando = controlador_saida.update(
                            None, forma, mapper=None, now=agora)
                        ultimo_controle_ocioso = agora
                        comando_atualizado = False

            if (
                args.drive
                and controlador_verde is not None
                and comando.state == controlador_verde.CONCLUIDO
                and comando.terminal
            ):
                marcador_chegada = controlador_verde.target_kind
                nome_marcador = (
                    "verde" if marcador_chegada == "green" else "vermelho")
                quantidade_destino = (
                    vitimas_prata_resgatadas
                    if marcador_chegada == "green"
                    else vitimas_pretas_resgatadas
                )
                print(
                    f"[resgate] chegada a 7 cm do {nome_marcador} "
                    f"confirmada; vitimas armazenadas={quantidade_destino}")
                controlador_verde = None
                monitor_chegada_verde = None
                deposito_cinza, comando = _preparar_deposito_final(
                    marcador_chegada,
                    quantidade_destino,
                )
                epoca_verde = None
                epoca_deposito_cinza = (
                    arduino.connection_epoch
                    if arduino is not None else None
                )
                ultimo_controle_ocioso = 0.0
                comando_atualizado = False

            # Uma falha visual do marcador (verde que saiu do quadro,
            # timeout de aproximacao ou leitura ultrassonica invalida) nao
            # pode devolver o robo ao segue-linha. Ele ja esta dentro da sala,
            # possivelmente carregando todas as vitimas; recomece apenas a
            # procura do MESMO destino e preserve esse inventario.
            if (
                args.drive
                and controlador_verde is not None
                and comando.terminal
            ):
                marcador_falhou = controlador_verde.target_kind
                nome_marcador = (
                    "verde" if marcador_falhou == "green" else "vermelho")
                print(
                    f"[resgate] rota do {nome_marcador} falhou "
                    f"({comando.state}); permanecendo no resgate")
                controlador_verde, comando = _recuperar_rota_deposito(
                    controlador_verde,
                    agora,
                )
                monitor_chegada_verde = None
                proxima_atualizacao_ultrassom_verde = 0.0
                epoca_verde = (
                    arduino.connection_epoch if arduino is not None else None)
                if marcadores is not None:
                    marcadores.reset()
                marcadores_atuais = {}
                ultimo_controle_ocioso = agora
                comando_atualizado = False

            if coleta_concluida is not None:
                vitimas_resgatadas += 1
                if coleta_concluida == "silver":
                    vitimas_prata_resgatadas += 1
                elif coleta_concluida == "black":
                    vitimas_pretas_resgatadas += 1
                contador_verde.reset()
                varreduras_sem_vitima = 0
                tentativas_recuperacao_coleta = 0
                epoca_coleta = None
                coleta = BallPickupSequencer(grippers_prepositioned=False)
                if trabalhador is not None:
                    trabalhador.reset_tracking()
                portao.reset()
                if marcadores is not None:
                    marcadores.reset()
                    marcadores_atuais = {}
                busca = _iniciar_busca_segura(arduino, start_time=agora)
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
                # A volta completa continua delimitando a cobertura, mas nao
                # encerra mais o resgate nem depende de ver o verde. Enquanto
                # a janela total estiver aberta, inicia outra volta.
                varreduras_sem_vitima += 1
                busca = _iniciar_busca_segura(arduino, start_time=agora)
                epoca_busca = None
                ultimo_controle_ocioso = 0.0
                restante = max(fim_busca_resgate_em - agora, 0.0)
                comando = MotionCommand(
                    "SEARCH_RESTART",
                    detail=(
                        f"volta {varreduras_sem_vitima} sem vitima; "
                        f"faltam {restante:.1f} s dos "
                        f"{cfg.RESCUE_SEARCH_DURATION_S:.0f} s de busca; "
                        "iniciando outra busca pulsada"),
                )
                comando_atualizado = False

            if (
                args.drive
                and not coleta.started
                and verificador_parede is None
                and comando.state == BallApproachController.NEAR
            ):
                tipo_confirmado = _validar_inicio_coleta(
                    comando,
                    coleta,
                    arduino,
                    movimento_enviado,
                    epoca_movimento,
                )
                iniciou_coleta = False
                modo_coleta_parede = False
                forma_alvo = (
                    frame_atual.shape if frame_atual is not None
                    else (cfg.RESCUE_CAMERA_MAX_HEIGHT,
                          cfg.RESCUE_CAMERA_MAX_WIDTH, 3))
                alvo_visual_coleta = deteccao_atual
                if (
                    alvo_visual_coleta is None
                    and resultado_atual is not None
                ):
                    alvo_visual_coleta = resultado_atual.locked_detection

                if (
                    coleta_apos_teste_parede is not None
                    and (
                        coleta_apos_teste_parede.target_kind
                        != tipo_confirmado
                        or not coleta_apos_teste_parede.matches(
                            alvo_visual_coleta,
                            forma_alvo,
                            agora,
                        )
                    )
                ):
                    # O detector mudou de alvo durante a reaproximacao. A
                    # decisao anterior nao vale para outra esfera.
                    coleta_apos_teste_parede = None

                if coleta_apos_teste_parede is not None:
                    modo_parede = coleta_apos_teste_parede.wall_mode
                    modo_coleta_parede = modo_parede
                    iniciou_coleta = _armar_coleta_confirmada(
                        comando,
                        coleta,
                        arduino,
                        movimento_enviado,
                        epoca_movimento,
                        modo_parede=modo_parede,
                    )
                    coleta_apos_teste_parede = None
                elif cfg.BALL_WALL_TEST_ENABLED:
                    if (
                        alvo_visual_coleta is None
                        or not alvo_visual_coleta.confirmed
                        or alvo_visual_coleta.kind != tipo_confirmado
                    ):
                        comando = MotionCommand(
                            "WALL_PROBE_FAULT",
                            detail=(
                                "deteccao original indisponivel; teste "
                                "lateral e garras bloqueados"
                            ),
                            terminal=True,
                        )
                    else:
                        verificador_parede = WallProbeController(
                            tipo_confirmado,
                            target_detection=alvo_visual_coleta,
                            start_time=agora,
                        )
                        epoca_parede = arduino.connection_epoch
                        busca = None
                        controlador = None
                        resultado_atual = None
                        deteccao_atual = None
                        portao.reset()
                        ultimo_controle_ocioso = 0.0
                        comando = MotionCommand(
                            "WALL_PROBE_START",
                            detail=(
                                f"vitima {tipo_confirmado} proxima; "
                                "medindo antes do teste lateral"
                            ),
                        )
                        comando_atualizado = False
                        print(
                            "[resgate] medindo o eixo; se houver eco proximo, "
                            "testara os dois lados com as garras bloqueadas")
                else:
                    iniciou_coleta = _armar_coleta_confirmada(
                        comando,
                        coleta,
                        arduino,
                        movimento_enviado,
                        epoca_movimento,
                    )

                if iniciou_coleta:
                    epoca_coleta = arduino.connection_epoch
                    if trabalhador is not None:
                        trabalhador.reset_tracking()
                    portao.reset()
                    busca = None
                    controlador = None
                    resultado_atual = None
                    deteccao_atual = None
                    ultimo_controle_ocioso = 0.0
                    modo = (
                        "parede: empurrando, dando re e "
                        if modo_coleta_parede else ""
                    )
                    print(
                        f"[coleta] vitima {coleta.target_kind} confirmada; "
                        f"{modo}baixando, fechando e selecionando")

            if (
                args.drive
                and controlador_saida is not None
                and comando.state == controlador_saida.DONE
                and comando.terminal
            ):
                from controle.direcao import steer

                steer()
                print(
                    "[saida] saida alinhada; trocando diretamente para a "
                    "camera de linha com o LED aceso para avancar ate a "
                    "proxima faixa")
                if trabalhador is not None:
                    trabalhador.close(timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
                    trabalhador = None
                if trabalhador_saida is not None:
                    trabalhador_saida.close(
                        timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
                    trabalhador_saida = None
                if captura is not None:
                    fechou = captura.close(
                        timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
                    if not fechou:
                        raise RuntimeError(
                            "a camera de resgate nao fechou antes da troca")
                    captura = None
                    fonte = None

                resultado_verificacao = _confirmar_saida_com_camera_linha(
                    arduino,
                    debug=args.debug,
                )
                frame_atual = None
                deteccao_saida = None
                faltas_saida = 0

                if resultado_verificacao == PRETA:
                    controlador_saida = None
                    portao_saida = None
                    epoca_saida = None
                    inicio_saida = None
                    codigo_saida = EXIT_OK
                    comando = MotionCommand(
                        "EXIT_BLACK_CONFIRMED",
                        detail=(
                            "faixa preta confirmada; segue-linha sera "
                            "iniciado apos liberar camera e serial"),
                        terminal=True,
                    )
                elif resultado_verificacao in (
                    RETOMADA_FALHOU,
                    LINHA_NAO_ENCONTRADA,
                ):
                    # Tanto uma faixa ausente quanto uma retomada geometrica
                    # incompleta falham paradas. Nao reabrir a busca frontal:
                    # isso repetiria a travessia e poderia sair da arena.
                    controlador_saida = None
                    portao_saida = None
                    epoca_saida = None
                    inicio_saida = None
                    codigo_saida = EXIT_INCOMPLETE
                    comando = MotionCommand(
                        "EXIT_LINE_RECOVERY_FAILED",
                        detail=(
                            "camera de linha nao concluiu a retomada; "
                            "robo parado, sem re nem nova rotacao"),
                        terminal=True,
                    )
                elif resultado_verificacao == NAO_PRETA:
                    # A re ja foi executada antes de fechar a camera de linha.
                    # Fechada essa camera, a frontal de resgate pode reabrir
                    # com seguranca e voltar aos pulsos de procura.
                    print(
                        "[saida] faixa prata; re de 1 segundo concluida; "
                        "reabrindo a camera de resgate para continuar a "
                        "busca")
                    from visao.captura_resgate import RescueCamera

                    fonte = RescueCamera(args.camera_index)
                    captura = LatestFrameSource(fonte)
                    sequencia_frame = 0
                    amostras_captura.clear()
                    agora_reinicio = time.monotonic()
                    if inicio_saida is None:
                        inicio_saida = agora_reinicio
                    controlador_saida = ExitPhaseController(
                        start_time=inicio_saida)
                    portao_saida, modelo_saida = novo_portao_saida(
                        modelo_saida)
                    trabalhador_saida = (
                        LatestFrameExitDetector(portao_saida.detector)
                        if modelo_saida is not None else None
                    )
                    sequencia_resultado_saida = 0
                    epoca_saida = arduino.connection_epoch
                    forma = (
                        cfg.RESCUE_CAMERA_MAX_HEIGHT,
                        cfg.RESCUE_CAMERA_MAX_WIDTH,
                        3,
                    )
                    comando = controlador_saida.update(
                        None,
                        forma,
                        mapper=None,
                        now=agora_reinicio,
                    )
                    ultimo_controle_ocioso = agora_reinicio
                else:
                    controlador_saida = None
                    portao_saida = None
                    epoca_saida = None
                    inicio_saida = None
                    codigo_saida = EXIT_INCOMPLETE
                    comando = MotionCommand(
                        "EXIT_CHECK_CANCELLED",
                        detail="verificacao cancelada; robo parado",
                        terminal=True,
                    )

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
                if controlador_saida is not None:
                    altura_saida, largura_saida = anotado.shape[:2]
                    topo_saida = int(round(
                        altura_saida * min(
                            cfg.EXIT_BLACK_ROI_TOP,
                            cfg.EXIT_LINE_ROI_TOP,
                        )
                    ))
                    fundo_saida = min(
                        int(round(
                            altura_saida * max(
                                cfg.EXIT_BLACK_ROI_BOTTOM,
                                cfg.EXIT_LINE_ROI_BOTTOM,
                            )
                        )),
                        altura_saida - 1,
                    )
                    cv2.line(
                        anotado,
                        (0, topo_saida),
                        (largura_saida - 1, topo_saida),
                        (255, 255, 0),
                        2,
                    )
                    cv2.line(
                        anotado,
                        (0, fundo_saida),
                        (largura_saida - 1, fundo_saida),
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        anotado,
                        "SAIDA: MODELO YOLO 640 ASYNC",
                        (8, max(topo_saida - 7, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    if modelo_saida is not None:
                        raw = modelo_saida.last_raw_detection
                        if raw is not None:
                            raw_x, raw_y, raw_w, raw_h = raw.bbox
                            cv2.rectangle(
                                anotado,
                                (int(raw_x), int(raw_y)),
                                (int(raw_x + raw_w - 1),
                                 int(raw_y + raw_h - 1)),
                                (0, 165, 255),
                                1,
                            )
                        cv2.putText(
                            anotado,
                            "NCNN raw="
                            f"{modelo_saida.last_raw_confidence:.0%} "
                            f"lim={modelo_saida.min_confidence:.0%} "
                            f"{modelo_saida.last_inference_ms:.0f}ms async "
                            "lock="
                            f"{'SIM' if portao_saida.track_locked else 'NAO'}",
                            (8, 84),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.46,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )
                if deteccao_saida is not None:
                    x, y, w, h = deteccao_saida.bbox
                    cv2.rectangle(
                        anotado,
                        (int(x), int(y)),
                        (int(x + w - 1), int(y + h - 1)),
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        anotado,
                        f"saida modelo={deteccao_saida.confidence:.0%}",
                        (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    centro_x = int(round(deteccao_saida.center_x))
                    centro_y = int(round(deteccao_saida.center_y))
                    comprimento = max(
                        int(round(deteccao_saida.width / 2)), 20)
                    angulo_rad = math.radians(deteccao_saida.angle_deg)
                    dx = int(round(math.cos(angulo_rad) * comprimento))
                    dy = int(round(math.sin(angulo_rad) * comprimento))
                    cv2.line(
                        anotado,
                        (centro_x - dx, centro_y - dy),
                        (centro_x + dx, centro_y + dy),
                        (0, 255, 255),
                        2,
                    )
                    cv2.circle(
                        anotado,
                        (largura_saida // 2, centro_y),
                        5,
                        (255, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        anotado,
                        f"centro={centro_x - largura_saida // 2:+d}px "
                        f"ang={deteccao_saida.angle_deg:+.1f}",
                        (8, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                if (
                    portao_saida is not None
                    and portao_saida.track_locked
                ):
                    cv2.putText(
                        anotado,
                        "FAIXA PRETA LOCK",
                        (8, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
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

                saida_concluida = _saida_concluida_libera_missao(
                    args, codigo_saida, comando)
                if saida_concluida:
                    print(
                        "[resgate] deposito e saida concluidos; devolvendo "
                        "o controle para mission.py")
                    break

                # Falhas de visão antes de a garra começar a sequência física
                # não invalidam a sala nem as vítimas já contadas. Volte ao
                # giro de busca; encerrar aqui devolveria o processo ao
                # supervisor, que não tem como saber em que yaw o robô ficou.
                falha_antes_da_coleta = (
                    args.drive
                    and not coleta.started
                    and (
                        controlador is not None
                        or verificador_parede is not None
                    )
                )
                if falha_antes_da_coleta:
                    print(
                        f"[resgate] falha recuperavel {comando.state}; "
                        "retomando a busca de vitimas sem sair da sala")
                    if trabalhador is not None:
                        trabalhador.reset_tracking()
                    portao.reset()
                    busca = _iniciar_busca_segura(arduino, start_time=agora)
                    controlador = None
                    verificador_parede = None
                    coleta_apos_teste_parede = None
                    resultado_atual = None
                    deteccao_atual = None
                    epoca_busca = None
                    epoca_parede = None
                    ultimo_controle_ocioso = agora
                    comando = MotionCommand(
                        "SEARCH_RECOVERY",
                        detail=(
                            "falha visual antes da coleta; procurando "
                            "novamente sem abandonar o resgate"),
                    )
                    time.sleep(TICK_S)
                    continue

                # A fase de saída só é criada depois dos dois depósitos. Se
                # ela perder a faixa, procurar de novo é seguro e preserva a
                # regra de que só a faixa preta confirmada libera a linha.
                if args.drive and controlador_saida is not None:
                    print(
                        f"[saida] falha recuperavel {comando.state}; "
                        "reiniciando a busca da faixa ainda dentro do resgate")
                    controlador_saida = ExitPhaseController(start_time=agora)
                    if portao_saida is not None:
                        portao_saida.reset(now=agora)
                    deteccao_saida = None
                    faltas_saida = 0
                    inicio_saida = agora
                    epoca_saida = (
                        arduino.connection_epoch if arduino is not None else None)
                    ultimo_controle_ocioso = agora
                    comando = MotionCommand(
                        "EXIT_RECOVERY",
                        detail=(
                            "falha ao localizar a saida; retomando a busca "
                            "sem liberar o segue-linha"),
                    )
                    time.sleep(TICK_S)
                    continue

                print(
                    f"[resgate] estado terminal {comando.state}; "
                    "motores parados")
                if args.drive and args.gerenciado_pela_missao:
                    _aguardar_desligamento_do_arduino(
                        arduino,
                        f"estado terminal {comando.state}",
                    )
                    raise ArduinoDisconnectedDuringRescue(
                        "Arduino desligado apos falha retida no resgate")
                if not args.debug or args.drive:
                    break

            time.sleep(TICK_S)

    except ArduinoDisconnectedDuringRescue as err:
        # O supervisor pode rearmar a missao somente neste caso explicito.
        # Qualquer outra falha termina parada, sem abrir o segue-linha.
        codigo_saida = EXIT_ARDUINO_DESCONECTADO
        print(f"[resgate] Arduino desconectado: {err}")
    except RuntimeError as err:
        # Em uma missão gerenciada, exceção de visão/serial não pode fechar o
        # resgate e deixar o supervisor reabrir o segue-linha. Se a placa
        # ainda responde, mantenha PARAR e aguarde o ciclo físico; se já caiu,
        # devolva o código explícito de desconexão.
        if (
            args.drive
            and args.gerenciado_pela_missao
            and arduino is not None
        ):
            if arduino.connected:
                _aguardar_desligamento_do_arduino(arduino, str(err))
            codigo_saida = EXIT_ARDUINO_DESCONECTADO
            print(f"[resgate] ERRO retido ate desconexao: {err}")
        else:
            codigo_saida = runtime_error_exit_code(
                args, arduino, codigo_saida)
            print(f"[resgate] ERRO: {err}")
    except KeyboardInterrupt:
        print("\n[resgate] Ctrl-C")
        if args.gerenciado_pela_missao:
            raise
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
        if trabalhador_saida is not None:
            _melhor_esforco(
                "encerrar o detector da saida",
                lambda: trabalhador_saida.close(
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
    return codigo_saida


def executar_resgate(args):
    """Entrada usada por ``mission.py`` sem criar outro programa."""
    return main(args)


if __name__ == "__main__":
    sys.exit(main())
