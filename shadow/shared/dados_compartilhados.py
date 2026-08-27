"""Guarda os valores compartilhados pelos processos de visão e controle."""

from collections import deque
import time
from multiprocessing import Array, Manager, Value
from typing import NamedTuple

import numpy as np

import config
from controle.estado_verde import (GREEN_OBSERVATION_ATOMIC_SIZE,
                                    GreenDecision, GreenManeuverState,
                                    GreenObservation, empty_observation)
from shared.gerenciadores import ConfigManager, Timer

config_manager = ConfigManager(str(config.CONFIG_INI_PATH))

manager = Manager()

# Números ficam em memória compartilhada real. ``Manager.Value`` passa cada
# leitura por um processo servidor; isso era especialmente caro nos loops de
# visão e controle. As quatro strings abaixo continuam no Manager para
# preservar a API atual sem criar uma migração arriscada de estados.
terminate = Value("b", False)
vision_ready = Value("b", False)   # SHADOW: controle espera a visao no boot

min_line_size = Value("i", config.MIN_LINE_SIZE_DEFAULT)

line_angle = Value("i", 0)
line_angle_y = Value("i", -1)
line_detected = Value("b", False)
line_ahead = Value("b", False)
line_crop = Value("d", config.LINE_CROP_INITIAL)
line_size = Value("d", 0.)
gap_angle = Value("d", 0.)
gap_center_x = Value("d", -180.)
gap_center_y = Value("d", -1.)
gap_end_width = Value("d", -1.)
red_detected = Value("b", False)
green_candidate = Value("b", False)
red_candidate = Value("b", False)
turn_dir = manager.Value("i", "straight")  # "straight"; "left"; "right"; "turn_around"
# Direcao armada: -1 esquerda, 0 nenhuma, 1 direita. O valor 2 neutraliza por
# alguns frames a memoria visual residual depois que o ramo ja foi alinhado.
# E Value real (nao Manager) para a visao consultar sem IPC caro.
green_turn_target = Value("b", 0)
black_average = Value("d", 0.)

last_bottom_point = Value("d", config.camera_x / 2)
last_bottom_point_y = Value("i", 0)
preferencia_linha_esquerda = Value("b", False)

line_status = manager.Value("i", "line_detected")  # "line_detected"; "gap_detected"; "gap_avoid"; "stop"

status = manager.Value("i", "Parado")

# Telemetria do controlador normal. O estado e numerico para nao consultar o
# Manager em cada frame: 0 TRACK, 1 CORNER, 2 LOST, 3 manobra especial.
STEERING_TRACK = 0
STEERING_CORNER = 1
STEERING_LOST = 2
STEERING_SPECIAL = 3
steering_state = Value("b", STEERING_TRACK)
steering_correction = Value("d", 0.)
steering_lateral_error = Value("d", 0.)
steering_heading = Value("d", 0.)
steering_left_pwm = Value("i", 0)
steering_right_pwm = Value("i", 0)


class ComandoMotores(NamedTuple):
    command_id: int
    publicado_em: float
    esquerda: int
    direita: int


_comando_motores = Array("d", 4, lock=True)


def publicar_comando_motores(esquerda, direita, *, publicado_em=None):
    with _comando_motores.get_lock():
        dados = _comando_motores.get_obj()
        dados[0] += 1
        dados[1] = (
            time.monotonic() if publicado_em is None else float(publicado_em))
        dados[2] = int(esquerda)
        dados[3] = int(direita)


def ler_comando_motores():
    with _comando_motores.get_lock():
        dados = tuple(_comando_motores.get_obj())
    return ComandoMotores(
        command_id=int(dados[0]),
        publicado_em=float(dados[1]),
        esquerda=int(dados[2]),
        direita=int(dados[3]),
    )

# ----------------------------------------------------------------------------
# Intersecoes e marcadores verdes
# ----------------------------------------------------------------------------
# A direcao e publicada como parte de uma unica observacao coerente. Os
# valores antigos ``turn_dir`` e ``green_turn_target`` permanecem apenas para
# compatibilidade com ferramentas/rotinas antigas; o controle competitivo nao
# deve montar uma decisao lendo esses campos separadamente.
INTERSECTION_NONE = int(GreenDecision.NONE)
INTERSECTION_PENDING = int(GreenDecision.PENDING)
INTERSECTION_STRAIGHT = int(GreenDecision.STRAIGHT)
INTERSECTION_LEFT = int(GreenDecision.LEFT)
INTERSECTION_RIGHT = int(GreenDecision.RIGHT)
INTERSECTION_UTURN = int(GreenDecision.UTURN)

GREEN_STATE_FOLLOW = int(GreenManeuverState.FOLLOW)
GREEN_STATE_OBSERVE = int(GreenManeuverState.OBSERVE)
GREEN_STATE_COMMITTED = int(GreenManeuverState.COMMITTED)
GREEN_STATE_APPROACH = int(GreenManeuverState.APPROACH)
GREEN_STATE_TURNING = int(GreenManeuverState.TURNING)
GREEN_STATE_REACQUIRE = int(GreenManeuverState.REACQUIRE)
GREEN_STATE_COOLDOWN = int(GreenManeuverState.COOLDOWN)
GREEN_STATE_FAULT_STOP = int(GreenManeuverState.FAULT_STOP)

green_calibration_ready = Value("b", False)
green_decision_consumed_id = Value("q", 0)
green_control_state = Value("b", GREEN_STATE_FOLLOW)
green_locked_decision = Value("b", INTERSECTION_NONE)
green_control_yaw = Value("d", float("nan"))
green_fault_stop = Value("b", False)
green_topology_junction_visible = Value("b", False)
# ACK da visão: decisão consumida cujo cooldown topológico realmente terminou.
green_rearmed_decision_id = Value("q", 0)

# ----------------------------------------------------------------------------
# Missão completa (shadow/mission.py)
# ----------------------------------------------------------------------------
# Estes valores existem sempre, mas só têm efeito quando `mission_mode` é
# ligado pelo supervisor da missão. Rodando `shadow/main.py` sozinho eles
# permanecem em falso e o segue-linha se comporta exatamente como antes.
mission_mode = Value("b", False)      # supervisor da missão no comando
entry_armed = Value("b", True)        # procurar a faixa prata agora?
entry_silver_detected = Value("b", False)   # candidato no frame atual
entry_silver_confirmed = Value("b", False)  # observação prata validada
entry_silver_votes = Value("i", 0)
entry_silver_reason = manager.Value("i", "")  # motivo da última rejeição
# Estado único publicado pela visão para sincronizar o controle e o verde.
ENTRY_SILVER_IDLE = 0
ENTRY_SILVER_VALIDATING = 1
ENTRY_SILVER_BLACK_FOLLOW = 2
entry_silver_state = Value("b", ENTRY_SILVER_IDLE)
rescue_requested = Value("b", False)  # controle pediu o handoff
red_finished = Value("b", False)      # faixa vermelha final cumprida
timer = Timer()


class ResultadoVisaoRapida(NamedTuple):
    """Um frame coerente usado apenas para decidir se é seguro acelerar."""

    sequencia: int
    publicado_em: float
    processamento_ms: float
    linha_detectada: bool
    linha_a_frente: bool
    angulo: float
    ponto_inferior_x: float
    ponto_inferior_y: float
    ponto_alvo_x: float
    ponto_alvo_y: float
    area_linha: float
    candidato_verde: bool
    candidato_vermelho: bool
    ponto_futuro_x: float
    ponto_futuro_y: float
    ponto_futuro_valido: bool
    faixa_transversal_y: float
    juncao_topologica_visivel: bool
    locked_branch_token: int
    locked_branch_valid: bool
    locked_branch_bottom_x: float
    locked_branch_bottom_y: float


# Um único bloqueio publica todos os campos. O controle nunca usa uma mistura
# do ângulo de um frame com o ponto inferior do frame seguinte.
_resultado_visao_rapida = Array("d", 22, lock=True)

# ``double`` mantem a estrutura contigua e barata entre processos. IDs sao
# contadores pequenos, portanto continuam exatamente representaveis.
_observacao_intersecao = Array(
    "d", empty_observation().as_atomic_values(), lock=True)


def publicar_observacao_intersecao(observacao):
    """Publica um GreenObservation inteiro sob um unico lock."""
    if not isinstance(observacao, GreenObservation):
        raise TypeError("observacao precisa ser GreenObservation")
    with _observacao_intersecao.get_lock():
        dados = _observacao_intersecao.get_obj()
        valores = observacao.as_atomic_values()
        for indice, valor in enumerate(valores):
            dados[indice] = valor


def ler_observacao_intersecao():
    with _observacao_intersecao.get_lock():
        dados = tuple(_observacao_intersecao.get_obj())
    if len(dados) != GREEN_OBSERVATION_ATOMIC_SIZE:
        raise RuntimeError("memoria compartilhada verde com tamanho invalido")
    return GreenObservation.from_atomic_values(dados)


def publicar_resultado_visao_rapida(
    *,
    sequencia=None,
    publicado_em,
    processamento_ms,
    linha_detectada,
    linha_a_frente,
    angulo,
    ponto_inferior_x,
    ponto_inferior_y,
    ponto_alvo_x,
    ponto_alvo_y,
    area_linha,
    candidato_verde,
    candidato_vermelho,
    ponto_futuro_x,
    ponto_futuro_y,
    ponto_futuro_valido,
    faixa_transversal_y,
    juncao_topologica_visivel,
    locked_branch_token=0,
    locked_branch_valid=False,
    locked_branch_bottom_x=-1.0,
    locked_branch_bottom_y=-1.0,
):
    with _resultado_visao_rapida.get_lock():
        dados = _resultado_visao_rapida.get_obj()
        if sequencia is None:
            dados[0] += 1
        else:
            sequencia = int(sequencia)
            if sequencia < 0:
                raise ValueError("sequencia da visao nao pode ser negativa")
            dados[0] = sequencia
        dados[1] = float(publicado_em)
        dados[2] = float(processamento_ms)
        dados[3] = bool(linha_detectada)
        dados[4] = bool(linha_a_frente)
        dados[5] = float(angulo)
        dados[6] = float(ponto_inferior_x)
        dados[7] = float(ponto_inferior_y)
        dados[8] = float(ponto_alvo_x)
        dados[9] = float(ponto_alvo_y)
        dados[10] = float(area_linha)
        dados[11] = bool(candidato_verde)
        dados[12] = bool(candidato_vermelho)
        dados[13] = float(ponto_futuro_x)
        dados[14] = float(ponto_futuro_y)
        dados[15] = bool(ponto_futuro_valido)
        dados[16] = float(faixa_transversal_y)
        dados[17] = bool(juncao_topologica_visivel)
        dados[18] = max(int(locked_branch_token), 0)
        dados[19] = bool(locked_branch_valid)
        dados[20] = float(locked_branch_bottom_x)
        dados[21] = float(locked_branch_bottom_y)


def ler_resultado_visao_rapida():
    with _resultado_visao_rapida.get_lock():
        dados = tuple(_resultado_visao_rapida.get_obj())
    return ResultadoVisaoRapida(
        sequencia=int(dados[0]),
        publicado_em=dados[1],
        processamento_ms=dados[2],
        linha_detectada=bool(dados[3]),
        linha_a_frente=bool(dados[4]),
        angulo=dados[5],
        ponto_inferior_x=dados[6],
        ponto_inferior_y=dados[7],
        ponto_alvo_x=dados[8],
        ponto_alvo_y=dados[9],
        area_linha=dados[10],
        candidato_verde=bool(dados[11]),
        candidato_vermelho=bool(dados[12]),
        ponto_futuro_x=dados[13],
        ponto_futuro_y=dados[14],
        ponto_futuro_valido=bool(dados[15]),
        faixa_transversal_y=dados[16],
        juncao_topologica_visivel=bool(dados[17]),
        locked_branch_token=int(dados[18]),
        locked_branch_valid=bool(dados[19]),
        locked_branch_bottom_x=dados[20],
        locked_branch_bottom_y=dados[21],
    )


def empty_time_arr(length: int = 240):
    return deque(maxlen=length)


def add_time_value(time_value_array, value):
    if isinstance(time_value_array, deque):
        time_value_array.append((time.perf_counter(), value))
        return time_value_array
    # Compatibilidade com ferramentas antigas que possam fornecer ndarray.
    return np.delete(
        np.vstack((time_value_array, [time.perf_counter(), value])),
        0,
        axis=0,
    )


def get_time_average(time_value_array, time_range):
    limite = time.perf_counter() - time_range
    if isinstance(time_value_array, deque):
        total = 0.
        quantidade = 0
        for instante, valor in reversed(time_value_array):
            if instante <= limite:
                break
            total += valor
            quantidade += 1
        return total / quantidade if quantidade else -1
    recentes = time_value_array[np.where(time_value_array[:, 0] > limite)]
    return np.mean(recentes[:, 1]) if recentes.size > 0 else -1
