"""Guarda os valores compartilhados pelos processos de visão e controle."""

from collections import deque
import time
from multiprocessing import Array, Manager, Value
from typing import NamedTuple

import numpy as np

import config
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
# Direcao que o controle ja armou para a manobra verde: -1 esquerda, 0 nenhuma,
# 1 direita. E Value real (nao Manager) para a visao preservar rapidamente o
# ramo escolhido durante a aproximacao e o giro.
green_turn_target = Value("b", 0)
black_average = Value("d", 0.)

last_bottom_point = Value("d", config.camera_x / 2)
last_bottom_point_y = Value("i", 0)
preferencia_linha_esquerda = Value("b", False)

line_status = manager.Value("i", "line_detected")  # "line_detected"; "gap_detected"; "gap_avoid"; "stop"

status = manager.Value("i", "Parado")

# ----------------------------------------------------------------------------
# Missão completa (shadow/mission.py)
# ----------------------------------------------------------------------------
# Estes valores existem sempre, mas só têm efeito quando `mission_mode` é
# ligado pelo supervisor da missão. Rodando `shadow/main.py` sozinho eles
# permanecem em falso e o segue-linha se comporta exatamente como antes.
mission_mode = Value("b", False)      # supervisor da missão no comando
entry_armed = Value("b", True)        # procurar a faixa prata agora?
entry_silver_detected = Value("b", False)   # candidato no frame atual
entry_silver_confirmed = Value("b", False)  # votação 3-de-5 fechada
entry_silver_votes = Value("i", 0)
entry_silver_reason = manager.Value("i", "")  # motivo da última rejeição
rescue_requested = Value("b", False)  # controle pediu o handoff
red_finished = Value("b", False)      # faixa vermelha final cumprida
# Armado exclusivamente pelo handoff resgate -> percurso. O main.py isolado
# nunca executa a busca pulsada especial da continuacao da saida.
exit_line_search_pending = Value("b", False)

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
    area_linha: float
    candidato_verde: bool
    candidato_vermelho: bool
    continuacao_saida_detectada: bool
    continuacao_saida_x: float
    continuacao_saida_y: float
    continuacao_saida_distancia: float


# Um único bloqueio publica todos os campos. O controle nunca usa uma mistura
# do ângulo de um frame com o ponto inferior do frame seguinte.
_resultado_visao_rapida = Array("d", 15, lock=True)


def publicar_resultado_visao_rapida(
    *,
    publicado_em,
    processamento_ms,
    linha_detectada,
    linha_a_frente,
    angulo,
    ponto_inferior_x,
    ponto_inferior_y,
    area_linha,
    candidato_verde,
    candidato_vermelho,
    continuacao_saida_detectada=False,
    continuacao_saida_x=-1.0,
    continuacao_saida_y=-1.0,
    continuacao_saida_distancia=0.0,
):
    with _resultado_visao_rapida.get_lock():
        dados = _resultado_visao_rapida.get_obj()
        dados[0] += 1
        dados[1] = float(publicado_em)
        dados[2] = float(processamento_ms)
        dados[3] = bool(linha_detectada)
        dados[4] = bool(linha_a_frente)
        dados[5] = float(angulo)
        dados[6] = float(ponto_inferior_x)
        dados[7] = float(ponto_inferior_y)
        dados[8] = float(area_linha)
        dados[9] = bool(candidato_verde)
        dados[10] = bool(candidato_vermelho)
        dados[11] = bool(continuacao_saida_detectada)
        dados[12] = float(continuacao_saida_x)
        dados[13] = float(continuacao_saida_y)
        dados[14] = float(continuacao_saida_distancia)


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
        area_linha=dados[8],
        candidato_verde=bool(dados[9]),
        candidato_vermelho=bool(dados[10]),
        continuacao_saida_detectada=bool(dados[11]),
        continuacao_saida_x=dados[12],
        continuacao_saida_y=dados[13],
        continuacao_saida_distancia=dados[14],
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
