"""Processa cada imagem da câmera do segue-linha."""

import time
import hashlib
from dataclasses import replace
from multiprocessing import shared_memory

import cv2
import numpy as np

import config
from config import (BLACK_AVG_SIDE_MASK, DEBUG_SHM_NAME, VISION_MAX_FRAMES,
                    camera_x, camera_y)
from shared.dados_compartilhados import (add_time_value, black_average,
                                         config_manager, empty_time_arr,
                                         entry_armed,
                                         green_candidate,
                                         green_calibration_ready,
                                         green_control_state,
                                         green_control_yaw,
                                         green_decision_consumed_id,
                                         green_rearmed_decision_id,
                                         green_locked_decision,
                                         green_topology_junction_visible,
                                         green_turn_target,
                                         get_time_average, last_bottom_point,
                                         last_bottom_point_y,
                                         line_ahead, line_angle, line_angle_y,
                                         line_detected, line_size,
                                         line_status, min_line_size,
                                         ler_comando_motores,
                                         publicar_observacao_intersecao,
                                         publicar_resultado_visao_rapida,
                                         preferencia_linha_esquerda,
                                         red_candidate, red_detected, status,
                                         steering_correction,
                                         steering_heading,
                                         steering_lateral_error,
                                         steering_state,
                                         terminate, timer, turn_dir,
                                         vision_ready)
from visao import linha as line_module
from controle.estado_verde import (GreenDecision, GreenDecisionTracker,
                                    GreenObservation, empty_observation)
from visao.calibracao_wide import (WideCalibrationError,
                                    carregar_calibracao)
from visao.captura import LineCamera
from visao.entrada_missao import build_entry_gate, update_entry_silver
from visao.gap import apply_gap_avoid_mask, publish_gap_geometry, reset_gap_values
from visao.linha import calculate_angle, determine_correct_line
from visao.trajetoria import extrair_ponto_futuro
from visao.intersecao_verde import (BranchKind, GreenTopologyTracker,
                                     MarkerPhase, TopologyConfig,
                                     draw_topology_debug)
from visao.faixa_verde import (altura_faixa_transversal,
                               tem_continuacao_reta)
from visao.gravacao_debug import GravadorVisao
from visao.rastreamento_ramo import (LockedBranchResult,
                                     LockedBranchTracker)
from visao.vermelho import ConfirmadorVermelho, check_contour_size
from visao.verde import (DirecaoVerdePersistente, check_green)

# Cores carregadas do config.ini (fallback: valores do config.py)
black_min = np.array(config.BLACK_MIN_DEFAULT)
black_max_normal_top = np.array(config.BLACK_MAX_NORMAL_TOP_DEFAULT)
black_max_normal_bottom = np.array(config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT)
black_max_ramp_down_top = np.array(config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT)
green_min = np.array(config.GREEN_MIN_DEFAULT)
green_max = np.array(config.GREEN_MAX_DEFAULT)
red_min_1 = np.array(config.RED_MIN_1_DEFAULT)
red_max_1 = np.array(config.RED_MAX_1_DEFAULT)
red_min_2 = np.array(config.RED_MIN_2_DEFAULT)
red_max_2 = np.array(config.RED_MAX_2_DEFAULT)


def _green_topology_mode(calibration):
    """Retorna o referencial autorizado para decidir o verde."""

    if calibration is not None:
        return "metric"
    if config.GREEN_PIXEL_FALLBACK_ENABLED:
        return "pixel"
    return "disabled"


def update_color_values():
    global black_max_normal_top, black_max_normal_bottom, \
        black_max_ramp_down_top, green_min, green_max, \
        red_min_1, red_max_1, red_min_2, red_max_2

    def read(name, fallback):
        value = config_manager.read_variable('color_values_line', name)
        return np.array(value) if value is not None else np.array(fallback)

    black_max_normal_top = read('black_max_normal_top', config.BLACK_MAX_NORMAL_TOP_DEFAULT)
    black_max_normal_bottom = read('black_max_normal_bottom', config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT)
    black_max_ramp_down_top = read(
        'black_max_ramp_down_top',
        config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT,
    )

    green_min = read('green_min', config.GREEN_MIN_DEFAULT)
    green_max = read('green_max', config.GREEN_MAX_DEFAULT)

    red_min_1 = read('red_min_1', config.RED_MIN_1_DEFAULT)
    red_max_1 = read('red_max_1', config.RED_MAX_1_DEFAULT)
    red_min_2 = read('red_min_2', config.RED_MIN_2_DEFAULT)
    red_max_2 = read('red_max_2', config.RED_MAX_2_DEFAULT)


def _has_black_ahead(mask):
    """Há preto persistente no corredor central à frente do robô?

    A mesma geometria avalia os dois perfis de preto: o normal, que orienta a
    linha, e o específico da rampa. Uma barra transversal isolada não basta;
    o preto precisa ocupar muitas linhas do corredor na direção de marcha.
    """
    if mask is None or mask.ndim != 2:
        raise ValueError("_has_black_ahead exige uma mascara 2D")
    height, width = mask.shape
    ahead = mask[
        0:int(height * config.GAP_AHEAD_Y_MAX),
        int(width * config.GAP_AHEAD_X_MIN):int(
            width * config.GAP_AHEAD_X_MAX),
    ]
    if not ahead.size:
        return False
    row_fill = np.count_nonzero(ahead, axis=1) / ahead.shape[1]
    return bool(
        np.mean(row_fill >= config.GAP_AHEAD_ROW_FILL)
        >= config.GAP_AHEAD_ROW_PERSISTENCE
    )


def _marker_ids_relevantes(observacao, decisao):
    if decisao in (GreenDecision.LEFT, GreenDecision.RIGHT,
                   GreenDecision.UTURN):
        marcadores = [
            marcador for marcador in observacao.markers
            if marcador.valid and marcador.phase == MarkerPhase.PRE
        ]
    elif decisao == GreenDecision.STRAIGHT:
        marcadores = [
            marcador for marcador in observacao.markers
            if (marcador.plausible and marcador.associated
                and marcador.phase == MarkerPhase.POST)
        ]
    else:
        marcadores = [
            marcador for marcador in observacao.markers
            if marcador.plausible and marcador.associated
        ]
    return tuple(sorted(
        marcador.marker_id for marcador in marcadores
        if marcador.marker_id > 0
    ))[:2]


def _alvo_topologico_no_frame_cru(alvo, target_to_raw=None):
    """Converte apenas o alvo de controle do frame retificado para o cru.

    A juncao continua no frame retificado porque o gatilho de 82% foi definido
    nesse espaco. O controlador de STRAIGHT, por outro lado, combina o alvo com
    ``ponto_inferior_x``, que pertence ao frame cru; publicar ambos no mesmo
    referencial evita uma correcao lateral criada pela distorcao da wide.
    """

    ponto = np.asarray(alvo, dtype=np.float64).reshape(1, 2)
    if target_to_raw is not None:
        ponto = np.asarray(
            target_to_raw(ponto), dtype=np.float64).reshape(-1, 2)
        if len(ponto) != 1:
            raise ValueError("target_to_raw deve devolver exatamente um ponto")
    if not np.all(np.isfinite(ponto)):
        raise ValueError("alvo topologico projetou coordenada nao finita")
    return float(ponto[0, 0]), float(ponto[0, 1])


def _evento_bruto_topologia(
    observacao,
    *,
    sequencia,
    timestamp,
    target_to_raw=None,
):
    decisao = GreenDecision(int(observacao.decision))
    juncao = (
        (-1., -1.)
        if observacao.junction_image is None
        else tuple(float(valor) for valor in observacao.junction_image)
    )
    alvo = (
        (-1., -1.)
        if observacao.target_branch is None
        else _alvo_topologico_no_frame_cru(
            observacao.target_branch.target_image,
            target_to_raw,
        )
    )
    propagada = bool(observacao.entry_propagated)
    juncao_visivel = bool(
        observacao.junction_image is not None and not propagada)
    pronto = bool(
        juncao_visivel
        and observacao.junction_image[1]
        >= camera_y * config.GREEN_TOPOLOGY_READY_Y_RATIO
    )
    return GreenObservation(
        sequence=sequencia,
        junction_id=observacao.junction_id,
        decision_id=0,
        timestamp=timestamp,
        decision=decisao,
        confidence=observacao.confidence,
        entry_tangent=observacao.entry_tangent,
        junction_center=juncao,
        target_branch=alvo,
        target_branch_token=(
            0 if observacao.target_branch is None
            else observacao.target_branch.branch_token
        ),
        ready_to_turn=pronto,
        junction_visible=juncao_visivel,
        geometry_predicted=propagada,
        marker_ids=_marker_ids_relevantes(observacao, decisao),
    )


def _fundir_verde_simples(observacao, direcao):
    """Usa o verde antigo como autorizacao e a topologia como caminho."""

    mapa = {
        "left": (GreenDecision.LEFT, BranchKind.LEFT),
        "right": (GreenDecision.RIGHT, BranchKind.RIGHT),
        "turn_around": (GreenDecision.UTURN, BranchKind.INCOMING),
    }
    if direcao not in mapa:
        if GreenDecision(int(observacao.decision)) in (
            GreenDecision.LEFT, GreenDecision.RIGHT, GreenDecision.UTURN,
        ):
            return replace(
                observacao,
                decision=GreenDecision.PENDING,
                target_branch=None,
                ready_to_turn=False,
                reason="marcador ainda nao confirmado pelo detector simples",
            )
        return observacao

    decisao, tipo_ramo = mapa[direcao]
    alvo = next(
        (ramo for ramo in observacao.branches if ramo.kind == tipo_ramo),
        None,
    )
    if observacao.junction_image is None or alvo is None:
        return replace(
            observacao,
            decision=GreenDecision.PENDING,
            target_branch=None,
            ready_to_turn=False,
            confidence=max(float(observacao.confidence), .65),
            reason=f"verde {direcao} confirmado; aguardando ramo futuro",
        )
    return replace(
        observacao,
        decision=decisao,
        target_branch=alvo,
        confidence=max(float(observacao.confidence), .90),
        reason=f"verde simples {direcao} + ramo futuro topologico",
    )


def _ids_verde_simples(direcao):
    if direcao == "turn_around":
        return (910001, 910002)
    if direcao == "left":
        return (910001,)
    if direcao == "right":
        return (910002,)
    return ()


def _preferencia_esquerda_permitida(preferencia_ativa, evento_verde):
    """Qualquer evidência topológica suspende a preferência de obstáculo."""

    return bool(
        preferencia_ativa
        and evento_verde.decision == GreenDecision.NONE
    )


def _juncao_presente_para_saida(observacao):
    """Falha fechado durante REACQUIRE/COOLDOWN.

    Uma entrada propagada representa uma perda curta da geometria, nao prova
    que o robo ja deixou a intersecao. Portanto ela impede tanto a liberacao
    do retorno quanto o rearme do mesmo ``decision_id``. Apenas um frame sem
    juncao *e* sem propagacao conta como juncao realmente ausente.
    """

    return bool(
        observacao.junction_image is not None
        or observacao.entry_propagated
    )


def vision_loop(debug=False, record_dir=None):
    green_calibration_ready.value = False
    green_topology_junction_visible.value = False
    green_rearmed_decision_id.value = 0
    # As operações trabalham em apenas 448×252. Paralelizá-las faz a Pi 5
    # acordar vários núcleos por poucos milissegundos sem mudar o resultado;
    # uma thread deixa o consumo mais uniforme e ainda cabe folgado em 25 ms.
    cv2.setNumThreads(max(1, int(config.VISION_OPENCV_THREADS)))
    line_module.init_tracker()
    inicio_aquecimento = time.perf_counter()
    print("[visão] preparando cálculos rápidos...")
    line_module.aquecer_numba()
    print("[visão] cálculos prontos em "
          f"{time.perf_counter() - inicio_aquecimento:.2f}s")

    bottom_y = camera_y

    time_line_angle = empty_time_arr()
    time_last_bottom_point_x = empty_time_arr()
    time_last_average_line_point = empty_time_arr()

    camera = LineCamera()
    # O verde usa proporcoes da propria imagem/linha. Nao depende de tabuleiro
    # nem de arquivo de calibracao; a configuracao wide da captura e mantida.
    calibracao_wide = None
    green_calibration_ready.value = True
    print("[visao] verde em pixels + caminho futuro; sem tabuleiro")

    rastreador_topologia = GreenTopologyTracker(config=TopologyConfig(
        marker_min_mm=config.GREEN_TOPOLOGY_MARKER_MIN_MM,
        marker_max_mm=config.GREEN_TOPOLOGY_MARKER_MAX_MM,
        pre_post_margin_sides=(
            config.GREEN_TOPOLOGY_PRE_POST_MARGIN_RATIO),
        min_branch_length_widths=(
            config.GREEN_TOPOLOGY_MIN_BRANCH_LINE_WIDTHS),
        tangent_history_frames=(
            config.GREEN_TOPOLOGY_ENTRY_HISTORY_FRAMES),
    ))
    rastreador_ramo_travado = LockedBranchTracker()
    direcao_verde_persistente = DirecaoVerdePersistente()
    confirmador_evento_verde = GreenDecisionTracker(
        confirm_frames=config.GREEN_TOPOLOGY_CONFIRM_FRAMES,
        window_frames=config.GREEN_TOPOLOGY_CONFIRM_WINDOW,
        second_marker_wait_s=config.GREEN_TOPOLOGY_SECOND_MARKER_WAIT_S,
        prediction_max_s=config.GREEN_TOPOLOGY_PREDICTION_MAX_S,
        rearm_frames=config.GREEN_TOPOLOGY_REARM_CLEAR_FRAMES,
        rearm_min_s=config.GREEN_TOPOLOGY_COOLDOWN_S,
    )
    sequencia_topologia = 0
    observacao_topologica = None
    evento_verde = empty_observation()
    ultimo_consumido_verde = 0
    decisao_em_cooldown_topologico = 0

    update_color_values()
    gravador = None
    if record_dir:
        try:
            calibracao_path = config.GREEN_WIDE_CALIBRATION_PATH
            calibracao_hash = (
                hashlib.sha256(calibracao_path.read_bytes()).hexdigest()
                if calibracao_path.is_file()
                else None
            )
            gravador = GravadorVisao(
                record_dir,
                largura=camera_x,
                altura=camera_y,
                fps=camera.capture_fps,
                manifest={
                    "calibration_path": str(calibracao_path),
                    "calibration_sha256": calibracao_hash,
                    "calibration_metadata": (
                        None if calibracao_wide is None
                        else calibracao_wide.metadata
                    ),
                    "black_min": black_min.tolist(),
                    "black_max_top": black_max_normal_top.tolist(),
                    "black_max_bottom": black_max_normal_bottom.tolist(),
                    "green_min": green_min.tolist(),
                    "green_max": green_max.tolist(),
                    "camera_index": config.LINE_CAMERA_INDEX,
                    "sensor": config.LINE_CAMERA_SENSOR_ID,
                    "capture_mode": camera.capture_mode_id,
                    "lens_position": camera.lens_position,
                },
            )
        except (OSError, ValueError, cv2.error) as err:
            print(f"[visao] nao foi possivel iniciar gravacao: {err}")

    shm = None
    shm_array = None
    if debug:
        shm = shared_memory.SharedMemory(name=DEBUG_SHM_NAME)
        shm_array = np.ndarray((camera_y, camera_x, 3), dtype=np.uint8, buffer=shm.buf)

    # Faixa prata de entrada. O detector vive aqui porque este é o único
    # processo que possui a câmera de linha; publicar o resultado por memória
    # compartilhada evita abrir a mesma câmera duas vezes. Sem `mission_mode`
    # ele nunca é construído e o custo é zero.
    entry_gate = build_entry_gate()
    ultimo_alinhamento_entrada = 0.

    # Matriz usada para reduzir ruídos das máscaras.
    kernal = np.ones((3, 3), np.uint8)
    confirmador_vermelho = ConfirmadorVermelho()

    # Contador e limitador de imagens por segundo.
    fps_time = time.perf_counter()
    counter = 0
    fps = 0
    fps_limit_time = time.perf_counter()

    timer.set_timer("multiple_bottom", .05)
    timer.set_timer("multiple_side_l", .05)
    timer.set_timer("multiple_side_r", .05)

    try:
        while not terminate.value:
            cv2_img = camera.get_frame()
            frame_cru_gravacao = (
                cv2_img.copy() if (gravador is not None or debug) else None)
            frame_captured_at = time.perf_counter()

            if time.perf_counter() - fps_limit_time <= 1 / VISION_MAX_FRAMES:
                continue
            fps_limit_time = time.perf_counter()
            inicio_processamento = time.perf_counter()
            sequencia_topologia += 1

            # Valores locais do MESMO frame. Só são publicados juntos depois
            # que toda a análise terminar.
            linha_detectada_frame = False
            linha_a_frente_frame = False
            angulo_frame = 0.
            ponto_inferior_x_frame = camera_x / 2
            ponto_inferior_y_frame = 0.
            ponto_alvo_x_frame = camera_x / 2
            ponto_alvo_y_frame = 0.
            ponto_futuro_x_frame = camera_x / 2
            ponto_futuro_y_frame = camera_y
            ponto_futuro_valido_frame = False
            faixa_transversal_y_frame = -1.
            area_linha_frame = 0.
            candidato_verde_frame = False
            candidato_vermelho_frame = False
            hsv_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
            green_image = cv2.inRange(hsv_image, green_min, green_max)
            red_image = cv2.bitwise_or(
                cv2.inRange(hsv_image, red_min_1, red_max_1),
                cv2.inRange(hsv_image, red_min_2, red_max_2),
            )

            black_image = cv2.inRange(cv2_img, black_min, black_max_normal_bottom)
            limite_topo = int(camera_y * .4)
            black_image[:limite_topo] = cv2.inRange(
                cv2_img[:limite_topo],
                black_min,
                black_max_normal_top,
            )
            black_image = cv2.bitwise_and(
                black_image,
                cv2.bitwise_not(green_image),
            )

            media_preto = float(np.mean(black_image))
            black_average.value = media_preto

            # A rampa tem seu proprio teto de preto, calibrado no grupo 3.
            # Esta mascara nunca substitui a mascara normal da linha: ela so
            # acrescenta um veto independente a entrada de resgate.
            black_ramp_image = np.zeros_like(black_image)
            black_ramp_image[:limite_topo] = cv2.inRange(
                cv2_img[:limite_topo],
                black_min,
                black_max_ramp_down_top,
            )
            black_ramp_image[:limite_topo] = cv2.bitwise_and(
                black_ramp_image[:limite_topo],
                cv2.bitwise_not(green_image[:limite_topo]),
            )
            # A mascara normal muda no segue-linha abaixo. Preserva a medida
            # bruta para compara-la com a caixa prata deste mesmo frame.
            entry_black_mask = black_image.copy()
            entry_ramp_black_mask = black_ramp_image

            # Topologia usa a segmentacao bruta, antes de gap, recortes
            # laterais e morfologia agressiva do seguidor comum.
            topologia_preta = black_image.copy()
            topologia_verde = green_image.copy()
            if calibracao_wide is not None:
                try:
                    topologia_preta = calibracao_wide.rectify_mask(
                        topologia_preta)
                    topologia_verde = calibracao_wide.rectify_mask(
                        topologia_verde)
                    ponto_entrada_topologia = tuple(
                        calibracao_wide.rectify_points((
                            (camera_x / 2, camera_y - 1),
                        ))[0]
                    )
                    homografia_topologia = calibracao_wide.pixel_to_ground
                except (WideCalibrationError, cv2.error, ValueError) as err:
                    print(
                        "[visao] calibracao wide falhou em runtime; "
                        f"mudando para modo PIXEL ({err})"
                    )
                    calibracao_wide = None
                    green_calibration_ready.value = bool(
                        config.GREEN_PIXEL_FALLBACK_ENABLED)
                    rastreador_ramo_travado.reset()
                    ponto_entrada_topologia = (camera_x / 2, camera_y - 1)
                    homografia_topologia = None
            else:
                ponto_entrada_topologia = (camera_x / 2, camera_y - 1)
                homografia_topologia = None

            modo_topologia_verde = _green_topology_mode(calibracao_wide)
            topologia_verde_ativa = modo_topologia_verde != "disabled"

            observacao_topologica = rastreador_topologia.update(
                topologia_preta,
                topologia_verde,
                image_to_ground=homografia_topologia,
                entry_point=ponto_entrada_topologia,
            )
            frente_entrada = None
            if (observacao_topologica.entry_tangent_image is not None
                    and observacao_topologica.junction_image is not None):
                frente_entrada = (
                    observacao_topologica.entry_tangent_image[0]
                    - observacao_topologica.junction_image[0],
                    observacao_topologica.entry_tangent_image[1]
                    - observacao_topologica.junction_image[1],
                )

            # Repete somente a limpeza leve usada pelo detector antigo. Ele
            # decide se existe marcador e de qual lado; nao decide movimento.
            verde_simples = cv2.erode(
                topologia_verde, kernal, iterations=1)
            verde_simples = cv2.dilate(
                verde_simples, kernal, iterations=11)
            verde_simples = cv2.erode(
                verde_simples, kernal, iterations=9)
            preto_simples = cv2.erode(
                topologia_preta, kernal, iterations=5)
            preto_simples = cv2.dilate(
                preto_simples, kernal, iterations=17)
            preto_simples = cv2.erode(
                preto_simples, kernal, iterations=9)
            contornos_verdes_simples, _ = cv2.findContours(
                verde_simples, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            direcao_verde_simples_frame = check_green(
                contornos_verdes_simples,
                preto_simples,
                debug_img=cv2_img if debug else None,
                entry_forward=frente_entrada,
            )
            instante_evento = time.monotonic()
            direcao_verde_autorizada = (
                direcao_verde_persistente.atualizar(
                    direcao_verde_simples_frame, instante_evento)
            )
            compromisso_existente = confirmador_evento_verde.committed
            if compromisso_existente is not None:
                direcao_verde_autorizada = {
                    GreenDecision.LEFT: "left",
                    GreenDecision.RIGHT: "right",
                    GreenDecision.UTURN: "turn_around",
                }.get(compromisso_existente.decision, "straight")

            observacao_topologica = _fundir_verde_simples(
                observacao_topologica, direcao_verde_autorizada)
            green_topology_junction_visible.value = (
                _juncao_presente_para_saida(observacao_topologica)
            )
            evento_bruto = _evento_bruto_topologia(
                observacao_topologica,
                sequencia=sequencia_topologia,
                timestamp=instante_evento,
                target_to_raw=(
                    None
                    if calibracao_wide is None
                    else calibracao_wide.unrectify_points
                ),
            )
            ids_simples = _ids_verde_simples(direcao_verde_autorizada)
            if ids_simples:
                evento_bruto = replace(
                    evento_bruto, marker_ids=ids_simples)
            if topologia_verde_ativa:
                consumido = int(green_decision_consumed_id.value)
                if consumido > ultimo_consumido_verde:
                    if confirmador_evento_verde.consume(
                        consumido, timestamp=instante_evento,
                    ):
                        decisao_em_cooldown_topologico = consumido
                    direcao_verde_persistente.reset()
                    ultimo_consumido_verde = consumido
                evento_verde = confirmador_evento_verde.update(evento_bruto)
            else:
                # A geometria continua visivel no debug, mas nao publica uma
                # decisao quando tanto a metrica quanto o fallback estao off.
                evento_verde = empty_observation(
                    sequencia_topologia, instante_evento)

            # LEFT/RIGHT carregam o ramo indicado pelo marcador; UTURN
            # carrega o INCOMING. O fluxo optico mantem esse mesmo pedaco
            # fisico de tinta identificado durante a manobra. Perder os
            # pontos invalida o token, sem procurar outra linha parecida.
            ramo_travado = LockedBranchResult()
            if (
                topologia_verde_ativa
                and evento_verde.committed
                and evento_verde.decision in (
                    GreenDecision.LEFT,
                    GreenDecision.RIGHT,
                    GreenDecision.UTURN,
                )
                and evento_verde.target_branch_token > 0
            ):
                frame_cinza_cru = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                if calibracao_wide is not None:
                    frame_ramo = calibracao_wide.rectify(frame_cinza_cru)
                    alvo_ramo_analise = calibracao_wide.rectify_points((
                        evento_verde.target_branch,
                    ))[0]
                else:
                    frame_ramo = frame_cinza_cru
                    alvo_ramo_analise = evento_verde.target_branch
                mesmo_ramo = bool(
                    rastreador_ramo_travado.decision_id
                    == evento_verde.decision_id
                    and rastreador_ramo_travado.branch_token
                    == evento_verde.target_branch_token
                )
                if mesmo_ramo:
                    ramo_travado = rastreador_ramo_travado.update(
                        frame_ramo,
                        topologia_preta,
                        sequence=sequencia_topologia,
                        decision_id=evento_verde.decision_id,
                        branch_token=evento_verde.target_branch_token,
                    )
                    if (
                        evento_verde.junction_visible
                        and not evento_verde.geometry_predicted
                    ):
                        try:
                            ramo_travado = (
                                rastreador_ramo_travado
                                .refresh_from_verified_geometry(
                                    frame_ramo,
                                    topologia_preta,
                                    sequence=sequencia_topologia,
                                    decision_id=evento_verde.decision_id,
                                    branch_token=(
                                        evento_verde.target_branch_token),
                                    junction=evento_verde.junction_center,
                                    target=alvo_ramo_analise,
                                    line_width_px=(
                                        observacao_topologica.line_width_px),
                                )
                            )
                        except (cv2.error, TypeError, ValueError):
                            # O fluxo antigo continua valido; uma falha de
                            # reseed nunca autoriza trocar de componente.
                            pass
                else:
                    try:
                        ramo_travado = rastreador_ramo_travado.arm(
                            frame_ramo,
                            topologia_preta,
                            sequence=sequencia_topologia,
                            decision_id=evento_verde.decision_id,
                            branch_token=evento_verde.target_branch_token,
                            junction=evento_verde.junction_center,
                            target=alvo_ramo_analise,
                            line_width_px=(
                                observacao_topologica.line_width_px),
                        )
                    except (cv2.error, TypeError, ValueError):
                        rastreador_ramo_travado.reset()
                        ramo_travado = LockedBranchResult()
            else:
                rastreador_ramo_travado.reset()

            ramo_travado_x_cru = -1.0
            ramo_travado_y_cru = -1.0
            if ramo_travado.valid:
                try:
                    ponto_ramo_analise = (
                        (ramo_travado.bottom_x, ramo_travado.bottom_y),
                    )
                    ponto_ramo_cru = (
                        calibracao_wide.unrectify_points(
                            ponto_ramo_analise)[0]
                        if calibracao_wide is not None
                        else ponto_ramo_analise[0]
                    )
                    ramo_travado_x_cru = float(ponto_ramo_cru[0])
                    ramo_travado_y_cru = float(ponto_ramo_cru[1])
                except (cv2.error, TypeError, ValueError):
                    ramo_travado = LockedBranchResult()

            # Esta leitura e exclusiva do detector de gap. O veto da prata
            # usa as duas mascaras brutas, mas apenas acima da caixa que o
            # YOLO encontrou no mesmo frame.
            linha_a_frente_frame = _has_black_ahead(black_image)
            line_ahead.value = linha_a_frente_frame

            # Recorta partes que não devem participar da decisão.
            if line_status.value == "gap_avoid":
                apply_gap_avoid_mask(black_image)

            if (
                bottom_y < camera_y * .95
                and media_preto < BLACK_AVG_SIDE_MASK
                and line_status.value == "line_detected"
            ):
                cv2.rectangle(black_image, (0, 0), (int(camera_x * .25), camera_y), 0, -1)
                cv2.rectangle(black_image, (int(camera_x * .75), 0), (camera_x, camera_y), 0, -1)

            # Redução de ruído.
            if line_status.value == "gap_avoid":
                black_image = cv2.erode(black_image, kernal, iterations=5)
                black_image = cv2.dilate(black_image, kernal, iterations=8)
            else:
                black_image = cv2.erode(black_image, kernal, iterations=5)
                black_image = cv2.dilate(black_image, kernal, iterations=17)
                black_image = cv2.erode(black_image, kernal, iterations=9)

            red_image = cv2.erode(red_image, kernal, iterations=1)
            red_image = cv2.dilate(red_image, kernal, iterations=11)
            red_image = cv2.erode(red_image, kernal, iterations=9)

            # Encontra os contornos.
            contours_red, _ = cv2.findContours(red_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_blk, _ = cv2.findContours(black_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            area_minima_linha = min_line_size.value
            contours_blk = [
                contorno
                for contorno in contours_blk
                if cv2.contourArea(contorno) > area_minima_linha
            ]

            # Procura a faixa vermelha.
            candidato_vermelho_frame = check_contour_size(
                contours_red,
                "red",
                debug_img=cv2_img if debug else None,
                frame_shape=red_image.shape,
            )
            red_candidate.value = candidato_vermelho_frame
            red_detected.value = confirmador_vermelho.atualizar(
                candidato_vermelho_frame)

            # A topologia ja foi calculada na mascara bruta. O evento abaixo
            # e a unica fonte de direcao; nao ha segunda memoria temporal.
            direcao_verde_bruta = direcao_verde_simples_frame
            candidato_verde_frame = bool(
                direcao_verde_simples_frame != "straight"
                or any(cv2.contourArea(contorno) > config.GREEN_MIN_AREA
                       for contorno in contornos_verdes_simples)
            )
            green_candidate.value = candidato_verde_frame
            turn_direction = {
                GreenDecision.LEFT: "left",
                GreenDecision.RIGHT: "right",
                GreenDecision.UTURN: "turn_around",
            }.get(evento_verde.decision, "straight")
            turn_dir.value = turn_direction

            # Escolhe o contorno correto da linha.
            if len(contours_blk) > 0:
                linha_detectada_frame = True
                line_detected.value = True
                # Uma preferencia residual do desvio de obstaculo pode
                # desempatar linha comum, mas nunca vence uma decisao verde
                # atomica que ja foi confirmada, inclusive o ramo STRAIGHT.
                preferir_esquerda = _preferencia_esquerda_permitida(
                    preferencia_linha_esquerda.value,
                    evento_verde,
                )
                alvo_verde = green_turn_target.value
                direcao_marcada = (
                    "left" if alvo_verde < 0 else
                    "right" if alvo_verde == 1 else
                    "straight" if alvo_verde == 2 else turn_dir.value
                )
                verde_autorizado = (
                    alvo_verde in (-1, 1)
                    or turn_direction in ("left", "right", "turn_around")
                )
                if topologia_verde_ativa:
                    # O ramo reto fica preso ao evento confirmado, e nao
                    # apenas ao classificador do frame atual. No meio da
                    # travessia a barra lateral pode sair da imagem e a
                    # topologia bruta deixar de dizer STRAIGHT; isso jamais
                    # pode devolver a escolha ao lookahead generico.
                    reto_confirmado = bool(
                        evento_verde.committed
                        and evento_verde.decision == GreenDecision.STRAIGHT
                    )
                    intersecao_sem_verde = bool(
                        reto_confirmado
                        or (
                            not verde_autorizado
                            and observacao_topologica.junction_image is not None
                            and GreenDecision(int(
                                observacao_topologica.decision
                            )) == GreenDecision.STRAIGHT
                        )
                    )
                else:
                    # Diagnostico sem calibracao nao muda o comportamento que
                    # ja existia para uma cruz sem marcador.
                    limite_lateral = config.GREEN_BRANCH_TRANSVERSE_MIN_RUN_PX
                    possui_extensao_lateral = any(
                        np.min(contorno[:, 0, 0])
                        < camera_x / 2 - limite_lateral
                        or np.max(contorno[:, 0, 0])
                        > camera_x / 2 + limite_lateral
                        for contorno in contours_blk
                    )
                    intersecao_sem_verde = bool(
                        not verde_autorizado
                        and possui_extensao_lateral
                        and tem_continuacao_reta(black_image)
                    )
                if intersecao_sem_verde:
                    direcao_marcada = "straight"
                direcao_geometria = (
                    "straight" if preferir_esquerda else direcao_marcada
                )
                preferir_esquerda_geometria = preferir_esquerda
                if (topologia_verde_ativa
                        and observacao_topologica.junction_image is not None
                        and direcao_marcada in ("left", "right")):
                    faixa_transversal_y_frame = float(
                        observacao_topologica.junction_image[1])
                else:
                    faixa_transversal_y_frame = altura_faixa_transversal(
                        black_image,
                        direcao_marcada,
                    )
                blackline, black_line_crop = determine_correct_line(
                    contours_blk,
                    preferir_esquerda=preferir_esquerda_geometria,
                    turn_direction=direcao_geometria,
                )
                area_linha_frame = float(cv2.contourArea(blackline))
                line_size.value = area_linha_frame

                # Calcula a geometria do gap.
                if line_status.value == "gap_detected":
                    publish_gap_geometry(blackline, cv2_img if debug else None)
                else:
                    reset_gap_values()

                # Calcula o ângulo de correção.
                last_bottom_point_x = float(get_time_average(time_last_bottom_point_x, .15))
                last_average_line_point = float(get_time_average(time_last_average_line_point, .15))

                line_angle.value, poi, bottom_point = calculate_angle(
                    blackline, black_line_crop,
                    float(get_time_average(time_line_angle, .3)),
                    direcao_geometria,
                    last_bottom_point_x,
                    last_average_line_point,
                    preferir_esquerda=preferir_esquerda_geometria,
                    preferir_reto=(
                        intersecao_sem_verde
                        or evento_verde.decision == GreenDecision.PENDING
                    ),
                )
                angulo_frame = float(line_angle.value)
                line_angle_y.value = int(poi[1])
                ponto_alvo_x_frame = float(poi[0])
                ponto_alvo_y_frame = float(poi[1])

                ponto_futuro = extrair_ponto_futuro(
                    blackline,
                    mascara_linha=black_image,
                    origem_x=bottom_point[0],
                )
                ponto_futuro_x_frame = ponto_futuro.x
                ponto_futuro_y_frame = ponto_futuro.y
                ponto_futuro_valido_frame = ponto_futuro.valido
                if intersecao_sem_verde:
                    # A busca distante pode cair no ramo lateral da cruz.
                    # Neste caso o alvo local central e a observacao segura.
                    ponto_futuro_valido_frame = False



                time_line_angle = add_time_value(time_line_angle, line_angle.value)
                time_last_bottom_point_x = add_time_value(time_last_bottom_point_x, bottom_point[0])

                # Projeta o vetor da parte inferior até o topo da imagem.
                if bottom_point[0] != poi[0] and bottom_point[1] != poi[1]:
                    slope = (bottom_point[1] - poi[1]) / (bottom_point[0] - poi[0])
                    x = min(max(poi[0] + (0 - poi[1]) / slope, 0), camera_x)
                else:
                    x = poi[0]

                time_last_average_line_point = add_time_value(time_last_average_line_point, x)

                bottom_y = bottom_point[1]
                ponto_inferior_x_frame = float(bottom_point[0])
                ponto_inferior_y_frame = float(bottom_point[1])

                # Publica os valores usados pelo controle.
                last_bottom_point.value = bottom_point[0]
                last_bottom_point_y.value = bottom_point[1]
                if debug:
                    cv2.drawContours(cv2_img, [blackline], -1, (255, 0, 0), 2)
                    cv2.circle(cv2_img, (int(last_average_line_point), 0), 5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (int(poi[0]), int(poi[1])), 5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (int(bottom_point[0]), int(bottom_point[1])), 5, (255, 255, 0), 1, cv2.LINE_AA)
                    if ponto_futuro.valido:
                        alvo_futuro = (
                            int(round(ponto_futuro.x)),
                            int(round(ponto_futuro.y)),
                        )
                        cv2.line(
                            cv2_img,
                            (camera_x // 2, camera_y - 1),
                            alvo_futuro,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            cv2_img, alvo_futuro, 6,
                            (0, 165, 255), 2, cv2.LINE_AA)
                    if faixa_transversal_y_frame >= 0.:
                        y_transversal = int(round(faixa_transversal_y_frame))
                        cv2.line(
                            cv2_img,
                            (0, y_transversal),
                            (camera_x - 1, y_transversal),
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.circle(cv2_img, (camera_x // 2, camera_y - 4),
                               5, (255, 0, 0), -1, cv2.LINE_AA)

            else:
                linha_detectada_frame = False
                line_detected.value = False
                line_angle.value = 0
                line_size.value = 0
                last_bottom_point_y.value = 0
                line_angle_y.value = -1
                reset_gap_values()

            # O modelo da entrada só roda na missão. O primeiro candidato usa
            # o mesmo frame em que a linha preta foi encontrada e centralizada;
            # depois de parar, a prata pode naturalmente cobrir esse fim.
            linha_alinhada = (
                linha_detectada_frame
                and abs(angulo_frame) <= config.ENTRY_LINE_MAX_ANGLE
                and abs(ponto_inferior_x_frame - camera_x / 2)
                <= config.ENTRY_LINE_MAX_BOTTOM_ERROR_PX
            )
            if linha_alinhada:
                ultimo_alinhamento_entrada = frame_captured_at
            # A faixa prata naturalmente tapa/termina a linha preta. Assim,
            # o modelo recebe até 0,5 s após o último alinhamento real, mas
            # nunca quando o robô já começou uma correção de linha perdida.
            entrada_alinhada = (
                frame_captured_at - ultimo_alinhamento_entrada
                <= config.ENTRY_ALIGNMENT_HOLD_S
            )
            update_entry_silver(
                entry_gate, cv2_img, frame_captured_at,
                line_aligned=entrada_alinhada,
                black_mask=entry_black_mask,
                ramp_black_mask=entry_ramp_black_mask)

            if topologia_verde_ativa:
                confirmador_evento_verde.note_rearm_frame(
                    junction_visible=_juncao_presente_para_saida(
                        observacao_topologica),
                    exit_line_stable=bool(
                        linha_detectada_frame
                        and abs(ponto_inferior_x_frame - camera_x / 2)
                        <= config.GREEN_TURN_CENTER_TOLERANCE_PX
                    ),
                    timestamp=instante_evento,
                )
                if (
                    decisao_em_cooldown_topologico > 0
                    and not confirmador_evento_verde.in_cooldown
                ):
                    green_rearmed_decision_id.value = max(
                        int(green_rearmed_decision_id.value),
                        decisao_em_cooldown_topologico,
                    )
                    decisao_em_cooldown_topologico = 0
            publicar_observacao_intersecao(evento_verde)

            processamento_ms = (
                time.perf_counter() - inicio_processamento
            ) * 1000.
            publicar_resultado_visao_rapida(
                sequencia=sequencia_topologia,
                publicado_em=time.monotonic(),
                processamento_ms=processamento_ms,
                linha_detectada=linha_detectada_frame,
                linha_a_frente=linha_a_frente_frame,
                angulo=angulo_frame,
                ponto_inferior_x=ponto_inferior_x_frame,
                ponto_inferior_y=ponto_inferior_y_frame,
                ponto_alvo_x=ponto_alvo_x_frame,
                ponto_alvo_y=ponto_alvo_y_frame,
                area_linha=area_linha_frame,
                candidato_verde=candidato_verde_frame,
                candidato_vermelho=candidato_vermelho_frame,
                ponto_futuro_x=ponto_futuro_x_frame,
                ponto_futuro_y=ponto_futuro_y_frame,
                ponto_futuro_valido=ponto_futuro_valido_frame,
                faixa_transversal_y=faixa_transversal_y_frame,
                juncao_topologica_visivel=(
                    _juncao_presente_para_saida(observacao_topologica)
                ),
                locked_branch_token=ramo_travado.token,
                locked_branch_valid=ramo_travado.valid,
                locked_branch_bottom_x=ramo_travado_x_cru,
                locked_branch_bottom_y=ramo_travado_y_cru,
            )

            if gravador is not None:
                try:
                    comando_motores = ler_comando_motores()
                    gravador.gravar(frame_cru_gravacao, {
                        "sequence": sequencia_topologia,
                        "timestamp": instante_evento,
                        "processing_ms": processamento_ms,
                        "raw_topology": observacao_topologica.decision.name,
                        "topology_reason": observacao_topologica.reason,
                        "junction_id": evento_verde.junction_id,
                        "decision_id": evento_verde.decision_id,
                        "decision": evento_verde.decision.name,
                        "confidence": evento_verde.confidence,
                        "coordinate_frames": {
                            "entry_tangent": (
                                "ground_xy_right_forward"
                                if calibracao_wide is not None
                                else "raw_pixel_xy_right_forward"
                            ),
                            "junction": (
                                "rectified_pixels"
                                if calibracao_wide is not None
                                else "raw_frame_pixels"
                            ),
                            "target_branch": "raw_frame_pixels",
                        },
                        "entry_tangent": evento_verde.entry_tangent,
                        "junction": evento_verde.junction_center,
                        "target_branch": evento_verde.target_branch,
                        "target_branch_token": (
                            evento_verde.target_branch_token),
                        "locked_branch_token": ramo_travado.token,
                        "locked_branch_valid": ramo_travado.valid,
                        "locked_branch_bottom_x": ramo_travado_x_cru,
                        "locked_branch_bottom_y": ramo_travado_y_cru,
                        "locked_branch_points": ramo_travado.tracked_points,
                        "ready_to_turn": evento_verde.ready_to_turn,
                        "junction_visible": evento_verde.junction_visible,
                        "geometry_predicted": evento_verde.geometry_predicted,
                        "marker_ids": evento_verde.marker_ids,
                        "control_state": int(green_control_state.value),
                        "locked_decision": int(green_locked_decision.value),
                        "motor_command_id": comando_motores.command_id,
                        "motor_command_timestamp": (
                            comando_motores.publicado_em),
                        "motor_left_pwm": comando_motores.esquerda,
                        "motor_right_pwm": comando_motores.direita,
                        "yaw": float(green_control_yaw.value),
                        "line_detected": linha_detectada_frame,
                        "line_angle": angulo_frame,
                        "bottom_point": (
                            ponto_inferior_x_frame,
                            ponto_inferior_y_frame,
                        ),
                    })
                except (OSError, ValueError, cv2.error) as err:
                    print(f"[visao] gravacao de diagnostico encerrada: {err}")
                    gravador.close()
                    gravador = None

            if not vision_ready.value:
                vision_ready.value = True
                print("[visão] primeiro frame processado — pipeline ativo")

            # FPS
            counter += 1
            if time.perf_counter() - fps_time > 1:
                fps = int(counter / (time.perf_counter() - fps_time))
                fps_time = time.perf_counter()
                counter = 0

            if debug:
                if calibracao_wide is not None:
                    vista_topologia = calibracao_wide.rectify(
                        frame_cru_gravacao)
                    draw_topology_debug(
                        vista_topologia, observacao_topologica)
                    largura_inset = 168
                    altura_inset = round(
                        largura_inset * camera_y / camera_x)
                    inset = cv2.resize(
                        vista_topologia, (largura_inset, altura_inset))
                    x_inset = camera_x - largura_inset
                    y_inset = 45
                    cv2_img[
                        y_inset:y_inset + altura_inset,
                        x_inset:camera_x,
                    ] = inset
                    cv2.rectangle(
                        cv2_img, (x_inset, y_inset),
                        (camera_x - 1, y_inset + altura_inset - 1),
                        (255, 255, 0), 1)
                    cv2.putText(
                        cv2_img, "TOPOLOGIA RETIFICADA",
                        (x_inset + 2, y_inset + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, .30,
                        (255, 255, 0), 1, cv2.LINE_AA)
                else:
                    draw_topology_debug(cv2_img, observacao_topologica)

                if entry_gate is not None and entry_armed.value:
                    entrada = entry_gate.last_detection
                    if entrada is not None:
                        x, y, w, h = entrada.bbox
                        cv2.rectangle(
                            cv2_img,
                            (int(x), int(y)),
                            (int(x + w - 1), int(y + h - 1)),
                            (255, 255, 0),
                            2,
                        )
                    motivo_entrada = (
                        entry_gate.last_reason or "candidata")
                    confianca_entrada = (
                        entrada.confidence if entrada is not None else 0.)
                    confianca_bruta = getattr(
                        getattr(entry_gate, "model", None),
                        "last_confidence",
                        None,
                    )
                    if entrada is None and confianca_bruta is not None:
                        confianca_entrada = confianca_bruta
                    cv2.putText(
                        cv2_img,
                        f"ONNX PRATA {entry_gate.votes}/"
                        f"{config.ENTRY_SILVER_VOTE_WINDOW} "
                        f"conf={confianca_entrada:.2f}/"
                        f"{config.ENTRY_MODEL_MIN_CONFIDENCE:.2f} "
                        f"{motivo_entrada}",
                        (5, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .35,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                nomes_controle = ("TRACK", "CORNER", "LOST", "SPECIAL")
                indice_controle = int(steering_state.value)
                nome_controle = (
                    nomes_controle[indice_controle]
                    if 0 <= indice_controle < len(nomes_controle)
                    else "?"
                )
                comando_debug = ler_comando_motores()
                cv2.putText(cv2_img, f"{fps} fps  ang={line_angle.value}  {line_status.value}",
                            (5, camera_y - 8), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.putText(
                    cv2_img,
                    f"ctl={nome_controle} c={steering_correction.value:+.2f} "
                    f"cmd={comando_debug.command_id} "
                    f"L={comando_debug.esquerda} R={comando_debug.direita}",
                    (5, camera_y - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    cv2_img,
                    f"lateral={steering_lateral_error.value:+.2f} "
                    f"rumo={steering_heading.value:+.0f}deg",
                    (5, camera_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    cv2_img,
                    f"verde raw={direcao_verde_bruta} "
                    f"evento={evento_verde.decision.name} "
                    f"id={evento_verde.decision_id} "
                    f"conf={evento_verde.confidence:.2f}",
                    (5, camera_y - 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (0, 200, 0),
                    1,
                    cv2.LINE_AA,
                )
                estados_verde = (
                    "FOLLOW", "OBSERVE", "COMMITTED", "APPROACH",
                    "TURNING", "REACQUIRE", "COOLDOWN", "FAULT_STOP",
                )
                indice_estado_verde = int(green_control_state.value)
                nome_estado_verde = (
                    estados_verde[indice_estado_verde]
                    if 0 <= indice_estado_verde < len(estados_verde)
                    else "?"
                )
                try:
                    nome_travado = GreenDecision(
                        int(green_locked_decision.value)).name
                except ValueError:
                    nome_travado = "?"
                cv2.putText(
                    cv2_img,
                    f"fsm={nome_estado_verde} lock={nome_travado} "
                    f"ready={int(evento_verde.ready_to_turn)} "
                    f"yaw={green_control_yaw.value:+.1f}",
                    (5, camera_y - 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (0, 200, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    cv2_img,
                    f"ramo token={ramo_travado.token}/"
                    f"{evento_verde.target_branch_token} "
                    f"ok={int(ramo_travado.valid)} "
                    f"pts={ramo_travado.tracked_points}",
                    (5, camera_y - 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (255, 140, 0),
                    1,
                    cv2.LINE_AA,
                )
                if ramo_travado.valid:
                    x_ramo = int(round(ramo_travado_x_cru))
                    if 0 <= x_ramo < camera_x:
                        cv2.circle(
                            cv2_img,
                            (x_ramo, int(round(ramo_travado_y_cru))),
                            7,
                            (255, 140, 0),
                            2,
                            cv2.LINE_AA,
                        )
                cv2.putText(cv2_img, str(status.value), (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.putText(
                    cv2_img,
                    f"proc={processamento_ms:.1f}ms  "
                    f"captura={camera.capture_fps:.0f}fps",
                    (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (0, 255, 255),
                    1,
                )
                shm_array[:] = cv2_img

    finally:
        green_calibration_ready.value = False
        green_topology_junction_visible.value = False
        publicar_observacao_intersecao(empty_observation())
        if entry_gate is not None:
            entry_gate.close()
        if gravador is not None:
            gravador.close()
        camera.close()
        if shm is not None:
            shm.close()
