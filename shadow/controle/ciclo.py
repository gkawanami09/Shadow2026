"""Ciclo principal de decisões e movimentos do segue-linha."""

import math
import time

import config
from config import (CONTROL_MAX_ITERATIONS, GAP_AVOID_RETREAT_TIME, GAP_AVOID_SPEED,
                    GAP_ENABLED,
                    GAP_AVOID_TIMEOUT, GAP_MIN_LINE_SIZE_RETREAT,
                    GAP_MISSING_CONFIRM_TIME, GAP_REJECT_COOLDOWN,
                    GREEN_APPROACH_SPEED, GREEN_APPROACH_TIME,
                    GREEN_TURN_BLIND_TIME, GREEN_TURN_CENTER_TOLERANCE_PX,
                    GREEN_TURN_SPEED, GREEN_TURN_TIMEOUT, LINE_FOLLOW_SPEED,
                    MIN_LINE_SIZE_DEFAULT, VISION_READY_TIMEOUT,
                    camera_x, camera_y)
from controle.orientacao_gap import drive_back_until_line, orientate_gap
from controle.parada_obstaculo import (
    MonitorObstaculo,
    desviar_obstaculo,
)
from controle.parada_vermelho import stop_for_red
from controle.manobra_verde import (alinhamento_verde_pode_concluir,
                                    correcao_aproximacao,
                                    correcao_reaquisicao_verde,
                                    correcao_ramo_reto,
                                    juncao_topologica_realmente_ausente,
                                    ramo_marcado_visto_pela_camera,
                                    ramo_travado_recente,
                                    saida_topologica_real_estavel)
from controle.estado_verde import (GreenDecision, GreenManeuverFSM,
                                    GreenManeuverState, SignedYawTracker,
                                    calibracao_permite_motores)
from controle.velocidade import get_speed
from controle.velocidade_adaptativa import ControladorVelocidadeAdaptativa
from controle.direcao import (init_steering, set_motion_guard,
                              set_motion_observer, sleep_steering, steer,
                              steer_line)
from controle.retorno import turn_around
from controle.seguidor_linha import CORNER, LOST, ControladorSegueLinha
from comunicacao_serial.arduino import Arduino
from shared.dados_compartilhados import (ENTRY_SILVER_BLACK_FOLLOW,
                               ENTRY_SILVER_IDLE, ENTRY_SILVER_VALIDATING,
                               entry_armed, entry_silver_confirmed,
                               entry_silver_detected, entry_silver_reason,
                               entry_silver_state, entry_silver_votes,
                               green_candidate,
                               green_calibration_ready,
                               green_control_state,
                               green_control_yaw,
                               green_decision_consumed_id,
                               green_fault_stop,
                               green_locked_decision,
                               green_rearmed_decision_id,
                               green_turn_target,
                               last_bottom_point,
                               last_bottom_point_y,
                               ler_resultado_visao_rapida,
                               line_ahead, line_angle,
                               line_detected,
                               line_size, line_status, min_line_size,
                               ler_observacao_intersecao,
                               publicar_comando_motores,
                               mission_mode,
                               preferencia_linha_esquerda,
                               red_candidate, red_detected, red_finished,
                               rescue_requested, status, terminate,
                               STEERING_CORNER, STEERING_LOST,
                               STEERING_SPECIAL, STEERING_TRACK,
                               steering_correction, steering_heading,
                               steering_lateral_error, steering_left_pwm,
                               steering_right_pwm, steering_state,
                               timer, turn_dir,
                               vision_ready)


def _enter_rescue_zone(arduino):
    """Entrega a câmera/serial ao resgate, que faz seu avanço de 1 segundo."""
    if steer() is False or not arduino.connected:
        return False
    # O LED só pode ser apagado enquanto esta serial ainda existe. O processo
    # de resgate reafirma o comando assim que abre a serial dele.
    if arduino.led("APAGADO") is False or not arduino.connected:
        return False
    entry_armed.value = False
    print("[controle] entrada confirmada; PARAR e LED APAGADO — "
          "resgate fará o avanço de 1 s")
    return True


def _enter_rescue_after_no_black(arduino):
    """Entrega a serial ao resgate apos a ausencia de preto confirmada."""
    if terminate.value:
        status.value = 'Entrada do resgate cancelada - encerrando controle'
        return False
    if steer() is False:
        status.value = 'Falha ao parar na entrada do resgate - tentando novamente'
        return False
    if not arduino.connected:
        status.value = 'Arduino desconectado na entrada do resgate'
        return False
    status.value = 'Linha preta ausente - entrando no resgate'
    if arduino.led("APAGADO") is False:
        status.value = 'Falha ao apagar LED na entrada do resgate - tentando novamente'
        return False
    if not arduino.connected:
        status.value = 'Arduino desconectado ao apagar LED da entrada'
        return False
    entry_armed.value = False
    _reset_entry_silver("entrada por ausencia de preto")
    print(
        "[controle] linha preta ausente por "
        f"{config.ENTRY_NO_BLACK_RESCUE_DELAY_S:.1f} s; PARAR e LED APAGADO "
        "— resgate avançará 1 s antes dos giros")
    return True


def _reset_entry_silver(reason):
    """Descarta qualquer candidatura de prata antes de mudar de percurso."""
    entry_silver_detected.value = False
    entry_silver_confirmed.value = False
    entry_silver_votes.value = 0
    entry_silver_reason.value = reason
    entry_silver_state.value = ENTRY_SILVER_IDLE


def _publicar_pwm_motores(esquerda, direita):
    publicar_comando_motores(esquerda, direita)
    steering_left_pwm.value = int(esquerda)
    steering_right_pwm.value = int(direita)


def control_loop():
    try:
        arduino = Arduino()
    except RuntimeError as erro:
        status.value = 'Aguardando Arduino para reiniciar a missao'
        print(f"[controle] Arduino indisponivel: {erro}")
        return
    init_steering(arduino)
    set_motion_observer(_publicar_pwm_motores)
    set_motion_guard(lambda: calibracao_permite_motores(
        obrigatoria=config.GREEN_WIDE_CALIBRATION_REQUIRED,
        pronta=green_calibration_ready.value,
    ))
    steer()  # motores parados desde o inicio
    # Nenhuma sessao reconecta em movimento. Se o Uno cair, seu watchdog para
    # os motores e esta execucao entra em falha; uma nova conexao exige um
    # reinicio explicito, inclusive em ``main.py`` standalone.
    arduino.travar_sessao()
    epoca_serial_sessao = arduino.connection_epoch

    last_turn_dir = "l"

    # espera a visao publicar o primeiro frame processado
    status.value = "Preparando visao - motores parados"
    print("[controle] aguardando primeiro frame da visao; motores parados")
    wait_start = time.perf_counter()
    while not vision_ready.value and not terminate.value:
        if not arduino.connected:
            status.value = 'Arduino desconectado - aguardando reinicio'
            print("[controle] Arduino desconectado durante a espera da visao")
            try:
                steer()
            finally:
                arduino.close()
            return
        if time.perf_counter() - wait_start > VISION_READY_TIMEOUT:
            status.value = 'Camera indisponivel - aguardando reinicio'
            print("[controle] visao nao ficou pronta em "
                  f"{VISION_READY_TIMEOUT} s; reiniciando sem mover")
            try:
                steer()
            finally:
                arduino.close()
            return
        arduino.refresh()
        time.sleep(.05)

    if terminate.value:
        try:
            steer()
        finally:
            arduino.close()
        return

    if not calibracao_permite_motores(
        obrigatoria=config.GREEN_WIDE_CALIBRATION_REQUIRED,
        pronta=green_calibration_ready.value,
    ):
        green_fault_stop.value = True
        green_control_state.value = int(GreenManeuverState.FAULT_STOP)
        green_locked_decision.value = int(GreenDecision.NONE)
        status.value = (
            'Calibracao wide obrigatoria ausente - motores bloqueados')
        arduino.led("APAGADO")
        print(
            "[controle] calibracao wide obrigatoria ausente/incompativel; "
            "LED APAGADO e motores bloqueados. Use --vision-only para "
            "diagnostico ou execute tools/calibrar_camera_wide.py"
        )
        while not terminate.value and arduino.connected:
            steer()
            arduino.refresh(fail_closed=True)
            time.sleep(.05)
        try:
            steer()
        finally:
            arduino.close()
        return

    # O LED passa a significar que camera e controle estao realmente prontos.
    # Assim nao existe mais uma espera silenciosa depois de ele acender.
    arduino.led("ACESO")
    print("[controle] LED ACESO: segue-linha pronto")
    line_status.value = "line_detected"
    status.value = "Shadow2026 pronto — aguardando linha"
    print("Shadow2026 ready — awaiting line")

    iteration_limit_time = time.perf_counter()
    max_iterations = CONTROL_MAX_ITERATIONS
    line_missing_since = None
    black_line_seen = False
    no_black_since = None
    gap_retry_after = 0.0
    controlador_linha = ControladorSegueLinha()
    green_direction = None
    green_turn_started = None
    green_target_seen = False
    green_transversal_frames = 0
    green_last_signed_error = None
    green_mpu_last_yaw = None
    green_mpu_turn_origin = None
    green_mpu_next_query = 0.
    green_fsm = GreenManeuverFSM()
    green_yaw_tracker = None
    green_yaw_progress = None
    green_mpu_last_timestamp = None
    green_mpu_last_generation = 0
    green_mpu_ativo = bool(
        config.GREEN_MPU_ENABLED
        and type(config.GREEN_MPU_POSITIVE_IS_RIGHT) is bool
    )
    if config.GREEN_MPU_ENABLED and not green_mpu_ativo:
        print(
            "[controle] MPU verde sem autoridade: configure "
            "GREEN_MPU_POSITIVE_IS_RIGHT após conferir o sinal físico do yaw"
        )
    green_exit_stable_frames = 0
    green_ready_last_sequence = -1
    green_exit_last_sequence = -1
    green_center_last_sequence = -1
    calibration_warning_printed = False
    monitor_obstaculo = MonitorObstaculo()
    velocidade_adaptativa = (
        ControladorVelocidadeAdaptativa()
        if config.RETA_RAPIDA_HABILITADA
        else None
    )
    modo_rapido_anterior = False
    obstaculo_retry_after = 0.
    preferencia_linha_esquerda.value = False
    green_turn_target.value = 0
    green_control_state.value = int(GreenManeuverState.FOLLOW)
    green_locked_decision.value = int(GreenDecision.NONE)
    green_fault_stop.value = False
    green_control_yaw.value = float("nan")
    preferencia_esquerda_inicio = 0.
    preferencia_esquerda_alinhada_desde = None
    entry_rearm_after = None
    estado_rampa = "PLANO"
    proxima_consulta_rampa = 0.

    try:
        while not terminate.value:
            if (not arduino.connected
                    or arduino.connection_epoch != epoca_serial_sessao):
                if not green_fsm.stopped:
                    green_fsm.fault(
                        "sessao serial perdida durante o controle",
                        now=time.monotonic(),
                    )
                green_fault_stop.value = True
                green_control_state.value = int(green_fsm.state)
                status.value = 'Arduino desconectado - aguardando reinicio'
                print(
                    "[controle] sessao serial perdida; "
                    "FAULT_STOP e encerrando sem reconectar"
                )
                steer()
                return

            # A calibracao e uma permissao global de movimento, nao apenas um
            # teste do bloco de segue-linha. A mesma trava tambem vive em
            # ``direcao`` e interrompe sleeps bloqueantes de gap/180 em ate
            # 50 ms; aqui persistimos o FAULT_STOP e apagamos o LED.
            if not calibracao_permite_motores(
                obrigatoria=config.GREEN_WIDE_CALIBRATION_REQUIRED,
                pronta=green_calibration_ready.value,
            ):
                agora_falha = time.monotonic()
                if not green_fsm.stopped:
                    green_fsm.fault(
                        "calibracao wide obrigatoria perdida em runtime",
                        now=agora_falha,
                    )
                green_fault_stop.value = True
                green_control_state.value = int(
                    GreenManeuverState.FAULT_STOP)
                green_locked_decision.value = int(
                    green_fsm.locked_direction)
                status.value = (
                    'Calibracao wide perdida - FAULT_STOP, motores parados')
                if not calibration_warning_printed:
                    calibration_warning_printed = True
                    arduino.led("APAGADO")
                    print(
                        "[controle] calibracao wide perdida em runtime; "
                        "LED APAGADO e motores bloqueados ate reiniciar")
                steer()
                arduino.refresh(fail_closed=True)
                time.sleep(.02)
                continue

            if green_fsm.stopped:
                # FAULT_STOP e persistente por projeto: somente reiniciar a
                # sessao pode devolver autoridade ao seguidor comum.
                green_fault_stop.value = True
                green_control_state.value = int(
                    GreenManeuverState.FAULT_STOP)
                status.value = (
                    'Verde FAULT_STOP - motores parados: '
                    + green_fsm.fault_reason)
                steer()
                arduino.refresh(fail_closed=True)
                time.sleep(.02)
                continue

            observacao_prioritaria = ler_observacao_intersecao()
            observacao_prioritaria_recente = bool(
                time.monotonic() - observacao_prioritaria.timestamp
                <= config.LINE_MAX_FRAME_AGE_S
            )
            manobra_verde_em_observacao = bool(
                observacao_prioritaria_recente
                and observacao_prioritaria.decision != GreenDecision.NONE
            )

            # A inclinacao vem do MPU no Arduino. A consulta e nao bloqueante:
            # enquanto uma resposta chega, o segue-linha continua enviando os
            # comandos normais e alimentando o watchdog do Uno.
            if (config.RAMPA_HABILITADA
                    and hasattr(arduino, "iniciar_rampa")
                    and hasattr(arduino, "poll_rampa")):
                agora_rampa = time.monotonic()
                concluida, leitura_rampa = arduino.poll_rampa()
                if concluida:
                    novo_estado = "PLANO"
                    novo_angulo = 0.
                    if leitura_rampa is not None:
                        novo_estado, novo_angulo = leitura_rampa
                    if novo_estado != estado_rampa:
                        print(
                            "[controle] rampa "
                            f"{estado_rampa.lower()} -> {novo_estado.lower()} "
                            f"({novo_angulo:+.1f} graus)")
                    estado_rampa = novo_estado
                if agora_rampa >= proxima_consulta_rampa:
                    arduino.iniciar_rampa()
                    proxima_consulta_rampa = (
                        agora_rampa + config.RAMPA_CONSULTA_INTERVALO_S)

            # A confirmacao tem prioridade sobre qualquer outra manobra: este
            # processo ainda e o dono seguro da serial e precisa parar antes
            # de o supervisor abrir o resgate.
            if (config.ENTRY_SILVER_ENABLED
                    and mission_mode.value and entry_armed.value
                    and entry_silver_confirmed.value):
                status.value = 'Faixa prata confirmada — entrando na sala'
                print("[controle] faixa PRATA confirmada; entrando na sala")
                if _enter_rescue_zone(arduino):
                    rescue_requested.value = True
                else:
                    status.value = (
                        'Falha serial na entrada - reiniciando missao')
                break

            # Primeiro positivo prata: pare sem bloquear a visao. Ela continua
            # recebendo frames por um segundo para procurar preto alem da
            # caixa antes de liberar a entrada da sala.
            if (config.ENTRY_SILVER_ENABLED
                    and mission_mode.value and entry_armed.value
                    and entry_silver_state.value == ENTRY_SILVER_VALIDATING):
                status.value = 'Prata candidata — parado validando'
                steer()
                sleep_steering(.02)
                continue

            # A confirmacao de preto depois da prata contradiz uma manobra de
            # gap: ha linha de novo. Volte ao estado comum e mantenha o
            # segue-linha enquanto o Gate bloqueia somente nova prata.
            if (config.ENTRY_SILVER_ENABLED
                    and mission_mode.value and entry_armed.value
                    and entry_silver_state.value == ENTRY_SILVER_BLACK_FOLLOW
                    and line_status.value in {"gap_detected", "gap_avoid"}):
                line_status.value = "line_detected"
                min_line_size.value = MIN_LINE_SIZE_DEFAULT

            # A preferência pós-obstáculo não gira o robô sozinha. A visão
            # apenas desempata contornos transversais para a esquerda e o
            # segue-linha proporcional executa a correção normal.
            if preferencia_linha_esquerda.value:
                agora_preferencia = time.monotonic()
                alinhada = (
                    agora_preferencia - preferencia_esquerda_inicio
                    >= config.OBSTACLE_LEFT_PREFERENCE_MIN_TIME_S
                    and line_detected.value
                    and abs(line_angle.value)
                    <= config.OBSTACLE_LEFT_PREFERENCE_MAX_ANGLE
                    and abs(last_bottom_point.value - camera_x / 2)
                    <= config.OBSTACLE_LEFT_PREFERENCE_BOTTOM_PX
                )
                if alinhada:
                    if preferencia_esquerda_alinhada_desde is None:
                        preferencia_esquerda_alinhada_desde = agora_preferencia
                    elif (
                        agora_preferencia
                        - preferencia_esquerda_alinhada_desde
                        >= config.OBSTACLE_LEFT_PREFERENCE_CONFIRM_TIME_S
                    ):
                        preferencia_linha_esquerda.value = False
                else:
                    preferencia_esquerda_alinhada_desde = None

                if (
                    agora_preferencia - preferencia_esquerda_inicio
                    >= config.OBSTACLE_LEFT_PREFERENCE_MAX_TIME_S
                ):
                    preferencia_linha_esquerda.value = False

                if not preferencia_linha_esquerda.value:
                    obstaculo_retry_after = max(
                        obstaculo_retry_after,
                        agora_preferencia + config.OBSTACLE_RETRY_COOLDOWN_S,
                    )
                    status.value = 'Preferência esquerda concluída'
                    print(
                        "[controle] preferência esquerda concluída; "
                        "segue-linha normal")

            # Segurança frontal independente da visão. Duas de três leituras
            # ultrassônicas precisam confirmar até 5 cm. Depois disso a
            # confirmação desloca o robô para a esquerda, avança pelo
            # obstáculo e retorna a mesma distância à direita. Ao terminar,
            # o laço volta diretamente ao segue-linha normal.
            if (
                config.OBSTACLE_STOP_ENABLED
                and green_fsm.state == GreenManeuverState.FOLLOW
                and not manobra_verde_em_observacao
                and not preferencia_linha_esquerda.value
                and time.monotonic() >= obstaculo_retry_after
                and monitor_obstaculo.atualizar(arduino)
            ):
                distancia_cm = (
                    monitor_obstaculo.distancia_confirmada_mm / 10.0)
                status.value = (
                    f'Obstáculo a {distancia_cm:.1f} cm — '
                    'executando desvio')
                print(
                    "[controle] obstáculo confirmado a "
                    f"{distancia_cm:.1f} cm; esquerda por "
                    f"{config.OBSTACLE_LATERAL_TIME_S:.1f} s e frente por "
                    f"{config.OBSTACLE_FORWARD_TIME_S:.1f} s; retorno "
                    "lateral à direita por "
                    f"{config.OBSTACLE_RETURN_LATERAL_TIME_S:.1f} s")
                try:
                    desviar_obstaculo(
                        arduino,
                        deve_encerrar=lambda: (
                            terminate.value
                            or not calibracao_permite_motores(
                                obrigatoria=(
                                    config.GREEN_WIDE_CALIBRATION_REQUIRED),
                                pronta=green_calibration_ready.value,
                            )
                        ),
                    )
                    if terminate.value:
                        break

                    status.value = 'Desvio concluído — segue-linha normal'
                    print(
                        "[controle] retorno lateral concluído; "
                        "retomando segue-linha normal")
                except RuntimeError as erro:
                    status.value = 'Falha no desvio do obstáculo — PARADO'
                    print(f"[controle] falha no desvio do obstáculo: {erro}")
                    while not terminate.value:
                        arduino.refresh(fail_closed=True)
                        time.sleep(.05)
                    break

                # Descarta o eco antigo e devolve o movimento ao segue-linha.
                arduino.cancelar_ultrassom()
                monitor_obstaculo.reiniciar()
                obstaculo_retry_after = (
                    time.monotonic()
                    + config.OBSTACLE_RETRY_COOLDOWN_S
                )
                line_status.value = "line_detected"
                line_missing_since = None
                controlador_linha.reset()
                green_direction = None
                green_turn_started = None
                green_target_seen = False
                green_turn_target.value = 0
                status.value = 'Seguindo linha'
                print("[controle] retomando segue-linha normal")
                continue

            # Faixa prata de entrada. Só existe no modo de missão completa;
            # rodando `shadow/main.py` sozinho este bloco nunca é atingido.
            # O 180 e uma manobra bloqueante: a visao continua recebendo
            # frames enquanto o controle gira e da a re de retomada. So
            # rearmamos a entrada depois de um trecho novo de segue-linha;
            # assim votos/candidatos vistos durante a manobra nao podem
            # solicitar o resgate quando ela retorna.
            if (
                mission_mode.value
                and entry_rearm_after is not None
                and time.monotonic() >= entry_rearm_after
                and turn_dir.value == "straight"
                and line_detected.value
            ):
                _reset_entry_silver("prata reiniciada apos giro de 180")
                entry_armed.value = True
                entry_rearm_after = None
                print(
                    "[controle] 180 concluido; entrada prata rearmada "
                    "com votos zerados")

            # Estado normal do segue-linha.
            if line_status.value == "line_detected":
                now = time.monotonic()

                if line_detected.value:
                    # Nunca inicie o teste apenas porque a camera acabou de
                    # ligar sem a linha no campo. Primeiro a linha precisa ter
                    # sido vista nesta fase do percurso.
                    black_line_seen = True
                    no_black_since = None

                test_no_black_active = (
                    config.ENTRY_NO_BLACK_RESCUE_TEST_ENABLED
                    and mission_mode.value
                    and entry_armed.value
                )
                can_count_no_black = (
                    test_no_black_active
                    and black_line_seen
                    and not line_detected.value
                    and not line_ahead.value
                    and turn_dir.value == "straight"
                    and not preferencia_linha_esquerda.value
                    and not green_candidate.value
                    and not red_candidate.value
                    and not red_detected.value
                    and green_direction is None
                )
                if can_count_no_black:
                    if no_black_since is None:
                        no_black_since = now
                        print(
                            "[controle] sem preto em reta; validando "
                            f"{config.ENTRY_NO_BLACK_RESCUE_DELAY_S:.1f} s")
                    elif (
                        now - no_black_since
                        >= config.ENTRY_NO_BLACK_RESCUE_DELAY_S
                    ):
                        if _enter_rescue_after_no_black(arduino):
                            rescue_requested.value = True
                            break
                        # O handoff ainda nao aconteceu: ``entry_armed``
                        # continua verdadeiro e o resgate nao pode assumir a
                        # serial. Mantenha os motores parados e tente de novo
                        # enquanto a sessao existir, em vez de encerrar o
                        # filho silenciosamente e reiniciar o segue-linha.
                        if arduino.connected and not terminate.value:
                            sleep_steering(.05)
                        continue
                else:
                    no_black_since = None

                # Durante este teste, ausencia de preto e o proprio gatilho
                # de entrada. Nao a transforme antes em uma manobra de gap,
                # que impediria a contagem de completar os tres segundos.
                gap_allowed = (
                    GAP_ENABLED
                    and not test_no_black_active
                    and green_fsm.state == GreenManeuverState.FOLLOW
                    and not green_candidate.value
                    and now >= gap_retry_after
                )
                if (gap_allowed and not line_detected.value
                        and not line_ahead.value):
                    if line_missing_since is None:
                        line_missing_since = now
                    elif now - line_missing_since >= GAP_MISSING_CONFIRM_TIME:
                        line_status.value = "gap_detected"
                        line_missing_since = None
                else:
                    line_missing_since = None

                if red_detected.value:
                    line_status.value = "stop"

            # Continua seguindo enquanto não muda de estado.
            if line_status.value == "line_detected":
                now = time.monotonic()
                observacao_verde = ler_observacao_intersecao()
                resultado_visao = ler_resultado_visao_rapida()
                # O deadline do estado atual precisa vencer antes que um
                # frame tardio possa promover OBSERVE->COMMITTED ou
                # APPROACH->TURNING e substituir esse deadline.
                if green_fsm.check_timeout(now=now):
                    green_fault_stop.value = True
                    green_control_state.value = int(green_fsm.state)
                    steer()
                    continue
                observacao_verde_recente = (
                    now - observacao_verde.timestamp
                    <= config.LINE_MAX_FRAME_AGE_S
                )
                if not green_calibration_ready.value:
                    if not calibration_warning_printed:
                        print(
                            "[controle] calibracao wide ausente/incompativel; "
                            "verde desarmado, segue-linha comum preservado"
                        )
                        calibration_warning_printed = True
                    if (config.GREEN_WIDE_CALIBRATION_REQUIRED
                            or green_fsm.state not in (
                                GreenManeuverState.FOLLOW,
                                GreenManeuverState.COOLDOWN,
                            )):
                        green_fsm.fault(
                            "calibracao wide obrigatoria perdida em runtime",
                            now=now,
                        )
                elif observacao_verde_recente:
                    evento_expirado_antes_do_controle = bool(
                        observacao_verde.committed
                        and green_fsm.state == GreenManeuverState.FOLLOW
                        and resultado_visao.sequencia
                        >= observacao_verde.sequence
                        and juncao_topologica_realmente_ausente(
                            resultado_visao, agora=now)
                    )
                    if evento_expirado_antes_do_controle:
                        green_fsm.fault(
                            "decisao verde expirou antes do controle",
                            now=now,
                        )
                        green_fault_stop.value = True
                        green_control_state.value = int(green_fsm.state)
                        status.value = (
                            "Decisao verde expirou - FAULT_STOP, "
                            "motores parados"
                        )
                        steer()
                        continue
                    else:
                        green_fsm.observe(
                            observacao_verde,
                            now=now,
                            observe_timeout_s=(
                                config.GREEN_TOPOLOGY_OBSERVE_TIMEOUT_S),
                        )
                    if (observacao_verde.decision == GreenDecision.NONE
                            and green_fsm.state
                            == GreenManeuverState.OBSERVE):
                        green_fsm.cancel_observation(now=now)

                if (green_fsm.state == GreenManeuverState.COMMITTED
                        and green_fsm.event is not None):
                    decisao = green_fsm.locked_direction
                    green_fsm.begin_approach(
                        now=now, timeout_s=GREEN_APPROACH_TIME)
                    green_direction = {
                        GreenDecision.LEFT: "left",
                        GreenDecision.RIGHT: "right",
                        GreenDecision.UTURN: "turn_around",
                    }.get(decisao)
                    green_turn_started = None
                    green_target_seen = False
                    green_transversal_frames = 0
                    green_ready_last_sequence = -1
                    green_exit_last_sequence = -1
                    green_center_last_sequence = -1
                    green_last_signed_error = None
                    green_mpu_last_yaw = None
                    green_mpu_turn_origin = None
                    green_mpu_last_timestamp = None
                    green_mpu_last_generation = 0
                    green_control_yaw.value = float("nan")
                    green_yaw_tracker = None
                    green_yaw_progress = None
                    green_mpu_next_query = now
                    if hasattr(arduino, "cancelar_mpu"):
                        arduino.cancelar_mpu()
                    green_turn_target.value = (
                        -1 if decisao == GreenDecision.LEFT else
                        1 if decisao == GreenDecision.RIGHT else
                        2 if decisao == GreenDecision.STRAIGHT else 0
                    )

                green_control_state.value = int(green_fsm.state)
                green_locked_decision.value = int(
                    green_fsm.locked_direction)
                direcao_visual = {
                    GreenDecision.LEFT: "left",
                    GreenDecision.RIGHT: "right",
                    GreenDecision.UTURN: "turn_around",
                }.get(green_fsm.locked_direction, "straight")

                status.value = (
                    'Seguindo Linha — preferência esquerda'
                    if preferencia_linha_esquerda.value
                    else 'Seguindo Linha'
                )

                # Mantem uma amostra recente durante a aproximacao. Quando o
                # tanque comeca ela vira a origem do giro, sem bloquear o
                # controle com um comando ``MPU ZERO``.
                if (green_direction is not None
                        and green_mpu_ativo
                        and hasattr(arduino, "iniciar_mpu")
                        and hasattr(arduino, "poll_mpu")):
                    mpu_concluido, leitura_mpu = arduino.poll_mpu()
                    if mpu_concluido and leitura_mpu is not None:
                        geracao_mpu = int(getattr(
                            leitura_mpu, "request_generation", 0))
                        instante_mpu = float(getattr(
                            leitura_mpu, "received_at", 0.0))
                        geracao_nova = bool(
                            geracao_mpu > 0
                            and geracao_mpu > green_mpu_last_generation
                            and math.isfinite(instante_mpu)
                            and 0. < instante_mpu <= now + .01
                        )
                        if geracao_nova:
                            green_mpu_last_yaw = leitura_mpu.yaw_graus
                            green_mpu_last_timestamp = instante_mpu
                            green_mpu_last_generation = max(
                                green_mpu_last_generation, geracao_mpu)
                            green_control_yaw.value = green_mpu_last_yaw
                            if green_yaw_tracker is not None:
                                green_yaw_progress = green_yaw_tracker.update(
                                    green_mpu_last_yaw,
                                    green_mpu_last_timestamp,
                                    now=now,
                                )
                    if now >= green_mpu_next_query:
                        arduino.iniciar_mpu(
                            timeout=config.GREEN_MPU_RESPONSE_TIMEOUT_S)
                        green_mpu_next_query = (
                            now + config.GREEN_MPU_QUERY_INTERVAL_S)
                linha_verde_recente = (
                    resultado_visao.linha_detectada
                    and now - resultado_visao.publicado_em
                    <= config.LINE_MAX_FRAME_AGE_S
                )
                evento_verde = green_fsm.event
                if (green_fsm.state == GreenManeuverState.APPROACH
                        and green_fsm.locked_direction
                        in (GreenDecision.LEFT, GreenDecision.RIGHT,
                            GreenDecision.UTURN)):
                    geometria_pronta = bool(
                        evento_verde is not None
                        and evento_verde.ready_to_turn
                        and evento_verde.junction_visible
                        and not evento_verde.geometry_predicted
                    )
                    if (evento_verde is not None
                            and evento_verde.sequence
                            != green_ready_last_sequence):
                        green_ready_last_sequence = evento_verde.sequence
                        # Precisam ser dois frames NOVOS e consecutivos. Uma
                        # perda/predicao entre eles zera a sequencia em vez de
                        # deixar um voto antigo sobreviver.
                        green_transversal_frames = (
                            green_transversal_frames + 1
                            if geometria_pronta else 0
                        )
                    if (evento_verde is not None
                            and not evento_verde.junction_visible
                            and not evento_verde.geometry_predicted
                            and not evento_verde.ready_to_turn):
                        green_fsm.fault(
                            "geometria da juncao perdida por mais de 0,20 s",
                            now=now,
                        )
                    elif (green_transversal_frames
                          >= config.GREEN_TOPOLOGY_READY_CONFIRM_FRAMES):
                        timeout_giro = (
                            config.GREEN_TURN_AROUND_TIMEOUT
                            if green_fsm.locked_direction
                            == GreenDecision.UTURN
                            else GREEN_TURN_TIMEOUT
                        )
                        green_fsm.begin_turn(
                            now=now, timeout_s=timeout_giro)
                        green_control_state.value = int(green_fsm.state)
                        green_turn_started = now
                        green_target_seen = False
                        green_transversal_frames = 0
                        green_mpu_turn_origin = green_mpu_last_yaw
                        if green_direction in (
                            "left", "right", "turn_around",
                        ):
                            decisao_yaw = {
                                "left": GreenDecision.LEFT,
                                "right": GreenDecision.RIGHT,
                                "turn_around": GreenDecision.UTURN,
                            }[green_direction]
                            green_yaw_tracker = SignedYawTracker(
                                decisao_yaw,
                                positive_is_right=(
                                    config.GREEN_MPU_POSITIVE_IS_RIGHT),
                                max_age_s=max(
                                    config.GREEN_MPU_RESPONSE_TIMEOUT_S * 2.,
                                    .20,
                                ),
                            )
                            if (green_mpu_last_yaw is not None
                                    and green_mpu_last_timestamp is not None):
                                green_yaw_progress = green_yaw_tracker.update(
                                    green_mpu_last_yaw,
                                    green_mpu_last_timestamp,
                                    now=now,
                                )

                if green_fsm.check_timeout(now=now):
                    green_fault_stop.value = True
                    green_control_state.value = int(green_fsm.state)
                    steer()
                    continue

                if (green_direction == "turn_around"
                        and green_fsm.state == GreenManeuverState.TURNING):
                    status.value = 'Girando 180° para a direita'
                    if mission_mode.value:
                        entry_armed.value = False
                        _reset_entry_silver(
                            "prata desarmada durante giro de 180")
                    epoca_serial_180 = arduino.connection_epoch
                    resultado_180 = turn_around(
                        last_turn_dir,
                        require_alignment=True,
                        arduino=(
                            arduino if green_mpu_ativo else None
                        ),
                        yaw_tracker=green_yaw_tracker,
                        yaw_callback=lambda value: setattr(
                            green_control_yaw, "value", value),
                        should_abort=lambda: bool(
                            terminate.value
                            or not arduino.connected
                            or arduino.connection_epoch != epoca_serial_180
                            or not green_calibration_ready.value
                            or (
                                green_fsm.deadline is not None
                                and time.monotonic() >= green_fsm.deadline
                            )
                        ),
                        expected_branch_token=(
                            0 if green_fsm.event is None
                            else green_fsm.event.target_branch_token
                        ),
                    )
                    fim_180 = time.monotonic()
                    if green_fsm.check_timeout(now=fim_180):
                        green_fault_stop.value = True
                        green_control_state.value = int(green_fsm.state)
                        steer()
                        continue
                    if resultado_180 is None:
                        green_fsm.fault(
                            "180 nao reencontrou a linha de saida", now=fim_180)
                        green_fault_stop.value = True
                        steer()
                        continue
                    last_turn_dir = resultado_180
                    green_fsm.begin_reacquire(now=time.monotonic())
                    consumido = green_fsm.complete(
                        now=time.monotonic(),
                        timeout_s=(
                            config.GREEN_TOPOLOGY_COOLDOWN_TIMEOUT_S),
                    )
                    green_decision_consumed_id.value = consumido
                    green_control_state.value = int(green_fsm.state)
                    controlador_linha.reset()
                    green_direction = None
                    green_turn_started = None
                    green_target_seen = False
                    green_turn_target.value = 2
                    green_exit_stable_frames = 0
                    if mission_mode.value:
                        entry_rearm_after = (
                            time.monotonic()
                            + config.ENTRY_TURN_AROUND_REARM_S
                        )
                        _reset_entry_silver(
                            "prata zerada apos giro de 180")
                    continue

                if (green_fsm.locked_direction == GreenDecision.STRAIGHT
                        and green_fsm.state == GreenManeuverState.APPROACH):
                    saida_reta_estavel = saida_topologica_real_estavel(
                        resultado_visao,
                        agora=now,
                    )
                    if (evento_verde is not None
                            and resultado_visao.sequencia
                            >= evento_verde.sequence
                            and resultado_visao.sequencia
                            != green_exit_last_sequence):
                        green_exit_last_sequence = resultado_visao.sequencia
                        green_exit_stable_frames = (
                            green_exit_stable_frames + 1
                            if saida_reta_estavel
                            else 0
                        )
                    if green_exit_stable_frames >= 3:
                        green_fsm.begin_turn(now=now)
                        green_fsm.begin_reacquire(now=now)
                        consumido = green_fsm.complete(
                            now=now,
                            timeout_s=(
                                config.GREEN_TOPOLOGY_COOLDOWN_TIMEOUT_S),
                        )
                        green_decision_consumed_id.value = consumido
                        green_control_state.value = int(green_fsm.state)
                        green_turn_target.value = 2
                        green_exit_stable_frames = 0

                controle_generico_bloqueado = bool(
                    green_direction is not None
                    or green_fsm.state in (
                        GreenManeuverState.OBSERVE,
                        GreenManeuverState.COMMITTED,
                        GreenManeuverState.APPROACH,
                        GreenManeuverState.TURNING,
                        GreenManeuverState.REACQUIRE,
                    )
                )
                if not controle_generico_bloqueado:
                    saida_linha = controlador_linha.atualizar(
                        sequencia=resultado_visao.sequencia,
                        publicado_em=resultado_visao.publicado_em,
                        linha_detectada=resultado_visao.linha_detectada,
                        linha_a_frente=resultado_visao.linha_a_frente,
                        ponto_inferior_x=resultado_visao.ponto_inferior_x,
                        ponto_inferior_y=resultado_visao.ponto_inferior_y,
                        ponto_alvo_x=resultado_visao.ponto_alvo_x,
                        ponto_alvo_y=resultado_visao.ponto_alvo_y,
                        ponto_futuro_x=resultado_visao.ponto_futuro_x,
                        ponto_futuro_y=resultado_visao.ponto_futuro_y,
                        # No verde o ponto lateral travado e a autoridade. O
                        # lookahead generico pode escolher o ramo reto de uma
                        # intersecao conectada e contrariar o marcador.
                        ponto_futuro_valido=(
                            resultado_visao.ponto_futuro_valido
                            and green_direction is None),
                        agora=now,
                    )
                else:
                    # Enquanto a FSM verde possui autoridade, a intersecao nao
                    # pode contaminar a memoria do seguidor generico. Ele so
                    # recebe um frame novo depois da saida estar estabilizada.
                    controlador_linha.suspender()
                    saida_linha = None

                velocidade_base = get_speed(line_angle.value)
                command_speed = velocidade_base
                if estado_rampa == "SUBINDO":
                    command_speed = config.RAMPA_SUBIDA_SPEED
                elif estado_rampa == "DESCENDO":
                    command_speed = config.RAMPA_DESCIDA_SPEED
                if config.RETA_RAPIDA_HABILITADA:
                    permitir_reta_rapida = (
                        green_direction is None
                        and green_fsm.state == GreenManeuverState.FOLLOW
                        and not preferencia_linha_esquerda.value
                        and line_detected.value
                        and line_ahead.value
                        and abs(line_angle.value)
                        <= config.ANGULO_MAXIMO_RETA_RAPIDA
                        and abs(last_bottom_point.value - camera_x / 2)
                        <= config.ERRO_INFERIOR_RETA_RAPIDA_PX
                        and last_bottom_point_y.value
                        >= camera_y * config.ALTURA_MINIMA_PONTO_INFERIOR_RAPIDA
                        and line_size.value >= config.AREA_MINIMA_LINHA_RAPIDA
                        and not green_candidate.value
                        and not red_candidate.value
                        and not red_detected.value
                        and not entry_silver_detected.value
                        and not entry_silver_confirmed.value
                        and not monitor_obstaculo.bloqueia_velocidade_rapida
                    )
                    command_speed = velocidade_adaptativa.atualizar(
                        ler_resultado_visao_rapida(),
                        velocidade_base=velocidade_base,
                        direcao=direcao_visual,
                        permitir_rapido=permitir_reta_rapida,
                    )
                    if (
                        velocidade_adaptativa.modo_rapido
                        != modo_rapido_anterior
                    ):
                        modo_rapido_anterior = (
                            velocidade_adaptativa.modo_rapido)
                        if modo_rapido_anterior:
                            print(
                                "[controle] reta estável confirmada — "
                                f"{velocidade_adaptativa.fps_visao:.0f} FPS, "
                                "acelerando até PWM "
                                f"{round(config.VELOCIDADE_RETA_RAPIDA * config.MAX_PWM)}"
                            )
                        else:
                            print(
                                "[controle] fim da reta rápida — "
                                f"voltando imediatamente ao PWM "
                                f"{round(LINE_FOLLOW_SPEED * config.MAX_PWM)}"
                            )

                usar_controle_linha = False
                correcao_linha = 0.

                if (green_fsm.state == GreenManeuverState.APPROACH
                        and green_fsm.locked_direction
                        == GreenDecision.STRAIGHT):
                    command_speed = min(
                        command_speed, config.GREEN_TOPOLOGY_PENDING_SPEED)

                if green_fsm.state == GreenManeuverState.COOLDOWN:
                    saida_estavel = saida_topologica_real_estavel(
                        resultado_visao,
                        agora=now,
                    )
                    if (resultado_visao.sequencia
                            != green_exit_last_sequence):
                        green_exit_last_sequence = resultado_visao.sequencia
                        green_exit_stable_frames = (
                            green_exit_stable_frames + 1
                            if saida_estavel
                            else 0
                        )
                    if (
                        green_exit_stable_frames
                        >= config.GREEN_TOPOLOGY_REARM_CLEAR_FRAMES
                        and now - green_fsm.state_since
                        >= config.GREEN_TOPOLOGY_COOLDOWN_S
                        and int(green_rearmed_decision_id.value)
                        >= green_fsm.decision_id
                    ):
                        green_fsm.release_cooldown(now=now)
                        green_control_state.value = int(green_fsm.state)
                        green_locked_decision.value = int(GreenDecision.NONE)
                        green_turn_target.value = 0
                        green_exit_stable_frames = 0

                if green_fsm.state == GreenManeuverState.OBSERVE:
                    command_speed = config.GREEN_TOPOLOGY_PENDING_SPEED
                    if linha_verde_recente:
                        usar_controle_linha = True
                        correcao_linha = correcao_aproximacao(
                            resultado_visao.ponto_inferior_x)
                        angle = round(correcao_linha * 180.)
                        status.value = (
                            'Verde PENDING - procurando segundo marcador')
                    else:
                        angle = 190
                        status.value = (
                            'Verde PENDING - PARADO sem linha de entrada')
                elif (green_fsm.state == GreenManeuverState.APPROACH
                      and green_fsm.locked_direction
                      == GreenDecision.STRAIGHT):
                    command_speed = config.GREEN_TOPOLOGY_PENDING_SPEED
                    alvo_reto_x = (
                        None if evento_verde is None
                        else evento_verde.target_branch[0]
                    )
                    correcao_reta = correcao_ramo_reto(
                        alvo_reto_x,
                        resultado_visao.ponto_inferior_x,
                    )
                    if correcao_reta is None:
                        green_fsm.fault(
                            "evento STRAIGHT confirmado sem ramo alvo",
                            now=now,
                        )
                        green_fault_stop.value = True
                        green_control_state.value = int(green_fsm.state)
                        angle = 190
                        status.value = (
                            'Verde STRAIGHT sem alvo topologico - FAULT_STOP')
                    elif linha_verde_recente:
                        usar_controle_linha = True
                        correcao_linha = correcao_reta
                        angle = round(correcao_linha * 180.)
                        status.value = (
                            'Intersecao - ramo reto topologico travado')
                    else:
                        angle = 190
                        status.value = (
                            'Intersecao reta - PARADO aguardando linha valida')
                elif green_direction is not None and green_turn_started is None:
                    # O ramo continua travado na visao, mas ainda nao comanda
                    # o rumo. Ate ele chegar perto da base, somente o ponto
                    # inferior centraliza a faixa de entrada; isso evita
                    # antecipar o tanque e ficar aquem da intersecao.
                    command_speed = GREEN_APPROACH_SPEED
                    aproximacao_valida = (
                        resultado_visao.linha_detectada
                        and now - resultado_visao.publicado_em
                        <= config.LINE_MAX_FRAME_AGE_S
                    )
                    if aproximacao_valida:
                        usar_controle_linha = True
                        correcao_linha = correcao_aproximacao(
                            resultado_visao.ponto_inferior_x)
                        angle = round(correcao_linha * 180.)
                    else:
                        angle = 190
                    status.value = (
                        f'Verde {green_direction} — ramo travado, '
                        + (
                            'centralizando entrada'
                            if aproximacao_valida
                            else 'PARADO aguardando linha valida'
                        )
                    )
                elif green_direction is not None:
                    command_speed = GREEN_TURN_SPEED
                    linha_ramo_recente = ramo_travado_recente(
                        resultado_visao,
                        (
                            0 if evento_verde is None
                            else evento_verde.target_branch_token
                        ),
                        agora=now,
                    )
                    erro_inferior = (
                        resultado_visao.locked_branch_bottom_x
                        - camera_x / 2
                        if linha_ramo_recente
                        else float("nan")
                    )
                    lado_esperado = (
                        -1 if green_direction == "left" else 1)

                    # Mesmo depois de o ramo aparecer, o seguidor generico
                    # nao reassume. Um controlador dedicado usa somente o
                    # ponto inferior da faixa ja selecionada pela direcao
                    # travada e jamais consegue inverter o sinal do evento.
                    if green_target_seen and linha_ramo_recente:
                        correcao_reaquisicao = correcao_reaquisicao_verde(
                            resultado_visao.locked_branch_bottom_x,
                            lado_esperado,
                        )
                        if correcao_reaquisicao is None:
                            green_fsm.fault(
                                "ramo verde publicou ponto inferior invalido",
                                now=now,
                            )
                            green_fault_stop.value = True
                            usar_controle_linha = False
                            angle = 190
                        else:
                            usar_controle_linha = True
                            correcao_linha = correcao_reaquisicao
                            angle = round(correcao_linha * 180.)
                    else:
                        # Conserva o sentido travado numa perda momentanea. O
                        # timeout e o MPU impedem giro indefinido.
                        angle = -180 if green_direction == "left" else 180

                    if usar_controle_linha:
                        usar_controle_linha = True
                        correcao_linha = (
                            min(correcao_linha, 0.)
                            if green_direction == "left"
                            else max(correcao_linha, 0.)
                        )
                        angle = round(correcao_linha * 180.)

                    elapsed_turn = now - green_turn_started
                    amostra_mpu_fresca = bool(
                        green_mpu_last_timestamp is not None
                        and 0. <= now - green_mpu_last_timestamp
                        <= max(
                            config.GREEN_MPU_RESPONSE_TIMEOUT_S * 2.,
                            .20,
                        )
                    )
                    giro_mpu = (
                        green_yaw_progress.progress_deg
                        if (green_yaw_progress is not None
                            and green_yaw_progress.valid
                            and amostra_mpu_fresca)
                        else None
                    )
                    if (amostra_mpu_fresca
                            and green_yaw_progress is not None
                            and green_yaw_progress.wrong_direction):
                        green_fsm.fault(
                            "MPU confirmou giro no sentido oposto", now=now)
                        green_fault_stop.value = True
                        usar_controle_linha = False
                        angle = 190
                    if (giro_mpu is not None
                            and giro_mpu >= config.GREEN_MPU_SLOWDOWN_DEG):
                        command_speed = min(
                            command_speed, config.GREEN_MPU_SLOW_SPEED)
                    if (giro_mpu is not None
                            and giro_mpu >= config.GREEN_MPU_HARD_LIMIT_DEG):
                        green_fsm.fault(
                            "limite angular do MPU sem ramo alinhado",
                            now=now,
                        )
                        green_fault_stop.value = True
                        green_control_state.value = int(green_fsm.state)
                        usar_controle_linha = False
                        angle = 190
                        status.value = (
                            'Verde limitado pelo MPU '
                            f'({giro_mpu:.0f} graus) — FAULT_STOP')
                    elif elapsed_turn < GREEN_TURN_BLIND_TIME:
                        # A linha de entrada ainda pode estar sob o robo. O
                        # controle visual ja atua, mas nao pode encerrar a
                        # manobra durante esta janela curta.
                        status.value = (
                            f'Verde {green_direction} — encaixando ramo '
                            f'({GREEN_TURN_BLIND_TIME:.1f} s)')
                    elif not green_target_seen:
                        # Aceita a linha apenas depois de ela aparecer no lado
                        # que o marcador escolheu. Isso evita capturar o ramo
                        # anterior que ainda cruza o campo da camera.
                        ramo_armado_pela_camera = (
                            ramo_marcado_visto_pela_camera(
                                linha_ramo_recente,
                                erro_inferior,
                                lado_esperado,
                            )
                        )
                        if ramo_armado_pela_camera:
                            green_target_seen = True
                            green_fsm.begin_reacquire(
                                now=now, timeout_s=GREEN_TURN_TIMEOUT)
                            green_control_state.value = int(green_fsm.state)
                            green_transversal_frames = 0
                            green_center_last_sequence = -1
                            green_last_signed_error = (
                                lado_esperado * erro_inferior)
                            status.value = (
                                f'Verde {green_direction} — ramo apareceu '
                                'no lado marcado')
                        else:
                            status.value = (
                                f'Verde {green_direction} — procurando '
                                'ramo no lado marcado')
                    elif linha_ramo_recente:
                        alinhamento_pronto = alinhamento_verde_pode_concluir(
                            erro_inferior,
                            green_last_signed_error,
                            lado_esperado,
                            giro_mpu,
                        )
                        if (resultado_visao.sequencia
                                != green_center_last_sequence):
                            green_center_last_sequence = (
                                resultado_visao.sequencia)
                            green_transversal_frames = (
                                green_transversal_frames + 1
                                if alinhamento_pronto else 0
                            )
                        if (green_transversal_frames
                                >= config.GREEN_TURN_CENTER_CONFIRM_FRAMES):
                            consumido = green_fsm.complete(
                                now=now,
                                timeout_s=(
                                    config.GREEN_TOPOLOGY_COOLDOWN_TIMEOUT_S),
                            )
                            if not consumido:
                                green_fsm.fault(
                                    "sequencia de estados verde invalida",
                                    now=now,
                                )
                                green_fault_stop.value = True
                            else:
                                green_decision_consumed_id.value = consumido
                            green_control_state.value = int(green_fsm.state)
                            green_direction = None
                            green_turn_started = None
                            green_target_seen = False
                            green_turn_target.value = 2
                            green_exit_stable_frames = 0
                            controlador_linha.reset()
                            usar_controle_linha = False
                            angle = 190
                            status.value = (
                                'Verde concluido — ramo alinhado no centro')
                        else:
                            if (resultado_visao.sequencia
                                    == green_center_last_sequence):
                                green_last_signed_error = (
                                    lado_esperado * erro_inferior)
                            status.value = (
                                f'Verde {green_direction} — trazendo ramo '
                                'para o centro '
                                f'({green_transversal_frames}/'
                                f'{config.GREEN_TURN_CENTER_CONFIRM_FRAMES})')
                    else:
                        if (resultado_visao.sequencia
                                != green_center_last_sequence):
                            green_center_last_sequence = (
                                resultado_visao.sequencia)
                            green_transversal_frames = 0
                        status.value = (
                            f'Verde {green_direction} — trazendo ramo '
                            'para o centro')
                elif saida_linha is not None and saida_linha.comando_valido:
                    usar_controle_linha = True
                    correcao_linha = saida_linha.correcao
                    angle = saida_linha.angulo_equivalente
                    if saida_linha.estado == CORNER:
                        lado = 'direita' if correcao_linha > 0 else 'esquerda'
                        status.value = (
                            f'Curva fechada {lado} — alinhando nova reta')
                    elif saida_linha.estado == LOST:
                        status.value = (
                            'Linha fora da imagem — avanco temporario')
                else:
                    angle = 190
                    status.value = (
                        'Linha nao reencontrada — parada de seguranca')

                # A candidata prata já foi tratada no início do ciclo: durante
                # a observação o robô fica parado; quando aparece preto além da
                # faixa este mesmo segue-linha continua em velocidade normal.
                if usar_controle_linha:
                    steering_correction.value = correcao_linha
                    if saida_linha is None:
                        # Aproximacao verde usa o mesmo mixer dos motores, mas
                        # deliberadamente nao possui uma SaidaSegueLinha: ela
                        # centraliza apenas o ponto inferior e ignora o rumo
                        # futuro do ramo ate liberar o controle continuo.
                        steering_state.value = STEERING_SPECIAL
                        steering_lateral_error.value = max(min(
                            (
                                resultado_visao.ponto_inferior_x
                                - camera_x / 2
                            ) / (camera_x / 2),
                            1.,
                        ), -1.)
                        steering_heading.value = 0.
                    else:
                        steering_state.value = (
                            STEERING_CORNER
                            if saida_linha.estado == CORNER
                            else STEERING_LOST
                            if saida_linha.estado == LOST
                            else STEERING_TRACK
                        )
                        steering_lateral_error.value = (
                            saida_linha.erro_lateral)
                        steering_heading.value = saida_linha.angulo_linha
                    steer_line(correcao_linha, command_speed)
                else:
                    steering_state.value = STEERING_SPECIAL
                    steering_correction.value = 0.
                    steering_lateral_error.value = 0.
                    steering_heading.value = 0.
                    steer(angle, command_speed)

            elif line_status.value == "stop":
                controlador_linha.suspender()
                stop_for_red()
                if mission_mode.value and not entry_armed.value:
                    # A sala de resgate já ficou para trás: esta é a faixa
                    # vermelha final da prova. O supervisor encerra a missão.
                    red_finished.value = True
                    status.value = 'Faixa vermelha final — prova concluida'
                    print("[controle] faixa vermelha final; prova concluida")
                    break
                line_status.value = "line_detected"
                continue

            elif line_status.value == "gap_detected":
                controlador_linha.suspender()
                verified_gap = orientate_gap()

                if verified_gap:
                    timer.set_timer("gap_avoid", GAP_AVOID_TIMEOUT)
                else:
                    # Qualquer validacao negativa refere-se ao mesmo elemento
                    # visual pelos proximos instantes; nao o valida novamente
                    # a cada frame enquanto o robo ainda termina a curva.
                    gap_retry_after = time.monotonic() + GAP_REJECT_COOLDOWN
                    line_status.value = "line_detected"
                    min_line_size.value = MIN_LINE_SIZE_DEFAULT
                    sleep_steering(.1)

                continue

            elif line_status.value == "gap_avoid":
                controlador_linha.suspender()
                status.value = 'Cruzando o gap'

                if line_detected.value:
                    min_line_size.value = MIN_LINE_SIZE_DEFAULT
                    line_status.value = "line_detected"
                    continue
                else:
                    steer(0, GAP_AVOID_SPEED)

                if timer.get_timer("gap_avoid"):
                    min_line_size.value = GAP_MIN_LINE_SIZE_RETREAT
                    steer(200, GAP_AVOID_SPEED)
                    sleep_steering(GAP_AVOID_RETREAT_TIME)
                    drive_back_until_line(.3, GAP_AVOID_SPEED)

                    line_status.value = "line_detected"
                    sleep_steering(.1)
                    continue

            # Limita o ciclo para não ocupar toda a CPU.
            if time.perf_counter() - iteration_limit_time < 1 / max_iterations:
                time.sleep(abs(1 / max_iterations - (time.perf_counter() - iteration_limit_time)))
            iteration_limit_time = time.perf_counter()

    finally:
        # Uma excecao Python durante uma manobra nao pode apagar o estado que
        # obriga o supervisor a exigir reposicionamento/ciclo fisico.
        if (
            green_fsm.state not in (
                GreenManeuverState.FOLLOW,
                GreenManeuverState.COOLDOWN,
            )
            or green_fsm.event is not None
        ):
            green_fault_stop.value = True
        preferencia_linha_esquerda.value = False
        green_turn_target.value = 0
        green_control_state.value = int(GreenManeuverState.FOLLOW)
        green_locked_decision.value = int(GreenDecision.NONE)
        green_control_yaw.value = float("nan")
        status.value = "Parado"
        try:
            steer()  # PARAR
        finally:
            set_motion_observer(None)
            arduino.close()
