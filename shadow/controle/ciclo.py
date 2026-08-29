"""Ciclo principal de decisões e movimentos do segue-linha."""

import time

import config
from config import (CONTROL_MAX_ITERATIONS, GAP_AVOID_RETREAT_TIME, GAP_AVOID_SPEED,
                    GAP_ENABLED,
                    GAP_AVOID_TIMEOUT, GAP_MIN_LINE_SIZE_RETREAT,
                    GAP_MISSING_CONFIRM_TIME, GAP_REJECT_COOLDOWN,
                    GREEN_APPROACH_SPEED, GREEN_APPROACH_TIME,
                    GREEN_REVERSE_SPEED, GREEN_REVERSE_TIME,
                    GREEN_TURN_BLIND_TIME, GREEN_TURN_CENTER_TOLERANCE_PX,
                    GREEN_TURN_SPEED, GREEN_TURN_TIMEOUT, LINE_FOLLOW_SPEED,
                    MIN_LINE_SIZE_DEFAULT, TURN_AROUND_GREEN_COOLDOWN,
                    VISION_READY_TIMEOUT, camera_x, camera_y)
from controle.orientacao_gap import drive_back_until_line, orientate_gap
from controle.parada_obstaculo import (
    MonitorObstaculo,
    desviar_obstaculo,
)
from controle.parada_vermelho import stop_for_red
from controle.manobra_verde import (alinhamento_verde_pode_concluir,
                                    controle_visual_verde_liberado,
                                    correcao_aproximacao,
                                    deve_iniciar_giro_verde,
                                    progresso_giro_mpu,
                                    ramo_marcado_visto_pela_camera,
                                    ramo_pronto_para_giro)
from controle.velocidade import get_speed
from controle.velocidade_adaptativa import ControladorVelocidadeAdaptativa
from controle.direcao import (atualizar_tanque_curva_fechada, init_steering,
                              mix_line_pwm, mix_tank_pwm, sleep_steering,
                              steer, steer_line)
from controle.retorno import turn_around
from controle.seguidor_linha import CORNER, LOST, ControladorSegueLinha
from comunicacao_serial.arduino import Arduino
from shared.dados_compartilhados import (ENTRY_SILVER_BLACK_FOLLOW,
                               ENTRY_SILVER_IDLE, ENTRY_SILVER_VALIDATING,
                               entry_armed, entry_silver_confirmed,
                               entry_silver_detected, entry_silver_reason,
                               entry_silver_state, entry_silver_votes,
                               green_candidate,
                               green_turn_target,
                               last_bottom_point,
                               last_bottom_point_y,
                               ler_resultado_visao_rapida,
                               line_ahead, line_angle,
                               line_detected,
                               line_size, line_status, min_line_size,
                               mission_mode,
                               preferencia_linha_esquerda,
                               red_candidate, red_detected, red_finished,
                               rescue_requested, rescue_yolo_confirmed,
                               status, terminate,
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
    entry_armed.value = False
    print("[controle] entrada confirmada; PARAR — "
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
    entry_armed.value = False
    _reset_entry_silver("entrada por ausencia de preto")
    print(
        "[controle] linha preta ausente por "
        f"{config.ENTRY_NO_BLACK_RESCUE_DELAY_S:.1f} s; PARAR "
        "— resgate avançará 1 s antes dos giros")
    return True


def _enter_rescue_from_yolo(arduino):
    """Consome o pedido do vigia sem entregar a serial de forma insegura."""
    if terminate.value or steer() is False or not arduino.connected:
        rescue_yolo_confirmed.value = False
        return False
    entry_armed.value = False
    _reset_entry_silver("vitima YOLO confirmada")
    status.value = "Vitima YOLO confirmada - entrando no resgate"
    print("[controle] vítima YOLO confirmada; PARAR - iniciando resgate")
    return True


def _reset_entry_silver(reason):
    """Descarta qualquer candidatura de prata antes de mudar de percurso."""
    entry_silver_detected.value = False
    entry_silver_confirmed.value = False
    entry_silver_votes.value = 0
    entry_silver_reason.value = reason
    entry_silver_state.value = ENTRY_SILVER_IDLE


def _executar_sequencia_partida(arduino):
    """Faz a varredura direita/esquerda antes de iniciar o segue-linha."""
    if not config.STARTUP_TURN_SEQUENCE_ENABLED:
        return True
    status.value = 'Executando sequencia de partida'
    tempo_total = sum(duracao for _sentido, duracao in config.STARTUP_TURN_SEQUENCE)
    print(
        "[controle] sequencia de partida: "
        f"{tempo_total:.1f} s de giro ativo balanceado")
    for indice, (sentido, duracao) in enumerate(
            config.STARTUP_TURN_SEQUENCE):
        if terminate.value or not arduino.connected:
            steer()
            return False
        lado = "direita" if sentido > 0 else "esquerda"
        print(f"[controle] partida: {lado} por {duracao:.1f} s")
        if steer_line(
                sentido, config.STARTUP_TURN_SPEED, tank=True) is False:
            steer()
            return False
        sleep_steering(duracao)
        if steer() is False or not arduino.connected:
            return False
        if indice < len(config.STARTUP_TURN_SEQUENCE) - 1:
            sleep_steering(config.STARTUP_TURN_PAUSE_S)
    print("[controle] sequencia de partida concluida; liberando segue-linha")
    return not terminate.value and arduino.connected


def control_loop():
    try:
        arduino = Arduino()
    except RuntimeError as erro:
        status.value = 'Aguardando Arduino para reiniciar a missao'
        print(f"[controle] Arduino indisponivel: {erro}")
        return
    init_steering(arduino)
    steer()  # motores parados desde o inicio
    if mission_mode.value:
        # Uma sessao da missao nunca reconecta em movimento. Se a placa cair,
        # o supervisor cria uma tentativa nova depois do reposicionamento.
        arduino.travar_sessao()

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

    # O vigia pode ter confirmado uma vítima enquanto a câmera de linha
    # aquecia. Não inicie sequer a sequência de partida nesse caso.
    if mission_mode.value and rescue_yolo_confirmed.value:
        if _enter_rescue_from_yolo(arduino):
            rescue_requested.value = True
        else:
            status.value = "Falha ao parar para vitima YOLO - reiniciando missao"
        arduino.close()
        return

    if not _executar_sequencia_partida(arduino):
        status.value = 'Sequencia de partida interrompida'
        try:
            steer()
        finally:
            arduino.close()
        return

    # A sequência de partida usa pulsos de no máximo um segundo; consome já o
    # pedido que chegou durante ela antes de qualquer comando de segue-linha.
    if mission_mode.value and rescue_yolo_confirmed.value:
        if _enter_rescue_from_yolo(arduino):
            rescue_requested.value = True
        else:
            status.value = "Falha ao parar para vitima YOLO - reiniciando missao"
        arduino.close()
        return

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
    tanque_curva_fechada = False
    green_direction = None
    green_approach_until = 0.
    green_turn_started = None
    green_reverse_until = None
    green_turn_deadline = 0.
    green_target_seen = False
    green_transversal_frames = 0
    green_last_signed_error = None
    green_mpu_last_yaw = None
    green_mpu_turn_origin = None
    green_mpu_next_query = 0.
    green_release_until = 0.
    green_armed = True
    green_rearm_after = 0.
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
    preferencia_esquerda_inicio = 0.
    preferencia_esquerda_alinhada_desde = None
    entry_rearm_after = None
    estado_rampa = "PLANO"
    proxima_consulta_rampa = 0.

    try:
        while not terminate.value:
            if not arduino.connected:
                status.value = 'Arduino desconectado - aguardando reinicio'
                print("[controle] Arduino desconectado; encerrando sessao")
                return

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
            if mission_mode.value and rescue_yolo_confirmed.value:
                if _enter_rescue_from_yolo(arduino):
                    rescue_requested.value = True
                else:
                    status.value = (
                        "Falha ao parar para vitima YOLO - reiniciando missao")
                break

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
                    green_rearm_after = max(
                        green_rearm_after,
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
                        deve_encerrar=lambda: terminate.value,
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
                    # A serial pode ter caído no meio da manobra. Não espere
                    # ``terminate`` aqui: esse loop mantinha o filho vivo e
                    # impedia mission.py de notar a falha e recriar a sessão
                    # de segue-linha após a reconexão do Arduino.
                    try:
                        steer()
                    finally:
                        arduino.close()
                    return

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
                green_reverse_until = None
                green_turn_deadline = 0.
                green_target_seen = False
                green_turn_target.value = 0
                green_armed = False
                green_rearm_after = obstaculo_retry_after
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

                if (green_turn_target.value == 2
                        and now >= green_release_until):
                    green_turn_target.value = 0
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
                    # Gap e somente uma ausencia em reta. Durante uma curva
                    # verde, perder a faixa por alguns frames e esperado e o
                    # ramo travado precisa continuar tendo prioridade.
                    and green_direction is None
                    and turn_dir.value == "straight"
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
                # Uma leitura da direção por iteração mantém todas as decisões
                # deste comando coerentes e evita várias consultas ao Manager.
                direcao_visual = turn_dir.value
                if (
                    not preferencia_linha_esquerda.value
                    and direcao_visual == "turn_around"
                ):
                    status.value = 'Girando 180° para a direita'

                    if mission_mode.value:
                        # Desarma ANTES do primeiro comando da manobra. A
                        # visao ainda roda em paralelo, mas o resultado dela
                        # sera descartado ate o rearme limpo pos-retorno.
                        entry_armed.value = False
                        _reset_entry_silver(
                            "prata desarmada durante giro de 180")

                    last_turn_dir = turn_around(last_turn_dir)
                    controlador_linha.reset()
                    # O filtro visual pode degradar "dois verdes" para apenas
                    # left/right por alguns frames. Nao iniciar uma segunda
                    # manobra com essa leitura residual.
                    green_direction = None
                    green_turn_started = None
                    green_reverse_until = None
                    green_turn_deadline = 0.
                    green_target_seen = False
                    green_transversal_frames = 0
                    green_last_signed_error = None
                    green_turn_target.value = 0
                    green_armed = False
                    green_rearm_after = (
                        time.monotonic() + TURN_AROUND_GREEN_COOLDOWN)
                    if mission_mode.value:
                        entry_rearm_after = (
                            time.monotonic()
                            + config.ENTRY_TURN_AROUND_REARM_S
                        )
                        _reset_entry_silver(
                            "prata zerada apos giro de 180")
                        print(
                            "[controle] 180 concluido; prata permanece "
                            "zerada ate o rearme")
                    continue

                status.value = (
                    'Seguindo Linha — preferência esquerda'
                    if preferencia_linha_esquerda.value
                    else 'Seguindo Linha'
                )

                now = time.monotonic()

                if (time.monotonic() >= green_rearm_after
                        and not preferencia_linha_esquerda.value
                        and direcao_visual == "straight"
                        and green_direction is None):
                    green_armed = True

                if (not preferencia_linha_esquerda.value
                        and green_armed and green_direction is None
                        and direcao_visual in ("left", "right")):
                    green_direction = direcao_visual
                    green_approach_until = now + GREEN_APPROACH_TIME
                    green_turn_started = None
                    green_reverse_until = None
                    green_turn_deadline = 0.
                    green_target_seen = False
                    green_transversal_frames = 0
                    green_last_signed_error = None
                    green_mpu_last_yaw = None
                    green_mpu_turn_origin = None
                    green_mpu_next_query = now
                    if hasattr(arduino, "cancelar_mpu"):
                        arduino.cancelar_mpu()
                    green_turn_target.value = (
                        -1 if direcao_visual == "left" else 1)
                    green_armed = False

                resultado_visao = ler_resultado_visao_rapida()

                # Mantem uma amostra recente durante a aproximacao. Quando o
                # tanque comeca ela vira a origem do giro, sem bloquear o
                # controle com um comando ``MPU ZERO``.
                if (green_direction is not None
                        and green_reverse_until is None
                        and config.GREEN_MPU_ENABLED
                        and hasattr(arduino, "iniciar_mpu")
                        and hasattr(arduino, "poll_mpu")):
                    mpu_concluido, leitura_mpu = arduino.poll_mpu()
                    if mpu_concluido and leitura_mpu is not None:
                        green_mpu_last_yaw = leitura_mpu.yaw_graus
                    if now >= green_mpu_next_query:
                        arduino.iniciar_mpu(
                            timeout=config.GREEN_MPU_RESPONSE_TIMEOUT_S)
                        green_mpu_next_query = (
                            now + config.GREEN_MPU_QUERY_INTERVAL_S)
                aproximando_ramo_verde = (
                    green_direction is not None
                    and green_turn_started is None
                    and green_reverse_until is None
                )
                transversal_pronta = (
                    aproximando_ramo_verde
                    and ramo_pronto_para_giro(
                        green_direction,
                        faixa_transversal_y=(
                            resultado_visao.faixa_transversal_y),
                    )
                )
                green_transversal_frames = (
                    green_transversal_frames + 1
                    if transversal_pronta else 0
                )
                linha_verde_recente = (
                    resultado_visao.linha_detectada
                    and now - resultado_visao.publicado_em
                    <= config.LINE_MAX_FRAME_AGE_S
                )
                if (
                    aproximando_ramo_verde
                    and deve_iniciar_giro_verde(
                        green_transversal_frames,
                        agora=now,
                        limite_aproximacao=green_approach_until,
                        linha_recente=linha_verde_recente,
                    )
                ):
                    green_turn_started = now
                    green_turn_deadline = now + GREEN_TURN_TIMEOUT
                    green_target_seen = False
                    green_transversal_frames = 0
                    green_mpu_turn_origin = green_mpu_last_yaw
                    aproximando_ramo_verde = False

                if (green_direction is None
                        or (green_turn_started is not None
                            and green_target_seen)):
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
                    # Na aproximacao e no inicio do tanque, a faixa de entrada
                    # ainda domina a imagem. Ela nao pode contaminar a memoria
                    # do controlador antes de o ramo marcado chegar pelo lado
                    # correto.
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

                if (green_direction is not None
                        and green_reverse_until is not None):
                    if now < green_reverse_until:
                        angle = 200
                        command_speed = GREEN_REVERSE_SPEED
                        status.value = 'Verde concluido — dando re curta'
                    else:
                        green_direction = None
                        green_turn_started = None
                        green_reverse_until = None
                        green_turn_deadline = 0.
                        green_target_seen = False
                        green_turn_target.value = 0
                        angle = line_angle.value if line_detected.value else 190
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
                    if green_turn_started is None:
                        green_turn_started = now
                        green_turn_deadline = now + GREEN_TURN_TIMEOUT
                        green_target_seen = False
                        green_mpu_turn_origin = green_mpu_last_yaw
                    command_speed = GREEN_TURN_SPEED

                    # O ramo marcado passa pelo mesmo controlador continuo do
                    # segue-linha. Curvas moderadas ficam diferenciais e o
                    # tanque aparece somente quando a geometria realmente
                    # pede uma correcao extrema.
                    controle_ramo_valido = controle_visual_verde_liberado(
                        green_target_seen,
                        saida_linha is not None
                        and saida_linha.comando_valido,
                    )
                    if controle_ramo_valido:
                        usar_controle_linha = True
                        correcao_linha = saida_linha.correcao
                        angle = saida_linha.angulo_equivalente
                    else:
                        # Conserva o sentido travado numa perda momentanea. O
                        # timeout e o MPU impedem giro indefinido.
                        angle = -180 if green_direction == "left" else 180

                    elapsed_turn = now - green_turn_started
                    giro_mpu = progresso_giro_mpu(
                        green_mpu_turn_origin, green_mpu_last_yaw)
                    if (giro_mpu is not None
                            and giro_mpu >= config.GREEN_MPU_SLOWDOWN_DEG):
                        command_speed = min(
                            command_speed, config.GREEN_MPU_SLOW_SPEED)
                    linha_ramo_recente = (
                        resultado_visao.linha_detectada
                        and now - resultado_visao.publicado_em
                        <= config.LINE_MAX_FRAME_AGE_S
                    )
                    erro_inferior = (
                        resultado_visao.ponto_inferior_x - camera_x / 2)
                    lado_esperado = -1 if green_direction == "left" else 1

                    if (giro_mpu is not None
                            and giro_mpu >= config.GREEN_MPU_HARD_LIMIT_DEG):
                        # Se a camera perder a faixa por um frame, o chassi
                        # ainda nao pode atravessar completamente os 90 graus.
                        green_direction = None
                        green_turn_started = None
                        green_reverse_until = None
                        green_turn_deadline = 0.
                        green_target_seen = False
                        green_turn_target.value = 2
                        green_release_until = (
                            now + config.GREEN_RELEASE_MEMORY_S)
                        green_armed = False
                        green_rearm_after = (
                            now + TURN_AROUND_GREEN_COOLDOWN)
                        controlador_linha.reset()
                        usar_controle_linha = False
                        angle = 190
                        status.value = (
                            'Verde limitado pelo MPU '
                            f'({giro_mpu:.0f} graus) — parada de seguranca')
                    elif elapsed_turn < GREEN_TURN_BLIND_TIME:
                        # A linha de entrada ainda pode estar sob o robo. O
                        # controle visual ja atua, mas nao pode encerrar a
                        # manobra durante esta janela curta.
                        status.value = (
                            f'Verde {green_direction} — encaixando ramo '
                            f'({GREEN_TURN_BLIND_TIME:.1f} s)')
                    elif now >= green_turn_deadline:
                        # Sem esta trava um falso contorno poderia deixar o
                        # tanque girando indefinidamente. Nao da re, pois o
                        # ramo esperado nunca foi confirmado.
                        green_direction = None
                        green_turn_started = None
                        green_reverse_until = None
                        green_turn_deadline = 0.
                        green_target_seen = False
                        green_turn_target.value = 2
                        green_release_until = (
                            now + config.GREEN_RELEASE_MEMORY_S)
                        green_armed = False
                        green_rearm_after = now + TURN_AROUND_GREEN_COOLDOWN
                        controlador_linha.reset()
                        usar_controle_linha = False
                        angle = 190
                        status.value = (
                            'Verde — ramo marcado nao foi encontrado; '
                            'parada de seguranca')
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
                            # O ramo marcado acabou de aparecer no lado
                            # correto. Arme imediatamente a memoria de canto
                            # para nao perder um Pacman se ele desaparecer no
                            # proximo frame durante o giro.
                            controlador_linha.forcar_canto(
                                lado_esperado,
                                agora=now,
                            )
                            green_transversal_frames = 0
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
                        green_transversal_frames = (
                            green_transversal_frames + 1
                            if alinhamento_pronto else 0
                        )
                        if (green_transversal_frames
                                >= config.GREEN_TURN_CENTER_CONFIRM_FRAMES):
                            green_direction = None
                            green_turn_started = None
                            green_reverse_until = None
                            green_turn_deadline = 0.
                            green_target_seen = False
                            green_turn_target.value = 2
                            green_release_until = (
                                now + config.GREEN_RELEASE_MEMORY_S)
                            controlador_linha.reset()
                            usar_controle_linha = False
                            angle = 190
                            status.value = (
                                'Verde concluido — ramo alinhado no centro')
                        else:
                            green_last_signed_error = (
                                lado_esperado * erro_inferior)
                            status.value = (
                                f'Verde {green_direction} — trazendo ramo '
                                'para o centro '
                                f'({green_transversal_frames}/'
                                f'{config.GREEN_TURN_CENTER_CONFIRM_FRAMES})')
                    else:
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
                    if saida_linha is not None:
                        tanque_curva_fechada = (
                            atualizar_tanque_curva_fechada(
                                saida_linha.estado,
                                correcao_linha,
                                tanque_curva_fechada,
                            )
                        )
                    else:
                        tanque_curva_fechada = False
                    misturador = (
                        mix_tank_pwm
                        if tanque_curva_fechada else mix_line_pwm
                    )
                    pwm_esquerda, pwm_direita = misturador(
                        correcao_linha, command_speed)
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
                    steering_left_pwm.value = pwm_esquerda
                    steering_right_pwm.value = pwm_direita
                    if tanque_curva_fechada:
                        status.value = (
                            'Curva fechada — tanque centralizando nova reta')
                    steer_line(
                        correcao_linha,
                        command_speed,
                        tank=tanque_curva_fechada,
                    )
                else:
                    tanque_curva_fechada = False
                    steering_state.value = STEERING_SPECIAL
                    steering_correction.value = 0.
                    steering_lateral_error.value = 0.
                    steering_heading.value = 0.
                    steering_left_pwm.value = 0
                    steering_right_pwm.value = 0
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
        preferencia_linha_esquerda.value = False
        green_turn_target.value = 0
        status.value = "Parado"
        try:
            steer()  # PARAR
        finally:
            arduino.close()
