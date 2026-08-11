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
                    GREEN_TURN_SIDE_MIN_ERROR_PX, GREEN_TURN_SPEED,
                    GREEN_TURN_TIMEOUT, LINE_FOLLOW_SPEED,
                    LINE_LOSS_STEER_HOLD, MIN_LINE_SIZE_DEFAULT,
                    PIVOT_BOTTOM_MIN_ERROR_PX,
                    PIVOT_RECOVERY_ASSIST_RAMP,
                    PIVOT_RECOVERY_ASSIST_START, PIVOT_RECOVERY_EXIT_ANGLE,
                    PIVOT_RECOVERY_SPEED, PIVOT_RECOVERY_TIMEOUT,
                    PIVOT_PROGRESS_PX, PIVOT_STALL_MIN_ANGLE,
                    PIVOT_STALL_RAMP_TIME, PIVOT_STALL_TIME,
                    TURN_AROUND_GREEN_COOLDOWN, VISION_READY_TIMEOUT,
                    FRONT_ANCHOR_FULL_ANGLE,
                    FRONT_ANCHOR_START_ANGLE, camera_x, camera_y)
from controle.orientacao_gap import drive_back_until_line, orientate_gap
from controle.parada_obstaculo import (
    MonitorObstaculo,
    desviar_obstaculo,
    orientacao_continuacao_saida,
    procurar_continuacao_saida_pivo_dianteiro,
)
from controle.parada_vermelho import stop_for_red
from controle.velocidade import get_speed
from controle.velocidade_adaptativa import ControladorVelocidadeAdaptativa
from controle.direcao import init_steering, sleep_steering, steer
from controle.retorno import turn_around
from comunicacao_serial.arduino import Arduino
from shared.dados_compartilhados import (add_time_value, empty_time_arr,
                               entry_armed, entry_silver_confirmed,
                               entry_silver_detected, entry_silver_reason,
                               entry_silver_votes, green_candidate,
                               green_turn_target,
                               exit_line_search_pending,
                               last_bottom_point,
                               last_bottom_point_y,
                               line_ahead, line_angle,
                               line_detected,
                               line_size, line_status, min_line_size,
                               mission_mode,
                               preferencia_linha_esquerda,
                               red_candidate, red_detected, red_finished,
                               rescue_requested, status, terminate,
                               ler_resultado_visao_rapida, timer, turn_dir,
                               vision_ready)


def _enter_rescue_zone(arduino):
    """Entrega a câmera/serial ao resgate, que faz seu avanço de 1 segundo."""
    steer()
    # O LED só pode ser apagado enquanto esta serial ainda existe. O processo
    # de resgate reafirma o comando assim que abre a serial dele.
    arduino.led("APAGADO")
    entry_armed.value = False
    print("[controle] entrada confirmada; PARAR e LED APAGADO — "
          "resgate fará o avanço de 1 s")


def control_loop():
    try:
        arduino = Arduino()
    except RuntimeError as erro:
        status.value = 'Aguardando Arduino para reiniciar a missao'
        print(f"[controle] Arduino indisponivel: {erro}")
        return
    init_steering(arduino)
    steer()  # motores parados desde o inicio
    arduino.led("ACESO")
    if mission_mode.value:
        # Uma sessao da missao nunca reconecta em movimento. Se a placa cair,
        # o supervisor cria uma tentativa nova depois do reposicionamento.
        arduino.travar_sessao()
    print("[controle] LED ACESO: modo segue-linha")

    last_turn_dir = "l"

    time_last_angles = empty_time_arr()

    # espera a visao publicar o primeiro frame processado
    wait_start = time.perf_counter()
    while not vision_ready.value and not terminate.value:
        if not arduino.connected:
            status.value = 'Arduino desconectado - aguardando reinicio'
            print("[controle] Arduino desconectado durante a espera da visao")
            return
        if time.perf_counter() - wait_start > VISION_READY_TIMEOUT:
            print("[controle] AVISO: visão não ficou pronta em "
                  f"{VISION_READY_TIMEOUT} s — seguindo mesmo assim")
            break
        arduino.refresh()
        time.sleep(.05)

    line_status.value = "line_detected"
    status.value = "Shadow2026 pronto — aguardando linha"
    print("Shadow2026 ready — awaiting line")

    iteration_limit_time = time.perf_counter()
    max_iterations = CONTROL_MAX_ITERATIONS
    line_missing_since = None
    gap_retry_after = 0.0
    pivot_sign = 0
    pivot_best_error = camera_x
    pivot_last_progress = time.monotonic()
    pivot_last_direction = 0
    pivot_line_lost_since = None
    last_follow_angle = 0
    last_line_seen = time.monotonic()
    last_rear_pivot_enabled = True
    green_direction = None
    green_approach_until = 0.
    green_turn_started = None
    green_reverse_until = None
    green_turn_deadline = 0.
    green_target_seen = False
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

    try:
        if mission_mode.value and exit_line_search_pending.value:
            status.value = 'Procurando continuacao da linha apos a saida'
            print(
                "[controle] faixa PRETA ja confirmada; procurando "
                "ramificacao: se estiver visivel, o segue-linha assume "
                "mesmo torta; sem ramo, o pivo dianteiro procura por "
                "0.5 s e depois cruza por 1.0 s")
            try:
                lado_encontrado = procurar_continuacao_saida_pivo_dianteiro(
                    arduino,
                    orientacao_ramificacao=lambda: orientacao_continuacao_saida(
                        ler_resultado_visao_rapida(),
                    ),
                    deve_encerrar=lambda: bool(terminate.value),
                )
            except RuntimeError as erro:
                lado_encontrado = None
                print(
                    "[controle] falha no pivo dianteiro pos-resgate: "
                    f"{erro}")
            finally:
                # A manobra e exclusiva da primeira retomada apos o resgate.
                # Nunca pode contaminar perdas de linha normais posteriores.
                exit_line_search_pending.value = False

            if terminate.value:
                return
            if lado_encontrado is None:
                status.value = (
                    'Continuacao da saida nao encontrada - PARADO')
                print(
                    "[controle] continuacao nao encontrada nos dois lados; "
                    "PARADO por seguranca")
                while not terminate.value:
                    arduino.refresh(fail_closed=True)
                    time.sleep(.05)
                return

            line_status.value = "line_detected"
            status.value = 'Continuacao encontrada - seguindo linha'
            print(
                "[controle] ramificacao confirmada em "
                f"{lado_encontrado}; entregando ao segue-linha normal")

        while not terminate.value:
            if not arduino.connected:
                status.value = 'Arduino desconectado - aguardando reinicio'
                print("[controle] Arduino desconectado; encerrando sessao")
                return

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
                    "lateral à direita pelo mesmo tempo")
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
                pivot_sign = 0
                pivot_best_error = camera_x
                pivot_last_progress = time.monotonic()
                pivot_last_direction = 0
                pivot_line_lost_since = None
                last_follow_angle = line_angle.value
                last_line_seen = time.monotonic()
                last_rear_pivot_enabled = True
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
            if (mission_mode.value and entry_armed.value
                    and entry_silver_confirmed.value):
                status.value = 'Faixa prata confirmada — entrando na sala'
                print("[controle] faixa PRATA confirmada; entrando na sala")
                # Para a inferência do modelo imediatamente. O subprocesso
                # resgate só é aberto depois de a câmera de linha fechar.
                _enter_rescue_zone(arduino)
                rescue_requested.value = True
                break

            # Estado normal do segue-linha.
            if line_status.value == "line_detected":

                gap_allowed = GAP_ENABLED and time.monotonic() >= gap_retry_after
                if (gap_allowed and not line_detected.value
                        and not line_ahead.value):
                    if line_missing_since is None:
                        line_missing_since = time.monotonic()
                    elif time.monotonic() - line_missing_since >= GAP_MISSING_CONFIRM_TIME:
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

                    last_turn_dir = turn_around(last_turn_dir)
                    # O filtro visual pode degradar "dois verdes" para apenas
                    # left/right por alguns frames. Nao iniciar uma segunda
                    # manobra com essa leitura residual.
                    green_direction = None
                    green_turn_started = None
                    green_reverse_until = None
                    green_turn_deadline = 0.
                    green_target_seen = False
                    green_turn_target.value = 0
                    green_armed = False
                    green_rearm_after = (
                        time.monotonic() + TURN_AROUND_GREEN_COOLDOWN)
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
                    green_turn_target.value = (
                        -1 if direcao_visual == "left" else 1)
                    green_armed = False

                if green_direction is not None:
                    # Recuperacao de linha do pivo nunca pode vazar para a
                    # manobra deliberada do marcador verde.
                    pivot_last_direction = 0
                    pivot_line_lost_since = None

                if line_detected.value:
                    last_line_seen = now
                    last_follow_angle = line_angle.value
                    last_rear_pivot_enabled = (
                        preferencia_linha_esquerda.value
                        or direcao_visual == "straight"
                    )

                    if (last_rear_pivot_enabled
                            and abs(line_angle.value) > FRONT_ANCHOR_START_ANGLE):
                        pivot_last_direction = 1 if line_angle.value > 0 else -1
                        pivot_line_lost_since = None
                    elif abs(line_angle.value) <= PIVOT_RECOVERY_EXIT_ANGLE:
                        pivot_last_direction = 0
                        pivot_line_lost_since = None
                elif not last_rear_pivot_enabled:
                    pivot_last_direction = 0
                    pivot_line_lost_since = None

                velocidade_base = get_speed(line_angle.value)
                command_speed = velocidade_base
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

                if (green_direction is not None
                        and green_reverse_until is not None):
                    if now < green_reverse_until:
                        angle = 200
                        command_speed = GREEN_REVERSE_SPEED
                        last_rear_pivot_enabled = False
                        status.value = 'Verde concluido — dando re curta'
                    else:
                        green_direction = None
                        green_turn_started = None
                        green_reverse_until = None
                        green_turn_deadline = 0.
                        green_target_seen = False
                        green_turn_target.value = 0
                        angle = line_angle.value if line_detected.value else 190
                        last_rear_pivot_enabled = True
                elif green_direction is not None and now < green_approach_until:
                    # A direcao ja foi memorizada: atravessa o marcador reto
                    # antes de iniciar qualquer rotacao.
                    angle = 0
                    command_speed = GREEN_APPROACH_SPEED
                    last_rear_pivot_enabled = False
                    status.value = f'Verde {green_direction} — avancando antes do giro'
                elif green_direction is not None:
                    if green_turn_started is None:
                        green_turn_started = now
                        green_turn_deadline = now + GREEN_TURN_TIMEOUT
                        green_target_seen = False
                    angle = -180 if green_direction == "left" else 180
                    command_speed = GREEN_TURN_SPEED
                    last_rear_pivot_enabled = False

                    elapsed_turn = now - green_turn_started
                    erro_inferior = last_bottom_point.value - camera_x / 2
                    lado_esperado = -1 if green_direction == "left" else 1

                    if elapsed_turn < GREEN_TURN_BLIND_TIME:
                        # A linha que ainda estava sob o robo nao pode encerrar
                        # a manobra: por este intervalo o giro e cego.
                        status.value = (
                            f'Verde {green_direction} — giro cego '
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
                        green_turn_target.value = 0
                        green_armed = False
                        green_rearm_after = now + TURN_AROUND_GREEN_COOLDOWN
                        angle = 190
                        status.value = (
                            'Verde — ramo marcado nao foi encontrado; '
                            'parada de seguranca')
                    elif not green_target_seen:
                        # Aceita a linha apenas depois de ela aparecer no lado
                        # que o marcador escolheu. Isso evita capturar o ramo
                        # anterior que ainda cruza o campo da camera.
                        if (line_detected.value
                                and lado_esperado * erro_inferior
                                >= GREEN_TURN_SIDE_MIN_ERROR_PX):
                            green_target_seen = True
                            status.value = (
                                f'Verde {green_direction} — ramo apareceu '
                                'no lado marcado')
                        else:
                            status.value = (
                                f'Verde {green_direction} — procurando '
                                'ramo no lado marcado')
                    elif (line_detected.value
                            and abs(erro_inferior)
                            <= GREEN_TURN_CENTER_TOLERANCE_PX):
                        green_reverse_until = now + GREEN_REVERSE_TIME
                        angle = 200
                        command_speed = GREEN_REVERSE_SPEED
                        last_rear_pivot_enabled = False
                        status.value = 'Verde concluido — dando re curta'
                    else:
                        status.value = (
                            f'Verde {green_direction} — trazendo ramo '
                            'para o centro')
                elif line_detected.value:
                    angle = last_follow_angle
                elif pivot_last_direction != 0:
                    if pivot_line_lost_since is None:
                        pivot_line_lost_since = now
                    recovery_time = now - pivot_line_lost_since
                    if recovery_time <= PIVOT_RECOVERY_TIMEOUT:
                        # Mantem o lado conhecido e um erro suficientemente
                        # alto para conservar o pivo traseiro durante a busca.
                        angle = pivot_last_direction * max(
                            abs(last_follow_angle), FRONT_ANCHOR_FULL_ANGLE)
                        command_speed = PIVOT_RECOVERY_SPEED
                        last_rear_pivot_enabled = True
                    else:
                        angle = 190
                        pivot_last_direction = 0
                        pivot_line_lost_since = None
                        status.value = 'Linha nao reencontrada — parada de seguranca'
                elif now - last_line_seen <= LINE_LOSS_STEER_HOLD:
                    # A linha saiu da imagem durante a curva: termina o giro
                    # atual em vez de substituir o comando por frente (0°).
                    angle = last_follow_angle
                else:
                    # Sem gap e sem linha por tempo demais, parar e mais seguro
                    # do que continuar reto para fora da pista.
                    angle = 190

                # O angulo pode mudar mesmo quando a linha apenas gira ao
                # redor da camera. O erro que importa e a distancia horizontal
                # do ponto inferior ate a bolinha central.
                error = abs(last_bottom_point.value - camera_x / 2)
                sign = 1 if angle > 0 else -1 if angle < 0 else 0
                front_reverse_assist = 0.
                # Marcadores verdes possuem uma direcao deliberada e precisam
                # do giro tanque original. O pivo traseiro fica reservado ao
                # alinhamento comum da linha, quando nao ha decisao verde.
                rear_pivot_enabled = last_rear_pivot_enabled and angle != 190

                if (not line_detected.value and rear_pivot_enabled
                        and pivot_line_lost_since is not None):
                    recovery_time = now - pivot_line_lost_since
                    front_reverse_assist = min(
                        PIVOT_RECOVERY_ASSIST_START
                        + recovery_time / PIVOT_RECOVERY_ASSIST_RAMP,
                        1.)
                    side = 'direita' if angle > 0 else 'esquerda'
                    status.value = (
                        f'Procurando linha — re dianteira {side} '
                        f'{round(front_reverse_assist * 100)}%')

                elif (rear_pivot_enabled and line_detected.value
                        and abs(angle) >= PIVOT_STALL_MIN_ANGLE
                        and error >= PIVOT_BOTTOM_MIN_ERROR_PX):
                    if sign != pivot_sign:
                        pivot_sign = sign
                        pivot_best_error = error
                        pivot_last_progress = now
                    elif error <= pivot_best_error - PIVOT_PROGRESS_PX:
                        pivot_best_error = error
                        pivot_last_progress = now
                    else:
                        stalled_for = now - pivot_last_progress
                        if stalled_for > PIVOT_STALL_TIME:
                            front_reverse_assist = min(
                                (stalled_for - PIVOT_STALL_TIME)
                                / PIVOT_STALL_RAMP_TIME,
                                1.)
                            side = 'direita' if angle > 0 else 'esquerda'
                            status.value = (
                                f'Ajudando pivo — re dianteira {side} '
                                f'{round(front_reverse_assist * 100)}%')
                else:
                    pivot_sign = 0
                    pivot_best_error = camera_x
                    pivot_last_progress = now

                steer(angle, command_speed,
                      front_reverse_assist=front_reverse_assist,
                      rear_pivot_enabled=rear_pivot_enabled)

                time_last_angles = add_time_value(time_last_angles, line_angle.value)
            elif line_status.value == "stop":
                stop_for_red()
                if mission_mode.value and not entry_armed.value:
                    # A sala de resgate já ficou para trás: esta é a faixa
                    # vermelha final da prova. O supervisor encerra a missão.
                    red_finished.value = True
                    status.value = 'Faixa vermelha final — missão concluída'
                    print("[controle] faixa vermelha final; missão concluída")
                    break
                line_status.value = "line_detected"
                continue

            elif line_status.value == "gap_detected":
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
