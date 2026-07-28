"""Ciclo principal de decisões e movimentos do segue-linha."""

import time

import config
from config import (CONTROL_MAX_ITERATIONS, GAP_AVOID_RETREAT_TIME, GAP_AVOID_SPEED,
                    GAP_ENABLED,
                    GAP_AVOID_TIMEOUT, GAP_MIN_LINE_SIZE_RETREAT,
                    GAP_MISSING_CONFIRM_TIME, GAP_REJECT_COOLDOWN,
                    GREEN_APPROACH_TIME, GREEN_TURN_EXIT_ANGLE,
                    GREEN_REVERSE_SPEED, GREEN_REVERSE_TIME,
                    GREEN_TURN_MIN_TIME, LINE_FOLLOW_SPEED,
                    LINE_LOSS_STEER_HOLD, MIN_LINE_SIZE_DEFAULT,
                    PIVOT_BOTTOM_MIN_ERROR_PX,
                    PIVOT_RECOVERY_ASSIST_RAMP,
                    PIVOT_RECOVERY_ASSIST_START, PIVOT_RECOVERY_EXIT_ANGLE,
                    PIVOT_RECOVERY_SPEED, PIVOT_RECOVERY_TIMEOUT,
                    PIVOT_PROGRESS_PX, PIVOT_STALL_MIN_ANGLE,
                    PIVOT_STALL_RAMP_TIME, PIVOT_STALL_TIME,
                    TURN_AROUND_GREEN_COOLDOWN, VISION_READY_TIMEOUT,
                    FRONT_ANCHOR_FULL_ANGLE,
                    FRONT_ANCHOR_START_ANGLE, camera_x)
from controle.orientacao_gap import drive_back_until_line, orientate_gap
from controle.parada_obstaculo import MonitorObstaculo
from controle.parada_vermelho import stop_for_red
from controle.velocidade import get_speed
from controle.direcao import init_steering, sleep_steering, steer
from controle.retorno import turn_around
from comunicacao_serial.arduino import Arduino
from shared.dados_compartilhados import (add_time_value, empty_time_arr,
                               entry_armed, entry_silver_confirmed,
                               entry_silver_detected, last_bottom_point,
                               line_ahead, line_angle, line_detected,
                               line_status, min_line_size, mission_mode,
                               ramp_ahead, red_detected, red_finished,
                               rescue_requested, status, terminate,
                               timer, turn_dir, vision_ready)


def _enter_rescue_zone(arduino):
    """Atravessa a soleira prata e entrega o robô parado, com o LED apagado.

    O tempo NÃO é a única evidência: o avanço termina assim que a faixa deixa
    de ser vista (ela passou por baixo do robô). O timeout existe apenas como
    limite de segurança para o caso de a faixa continuar visível por erro de
    detecção — sem ele o robô atravessaria a sala inteira em linha reta.
    """
    started = time.monotonic()
    steer(0, config.ENTRY_ADVANCE_SPEED)
    motivo = "timeout"
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= config.ENTRY_ADVANCE_TIMEOUT_S:
            break
        if (
            elapsed >= config.ENTRY_ADVANCE_MIN_S
            and not entry_silver_detected.value
        ):
            motivo = "faixa passou para trás"
            break
        sleep_steering(.02)

    steer()  # PARAR antes de qualquer outra coisa
    # O LED só pode ser apagado enquanto esta serial ainda existe. O processo
    # de resgate reafirma o comando assim que abre a serial dele.
    arduino.led("APAGADO")
    entry_armed.value = False
    print(
        f"[controle] entrada concluída ({motivo}, {elapsed:.2f} s); "
        "PARAR enviado e LED APAGADO — liberando a serial para o resgate")


def control_loop():
    arduino = Arduino()
    init_steering(arduino)
    steer()  # motores parados desde o inicio
    arduino.led("ACESO")
    print("[controle] LED ACESO: modo segue-linha")

    last_turn_dir = "l"

    time_last_angles = empty_time_arr()

    timer.set_timer("ramp_ahead", .01)

    # espera a visao publicar o primeiro frame processado
    wait_start = time.perf_counter()
    while not vision_ready.value and not terminate.value:
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
    green_armed = True
    green_rearm_after = 0.
    monitor_obstaculo = MonitorObstaculo()

    try:
        while not terminate.value:

            # Segurança frontal independente da visão. Duas de três leituras
            # ultrassônicas precisam confirmar até 10 cm. Depois disso a
            # parada fica travada até o programa ser encerrado.
            if (
                config.OBSTACLE_STOP_ENABLED
                and monitor_obstaculo.atualizar(arduino)
            ):
                steer()
                distancia_cm = (
                    monitor_obstaculo.distancia_confirmada_mm / 10.0)
                status.value = (
                    f'Obstáculo confirmado a {distancia_cm:.1f} cm — PARADO')
                print(
                    "[controle] obstáculo confirmado a "
                    f"{distancia_cm:.1f} cm; parada de segurança travada")
                while not terminate.value:
                    arduino.refresh()
                    time.sleep(.05)
                break

            # Faixa prata de entrada. Só existe no modo de missão completa;
            # rodando `shadow/main.py` sozinho este bloco nunca é atingido.
            if (mission_mode.value and entry_armed.value
                    and entry_silver_confirmed.value):
                status.value = 'Faixa prata confirmada — entrando na sala'
                print("[controle] faixa PRATA confirmada; entrando na sala")
                _enter_rescue_zone(arduino)
                rescue_requested.value = True
                break

            # Estado normal do segue-linha.
            if line_status.value == "line_detected":

                gap_allowed = GAP_ENABLED and time.monotonic() >= gap_retry_after
                if (gap_allowed and not line_detected.value
                        and not line_ahead.value and not ramp_ahead.value):
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
                if turn_dir.value == "turn_around":
                    status.value = f'Girando 180° para a {"direita" if last_turn_dir == "r" else "esquerda"}'

                    last_turn_dir = turn_around(last_turn_dir)
                    # O filtro visual pode degradar "dois verdes" para apenas
                    # left/right por alguns frames. Nao iniciar uma segunda
                    # manobra com essa leitura residual.
                    green_direction = None
                    green_turn_started = None
                    green_reverse_until = None
                    green_armed = False
                    green_rearm_after = (
                        time.monotonic() + TURN_AROUND_GREEN_COOLDOWN)
                    continue

                status.value = 'Seguindo Linha'

                now = time.monotonic()

                if (time.monotonic() >= green_rearm_after
                        and turn_dir.value == "straight"
                        and green_direction is None):
                    green_armed = True

                if (green_armed and green_direction is None
                        and turn_dir.value in ("left", "right")):
                    green_direction = turn_dir.value
                    green_approach_until = now + GREEN_APPROACH_TIME
                    green_turn_started = None
                    green_reverse_until = None
                    green_armed = False

                if green_direction is not None:
                    # Recuperacao de linha do pivo nunca pode vazar para a
                    # manobra deliberada do marcador verde.
                    pivot_last_direction = 0
                    pivot_line_lost_since = None

                if line_detected.value:
                    last_line_seen = now
                    last_follow_angle = line_angle.value
                    last_rear_pivot_enabled = turn_dir.value == "straight"

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

                command_speed = get_speed(line_angle.value)

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
                        angle = line_angle.value if line_detected.value else 190
                        last_rear_pivot_enabled = True
                elif green_direction is not None and now < green_approach_until:
                    # A direcao ja foi memorizada: atravessa o marcador reto
                    # antes de iniciar qualquer rotacao.
                    angle = 0
                    command_speed = LINE_FOLLOW_SPEED
                    last_rear_pivot_enabled = False
                    status.value = f'Verde {green_direction} — avancando antes do giro'
                elif green_direction is not None:
                    if green_turn_started is None:
                        green_turn_started = now
                    angle = -180 if green_direction == "left" else 180
                    last_rear_pivot_enabled = False
                    status.value = f'Verde {green_direction} — girando tanque'

                    if (now - green_turn_started >= GREEN_TURN_MIN_TIME
                            and turn_dir.value == "straight"
                            and line_detected.value
                            and abs(line_angle.value) <= GREEN_TURN_EXIT_ANGLE):
                        green_reverse_until = now + GREEN_REVERSE_TIME
                        angle = 200
                        command_speed = GREEN_REVERSE_SPEED
                        last_rear_pivot_enabled = False
                        status.value = 'Verde concluido — dando re curta'
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
        status.value = "Parado"
        try:
            steer()  # PARAR
        finally:
            arduino.close()
