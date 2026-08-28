"""Converte ângulo e velocidade nos comandos dos dois lados do robô."""

import time

from config import (FRONT_ANCHORED_STEERING, FRONT_ANCHOR_FULL_ANGLE,
                    FRONT_ANCHOR_MAX_BLEND, FRONT_ANCHOR_REAR_SCALE,
                    FRONT_ANCHOR_START_ANGLE,
                    MAX_PWM, PIVOT_FRONT_REVERSE_MIN_PWM,
                    PIVOT_FRONT_REVERSE_SCALE, LINE_TANK_ENTRY_CORRECTION,
                    LINE_TANK_EXIT_CORRECTION, left_correction,
                    max_turn_angle, right_correction)

# Instancia definida por init_steering() no processo de controle (ou nos tools).
arduino = None


def init_steering(arduino_instance):
    global arduino
    arduino = arduino_instance


def mix_line_pwm(correction, speed):
    """Mistura continua do segue-linha, sem separar frente e traseira.

    ``correction`` usa -1..1. Em zero os dois lados avancam; em modulo 0.5 a
    roda interna para; em modulo 1 ela chega ao pivo tanque. O lado externo
    permanece no PWM base, evitando o salto que existia acima de 110 graus.
    """
    correction = max(min(float(correction), 1.), -1.)
    speed = max(min(float(speed), 1.), 0.)
    outer = speed
    inner = speed * (1. - 2. * abs(correction))

    if correction >= 0:
        speed_left, speed_right = outer, inner
    else:
        speed_left, speed_right = inner, outer

    speed_left = max(min(speed_left * left_correction, 1.), -1.)
    speed_right = max(min(speed_right * right_correction, 1.), -1.)
    return round(speed_left * MAX_PWM), round(speed_right * MAX_PWM)


def atualizar_tanque_curva_fechada(estado, correction, ativo=False):
    """Histerese do tanque: entra forte e sai perto do alinhamento."""
    if estado not in ("CORNER", "LOST"):
        return False
    limite = (
        LINE_TANK_EXIT_CORRECTION if ativo
        else LINE_TANK_ENTRY_CORRECTION
    )
    return abs(float(correction)) >= limite


def mix_tank_pwm(correction, speed):
    """Gira no centro do chassi com os dois lados em sentidos opostos."""
    correction = max(min(float(correction), 1.), -1.)
    speed = max(min(float(speed), 1.), 0.)
    pwm_esquerda = round(speed * left_correction * MAX_PWM)
    pwm_direita = round(speed * right_correction * MAX_PWM)
    if correction > 0.:
        return pwm_esquerda, -pwm_direita
    if correction < 0.:
        return -pwm_esquerda, pwm_direita
    return 0, 0


def steer_line(correction, speed, *, tank=False):
    """Envia o controle normal por pares: FE=TE e FD=TD."""
    misturador = mix_tank_pwm if tank else mix_line_pwm
    speed_left, speed_right = misturador(correction, speed)
    return arduino.lado(speed_left, speed_right)


def steer(angle=190., speed=.8, front_reverse_assist=0., rear_pivot_enabled=False,
          toque_frente_direita_pwm=0):
    """Transforma ângulo e velocidade no movimento das quatro rodas.

    O ângulo 190 para o robô e o ângulo 200 dá ré. Ângulos entre -180 e
    180 movem o robô para a frente: valores positivos viram para a direita
    e valores negativos viram para a esquerda.
    """

    # stop
    if angle == 190:
        return arduino.parar()

    # backward
    elif angle == 200:
        speed_left = -max(speed * left_correction, 0)
        speed_right = -max(speed * right_correction, 0)

    # forward
    elif -180 <= angle <= 180:

        # right
        if angle >= 0:
            if angle > max_turn_angle:
                # pivot: roda interna (direita) inverte o sentido
                speed_left = min(speed * left_correction * 1.2, 1)
                speed_right = -min(speed * right_correction * 1.2, 1)
            else:
                speed_left = min(speed * left_correction, 1)
                speed_right = min(speed * right_correction * ((max_turn_angle - angle) / max_turn_angle), 1)

        # left
        else:
            if angle < -max_turn_angle:
                # pivot: roda interna (esquerda) inverte o sentido
                speed_left = -min(speed * left_correction * 1.2, 1)
                speed_right = min(speed * right_correction * 1.2, 1)
            else:
                speed_left = min(speed * left_correction * ((max_turn_angle + angle) / max_turn_angle), 1)
                speed_right = min(speed * right_correction, 1)

    else:
        # angulo fora do vocabulario: para por seguranca
        return arduino.parar()

    # Para erros grandes, desloca progressivamente o centro de giro para a
    # frente do chassi. No limite, as rodas dianteiras ficam quase paradas e
    # somente a traseira gira em sentidos opostos. Isso faz a traseira buscar
    # o alinhamento apontado pela bolinha inferior sem um caso especial de 90°.
    if FRONT_ANCHORED_STEERING and rear_pivot_enabled and -180 <= angle <= 180 and \
            abs(angle) > FRONT_ANCHOR_START_ANGLE:
        span = max(FRONT_ANCHOR_FULL_ANGLE - FRONT_ANCHOR_START_ANGLE, 1)
        blend = min((abs(angle) - FRONT_ANCHOR_START_ANGLE) / span,
                    FRONT_ANCHOR_MAX_BLEND)
        rear_speed = min(speed * FRONT_ANCHOR_REAR_SCALE, 1.)

        if angle > 0:  # direita: traseira esquerda avanca, direita recua
            anchor_te, anchor_td = rear_speed, -rear_speed
        else:          # esquerda: traseira direita avanca, esquerda recua
            anchor_te, anchor_td = -rear_speed, rear_speed

        front_left = speed_left * (1 - blend)
        front_right = speed_right * (1 - blend)
        rear_left = speed_left * (1 - blend) + anchor_te * blend
        rear_right = speed_right * (1 - blend) + anchor_td * blend

        # Se o pivo nao estiver aproximando a linha do centro, ajuda apenas a
        # roda dianteira do lado interno da curva. O controle externo fornece
        # assistencia continua em [0, 1], nunca uma manobra temporizada fixa.
        assist = min(max(float(front_reverse_assist), 0.), 1.)
        front_reverse = min(max(speed * PIVOT_FRONT_REVERSE_SCALE,
                                PIVOT_FRONT_REVERSE_MIN_PWM / MAX_PWM), 1.)
        if angle > 0:  # direita: re somente na dianteira direita
            front_right = ((1 - assist) * front_right
                           - assist * front_reverse)
        else:          # esquerda: re somente na dianteira esquerda
            front_left = ((1 - assist) * front_left
                          - assist * front_reverse)

        # Na saida do resgate, pequenos impulsos na dianteira direita fazem
        # o chassi continuar progredindo enquanto a traseira puxa a curva.
        # O parametro e opt-in e nao altera o segue-linha normal.
        toque = float(toque_frente_direita_pwm) / MAX_PWM
        front_right = min(max(front_right + toque, -1.), 1.)

        return arduino.rodas(
            round(front_left * MAX_PWM),
            round(rear_left * MAX_PWM),
            round(front_right * MAX_PWM),
            round(rear_right * MAX_PWM),
        )
    else:
        return arduino.lado(
            round(speed_left * MAX_PWM),
            round(speed_right * MAX_PWM),
        )


def sleep_steering(duration):
    """time.sleep() que mantém o watchdog do Uno alimentado (keepalive)."""
    end = time.monotonic() + duration
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        if arduino is not None:
            arduino.refresh()
        time.sleep(min(.05, remaining))
