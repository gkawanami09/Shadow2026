"""Converte ângulo e velocidade nos comandos dos dois lados do robô."""

import time

from config import (FRONT_ANCHORED_STEERING, FRONT_ANCHOR_FULL_ANGLE,
                    FRONT_ANCHOR_MAX_BLEND, FRONT_ANCHOR_REAR_SCALE,
                    FRONT_ANCHOR_START_ANGLE,
                    MAX_PWM, PIVOT_FRONT_REVERSE_MIN_PWM,
                    PIVOT_FRONT_REVERSE_SCALE, left_correction, max_turn_angle,
                    right_correction)

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


def mix_line_wheels(correction, speed, *, front_anchor=False):
    """Distribui o segue-linha nas quatro rodas.

    Em TRACK, preserva exatamente o comando por lados. Em um canto fechado,
    reduz a frente e concentra o giro na traseira para que o eixo da camera
    nao avance alem do vertice antes de enxergar a nova reta.
    """
    correction = max(min(float(correction), 1.), -1.)
    speed = max(min(float(speed), 1.), 0.)
    left, right = mix_line_pwm(correction, speed)
    wheels = (left, left, right, right)  # FE, TE, FD, TD

    angle = abs(correction) * 180.
    if (
        not front_anchor
        or not FRONT_ANCHORED_STEERING
        or angle <= FRONT_ANCHOR_START_ANGLE
    ):
        return wheels

    span = max(FRONT_ANCHOR_FULL_ANGLE - FRONT_ANCHOR_START_ANGLE, 1)
    blend = min(
        (angle - FRONT_ANCHOR_START_ANGLE) / span,
        FRONT_ANCHOR_MAX_BLEND,
    )
    left_limit = round(min(speed * left_correction, 1.) * MAX_PWM)
    right_limit = round(min(speed * right_correction, 1.) * MAX_PWM)

    if correction > 0:  # direita: traseira esquerda avanca e direita recua
        anchor_te, anchor_td = left_limit, -right_limit
    else:               # esquerda: traseira esquerda recua e direita avanca
        anchor_te, anchor_td = -left_limit, right_limit

    front_left = round(left * (1. - blend))
    rear_left = round(left * (1. - blend) + anchor_te * blend)
    front_right = round(right * (1. - blend))
    rear_right = round(right * (1. - blend) + anchor_td * blend)
    return front_left, rear_left, front_right, rear_right


def steer_line(correction, speed, *, front_anchor=False):
    """Envia o controle normal, ancorando a frente apenas em cantos."""
    fe, te, fd, td = mix_line_wheels(
        correction, speed, front_anchor=front_anchor)
    if fe == te and fd == td:
        return arduino.lado(fe, fd)
    return arduino.rodas(fe, te, fd, td)


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
