"""Converte ângulo e velocidade nos comandos dos dois lados do robô."""

import time

from config import (FRONT_ANCHORED_STEERING, FRONT_ANCHOR_FULL_ANGLE,
                    FRONT_ANCHOR_MAX_BLEND, FRONT_ANCHOR_REAR_SCALE,
                    FRONT_ANCHOR_START_ANGLE,
                    LINE_TANK_FULL_ANGLE, LINE_TANK_SPEED_REDUCTION,
                    LINE_TANK_TURN_GAIN,
                    MAX_PWM, PIVOT_FRONT_REVERSE_MIN_PWM,
                    PIVOT_FRONT_REVERSE_SCALE, left_correction, max_turn_angle,
                    right_correction)

# Instancia definida por init_steering() no processo de controle (ou nos tools).
arduino = None


def init_steering(arduino_instance):
    global arduino
    arduino = arduino_instance


def steer(angle=190., speed=.8, front_reverse_assist=0., rear_pivot_enabled=False,
          toque_frente_direita_pwm=0, center_pivot=None,
          rear_pivot_start_angle=None, rear_pivot_full_angle=None,
          rear_pivot_max_blend=None, rear_pivot_rear_scale=None):
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
        if center_pivot is None:
            # Mantem a semantica antiga para manobras especiais que ainda
            # chamam ``steer(180)`` sem declarar o modo.
            center_pivot = abs(angle) > max_turn_angle

        if center_pivot:
            # Giro no centro: reservado para verde, retorno e buscas.
            outer = min(speed * 1.2, 1.)
            if angle >= 0:
                speed_left = outer * left_correction
                speed_right = -outer * right_correction
            else:
                speed_left = -outer * left_correction
                speed_right = outer * right_correction
        else:
            # iguais e a roda interna nunca muda abruptamente para rÃ©.
            # Tank steering contínuo: cada lado recebe sua própria velocidade.
            # Erro grande admite ré no lado interno para corrigir a tempo.
            turn = max(-1., min(float(angle) / LINE_TANK_FULL_ANGLE, 1.))
            linear = speed * (1 - LINE_TANK_SPEED_REDUCTION * abs(turn))
            rotation = speed * LINE_TANK_TURN_GAIN * turn
            # O lado externo não acelera acima da velocidade pedida. Em uma
            # curva de 90 graus, ganhar PWM era justamente o que fazia o robô
            # atravessar a linha antes de conseguir fazer a correção.
            speed_left = max(
                -1., min((linear + rotation) * left_correction,
                           min(speed * left_correction, 1.)))
            speed_right = max(
                -1., min((linear - rotation) * right_correction,
                           min(speed * right_correction, 1.)))

    else:
        # angulo fora do vocabulario: para por seguranca
        return arduino.parar()

    # Para erros grandes, desloca progressivamente o centro de giro para a
    # frente do chassi. No limite, as rodas dianteiras ficam quase paradas e
    # somente a traseira gira em sentidos opostos. Isso faz a traseira buscar
    # o alinhamento apontado pela bolinha inferior sem um caso especial de 90°.
    anchor_start = (FRONT_ANCHOR_START_ANGLE if rear_pivot_start_angle is None
                    else rear_pivot_start_angle)
    anchor_full = (FRONT_ANCHOR_FULL_ANGLE if rear_pivot_full_angle is None
                   else rear_pivot_full_angle)
    anchor_blend = (FRONT_ANCHOR_MAX_BLEND if rear_pivot_max_blend is None
                    else rear_pivot_max_blend)
    anchor_rear_scale = (FRONT_ANCHOR_REAR_SCALE
                         if rear_pivot_rear_scale is None
                         else rear_pivot_rear_scale)
    if FRONT_ANCHORED_STEERING and rear_pivot_enabled and -180 <= angle <= 180 and \
            abs(angle) > anchor_start:
        span = max(anchor_full - anchor_start, 1)
        blend = min((abs(angle) - anchor_start) / span, anchor_blend)
        rear_speed = min(speed * anchor_rear_scale, 1.)

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
