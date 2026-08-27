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
_motion_guard = None
_motion_observer = None


def init_steering(arduino_instance):
    global arduino, _motion_guard, _motion_observer
    arduino = arduino_instance
    # Cada processo/sessao comeca sem herdar a trava da utilizacao anterior.
    # O controle competitivo instala seu predicado logo depois desta chamada;
    # ferramentas e o processo de resgate continuam com o comportamento antigo.
    _motion_guard = None
    _motion_observer = None


def set_motion_observer(callback=None):
    """Publica o par PWM efetivamente enviado, inclusive giros bloqueantes."""

    global _motion_observer
    if callback is not None and not callable(callback):
        raise TypeError("motion observer precisa ser callable ou None")
    _motion_observer = callback


def _publish_motion(left_pwm, right_pwm):
    if _motion_observer is None:
        return
    try:
        _motion_observer(int(round(left_pwm)), int(round(right_pwm)))
    except Exception:
        # Diagnostico nunca pode impedir uma parada/comando de seguranca.
        pass


def set_motion_guard(predicate=None):
    """Instala uma permissao fail-closed avaliada antes e durante movimentos.

    O predicado pertence ao processo que possui a serial. Se ele devolver
    falso ou levantar uma excecao, qualquer comando passa a ser PARAR. Isso
    tambem interrompe esperas bloqueantes de gap/retorno em no maximo 50 ms.
    """
    global _motion_guard
    if predicate is not None and not callable(predicate):
        raise TypeError("motion guard precisa ser callable ou None")
    _motion_guard = predicate


def motion_allowed():
    if _motion_guard is None:
        return True
    try:
        return bool(_motion_guard())
    except Exception:
        return False


def _stop_if_guarded():
    if motion_allowed():
        return False
    if arduino is not None:
        arduino.parar()
    _publish_motion(0, 0)
    return True


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


def steer_line(correction, speed):
    """Envia o controle normal por pares: FE=TE e FD=TD."""
    if _stop_if_guarded():
        return False
    speed_left, speed_right = mix_line_pwm(correction, speed)
    sent = arduino.lado(speed_left, speed_right)
    _publish_motion(
        speed_left if sent is not False else 0,
        speed_right if sent is not False else 0,
    )
    return sent


def steer(angle=190., speed=.8, front_reverse_assist=0., rear_pivot_enabled=False,
          toque_frente_direita_pwm=0):
    """Transforma ângulo e velocidade no movimento das quatro rodas.

    O ângulo 190 para o robô e o ângulo 200 dá ré. Ângulos entre -180 e
    180 movem o robô para a frente: valores positivos viram para a direita
    e valores negativos viram para a esquerda.
    """

    # PARAR sempre continua permitido. Para qualquer outro comando, uma trava
    # negada transforma a tentativa em parada sem depender do chamador.
    if angle != 190 and _stop_if_guarded():
        return False

    # stop
    if angle == 190:
        sent = arduino.parar()
        _publish_motion(0, 0)
        return sent

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
        sent = arduino.parar()
        _publish_motion(0, 0)
        return sent

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

        fe = round(front_left * MAX_PWM)
        te = round(rear_left * MAX_PWM)
        fd = round(front_right * MAX_PWM)
        td = round(rear_right * MAX_PWM)
        sent = arduino.rodas(fe, te, fd, td)
        _publish_motion(
            round((fe + te) / 2) if sent is not False else 0,
            round((fd + td) / 2) if sent is not False else 0,
        )
        return sent
    else:
        left_pwm = round(speed_left * MAX_PWM)
        right_pwm = round(speed_right * MAX_PWM)
        sent = arduino.lado(left_pwm, right_pwm)
        _publish_motion(
            left_pwm if sent is not False else 0,
            right_pwm if sent is not False else 0,
        )
        return sent


def sleep_steering(duration):
    """time.sleep() que mantém o watchdog do Uno alimentado (keepalive)."""
    end = time.monotonic() + duration
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        if _stop_if_guarded():
            return False
        if arduino is not None:
            arduino.refresh()
        time.sleep(min(.05, remaining))
    return True
