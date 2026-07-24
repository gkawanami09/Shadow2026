"""Executa o retorno de 180 graus indicado por dois verdes."""

import time

from config import (T_180, T_180_CONFIRM_TIME, T_180_EXIT_BOTTOM_PX,
                    T_180_SEARCH_SPEED,
                    T_180_SEARCH_TIMEOUT, T_180_SPEED, T_180_TEST_STOP,
                    TURN_AROUND_PREROLL, TURN_AROUND_REVERSE,
                    TURN_AROUND_REVERSE_EXTRA, TURN_AROUND_SMALL_LINE,
                    camera_x)
from controle.direcao import sleep_steering, steer
from shared.dados_compartilhados import (last_bottom_point, line_detected, line_size,
                               status, terminate, timer)


def turn_around(last_turn_dir):
    """Executes the 180° and returns the NEXT turn direction ("l"/"r")."""
    # avanca por cima do marcador duplo
    steer(0, .7)
    sleep_steering(TURN_AROUND_PREROLL)

    # Pivô temporizado, pois o robô não possui giroscópio.
    steer(180 if last_turn_dir == "r" else -180, T_180_SPEED)
    sleep_steering(T_180)
    steer()

    # Modo temporario de afericao: isola somente o giro cronometrado. Mantem
    # PARAR ate o operador encerrar o programa, sem busca visual, re ou
    # retomada automatica do segue-linha mascararem o angulo obtido.
    if T_180_TEST_STOP:
        status.value = 'Teste 180 concluido — parado apos o giro'
        while not terminate.value:
            sleep_steering(.05)
        return last_turn_dir

    # Depois da parte cega, reduz a velocidade e continua no mesmo sentido ate
    # a camera confirmar a linha centralizada. Somente a posicao inferior pode
    # concluir o giro, pois ela representa diretamente a bolinha azul.
    steer(180 if last_turn_dir == "r" else -180, T_180_SEARCH_SPEED)
    status.value = 'Completando 180 — procurando linha no centro'
    search_end = time.monotonic() + T_180_SEARCH_TIMEOUT
    aligned_since = None
    while time.monotonic() < search_end:
        bottom_aligned = abs(last_bottom_point.value - camera_x / 2) <= T_180_EXIT_BOTTOM_PX
        aligned = line_detected.value and bottom_aligned
        if aligned:
            if aligned_since is None:
                aligned_since = time.monotonic()
            elif time.monotonic() - aligned_since >= T_180_CONFIRM_TIME:
                break
        else:
            aligned_since = None
        sleep_steering(.01)

    steer()

    # Dá ré até a câmera voltar a encontrar a linha.
    steer(200, .7)
    sleep_steering(TURN_AROUND_REVERSE)
    steer()

    if line_size.value < TURN_AROUND_SMALL_LINE:
        steer(200, .7)
        sleep_steering(TURN_AROUND_REVERSE_EXTRA)
        steer()


    return "r" if last_turn_dir == "l" else "l"
