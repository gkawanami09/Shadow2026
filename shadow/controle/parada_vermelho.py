"""Para o robô durante o tempo definido para a faixa vermelha."""

from config import wait_time_red
from controle.direcao import sleep_steering, steer
from shared.dados_compartilhados import status, terminate


def stop_for_red():
    steer()
    for i in range(wait_time_red):
        if terminate.value:
            break

        status.value = f'Parada por vermelho: {wait_time_red - i} s restantes'
        sleep_steering(1)
