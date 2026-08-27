"""Decisoes geometricas da aproximacao ao ramo marcado em verde."""

import math

import config


def ramo_pronto_para_giro(
    direcao,
    *,
    ponto_alvo_x,
    ponto_alvo_y,
    largura=config.camera_x,
    altura=config.camera_y,
):
    """Indica quando o ramo escolhido chegou perto do eixo do robo.

    A direcao verde ja esta travada na visao. Aqui nao escolhemos outro
    contorno: apenas esperamos o alvo desse ramo aparecer no lado correto e
    suficientemente baixo na imagem para iniciar o tanque sem atravessar a
    intersecao inteira.
    """
    sinal = -1 if direcao == "left" else 1 if direcao == "right" else 0
    if not sinal:
        return False

    x = float(ponto_alvo_x)
    y = float(ponto_alvo_y)
    if not math.isfinite(x) or not math.isfinite(y):
        return False

    deslocamento_lateral = sinal * (x - float(largura) / 2.)
    alvo_baixo = y >= float(altura) * config.GREEN_APPROACH_BRANCH_MIN_Y_RATIO
    return (
        deslocamento_lateral >= config.GREEN_APPROACH_BRANCH_SIDE_MIN_PX
        and alvo_baixo
    )
