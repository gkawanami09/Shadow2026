"""Decisoes geometricas da aproximacao ao ramo marcado em verde."""

import math

import config


def correcao_aproximacao(ponto_inferior_x, *, largura=config.camera_x):
    """Centraliza somente a faixa de entrada, sem antecipar o ramo lateral."""
    x = float(ponto_inferior_x)
    if not math.isfinite(x):
        return 0.

    meio = float(largura) / 2.
    erro = max(min((x - meio) / meio, 1.), -1.)
    correcao = config.LINE_LATERAL_GAIN * erro
    limite = config.GREEN_APPROACH_MAX_CORRECTION
    correcao = max(min(correcao, limite), -limite)
    if abs(correcao) < config.LINE_CORRECTION_DEADBAND:
        return 0.
    return correcao


def ramo_pronto_para_giro(
    direcao,
    *,
    faixa_transversal_y,
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

    y = float(faixa_transversal_y)
    if not math.isfinite(y) or y < 0.:
        return False

    return y >= float(altura) * config.GREEN_APPROACH_BRANCH_MIN_Y_RATIO
