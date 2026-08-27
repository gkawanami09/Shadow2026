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


def ramo_chegou_ao_centro(
    erro_inferior,
    erro_assinado_anterior,
    lado_esperado,
):
    """Aceita tanto a zona central quanto um salto que cruzou essa zona."""
    erro = float(erro_inferior)
    if not math.isfinite(erro):
        return False
    if abs(erro) <= config.GREEN_TURN_CENTER_TOLERANCE_PX:
        return True
    if erro_assinado_anterior is None:
        return False

    erro_assinado = float(lado_esperado) * erro
    return float(erro_assinado_anterior) > 0. and erro_assinado <= 0.


def progresso_giro_mpu(yaw_inicial, yaw_atual):
    """Retorna quantos graus o chassi girou desde o inicio da manobra.

    O yaw do MPU6050 e relativo e pode derivar ao longo da prova, mas a
    diferenca entre duas amostras de um giro curto independe desse zero.
    """
    if yaw_inicial is None or yaw_atual is None:
        return None
    inicial = float(yaw_inicial)
    atual = float(yaw_atual)
    if not math.isfinite(inicial) or not math.isfinite(atual):
        return None
    return abs(atual - inicial)
