"""Decisoes geometricas da aproximacao ao ramo marcado em verde."""

import math

import config


def ramo_travado_recente(resultado, branch_token, *, agora):
    """Valida identidade, atualidade e coordenada no mesmo frame atomico."""

    try:
        esperado = int(branch_token)
        recebido = int(resultado.locked_branch_token)
        idade = float(agora) - float(resultado.publicado_em)
        ponto_x = float(resultado.locked_branch_bottom_x)
        ponto_y = float(resultado.locked_branch_bottom_y)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        esperado > 0
        and recebido == esperado
        and bool(resultado.locked_branch_valid)
        and 0. <= idade <= config.LINE_MAX_FRAME_AGE_S
        and math.isfinite(ponto_x)
        and 0. <= ponto_x <= config.camera_x
        and math.isfinite(ponto_y)
        and config.camera_y * config.GREEN_LOCKED_BRANCH_MIN_Y_RATIO
        <= ponto_y <= config.camera_y
    )


def juncao_topologica_realmente_ausente(resultado, *, agora):
    """Prova atomica e recente de ausencia da juncao bruta."""

    try:
        sequencia = int(resultado.sequencia)
        idade = float(agora) - float(resultado.publicado_em)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        sequencia > 0
        and 0. <= idade <= config.LINE_MAX_FRAME_AGE_S
        and not bool(resultado.juncao_topologica_visivel)
    )


def saida_topologica_real_estavel(resultado, *, agora):
    """Prova atomica de que o robo deixou a juncao sobre linha central.

    A visibilidade filtrada do evento confirmado nao serve para este gate:
    ela pode expirar quando um frame da mesma juncao discorda da direcao ja
    travada. ``ResultadoVisaoRapida`` carrega linha, ponto inferior e presenca
    topologica crus do mesmo frame; propagacao curta conta como presenca.
    """

    try:
        ponto_x = float(resultado.ponto_inferior_x)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        juncao_topologica_realmente_ausente(resultado, agora=agora)
        and bool(resultado.linha_detectada)
        and math.isfinite(ponto_x)
        and abs(ponto_x - config.camera_x / 2)
        <= config.GREEN_TURN_CENTER_TOLERANCE_PX
    )


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


def correcao_ramo_reto(
    ponto_ramo_x,
    ponto_inferior_x,
    *,
    largura=config.camera_x,
    peso_ramo=config.GREEN_STRAIGHT_TARGET_WEIGHT,
):
    """Segue o ramo STRAIGHT escolhido no referencial da linha de entrada.

    ``ponto_ramo_x`` vem do grafo topologico retificado. O ponto inferior
    continua participando apenas para manter a faixa de entrada sob o centro
    do chassi. O topo do contorno legado nunca entra nesta decisao.
    """
    try:
        alvo = float(ponto_ramo_x)
        inferior = float(ponto_inferior_x)
        largura = float(largura)
        peso = float(peso_ramo)
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(valor) for valor in (alvo, inferior, largura, peso))
        or largura <= 0.
        or not 0. <= alvo <= largura
        or not 0. <= inferior <= largura
        or not 0. <= peso <= 1.
    ):
        return None

    meio = largura / 2.
    erro_alvo = max(min((alvo - meio) / meio, 1.), -1.)
    erro_inferior = max(min((inferior - meio) / meio, 1.), -1.)
    erro = peso * erro_alvo + (1. - peso) * erro_inferior
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


def deve_iniciar_giro_verde(
    quadros_transversais,
    *,
    agora,
    limite_aproximacao,
    linha_recente,
):
    """Compatibilidade: apenas geometria confirmada inicia o giro.

    O relógio agora é exclusivamente um limite de segurança que leva a
    FAULT_STOP; nunca mais é um gatilho para movimentar os motores.
    """
    del agora, limite_aproximacao, linha_recente
    return bool(
        int(quadros_transversais) >= config.GREEN_BRANCH_CONFIRM_FRAMES
    )


def ramo_marcado_visto_pela_camera(
    linha_recente,
    erro_inferior,
    lado_esperado,
):
    """Confirma que a faixa escolhida chegou pela lateral correta.

    O MPU mede quanto o chassi girou, mas nao distingue a faixa de entrada do
    ramo marcado. Autorizar o controle visual apenas pelo yaw fazia a linha de
    entrada reassumir o comando no meio do 90 e desfazer o giro verde.
    """
    try:
        erro = float(erro_inferior)
        lado = float(lado_esperado)
    except (TypeError, ValueError):
        return False
    return bool(
        linha_recente
        and math.isfinite(erro)
        and lado in (-1., 1.)
        and lado * erro >= config.GREEN_TURN_SIDE_MIN_ERROR_PX
    )


def controle_visual_verde_liberado(ramo_camera_visto, comando_valido):
    """So entrega os motores ao seguidor depois de ver o ramo marcado."""
    return bool(ramo_camera_visto and comando_valido)


def correcao_reaquisicao_verde(
    ponto_inferior_x,
    lado_esperado,
    *,
    largura=config.camera_x,
):
    """Traz o ramo travado ao centro sem devolver autoridade ao seguidor.

    O sinal vem exclusivamente da decisao imutavel. Fora da zona central a
    magnitude cai progressivamente de tanque para diferencial; dentro dela o
    robo segue reto enquanto tres frames novos confirmam o alinhamento. Se a
    linha ja cruzou o centro, nunca se inventa uma correcao oposta.
    """
    try:
        x = float(ponto_inferior_x)
        lado = int(lado_esperado)
        largura = float(largura)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(x)
        or not math.isfinite(largura)
        or largura <= 0.
        or not 0. <= x <= largura
        or lado not in (-1, 1)
    ):
        return None

    meio = largura / 2.
    erro_assinado = lado * (x - meio)
    tolerancia = float(config.GREEN_TURN_CENTER_TOLERANCE_PX)
    if erro_assinado <= tolerancia:
        return 0.

    alcance = max(meio - tolerancia, 1.)
    progresso = min(max((erro_assinado - tolerancia) / alcance, 0.), 1.)
    minimo = float(config.GREEN_REACQUIRE_MIN_CORRECTION)
    maximo = float(config.GREEN_REACQUIRE_MAX_CORRECTION)
    magnitude = minimo + (maximo - minimo) * progresso
    return lado * min(max(magnitude, 0.), 1.)


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


def alinhamento_verde_pode_concluir(
    erro_inferior,
    erro_assinado_anterior,
    lado_esperado,
    progresso_mpu,
):
    """Exige giro material antes de aceitar a linha central como saida.

    Sem essa trava, a linha de entrada ainda central podia encerrar a manobra
    logo depois de o MPU apenas armar a procura do ramo. Sem leitura do MPU, a
    camera continua suficiente: nesse caso o ramo obrigatoriamente ja apareceu
    no lado marcado antes de chegar aqui.
    """
    if progresso_mpu is not None:
        progresso = float(progresso_mpu)
        if (not math.isfinite(progresso)
                or progresso < config.GREEN_MPU_COMPLETION_MIN_DEG):
            return False
    return ramo_chegou_ao_centro(
        erro_inferior,
        erro_assinado_anterior,
        lado_esperado,
    )


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
    return abs((atual - inicial + 180.) % 360. - 180.)
