"""Confirmacao preto/prata e retomada da linha usando a camera inferior."""

import time

import cv2

import config_resgate as cfg
from controle.retomada_saida import (
    ControladorRetomadaSaida,
    ErroRetomadaSaida,
)
from visao.confirmacao_saida_linha import (
    INCONCLUSIVA,
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
    ConfirmadorFaixaSaidaLinha,
    anotar_confirmacao,
    faixa_centralizada,
    posicao_vertical_faixa,
)
from visao.continuacao_saida import anotar_analise_saida


RETOMADA_FALHOU = "retomada_falhou"


def _mover_por_tempo(
    arduino,
    acao_direcao,
    angulo,
    velocidade,
    duracao,
    epoca_serial,
    relogio,
    dormir,
):
    if acao_direcao(angulo, velocidade) is False:
        raise RuntimeError("comando da verificacao de saida nao foi enviado")
    prazo = relogio() + max(float(duracao), 0.0)
    while relogio() < prazo:
        arduino.refresh(fail_closed=True)
        if (
            not arduino.connected
            or arduino.connection_epoch != epoca_serial
        ):
            acao_direcao()
            raise RuntimeError(
                "serial mudou durante a verificacao da faixa de saida")
        dormir(min(0.02, max(prazo - relogio(), 0.0)))


def confirmar_saida_com_camera_linha(
    arduino,
    acao_direcao,
    camera_factory,
    *,
    debug=False,
    janela_debug="Shadow2026 - resgate (visao)",
    relogio=time.monotonic,
    dormir=time.sleep,
    retomada_factory=ControladorRetomadaSaida,
):
    """Devolve PRETA somente depois de achar a terceira linha.

    NAO_PRETA executa exatamente um segundo de re. Qualquer discordancia,
    timeout ou exposicao instavel falha fechada e nunca libera o segue-linha.
    """
    camera = None
    epoca_serial = arduino.connection_epoch
    inicio = None
    ultimo_log = 0.0
    ultimo_resumo = None
    confirmador = None
    confirmacao_iniciada_em = None
    hipotese_inicial = None
    rechecando = False
    deslocamento_temporal = 0.0
    ultima_posicao_faixa = None
    manter_led_aceso = False

    def parar():
        if acao_direcao() is False:
            raise RuntimeError("nao foi possivel parar na saida")

    def parar_melhor_esforco():
        try:
            parou = acao_direcao()
            if parou is False:
                arduino.parar()
        except Exception:
            try:
                arduino.parar()
            except Exception:
                pass

    def mover(angulo, velocidade, duracao):
        _mover_por_tempo(
            arduino,
            acao_direcao,
            angulo,
            velocidade,
            duracao,
            epoca_serial,
            relogio,
            dormir,
        )

    def vigiar():
        arduino.refresh(fail_closed=True)
        if (
            not arduino.connected
            or arduino.connection_epoch != epoca_serial
        ):
            raise RuntimeError(
                "serial mudou durante a confirmacao preta/prata")

    def recuar_e_retornar(resultado, duracao=None):
        nonlocal deslocamento_temporal
        if duracao is None:
            duracao = max(
                cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_S,
                max(deslocamento_temporal, 0.0)
                + cfg.EXIT_LINE_VERIFY_MISSED_REVERSE_MARGIN_S,
            )
        parar()
        mover(
            200,
            cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_SPEED,
            duracao,
        )
        deslocamento_temporal -= float(duracao)
        parar()
        return resultado

    def debug_retomada(frame, analise, fase):
        if not debug:
            return True
        canvas = anotar_analise_saida(frame, analise)
        cv2.putText(
            canvas, fase, (8, 82), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(janela_debug, canvas)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    try:
        parar()
        arduino.led("ACESO")
        print("[saida] LED ACESO: entrando na camera do segue-linha")
        camera = camera_factory()

        print(
            "[saida] camera de linha aberta com o robo PARADO; "
            f"descartando {cfg.EXIT_LINE_CAMERA_WARMUP_S:.2f} s para "
            "autoexposicao estabilizar")
        prazo_aquecimento = relogio() + cfg.EXIT_LINE_CAMERA_WARMUP_S
        while relogio() < prazo_aquecimento:
            camera.get_frame()
            vigiar()
        # Inicializacao do driver e aquecimento nao consomem o prazo fisico
        # de aproximacao da faixa.
        inicio = relogio()

        print(
            f"[saida] procurando a faixa em passos de "
            f"{cfg.EXIT_LINE_VERIFY_STEP_S:.2f} s a PWM "
            f"{cfg.EXIT_LINE_VERIFY_PWM}; votos so contam no centro e com "
            "luz estavel")

        while True:
            frame = camera.get_frame()
            agora = relogio()
            posicao_faixa = None
            if confirmador is None:
                resultado = ClassificadorFaixaSaidaLinha().classificar(
                    frame, timestamp=agora)
                decisao = None
                posicao_faixa = posicao_vertical_faixa(resultado)
                if posicao_faixa is not None:
                    ultima_posicao_faixa = posicao_faixa
                if faixa_centralizada(resultado):
                    parar()
                    dormir(cfg.EXIT_LINE_VERIFY_STEP_SETTLE_S)
                    confirmador = ConfirmadorFaixaSaidaLinha()
                    confirmacao_iniciada_em = relogio()
                    fase_confirmacao = (
                        "faixa centralizada; estabilizando exposicao")
                    print(
                        "[saida] faixa no centro util; AVANCO TRAVADO e "
                        "votacao iniciada com frames posteriores")
                else:
                    fase_confirmacao = (
                        "procurando faixa entre passos"
                        if posicao_faixa is None
                        else f"recentralizando faixa ({posicao_faixa:.0%})"
                    )
            else:
                decisao, resultado = confirmador.update(
                    frame, timestamp=agora, now=agora)
                fase_confirmacao = (
                    "rechecagem simetrica parada"
                    if rechecando else "confirmacao primaria parada")

            vigiar()

            resumo = (
                fase_confirmacao,
                resultado.classificacao,
                confirmador.votos_pretos if confirmador else 0,
                confirmador.votos_nao_pretos if confirmador else 0,
                rechecando,
            )
            if resumo != ultimo_resumo and agora - ultimo_log >= 0.15:
                print(
                    f"[saida] {fase_confirmacao}: "
                    f"{resultado.classificacao}; "
                    f"preta={confirmador.votos_pretos if confirmador else 0} "
                    f"prata={confirmador.votos_nao_pretos if confirmador else 0} "
                    f"brilho_rel={resultado.brilho_relativo:.2f} "
                    f"textura_rel={resultado.textura_relativa:.2f}")
                ultimo_resumo = resumo
                ultimo_log = agora

            if debug:
                cv2.imshow(
                    janela_debug,
                    anotar_confirmacao(frame, resultado, decisao),
                )
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    parar()
                    return None

            if decisao in (PRETA, NAO_PRETA):
                if not rechecando:
                    hipotese_inicial = decisao
                    print(
                        f"[saida] hipotese primaria={decisao}; mantendo "
                        "o robo parado para uma votacao independente")
                    parar()
                    dormir(cfg.EXIT_LINE_VERIFY_RECHECK_SETTLE_S)
                    confirmador = ConfirmadorFaixaSaidaLinha(
                        tamanho_janela=cfg.EXIT_LINE_VERIFY_RECHECK_WINDOW,
                        votos_pretos=cfg.EXIT_LINE_VERIFY_RECHECK_BLACK_VOTES,
                        votos_nao_pretos=(
                            cfg.EXIT_LINE_VERIFY_RECHECK_SILVER_VOTES),
                    )
                    confirmacao_iniciada_em = relogio()
                    rechecando = True
                    ultimo_resumo = None
                    continue

                if decisao != hipotese_inicial:
                    print(
                        "[saida] votacoes discordaram; PRETO nao sera "
                        "liberado")
                    return recuar_e_retornar(INCONCLUSIVA)

                if decisao == NAO_PRETA:
                    print(
                        "[saida] faixa PRATA reconfirmada; re de 1,00 s e "
                        "retorno aos pulsos da camera frontal")
                    return recuar_e_retornar(
                        NAO_PRETA,
                        cfg.EXIT_LINE_VERIFY_REJECT_REVERSE_S,
                    )

                print(
                    "[saida] faixa PRETA reconfirmada; medindo inclinacao e "
                    "procurando a terceira linha")
                try:
                    retomada = retomada_factory(
                        camera,
                        arduino,
                        acao_direcao,
                        relogio=relogio,
                        dormir=dormir,
                        debug_callback=(
                            debug_retomada if debug else None),
                    ).executar()
                except ErroRetomadaSaida as erro:
                    print(f"[saida] retomada falhou: {erro}; PARADO")
                    return RETOMADA_FALHOU
                manter_led_aceso = True
                print(
                    "[saida] retomada concluida: "
                    f"pose={retomada.orientacao_soleira}, "
                    f"fase={retomada.fase_encontro}; segue-linha normal "
                    "recebera o robo parado sobre a continuacao")
                return PRETA

            esgotou_aproximacao = (
                confirmador is None
                and inicio is not None
                and agora - inicio >= cfg.EXIT_LINE_VERIFY_TIMEOUT_S)
            limite_confirmacao = (
                cfg.EXIT_LINE_VERIFY_RECHECK_TIMEOUT_S
                if rechecando else cfg.EXIT_LINE_VERIFY_CONFIRM_TIMEOUT_S)
            esgotou_confirmacao = (
                confirmador is not None
                and confirmacao_iniciada_em is not None
                and agora - confirmacao_iniciada_em >= limite_confirmacao)
            if esgotou_aproximacao or esgotou_confirmacao:
                print(
                    "[saida] cor nao fechou com centro/luz estaveis; "
                    "desfazendo a aproximacao")
                return recuar_e_retornar(INCONCLUSIVA)

            if confirmador is None:
                alvo = cfg.EXIT_LINE_VERIFY_CENTER_Y_RATIO
                tolerancia = cfg.EXIT_LINE_VERIFY_CENTER_Y_TOLERANCE
                if (
                    (
                        posicao_faixa is not None
                        and posicao_faixa > alvo + tolerancia
                    )
                    or (
                        posicao_faixa is None
                        and ultima_posicao_faixa is not None
                        and ultima_posicao_faixa > alvo
                    )
                ):
                    mover(
                        200,
                        cfg.EXIT_LINE_VERIFY_SPEED,
                        cfg.EXIT_LINE_VERIFY_REVERSE_STEP_S,
                    )
                    deslocamento_temporal -= (
                        cfg.EXIT_LINE_VERIFY_REVERSE_STEP_S)
                else:
                    mover(
                        0,
                        cfg.EXIT_LINE_VERIFY_SPEED,
                        cfg.EXIT_LINE_VERIFY_STEP_S,
                    )
                    deslocamento_temporal += cfg.EXIT_LINE_VERIFY_STEP_S
                parar()
                dormir(cfg.EXIT_LINE_VERIFY_STEP_SETTLE_S)
    finally:
        parar_melhor_esforco()
        erro_limpeza = None
        if camera is not None:
            try:
                camera.close()
            except Exception as erro:
                if erro_limpeza is None:
                    erro_limpeza = erro
        if not manter_led_aceso:
            try:
                arduino.led("APAGADO")
            except Exception as erro:
                if erro_limpeza is None:
                    erro_limpeza = erro
            print("[saida] LED APAGADO: retornando a camera de resgate")
        if erro_limpeza is not None:
            raise RuntimeError(
                "falha ao liberar a camera da saida") from erro_limpeza
