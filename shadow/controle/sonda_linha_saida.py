"""Teste curto da abertura usando somente a camera inferior."""

import time

import config_resgate as cfg
from controle.retomada_saida import ControladorRetomadaSaida, ErroRetomadaSaida
from controle.saida_parede_resgate import ResultadoSondaLinha
from visao.captura import LineCamera
from visao.confirmacao_saida_linha import (
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
    faixa_pronta_para_confirmacao,
)


RETOMADA_FALHOU = "retomada_falhou"
LINHA_NAO_ENCONTRADA = "linha_nao_encontrada"


def testar_abertura_com_camera_linha(
    arduino,
    acao_direcao,
    *,
    debug=False,
    relogio=time.monotonic,
    dormir=time.sleep,
    camera_factory=LineCamera,
):
    """Procura exclusivamente preto durante poucos pulsos dentro do vao.

    A funcao e chamada somente com a camera frontal ja fechada. Qualquer
    faixa horizontal que chegue a regiao de confirmacao e nao seja PRETA
    rejeita o vao, sem tentar chamar essa faixa de prata. O tempo real de
    avanco e devolvido para o controlador desfazer a manobra.
    """

    camera = None
    avancado_s = 0.0
    preto_hits = 0
    epoca_serial = arduino.connection_epoch
    sucesso = False

    def vigiar():
        arduino.refresh(fail_closed=True)
        if (
            not arduino.connected
            or arduino.connection_epoch != epoca_serial
        ):
            raise RuntimeError("serial mudou durante a sonda da saida")

    def parar():
        if acao_direcao() is False:
            raise RuntimeError("nao foi possivel parar na sonda da saida")

    try:
        parar()
        camera = camera_factory()
        fim_aquecimento = relogio() + cfg.EXIT_LINE_CAMERA_WARMUP_S
        while relogio() < fim_aquecimento:
            camera.get_frame()
            vigiar()

        classificador = ClassificadorFaixaSaidaLinha()
        inicio = relogio()
        while relogio() - inicio < cfg.SAIDA_PAREDE_TIMEOUT_SONDA_LINHA_S:
            inicio_pulso = relogio()
            pulso_em_movimento = True
            if acao_direcao(
                0,
                cfg.SAIDA_PAREDE_PWM_SONDA_LINHA / 120.0,
            ) is False:
                raise RuntimeError("nao foi possivel avancar na sonda da saida")
            fim_pulso = min(
                inicio + cfg.SAIDA_PAREDE_TIMEOUT_SONDA_LINHA_S,
                inicio_pulso + cfg.SAIDA_PAREDE_PULSO_SONDA_LINHA_S,
            )
            while relogio() < fim_pulso:
                frame = camera.get_frame()
                agora = relogio()
                resultado = classificador.classificar(frame, timestamp=agora)
                vigiar()
                if faixa_pronta_para_confirmacao(resultado):
                    # O recuo posterior deve cobrir tambem a fracao deste
                    # pulso que ja ocorreu antes de a faixa aparecer.
                    avancado_s += max(agora - inicio_pulso, 0.0)
                    pulso_em_movimento = False
                    parar()
                    if resultado.classificacao == PRETA:
                        preto_hits += 1
                        # A segunda imagem e observada com o chassi parado;
                        # movimento nunca acumula dois votos para a saida.
                        frame_parado = camera.get_frame()
                        resultado_parado = classificador.classificar(
                            frame_parado, timestamp=relogio())
                        vigiar()
                        if (
                            faixa_pronta_para_confirmacao(resultado_parado)
                            and resultado_parado.classificacao == PRETA
                        ):
                            preto_hits += 1
                        else:
                            preto_hits = 0
                    else:
                        return ResultadoSondaLinha(NAO_PRETA, avancado_s)
                    if preto_hits >= cfg.SAIDA_PAREDE_CONFIRMACOES_PRETO:
                        try:
                            ControladorRetomadaSaida(
                                camera,
                                arduino,
                                acao_direcao,
                            ).executar()
                        except ErroRetomadaSaida:
                            return ResultadoSondaLinha(
                                RETOMADA_FALHOU, avancado_s)
                        sucesso = True
                        return ResultadoSondaLinha(PRETA, avancado_s)
                    break
                dormir(0.005)

            agora = relogio()
            if pulso_em_movimento:
                avancado_s += max(agora - inicio_pulso, 0.0)
            parar()
            vigiar()

        return ResultadoSondaLinha(LINHA_NAO_ENCONTRADA, avancado_s)
    finally:
        try:
            parar()
        except Exception:
            pass
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
