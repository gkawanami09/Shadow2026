"""Acelera somente depois de confirmar uma reta visualmente estável.

Este controlador não decide curvas, verdes ou recuperações. Ele recebe o
resultado coerente já publicado pela visão e pode fazer apenas duas coisas:

* conservar a velocidade normal;
* subir aos poucos até o teto de reta rápida.

Qualquer dúvida cancela o ganho imediatamente. Quadros repetidos nunca contam
como novas confirmações.
"""

from collections import deque
import time

from config import (
    ALTURA_MINIMA_PONTO_INFERIOR_RAPIDA,
    ANGULO_MAXIMO_RETA_RAPIDA,
    AREA_MINIMA_LINHA_RAPIDA,
    ERRO_INFERIOR_RETA_RAPIDA_PX,
    FPS_MINIMO_RETA_RAPIDA,
    FRAMES_PARA_RETA_RAPIDA,
    IDADE_MAXIMA_VISAO_RAPIDA_S,
    JANELA_FPS_RETA_RAPIDA,
    LINE_FOLLOW_SPEED,
    PASSO_VELOCIDADE_RETA_RAPIDA,
    VARIACAO_ANGULO_RETA_RAPIDA,
    VARIACAO_INFERIOR_RETA_RAPIDA_PX,
    VELOCIDADE_RETA_RAPIDA,
    camera_x,
    camera_y,
)


class ControladorVelocidadeAdaptativa:
    """Libera PWM 72 apenas em frames novos de uma reta confirmada."""

    def __init__(self, relogio=None):
        self._relogio = time.monotonic if relogio is None else relogio
        self._tempos_dos_frames = deque(maxlen=JANELA_FPS_RETA_RAPIDA)
        self._ultima_sequencia = None
        self._ultimo_angulo = None
        self._ultimo_ponto_x = None
        self._frames_estaveis = 0
        self._velocidade = LINE_FOLLOW_SPEED
        self._modo_rapido = False

    @property
    def modo_rapido(self):
        return self._modo_rapido

    @property
    def velocidade(self):
        return self._velocidade

    @property
    def fps_visao(self):
        if len(self._tempos_dos_frames) < JANELA_FPS_RETA_RAPIDA:
            return 0.
        intervalo = self._tempos_dos_frames[-1] - self._tempos_dos_frames[0]
        if intervalo <= 0:
            return 0.
        return (len(self._tempos_dos_frames) - 1) / intervalo

    def _observar_frame_novo(self, resultado, condicoes_basicas):
        if resultado.sequencia <= 0:
            return False
        if resultado.sequencia == self._ultima_sequencia:
            return False

        if (
            self._tempos_dos_frames
            and resultado.publicado_em <= self._tempos_dos_frames[-1]
        ):
            # Reinício da visão ou relógio inválido: uma janela antiga não pode
            # autorizar aceleração na execução nova.
            self._tempos_dos_frames.clear()
        self._tempos_dos_frames.append(resultado.publicado_em)

        variacao_angulo_ok = (
            self._ultimo_angulo is None
            or abs(resultado.angulo - self._ultimo_angulo)
            <= VARIACAO_ANGULO_RETA_RAPIDA
        )
        variacao_ponto_ok = (
            self._ultimo_ponto_x is None
            or abs(resultado.ponto_inferior_x - self._ultimo_ponto_x)
            <= VARIACAO_INFERIOR_RETA_RAPIDA_PX
        )

        self._ultima_sequencia = resultado.sequencia
        self._ultimo_angulo = resultado.angulo
        self._ultimo_ponto_x = resultado.ponto_inferior_x

        if condicoes_basicas and variacao_angulo_ok and variacao_ponto_ok:
            self._frames_estaveis += 1
        else:
            self._frames_estaveis = 0
            self._modo_rapido = False
        return True

    def _condicoes_basicas(
        self,
        resultado,
        *,
        agora,
        velocidade_base,
        direcao,
        permitir_rapido,
    ):
        idade = agora - resultado.publicado_em
        return (
            permitir_rapido
            and velocidade_base == LINE_FOLLOW_SPEED
            and direcao == "straight"
            and resultado.linha_detectada
            and resultado.linha_a_frente
            and not resultado.candidato_verde
            and not resultado.candidato_vermelho
            and not resultado.rampa
            and 0 <= idade <= IDADE_MAXIMA_VISAO_RAPIDA_S
            and abs(resultado.angulo) <= ANGULO_MAXIMO_RETA_RAPIDA
            and abs(resultado.ponto_inferior_x - camera_x / 2)
            <= ERRO_INFERIOR_RETA_RAPIDA_PX
            and resultado.ponto_inferior_y
            >= camera_y * ALTURA_MINIMA_PONTO_INFERIOR_RAPIDA
            and resultado.area_linha >= AREA_MINIMA_LINHA_RAPIDA
        )

    def atualizar(
        self,
        resultado,
        *,
        velocidade_base,
        direcao,
        permitir_rapido,
    ):
        """Devolve uma velocidade normalizada entre 0 e 1."""
        agora = self._relogio()
        condicoes = self._condicoes_basicas(
            resultado,
            agora=agora,
            velocidade_base=velocidade_base,
            direcao=direcao,
            permitir_rapido=permitir_rapido,
        )
        frame_novo = self._observar_frame_novo(resultado, condicoes)

        # Desaceleração é imediata. Isso também cobre verde, prata, obstáculo,
        # curva, linha perdida, visão travada e qualquer velocidade especial.
        if not condicoes:
            self._frames_estaveis = 0
            self._modo_rapido = False
            self._velocidade = velocidade_base
            return self._velocidade

        pode_acelerar = (
            self._frames_estaveis >= FRAMES_PARA_RETA_RAPIDA
            and self.fps_visao >= FPS_MINIMO_RETA_RAPIDA
        )
        if not pode_acelerar:
            self._modo_rapido = False
            self._velocidade = velocidade_base
            return self._velocidade

        self._modo_rapido = True
        if frame_novo:
            self._velocidade = min(
                VELOCIDADE_RETA_RAPIDA,
                max(self._velocidade, velocidade_base)
                + PASSO_VELOCIDADE_RETA_RAPIDA,
            )
        return self._velocidade

