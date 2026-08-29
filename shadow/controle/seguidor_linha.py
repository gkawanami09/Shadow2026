"""Controle geometrico do segue-linha normal.

Este modulo nao conhece Arduino, OpenCV ou multiprocessing. Ele recebe a
geometria coerente de um unico frame e devolve uma correcao normalizada entre
-1 (pivo total para a esquerda) e +1 (pivo total para a direita).
"""

from dataclasses import dataclass
import math
import time

import config


TRACK = "TRACK"
CORNER = "CORNER"
LOST = "LOST"


@dataclass(frozen=True)
class SaidaSegueLinha:
    correcao: float
    estado: str
    erro_lateral: float
    angulo_linha: float
    comando_valido: bool

    @property
    def angulo_equivalente(self):
        """Angulo apenas para telemetria e compatibilidade visual."""
        return int(round(self.correcao * 180.))


def erros_da_geometria(
    ponto_inferior_x,
    ponto_inferior_y,
    ponto_alvo_x,
    ponto_alvo_y,
    *,
    largura=config.camera_x,
):
    """Retorna erro lateral normalizado e rumo da linha em graus."""
    meio = largura / 2.
    erro_lateral = max(min((float(ponto_inferior_x) - meio) / meio, 1.), -1.)

    dx = float(ponto_alvo_x) - float(ponto_inferior_x)
    dy_frente = float(ponto_inferior_y) - float(ponto_alvo_y)
    angulo_linha = math.degrees(math.atan2(dx, max(dy_frente, 0.)))
    angulo_linha = max(min(angulo_linha, 90.), -90.)
    return erro_lateral, angulo_linha


def angulo_para_ponto_futuro(
    ponto_futuro_x,
    ponto_futuro_y,
    *,
    largura=config.camera_x,
    altura=config.camera_y,
):
    """Rumo do centro fisico do robo ate o lookahead visivel."""
    dx = float(ponto_futuro_x) - float(largura) / 2.
    dy_frente = float(altura) - float(ponto_futuro_y)
    angulo = math.degrees(math.atan2(dx, max(dy_frente, 0.)))
    return max(min(angulo, 90.), -90.)


class ControladorSegueLinha:
    """Combina posicao, rumo, derivada e memoria de cantos de 90 graus."""

    def __init__(self, *, largura=config.camera_x):
        self.largura = float(largura)
        self.reset()

    def reset(self):
        self.estado = TRACK
        self._ultima_sequencia = -1
        self._ultimo_publicado_em = None
        self._ultimo_erro_base = None
        self._angulo_futuro_filtrado = None
        self._derivada_filtrada = 0.
        self._ultima_correcao = 0.
        self._ultimo_visto_em = None
        self._perdida_desde = None
        self._canto_sinal = 0
        self._canto_candidato_sinal = 0
        self._canto_candidato_frames = 0
        self._canto_alinhado_frames = 0
        self._canto_retorno_frames = 0
        self._canto_iniciado_em = None
        self._ultima_saida = SaidaSegueLinha(0., TRACK, 0., 0., False)

    def suspender(self):
        """Descarta a memoria durante verde, gap ou outra manobra deliberada."""
        self.reset()

    def forcar_canto(self, sinal, *, agora=None):
        """Arma a memoria de curva para uma ramificacao ja confirmada.

        O verde ja provou qual lado e a camera acabou de ver o ramo naquele
        lado. Em curvas fechadas, especialmente no Pacman, a faixa pode
        desaparecer no frame seguinte; aguardar dois frames normais deixava
        a perda ser tratada como reta/gap antes de a memoria de canto existir.
        """
        sinal = self._sinal(float(sinal))
        if not sinal:
            return False
        agora = time.monotonic() if agora is None else float(agora)
        self._canto_sinal = sinal
        self._canto_candidato_sinal = 0
        self._canto_candidato_frames = 0
        self._canto_alinhado_frames = 0
        self._canto_retorno_frames = 0
        self._canto_iniciado_em = agora
        self._ultimo_visto_em = agora
        self._perdida_desde = None
        self.estado = CORNER
        self._ultima_correcao = sinal * max(
            abs(self._ultima_correcao), config.LINE_CORNER_MIN_CORRECTION)
        return True

    @staticmethod
    def _sinal(valor):
        return 1 if valor > 0 else -1 if valor < 0 else 0

    def _saida_perdida(self, agora):
        if self._perdida_desde is None:
            self._perdida_desde = agora
        self.estado = LOST

        desde_vista = (
            float("inf") if self._ultimo_visto_em is None
            else agora - self._ultimo_visto_em
        )
        em_canto = self._canto_sinal != 0
        candidato_canto = (
            not em_canto
            and self._canto_candidato_frames > 0
            and self._canto_candidato_sinal != 0
        )
        limite = (
            config.LINE_CORNER_LOST_HOLD_S
            if em_canto else config.LINE_TRACK_LOST_HOLD_S
        )
        if candidato_canto:
            limite = config.LINE_CORNER_CANDIDATE_LOST_HOLD_S
        comando_valido = desde_vista <= limite
        # Branco em uma reta deve produzir um avanco previsivel, sem prolongar
        # a ultima correcao lateral. Em um canto confirmado, preserve o giro
        # porque zerar a correcao faria o robo escapar antes da nova reta.
        correcao = self._ultima_correcao if (
            comando_valido and (em_canto or candidato_canto)
        ) else 0.
        if comando_valido and em_canto:
            correcao = self._canto_sinal * max(
                abs(correcao), config.LINE_CORNER_MIN_CORRECTION)
        elif comando_valido and candidato_canto:
            correcao = self._canto_candidato_sinal * max(
                abs(correcao),
                config.LINE_CORNER_CANDIDATE_MIN_CORRECTION,
            )

        self._ultima_saida = SaidaSegueLinha(
            correcao=max(min(correcao, 1.), -1.),
            estado=LOST,
            erro_lateral=self._ultima_saida.erro_lateral,
            angulo_linha=self._ultima_saida.angulo_linha,
            comando_valido=comando_valido,
        )
        return self._ultima_saida

    def _atualizar_canto(
        self,
        *,
        angulo_linha,
        erro_lateral,
        erro_alvo,
        linha_a_frente,
        correcao,
        agora,
        permitir_novo_canto=True,
        cancelar_canto=False,
    ):
        sinal = self._sinal(
            angulo_linha if abs(angulo_linha) >= 5. else correcao)
        evidencia = (
            abs(angulo_linha) >= config.LINE_CORNER_ENTRY_HEADING_DEG
            or (
                not linha_a_frente
                and abs(angulo_linha) >= config.LINE_CORNER_SIDE_HEADING_DEG
                and abs(erro_alvo) >= config.LINE_CORNER_TARGET_MIN
            )
        )

        if self._canto_sinal == 0:
            if not permitir_novo_canto:
                self._canto_candidato_sinal = 0
                self._canto_candidato_frames = 0
            elif evidencia and sinal:
                if sinal == self._canto_candidato_sinal:
                    self._canto_candidato_frames += 1
                else:
                    self._canto_candidato_sinal = sinal
                    self._canto_candidato_frames = 1
            else:
                self._canto_candidato_sinal = 0
                self._canto_candidato_frames = 0

            if self._canto_candidato_frames >= config.LINE_CORNER_CONFIRM_FRAMES:
                self._canto_sinal = self._canto_candidato_sinal
                self._canto_iniciado_em = agora
                self._canto_alinhado_frames = 0

        if self._canto_sinal == 0:
            self.estado = TRACK
            return correcao

        # Um zig-zag pode parecer um canto real nos primeiros quadros. Quando
        # o ponto distante revela de forma persistente que a faixa voltou, a
        # memoria antiga nao pode continuar impondo o giro oposto ao alvo.
        self._canto_retorno_frames = (
            self._canto_retorno_frames + 1
            if cancelar_canto else 0
        )
        if (
            self._canto_retorno_frames
            >= config.LINE_CORNER_RETURN_CANCEL_FRAMES
        ):
            self._limpar_canto()
            self.estado = TRACK
            return correcao

        if (
            self._canto_iniciado_em is not None
            and agora - self._canto_iniciado_em > config.LINE_CORNER_TIMEOUT_S
        ):
            self._limpar_canto()
            self.estado = TRACK
            return correcao

        alinhada = (
            abs(angulo_linha) <= config.LINE_CORNER_EXIT_HEADING_DEG
            and abs(erro_lateral) <= config.LINE_CORNER_EXIT_LATERAL
            and abs(erro_alvo) <= config.LINE_CORNER_EXIT_TARGET
        )
        self._canto_alinhado_frames = (
            self._canto_alinhado_frames + 1 if alinhada else 0)
        if self._canto_alinhado_frames >= config.LINE_CORNER_EXIT_FRAMES:
            self._limpar_canto()
            # A queda brusca do erro e consequencia do giro concluido, nao um
            # pedido para corrigir no sentido contrario.
            self._derivada_filtrada = 0.
            self._ultimo_erro_base = None
            self.estado = TRACK
            return 0.

        self.estado = CORNER
        quase_alinhada = (
            abs(angulo_linha) <= config.LINE_CORNER_EXIT_HEADING_DEG * 1.5
            and abs(erro_lateral) <= config.LINE_CORNER_EXIT_LATERAL * 1.5
        )
        minimo = (
            config.LINE_CORNER_FINISH_CORRECTION
            if quase_alinhada else config.LINE_CORNER_MIN_CORRECTION)
        return self._canto_sinal * max(abs(correcao), minimo)

    def _limpar_canto(self):
        self._canto_sinal = 0
        self._canto_candidato_sinal = 0
        self._canto_candidato_frames = 0
        self._canto_alinhado_frames = 0
        self._canto_retorno_frames = 0
        self._canto_iniciado_em = None

    def atualizar(
        self,
        *,
        sequencia,
        publicado_em,
        linha_detectada,
        linha_a_frente,
        ponto_inferior_x,
        ponto_inferior_y,
        ponto_alvo_x,
        ponto_alvo_y,
        ponto_futuro_x=None,
        ponto_futuro_y=None,
        ponto_futuro_valido=False,
        agora=None,
    ):
        agora = time.monotonic() if agora is None else float(agora)
        publicado_em = float(publicado_em)
        frame_recente = agora - publicado_em <= config.LINE_MAX_FRAME_AGE_S
        novo_frame = int(sequencia) != self._ultima_sequencia

        if not linha_detectada or not frame_recente:
            return self._saida_perdida(agora)

        if not novo_frame:
            return self._ultima_saida

        erro_lateral, angulo_linha = erros_da_geometria(
            ponto_inferior_x,
            ponto_inferior_y,
            ponto_alvo_x,
            ponto_alvo_y,
            largura=self.largura,
        )
        meio = self.largura / 2.
        erro_alvo = max(min((float(ponto_alvo_x) - meio) / meio, 1.), -1.)
        futuro_valido = (
            bool(ponto_futuro_valido)
            and ponto_futuro_x is not None
            and ponto_futuro_y is not None
            and math.isfinite(float(ponto_futuro_x))
            and math.isfinite(float(ponto_futuro_y))
        )
        # O ponto inferior representa onde a faixa realmente cruza a base do
        # robo. Ele nunca pode desaparecer do comando: o ponto futuro antecipa
        # o rumo, mas esta parcela fecha continuamente o alinhamento fisico.
        erro_inferior = config.LINE_LATERAL_GAIN * erro_lateral
        angulo_futuro_bruto = angulo_linha
        if futuro_valido:
            angulo_futuro_bruto = angulo_para_ponto_futuro(
                ponto_futuro_x,
                ponto_futuro_y,
                largura=self.largura,
                altura=config.camera_y,
            )
            if self._angulo_futuro_filtrado is None:
                self._angulo_futuro_filtrado = angulo_futuro_bruto
            else:
                alpha = config.LINE_FUTURE_FILTER
                self._angulo_futuro_filtrado = (
                    alpha * angulo_futuro_bruto
                    + (1. - alpha) * self._angulo_futuro_filtrado
                )
            angulo_controle = self._angulo_futuro_filtrado
            erro_base = (
                erro_inferior
                + config.LINE_FUTURE_GAIN
                * math.sin(math.radians(angulo_controle))
            )
        else:
            self._angulo_futuro_filtrado = None
            angulo_controle = angulo_linha
            erro_base = (
                erro_inferior
                + config.LINE_HEADING_GAIN
                * math.sin(math.radians(angulo_linha))
            )

        derivada = 0.
        if self._ultimo_erro_base is not None and self._ultimo_publicado_em is not None:
            dt = max(min(publicado_em - self._ultimo_publicado_em, .10), .005)
            derivada_bruta = (erro_base - self._ultimo_erro_base) / dt
            alpha = config.LINE_DERIVATIVE_FILTER
            self._derivada_filtrada = (
                alpha * derivada_bruta
                + (1. - alpha) * self._derivada_filtrada)
            derivada = max(min(
                config.LINE_DERIVATIVE_GAIN * self._derivada_filtrada,
                config.LINE_DERIVATIVE_LIMIT,
            ), -config.LINE_DERIVATIVE_LIMIT)

        correcao = erro_base + derivada
        if abs(correcao) < config.LINE_CORRECTION_DEADBAND:
            correcao = 0.
        correcao = max(min(correcao, 1.), -1.)

        self._ultima_sequencia = int(sequencia)
        self._ultimo_publicado_em = publicado_em
        self._ultimo_erro_base = erro_base
        self._ultimo_visto_em = agora
        self._perdida_desde = None

        retorno_visivel = (
            futuro_valido
            and float(ponto_futuro_y)
            <= config.camera_y * config.LINE_FUTURE_RETURN_MAX_Y_RATIO
            and abs(angulo_linha) >= config.LINE_CORNER_SIDE_HEADING_DEG
            and (
                angulo_linha * angulo_futuro_bruto <= 0.
                or abs(angulo_futuro_bruto)
                <= abs(angulo_linha) * config.LINE_FUTURE_RETURN_RATIO
            )
        )
        futuro_contradiz_canto = (
            futuro_valido
            and self._canto_sinal != 0
            and float(ponto_futuro_y)
            <= config.camera_y * config.LINE_FUTURE_RETURN_MAX_Y_RATIO
            and (
                self._canto_sinal * angulo_futuro_bruto < 0.
                or abs(angulo_futuro_bruto)
                <= config.LINE_CORNER_EXIT_HEADING_DEG
            )
        )
        # O horizonte sozinho pode ficar central durante um 90 verdadeiro.
        # So solte uma curva ja confirmada quando a geometria junto da base
        # tambem tiver mudado de lado; caso contrario o robo abandona o giro
        # antes de trazer a nova reta para o centro inferior.
        base_cruzou_centro = (
            self._canto_sinal != 0
            and self._canto_sinal * erro_lateral
            < -config.LINE_CORNER_EXIT_LATERAL
        )
        rumo_local_contradiz_canto = (
            self._canto_sinal != 0
            and self._canto_sinal * angulo_linha < -5.
        )
        correcao = self._atualizar_canto(
            angulo_linha=angulo_linha,
            erro_lateral=erro_lateral,
            erro_alvo=erro_alvo,
            linha_a_frente=bool(linha_a_frente),
            correcao=correcao,
            agora=agora,
            permitir_novo_canto=not retorno_visivel,
            cancelar_canto=(
                base_cruzou_centro
                or (
                    futuro_contradiz_canto
                    and rumo_local_contradiz_canto
                )
            ),
        )
        correcao = max(min(correcao, 1.), -1.)
        self._ultima_correcao = correcao
        self._ultima_saida = SaidaSegueLinha(
            correcao=correcao,
            estado=self.estado,
            erro_lateral=erro_lateral,
            angulo_linha=angulo_controle,
            comando_valido=True,
        )
        return self._ultima_saida
