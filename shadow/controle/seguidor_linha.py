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
ZIGZAG = "ZIGZAG"


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
        self._derivada_filtrada = 0.
        self._ultima_correcao = 0.
        self._ultimo_visto_em = None
        self._perdida_desde = None
        self._canto_sinal = 0
        self._canto_candidato_sinal = 0
        self._canto_candidato_frames = 0
        self._canto_alinhado_frames = 0
        self._canto_iniciado_em = None
        self._zigzag_candidato_frames = 0
        self._zigzag_ativo_ate = None
        self._ultima_saida = SaidaSegueLinha(0., TRACK, 0., 0., False)

    def suspender(self):
        """Descarta a memoria durante verde, gap ou outra manobra deliberada."""
        self.reset()

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
        limite = (
            config.LINE_CORNER_LOST_HOLD_S
            if em_canto else config.LINE_TRACK_LOST_HOLD_S
        )
        comando_valido = desde_vista <= limite
        correcao = self._ultima_correcao if comando_valido else 0.
        if comando_valido and em_canto:
            correcao = self._canto_sinal * max(
                abs(correcao), config.LINE_CORNER_MIN_CORRECTION)

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
            if evidencia and sinal:
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
        self._canto_iniciado_em = None

    def _atualizar_zigzag(
        self,
        *,
        detectado,
        linha_a_frente,
        angulo_linha,
        agora,
    ):
        """Confirma a geometria e conserva um avanco reto por poucos frames."""
        if detectado:
            self._zigzag_candidato_frames += 1
            if self._zigzag_candidato_frames >= config.ZIGZAG_CONFIRM_FRAMES:
                self._zigzag_ativo_ate = agora + config.ZIGZAG_HOLD_S
        else:
            self._zigzag_candidato_frames = 0
            # Nao carregar o atalho reto para dentro de um 90 graus real que
            # apareca imediatamente depois do zigue-zague.
            canto_forte = (
                not linha_a_frente
                and abs(angulo_linha)
                >= config.ZIGZAG_CORNER_RELEASE_HEADING_DEG
            )
            if canto_forte:
                self._zigzag_ativo_ate = None

        ativo = (
            self._zigzag_ativo_ate is not None
            and agora <= self._zigzag_ativo_ate
        )
        if not ativo:
            self._zigzag_ativo_ate = None
        return ativo

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
        zigzag_detectado=False,
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
        erro_base = (
            config.LINE_LATERAL_GAIN * erro_lateral
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

        zigzag_ativo = self._atualizar_zigzag(
            detectado=bool(zigzag_detectado),
            linha_a_frente=bool(linha_a_frente),
            angulo_linha=angulo_linha,
            agora=agora,
        )
        if zigzag_ativo:
            self._limpar_canto()
            self.estado = ZIGZAG
            correcao = 0.
            # As diagonais ignoradas nao podem produzir um pico derivativo ao
            # sair do ladrilho e reencontrar a reta.
            self._derivada_filtrada = 0.
            self._ultimo_erro_base = None
        elif zigzag_detectado:
            # Primeiro frame: ainda nao corta caminho, mas tambem nao deixa a
            # diagonal armar a memoria de um canto de 90 graus.
            self._limpar_canto()
            self.estado = TRACK
        else:
            correcao = self._atualizar_canto(
                angulo_linha=angulo_linha,
                erro_lateral=erro_lateral,
                erro_alvo=erro_alvo,
                linha_a_frente=bool(linha_a_frente),
                correcao=correcao,
                agora=agora,
            )
        correcao = max(min(correcao, 1.), -1.)
        self._ultima_correcao = correcao
        self._ultima_saida = SaidaSegueLinha(
            correcao=correcao,
            estado=self.estado,
            erro_lateral=erro_lateral,
            angulo_linha=angulo_linha,
            comando_valido=True,
        )
        return self._ultima_saida
