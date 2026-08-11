"""Confirma se a faixa de saida e preta usando a camera do segue-linha.

Esta verificacao acontece somente depois do deposito vermelho. A camera de
resgate ja encontrou e aproximou o robo da soleira; aqui a camera apontada
para o chao decide entre a faixa preta e a prata.

As duas evidencias usadas sao independentes:

* faixa transversal: uma borda horizontal ocupa boa parte da imagem;
* aparencia: preto e escuro e quase liso, enquanto prata tem reflexos e
  textura forte.

Um unico frame nunca decide. Preto e prata precisam obter maioria dentro das
cinco imagens novas; resultados inconclusivos nao contam para nenhum lado.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import config
import config_resgate as cfg


PRETA = "preta"
NAO_PRETA = "nao_preta"
INCONCLUSIVA = "inconclusiva"


@dataclass(frozen=True)
class ResultadoFaixaSaidaLinha:
    classificacao: str
    faixa_presente: bool
    textura: float
    preenchimento_borda: float
    altura_preta_ratio: float
    confianca: float
    bbox: tuple
    timestamp: float


def _maior_sequencia(flags):
    melhor_inicio = -1
    melhor_fim = -1
    melhor_tamanho = 0
    inicio = None
    for indice, ativo in enumerate(flags):
        if ativo and inicio is None:
            inicio = indice
        if not ativo and inicio is not None:
            tamanho = indice - inicio
            if tamanho > melhor_tamanho:
                melhor_inicio, melhor_fim = inicio, indice - 1
                melhor_tamanho = tamanho
            inicio = None
    if inicio is not None:
        tamanho = len(flags) - inicio
        if tamanho > melhor_tamanho:
            melhor_inicio, melhor_fim = inicio, len(flags) - 1
    return melhor_inicio, melhor_fim


def posicao_vertical_faixa(resultado):
    """Centro vertical normalizado da faixa encontrada, ou ``None``."""
    if resultado is None or not resultado.faixa_presente:
        return None
    _x, y, _w, h = resultado.bbox
    if h <= 0:
        return None
    return float(
        (float(y) + float(h) / 2.0)
        / max(float(config.camera_y), 1.0)
    )


def faixa_centralizada(resultado):
    """A faixa ja esta no centro util para comparar preto com prata?"""
    posicao = posicao_vertical_faixa(resultado)
    return (
        posicao is not None
        and abs(posicao - cfg.EXIT_LINE_VERIFY_CENTER_Y_RATIO)
        <= cfg.EXIT_LINE_VERIFY_CENTER_Y_TOLERANCE
    )


class ClassificadorFaixaSaidaLinha:
    """Diferencia a faixa preta da prata no ponto final da aproximacao."""

    def __init__(self):
        self.last_reason = "inicio"

    def classificar(self, frame_bgr, timestamp=None):
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("a confirmacao da saida exige um frame BGR")

        timestamp = 0.0 if timestamp is None else float(timestamp)
        frame = cv2.resize(
            frame_bgr,
            (config.camera_x, config.camera_y),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        valor = hsv[:, :, 2]
        altura, largura = valor.shape

        kernel = np.ones((7, 7), dtype=np.uint8)
        maximo = cv2.dilate(valor, kernel)
        minimo = cv2.erode(valor, kernel)
        variacao_local = cv2.subtract(maximo, minimo)
        topo_textura_global = int(round(
            altura * cfg.EXIT_LINE_VERIFY_TEXTURE_ROI_TOP))
        base_textura_global = int(round(
            altura * cfg.EXIT_LINE_VERIFY_TEXTURE_ROI_BOTTOM))
        roi_textura_global = variacao_local[
            topo_textura_global:base_textura_global, :]
        textura_global = float(np.median(roi_textura_global))

        # Uma fronteira horizontal extensa mostra que a soleira entrou no
        # quadro. A diferenca usa duas linhas de distancia para nao depender
        # de uma borda de apenas um pixel.
        valor_i16 = valor.astype(np.int16)
        diferenca = np.abs(valor_i16[2:, :] - valor_i16[:-2, :])
        preenchimento_por_linha = np.mean(
            diferenca >= cfg.EXIT_LINE_VERIFY_EDGE_MIN,
            axis=1,
        )
        topo_borda = int(round(altura * 0.10))
        base_borda = min(int(round(altura * 0.93)), diferenca.shape[0])
        trecho_borda = preenchimento_por_linha[topo_borda:base_borda]
        if trecho_borda.size:
            indice_borda = int(np.argmax(trecho_borda)) + topo_borda
            preenchimento_borda = float(
                preenchimento_por_linha[indice_borda])
        else:
            indice_borda = altura // 2
            preenchimento_borda = 0.0

        # Mede a aparencia logo depois da borda transversal encontrada. Isso
        # acompanha a faixa quando o robo a leva ao centro da imagem. A janela
        # fixa anterior podia ficar acima da prata e medir apenas piso liso.
        base_textura = min(
            indice_borda + max(
                int(round(
                    altura * cfg.EXIT_LINE_VERIFY_TEXTURE_BAND_HEIGHT_RATIO
                )),
                1,
            ),
            altura,
        )
        roi_textura = variacao_local[indice_borda:base_textura, :]
        textura_faixa = (
            float(np.median(roi_textura))
            if roi_textura.size else 0.0
        )
        textura = max(textura_global, textura_faixa)

        # A faixa preta real das fotos ocupa varias linhas vizinhas. Exigir
        # baixa variacao local elimina a prata que ficou escura por reflexo.
        mascara_preta_lisa = (
            (valor <= cfg.EXIT_LINE_VERIFY_DARK_VALUE_MAX)
            & (variacao_local <= cfg.EXIT_LINE_VERIFY_DARK_LOCAL_MAX)
        )
        preenchimento_preto = np.mean(mascara_preta_lisa, axis=1)
        linhas_pretas = (
            preenchimento_preto >= cfg.EXIT_LINE_VERIFY_DARK_ROW_FILL)
        inicio_preto, fim_preto = _maior_sequencia(linhas_pretas)
        altura_preta = (
            fim_preto - inicio_preto + 1 if inicio_preto >= 0 else 0)
        altura_preta_ratio = altura_preta / max(float(altura), 1.0)

        faixa_presente = bool(
            preenchimento_borda >= cfg.EXIT_LINE_VERIFY_EDGE_FILL
            or altura_preta_ratio
            >= cfg.EXIT_LINE_VERIFY_DARK_MIN_HEIGHT_RATIO
        )
        preta = bool(
            faixa_presente
            and altura_preta_ratio
            >= cfg.EXIT_LINE_VERIFY_DARK_MIN_HEIGHT_RATIO
            and textura_global <= cfg.EXIT_LINE_VERIFY_BLACK_TEXTURE_MAX
            and textura_faixa
            <= cfg.EXIT_LINE_VERIFY_BLACK_LOCAL_TEXTURE_MAX
        )
        prata = bool(
            faixa_presente
            and (
                textura_global >= cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_MIN
                or textura_faixa >= cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_MIN
            )
        )
        margem_borda = max(int(round(altura * 0.08)), 1)
        bbox_borda = (
            0,
            max(indice_borda - margem_borda, 0),
            largura,
            min(margem_borda * 2 + 1, altura),
        )

        if preta:
            classificacao = PRETA
            separacao = (
                cfg.EXIT_LINE_VERIFY_BLACK_TEXTURE_MAX - textura)
            confianca = float(np.clip(
                0.70 + separacao / 20.0
                + min(altura_preta_ratio, 0.30),
                0.0,
                1.0,
            ))
            bbox = (
                0,
                max(inicio_preto, 0),
                largura,
                max(altura_preta, 1),
            )
            self.last_reason = "preta_lisa"
        elif prata:
            classificacao = NAO_PRETA
            separacao = (
                textura - cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_MIN)
            confianca = float(np.clip(
                0.70 + separacao / 20.0
                + min(preenchimento_borda, 1.0) * 0.10,
                0.0,
                1.0,
            ))
            bbox = bbox_borda
            self.last_reason = "reflexiva_texturizada"
        else:
            classificacao = INCONCLUSIVA
            confianca = 0.0
            bbox = bbox_borda if faixa_presente else (0, 0, 0, 0)
            self.last_reason = (
                "sem_faixa" if not faixa_presente else "zona_inconclusiva")

        return ResultadoFaixaSaidaLinha(
            classificacao=classificacao,
            faixa_presente=faixa_presente,
            textura=textura,
            preenchimento_borda=preenchimento_borda,
            altura_preta_ratio=altura_preta_ratio,
            confianca=confianca,
            bbox=bbox,
            timestamp=timestamp,
        )


class ConfirmadorFaixaSaidaLinha:
    """Votacao temporal que trava em PRETA ou NAO_PRETA."""

    def __init__(
        self,
        classificador=None,
        tamanho_janela=None,
        votos_pretos=None,
        votos_nao_pretos=None,
    ):
        self.classificador = (
            ClassificadorFaixaSaidaLinha()
            if classificador is None else classificador)
        self.tamanho_janela = int(
            cfg.EXIT_LINE_VERIFY_WINDOW
            if tamanho_janela is None else tamanho_janela)
        self.votos_pretos_necessarios = int(
            cfg.EXIT_LINE_VERIFY_BLACK_VOTES
            if votos_pretos is None else votos_pretos)
        self.votos_nao_pretos_necessarios = int(
            cfg.EXIT_LINE_VERIFY_SILVER_VOTES
            if votos_nao_pretos is None else votos_nao_pretos)
        if self.tamanho_janela < 1:
            raise ValueError("a janela de confirmacao precisa ser positiva")
        if not 1 <= self.votos_pretos_necessarios <= self.tamanho_janela:
            raise ValueError("votos pretos invalidos para a janela")
        if not 1 <= self.votos_nao_pretos_necessarios <= self.tamanho_janela:
            raise ValueError("votos nao-pretos invalidos para a janela")
        self._votos = deque(maxlen=self.tamanho_janela)
        self._ultimo_timestamp = None
        self._decisao = None
        self.ultimo_resultado = None

    @property
    def decisao(self):
        return self._decisao

    @property
    def votos_pretos(self):
        return sum(voto == PRETA for voto in self._votos)

    @property
    def votos_nao_pretos(self):
        return sum(voto == NAO_PRETA for voto in self._votos)

    def update(self, frame_bgr, timestamp=None, now=None):
        timestamp = 0.0 if timestamp is None else float(timestamp)
        now = timestamp if now is None else float(now)
        resultado = self.classificador.classificar(
            frame_bgr, timestamp=timestamp)
        self.ultimo_resultado = resultado

        if self._decisao is not None:
            return self._decisao, resultado
        if (
            self._ultimo_timestamp is not None
            and timestamp <= self._ultimo_timestamp + 1e-9
        ):
            return None, resultado
        self._ultimo_timestamp = timestamp
        if now - timestamp > cfg.EXIT_LINE_VERIFY_MAX_AGE_S:
            return None, resultado

        self._votos.append(resultado.classificacao)
        if self.votos_pretos >= self.votos_pretos_necessarios:
            self._decisao = PRETA
        elif (
            self.votos_nao_pretos
            >= self.votos_nao_pretos_necessarios
        ):
            self._decisao = NAO_PRETA
        return self._decisao, resultado


def anotar_confirmacao(frame_bgr, resultado, decisao=None):
    """Desenha somente o diagnostico desta confirmacao final."""
    canvas = frame_bgr.copy()
    if resultado is None:
        return canvas
    x, y, w, h = resultado.bbox
    if w > 0 and h > 0:
        cor = (0, 255, 0) if resultado.classificacao == PRETA else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), cor, 2)
    altura, largura = canvas.shape[:2]
    alvo = int(round(
        altura * cfg.EXIT_LINE_VERIFY_CENTER_Y_RATIO))
    tolerancia = int(round(
        altura * cfg.EXIT_LINE_VERIFY_CENTER_Y_TOLERANCE))
    cv2.line(
        canvas,
        (0, max(alvo - tolerancia, 0)),
        (largura - 1, max(alvo - tolerancia, 0)),
        (255, 255, 0),
        1,
    )
    cv2.line(
        canvas,
        (0, min(alvo + tolerancia, altura - 1)),
        (largura - 1, min(alvo + tolerancia, altura - 1)),
        (255, 255, 0),
        1,
    )
    texto = (
        f"saida={decisao or resultado.classificacao} "
        f"textura={resultado.textura:.1f} "
        f"borda={resultado.preenchimento_borda:.0%}"
    )
    cv2.putText(
        canvas, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas
