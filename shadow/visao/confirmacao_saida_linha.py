"""Confirma se a faixa de saida e preta usando a camera do segue-linha.

Esta verificacao acontece somente depois do deposito vermelho. A camera de
resgate ja encontrou e aproximou o robo da soleira; aqui a camera apontada
para o chao decide entre a faixa preta e a prata.

As duas evidencias usadas sao independentes:

* faixa transversal: uma borda horizontal ocupa boa parte da imagem;
* aparencia: preto e escuro e quase liso, enquanto prata tem reflexos e
  textura forte.

Um unico frame nunca decide. Como o erro perigoso e aceitar prata como preta,
a votacao e conservadora: dois votos de prata bloqueiam a saida, enquanto
preto exige quatro votos dentro das cinco imagens novas.
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

        topo_textura = int(round(
            altura * cfg.EXIT_LINE_VERIFY_TEXTURE_ROI_TOP))
        base_textura = int(round(
            altura * cfg.EXIT_LINE_VERIFY_TEXTURE_ROI_BOTTOM))
        roi_textura = variacao_local[topo_textura:base_textura, :]
        textura = float(np.median(roi_textura))

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
            and textura <= cfg.EXIT_LINE_VERIFY_BLACK_TEXTURE_MAX
        )
        prata = bool(
            faixa_presente
            and textura >= cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_MIN
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
            margem = max(int(round(altura * 0.08)), 1)
            bbox = (
                0,
                max(indice_borda - margem, 0),
                largura,
                min(margem * 2 + 1, altura),
            )
            self.last_reason = "reflexiva_texturizada"
        else:
            classificacao = INCONCLUSIVA
            confianca = 0.0
            bbox = (0, 0, 0, 0)
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

    def __init__(self, classificador=None):
        self.classificador = (
            ClassificadorFaixaSaidaLinha()
            if classificador is None else classificador)
        self._votos = deque(maxlen=cfg.EXIT_LINE_VERIFY_WINDOW)
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
        if self.votos_pretos >= cfg.EXIT_LINE_VERIFY_BLACK_VOTES:
            self._decisao = PRETA
        elif (
            self.votos_nao_pretos
            >= cfg.EXIT_LINE_VERIFY_SILVER_VOTES
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
    texto = (
        f"saida={decisao or resultado.classificacao} "
        f"textura={resultado.textura:.1f} "
        f"borda={resultado.preenchimento_borda:.0%}"
    )
    cv2.putText(
        canvas, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas
