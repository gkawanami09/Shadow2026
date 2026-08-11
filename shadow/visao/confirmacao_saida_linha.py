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


@dataclass(frozen=True)
class ResultadoLinhaPercurso:
    """Geometria da linha longitudinal usada no handoff ao segue-linha."""

    encontrada: bool
    centralizada: bool
    centro_x_ratio: float
    alcance_vertical_ratio: float
    aspecto: float
    area: float
    bbox: tuple


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


class DetectorLinhaPercurso:
    """Encontra a linha longitudinal sem aceitar a soleira transversal.

    Usa os mesmos limites BGR calibrados do segue-linha. A forma precisa
    atravessar a metade vertical da imagem, ter forma predominantemente
    longitudinal e estar no corredor central. Assim a faixa preta horizontal
    que acabou de ser confirmada nao encerra o avanco antes de o robo
    realmente atravessa-la.
    """

    def __init__(self):
        from shared.gerenciadores import ConfigManager

        configuracao = ConfigManager(str(config.CONFIG_INI_PATH))

        def ler(nome, fallback):
            valor = configuracao.read_variable("color_values_line", nome)
            return np.asarray(
                fallback if valor is None else valor,
                dtype=np.uint8,
            )

        self.preto_min = np.asarray(
            config.BLACK_MIN_DEFAULT,
            dtype=np.uint8,
        )
        self.preto_max_topo = ler(
            "black_max_normal_top",
            config.BLACK_MAX_NORMAL_TOP_DEFAULT,
        )
        self.preto_max_base = ler(
            "black_max_normal_bottom",
            config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT,
        )

    def detectar(self, frame_bgr):
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("a deteccao da linha exige um frame BGR")

        frame = cv2.resize(
            frame_bgr,
            (config.camera_x, config.camera_y),
            interpolation=cv2.INTER_AREA,
        )
        altura, largura = frame.shape[:2]
        mascara = cv2.inRange(
            frame,
            self.preto_min,
            self.preto_max_base,
        )
        limite_topo = int(round(altura * 0.40))
        mascara[:limite_topo] = cv2.inRange(
            frame[:limite_topo],
            self.preto_min,
            self.preto_max_topo,
        )

        # Fecha pequenas falhas de iluminacao sem unir uma linha lateral a
        # outra forma distante.
        nucleo = np.ones((3, 3), dtype=np.uint8)
        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_CLOSE,
            nucleo,
            iterations=2,
        )
        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        meio_y = int(round(altura * cfg.EXIT_LINE_HANDOFF_MID_Y_RATIO))
        meia_faixa = max(int(round(
            altura * cfg.EXIT_LINE_HANDOFF_MID_BAND_RATIO / 2.0
        )), 1)
        topo_faixa = max(meio_y - meia_faixa, 0)
        base_faixa = min(meio_y + meia_faixa + 1, altura)
        melhor = None
        melhor_pontuacao = float("-inf")

        for contorno in contornos:
            area = float(cv2.contourArea(contorno))
            if area < cfg.EXIT_LINE_HANDOFF_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contorno)
            alcance_vertical = h / max(float(altura), 1.0)
            largura_ratio = w / max(float(largura), 1.0)
            aspecto = h / max(float(w), 1.0)
            if (
                alcance_vertical
                < cfg.EXIT_LINE_HANDOFF_MIN_VERTICAL_SPAN_RATIO
                or largura_ratio > cfg.EXIT_LINE_HANDOFF_MAX_WIDTH_RATIO
                or aspecto < cfg.EXIT_LINE_HANDOFF_MIN_ASPECT
                or y > topo_faixa
                or y + h < base_faixa
            ):
                continue

            mascara_contorno = np.zeros_like(mascara)
            cv2.drawContours(
                mascara_contorno,
                [contorno],
                -1,
                255,
                thickness=-1,
            )
            _ys, xs = np.nonzero(
                mascara_contorno[topo_faixa:base_faixa, :])
            if xs.size == 0:
                continue
            centro_x_ratio = float(np.median(xs)) / max(
                float(largura - 1), 1.0)
            centralizada = bool(
                abs(centro_x_ratio - 0.50)
                <= cfg.EXIT_LINE_HANDOFF_CENTER_X_TOLERANCE_RATIO
            )
            pontuacao = (
                alcance_vertical
                + min(area / max(float(altura * largura), 1.0), 1.0)
                - abs(centro_x_ratio - 0.50)
            )
            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor = ResultadoLinhaPercurso(
                    encontrada=True,
                    centralizada=centralizada,
                    centro_x_ratio=centro_x_ratio,
                    alcance_vertical_ratio=alcance_vertical,
                    aspecto=aspecto,
                    area=area,
                    bbox=(x, y, w, h),
                )

        if melhor is not None:
            return melhor
        return ResultadoLinhaPercurso(
            encontrada=False,
            centralizada=False,
            centro_x_ratio=-1.0,
            alcance_vertical_ratio=0.0,
            aspecto=0.0,
            area=0.0,
            bbox=(0, 0, 0, 0),
        )


def detectar_linha_percurso(frame_bgr):
    """Atalho sem estado para ferramentas e testes."""
    return DetectorLinhaPercurso().detectar(frame_bgr)


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


def anotar_linha_percurso(frame_bgr, resultado, votos=0):
    """Mostra a decisao visual usada no ultimo avanco da saida."""
    canvas = frame_bgr.copy()
    altura, largura = canvas.shape[:2]
    meio_y = int(round(altura * cfg.EXIT_LINE_HANDOFF_MID_Y_RATIO))
    meia_faixa = max(int(round(
        altura * cfg.EXIT_LINE_HANDOFF_MID_BAND_RATIO / 2.0
    )), 1)
    tolerancia_x = int(round(
        largura * cfg.EXIT_LINE_HANDOFF_CENTER_X_TOLERANCE_RATIO))
    centro_x = largura // 2
    cv2.rectangle(
        canvas,
        (max(centro_x - tolerancia_x, 0), max(meio_y - meia_faixa, 0)),
        (min(centro_x + tolerancia_x, largura - 1),
         min(meio_y + meia_faixa, altura - 1)),
        (255, 255, 0),
        1,
    )
    x, y, w, h = resultado.bbox
    if w > 0 and h > 0:
        cor = (0, 255, 0) if resultado.centralizada else (0, 165, 255)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), cor, 2)
    estado = (
        "CENTRAL" if resultado.centralizada
        else "LATERAL" if resultado.encontrada
        else "PROCURANDO"
    )
    cv2.putText(
        canvas,
        f"linha={estado} votos={votos}/{cfg.EXIT_LINE_HANDOFF_VOTES}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
