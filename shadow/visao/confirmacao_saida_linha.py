"""Confirma preto ou prata com a camera de linha, perto da soleira.

A exposicao automatica da OV5647 impede que um valor absoluto de brilho seja
uma medida confiavel de cor. Este modulo localiza a faixa pelo seu perfil
horizontal e compara sua aparencia com o piso do *mesmo frame*. Uma mudanca
uniforme de luz altera ambos e praticamente preserva a razao.

Os 28% externos de cada lado do quadro sao usados para medir a soleira. Isso
remove da amostra a linha preta longitudinal que aparece no centro nos casos
T/L. Frames repetidos, faixa fora do centro e exposicao instavel nao votam.
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
    brilho_relativo: float
    textura_relativa: float
    referencia_luz: float
    preenchimento_escuro: float


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
    """A faixa esta na zona em que preto e prata podem ser comparados?"""
    posicao = posicao_vertical_faixa(resultado)
    return (
        posicao is not None
        and abs(posicao - cfg.EXIT_LINE_VERIFY_CENTER_Y_RATIO)
        <= cfg.EXIT_LINE_VERIFY_CENTER_Y_TOLERANCE
    )


def _grupos_colunas_laterais(largura):
    inicio = int(round(
        largura * cfg.EXIT_LINE_VERIFY_SIDE_X_MIN_RATIO))
    fim = int(round(
        largura * cfg.EXIT_LINE_VERIFY_SIDE_X_MAX_RATIO))
    inicio_direita = largura - fim
    fim_direita = largura - inicio
    esquerda = np.arange(max(inicio, 0), min(fim, largura))
    direita = np.arange(
        max(inicio_direita, 0), min(fim_direita, largura))
    return esquerda, direita


def _mediana_ou_none(array):
    if array is None or not getattr(array, "size", 0):
        return None
    return float(np.median(array))


def _localizar_bordas_laterais(valor, colunas):
    """Localiza topo/base no perfil de um lado, sem misturar diagonais."""
    altura = valor.shape[0]
    perfil = np.median(valor[:, colunas], axis=1).astype(np.float32)
    perfil = cv2.GaussianBlur(
        perfil.reshape((-1, 1)), (1, 9), 0).reshape((-1,))
    derivada = perfil[2:] - perfil[:-2]
    busca_inicio = max(int(round(
        altura * cfg.EXIT_LINE_VERIFY_EDGE_SEARCH_TOP_RATIO)), 1)
    busca_fim = min(int(round(
        altura * cfg.EXIT_LINE_VERIFY_EDGE_SEARCH_BOTTOM_RATIO)),
        len(derivada),
    )
    if busca_fim <= busca_inicio:
        return None

    topo = busca_inicio + int(np.argmin(
        derivada[busca_inicio:busca_fim])) + 1
    altura_minima = max(int(round(
        altura * cfg.EXIT_LINE_VERIFY_MIN_HEIGHT_RATIO)), 2)
    altura_maxima = max(int(round(
        altura * cfg.EXIT_LINE_VERIFY_MAX_HEIGHT_RATIO)), altura_minima)
    inicio_base = min(topo + altura_minima, len(derivada) - 1)
    fim_base = min(topo + altura_maxima, len(derivada))
    if fim_base <= inicio_base:
        return None
    base = inicio_base + int(np.argmax(
        derivada[inicio_base:fim_base])) + 1
    espessura = base - topo
    if not altura_minima <= espessura <= altura_maxima:
        return None
    return int(topo), int(base)


class ClassificadorFaixaSaidaLinha:
    """Localiza a soleira e separa preto de prata por contraste relativo."""

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
        valor = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        altura, largura = valor.shape
        grupos_colunas = _grupos_colunas_laterais(largura)
        colunas = np.concatenate(grupos_colunas)
        if colunas.size < 4:
            raise ValueError("largura insuficiente para medir a soleira")

        bordas = tuple(
            _localizar_bordas_laterais(valor, grupo)
            for grupo in grupos_colunas
        )
        if any(borda is None for borda in bordas):
            return self._inconclusiva(timestamp, "sem_bordas_nos_dois_lados")

        # Cada lado e localizado separadamente. Misturar os dois perfis fazia
        # uma faixa preta bem diagonal parecer metade preta/metade piso e,
        # portanto, prata. As retas abaixo acompanham a inclinacao e amostram
        # somente o interior da soleira em cada coluna lateral.
        centros_x = np.asarray([
            float(np.median(grupo)) for grupo in grupos_colunas
        ])
        topos = np.asarray([borda[0] for borda in bordas], dtype=np.float64)
        bases = np.asarray([borda[1] for borda in bordas], dtype=np.float64)
        if abs(centros_x[1] - centros_x[0]) < 1.0:
            return self._inconclusiva(timestamp, "lados_sem_separacao")
        coef_topo = np.polyfit(centros_x, topos, 1)
        coef_base = np.polyfit(centros_x, bases, 1)

        margem = max(int(round(altura * 0.018)), 2)
        alcance_referencia = max(int(round(altura * 0.14)), 4)
        pixels_faixa = []
        pixels_referencia_topo = []
        pixels_referencia_base = []
        coordenadas_faixa = []
        for x in colunas:
            topo_x = int(round(np.polyval(coef_topo, x)))
            base_x = int(round(np.polyval(coef_base, x)))
            miolo_topo = max(min(topo_x + margem, altura - 1), 0)
            miolo_base = max(min(base_x - margem, altura), miolo_topo + 1)
            if miolo_base <= miolo_topo:
                continue
            pixels_faixa.append(valor[miolo_topo:miolo_base, x])
            coordenadas_faixa.extend(
                (y, int(x)) for y in range(miolo_topo, miolo_base))

            fim_topo = max(min(topo_x - margem, altura), 0)
            inicio_topo = max(fim_topo - alcance_referencia, 0)
            if fim_topo > inicio_topo:
                pixels_referencia_topo.append(
                    valor[inicio_topo:fim_topo, x])
            inicio_base = max(min(base_x + margem, altura), 0)
            fim_base = min(inicio_base + alcance_referencia, altura)
            if fim_base > inicio_base:
                pixels_referencia_base.append(
                    valor[inicio_base:fim_base, x])

        if not pixels_faixa:
            return self._inconclusiva(timestamp, "faixa_vazia")
        roi_faixa = np.concatenate(pixels_faixa)
        referencias = [
            np.concatenate(grupo)
            for grupo in (
                pixels_referencia_topo,
                pixels_referencia_base,
            )
            if grupo
        ]
        if not referencias:
            return self._inconclusiva(timestamp, "sem_piso_referencia")

        # O lado mais iluminado e a referencia. Vignetting deixa o topo da
        # imagem naturalmente mais escuro; usar a media faria a prata parecer
        # preta justamente quando a faixa estivesse alta.
        referencia_luz = max(
            valor_ref
            for valor_ref in map(_mediana_ou_none, referencias)
            if valor_ref is not None
        )
        brilho_faixa = float(np.median(roi_faixa))
        brilho_relativo = brilho_faixa / max(referencia_luz, 1.0)

        # Um blur minimo remove ruido de leitura independente da camera sem
        # apagar a textura metalica espacial da prata. Sem isso, apenas dois
        # niveis de ruido BGR ja podiam empurrar preto real para a zona prata.
        valor_textura = cv2.GaussianBlur(valor, (3, 3), 0)
        gradiente = cv2.morphologyEx(
            valor_textura,
            cv2.MORPH_GRADIENT,
            np.ones((7, 7), dtype=np.uint8),
        )
        textura = float(np.median([
            gradiente[y, x] for y, x in coordenadas_faixa
        ]))
        textura_relativa = textura / max(referencia_luz, 1.0)
        preenchimento_escuro = float(np.mean(
            roi_faixa
            <= referencia_luz * cfg.EXIT_LINE_VERIFY_DARK_PIXEL_RATIO
        ))
        contraste = max(referencia_luz - brilho_faixa, 0.0)
        faixa_presente = bool(
            contraste >= cfg.EXIT_LINE_VERIFY_EDGE_CONTRAST_MIN)
        preenchimento_borda = float(np.clip(
            contraste
            / max(cfg.EXIT_LINE_VERIFY_EDGE_CONTRAST_MIN * 4.0, 1.0),
            0.0,
            1.0,
        ))
        espessura = float(np.median(bases - topos))
        espessura_ratio = espessura / max(float(altura), 1.0)

        preta = bool(
            faixa_presente
            and brilho_relativo
            <= cfg.EXIT_LINE_VERIFY_BLACK_BRIGHTNESS_RATIO_MAX
            and preenchimento_escuro
            >= cfg.EXIT_LINE_VERIFY_BLACK_DARK_FILL_MIN
            and textura_relativa
            <= cfg.EXIT_LINE_VERIFY_BLACK_TEXTURE_RATIO_MAX
            and espessura_ratio
            <= cfg.EXIT_LINE_VERIFY_BLACK_MAX_HEIGHT_RATIO
        )
        prata = bool(
            faixa_presente
            and (
                brilho_relativo
                >= cfg.EXIT_LINE_VERIFY_SILVER_BRIGHTNESS_RATIO_MIN
                or (
                    textura_relativa
                    >= cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_RATIO_MIN
                    and preenchimento_escuro
                    <= cfg.EXIT_LINE_VERIFY_SILVER_DARK_FILL_MAX
                )
            )
        )
        topo_bbox = max(int(np.floor(np.min(topos))), 0)
        base_bbox = min(int(np.ceil(np.max(bases))) + 1, altura)
        bbox = (
            0,
            topo_bbox,
            int(largura),
            max(base_bbox - topo_bbox, 1),
        )

        if preta:
            classificacao = PRETA
            margem_cor = (
                cfg.EXIT_LINE_VERIFY_BLACK_BRIGHTNESS_RATIO_MAX
                - brilho_relativo
            )
            confianca = float(np.clip(
                0.65 + margem_cor * 1.2
                + min(preenchimento_escuro, 1.0) * 0.15,
                0.0,
                1.0,
            ))
            altura_preta_ratio = espessura_ratio
            self.last_reason = "preta_por_contraste_relativo"
        elif prata:
            classificacao = NAO_PRETA
            margem_cor = max(
                brilho_relativo
                - cfg.EXIT_LINE_VERIFY_SILVER_BRIGHTNESS_RATIO_MIN,
                textura_relativa
                - cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_RATIO_MIN,
                0.0,
            )
            confianca = float(np.clip(
                0.65 + margem_cor * 1.2
                + min(textura_relativa, 0.20),
                0.0,
                1.0,
            ))
            altura_preta_ratio = 0.0
            self.last_reason = "prata_por_brilho_ou_textura_relativa"
        else:
            classificacao = INCONCLUSIVA
            confianca = 0.0
            altura_preta_ratio = 0.0
            self.last_reason = (
                "sem_faixa" if not faixa_presente
                else "zona_relativa_inconclusiva"
            )

        return ResultadoFaixaSaidaLinha(
            classificacao=classificacao,
            faixa_presente=faixa_presente,
            textura=textura,
            preenchimento_borda=preenchimento_borda,
            altura_preta_ratio=altura_preta_ratio,
            confianca=confianca,
            bbox=bbox if faixa_presente else (0, 0, 0, 0),
            timestamp=timestamp,
            brilho_relativo=float(brilho_relativo),
            textura_relativa=float(textura_relativa),
            referencia_luz=float(referencia_luz),
            preenchimento_escuro=float(preenchimento_escuro),
        )

    def _inconclusiva(self, timestamp, motivo):
        self.last_reason = motivo
        return ResultadoFaixaSaidaLinha(
            classificacao=INCONCLUSIVA,
            faixa_presente=False,
            textura=0.0,
            preenchimento_borda=0.0,
            altura_preta_ratio=0.0,
            confianca=0.0,
            bbox=(0, 0, 0, 0),
            timestamp=float(timestamp),
            brilho_relativo=1.0,
            textura_relativa=0.0,
            referencia_luz=0.0,
            preenchimento_escuro=0.0,
        )


class ConfirmadorFaixaSaidaLinha:
    """Votacao temporal com centro e estabilidade de exposicao obrigatorios."""

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
        self._ultima_referencia_luz = None
        self._frames_exposicao_estavel = 0
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

    @property
    def exposicao_estavel(self):
        return (
            self._frames_exposicao_estavel
            >= cfg.EXIT_LINE_VERIFY_EXPOSURE_STABLE_FRAMES
        )

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

        if not faixa_centralizada(resultado):
            self._votos.clear()
            self._frames_exposicao_estavel = 0
            self._ultima_referencia_luz = None
            return None, resultado

        referencia = float(resultado.referencia_luz)
        if self._ultima_referencia_luz is None:
            self._frames_exposicao_estavel = 1
        else:
            mudanca = abs(referencia - self._ultima_referencia_luz) / max(
                self._ultima_referencia_luz, 1.0)
            if mudanca <= cfg.EXIT_LINE_VERIFY_EXPOSURE_MAX_REL_CHANGE:
                self._frames_exposicao_estavel += 1
            else:
                # AEC ainda se acomodando: nenhum voto anterior atravessa a
                # mudanca de exposicao.
                self._votos.clear()
                self._frames_exposicao_estavel = 1
        self._ultima_referencia_luz = referencia
        if not self.exposicao_estavel:
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
    """Desenha apenas diagnostico; nunca realimenta o classificador."""
    canvas = frame_bgr.copy()
    if resultado is None:
        return canvas
    x, y, w, h = resultado.bbox
    if w > 0 and h > 0:
        cor = (0, 255, 0) if resultado.classificacao == PRETA else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), cor, 2)
    altura, largura = canvas.shape[:2]
    alvo = int(round(altura * cfg.EXIT_LINE_VERIFY_CENTER_Y_RATIO))
    tolerancia = int(round(
        altura * cfg.EXIT_LINE_VERIFY_CENTER_Y_TOLERANCE))
    cv2.line(
        canvas, (0, max(alvo - tolerancia, 0)),
        (largura - 1, max(alvo - tolerancia, 0)), (255, 255, 0), 1)
    cv2.line(
        canvas, (0, min(alvo + tolerancia, altura - 1)),
        (largura - 1, min(alvo + tolerancia, altura - 1)),
        (255, 255, 0), 1)
    texto = (
        f"saida={decisao or resultado.classificacao} "
        f"brilho={resultado.brilho_relativo:.2f} "
        f"textura={resultado.textura_relativa:.2f}"
    )
    cv2.putText(
        canvas, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas
