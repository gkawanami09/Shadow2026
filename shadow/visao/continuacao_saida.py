"""Geometria da soleira preta e da terceira linha depois do resgate.

A antiga implementacao escolhia o ponto mais distante de um contorno unico.
Quando a soleira estava diagonal, uma de suas pontas podia vencer a haste do
T e ser entregue como linha. Aqui as duas estruturas sao separadas:

1. a borda inferior da faixa larga e ajustada nos lados do quadro;
2. toda a espessura dessa faixa e apagada;
3. somente um componente que progride para a frente e tocava a faixa pode ser
   considerado continuacao.

Depois que a soleira sai do quadro, uma linha longitudinal comum tambem e
aceita. A mascara usa os mesmos limites BGR calibrados do segue-linha.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
import config_resgate as cfg
from shared.gerenciadores import ConfigManager


DIREITA_BAIXA = "direita_baixa"
ESQUERDA_BAIXA = "esquerda_baixa"
NIVEL = "nivel"


@dataclass(frozen=True)
class DeteccaoSoleira:
    orientacao: str
    y_esquerda: float
    y_direita: float
    delta_y_ratio: float
    inclinacao: float
    intercepto_inferior: float
    inclinacao_superior: float
    intercepto_superior: float
    espessura: float
    cobertura_esquerda: float
    cobertura_direita: float
    erro_mediano: float
    confianca: float
    bbox: tuple[int, int, int, int]

    def y_inferior(self, x):
        return float(self.inclinacao * float(x) + self.intercepto_inferior)

    def y_superior(self, x):
        return float(
            self.inclinacao_superior * float(x)
            + self.intercepto_superior
        )


@dataclass(frozen=True)
class DeteccaoContinuacaoSaida:
    alvo_x: float
    alvo_y: float
    confianca: float
    area: float
    altura_ratio: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class AnaliseSaidaPreta:
    soleira: DeteccaoSoleira | None
    continuacao: DeteccaoContinuacaoSaida | None
    mascara: np.ndarray


def _ler_limite(nome, fallback):
    manager = ConfigManager(str(Path(config.CONFIG_INI_PATH)))
    valor = manager.read_variable("color_values_line", nome)
    return np.asarray(fallback if valor is None else valor, dtype=np.uint8)


class SegmentadorPretoLinha:
    """Aplica exatamente o perfil de preto configurado para a camera 1."""

    def __init__(self, minimo=None, maximo_topo=None, maximo_base=None):
        self.minimo = np.asarray(
            config.BLACK_MIN_DEFAULT if minimo is None else minimo,
            dtype=np.uint8,
        )
        self.maximo_topo = (
            _ler_limite(
                "black_max_normal_top",
                config.BLACK_MAX_NORMAL_TOP_DEFAULT,
            )
            if maximo_topo is None
            else np.asarray(maximo_topo, dtype=np.uint8)
        )
        self.maximo_base = (
            _ler_limite(
                "black_max_normal_bottom",
                config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT,
            )
            if maximo_base is None
            else np.asarray(maximo_base, dtype=np.uint8)
        )

    def segmentar(self, frame_bgr):
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("a geometria da saida exige um frame BGR")
        frame = cv2.resize(
            frame_bgr,
            (config.camera_x, config.camera_y),
            interpolation=cv2.INTER_AREA,
        )
        mascara = cv2.inRange(frame, self.minimo, self.maximo_base)
        limite_topo = int(round(config.camera_y * 0.40))
        mascara[:limite_topo] = cv2.inRange(
            frame[:limite_topo], self.minimo, self.maximo_topo)
        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        return cv2.morphologyEx(
            mascara,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )


def _runs_coluna(coluna):
    ativa = np.asarray(coluna) > 0
    padded = np.pad(ativa, (1, 1), constant_values=False)
    mudancas = np.diff(padded.astype(np.int8))
    inicios = np.flatnonzero(mudancas == 1)
    finais = np.flatnonzero(mudancas == -1) - 1
    return inicios, finais


def _run_inferior_valido(coluna, comprimento_minimo):
    inicios, finais = _runs_coluna(coluna)
    if not len(inicios):
        return None
    comprimentos = finais - inicios + 1
    validos = comprimentos >= int(comprimento_minimo)
    if not np.any(validos):
        return None
    inicios = inicios[validos]
    finais = finais[validos]
    comprimentos = comprimentos[validos]
    indice = int(np.argmax(finais))
    return (
        int(inicios[indice]),
        int(finais[indice]),
        int(comprimentos[indice]),
    )


def _ajustar_reta_robusta(xs, ys, mascara_inicial=None):
    """Ajusta uma borda sem deixar poucos outliers inclinarem a reta."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    manter = (
        np.ones(len(xs), dtype=bool)
        if mascara_inicial is None
        else np.asarray(mascara_inicial, dtype=bool).copy()
    )
    for _ in range(3):
        if np.count_nonzero(manter) < 6:
            return None
        coeficientes = np.polyfit(xs[manter], ys[manter], 1)
        residuos = np.abs(ys - np.polyval(coeficientes, xs))
        mad = float(np.median(residuos[manter]))
        manter &= residuos <= max(3.0, mad * 3.0)
    if np.count_nonzero(manter) < 6:
        return None
    coeficientes = np.polyfit(xs[manter], ys[manter], 1)
    residuos = np.abs(ys - np.polyval(coeficientes, xs))
    erro = float(np.median(residuos[manter]))
    return coeficientes, manter, erro


def detectar_soleira(mascara):
    """Mede qual lado da faixa larga esta mais baixo na imagem."""
    if mascara is None or getattr(mascara, "ndim", 0) != 2:
        raise ValueError("a soleira exige uma mascara binaria")
    altura, largura = mascara.shape
    if altura < 8 or largura < 8:
        return None

    comprimento_minimo = max(int(round(
        altura * cfg.EXIT_POST_COLUMN_MIN_RUN_RATIO)), 2)
    intervalos = (
        (
            int(round(largura * cfg.EXIT_LINE_VERIFY_SIDE_X_MIN_RATIO)),
            int(round(largura * cfg.EXIT_LINE_VERIFY_SIDE_X_MAX_RATIO)),
        ),
        (
            int(round(largura * (1.0 - cfg.EXIT_LINE_VERIFY_SIDE_X_MAX_RATIO))),
            int(round(largura * (1.0 - cfg.EXIT_LINE_VERIFY_SIDE_X_MIN_RATIO))),
        ),
    )

    grupos = []
    for inicio, fim in intervalos:
        pontos = []
        for x in range(max(inicio, 0), min(fim, largura)):
            run = _run_inferior_valido(mascara[:, x], comprimento_minimo)
            if run is not None:
                topo, base, espessura = run
                pontos.append((x, base, espessura, topo))
        grupos.append((pontos, max(fim - inicio, 1)))

    coberturas = tuple(
        len(pontos) / float(total) for pontos, total in grupos)
    if min(coberturas) < cfg.EXIT_POST_SIDE_MIN_COVERAGE:
        return None

    pontos = np.asarray(
        [item for grupo, _ in grupos for item in grupo],
        dtype=np.float64,
    )
    if len(pontos) < 8:
        return None

    ajuste_inferior = _ajustar_reta_robusta(pontos[:, 0], pontos[:, 1])
    if ajuste_inferior is None:
        return None
    coeficientes, manter_inferior, erro_inferior = ajuste_inferior
    ajuste_superior = _ajustar_reta_robusta(
        pontos[:, 0],
        pontos[:, 3],
        mascara_inicial=manter_inferior,
    )
    if ajuste_superior is None:
        return None
    coeficientes_superiores, manter_superior, erro_superior = (
        ajuste_superior
    )
    manter = manter_inferior & manter_superior
    if np.count_nonzero(manter) < 6:
        return None
    # Reajusta as duas bordas sobre exatamente as mesmas colunas. Assim a
    # espessura pode variar com a perspectiva sem deixar uma cunha residual
    # que pareca a terceira linha depois que a soleira for apagada.
    coeficientes = np.polyfit(
        pontos[manter, 0], pontos[manter, 1], 1)
    coeficientes_superiores = np.polyfit(
        pontos[manter, 0], pontos[manter, 3], 1)
    erro_inferior = float(np.median(np.abs(
        pontos[manter, 1]
        - np.polyval(coeficientes, pontos[manter, 0])
    )))
    erro_superior = float(np.median(np.abs(
        pontos[manter, 3]
        - np.polyval(coeficientes_superiores, pontos[manter, 0])
    )))
    erro_mediano = max(erro_inferior, erro_superior)
    if erro_mediano > altura * cfg.EXIT_POST_GEOMETRY_MAX_MAD_RATIO:
        return None

    inclinacao, intercepto = map(float, coeficientes)
    inclinacao_superior, intercepto_superior = map(
        float, coeficientes_superiores)
    espessuras_ajustadas = (
        np.polyval(coeficientes, pontos[manter, 0])
        - np.polyval(coeficientes_superiores, pontos[manter, 0])
    )
    if np.min(espessuras_ajustadas) < comprimento_minimo - 1:
        return None
    espessura = float(np.median(espessuras_ajustadas))
    x_esquerda = largura * 0.18
    x_direita = largura * 0.82
    y_esquerda = float(np.polyval(coeficientes, x_esquerda))
    y_direita = float(np.polyval(coeficientes, x_direita))
    if not (
        -espessura <= y_esquerda <= altura + espessura
        and -espessura <= y_direita <= altura + espessura
    ):
        return None
    delta_ratio = (y_direita - y_esquerda) / max(float(altura), 1.0)
    if delta_ratio > cfg.EXIT_POST_LEVEL_DELTA_RATIO:
        orientacao = DIREITA_BAIXA
    elif delta_ratio < -cfg.EXIT_POST_LEVEL_DELTA_RATIO:
        orientacao = ESQUERDA_BAIXA
    else:
        orientacao = NIVEL

    topo_minimo = min(
        float(np.polyval(coeficientes_superiores, x_esquerda)),
        float(np.polyval(coeficientes_superiores, x_direita)),
    )
    base_maxima = max(y_esquerda, y_direita)
    bbox_y = max(int(np.floor(topo_minimo)), 0)
    bbox_base = min(int(np.ceil(base_maxima)) + 1, altura)
    cobertura_media = (coberturas[0] + coberturas[1]) / 2.0
    confianca = float(np.clip(
        0.55
        + cobertura_media * 0.30
        + (1.0 - min(erro_mediano / max(altura * 0.04, 1.0), 1.0))
        * 0.15,
        0.0,
        1.0,
    ))
    return DeteccaoSoleira(
        orientacao=orientacao,
        y_esquerda=y_esquerda,
        y_direita=y_direita,
        delta_y_ratio=float(delta_ratio),
        inclinacao=inclinacao,
        intercepto_inferior=intercepto,
        inclinacao_superior=inclinacao_superior,
        intercepto_superior=intercepto_superior,
        espessura=espessura,
        cobertura_esquerda=float(coberturas[0]),
        cobertura_direita=float(coberturas[1]),
        erro_mediano=erro_mediano,
        confianca=confianca,
        bbox=(0, bbox_y, largura, max(bbox_base - bbox_y, 1)),
    )


def _mascara_sem_soleira(mascara, soleira):
    limpa = np.where(mascara > 0, 255, 0).astype(np.uint8)
    altura, largura = limpa.shape
    margem = max(
        altura * cfg.EXIT_POST_BAND_ERASE_MARGIN_RATIO,
        3.0,
    )
    # Tudo da borda superior da soleira para tras e removido. Sobra somente a
    # haste que aponta para a frente, sem as pontas da faixa transversal.
    for x in range(largura):
        limite = int(np.floor(soleira.y_superior(x) - margem))
        limpa[max(limite, 0):, x] = 0
    return limpa


def detectar_continuacao_saida(mascara, soleira=None):
    """Encontra a terceira linha, nunca uma ponta da faixa transversal."""
    if mascara is None or getattr(mascara, "ndim", 0) != 2:
        raise ValueError("a busca da continuacao exige uma mascara binaria")
    altura, largura = mascara.shape
    if altura < 2 or largura < 2:
        return None

    limpa = (
        _mascara_sem_soleira(mascara, soleira)
        if soleira is not None
        else np.where(mascara > 0, 255, 0).astype(np.uint8)
    )
    quantidade, rotulos, estatisticas, _centroides = (
        cv2.connectedComponentsWithStats(limpa, connectivity=8)
    )
    area_quadro = float(altura * largura)
    melhor = None
    melhor_pontuacao = float("-inf")

    for rotulo in range(1, quantidade):
        x, y, w, h, area = map(int, estatisticas[rotulo])
        if area < area_quadro * cfg.EXIT_POST_CONTINUATION_MIN_AREA_RATIO:
            continue
        if h < altura * cfg.EXIT_POST_CONTINUATION_MIN_HEIGHT_RATIO:
            continue
        if (
            w > largura * cfg.EXIT_POST_CONTINUATION_MAX_HORIZONTAL_SPAN_RATIO
            and h
            < altura * cfg.EXIT_POST_CONTINUATION_MAX_HORIZONTAL_HEIGHT_RATIO
        ):
            continue

        ys, xs = np.where(rotulos == rotulo)
        if not len(xs):
            continue
        if soleira is not None:
            distancia_conexao = np.min(np.abs(
                np.asarray([
                    soleira.y_superior(px) for px in xs
                ]) - ys
            ))
            if (
                distancia_conexao
                > altura * cfg.EXIT_POST_CONTINUATION_CONNECT_TOLERANCE_RATIO
            ):
                continue
        else:
            # Depois que a soleira some, a mascara nao oferece mais uma reta
            # de referencia para apaga-la. Um pedaco diagonal da propria
            # soleira pode ficar preso entre uma lateral e o fundo da imagem;
            # ele jamais e uma linha longitudinal valida.
            margem_borda = max(int(round(
                min(altura, largura)
                * cfg.EXIT_POST_FALLBACK_EDGE_MARGIN_RATIO
            )), 2)
            toca_lateral = (
                x <= margem_borda
                or x + w >= largura - margem_borda
            )
            toca_fundo = y + h >= altura - margem_borda
            if toca_lateral and toca_fundo:
                continue

            # PCA mede a direcao real do componente, sem depender da caixa
            # delimitadora. A terceira linha precisa avancar mais no eixo Y
            # do que deriva lateralmente; uma soleira residual nao passa.
            pontos = np.column_stack((xs, ys)).astype(np.float64)
            pontos -= np.mean(pontos, axis=0, keepdims=True)
            covariancia = np.cov(pontos, rowvar=False)
            if covariancia.shape != (2, 2) or not np.all(
                np.isfinite(covariancia)
            ):
                continue
            autovalores, autovetores = np.linalg.eigh(covariancia)
            eixo_principal = autovetores[:, int(np.argmax(autovalores))]
            deriva_x = abs(float(eixo_principal[0]))
            avanco_y = abs(float(eixo_principal[1]))
            if (
                avanco_y <= 1e-9
                or deriva_x / avanco_y
                > cfg.EXIT_POST_FALLBACK_MAX_LATERAL_PER_FORWARD
            ):
                continue

        quantidade_topo = max(int(round(len(ys) * 0.12)), 1)
        indices_topo = np.argpartition(
            ys, quantidade_topo - 1)[:quantidade_topo]
        alvo_x = float(np.mean(xs[indices_topo]))
        alvo_y = float(np.mean(ys[indices_topo]))
        if (
            soleira is not None
            and soleira.y_superior(alvo_x) - alvo_y
            < altura
            * cfg.EXIT_POST_CONTINUATION_MIN_FORWARD_PROGRESS_RATIO
        ):
            # Um pedaco irregular da propria borda pode continuar tocando a
            # banda apagada, mas nao progride de verdade para o horizonte.
            continue
        altura_ratio = h / max(float(altura), 1.0)
        area_ratio = area / max(area_quadro, 1.0)
        confianca = float(np.clip(
            0.55 + min(altura_ratio, 0.60) * 0.55
            + min(area_ratio, 0.08),
            0.0,
            1.0,
        ))
        pontuacao = altura_ratio + area_ratio * 2.0 - alvo_y / altura * 0.10
        if pontuacao <= melhor_pontuacao:
            continue
        melhor_pontuacao = pontuacao
        melhor = DeteccaoContinuacaoSaida(
            alvo_x=alvo_x,
            alvo_y=alvo_y,
            confianca=confianca,
            area=float(area),
            altura_ratio=float(altura_ratio),
            bbox=(x, y, w, h),
        )
    return melhor


class AnalisadorSaidaPreta:
    def __init__(self, segmentador=None):
        self.segmentador = (
            SegmentadorPretoLinha() if segmentador is None else segmentador)

    def analisar(self, frame_bgr):
        mascara = self.segmentador.segmentar(frame_bgr)
        soleira = detectar_soleira(mascara)
        continuacao = detectar_continuacao_saida(mascara, soleira=soleira)
        return AnaliseSaidaPreta(
            soleira=soleira,
            continuacao=continuacao,
            mascara=mascara,
        )


def anotar_analise_saida(frame_bgr, analise):
    """Overlay de debug; a imagem anotada nunca volta para a decisao."""
    canvas = cv2.resize(
        frame_bgr,
        (config.camera_x, config.camera_y),
        interpolation=cv2.INTER_AREA,
    ).copy()
    if analise is None:
        return canvas
    if analise.soleira is not None:
        soleira = analise.soleira
        for x in range(canvas.shape[1] - 1):
            p1 = (x, int(round(soleira.y_inferior(x))))
            p2 = (x + 1, int(round(soleira.y_inferior(x + 1))))
            cv2.line(canvas, p1, p2, (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"pose={soleira.orientacao} dY={soleira.delta_y_ratio:+.2f}",
            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (0, 255, 255), 1, cv2.LINE_AA,
        )
    if analise.continuacao is not None:
        alvo = (
            int(round(analise.continuacao.alvo_x)),
            int(round(analise.continuacao.alvo_y)),
        )
        cv2.circle(canvas, alvo, 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "TERCEIRA LINHA", (8, 62),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
    return canvas
