"""Estima a trajetoria da linha usando varios cortes horizontais.

O estimador e propositalmente independente de MPU e de temporizacao dos
motores. Ele transforma o contorno escolhido pelo rastreador existente em
tres sinais visuais normalizados: deslocamento lateral, orientacao e
curvatura. Estruturas largas/ambiguas ficam invalidas e usam o fallback do
seguidor legado.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EstimativaTrajetoria:
    valida: bool = False
    lateral: float = 0.
    orientacao: float = 0.
    curvatura: float = 0.
    confianca: float = 0.
    largura_normalizada: float = 0.
    amostras: int = 0
    pontos: tuple = ()


def _largura_e_centro(mask, y, meia_banda=2):
    """Retorna centro/largura medianos sem alargar uma linha inclinada."""
    centros = []
    larguras = []
    y0 = max(0, int(y) - meia_banda)
    y1 = min(mask.shape[0], int(y) + meia_banda + 1)
    for linha in mask[y0:y1]:
        xs = np.flatnonzero(linha)
        if xs.size:
            centros.append((float(xs[0]) + float(xs[-1])) / 2.)
            larguras.append(float(xs[-1] - xs[0] + 1))
    if not centros:
        return None
    return float(np.median(centros)), float(np.median(larguras))


def estimar_trajetoria(contorno, formato_imagem):
    """Extrai uma curva quadratica normalizada do contorno selecionado."""
    altura, largura = tuple(formato_imagem)[:2]
    if contorno is None or altura < 20 or largura < 20:
        return EstimativaTrajetoria()

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.drawContours(mascara, [np.asarray(contorno)], -1, 255, -1)

    y_perto = int(round((altura - 1) * .94))
    y_longe = int(round((altura - 1) * .28))
    niveis = np.linspace(y_perto, y_longe, 11)
    pontos = []
    larguras = []
    for y in niveis:
        medida = _largura_e_centro(mascara, y)
        if medida is None:
            continue
        centro, largura_linha = medida
        pontos.append((centro, float(y)))
        larguras.append(largura_linha)

    if len(pontos) < 6:
        return EstimativaTrajetoria(
            amostras=len(pontos), pontos=tuple(pontos))

    pontos_np = np.asarray(pontos, dtype=np.float64)
    u = (pontos_np[:, 0] - largura / 2.) / (largura / 2.)
    alcance_y = max(float(y_perto - y_longe), 1.)
    v = (y_perto - pontos_np[:, 1]) / alcance_y

    # Um segundo ajuste sem o pior residuo evita que um pequeno esporao de
    # intersecao domine a curva, sem esconder uma estrutura toda ambigua.
    coef = np.polyfit(v, u, 2)
    previsto = np.polyval(coef, v)
    residuos = np.abs(u - previsto)
    if len(u) >= 8 and float(np.max(residuos)) > .08:
        manter = np.ones(len(u), dtype=bool)
        manter[int(np.argmax(residuos))] = False
        coef = np.polyfit(v[manter], u[manter], 2)
        previsto = np.polyval(coef, v)
        residuos = np.abs(u - previsto)

    a, b, c = (float(valor) for valor in coef)
    lateral = float(np.clip(c, -1., 1.))
    orientacao = float(np.clip(b, -1., 1.))
    curvatura = float(np.clip(2. * a, -1., 1.))
    largura_norm = float(np.median(larguras) / largura)
    cobertura = min(len(pontos) / len(niveis), 1.)
    erro_ajuste = float(np.sqrt(np.mean(residuos ** 2)))
    qualidade_ajuste = float(np.clip(1. - erro_ajuste / .16, 0., 1.))
    largura_ok = float(np.clip((.30 - largura_norm) / .18, 0., 1.))
    confianca = cobertura * qualidade_ajuste * largura_ok

    # Uma barra transversal ou intersecao inteira nao e uma trajetoria unica.
    valida = bool(
        len(pontos) >= 6
        and largura_norm >= .008
        and largura_norm < .28
        and confianca >= .35
        and np.isfinite((lateral, orientacao, curvatura, confianca)).all()
    )
    return EstimativaTrajetoria(
        valida=valida,
        lateral=lateral,
        orientacao=orientacao,
        curvatura=curvatura,
        confianca=confianca,
        largura_normalizada=largura_norm,
        amostras=len(pontos),
        pontos=tuple((float(x), float(y)) for x, y in pontos),
    )
