"""Encontra o ramo do trajeto final conectado a faixa preta de saida."""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class DeteccaoContinuacaoSaida:
    alvo_x: float
    alvo_y: float
    distancia_normalizada: float
    distancia_ancora_normalizada: float
    nao_linearidade: float
    area: float
    bbox: tuple[int, int, int, int]


def detectar_continuacao_saida(mascara):
    """Escolhe a extremidade preta mais distante da base central da camera.

    A faixa da porta sozinha e quase uma barra reta. Um T, L ou curva e
    aceito pela sua geometria nao linear. Uma linha reta tambem e aceita
    quando sua extremidade distante ja aponta para a frente. A deteccao guia
    a varredura dianteira, mas o segue-linha so recebe o ramo quando a ponta
    distante cruza o centro da camera.
    """
    if mascara is None or getattr(mascara, "ndim", 0) != 2:
        raise ValueError("a busca da continuacao exige uma mascara binaria")

    altura, largura = mascara.shape
    if altura < 2 or largura < 2:
        return None

    binaria = np.where(mascara > 0, 255, 0).astype(np.uint8)
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None

    ancora = np.array([largura / 2.0, altura - 1.0], dtype=np.float32)
    diagonal = max(float(np.hypot(largura, altura)), 1.0)
    area_quadro = float(largura * altura)
    melhor = None
    melhor_pontuacao = float("-inf")

    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        if area < area_quadro * config.EXIT_CONTINUATION_MIN_AREA_RATIO:
            continue

        pontos = contorno.reshape(-1, 2).astype(np.float32)
        if len(pontos) < 5:
            continue
        distancias = np.linalg.norm(pontos - ancora, axis=1)
        indice_distante = int(np.argmax(distancias))
        ponto_extremo = pontos[indice_distante]
        distancia_alvo = float(distancias[indice_distante] / diagonal)
        distancia_ancora = float(np.min(distancias) / diagonal)
        if (
            distancia_alvo < config.EXIT_CONTINUATION_MIN_TARGET_DISTANCE_RATIO
            or distancia_ancora
            > config.EXIT_CONTINUATION_MAX_ANCHOR_DISTANCE_RATIO
        ):
            continue

        # Faz a media apenas da pequena vizinhanca da extremidade vencedora.
        # Assim ruido de um pixel nao desloca o alvo e extremos concorrentes
        # de um T nao sao misturados entre si.
        raio = max(diagonal * config.EXIT_CONTINUATION_TARGET_CLUSTER_RATIO, 2.0)
        vizinhos = pontos[
            np.linalg.norm(pontos - ponto_extremo, axis=1) <= raio]
        alvo = np.mean(vizinhos, axis=0) if len(vizinhos) else ponto_extremo
        alvo_x, alvo_y = float(alvo[0]), float(alvo[1])
        if alvo_y > altura * config.EXIT_CONTINUATION_MAX_TARGET_Y_RATIO:
            continue

        centralizados = pontos - np.mean(pontos, axis=0)
        covariancia = np.cov(centralizados, rowvar=False)
        autovalores = np.linalg.eigvalsh(covariancia)
        maior = max(float(autovalores[-1]), 1e-6)
        nao_linearidade = max(float(autovalores[0]), 0.0) / maior

        x, y, w, h = cv2.boundingRect(contorno)
        forma_bidimensional = (
            nao_linearidade >= config.EXIT_CONTINUATION_MIN_PCA_RATIO
            and w >= largura * config.EXIT_CONTINUATION_MIN_BBOX_WIDTH_RATIO
            and h >= altura * config.EXIT_CONTINUATION_MIN_BBOX_HEIGHT_RATIO
        )
        # Uma reta inclinada tambem e uma continuacao valida. O avanco
        # vertical distingue essa linha da faixa transversal isolada, sem
        # exigir que ela ja esteja centralizada na camera.
        reta_com_avanco = (
            alvo_y <= altura * config.EXIT_CONTINUATION_FORWARD_TARGET_Y_RATIO
            and h >= altura * config.EXIT_CONTINUATION_MIN_BBOX_HEIGHT_RATIO
        )
        if not (forma_bidimensional or reta_com_avanco):
            continue

        pontuacao = (
            distancia_alvo
            - .45 * distancia_ancora
            + .15 * min(nao_linearidade, 1.0)
            + .05 * min(area / area_quadro, 1.0)
        )
        if pontuacao <= melhor_pontuacao:
            continue
        melhor_pontuacao = pontuacao
        melhor = DeteccaoContinuacaoSaida(
            alvo_x=alvo_x,
            alvo_y=alvo_y,
            distancia_normalizada=distancia_alvo,
            distancia_ancora_normalizada=distancia_ancora,
            nao_linearidade=nao_linearidade,
            area=area,
            bbox=(int(x), int(y), int(w), int(h)),
        )

    return melhor
