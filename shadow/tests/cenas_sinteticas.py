"""Cenas sintéticas usadas pelos testes de faixa de entrada e de saída.

Não substituem imagens reais da arena. Elas existem para provar as REGRAS do
detector (forma, neutralidade, reflexo, contraste, temporalidade) de forma
determinística e sem câmera. Os limiares numéricos continuam precisando de
calibração com fotos reais — isso está registrado no relatório final.
"""

import numpy as np


LINE_FRAME = (252, 448)
RESCUE_FRAME = (480, 640)


def _blank(shape, value):
    frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    frame[:] = value
    return frame


def piso_neutro(shape=LINE_FRAME, valor=120):
    """Piso liso e neutro, abaixo do limiar de brilho da fita prata."""
    return _blank(shape, valor)


def faixa_prata(
    shape=LINE_FRAME,
    topo=0.78,
    espessura=0.10,
    piso=120,
    base=200,
    brilho=255,
    densidade_brilho=0.12,
    seed=7,
):
    """Fita refletiva: banda clara, neutra e com brilho especular esparso."""
    height, width = shape
    frame = piso_neutro(shape, piso)
    top = int(round(height * topo))
    bottom = min(int(round(top + height * espessura)), height)
    frame[top:bottom, :] = base
    rng = np.random.default_rng(seed)
    speckle = rng.random((bottom - top, width)) < densidade_brilho
    band = frame[top:bottom, :]
    band[speckle] = brilho
    frame[top:bottom, :] = band
    return frame


def piso_branco(shape=LINE_FRAME, valor=235):
    """Piso branco liso ocupando toda a imagem — negativo obrigatório."""
    return _blank(shape, valor)


def reflexo_pontual(shape=LINE_FRAME, piso=120, valor=255, largura=34,
                    altura=22):
    """Reflexo de LED / parafuso brilhante: pequeno e isolado."""
    height, width = shape
    frame = piso_neutro(shape, piso)
    cx, cy = width // 2, int(height * 0.85)
    frame[cy - altura // 2:cy + altura // 2,
          cx - largura // 2:cx + largura // 2] = valor
    return frame


def esfera(shape, raio, valor, piso, centro=None):
    """Disco cheio — usado como vítima prateada ou vítima preta."""
    height, width = shape
    frame = _blank(shape, piso)
    cx, cy = (width // 2, int(height * 0.80)) if centro is None else centro
    ys, xs = np.ogrid[:height, :width]
    disk = (xs - cx) ** 2 + (ys - cy) ** 2 <= raio * raio
    frame[disk] = valor
    return frame


def faixa_preta(
    shape=RESCUE_FRAME,
    topo=0.74,
    espessura=0.09,
    piso=185,
    valor=18,
):
    """Soleira preta de saída sobre piso claro."""
    height, width = shape
    frame = _blank(shape, piso)
    top = int(round(height * topo))
    bottom = min(int(round(top + height * espessura)), height)
    frame[top:bottom, :] = valor
    return frame


def interseccao(
    shape=LINE_FRAME,
    topo=0.76,
    espessura=0.09,
    piso=225,
    valor=22,
    coluna=(200, 250),
):
    """Cruzamento de linhas pretas: a transversal + a linha que o robô segue.

    É o falso positivo que mais custa caro: a transversal tem exatamente a
    forma da fita de entrada. O que a denuncia é ser PRETA — a auréola clara
    da borda até entra na máscara, mas a caixa continua escura.
    """
    frame = faixa_preta(
        shape, topo=topo, espessura=espessura, piso=piso, valor=valor)
    frame[:, coluna[0]:coluna[1]] = valor
    return frame


def faixa_preta_salpicada(
    shape=LINE_FRAME,
    topo=0.74,
    espessura=0.14,
    piso=225,
    valor=20,
    brilho=245,
    densidade_brilho=0.5,
    seed=11,
):
    """Faixa preta com reflexo salpicado por cima.

    Existe para provar o veto de escuro isoladamente: o salpico é claro e
    texturizado o bastante para a máscara formar uma faixa com a geometria
    certa, mas a caixa dessa faixa continua majoritariamente preta.
    """
    height, width = shape
    frame = _blank(shape, piso)
    top = int(round(height * topo))
    bottom = min(int(round(top + height * espessura)), height)
    frame[top:bottom, :] = valor
    rng = np.random.default_rng(seed)
    speckle = rng.random((bottom - top, width)) < densidade_brilho
    band = frame[top:bottom, :]
    band[speckle] = brilho
    frame[top:bottom, :] = band
    return frame


def girar(frame, graus):
    """Gira a cena — o robô chegando torto na soleira.

    Replica a borda em vez de preencher com preto: cantos pretos artificiais
    disparariam o veto de escuro e o teste passaria a medir um defeito do
    próprio gerador de cena, não o detector.
    """
    import cv2

    altura, largura = frame.shape[:2]
    matriz = cv2.getRotationMatrix2D(
        (largura / 2.0, altura / 2.0), graus, 1.0)
    return cv2.warpAffine(
        frame, matriz, (largura, altura),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def sombra_ampla(shape=RESCUE_FRAME, piso=60, valor=30):
    """Sombra grande sobre piso já escuro: sem contraste real com o entorno."""
    height, width = shape
    frame = _blank(shape, piso)
    frame[int(height * 0.55):, :] = valor
    return frame


def madeira(shape=RESCUE_FRAME, seed=3):
    """Textura amadeirada saturada: negativo para prata e para faixa."""
    height, width = shape
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    base = rng.integers(90, 150, size=(height, width))
    frame[:, :, 0] = np.clip(base * 0.45, 0, 255)
    frame[:, :, 1] = np.clip(base * 0.75, 0, 255)
    frame[:, :, 2] = np.clip(base * 1.15, 0, 255)
    return frame
