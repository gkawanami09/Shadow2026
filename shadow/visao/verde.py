"""Detecta e confirma os marcadores verdes do percurso."""

from collections import deque

import cv2
import numpy as np

from config import (GREEN_BLACK_MAX_GAP_RATIO, GREEN_BLACK_MIN_RUN_RATIO,
                    GREEN_BLACK_ROI_SCALE, GREEN_CONFIRM_FRAMES,
                    GREEN_CONFIRM_WINDOW_FRAMES,
                    GREEN_MARKER_MAX_ASPECT, GREEN_MARKER_MEMORY,
                    GREEN_MARKER_MIN_ASPECT, GREEN_MARKER_MIN_RECT_FILL,
                    GREEN_MIN_AREA, GREEN_TURN_AROUND_CONFIRM_FRAMES,
                    GREEN_VOTE_THRESHOLD, GREEN_VOTE_WINDOW,
                    LINE_CROP_GREEN, LINE_CROP_NORMAL)
from shared.dados_compartilhados import (add_time_value, get_time_average,
                                         line_crop, timer, turn_dir)


class ConfirmadorVerde:
    """Libera uma maioria coerente dentro de uma janela curta de quadros."""

    def __init__(self, frames=GREEN_CONFIRM_FRAMES,
                 frames_180=GREEN_TURN_AROUND_CONFIRM_FRAMES,
                 window=GREEN_CONFIRM_WINDOW_FRAMES):
        self.frames = max(1, int(frames))
        self.frames_180 = max(1, int(frames_180))
        self.window = max(self.frames, int(window))
        self._historico = deque(maxlen=self.window)
        self._candidatos = deque(maxlen=self.window)
        self._contagem_180 = 0

    @property
    def candidato_ativo(self):
        """Mantem na memoria um verde plausivel durante a janela de votos."""
        return any(self._candidatos)

    def atualizar(self, direcao, candidato=None):
        if candidato is None:
            candidato = direcao != "straight"
        self._candidatos.append(bool(candidato))
        if direcao == "turn_around":
            # Dois marcadores validos no mesmo quadro tem prioridade sobre
            # qualquer voto parcial de 90 graus.
            self._historico.clear()
            self._contagem_180 += 1
            return (
                "turn_around"
                if self._contagem_180 >= self.frames_180
                else "straight"
            )

        self._contagem_180 = 0
        direcao = direcao if direcao in ("left", "right") else "straight"
        self._historico.append(direcao)
        if direcao == "straight":
            return "straight"

        oposta = "right" if direcao == "left" else "left"
        votos = self._historico.count(direcao)
        votos_opostos = self._historico.count(oposta)
        return (
            direcao
            if votos >= self.frames and votos > votos_opostos
            else "straight"
        )


def _marcador_plausivel(contour):
    """Rejeita manchas verdes que nao podem ser um quadrado da pista."""
    area = float(cv2.contourArea(contour))
    if area <= GREEN_MIN_AREA:
        return False
    _centro, (largura, altura), _angulo = cv2.minAreaRect(contour)
    largura = float(largura)
    altura = float(altura)
    if largura <= 0. or altura <= 0.:
        return False
    aspecto = largura / altura
    preenchimento = area / max(largura * altura, 1.)
    return bool(
        GREEN_MARKER_MIN_ASPECT <= aspecto <= GREEN_MARKER_MAX_ASPECT
        and preenchimento >= GREEN_MARKER_MIN_RECT_FILL
    )


def has_plausible_green(contours_grn):
    """Informa se algum verde merece ser guardado na memoria curta."""
    return any(_marcador_plausivel(contorno) for contorno in contours_grn)


def _forca_preto_proximo(roi, borda_interna, lado):
    """Mede uma faixa preta conectada, mesmo curva ou diagonal."""
    if roi.size == 0:
        return 0
    mascara = (roi > 0).astype(np.uint8) * 255
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    quantidade, _rotulos, stats, _centros = cv2.connectedComponentsWithStats(
        mascara, connectivity=8)
    alcance = max(3, int(round(lado * GREEN_BLACK_MAX_GAP_RATIO)))
    area_minima = max(10, int(round(lado * lado * .025)))
    extensao_minima = max(4, int(round(lado * GREEN_BLACK_MIN_RUN_RATIO * .65)))

    maior = 0
    for indice in range(1, quantidade):
        x, y, w, h, area = stats[indice]
        if area < area_minima or max(w, h) < extensao_minima:
            continue
        if borda_interna == "bottom":
            toca = y + h >= roi.shape[0] - alcance
        elif borda_interna == "top":
            toca = y <= alcance
        elif borda_interna == "right":
            toca = x + w >= roi.shape[1] - alcance
        else:
            toca = x <= alcance
        if toca:
            maior = max(maior, int(area))
    return maior


def _tem_preto_inferior(roi, lado):
    """Rejeita uma faixa larga sob o verde, ignorando risco lateral fino."""
    if roi.size == 0:
        return False
    mascara = (roi > 0).astype(np.uint8) * 255
    quantidade, _rotulos, stats, _centros = cv2.connectedComponentsWithStats(
        mascara, connectivity=8)
    largura_minima = max(
        4, int(round(lado * GREEN_BLACK_MIN_RUN_RATIO)))
    alcance = max(3, int(round(lado * GREEN_BLACK_MAX_GAP_RATIO)))
    for indice in range(1, quantidade):
        _x, y, w, _h, area = stats[indice]
        if y <= alcance and w >= largura_minima and area >= largura_minima * 2:
            return True
    return False


def check_green(contours_grn, black_image, debug_img=None):
    """Retorna apenas direcoes sustentadas pela geometria preta obrigatoria."""
    black_around_sign = np.zeros((len(contours_grn), 5), dtype=np.int32)

    for i, contour in enumerate(contours_grn):
        if not _marcador_plausivel(contour):
            continue

        green_box = cv2.boxPoints(cv2.minAreaRect(contour))
        check_black(black_around_sign, i, green_box, black_image)
        if debug_img is not None:
            leitura = black_around_sign[i]
            geometria_valida = bool(
                not leitura[0]
                and leitura[1]
                and (leitura[2] or leitura[3])
            )
            # Vermelho significa apenas quadrado verde plausivel; verde
            # significa que topo + lado ja autorizaram uma ordem de curva.
            cor = (0, 255, 0) if geometria_valida else (0, 0, 255)
            cv2.drawContours(
                debug_img, [np.intp(green_box)], -1, cor, 2)

    turn_left, turn_right = determine_turn_direction(black_around_sign)
    # O par tem prioridade absoluta. Sem esta ordem, perder um dos contornos
    # por um frame pode deixar a memoria de 90 graus vencer o retorno.
    if turn_left and turn_right:
        return "turn_around"
    if turn_left and not turn_right:
        return "left"
    if turn_right and not turn_left:
        return "right"
    return "straight"


def check_black(black_around_sign, i, green_box, black_image):
    """Mede preto acima e nos lados, sem exigir segmentos a 90 graus."""
    altura_img, largura_img = black_image.shape[:2]
    x, y, w, h = cv2.boundingRect(np.asarray(green_box, dtype=np.float32))
    lado = max(3, int(round(max(w, h))))
    margem = max(4, int(round(lado * GREEN_BLACK_ROI_SCALE)))
    abertura = max(2, int(round(lado * .35)))

    x0 = max(0, x - abertura)
    x1 = min(largura_img, x + w + abertura)
    # Uma pequena sobreposicao aceita arco/diagonal tocando o canto. A forca
    # das componentes abaixo impede a faixa superior de virar dois lados.
    y0 = max(0, y - int(round(lado * .18)))
    y1 = min(altura_img, y + h + abertura)
    topo = black_image[max(0, y - margem):y, x0:x1]
    # Copia a regiao inferior antiga usando a aresta real do retangulo
    # rotacionado. Assim a lateral de um verde diagonal nao vira "embaixo".
    cantos_y = green_box[np.argsort(green_box[:, 1])]
    altura_marcador = max(1., float(
        cantos_y[-1, 1] - cantos_y[0, 1]))
    baixo_y0 = max(0, int(cantos_y[2, 1]))
    baixo_y1 = min(
        altura_img,
        int(round(cantos_y[2, 1] + altura_marcador * .8)),
    )
    baixo_x0 = max(0, int(min(cantos_y[2, 0], cantos_y[3, 0])))
    baixo_x1 = min(
        largura_img,
        int(max(cantos_y[2, 0], cantos_y[3, 0])),
    )
    baixo = black_image[baixo_y0:baixo_y1, baixo_x0:baixo_x1]
    esquerda = black_image[y0:y1, max(0, x - margem):x]
    direita = black_image[y0:y1, x + w:min(largura_img, x + w + margem)]

    black_around_sign[i, 0] = int(_tem_preto_inferior(baixo, lado))
    black_around_sign[i, 1] = _forca_preto_proximo(
        topo, "bottom", lado)
    black_around_sign[i, 2] = _forca_preto_proximo(
        esquerda, "right", lado)
    black_around_sign[i, 3] = _forca_preto_proximo(
        direita, "left", lado)
    black_around_sign[i, 4] = int(np.ceil(np.max(green_box[:, 1])))
    return black_around_sign


def aquecer_numba():
    """Mantem a API de inicializacao; a validacao agora usa OpenCV."""


def determine_turn_direction(black_around_sign):
    turn_left = False
    turn_right = False

    for leitura in black_around_sign:
        # Repete a regra antiga: preto acima e obrigatorio, enquanto qualquer
        # preto conectado abaixo invalida completamente este marcador.
        tem_baixo = leitura[0] > 0
        tem_topo = leitura[1] > 0
        if tem_baixo or not tem_topo:
            continue

        # Se um arco invade os dois recortes, vale o lado claramente mais
        # conectado; lados equivalentes seguem ambiguos e nao inventam curva.
        forca_esquerda = leitura[2]
        forca_direita = leitura[3]
        tem_esquerda = (
            forca_esquerda > 0
            and (forca_direita == 0 or forca_esquerda > forca_direita * 1.25)
        )
        tem_direita = (
            forca_direita > 0
            and (forca_esquerda == 0 or forca_direita > forca_esquerda * 1.25)
        )
        if tem_esquerda and not tem_direita:
            turn_right = True
        elif tem_direita and not tem_esquerda:
            turn_left = True

    return turn_left, turn_right


def average_direction(turn_direction):
    if turn_direction == "left":
        return -1
    if turn_direction == "right":
        return 1
    return 0


def latch_turn_direction(turn_direction, time_turn_direction):
    """Confirma a direcao por varios quadros e guarda uma memoria curta."""
    time_turn_direction = add_time_value(
        time_turn_direction, average_direction(turn_direction))
    avg_turn_dir = get_time_average(time_turn_direction, GREEN_VOTE_WINDOW)

    if avg_turn_dir > GREEN_VOTE_THRESHOLD:
        timer.set_timer("right_marker", GREEN_MARKER_MEMORY)
    elif avg_turn_dir < -GREEN_VOTE_THRESHOLD:
        timer.set_timer("left_marker", GREEN_MARKER_MEMORY)

    if (not timer.get_timer("right_marker")
            and turn_direction != "turn_around" and avg_turn_dir >= 0):
        turn_dir.value = "right"
        line_crop.value = LINE_CROP_GREEN
    elif (not timer.get_timer("left_marker")
          and turn_direction != "turn_around" and avg_turn_dir <= 0):
        turn_dir.value = "left"
        line_crop.value = LINE_CROP_GREEN
    else:
        turn_dir.value = turn_direction
        line_crop.value = LINE_CROP_NORMAL

    return time_turn_direction
