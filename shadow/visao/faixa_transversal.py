"""Geometria e confirmacao temporal de faixas transversais.

A faixa PRATA de entrada e a faixa PRETA de saida sao objetos diferentes, em
cameras diferentes, com iluminacao diferente. O que elas tem em comum e apenas
a *forma*: uma barra alongada que atravessa o campo de visao perto da base.

Este modulo isola somente essa parte comum. Ele nao conhece cor, nao le
configuracao e nao decide nada sobre a missao: recebe uma mascara binaria ja
pronta e responde "existe uma barra transversal aqui?". Os detectores
específicos (como ``faixa_saida``) aplicam suas próprias cores,
seus proprios limiares e suas proprias evidencias de contexto.

Motivo de existir separado: a analise por perfil de linhas abaixo e o unico
teste que distingue uma FAIXA de uma MANCHA. Concentra-la em um lugar permite
testa-la com imagens sinteticas, sem camera, e reaproveitar exatamente o mesmo
criterio nas duas pontas da sala de resgate.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandGeometry:
    """Parametros de forma de uma faixa transversal, em proporcoes do frame.

    Todos os limites sao razoes (0..1) e nao pixels: assim o mesmo perfil vale
    para a camera de linha (448x252) e para a de resgate (640x480), e continua
    valido se a resolucao de trabalho mudar.
    """

    # Fracao minima da largura preenchida para uma linha horizontal ser
    # considerada "parte da faixa". Tolera furos, brilho e oclusao parcial.
    min_row_fill: float = 0.45
    # Largura horizontal minima/maxima ocupada pela faixa.
    min_span_ratio: float = 0.60
    max_span_ratio: float = 1.00
    # Espessura vertical aceitavel. O limite superior e o que separa uma faixa
    # de um piso inteiro claro: um piso preenche a ROI, uma fita nao. Junto
    # com ``min_span_ratio`` ele tambem exclui qualquer disco: largura >= 0.60W
    # exige raio >= 0.30W, o que ja impoe espessura >= 0.397W.
    min_thickness_ratio: float = 0.03
    max_thickness_ratio: float = 0.30
    # Preenchimento dentro da propria caixa da faixa.
    min_fill_ratio: float = 0.55
    # Alongamento (largura/espessura). Uma esfera fica proxima de 1; uma fita
    # transversal fica muito acima. Este e o veto direto contra a vitima.
    min_aspect: float = 3.0


@dataclass(frozen=True)
class TransversalBand:
    """Faixa transversal encontrada, em coordenadas do frame completo."""

    top_y: int
    bottom_y: int
    left_x: int
    right_x: int
    row_count: int
    span_ratio: float
    thickness_ratio: float
    fill_ratio: float
    aspect: float
    coverage: float

    @property
    def height(self):
        return self.bottom_y - self.top_y + 1

    @property
    def width(self):
        return self.right_x - self.left_x + 1

    @property
    def center_x(self):
        return (self.left_x + self.right_x) / 2.0

    @property
    def center_y(self):
        return (self.top_y + self.bottom_y) / 2.0

    @property
    def bbox(self):
        return self.left_x, self.top_y, self.width, self.height


def _longest_true_run(flags):
    """Maior sequencia contigua de ``True``; retorna ``(inicio, fim)``.

    Usa a sequencia mais longa em vez de "qualquer linha acesa" porque brilho
    especular e ruido produzem linhas isoladas espalhadas pela ROI. Uma fita
    real gera linhas vizinhas continuas.
    """
    best_start = best_end = -1
    best_length = 0
    start = None
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
        elif start is not None:
            length = index - start
            if length > best_length:
                best_length, best_start, best_end = length, start, index - 1
            start = None
    if start is not None:
        length = len(flags) - start
        if length > best_length:
            best_length, best_start, best_end = length, start, len(flags) - 1
    return best_start, best_end


def find_transversal_band(
    mask,
    geometry=None,
    roi_top_ratio=0.0,
    roi_bottom_ratio=1.0,
):
    """Procura uma barra alongada e transversal dentro da ROI da mascara.

    Retorna ``(TransversalBand, "")`` quando aceita e ``(None, motivo)`` quando
    rejeita. O motivo e devolvido literalmente para o calibrador e o dataset
    poderem mostrar *por que* um candidato caiu, em vez de apenas "nada".
    """
    geometry = BandGeometry() if geometry is None else geometry
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("find_transversal_band exige uma mascara 2D")

    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return None, "frame_vazio"

    top = int(round(height * float(roi_top_ratio)))
    bottom = int(round(height * float(roi_bottom_ratio)))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    roi = mask[top:bottom, :]
    if roi.size == 0:
        return None, "roi_vazia"

    binary = roi > 0
    row_fill = binary.sum(axis=1) / float(width)
    hot_rows = row_fill >= geometry.min_row_fill
    first_row, last_row = _longest_true_run(hot_rows)
    if first_row < 0:
        return None, "sem_linha_cheia"

    if first_row == 0:
        # A faixa encosta no TOPO da ROI: sua extensao real continua acima do
        # corte e nao pode ser medida. Quase sempre e uma regiao grande
        # (roupa escura, sombra, parede) truncada pela ROI, que passa a
        # parecer uma fita fina. Medido nas capturas reais da arena: as seis
        # falsas soleiras de saida comecavam todas exatamente nesta linha.
        # Encostar embaixo continua valido — a fita fica assim quando o robo
        # chega perto dela.
        return None, "cortada_no_topo"

    row_count = last_row - first_row + 1
    thickness_ratio = row_count / float(height)
    if thickness_ratio < geometry.min_thickness_ratio:
        return None, "fina"
    if thickness_ratio > geometry.max_thickness_ratio:
        # Um piso claro inteiro ou uma sombra grande entram aqui: elas
        # preenchem a ROI em vez de formar uma fita.
        return None, "espessa"

    band = binary[first_row:last_row + 1, :]
    columns = band.any(axis=0)
    active = np.flatnonzero(columns)
    if active.size == 0:
        return None, "sem_coluna"
    left_x = int(active[0])
    right_x = int(active[-1])
    span = right_x - left_x + 1
    span_ratio = span / float(width)
    if span_ratio < geometry.min_span_ratio:
        return None, "estreita"
    if span_ratio > geometry.max_span_ratio:
        return None, "larga"

    box_area = float(span * row_count)
    fill_ratio = float(band[:, left_x:right_x + 1].sum()) / max(box_area, 1.0)
    if fill_ratio < geometry.min_fill_ratio:
        return None, "vazada"

    aspect = span / float(max(row_count, 1))
    if aspect < geometry.min_aspect:
        # Uma esfera prateada ou preta tem proporcao proxima de 1 e morre aqui,
        # mesmo que sua cor combine perfeitamente com a da fita.
        return None, "compacta"

    coverage = float(binary.sum()) / float(binary.size)
    return TransversalBand(
        top_y=top + first_row,
        bottom_y=top + last_row,
        left_x=left_x,
        right_x=right_x,
        row_count=int(row_count),
        span_ratio=float(span_ratio),
        thickness_ratio=float(thickness_ratio),
        fill_ratio=float(fill_ratio),
        aspect=float(aspect),
        coverage=coverage,
    ), ""


class StripeConfirmer:
    """Votacao temporal N-de-M com frames distintos, histerese e cooldown.

    Regras que o resto do codigo depende:

    * um unico frame positivo nunca confirma (reflexo isolado);
    * dois resultados com o mesmo timestamp contam como UM voto (o detector
      assincrono pode reentregar o mesmo frame);
    * um resultado velho demais nao vota, mesmo sendo positivo;
    * depois de confirmar, o estado gruda (histerese) ate ``reset()``;
    * apos ``reset()``, nenhuma confirmacao nova ocorre durante o cooldown.
    """

    def __init__(
        self,
        votes_needed=3,
        window=5,
        max_age_s=0.75,
        cooldown_s=0.0,
    ):
        if votes_needed <= 0:
            raise ValueError("votes_needed deve ser positivo")
        if window < votes_needed:
            raise ValueError("a janela nao pode ser menor que os votos")
        self.votes_needed = int(votes_needed)
        self.window = int(window)
        self.max_age_s = float(max_age_s)
        self.cooldown_s = float(cooldown_s)
        self._votes = []
        self._confirmed = False
        self._last_timestamp = None
        self._blocked_until = None

    @property
    def confirmed(self):
        return self._confirmed

    @property
    def votes(self):
        """Positivos dentro da janela atual."""
        return sum(1 for vote in self._votes if vote)

    @property
    def samples(self):
        return len(self._votes)

    def reset(self, now=None):
        """Descarta os votos e arma o cooldown a partir de ``now``."""
        self._votes = []
        self._confirmed = False
        self._last_timestamp = None
        if self.cooldown_s > 0.0 and now is not None:
            self._blocked_until = float(now) + self.cooldown_s

    def blocked(self, now):
        return (
            self._blocked_until is not None
            and float(now) < self._blocked_until
        )

    def update(self, positive, timestamp, now=None):
        """Registra um frame e devolve o estado confirmado (com histerese)."""
        now = float(timestamp) if now is None else float(now)
        timestamp = float(timestamp)

        if self._confirmed:
            # Histerese: quem desarma e a missao, via reset(), e nao um frame
            # ruim isolado logo depois de confirmar a entrada.
            return True
        if self.blocked(now):
            return False
        if (
            self._last_timestamp is not None
            and timestamp <= self._last_timestamp + 1e-9
        ):
            # Frame repetido: nao gera um segundo voto.
            return False
        if now - timestamp > self.max_age_s:
            # Resultado stale nao vota; o frame ja nao descreve o presente.
            return False

        self._last_timestamp = timestamp
        self._votes.append(bool(positive))
        if len(self._votes) > self.window:
            self._votes = self._votes[-self.window:]

        if self.votes >= self.votes_needed:
            self._confirmed = True
        return self._confirmed
