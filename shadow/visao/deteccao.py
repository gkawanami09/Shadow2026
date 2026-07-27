"""Tipos compartilhados da visão de vítimas.

Este módulo define o CONTRATO entre quem detecta e quem controla. Ele não
detecta nada e não depende de OpenCV nem de modelo nenhum — por isso pode ser
importado por qualquer lado sem arrastar peso junto.

O contrato foi extraído do que os controladores já consumiam do detector
clássico, para que trocar o detector não obrigue a reescrever a aproximação,
a busca pulsada nem o portão assíncrono.
"""

from dataclasses import dataclass


#: Cores válidas de vítima. Prata = viva, preta = morta.
CORES_VALIDAS = ("silver", "black")


@dataclass(frozen=True)
class VictimDetection:
    """Uma vítima detectada e acompanhada no tempo.

    Os campos são exatamente os que ``aproximacao_resgate``,
    ``busca_pulsada`` e ``resgate_assincrono`` consomem. Qualquer detector
    novo só precisa produzir isto.
    """

    kind: str
    center_x: float
    center_y: float
    radius: float
    confidence: float
    #: ``True`` após acumular confirmações temporais suficientes.
    confirmed: bool
    hits: int
    timestamp: float
    #: ``True`` quando o track já está travado e não pode ser roubado.
    track_locked: bool = False
    #: A vítima encosta na borda lateral do quadro, então parte dela está
    #: fora da imagem. Continua sendo vítima e o robô deve girar na direção
    #: dela, mas a coleta fica bloqueada até o alinhamento trazê-la inteira.
    truncated: bool = False

    def __post_init__(self):
        if self.kind not in CORES_VALIDAS:
            raise ValueError(
                f"cor de vítima inválida: {self.kind!r}; "
                f"esperado um de {CORES_VALIDAS}")

    @property
    def diameter(self):
        return self.radius * 2.0

    @property
    def bottom_y(self):
        return self.center_y + self.radius

    @property
    def top_y(self):
        return self.center_y - self.radius

    @property
    def bbox(self):
        """``(x, y, largura, altura)`` do quadrado que envolve a esfera."""
        return (
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.radius * 2.0,
            self.radius * 2.0,
        )

    def horizontal_error(self, frame_width):
        """Erro lateral normalizado: −1 na esquerda, +1 na direita."""
        metade = max(float(frame_width) / 2.0, 1.0)
        bruto = (self.center_x - metade) / metade
        return float(min(max(bruto, -1.0), 1.0))


@dataclass(frozen=True)
class CloseCrescentEvidence:
    """Evidência da borda larga da esfera quando ela é cortada pelo quadro.

    Vinha do detector clássico, que foi aposentado. O tipo sobrevive porque
    ``aproximacao_resgate`` sabe consumi-lo como rota alternativa do gate de
    proximidade — perto demais, a esfera fica maior que o quadro e nenhum
    detector de caixa a enxerga inteira.

    Hoje **nada produz esta evidência**: o detector por modelo entrega caixa,
    não meia-lua. O campo continua opcional em toda a cadeia e o controlador
    cai na rota do círculo travado quando ela é ``None``. Quando a coleta
    voltar ao escopo, é aqui que a evidência de contato deve ser reintroduzida
    — provavelmente como uma classe própria do modelo ("vitima_muito_perto"),
    que é mais robusta do que reconstruir o arco por gradiente.
    """

    accepted: bool
    confidence: float
    support: float
    left_support: float
    center_support: float
    right_support: float
    contrast: float
    center_x_ratio: float
    top_y_ratio: float
    halfspan_ratio: float
    bottom_y_ratio: float
    timestamp: float
    gradient_polarity: float = 0.0
    profile_support: float = 0.0
    profile_polarity: float = 0.0
    coherent_run: float = 0.0
    circle_rmse_ratio: float = 1.0
    curvature_score: float = 0.0
    foil_fallback: bool = False
    foil_texture_bins: int = 0
    foil_valid_bins: int = 0
    interior_edge_density: float = 0.0
    background_edge_density: float = 1.0


@dataclass(frozen=True)
class VictimCandidate:
    """Candidato cru, antes da plausibilidade física e do tempo.

    É o que um modelo de detecção produz por frame: uma caixa, uma classe e
    uma confiança. Sem histórico e sem garantia nenhuma.
    """

    kind: str
    center_x: float
    center_y: float
    radius: float
    confidence: float

    @property
    def bottom_y(self):
        return self.center_y + self.radius

    @property
    def top_y(self):
        return self.center_y - self.radius

    @classmethod
    def from_xyxy(cls, kind, x0, y0, x1, y1, confidence):
        """Constrói a partir de uma caixa no formato do YOLO."""
        largura = max(float(x1) - float(x0), 0.0)
        altura = max(float(y1) - float(y0), 0.0)
        return cls(
            kind=kind,
            center_x=(float(x0) + float(x1)) / 2.0,
            center_y=(float(y0) + float(y1)) / 2.0,
            # A vítima é esférica: o raio é a média dos dois semi-eixos, o
            # que tolera uma caixa levemente retangular sem distorcer.
            radius=(largura + altura) / 4.0,
            confidence=float(confidence),
        )
