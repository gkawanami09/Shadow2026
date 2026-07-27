"""Camada de plausibilidade física — o filtro que NÃO depende da arena.

Por que esta camada existe separada
-----------------------------------
O detector clássico anterior encadeava dez portões, e quase todos julgavam
APARÊNCIA: cor, brilho, textura, contraste, reflexo. Aparência muda quando a
arena muda, e dez portões em série multiplicam essa fragilidade — medido, o
recall caía de 45% para 20% com uma piora modesta em cada portão.

Aqui ficam só regras de GEOMETRIA DE CÂMERA. Elas valem em qualquer arena,
com qualquer piso, qualquer parede e qualquer iluminação, porque descrevem
como o mundo se projeta na imagem — não como o mundo é pintado:

1. o centro da vítima precisa estar dentro do quadro;
2. a vítima está no chão, então aparece abaixo do horizonte útil;
3. o tamanho aparente precisa ser compatível com a linha em que ela está —
   perspectiva: quanto mais baixo no quadro, mais perto, maior.

Quem julga aparência é o modelo treinado. Esta camada só descarta o que é
geometricamente impossível.

Calibração
----------
A envoltória tamanho×linha depende da ALTURA E DO ÂNGULO da câmera. Os
valores em ``config_resgate`` são propositalmente largos e vieram das 18
fotos reais deste robô, que cobrem apenas vítimas próximas. Assim que houver
dataset rotulado, rode ``tools/ajustar_plausibilidade.py`` para ajustá-la aos
dados — apertar antes disso rejeitaria vítimas distantes que eu nunca vi.
"""

from dataclasses import dataclass

import config_resgate as cfg


@dataclass(frozen=True)
class PlausibilityResult:
    """Veredito da camada física sobre um candidato."""

    accepted: bool
    reason: str
    #: Encosta na borda lateral: continua sendo vítima, mas não pode
    #: disparar a coleta antes de o alinhamento trazê-la inteira ao quadro.
    truncated: bool = False

    def __bool__(self):
        return self.accepted


def envelope_de_raio(center_y, frame_height, frame_width):
    """Faixa de raio plausível para uma vítima naquela linha da imagem.

    Modelo linear simples de perspectiva: mais baixo no quadro = mais perto
    = maior. Devolve ``(minimo, maximo)`` em pixels.
    """
    altura = max(float(frame_height), 1.0)
    largura = max(float(frame_width), 1.0)
    razao_y = float(center_y) / altura

    inclinacao = (
        cfg.PLAUSIBLE_RADIUS_AT_BOTTOM - cfg.PLAUSIBLE_RADIUS_AT_TOP
    ) / max(
        cfg.PLAUSIBLE_ROW_BOTTOM - cfg.PLAUSIBLE_ROW_TOP, 1e-6)
    esperado = (
        cfg.PLAUSIBLE_RADIUS_AT_TOP
        + inclinacao * (razao_y - cfg.PLAUSIBLE_ROW_TOP)
    ) * largura
    esperado = max(esperado, 1.0)
    return (
        esperado * cfg.PLAUSIBLE_RADIUS_TOLERANCE_LOW,
        esperado * cfg.PLAUSIBLE_RADIUS_TOLERANCE_HIGH,
    )


class PlausibilityGuard:
    """Aplica as regras físicas a um candidato do detector."""

    def __init__(self, enabled=None):
        self.enabled = bool(
            cfg.PLAUSIBLE_ENABLED if enabled is None else enabled)
        self.last_reason = ""

    def check(self, candidate, frame_shape):
        altura, largura = frame_shape[:2]
        margem = cfg.PLAUSIBLE_EDGE_MARGIN_PX

        if not self.enabled:
            self.last_reason = ""
            return PlausibilityResult(True, "")

        # 1. O centro precisa estar dentro do quadro. Um círculo cujo centro
        #    caiu fora não tem geometria mensurável.
        if not (margem <= candidate.center_x < largura - margem):
            self.last_reason = "centro_fora"
            return PlausibilityResult(False, "centro_fora")

        # 2. A vítima está no chão. Acima do horizonte útil só existe
        #    parede, público, cadeira e mesa — nunca uma vítima.
        if candidate.center_y < altura * cfg.PLAUSIBLE_MIN_CENTER_Y_RATIO:
            self.last_reason = "acima_do_horizonte"
            return PlausibilityResult(False, "acima_do_horizonte")

        # 3. Perspectiva: o tamanho precisa combinar com a linha.
        minimo, maximo = envelope_de_raio(
            candidate.center_y, altura, largura)
        if candidate.radius < minimo:
            self.last_reason = "pequena_demais_para_a_linha"
            return PlausibilityResult(
                False, "pequena_demais_para_a_linha")
        if candidate.radius > maximo:
            self.last_reason = "grande_demais_para_a_linha"
            return PlausibilityResult(
                False, "grande_demais_para_a_linha")

        # Encostar na lateral NÃO reprova — só marca. Ver o docstring de
        # VictimDetection.truncated.
        truncada = (
            candidate.center_x - candidate.radius < margem
            or candidate.center_x + candidate.radius >= largura - margem
        )
        self.last_reason = ""
        return PlausibilityResult(True, "", truncated=truncada)
