"""Faixa PRATA de entrada da sala de resgate, vista pela CÂMERA DE LINHA.

Este detector é deliberadamente separado do detector de vítimas prateadas.
São câmeras diferentes, alturas diferentes, iluminação diferente e objetos
diferentes: a vítima é uma esfera de ~4 cm e a entrada é uma fita de
25 x 250 mm colada no chão. Compartilhar limiares entre os dois seria o
caminho mais rápido para o robô entrar na sala ao ver uma vítima, ou ignorar
a entrada por estar calibrado para uma esfera.

A confirmação exige evidência CONJUNTA, nunca um único sinal:

1. região neutra e clara na ROI inferior (máscara HSV calibrável);
2. forma alongada e transversal (``faixa_transversal``);
3. largura mínima proporcional ao quadro;
4. baixa saturação (neutralidade metálica);
5. assinatura reflexiva — faixa dinâmica e brilho especular concentrado,
   que é o que separa fita refletiva de papel branco fosco;
6. contraste contra a vizinhança imediata, acima e abaixo da faixa;
7. fim coerente da linha preta à frente;
8. votação temporal 3-de-5 em frames distintos e recentes, com cooldown.

Nenhum desses limiares foi validado com a fita real sob a luz da competição.
Eles são um ponto de partida seguro e existem para ser medidos com
``tools/calibrar_cores.py`` (grupo 7).
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config
from visao.faixa_transversal import (BandGeometry, StripeConfirmer,
                                     find_transversal_band)


@dataclass(frozen=True)
class EntryStripeDetection:
    """Candidato de faixa prata já aprovado em forma e aparência."""

    center_x: float
    center_y: float
    width: int
    height: int
    top_y: int
    bottom_y: int
    span_ratio: float
    thickness_ratio: float
    aspect: float
    saturation: float
    value: float
    dynamic_range: float
    highlight_fraction: float
    surround_contrast: float
    confidence: float
    timestamp: float
    bbox: tuple


def default_geometry():
    return BandGeometry(
        min_row_fill=config.ENTRY_SILVER_MIN_ROW_FILL,
        min_span_ratio=config.ENTRY_SILVER_MIN_SPAN_RATIO,
        max_span_ratio=config.ENTRY_SILVER_MAX_SPAN_RATIO,
        min_thickness_ratio=config.ENTRY_SILVER_MIN_THICKNESS_RATIO,
        max_thickness_ratio=config.ENTRY_SILVER_MAX_THICKNESS_RATIO,
        min_fill_ratio=config.ENTRY_SILVER_MIN_FILL_RATIO,
        min_aspect=config.ENTRY_SILVER_MIN_ASPECT,
    )


def load_bounds(config_manager=None):
    """Lê ``entry_silver_min/max`` do config.ini da CÂMERA DE LINHA.

    A seção é ``color_values_line`` porque este perfil é da câmera de linha.
    Os limites da câmera de resgate vivem em ``config_resgate.py`` e nunca
    são gravados aqui.
    """
    minimum = list(config.ENTRY_SILVER_MIN_DEFAULT)
    maximum = list(config.ENTRY_SILVER_MAX_DEFAULT)
    if config_manager is None:
        return minimum, maximum
    stored_min = config_manager.read_variable(
        'color_values_line', 'entry_silver_min')
    stored_max = config_manager.read_variable(
        'color_values_line', 'entry_silver_max')
    if stored_min is not None:
        minimum = list(stored_min)
    if stored_max is not None:
        maximum = list(stored_max)
    return minimum, maximum


def local_range_map(value_channel, window=None):
    """Amplitude local (máx − mín) do brilho numa janela pequena.

    É a medida barata de "textura de luz": metal reflexivo concentra brilho
    em pontos e produz amplitude alta; piso fosco é uniforme e produz
    amplitude baixa, por mais claro que seja.
    """
    window = (
        config.ENTRY_SILVER_LOCAL_WINDOW_PX if window is None else window)
    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1
    kernel = np.ones((window, window), np.uint8)
    maximo = cv2.dilate(value_channel, kernel)
    minimo = cv2.erode(value_channel, kernel)
    return cv2.subtract(maximo, minimo)


def silver_mask(frame_bgr, hsv_min, hsv_max, min_local_range=None,
                hsv_image=None):
    """Máscara neutra/clara da faixa. Função pura, reutilizada no calibrador.

    Além da faixa HSV, exige textura de luz mínima. Sem isso, um piso cinza
    com o mesmo brilho e a mesma neutralidade da fita entra inteiro na
    máscara e o candidato morre na geometria antes de qualquer teste de
    aparência — foi exatamente o que aconteceu na arena real.
    """
    hsv = (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if hsv_image is None else hsv_image
    )
    mask = cv2.inRange(
        hsv,
        np.asarray(hsv_min, dtype=np.uint8),
        np.asarray(hsv_max, dtype=np.uint8),
    )
    limite = (
        config.ENTRY_SILVER_MIN_LOCAL_RANGE
        if min_local_range is None else min_local_range
    )
    if limite > 0:
        amplitude = local_range_map(hsv[:, :, 2])
        mask[amplitude < limite] = 0
    return mask


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


class EntrySilverDetector:
    """Encontra a fita prata de entrada em um frame da câmera de linha."""

    def __init__(self, hsv_min=None, hsv_max=None, geometry=None,
                 min_local_range=None):
        self.hsv_min = list(
            config.ENTRY_SILVER_MIN_DEFAULT if hsv_min is None else hsv_min)
        self.hsv_max = list(
            config.ENTRY_SILVER_MAX_DEFAULT if hsv_max is None else hsv_max)
        self.geometry = default_geometry() if geometry is None else geometry
        #: Amplitude local mínima; ``None`` usa o valor do config.
        self.min_local_range = min_local_range
        self.last_reason = "inicio"
        self.last_mask = None
        self.last_band = None

    def detect(self, frame_bgr, line_ahead=None, timestamp=None,
               hsv_image=None):
        """Retorna ``EntryStripeDetection`` ou ``None`` com ``last_reason``.

        ``line_ahead`` é a evidência de que a linha preta continua à frente,
        publicada pelo pipeline do segue-linha. Quando ela é ``True`` e a
        configuração exige o fim da linha, o candidato é vetado: um brilho
        sobre a linha não é a entrada da sala.
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("EntrySilverDetector exige um frame BGR")

        timestamp = 0.0 if timestamp is None else float(timestamp)
        self.last_band = None

        mask = silver_mask(
            frame_bgr, self.hsv_min, self.hsv_max,
            min_local_range=self.min_local_range,
            hsv_image=hsv_image)
        self.last_mask = mask

        band, reason = find_transversal_band(
            mask,
            self.geometry,
            roi_top_ratio=config.ENTRY_SILVER_ROI_TOP,
            roi_bottom_ratio=config.ENTRY_SILVER_ROI_BOTTOM,
        )
        usou_reflexo_flexivel = False
        limite_principal = (
            config.ENTRY_SILVER_MIN_LOCAL_RANGE
            if self.min_local_range is None
            else float(self.min_local_range)
        )
        limite_flexivel = min(
            limite_principal,
            float(config.ENTRY_SILVER_FALLBACK_LOCAL_RANGE),
        )
        if (
            band is None
            and not bool(line_ahead)
            and limite_flexivel < limite_principal
        ):
            # Perto da câmera, a fita ocupa muitos pixels e o reflexo fica
            # espalhado. A máscara rígida pode virar várias linhas finas.
            # Uma segunda máscara apenas recompõe a forma; todos os testes de
            # largura, aspecto, neutralidade, contraste e confiança continuam.
            mask_flexivel = silver_mask(
                frame_bgr,
                self.hsv_min,
                self.hsv_max,
                min_local_range=limite_flexivel,
                hsv_image=hsv_image,
            )
            banda_flexivel, motivo_flexivel = find_transversal_band(
                mask_flexivel,
                self.geometry,
                roi_top_ratio=config.ENTRY_SILVER_ROI_TOP,
                roi_bottom_ratio=config.ENTRY_SILVER_ROI_BOTTOM,
            )
            if banda_flexivel is not None:
                mask = mask_flexivel
                self.last_mask = mask
                band = banda_flexivel
                reason = ""
                usou_reflexo_flexivel = True
            else:
                reason = motivo_flexivel
        if band is None:
            self.last_reason = reason
            return None
        self.last_band = band

        if config.ENTRY_SILVER_REQUIRE_LINE_END and bool(line_ahead):
            # A linha preta ainda continua adiante: isto é brilho sobre a
            # pista, não a soleira da sala.
            self.last_reason = "linha_continua"
            return None

        hsv = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            if hsv_image is None else hsv_image
        )
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)

        inside = np.zeros(mask.shape, dtype=bool)
        inside[band.top_y:band.bottom_y + 1, band.left_x:band.right_x + 1] = True
        inside &= mask > 0
        if not inside.any():
            self.last_reason = "faixa_vazia"
            return None

        inside_saturation = float(np.median(saturation[inside]))
        if inside_saturation > config.ENTRY_SILVER_MAX_SATURATION:
            self.last_reason = "saturada"
            return None

        inside_values = value[inside]
        inside_value = float(np.median(inside_values))
        dynamic_range = float(
            np.percentile(inside_values, 95) - np.percentile(inside_values, 5))
        highlight_fraction = float(
            np.count_nonzero(inside_values >= config.ENTRY_SILVER_HIGHLIGHT_V)
            / float(inside_values.size))
        # Fita refletiva concentra brilho especular OU espalha muita faixa
        # dinâmica. Papel branco fosco e piso claro uniforme falham nos dois.
        reflective = (
            dynamic_range >= config.ENTRY_SILVER_MIN_DYNAMIC_RANGE
            or highlight_fraction
            >= config.ENTRY_SILVER_MIN_HIGHLIGHT_FRACTION
        )
        if not reflective:
            self.last_reason = "sem_assinatura_reflexiva"
            return None

        # Contraste com a vizinhança, medido de DUAS formas independentes.
        # Vale a maior: numa arena onde o piso tem o mesmo brilho da fita, o
        # contraste de brilho é zero e quem denuncia a fita é a textura da
        # luz. Exigir contraste de brilho ali rejeitaria a fita verdadeira.
        surround_contrast = max(
            self._surround_contrast(value, band, mask),
            self._surround_contrast(
                local_range_map(hsv[:, :, 2]).astype(np.float32),
                band, mask),
        )
        if surround_contrast < config.ENTRY_SILVER_MIN_SURROUND_CONTRAST:
            # Indistinguível do próprio piso: sem borda, não é uma fita.
            self.last_reason = "sem_contraste"
            return None

        confidence = self._confidence(
            band, dynamic_range, highlight_fraction, surround_contrast)
        confianca_minima = (
            config.ENTRY_SILVER_FALLBACK_MIN_CONFIDENCE
            if usou_reflexo_flexivel
            else config.ENTRY_SILVER_MIN_CONFIDENCE
        )
        if confidence < confianca_minima:
            self.last_reason = "confianca"
            return None

        self.last_reason = ""
        return EntryStripeDetection(
            center_x=band.center_x,
            center_y=band.center_y,
            width=band.width,
            height=band.height,
            top_y=band.top_y,
            bottom_y=band.bottom_y,
            span_ratio=band.span_ratio,
            thickness_ratio=band.thickness_ratio,
            aspect=band.aspect,
            saturation=inside_saturation,
            value=inside_value,
            dynamic_range=dynamic_range,
            highlight_fraction=highlight_fraction,
            surround_contrast=surround_contrast,
            confidence=confidence,
            timestamp=timestamp,
            bbox=band.bbox,
        )

    @staticmethod
    def _surround_contrast(value, band, mask):
        """|mediana da faixa − mediana da vizinhança| acima e abaixo dela.

        A polaridade não importa: dependendo do ângulo, a fita refletiva pode
        aparecer mais clara ou mais escura que o piso. O que ela não pode é
        ser idêntica ao piso, porque então não existe fita nenhuma.
        """
        height = value.shape[0]
        margin = max(
            int(round(height * config.ENTRY_SILVER_SURROUND_MARGIN_RATIO)), 1)
        inside = mask[band.top_y:band.bottom_y + 1,
                      band.left_x:band.right_x + 1] > 0
        patch = value[band.top_y:band.bottom_y + 1,
                      band.left_x:band.right_x + 1]
        if not inside.any():
            return 0.0
        inside_value = float(np.median(patch[inside]))

        samples = []
        above_top = max(band.top_y - margin, 0)
        if above_top < band.top_y:
            samples.append(
                value[above_top:band.top_y, band.left_x:band.right_x + 1])
        below_bottom = min(band.bottom_y + 1 + margin, height)
        if below_bottom > band.bottom_y + 1:
            samples.append(
                value[band.bottom_y + 1:below_bottom,
                      band.left_x:band.right_x + 1])
        contrasts = [
            abs(inside_value - float(np.median(sample)))
            for sample in samples if sample.size
        ]
        # A faixa pode encostar na base do quadro; nesse caso só existe a
        # vizinhança de cima e ela decide sozinha.
        return max(contrasts) if contrasts else 0.0

    @staticmethod
    def _confidence(band, dynamic_range, highlight_fraction,
                    surround_contrast):
        span_score = _clip01(
            (band.span_ratio - config.ENTRY_SILVER_MIN_SPAN_RATIO)
            / max(1.0 - config.ENTRY_SILVER_MIN_SPAN_RATIO, 1e-6))
        fill_score = _clip01(
            (band.fill_ratio - config.ENTRY_SILVER_MIN_FILL_RATIO)
            / max(1.0 - config.ENTRY_SILVER_MIN_FILL_RATIO, 1e-6))
        range_score = _clip01(
            dynamic_range / max(config.ENTRY_SILVER_MIN_DYNAMIC_RANGE * 2.0,
                                1e-6))
        highlight_score = _clip01(
            highlight_fraction
            / max(config.ENTRY_SILVER_MIN_HIGHLIGHT_FRACTION * 4.0, 1e-6))
        contrast_score = _clip01(
            surround_contrast
            / max(config.ENTRY_SILVER_MIN_SURROUND_CONTRAST * 3.0, 1e-6))
        aspect_score = _clip01(
            band.aspect / max(config.ENTRY_SILVER_MIN_ASPECT * 2.0, 1e-6))
        return float(
            0.26 * span_score
            + 0.14 * fill_score
            + 0.18 * range_score
            + 0.14 * highlight_score
            + 0.18 * contrast_score
            + 0.10 * aspect_score
        )


class EntrySilverGate:
    """Detector + votação temporal. É esta classe que a missão consulta.

    Somente deteccoes completas votam. Motivos de rejeicao como ``fina`` sao
    diagnosticos geometricos, nao evidencia de prata: uma curva de 90 graus
    sobre piso claro produz exatamente essa assinatura.
    """

    def __init__(self, detector=None, confirmer=None, weak_confirmer=None):
        self.detector = (
            EntrySilverDetector() if detector is None else detector)
        self.confirmer = (
            StripeConfirmer(
                votes_needed=config.ENTRY_SILVER_VOTES_NEEDED,
                window=config.ENTRY_SILVER_VOTE_WINDOW,
                max_age_s=config.ENTRY_SILVER_MAX_AGE_S,
                cooldown_s=config.ENTRY_SILVER_COOLDOWN_S,
            )
            if confirmer is None else confirmer
        )
        self.weak_confirmer = (
            StripeConfirmer(
                votes_needed=config.ENTRY_SILVER_WEAK_VOTES_NEEDED,
                window=config.ENTRY_SILVER_WEAK_VOTE_WINDOW,
                max_age_s=config.ENTRY_SILVER_MAX_AGE_S,
                cooldown_s=config.ENTRY_SILVER_COOLDOWN_S,
            )
            if weak_confirmer is None else weak_confirmer
        )
        self.last_detection = None

    @property
    def confirmed(self):
        return self.confirmer.confirmed or self.weak_confirmer.confirmed

    @property
    def votes(self):
        return max(self.confirmer.votes, self.weak_confirmer.votes)

    def reset(self, now=None):
        """Desarma e inicia o cooldown — usado ao voltar ao segue-linha."""
        self.confirmer.reset(now=now)
        self.weak_confirmer.reset(now=now)
        self.last_detection = None

    def update(self, frame_bgr, line_ahead=None, timestamp=None, now=None,
               hsv_image=None):
        if hsv_image is None:
            # Preserva compatibilidade com detectores de teste/terceiros que
            # implementam a assinatura antiga.
            detection = self.detector.detect(
                frame_bgr,
                line_ahead=line_ahead,
                timestamp=timestamp,
            )
        else:
            detection = self.detector.detect(
                frame_bgr,
                line_ahead=line_ahead,
                timestamp=timestamp,
                hsv_image=hsv_image,
            )
        self.last_detection = detection
        confirmed = self.confirmer.update(
            detection is not None,
            timestamp=0.0 if timestamp is None else timestamp,
            now=now,
        )
        # O confirmador fraco permanece para compatibilidade com objetos de
        # teste e estados ja serializados, mas nunca recebe uma rejeicao como
        # voto positivo. Antes, duas respostas ``fina`` confirmavam a entrada
        # mesmo com zero deteccoes — causa do falso resgate nas curvas de 90°.
        evidencia_fraca = detection is not None
        weak_confirmed = self.weak_confirmer.update(
            evidencia_fraca,
            timestamp=0.0 if timestamp is None else timestamp,
            now=now,
        )
        return bool(confirmed or weak_confirmed), detection

