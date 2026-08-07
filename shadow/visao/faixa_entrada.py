"""Faixa PRATA de entrada da sala de resgate, vista pela CÂMERA DE LINHA.

Este detector é deliberadamente separado do detector de vítimas prateadas.
São câmeras diferentes, alturas diferentes, iluminação diferente e objetos
diferentes: a vítima é uma esfera de ~4 cm e a entrada é uma fita de
25 x 250 mm colada no chão. Compartilhar limiares entre os dois seria o
caminho mais rápido para o robô entrar na sala ao ver uma vítima, ou ignorar
a entrada por estar calibrado para uma esfera.

A confirmação exige evidência CONJUNTA, nunca um único sinal:

1. região neutra e não-preta na ROI inferior (máscara HSV calibrável);
2. forma alongada e transversal (``faixa_transversal``);
3. largura mínima proporcional ao quadro;
4. caixa não dominada por preto — veto direto contra intersecção e linha
   transversal, medido na imagem crua;
5. baixa saturação (neutralidade metálica);
6. assinatura reflexiva — faixa dinâmica e brilho especular concentrado,
   que é o que separa fita refletiva de papel branco fosco;
7. contraste contra a vizinhança imediata, acima e abaixo da faixa;
8. fim coerente da linha preta à frente;
9. votação temporal em frames distintos e recentes, com cooldown.

O peso da separação NÃO está na janela HSV — ela é larga de propósito, porque
a fita real muda muito de brilho com o ângulo. Quem separa fita de piso é o
filtro de textura da luz (``ENTRY_SILVER_MIN_LOCAL_RANGE``); quem separa fita
de linha/intersecção é o veto de escuro somado à geometria transversal.

A busca é feita por LINHAS horizontais, então uma faixa inclinada — o robô
chegando torto na soleira — não preenche linha nenhuma e escapa de todos os
testes acima sem nunca ser julgada. Para isso existe a escada de ângulos: se
a busca direta falha, o quadro é girado alguns graus por vez e a MESMA busca
roda de novo. Nenhum limiar é afrouxado; muda só a orientação do quadro.

Os limiares seguem ajustáveis com ``tools/calibrar_cores.py`` (grupo 7).
"""

from dataclasses import dataclass, replace

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
    #: Graus que a imagem precisou girar para a faixa ficar horizontal. Zero
    #: quando a faixa foi encontrada direto, sem endireitar nada.
    tilt_deg: float = 0.0


def escada_de_inclinacao(passo=None, maximo=None):
    """Ângulos a tentar, do menos torto para o mais torto.

    Alterna os dois lados (-8, +8, -16, +16…) para que a chegada torta mais
    provável — a de poucos graus — seja resolvida nos primeiros degraus, sem
    pagar a escada inteira.
    """
    passo = config.ENTRY_SILVER_TILT_STEP_DEG if passo is None else float(
        passo)
    maximo = config.ENTRY_SILVER_MAX_TILT_DEG if maximo is None else float(
        maximo)
    if passo <= 0 or maximo < passo:
        return ()
    angulos = []
    atual = passo
    while atual <= maximo + 1e-9:
        angulos.extend((-atual, atual))
        atual += passo
    return tuple(angulos)


def _rotacionar(imagem, graus, borda=cv2.BORDER_REPLICATE,
                interpolacao=cv2.INTER_NEAREST):
    """Gira em torno do centro do quadro.

    Três escolhas que parecem detalhe e não são:

    * **Sem interpolação.** Girar com ``INTER_LINEAR`` mistura o preto da
      linha com o branco do piso vizinho e clareia os pixels escuros; uma
      faixa PRETA inclinada escapava do veto de escuro só por causa disso.
      ``INTER_NEAREST`` copia o pixel original inteiro. O detector decide por
      medianas e frações, então o serrilhado não custa nada.
    * **Borda replicada no quadro.** Cantos pretos artificiais entrariam no
      veto de escuro e reprovariam uma faixa boa por um defeito que o próprio
      detector criou.
    * **Borda zerada na máscara.** Replicar máscara inventaria linhas cheias
      que não existem — exatamente o falso positivo a evitar.
    """
    altura, largura = imagem.shape[:2]
    matriz = cv2.getRotationMatrix2D(
        (largura / 2.0, altura / 2.0), graus, 1.0)
    girada = cv2.warpAffine(
        imagem,
        matriz,
        (largura, altura),
        flags=interpolacao,
        borderMode=borda,
    )
    return girada, matriz


def _bbox_para_o_quadro_original(bbox, matriz):
    """Traz a caixa achada no quadro girado de volta ao quadro real.

    A caixa volta inclinada; guardamos a caixa alinhada que a contém. É uma
    aproximação, e ela só alimenta o desenho do HUD — nenhuma decisão do
    detector usa este valor.
    """
    x, y, largura, altura = bbox
    cantos = np.array(
        [[x, y],
         [x + largura - 1, y],
         [x + largura - 1, y + altura - 1],
         [x, y + altura - 1]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    de_volta = cv2.transform(
        cantos, cv2.invertAffineTransform(matriz)).reshape(-1, 2)
    x0, y0 = de_volta.min(axis=0)
    x1, y1 = de_volta.max(axis=0)
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x1 - x0 + 1)),
        int(round(y1 - y0 + 1)),
    )


def default_geometry():
    return BandGeometry(
        min_row_fill=config.ENTRY_SILVER_MIN_ROW_FILL,
        min_span_ratio=config.ENTRY_SILVER_MIN_SPAN_RATIO,
        max_span_ratio=config.ENTRY_SILVER_MAX_SPAN_RATIO,
        min_thickness_ratio=config.ENTRY_SILVER_MIN_THICKNESS_RATIO,
        max_thickness_ratio=config.ENTRY_SILVER_MAX_THICKNESS_RATIO,
        min_fill_ratio=config.ENTRY_SILVER_MIN_FILL_RATIO,
        min_aspect=config.ENTRY_SILVER_MIN_ASPECT,
        allow_top_touch=config.ENTRY_SILVER_ALLOW_TOP_TOUCH,
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

        Quando a busca direta falha e a mancha na máscara está claramente
        inclinada, a imagem é girada de volta ao horizontal e a MESMA busca
        roda outra vez. É o caso do robô chegando torto na soleira: a faixa
        está inteira e nítida no quadro, mas atravessa as linhas da imagem em
        diagonal e nenhuma delas fica cheia.
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("EntrySilverDetector exige um frame BGR")

        timestamp = 0.0 if timestamp is None else float(timestamp)
        deteccao = self._detectar_direto(
            frame_bgr, line_ahead, timestamp, hsv_image)
        if deteccao is not None or not config.ENTRY_SILVER_DESKEW_ENABLED:
            return deteccao

        motivo_torto = self.last_reason
        mask_torta = self.last_mask
        deteccao = self._procurar_inclinada(
            frame_bgr, hsv_image, line_ahead, timestamp, mask_torta)
        if deteccao is None:
            # Nenhum ângulo funcionou: o diagnóstico útil é o do quadro
            # original, não o do último degrau tentado.
            self.last_reason = motivo_torto
            self.last_mask = mask_torta
        return deteccao

    def _procurar_inclinada(self, frame_bgr, hsv_image, line_ahead, timestamp,
                            mask):
        """Varre a escada de ângulos procurando a faixa endireitada.

        A sonda é barata de propósito: gira só a MÁSCARA e pergunta à
        geometria se apareceu uma faixa. O quadro inteiro — que custa uma
        conversão HSV e todos os testes de aparência — só é girado no ângulo
        em que a geometria fechou. Num quadro sem nada, a máscara é vazia e a
        escada inteira custa menos que uma detecção completa.
        """
        if mask is None or not self._pode_haver_faixa(mask):
            return None
        for graus in escada_de_inclinacao():
            mask_girada, _matriz_mask = _rotacionar(
                mask, graus,
                borda=cv2.BORDER_CONSTANT,
                interpolacao=cv2.INTER_NEAREST,
            )
            banda, _motivo = find_transversal_band(
                mask_girada,
                self.geometry,
                roi_top_ratio=config.ENTRY_SILVER_ROI_TOP,
                roi_bottom_ratio=config.ENTRY_SILVER_ROI_BOTTOM,
            )
            if banda is None:
                continue

            girado, matriz = _rotacionar(frame_bgr, graus)
            hsv_girado = (
                None if hsv_image is None
                else _rotacionar(hsv_image, graus)[0]
            )
            deteccao = self._detectar_direto(
                girado, line_ahead, timestamp, hsv_girado)
            if deteccao is None:
                continue
            if deteccao.confidence < config.ENTRY_SILVER_DESKEW_MIN_CONFIDENCE:
                continue

            self.last_reason = ""
            return replace(
                deteccao,
                tilt_deg=float(graus),
                bbox=_bbox_para_o_quadro_original(deteccao.bbox, matriz),
            )
        return None

    @staticmethod
    def _limiar_de_escuro(value):
        """Abaixo de quanto um pixel conta como preto NESTA cena.

        Ancorado no percentil 75 do brilho da ROI — na prática, o piso, que
        domina a região mesmo quando a fita ocupa boa parte do quadro. Sob luz
        fraca a âncora desce junto com a cena e a fita continua sendo lida
        como fita, em vez de virar "preto" por causa da lâmpada.
        """
        altura = value.shape[0]
        topo = int(round(altura * config.ENTRY_SILVER_ROI_TOP))
        topo = max(0, min(topo, altura - 1))
        claro = float(np.percentile(value[topo:, :], 75))
        return float(np.clip(
            claro * config.ENTRY_SILVER_DARK_V_RATIO,
            config.ENTRY_SILVER_DARK_V_MIN,
            config.ENTRY_SILVER_DARK_V_MAX,
        ))

    def _pode_haver_faixa(self, mask):
        """Há pixels suficientes na ROI para QUALQUER faixa válida existir?

        Não é um palpite: é a contagem exata de pixels da menor faixa que a
        geometria aceitaria (largura mínima × espessura mínima × preenchimento
        mínimo). Girar não cria pixel — só pode perder, nos cantos. Então
        abaixo desta conta nenhum ângulo da escada teria como dar certo, e
        pular é seguro por construção, não por tolerância.

        Na prática é isto que mantém o custo baixo: no percurso normal a
        máscara está praticamente vazia e a escada nem começa.
        """
        altura, largura = mask.shape[:2]
        topo = int(round(altura * config.ENTRY_SILVER_ROI_TOP))
        base = int(round(altura * config.ENTRY_SILVER_ROI_BOTTOM))
        topo = max(0, min(topo, altura - 1))
        base = max(topo + 1, min(base, altura))
        minimo = (
            self.geometry.min_span_ratio * largura
            * self.geometry.min_thickness_ratio * altura
            * self.geometry.min_fill_ratio
        )
        return cv2.countNonZero(mask[topo:base, :]) >= minimo

    def _detectar_direto(self, frame_bgr, line_ahead, timestamp, hsv_image):
        """Busca a faixa neste quadro, exatamente como ele chegou."""
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

        # Veto de escuro: a caixa da candidata, medida na imagem CRUA, não pode
        # ser dominada por preto. Numa intersecção a máscara pega apenas a
        # auréola clara na borda da linha e a forma até parece uma fita — mas o
        # miolo da caixa continua preto, e é isso que o teste enxerga. A fita
        # prata vista de raso fica escura, nunca preta nesse grau.
        caixa = value[band.top_y:band.bottom_y + 1,
                      band.left_x:band.right_x + 1]
        dark_fraction = float(
            np.count_nonzero(caixa < self._limiar_de_escuro(value))
            / float(max(caixa.size, 1)))
        if dark_fraction > config.ENTRY_SILVER_MAX_DARK_FRACTION:
            self.last_reason = "escura"
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
