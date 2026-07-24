"""Configuracao isolada da primeira etapa do resgate do Shadow.

Este modulo nao e importado pelo segue-linha. Alterar valores aqui nao muda o
comportamento de ``shadow/main.py``.
"""

# Camera de resgate. O dual_camera_viewer existente chama a camera 0 de
# "resgate". O indice continua exposto na CLI porque o pipeline de linha antigo
# ainda abre a camera padrao sem registrar seu indice explicitamente.
RESCUE_CAMERA_INDEX = 0
# A saida preserva a proporcao do modo de sensor com maior campo de visao.
# 640x480 reduz em 56% os pixels do antigo 960x720 sem reduzir o campo de
# visao: quem define o FoV e o modo/crop do sensor, nao a escala da saida.
RESCUE_CAMERA_MAX_WIDTH = 640
RESCUE_CAMERA_MAX_HEIGHT = 480
RESCUE_CAMERA_FPS = 30
RESCUE_LENS_POSITION = None
RESCUE_REQUIRE_TWO_CAMERAS = True
RESCUE_ROTATE_180 = True

# Modo full-FoV ja identificado no hardware frontal OV5647 do Shadow. Usar o
# modo conhecido evita consultar ``sensor_modes`` a cada partida (essa consulta
# para e reconfigura a camera varias vezes). Outros sensores usam descoberta.
RESCUE_KNOWN_SENSOR_MODES = {
    "ov5647": {
        "size": (1296, 972),
        "bit_depth": 10,
        "fps": 43.25,
        "crop_limits": (0, 0, 2592, 1944),
    },
}

# O preview permanece 640x480/full-FoV, mas a visao trabalha em 320x240. No Pi,
# o Hough a 640x480 levou 1.5--2.0 s e toda deteccao chegava stale; reduzir a
# matriz nao recorta o campo de visao e as coordenadas voltam para o preview.
RESCUE_DETECTOR_MAX_WIDTH = 320
RESCUE_DETECTOR_MAX_HEIGHT = 240
RESCUE_ARM_DELAY_S = 3.0
RESCUE_WORKER_JOIN_TIMEOUT_S = 2.0

# Melhoria de iluminacao ja experimentada no dual_camera_viewer.
RESCUE_CLAHE_CLIP = 2.0
RESCUE_CLAHE_GRID = (8, 8)
RESCUE_GAMMA = 1.5

# Regiao e propostas geometricas.
# Limiares em pixels abaixo foram calibrados em 640x480 e sao escalados
# automaticamente sem confundir uma imagem mais larga com uma esfera maior.
BALL_BASE_WIDTH = 640
BALL_BASE_HEIGHT = 480


def ball_pixel_scale(frame_width, frame_height):
    """Escala isotropica dos limiares da calibracao 640x480."""
    return max(min(
        float(frame_width) / BALL_BASE_WIDTH,
        float(frame_height) / BALL_BASE_HEIGHT,
    ), 0.25)


BALL_ROI_TOP = 0.12
# O circulo pode continuar valido ate a ultima linha. Uma tolerancia pequena
# deixa o lock sobreviver quando a base acabou de ser cortada, sem mover para
# cima o ponto fisico que dispara a coleta.
BALL_ROI_BOTTOM = 1.00
BALL_ROI_BOTTOM_OVERFLOW_RATIO = 0.03
BALL_MIN_RADIUS_PX = 9
BALL_MAX_RADIUS_PX = 135
BALL_MIN_CIRCULARITY = 0.56
BALL_MIN_FILL_RATIO = 0.50
BALL_MAX_ASPECT_RATIO = 1.32
BALL_MIN_EDGE_SUPPORT = 0.22

# Hough + bordas. Os contornos de mascara cobrem a esfera preta; Hough e
# contraste local cobrem a esfera prateada/reflexiva.
BALL_MEDIAN_BLUR = 5
BALL_CANNY_SIGMA = 0.33
BALL_HOUGH_DP = 1.2
BALL_HOUGH_MIN_DIST_PX = 28
BALL_HOUGH_PARAM1 = 105
BALL_HOUGH_PARAM2 = 18
BALL_HOUGH_MIN_CONFIDENCE = 0.66
# Um contorno forte ja passou por circularidade, borda e aparencia. Nessa
# situacao, Hough redundante durante os 3 hits de aquisicao so adiciona atraso.
BALL_CONTOUR_FAST_CONFIDENCE = 0.78

# Aparencia no frame original (classificacao nunca usa o gamma).
BALL_BLACK_V_MAX = 105
BALL_BLACK_DARK_FRACTION_MIN = 0.52
BALL_BLACK_LOCAL_CONTRAST_MIN = 8.0
BALL_SILVER_S_MAX = 88
# Referencia de bonus, nao gate: aluminio reflete a cor do iluminante e pode
# ficar ciano/verde com saturacao alta mesmo continuando metalico.
BALL_SILVER_LOW_SAT_FRACTION_MIN = 0.62
BALL_SILVER_DYNAMIC_RANGE_MIN = 20.0
BALL_SILVER_HIGHLIGHT_V = 195
BALL_SILVER_HIGHLIGHT_FRACTION_MIN = 0.015
# Rota conservadora para aluminio refletindo luz ciano/verde. Ela dispensa a
# neutralidade global somente quando textura, brilho quase neutro e borda sao
# simultaneamente muito fortes.
BALL_SILVER_TINTED_INNER_V_MIN = 110
BALL_SILVER_TINTED_DYNAMIC_RANGE_MIN = 40.0
BALL_SILVER_TINTED_HIGHLIGHT_FRACTION_MIN = 0.05
BALL_SILVER_TINTED_NEUTRAL_S_MAX = 120
BALL_SILVER_TINTED_NEUTRAL_HIGHLIGHT_MIN = 0.015
BALL_SILVER_TINTED_EDGE_SUPPORT_MIN = 0.35
BALL_MIN_CONFIDENCE = 0.56

# Rastreamento e confirmacao temporal.
BALL_ACQUIRE_HITS = 3
BALL_MAX_TRACK_MISSES = 2
# Uma falha isolada nao troca a identidade nem apaga a confirmacao do track.
# O controle ainda manda PARAR naquele frame; somente a memoria visual sobrevive.
BALL_TRACK_COAST_MISSES = 1
BALL_ASSOCIATION_MIN_PX = 34
BALL_ASSOCIATION_RADIUS_FACTOR = 1.05
BALL_RADIUS_RATIO_MIN = 0.62
BALL_RADIUS_RATIO_MAX = 1.60
# Antes dos 3 hits, reflexos internos nao podem ser associados como se fossem
# o mesmo perimetro externo. Depois da confirmacao, os limites amplos acima
# continuam cobrindo o movimento real do robo e pequenas perdas de quadro.
BALL_ACQUIRE_ASSOCIATION_MIN_PX = 16
BALL_ACQUIRE_ASSOCIATION_RADIUS_FACTOR = 0.80
BALL_ACQUIRE_RADIUS_RATIO_MIN = 0.72
BALL_ACQUIRE_RADIUS_RATIO_MAX = 1.40
BALL_TRACK_EMA_ALPHA = 0.40
# O segundo gate temporal do worker tambem tolera somente uma falha entre
# resultados novos do mesmo track bloqueado.
BALL_FRESH_GATE_MAX_MISSES = 1

# Propostas quase identicas sao redundantes; circulos concentricos com raios
# diferentes precisam chegar a classificacao para um halo invalido nao apagar
# o perimetro verdadeiro.
BALL_DUPLICATE_CENTER_FACTOR = 0.25
BALL_DUPLICATE_RADIUS_RATIO_MIN = 0.82

# Entre candidatos ja validados, um circulo menor pode ser apenas um reflexo
# dentro da esfera. A preferencia pelo envelope externo so vale quando existe
# contencao geometrica e a confianca externa permanece proxima da interna.
BALL_OUTER_MIN_RADIUS_RATIO = 1.15
BALL_OUTER_MAX_RADIUS_RATIO = 1.80
BALL_OUTER_CENTER_FACTOR = 0.45
BALL_OUTER_CONTAINMENT_SLACK = 1.15
BALL_OUTER_CONFIDENCE_TOLERANCE = 0.18

# Controle de aproximacao. O comando usa a lei steer() ja existente:
# positivo=direita, negativo=esquerda, |angulo|>110=pivo, 190=PARAR.
BALL_CENTER_DEADBAND = 0.10
# ALIGN entra somente acima de 0,24 e, depois de entrar, sai abaixo de 0,15.
# Em vez do pivo brusco |angulo|>110, usa arco proporcional 65..82. Pela lei
# steer, a roda interna preserva cerca de 26%..41% da externa, em vez de ficar
# com apenas 1..5 PWM e se comportar como um pivo apoiado em uma roda.
BALL_ALIGN_THRESHOLD = 0.24
BALL_ALIGN_EXIT_THRESHOLD = 0.15
BALL_ALIGN_ARC_MIN_ANGLE = 65
BALL_ALIGN_ARC_MAX_ANGLE = 82
BALL_ALIGN_SPEED_MIN = 0.30
BALL_ALIGN_SPEED_MAX = 0.34
BALL_STEER_MAX_ANGLE = 60
BALL_APPROACH_SPEED_FAR = 0.45
BALL_APPROACH_SPEED_NEAR = 0.35
BALL_SLOW_RADIUS_PX = 48

# O raio ainda controla apenas a desaceleracao. A coleta nao e mais disparada
# por "circulo dentro de um retangulo": perto da garra a esfera fica maior que
# o quadro e o Hough rejeita o contorno cortado. O gate definitivo usa a
# meia-lua larga descrita abaixo.
BALL_STOP_RADIUS_PX = 76
BALL_STOP_CONFIRM_FRAMES = 3
# Gate primario pedido no teste fisico: o circulo temporal ja bloqueado na
# vitima cobre um ponto perto da base. Tamanho, centro, crescimento anterior
# e dois timestamps frescos impedem um reflexo pequeno de acionar a coleta.
BALL_LOCKED_CIRCLE_POINT_X_RATIO = 0.50
# 0,98 ficou dentro da zona em que o Hough ja perde o circulo cortado. Em
# 0,95 a borda ainda esta muito baixa, mas sobra uma janela real para obter as
# duas medicoes frescas apesar do EMA, de um drop ou de pequeno erro lateral.
BALL_LOCKED_CIRCLE_POINT_Y_RATIO = 0.95
BALL_LOCKED_CIRCLE_POINT_SLACK_RATIO = 0.00
BALL_LOCKED_CIRCLE_MIN_RADIUS_RATIO = 0.085
BALL_LOCKED_CIRCLE_MAX_CENTER_ERROR = 0.16
BALL_LOCKED_CIRCLE_CONFIRM_FRAMES = 2
# Somente uma medicao ausente pode separar as duas confirmacoes. A falha
# preserva o contador por poucos milissegundos, mas sempre deixa as rodas em
# PARAR e nunca incrementa sozinha.
BALL_NEAR_CONFIRM_MAX_MISSES = 1
BALL_NEAR_CONFIRM_GRACE_S = 0.18
BALL_NEAR_CONFIRM_WINDOW_S = 0.35

# Gate de proximidade pela borda superior da esfera enorme/cortada. Cada
# template é o arco circular que passa pelo ápice e pelos dois ombros.
# Os frames reais colocaram o ápice entre 0,62H e 0,74H. Uma esfera distante
# não cobre simultaneamente os dois ombros e o centro.
BALL_CRESCENT_TOP_RATIOS = (0.62, 0.66, 0.70, 0.74)
# Os brutos reais ocupam aproximadamente 80–92% da largura. Exigir essa meia
# lua larga impede que a perspectiva de uma bolinha distante arme a coleta.
BALL_CRESCENT_HALFSPAN_RATIOS = (0.40, 0.46)
BALL_CRESCENT_CENTER_RATIOS = (0.44, 0.48, 0.50, 0.52, 0.56)
BALL_CRESCENT_BOTTOM_RATIO = 0.98
BALL_CRESCENT_DEFAULT_TOP_RATIO = 0.70
BALL_CRESCENT_DEFAULT_HALFSPAN_RATIO = 0.46
BALL_CRESCENT_BAND_RATIO = 0.035
BALL_CRESCENT_CONTRAST_OFFSET_RATIO = 0.025
BALL_CRESCENT_OUTSIDE_CONTRAST_OFFSET_RATIO = 0.050
BALL_CRESCENT_DEEP_CONTRAST_OFFSET_RATIO = 0.075
BALL_CRESCENT_DEEP_INNER_X_RATIO = 0.70
BALL_CRESCENT_SAMPLES = 73
BALL_CRESCENT_MIN_SUPPORT = 0.55
BALL_CRESCENT_MIN_SHOULDER_SUPPORT = 0.40
BALL_CRESCENT_MIN_CENTER_SUPPORT = 0.55
BALL_CRESCENT_MIN_CONTRAST = 10.0
BALL_CRESCENT_MIN_GRADIENT = 12.0
BALL_CRESCENT_MIN_GRADIENT_ALIGNMENT = 0.82
BALL_CRESCENT_MIN_GRADIENT_POLARITY = 0.62
BALL_CRESCENT_MIN_PROFILE_SUPPORT = 0.55
BALL_CRESCENT_MIN_PROFILE_POLARITY = 0.62
BALL_CRESCENT_MIN_COHERENT_RUN = 0.18
# A silhueta da esfera de foil tem pequenos dentes. A suavização é aplicada
# apenas aos pontos já validados da borda, antes dos testes de forma global.
BALL_CRESCENT_SMOOTH_SAMPLES = 9
BALL_CRESCENT_MAX_CIRCLE_RMSE_RATIO = 0.008
BALL_CRESCENT_CURVATURE_BINS = 7
BALL_CRESCENT_MIN_CURVATURE_SCORE = 0.95
# Exige curvatura também nos ombros. Um V ligado a uma bolinha ainda pequena
# pode parecer circular no miolo, mas seus incrementos externos ficam < 0.08.
BALL_CRESCENT_MIN_SLOPE_STEP = 0.08
BALL_CRESCENT_MIN_SLOPE_SPAN = 0.45
BALL_CRESCENT_MAX_CENTER_ERROR = 0.12

# Segunda rota exclusiva para o papel-alumínio amassado/desfocado. A forma
# pode perder Canny e circularidade local, mas precisa ter reflexos distribuídos
# dentro do domo e fundo muito mais limpo; sombras e rampas sólidas não passam.
BALL_CRESCENT_FOIL_MIN_SUPPORT = 0.45
BALL_CRESCENT_FOIL_MIN_SHOULDER_SUPPORT = 0.35
BALL_CRESCENT_FOIL_MIN_CENTER_SUPPORT = 0.45
BALL_CRESCENT_FOIL_MIN_COHERENT_RUN = 0.16
BALL_CRESCENT_FOIL_MAX_CIRCLE_RMSE_RATIO = 0.025
# Um candidato por par (altura, largura); evita gastar as tres vagas apenas
# com pequenos deslocamentos horizontais da mesma forma.
BALL_CRESCENT_FOIL_MAX_CANDIDATES = 3
BALL_CRESCENT_FOIL_TEXTURE_BINS = 5
BALL_CRESCENT_FOIL_MIN_TEXTURE_BINS = 4
BALL_CRESCENT_FOIL_MIN_DYNAMIC_RANGE = 35.0
BALL_CRESCENT_FOIL_INNER_X_RATIO = 0.75
BALL_CRESCENT_FOIL_INSIDE_OFFSETS = (0.035, 0.070, 0.105, 0.140)
BALL_CRESCENT_FOIL_OUTSIDE_OFFSETS = (0.060, 0.100, 0.140)
BALL_CRESCENT_FOIL_MIN_INTERIOR_EDGE_DENSITY = 0.02
BALL_CRESCENT_FOIL_MAX_BACKGROUND_EDGE_DENSITY = 0.04
BALL_CRESCENT_FOIL_BACKGROUND_EDGE_RATIO = 0.60

# A meia-lua so pode concluir uma aproximacao visual real. O token e armado
# por uma serie temporal de circulos centralizados, crescentes e ja baixos no
# quadro; ele sobrevive por pouco tempo quando o perimetro sai do ROI.
BALL_CRESCENT_HISTORY_S = 1.80
BALL_CRESCENT_HISTORY_MIN_SAMPLES = 4
BALL_CRESCENT_HISTORY_MIN_SPAN_S = 0.20
BALL_CRESCENT_HISTORY_MIN_FORWARD_S = 0.12
BALL_CRESCENT_ARM_RADIUS_RATIO = 0.07
BALL_CRESCENT_ARM_BOTTOM_RATIO = 0.76
BALL_CRESCENT_ARM_RADIUS_GROWTH_RATIO = 0.012
BALL_CRESCENT_ARM_BOTTOM_GROWTH_RATIO = 0.025
BALL_CRESCENT_ARM_MAX_CENTER_ERROR = 0.24
BALL_CRESCENT_ASSOCIATION_X_RATIO = 0.10
# Perto demais, o Hough pode escolher um reflexo interno deslocado (no frame
# medido, x=424 enquanto a esfera externa estava centrada em 640 px).
BALL_CRESCENT_INNER_ASSOCIATION_X_RATIO = 0.22
BALL_CRESCENT_INNER_BOTTOM_RATIO = 0.82
BALL_CRESCENT_TOKEN_TTL_S = 0.80

# Coleta depois que a aproximacao visual termina. Nao existe etapa de re.
# O avanco usa a velocidade conservadora ja validada perto da esfera. O Futaba
# e continuo:
# -20 e potencia de descida por 1500 ms, nao um angulo. A margem garante que
# CH3 ja foi desligado pelo firmware antes do avanco com as duas garras.
BALL_PICKUP_FUTABA_POWER = -20
BALL_PICKUP_FUTABA_MS = 1500
BALL_PICKUP_FUTABA_GUARD_S = 0.10
BALL_PICKUP_LEFT_DELTA = -50
BALL_PICKUP_RIGHT_DELTA = 50
# O motor recebe o avanco antes das garras. Esta curta vantagem deixa as rodas
# vencerem a inercia; depois ambas as garras fecham no mesmo lote USB. Os
# 2,00 s sao contados desde o comando das rodas, incluindo essa vantagem.
BALL_PICKUP_FORWARD_LEAD_S = 0.12
BALL_PICKUP_FORWARD_S = 2.00
BALL_PICKUP_FORWARD_SPEED = BALL_APPROACH_SPEED_NEAR

# Hough + filtros medidos no Pi podem ultrapassar 0.20 s. O timestamp agora e
# tirado depois da captura; 0.75 s ainda impede movimento com imagem congelada,
# mas nao rejeita todo frame valido como ocorreu no primeiro teste fisico.
BALL_FRAME_STALE_S = 0.75
BALL_REACQUIRE_TIMEOUT_S = 1.0
BALL_MAX_WAIT_S = 30.0
BALL_MAX_ACTIVE_S = 45.0
BALL_PROGRESS_WINDOW_S = 3.0
BALL_PROGRESS_MIN_RADIUS_PX = 3.0
BALL_PROGRESS_MIN_BOTTOM_Y_PX = 8.0
