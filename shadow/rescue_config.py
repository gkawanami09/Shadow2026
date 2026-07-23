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
BALL_ROI_BOTTOM = 0.98
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
BALL_ASSOCIATION_MIN_PX = 34
BALL_ASSOCIATION_RADIUS_FACTOR = 1.7
BALL_RADIUS_RATIO_MIN = 0.55
BALL_RADIUS_RATIO_MAX = 1.80
# Antes dos 3 hits, reflexos internos nao podem ser associados como se fossem
# o mesmo perimetro externo. Depois da confirmacao, os limites amplos acima
# continuam cobrindo o movimento real do robo e pequenas perdas de quadro.
BALL_ACQUIRE_ASSOCIATION_MIN_PX = 16
BALL_ACQUIRE_ASSOCIATION_RADIUS_FACTOR = 0.80
BALL_ACQUIRE_RADIUS_RATIO_MIN = 0.72
BALL_ACQUIRE_RADIUS_RATIO_MAX = 1.40
BALL_TRACK_EMA_ALPHA = 0.40

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
BALL_CENTER_DEADBAND = 0.085
BALL_ALIGN_THRESHOLD = 0.19
BALL_ALIGN_ANGLE = 180
# Os valores anteriores (0.20..0.38) geravam apenas pulsos de PWM 24..46,
# intercalados por PARAR, e nao venceram a inercia no teste fisico. Estes
# valores continuam abaixo dos 60 PWM usados pelo segue-linha, mas deixam uma
# margem real para os quatro motores com a LiPo 2S.
BALL_ALIGN_SPEED = 0.35
BALL_STEER_MAX_ANGLE = 82
BALL_APPROACH_SPEED_FAR = 0.45
BALL_APPROACH_SPEED_NEAR = 0.35
BALL_SLOW_RADIUS_PX = 48

# Parada perto da esfera. Estes dois valores sao deliberadamente conservadores
# e precisam ser calibrados com a camera montada no robo.
BALL_STOP_RADIUS_PX = 76
BALL_STOP_BOTTOM_Y_RATIO = 0.78
BALL_STOP_CENTER_ERROR = 0.12
BALL_STOP_CONFIRM_FRAMES = 3

# Coleta depois que a aproximacao visual termina. A re usa a mesma velocidade
# conservadora ja validada perto da esfera. O Futaba e continuo: -20 e potencia
# de descida por 1500 ms, nao um angulo. A margem garante que CH3 ja foi
# desligado pelo firmware antes de mover as duas garras.
BALL_PICKUP_REVERSE_S = 0.50
BALL_PICKUP_REVERSE_SPEED = BALL_APPROACH_SPEED_NEAR
BALL_PICKUP_FUTABA_POWER = -20
BALL_PICKUP_FUTABA_MS = 1500
BALL_PICKUP_FUTABA_GUARD_S = 0.10
BALL_PICKUP_LEFT_DELTA = -50
BALL_PICKUP_RIGHT_DELTA = 50
BALL_PICKUP_GRIPPER_SETTLE_S = 0.50

# Ultrassom e travas de seguranca. O HC-SR04 e somente uma barreira auxiliar:
# esfera pequena pode nao devolver eco e parede pode devolver.
BALL_ULTRASONIC_POLL_S = 0.20
BALL_ULTRASONIC_TIMEOUT_S = 0.05
BALL_ULTRASONIC_MIN_VALID_MM = 35
BALL_ULTRASONIC_STOP_MM = 145
BALL_ULTRASONIC_CONFIRM_READS = 2
BALL_ULTRASONIC_HOLD_TIMEOUT_S = 1.0

# Hough + filtros medidos no Pi podem ultrapassar 0.20 s. O timestamp agora e
# tirado depois da captura; 0.75 s ainda impede movimento com imagem congelada,
# mas nao rejeita todo frame valido como ocorreu no primeiro teste fisico.
BALL_FRAME_STALE_S = 0.75
BALL_REACQUIRE_TIMEOUT_S = 1.0
BALL_MAX_WAIT_S = 30.0
BALL_MAX_ACTIVE_S = 45.0
BALL_PROGRESS_WINDOW_S = 3.0
BALL_PROGRESS_MIN_RADIUS_PX = 3.0
