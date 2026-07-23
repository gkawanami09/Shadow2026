"""Configuracao isolada da primeira etapa do resgate do Shadow.

Este modulo nao e importado pelo segue-linha. Alterar valores aqui nao muda o
comportamento de ``shadow/main.py``.
"""

# Camera de resgate. O dual_camera_viewer existente chama a camera 0 de
# "resgate". O indice continua exposto na CLI porque o pipeline de linha antigo
# ainda abre a camera padrao sem registrar seu indice explicitamente.
RESCUE_CAMERA_INDEX = 0
# A saida preserva a proporcao do modo de sensor com maior campo de visao.
# Estes limites produzem 960x540 em sensores 16:9 e 960x720 em sensores 4:3.
RESCUE_CAMERA_MAX_WIDTH = 960
RESCUE_CAMERA_MAX_HEIGHT = 720
RESCUE_CAMERA_FPS = 30
RESCUE_LENS_POSITION = None
RESCUE_REQUIRE_TWO_CAMERAS = True
RESCUE_ROTATE_180 = True

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
BALL_MEDIAN_BLUR = 7
BALL_CANNY_SIGMA = 0.33
BALL_HOUGH_DP = 1.2
BALL_HOUGH_MIN_DIST_PX = 28
BALL_HOUGH_PARAM1 = 105
BALL_HOUGH_PARAM2 = 18
BALL_HOUGH_MIN_CONFIDENCE = 0.66

# Aparencia no frame original (classificacao nunca usa o gamma).
BALL_BLACK_V_MAX = 105
BALL_BLACK_DARK_FRACTION_MIN = 0.52
BALL_BLACK_LOCAL_CONTRAST_MIN = 8.0
BALL_SILVER_S_MAX = 88
BALL_SILVER_LOW_SAT_FRACTION_MIN = 0.62
BALL_SILVER_DYNAMIC_RANGE_MIN = 20.0
BALL_SILVER_HIGHLIGHT_V = 195
BALL_SILVER_HIGHLIGHT_FRACTION_MIN = 0.015
BALL_MIN_CONFIDENCE = 0.56

# Rastreamento e confirmacao temporal.
BALL_ACQUIRE_HITS = 3
BALL_MAX_TRACK_MISSES = 2
BALL_ASSOCIATION_MIN_PX = 34
BALL_ASSOCIATION_RADIUS_FACTOR = 1.7
BALL_RADIUS_RATIO_MIN = 0.55
BALL_RADIUS_RATIO_MAX = 1.80
BALL_TRACK_EMA_ALPHA = 0.40

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

# Ultrassom e travas de seguranca. O HC-SR04 e somente uma barreira auxiliar:
# esfera pequena pode nao devolver eco e parede pode devolver.
BALL_ULTRASONIC_POLL_S = 0.20
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
