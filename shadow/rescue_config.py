"""Configuracao isolada da primeira etapa do resgate do Shadow.

Este modulo nao e importado pelo segue-linha. Alterar valores aqui nao muda o
comportamento de ``shadow/main.py``.
"""

# Camera de resgate. O dual_camera_viewer existente chama a camera 0 de
# "resgate". O indice continua exposto na CLI porque o pipeline de linha antigo
# ainda abre a camera padrao sem registrar seu indice explicitamente.
RESCUE_CAMERA_INDEX = 0
RESCUE_CAMERA_WIDTH = 640
RESCUE_CAMERA_HEIGHT = 480
RESCUE_CAMERA_FPS = 30
RESCUE_LENS_POSITION = None
RESCUE_REQUIRE_TWO_CAMERAS = True

# Melhoria de iluminacao ja experimentada no dual_camera_viewer.
RESCUE_CLAHE_CLIP = 2.0
RESCUE_CLAHE_GRID = (8, 8)
RESCUE_GAMMA = 1.5

# Regiao e propostas geometricas.
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
BALL_ALIGN_SPEED = 0.22
BALL_STEER_MAX_ANGLE = 82
BALL_APPROACH_SPEED_FAR = 0.38
BALL_APPROACH_SPEED_NEAR = 0.20
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

BALL_FRAME_STALE_S = 0.20
BALL_REACQUIRE_TIMEOUT_S = 1.0
BALL_MAX_WAIT_S = 30.0
BALL_MAX_ACTIVE_S = 30.0
BALL_PROGRESS_WINDOW_S = 3.0
BALL_PROGRESS_MIN_RADIUS_PX = 3.0
