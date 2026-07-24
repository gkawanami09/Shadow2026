"""Configurações do segue-linha e do hardware do Shadow2026."""

from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos do projeto
# ----------------------------------------------------------------------------
SHADOW_ROOT = Path(__file__).resolve().parent
CONFIG_INI_PATH = SHADOW_ROOT / "config.ini"

# ----------------------------------------------------------------------------
# Serial e Arduino Uno
# ----------------------------------------------------------------------------
SERIAL_BAUD = 115200
# Ordem de sondagem das portas; COM* fica por ultimo (apenas teste em bancada).
SERIAL_PORT_PREFIXES = ("/dev/ttyACM", "/dev/ttyUSB", "COM")
SERIAL_BANNER = "Arduino pronto"          # banner do firmware SPEC 01
SERIAL_HANDSHAKE_TIMEOUT = 5.0            # s — tempo total de auto-deteccao
SERIAL_RETRY_BACKOFF = 0.5                # s — espera entre tentativas
SERIAL_KEEPALIVE_S = 0.25                 # s — reenvio do ultimo comando (watchdog 1 s no Uno)
SERIAL_MIN_RESEND_S = 0.05                # s — dedupe de comandos identicos
SERIAL_RECONNECT_BACKOFF = 0.5            # s — espera minima entre tentativas de reconexao
MAX_PWM = 120                             # teto absoluto; firmware tambem trava em 120

# ----------------------------------------------------------------------------
# Câmera: captura 640×480 e processamento em 448×252
# ----------------------------------------------------------------------------
# Mapeamento físico atual do Pi 5: índice 0 = resgate; índice 1 = segue-linha
# no flat 2. Nunca deixar Picamera2 escolher a câmera padrão neste processo.
LINE_CAMERA_INDEX = 1
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 40                          # FrameDurationLimits = 1e6/40 = 25000 µs
camera_x = 448                            # resolucao do algoritmo
camera_y = 252
LENS_POSITION = None                      # None = foco fixo; ajuste se o modulo tiver AF
DEBUG_SHM_NAME = "shadow_shm_cam"
DEBUG_SHM_SIZE = camera_x * camera_y * 3  # 338688 B

# ----------------------------------------------------------------------------
# Controle proporcional da direção
# ----------------------------------------------------------------------------
max_turn_angle = 110                      # acima disso: pivot no lugar
left_correction = 1                       # trim por lado
right_correction = 1

# ----------------------------------------------------------------------------
# Velocidades
# ----------------------------------------------------------------------------
LINE_FOLLOW_SPEED = .5                    # PWM 60: .5 * MAX_PWM (120)
LINE_LOSS_STEER_HOLD = .7                 # s — conserva a curva ao sair brevemente da imagem
RAMP_AHEAD_HOLD = 2                       # s segurando velocidade reduzida
RAMP_AHEAD_SPEED_PIVOT = .65
RAMP_AHEAD_SPEED_ARC = .4
RAMP_AHEAD_SPEED_STRAIGHT = .3

# ----------------------------------------------------------------------------
# Detecção de linha
# ----------------------------------------------------------------------------
MIN_LINE_SIZE_DEFAULT = 3000              # area minima do contorno
RAMP_SWAP_TRIGGER = 90                    # media da banda 25% superior
RAMP_SWAP_MARGIN = 30                     # melhora minima p/ trocar teto
BLACK_AVG_SIDE_MASK = 21                  # mascara lateral se imagem limpa
LINE_CROP_INITIAL = .6
LINE_CROP_NORMAL = .6
LINE_CROP_GREEN = .45                     # durante curva verde

# Mantem a linha sob o centro inferior da camera frontal. O ponto proximo tem
# prioridade, mas parte do POI original preserva antecipacao de curvas.
BOTTOM_CENTER_CONTROL = True
BOTTOM_CENTER_WEIGHT = .7
BOTTOM_CENTER_MIN_Y = .75

# Em correcoes fortes, aproxima o centro de giro da frente do robo: as rodas
# dianteiras perdem velocidade e a traseira descreve o arco. A transicao e
# proporcional ao mesmo angulo produzido pelo controle da bolinha inferior;
# nao existe deteccao ou sequencia temporizada especifica para curvas de 90°.
FRONT_ANCHORED_STEERING = True
FRONT_ANCHOR_START_ANGLE = 65
FRONT_ANCHOR_FULL_ANGLE = 120
FRONT_ANCHOR_REAR_SCALE = 1.30
# Nunca fixa completamente a frente: com blend 1.0 a camera vira o centro de
# rotacao, a linha apenas gira na imagem e nao consegue chegar a bolinha. A
# parcela restante faz a camera descrever um arco curto ate reencontrar a linha.
FRONT_ANCHOR_MAX_BLEND = .78

# Assistencia adaptativa do pivo: se o erro para de diminuir, a roda dianteira
# interna recebe uma re leve e progressiva. A ajuda desaparece assim que a
# linha volta a se aproximar do centro.
PIVOT_STALL_MIN_ANGLE = 85
PIVOT_STALL_TIME = .35
PIVOT_STALL_RAMP_TIME = .35
PIVOT_PROGRESS_PX = 8
PIVOT_BOTTOM_MIN_ERROR_PX = 45
PIVOT_FRONT_REVERSE_SCALE = .8
PIVOT_FRONT_REVERSE_MIN_PWM = 45
PIVOT_RECOVERY_SPEED = .6                 # PWM base 72 durante busca da linha
PIVOT_RECOVERY_ASSIST_START = .55
PIVOT_RECOVERY_ASSIST_RAMP = .35          # s ate chegar a 100% da ajuda
PIVOT_RECOVERY_TIMEOUT = 2.0              # seguranca contra giro indefinido
PIVOT_RECOVERY_EXIT_ANGLE = 40

# ----------------------------------------------------------------------------
# Verde
# ----------------------------------------------------------------------------
GREEN_MIN_AREA = 2500                     # area minima do marcador
GREEN_ROI_MEAN = 125                      # "lado e preto" se media > 125
GREEN_VOTE_WINDOW = .2                    # janela da media de votos
GREEN_VOTE_THRESHOLD = .1                 # |media| que arma memoria
GREEN_MARKER_MEMORY = .5                  # memoria do marcador (plano)
GREEN_APPROACH_TIME = .7                  # s — avanca reto antes do giro verde
GREEN_TURN_MIN_TIME = .2                  # s — evita encerrar o tanque no primeiro frame
GREEN_TURN_EXIT_ANGLE = 35                # graus — linha realinhada apos o giro
GREEN_REVERSE_TIME = .5
GREEN_REVERSE_SPEED = .4                  # PWM 48


# ----------------------------------------------------------------------------
# Vermelho
# ----------------------------------------------------------------------------
RED_MIN_CONTOUR = 15000                   # gatilho de frame unico
wait_time_red = 9                         # s parado no vermelho

# ----------------------------------------------------------------------------
# Gap
# ----------------------------------------------------------------------------
GAP_ENABLED = False                       # temporariamente desabilitado para testes
GAP_CORRECTION_CYCLES = 7                 # ciclos de square-up
GAP_MIN_LINE_SIZE_ORIENT = 9000           # durante re-approach
GAP_MIN_LINE_SIZE_COMMIT = 4000           # ao entrar em gap_avoid
GAP_MIN_LINE_SIZE_RETREAT = 4500          # na retirada do gap_avoid
GAP_NOT_A_STUB_SIZE = 17000               # "linha inteira, nao toco"
GAP_BLACK_AVG_MAX = 40                    # acima disso nao e gap
# Trava contra falso gap: uma linha que ocupa varias linhas horizontais dentro
# do corredor central representa continuacao material a frente. Uma barra
# transversal isolada (o canto de um L) nao satisfaz a persistencia vertical.
GAP_AHEAD_X_MIN = .38
GAP_AHEAD_X_MAX = .62
GAP_AHEAD_Y_MAX = .72
GAP_AHEAD_ROW_FILL = .08
GAP_AHEAD_ROW_PERSISTENCE = .38
GAP_MISSING_CONFIRM_TIME = .12
GAP_REJECT_COOLDOWN = 2.0
# Campo calibrado em aproximadamente 8 cm / 448 px: 2 cm = 112 px.
# Uma borda terminal maior que isso e uma intersecao/canto, nunca um gap.
GAP_MAX_END_WIDTH_PX = 112
GAP_AVOID_TIMEOUT = .4                    # timer da travessia cega
GAP_AVOID_SPEED = .6
GAP_AVOID_RETREAT_TIME = 1.35
GAP_COMMIT_TIME = .8
GAP_COMMIT_SPEED = .7

# ----------------------------------------------------------------------------
# Movimentos temporizados usados porque o robô não possui IMU
# ----------------------------------------------------------------------------
T_SWEEP_RIGHT = .35                       # s — varredura direita na busca do gap (esq = 2×)
SWEEP_SPEED = .6                          # velocidade da varredura temporizada
LINE_SEARCH_CREEP = 1.2                   # avanco final procurando linha
T_180 = .82                               # s — teste mostrou .70 s ~= 90°; inicia perto de 105°
T_180_SPEED = .7                          # velocidade do pivot de 180°
T_180_TEST_STOP = False                   # True isola e para definitivamente apos o giro cego
T_180_SEARCH_SPEED = .4                   # procura devagar para nao atravessar a linha entre frames
T_180_SEARCH_TIMEOUT = 1.5                # s — complemento visual maximo
T_180_EXIT_BOTTOM_PX = 30                 # px — tolerancia ao redor da bolinha inferior central
T_180_CONFIRM_TIME = .10                  # s — evita parar por um frame isolado
TURN_AROUND_PREROLL = .55                 # avanca sobre o marcador
TURN_AROUND_REVERSE = .3                  # re-aquisicao da linha
TURN_AROUND_REVERSE_EXTRA = .4            # extra se line_size < 5500
TURN_AROUND_SMALL_LINE = 5500
TURN_AROUND_GREEN_COOLDOWN = 1.0          # ignora memoria residual dos dois verdes

# ----------------------------------------------------------------------------
# Loops
# ----------------------------------------------------------------------------
CONTROL_MAX_ITERATIONS = 60               # teto do loop de controle
VISION_MAX_FRAMES = 90                    # teto de processamento
VISION_READY_TIMEOUT = 15                 # s que o controle espera a visao no boot

# ----------------------------------------------------------------------------
# Cores usadas quando uma chave não existe no config.ini
# ----------------------------------------------------------------------------
BLACK_MIN_DEFAULT = [0, 0, 0]
BLACK_MAX_NORMAL_TOP_DEFAULT = [82, 83, 84]         # BGR
BLACK_MAX_NORMAL_BOTTOM_DEFAULT = [133, 133, 135]   # BGR
BLACK_MAX_RAMP_DOWN_TOP_DEFAULT = [27, 27, 26]      # BGR
GREEN_MIN_DEFAULT = [58, 95, 39]                    # HSV
GREEN_MAX_DEFAULT = [98, 255, 255]                  # HSV
RED_MIN_1_DEFAULT = [0, 100, 90]                    # HSV
RED_MAX_1_DEFAULT = [10, 255, 255]                  # HSV
RED_MIN_2_DEFAULT = [170, 100, 100]                 # HSV
RED_MAX_2_DEFAULT = [180, 255, 255]                 # HSV
