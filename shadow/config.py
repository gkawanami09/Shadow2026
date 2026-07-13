"""
config.py — every tunable constant of the Shadow2026 line-follower port.

Values marked [OE2] are copied VERBATIM from the Overengineering² Reading
Dossier, Section 5 (do not retune silently — recalibrate via tools/color_slider.py
and docs/CALIBRATION_GUIDE.md). Values marked [SHADOW] are Shadow2026 hardware
constants or IMU-replacement timings from the mission spec (§2, §3.2).

Color thresholds live in config.ini (runtime-tunable via tools/color_slider.py);
the *_DEFAULT values below are only fallbacks when config.ini is missing a key.

Note on deep geometry fractions: the POI-cascade fractions inside
vision/line.py (0.1/0.02/0.98/0.75/0.5 …) are kept inline for verbatim parity
with the dossier's quoted code; they are documented in the dossier §5 table and
in docs/CALIBRATION_GUIDE.md rather than duplicated here.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos (hardware do projeto — nada aqui vem do OE²)
# ----------------------------------------------------------------------------
SHADOW_ROOT = Path(__file__).resolve().parent
CONFIG_INI_PATH = SHADOW_ROOT / "config.ini"

# ----------------------------------------------------------------------------
# Serial / Arduino Uno  [SHADOW — mission §2.1, §2.7; firmware SPEC 01]
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
MAX_PWM = 120                             # [SHADOW §2.7] teto absoluto; firmware tambem trava em 120

# ----------------------------------------------------------------------------
# Camera  [SHADOW §2.4] — captura 640×480 RGB, algoritmo em 448×252 BGR
# ----------------------------------------------------------------------------
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 40                          # FrameDurationLimits = 1e6/40 = 25000 µs
camera_x = 448                            # [OE2] line_cam.py:20 — resolucao do algoritmo
camera_y = 252                            # [OE2] line_cam.py:21
LENS_POSITION = None                      # [SHADOW] None = foco fixo; ajuste se o modulo tiver AF
DEBUG_SHM_NAME = "shadow_shm_cam"
DEBUG_SHM_SIZE = camera_x * camera_y * 3  # 338688 B

# ----------------------------------------------------------------------------
# Lei de direcao (P-only)  [OE2 Hotspot 2]
# ----------------------------------------------------------------------------
max_turn_angle = 110                      # [OE2] control.py:16 — acima disso: pivot no lugar
left_correction = 1                       # [OE2] control.py:76 — trim por lado
right_correction = 1                      # [OE2] control.py:77

# ----------------------------------------------------------------------------
# Velocidades / get_speed  [OE2 Hotspot 2]
# ----------------------------------------------------------------------------
LINE_FOLLOW_SPEED = .5                    # PWM 60: .5 * MAX_PWM (120)
LINE_LOSS_STEER_HOLD = .7                 # s — conserva a curva ao sair brevemente da imagem
RAMP_AHEAD_HOLD = 2                       # [OE2] control.py:288 — s segurando velocidade reduzida
RAMP_AHEAD_SPEED_PIVOT = .65              # [OE2] control.py:291
RAMP_AHEAD_SPEED_ARC = .4                 # [OE2] control.py:293
RAMP_AHEAD_SPEED_STRAIGHT = .3            # [OE2] control.py:295

# ----------------------------------------------------------------------------
# Deteccao de linha  [OE2 Hotspot 1]
# ----------------------------------------------------------------------------
MIN_LINE_SIZE_DEFAULT = 3000              # [OE2] mp_manager.py:48 — area minima do contorno
RAMP_SWAP_TRIGGER = 90                    # [OE2] line_cam.py:642 — media da banda 25% superior
RAMP_SWAP_MARGIN = 30                     # [OE2] line_cam.py:649 — melhora minima p/ trocar teto
BLACK_AVG_SIDE_MASK = 21                  # [OE2] line_cam.py:680 — mascara lateral se imagem limpa
LINE_CROP_INITIAL = .6                    # [OE2] mp_manager.py:53
LINE_CROP_NORMAL = .6                    # [OE2] line_cam.py:758
LINE_CROP_GREEN = .45                     # [OE2] line_cam.py:746/752 — durante curva verde

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
FRONT_ANCHOR_REAR_SCALE = 1.15
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

SIMILARITY_CHECK_EVERY = 30               # [OE2] line_cam.py:584 — SSIM a cada 30 frames

# ----------------------------------------------------------------------------
# Verde  [OE2 Hotspot 4]
# ----------------------------------------------------------------------------
GREEN_MIN_AREA = 2500                     # [OE2] line_cam.py:120 — area minima do marcador
GREEN_ROI_MEAN = 125                      # [OE2] line_cam.py:150-173 — "lado e preto" se media > 125
GREEN_VOTE_WINDOW = .2                    # [OE2] line_cam.py:733 — janela da media de votos
GREEN_VOTE_THRESHOLD = .1                 # [OE2] line_cam.py:735/739 — |media| que arma memoria
GREEN_MARKER_MEMORY = .5                  # [OE2] line_cam.py:736/740 — memoria do marcador (plano)
GREEN_APPROACH_TIME = .7                  # s — avanca reto antes do giro verde
GREEN_TURN_MIN_TIME = .2                  # s — evita encerrar o tanque no primeiro frame
GREEN_TURN_EXIT_ANGLE = 35                # graus — linha realinhada apos o giro
GREEN_REVERSE_TIME = .5
GREEN_REVERSE_SPEED = .4                  # PWM 48

# Canto preto de 90 graus. Os limites sao conservadores: so substitui o
# controle da bolinha depois de varios frames com dois bracos retos, longos e
# perpendiculares formando um unico vertice.
CORNER_90_ENABLED = True
# A geometria ja possui varias travas independentes. Publica no primeiro frame
# valido e segura o resultado tempo suficiente para o processo de controle
# recebe-lo antes que o proprio movimento mude a imagem.
CORNER_90_CONFIRM_FRAMES = 1
CORNER_90_RELEASE_FRAMES = 10
CORNER_90_APPROACH_TIME = .9
CORNER_90_TURN_MIN_TIME = .2
CORNER_90_EXIT_ANGLE = 35
CORNER_90_REVERSE_TIME = .5
CORNER_90_REVERSE_SPEED = .4              # PWM 48
CORNER_90_HOUGH_THRESHOLD = 30
CORNER_90_MIN_ARM_PX = 55
CORNER_90_MAX_AXIS_ERROR_DEG = 20
CORNER_90_VERTEX_TOLERANCE_PX = 30
CORNER_90_CLUSTER_TOLERANCE_PX = 55

# ----------------------------------------------------------------------------
# Vermelho  [OE2 Hotspot 5]
# ----------------------------------------------------------------------------
RED_MIN_CONTOUR = 15000                   # [OE2] line_cam.py:96 — gatilho de frame unico
wait_time_red = 9                         # [OE2] control.py:14 — s parado no vermelho

# ----------------------------------------------------------------------------
# Gap  [OE2 Hotspot 3]
# ----------------------------------------------------------------------------
GAP_ENABLED = False                       # temporariamente desabilitado para testes
GAP_CORRECTION_CYCLES = 7                 # [OE2] control.py:520 — ciclos de square-up
GAP_MIN_LINE_SIZE_ORIENT = 9000           # [OE2] control.py:563 — durante re-approach
GAP_MIN_LINE_SIZE_COMMIT = 4000           # [OE2] control.py:640 — ao entrar em gap_avoid
GAP_MIN_LINE_SIZE_RETREAT = 4500          # [OE2] control.py:1964 — na retirada do gap_avoid
GAP_NOT_A_STUB_SIZE = 17000               # [OE2] control.py:483/570/615 — "linha inteira, nao toco"
GAP_BLACK_AVG_MAX = 40                    # [OE2] control.py:512/645 — acima disso nao e gap
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
GAP_AVOID_TIMEOUT = .4                    # [OE2] control.py:1942 — timer da travessia cega
GAP_AVOID_SPEED = .6                      # [OE2] control.py:1961
GAP_AVOID_RETREAT_TIME = 1.35             # [OE2] control.py:1966
GAP_COMMIT_TIME = .8                      # [OE2] control.py:642
GAP_COMMIT_SPEED = .7                     # [OE2] control.py:641

# ----------------------------------------------------------------------------
# Substituicoes de IMU  [SHADOW — mission §3.2; marcadas # IMU_REPLACEMENT no codigo]
# ----------------------------------------------------------------------------
T_SWEEP_RIGHT = .35                       # s — varredura direita na busca do gap (esq = 2×)
SWEEP_SPEED = .6                          # velocidade da varredura temporizada
LINE_SEARCH_CREEP = 1.2                   # [OE2] control.py:670 — avanco final procurando linha
T_180 = .9                                # s — pivot temporizado do giro de 180° (calibrar!)
T_180_SPEED = .7                          # velocidade do pivot de 180°
TURN_AROUND_PREROLL = .55                 # [OE2] control.py:714 — avanca sobre o marcador
TURN_AROUND_REVERSE = .3                  # [OE2] control.py:719 — re-aquisicao da linha
TURN_AROUND_REVERSE_EXTRA = .4            # [OE2] control.py:723 — extra se line_size < 5500
TURN_AROUND_SMALL_LINE = 5500             # [OE2] control.py:722

# ----------------------------------------------------------------------------
# Anti-travamento  [OE2 Hotspot 2]
# ----------------------------------------------------------------------------
STUCK_SIM_THRESHOLD = .88                 # [OE2] control.py:1927 — SSIM medio p/ "preso"
STUCK_SIM_WINDOW = 15                     # [OE2] control.py:1927 — s da janela
STUCK_COOLDOWN = 4                        # [OE2] control.py:1929 — s entre recuperacoes (plano)

# ----------------------------------------------------------------------------
# Loops
# ----------------------------------------------------------------------------
CONTROL_MAX_ITERATIONS = 60               # [OE2] control.py:1766 — teto do loop de controle
VISION_MAX_FRAMES = 90                    # [OE2] line_cam.py:568 — teto de processamento
VISION_READY_TIMEOUT = 15                 # [SHADOW] s que o controle espera a visao no boot

# ----------------------------------------------------------------------------
# Cores — fallback caso config.ini nao exista (valores [OE2] config.ini:9-29)
# ----------------------------------------------------------------------------
BLACK_MIN_DEFAULT = [0, 0, 0]                       # [OE2] line_cam.py:26
BLACK_MAX_NORMAL_TOP_DEFAULT = [82, 83, 84]         # BGR
BLACK_MAX_NORMAL_BOTTOM_DEFAULT = [133, 133, 135]   # BGR
BLACK_MAX_RAMP_DOWN_TOP_DEFAULT = [27, 27, 26]      # BGR
GREEN_MIN_DEFAULT = [58, 95, 39]                    # HSV
GREEN_MAX_DEFAULT = [98, 255, 255]                  # HSV
RED_MIN_1_DEFAULT = [0, 100, 90]                    # HSV
RED_MAX_1_DEFAULT = [10, 255, 255]                  # HSV
RED_MIN_2_DEFAULT = [170, 100, 100]                 # HSV
RED_MAX_2_DEFAULT = [180, 255, 255]                 # HSV
