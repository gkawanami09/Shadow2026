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
# A missao se recupera de uma queda do Arduino sem encerrar o supervisor.
MISSION_RECOVERY_DELAY_S = 1.0
MAX_PWM = 120                             # teto absoluto; firmware tambem trava em 120

# ----------------------------------------------------------------------------
# Parada de segurança por obstáculo
# ----------------------------------------------------------------------------
# O ultrassônico devolve milímetros. A parada exige duas leituras próximas
# dentro de uma janela curta: um eco isolado não consegue parar o robô.
OBSTACLE_STOP_ENABLED = True
OBSTACLE_STOP_DISTANCE_MM = 50             # 5 cm, inclusive
# Uma leitura isolada nesta faixa NÃO inicia o desvio; ela apenas bloqueia uma
# eventual aceleração adaptativa enquanto a confirmação 2-de-3 continua.
OBSTACLE_FAST_SPEED_BLOCK_MM = 100          # 10 cm
OBSTACLE_SAMPLE_INTERVAL_S = .06           # respeita o intervalo do HC-SR04
OBSTACLE_READ_TIMEOUT_S = .08              # firmware espera eco por até 30 ms
OBSTACLE_CONFIRM_READINGS = 2
OBSTACLE_HISTORY_SIZE = 3
OBSTACLE_CONFIRM_WINDOW_S = .20
OBSTACLE_MIN_VALID_MM = 1
OBSTACLE_MAX_VALID_MM = 4000
OBSTACLE_LATERAL_PWM = 60                  # translação com rodas omnidirecionais
OBSTACLE_LATERAL_TIME_S = 1.5              # s deslizando para a esquerda
OBSTACLE_FORWARD_PWM = 60
# Utilitario independente para ensaios/reaproximacao de obstaculo. Nao faz
# parte da logica especial de saida do resgate.
OBSTACLE_LINE_SEARCH_PWM = 60
OBSTACLE_LINE_SEARCH_TIMEOUT_S = 4.0
OBSTACLE_LINE_CONFIRM_TIME_S = .10
OBSTACLE_FORWARD_TIME_S = 2.0              # s avançando depois do lateral
# Ao encontrar uma linha transversal, o segue-linha recebe preferência
# temporária pelo ramo esquerdo, sem executar um giro tanque separado.
OBSTACLE_LEFT_PREFERENCE_MIN_TIME_S = .20
OBSTACLE_LEFT_PREFERENCE_MAX_TIME_S = 3.0
OBSTACLE_LEFT_PREFERENCE_MAX_ANGLE = 35
OBSTACLE_LEFT_PREFERENCE_BOTTOM_PX = 35
OBSTACLE_LEFT_PREFERENCE_MIN_SPAN_RATIO = .45
OBSTACLE_LEFT_PREFERENCE_ARM_TIME_S = .08
OBSTACLE_LEFT_PREFERENCE_CONFIRM_TIME_S = .12
OBSTACLE_RETRY_COOLDOWN_S = 1.0

# ----------------------------------------------------------------------------
# Câmera: captura 640×480 e processamento em 448×252
# ----------------------------------------------------------------------------
# Mapeamento físico atual do Pi 5: índice 0 = resgate; índice 1 = segue-linha
# no flat 2. Nunca deixar Picamera2 escolher a câmera padrão neste processo.
LINE_CAMERA_INDEX = 1
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# A OV5647 possui modo VGA mais rápido. A captura escolhe até este alvo usando
# apenas modos realmente anunciados pelo driver e volta para 40 FPS quando não
# consegue confirmar um modo mais rápido. A captura rápida continua ativa
# mesmo quando o PWM do segue-linha fica fixo.
CAPTURE_FPS = 60
CAPTURE_FPS_FALLBACK = 40
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
LINE_FOLLOW_PWM = 80
LINE_FOLLOW_SPEED = LINE_FOLLOW_PWM / MAX_PWM
LINE_LOSS_STEER_HOLD = .7                 # s — conserva a curva ao sair brevemente da imagem

# O segue-linha usa PWM 80 diretamente. O controlador adaptativo permanece no
# projeto para uma calibração futura, mas não participa dos comandos atuais.
RETA_RAPIDA_HABILITADA = False
VELOCIDADE_RETA_RAPIDA = LINE_FOLLOW_SPEED
FRAMES_PARA_RETA_RAPIDA = 6
FPS_MINIMO_RETA_RAPIDA = 50.
JANELA_FPS_RETA_RAPIDA = 6
IDADE_MAXIMA_VISAO_RAPIDA_S = .05          # 2,5 períodos no mínimo de 50 FPS
ANGULO_MAXIMO_RETA_RAPIDA = 10
VARIACAO_ANGULO_RETA_RAPIDA = 6
ERRO_INFERIOR_RETA_RAPIDA_PX = 18
VARIACAO_INFERIOR_RETA_RAPIDA_PX = 8
ALTURA_MINIMA_PONTO_INFERIOR_RAPIDA = .85
AREA_MINIMA_LINHA_RAPIDA = 4500
PASSO_VELOCIDADE_RETA_RAPIDA = .01

# ----------------------------------------------------------------------------
# Detecção de linha
# ----------------------------------------------------------------------------
MIN_LINE_SIZE_DEFAULT = 3000              # area minima do contorno
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
GREEN_APPROACH_SPEED = .5                 # base PWM 60, preserva a manobra
GREEN_TURN_BLIND_TIME = .30                # s — giro sem aceitar leitura da camera
GREEN_TURN_SIDE_MIN_ERROR_PX = 55          # linha alvo deve primeiro entrar pelo lado marcado
GREEN_TURN_CENTER_TOLERANCE_PX = 35        # px — ponto inferior aceito no centro da camera
GREEN_TURN_TIMEOUT = 2.0                   # s — para com seguranca se nao reencontrar o ramo
GREEN_TURN_SPEED = .5                     # base PWM 60, preserva o giro
GREEN_REVERSE_TIME = .5
GREEN_REVERSE_SPEED = .4                  # PWM 48
# Quando um verde confirmado indica uma curva, desloca o historico usado
# para escolher entre ramos concorrentes. Isso faz o ramo indicado vencer a
# correcao da linha que o robo vinha seguindo no mesmo frame.
GREEN_BRANCH_TRACKER_OFFSET_PX = 150


# ----------------------------------------------------------------------------
# Vermelho
# ----------------------------------------------------------------------------
RED_MIN_CONTOUR = 15000                   # area grande: aceita faixa ja proxima
# Faixa distante: pode ter pouca area, mas precisa ser comprida, transversal
# e fina. Isso antecipa a desaceleracao/confirmacao sem aceitar uma mancha.
RED_FAR_MIN_CONTOUR = 900
RED_FAR_MIN_SPAN_RATIO = .42
RED_FAR_MIN_ASPECT_RATIO = 3.0
RED_FAR_MAX_ANGLE_DEG = 30.0
RED_CONFIRM_WINDOW_FRAMES = 3
RED_CONFIRM_READINGS = 2                  # 2-de-3: rejeita um frame vermelho isolado
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
T_180_BLIND_EXTRA = .30                   # s extras girando à direita sem procurar linha
T_180_SPEED = .7                          # velocidade do pivot de 180°
T_180_TEST_STOP = False                   # True isola e para definitivamente apos o giro cego
T_180_SEARCH_SPEED = .4                   # procura devagar para nao atravessar a linha entre frames
T_180_SEARCH_TIMEOUT = 1.5                # s — complemento visual maximo
T_180_EXIT_BOTTOM_PX = 30                 # px — tolerancia ao redor da bolinha inferior central
T_180_CONFIRM_TIME = .10                  # s — evita parar por um frame isolado
TURN_AROUND_PREROLL = .55                 # avanca sobre o marcador
TURN_AROUND_REVERSE = .15                 # metade da ré após re-aquisitar a linha
TURN_AROUND_REVERSE_EXTRA = .20           # metade do extra se line_size < 5500
TURN_AROUND_SMALL_LINE = 5500
TURN_AROUND_GREEN_COOLDOWN = 1.0          # ignora memoria residual dos dois verdes
# Durante o 180 a visao nao pode confirmar a entrada. Depois de parar e dar
# a re de retomada, aguardamos este trecho de segue-linha antes de rearmar o
# modelo de prata com votos novos.
ENTRY_TURN_AROUND_REARM_S = 1.0

# ----------------------------------------------------------------------------
# Loops
# ----------------------------------------------------------------------------
CONTROL_MAX_ITERATIONS = 60               # teto do loop de controle
VISION_MAX_FRAMES = 90                    # teto de processamento
VISION_READY_TIMEOUT = 15                 # s que o controle espera a visao no boot

# ----------------------------------------------------------------------------
# Entrada da sala de resgate por modelo ONNX — CÂMERA DE LINHA
# ----------------------------------------------------------------------------
# Este perfil pertence exclusivamente à câmera de linha (índice 1). Ele NÃO
# compartilha limites com a esfera prateada da vítima: aquela é vista pela
# câmera de resgate, de outro ângulo, com outra iluminação e outro tamanho
# aparente. Misturar os dois perfis foi explicitamente evitado.
#
ENTRY_SILVER_ENABLED = True
# `entrada.onnx` é um YOLO de uma classe, exportado em 640×640.
ENTRY_MODEL_PATH = "modelos/entrada.onnx"
# Export NCNN 416×416 do mesmo `entrada.pt`. No Pi, NCNN é mais adequado ao
# CPU ARM. O modo `auto` tenta NCNN primeiro e mantém o ONNX como contingência.
ENTRY_MODEL_BACKEND = "auto"
ENTRY_NCNN_MODEL_PATH = "modelos/entrada_416_ncnn_model"
ENTRY_MODEL_INPUT = 640
ENTRY_NCNN_MODEL_INPUT = 416
# A faixa prata muda muito de brilho com a luz da pista. Aceita candidatos a
# partir de 45%; a linha alinhada e o preto depois da caixa protegem contra
# um falso resgate sem perder uma faixa atravessada em velocidade.
ENTRY_MODEL_MIN_CONFIDENCE = .45
# Limita o runtime do modelo para não disputar todos os núcleos com a linha.
ENTRY_MODEL_THREADS = 2
# Uma candidata alinhada entra imediatamente quando não há preto depois dela.
# Para voltar à observação parada, aumente os votos e a duração abaixo juntos.
ENTRY_SILVER_VOTES_NEEDED = 1
ENTRY_SILVER_VOTE_WINDOW = 3
# Contexto da prata: a imagem clara no final da rampa só é um falso
# candidato se a linha preta continuar DEPOIS dela, na direção de marcha.
# Esta é uma evidência muito mais estável do que exigir que a faixa prata
# tenha sempre a mesma textura/espessura na câmera. A entrada verdadeira
# encerra a linha, portanto não deve haver preto além da caixa.
ENTRY_REJECT_SILVER_WITH_BLACK_AHEAD = True
# Ignora alguns pixels imediatamente acima da caixa prata para nao confundir
# sua borda com a continuacao preta da rampa.
ENTRY_BLACK_AFTER_SILVER_GUARD_RATIO = .03
# Com um voto a confirmação é imediata. Este valor só tem efeito quando a
# configuração exigir mais de um voto, para voltar à observação parada.
ENTRY_SILVER_VALIDATION_S = 0.0
# Se houver preto alem da prata, mantenha o segue-linha e bloqueie apenas uma
# nova candidatura prata por este periodo. Cada frame com preto renova o prazo.
ENTRY_BLACK_FOLLOW_TIMEOUT_S = 1.0
# Além do modelo, o primeiro voto precisa ter a linha preta rastreada e
# centralizada. Isso evita parar atravessado na faixa; durante a observação a
# prata pode cobrir o fim da linha.
ENTRY_LINE_MAX_ANGLE = 18
ENTRY_LINE_MAX_BOTTOM_ERROR_PX = 55
# A faixa pode cobrir o fim da linha no frame seguinte. Conserva o último
# alinhamento comprovado por este intervalo, sem aceitar uma linha antiga.
ENTRY_ALIGNMENT_HOLD_S = .50

# ----------------------------------------------------------------------------
# Cores usadas quando uma chave não existe no config.ini
# ----------------------------------------------------------------------------
BLACK_MIN_DEFAULT = [0, 0, 0]
BLACK_MAX_NORMAL_TOP_DEFAULT = [82, 83, 84]         # BGR
BLACK_MAX_NORMAL_BOTTOM_DEFAULT = [133, 133, 135]   # BGR
# Perfil independente da parte escura da rampa. Ele nao substitui o preto
# normal que guia o robo: so e consultado como uma segunda prova de que ainda
# existe linha preta alem da candidata a prata, antes de liberar o resgate.
# Calibre pelo grupo 3 de `tools/calibrar_cores.py`.
BLACK_MAX_RAMP_DOWN_TOP_DEFAULT = [27, 27, 26]      # BGR
# Matizes MIGRADOS quando a troca R<->B da câmera de linha foi corrigida
# (ver visao/captura.py). A conversão é exata: H_correto = 120 − H_trocado.
# Antes: 58..98 na imagem trocada. Depois: 22..62 na imagem correta.
# S e V não mudam — trocar dois canais não altera máximo nem mínimo.
GREEN_MIN_DEFAULT = [22, 95, 39]                    # HSV (era H=58)
GREEN_MAX_DEFAULT = [62, 255, 255]                  # HSV (era H=98)
RED_MIN_1_DEFAULT = [0, 100, 90]                    # HSV
RED_MAX_1_DEFAULT = [10, 255, 255]                  # HSV
RED_MIN_2_DEFAULT = [170, 100, 100]                 # HSV
RED_MAX_2_DEFAULT = [180, 255, 255]                 # HSV
