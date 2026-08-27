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
# Uma queda USB curta nao e um reinicio intencional da placa. Para rearmar o
# percurso depois de uma falha no resgate, o Arduino deve ficar ausente por
# este tempo continuo antes de reaparecer.
MISSION_ARDUINO_DESLIGAMENTO_MINIMO_S = 3.0
MAX_PWM = 120                             # teto absoluto; firmware tambem trava em 120

# ----------------------------------------------------------------------------
# Parada de segurança por obstáculo
# ----------------------------------------------------------------------------
# O ultrassônico devolve milímetros. A parada exige duas leituras próximas E
# consecutivas dentro de uma janela curta: eco distante, ausência de eco ou
# medida inválida reinicia a confirmação.
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
# O HC-SR04 não mede de forma confiável dentro dos primeiros 2 cm. Pulsos
# menores normalmente são ruído elétrico/reflexo do próprio chassi e jamais
# podem iniciar uma manobra de obstáculo.
OBSTACLE_MIN_VALID_MM = 20
OBSTACLE_MAX_VALID_MM = 4000
OBSTACLE_LATERAL_PWM = 60                  # translação com rodas omnidirecionais
OBSTACLE_LATERAL_TIME_S = 1.8              # s: desvio lateral de ida
OBSTACLE_RETURN_LATERAL_TIME_S = 1.6       # s: retorno lateral apos o reto
OBSTACLE_FORWARD_PWM = 60
# Utilitario independente para ensaios/reaproximacao de obstaculo. Nao faz
# parte da logica especial de saida do resgate.
OBSTACLE_LINE_SEARCH_PWM = 60
OBSTACLE_LINE_SEARCH_TIMEOUT_S = 4.0
OBSTACLE_LINE_CONFIRM_TIME_S = .10
OBSTACLE_FORWARD_TIME_S = 2.2              # s avançando depois do lateral
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
# Câmera: captura e processamento em 16:9, sem cortar as laterais
# ----------------------------------------------------------------------------
# Mapeamento físico atual do Pi 5: índice 0 = resgate; índice 1 = segue-linha
# no flat 2. Nunca deixar Picamera2 escolher a câmera padrão neste processo.
LINE_CAMERA_INDEX = 1
# A Camera Module 3 Wide usa um sensor 16:9. A saída já é pedida no tamanho
# exato consumido pelo algoritmo: isso preserva o mesmo campo de visão e os
# mesmos pontos de controle, mas elimina um resize OpenCV e uma cópia de
# 640×360 em todos os frames.
CAPTURE_WIDTH = 448
CAPTURE_HEIGHT = 252
# 40 FPS ainda usa o modo full-FoV 2304×1296 da IMX708, deixando margem para
# a Pi, a câmera e o conversor de alimentação sem alterar o PWM 80.
CAPTURE_FPS = 40
CAPTURE_FPS_FALLBACK = 40
camera_x = 448                            # resolução do algoritmo
camera_y = 252
# None executa um único autofocus na partida e trava a posição encontrada.
# Um número continua permitindo a calibração manual em dioptrias.
LENS_POSITION = None
# Mantém exposição e balanço de branco automáticos. Fixar estes controles sem
# calibração específica pode estourar toda a pista para branco.
LINE_CAMERA_LOCK_AUTO_CONTROLS = False
LINE_CAMERA_WARMUP_S = .45
LINE_CAMERA_EXPOSURE_VALUE = 0.0
DEBUG_SHM_NAME = "shadow_shm_cam"
DEBUG_SHM_SIZE = camera_x * camera_y * 3  # 338688 B
# A janela é apenas telemetria; a visão continua a 40 FPS. Atualizá-la menos
# vezes reduz CPU/GPU durante testes sem tocar no controle do robô.
DEBUG_DISPLAY_FPS = 15

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

# Rampa lida pelo MPU6050 do Arduino. A consulta e assincrona, portanto nao
# interrompe os comandos de movimento nem o watchdog. A correcao proporcional
# do segue-linha recebe a propria velocidade abaixo; assim, ela escala junto
# com o PWM em subida e descida.
# O MPU esta ativo; a troca automatica de PWM em rampas fica isolada enquanto
# o segue-linha e calibrado. O yaw da manobra verde independe desta opcao.
RAMPA_HABILITADA = False  # MPU ativo; ajuste automatico de rampa ainda isolado
RAMPA_CONSULTA_INTERVALO_S = .10
RAMPA_RESPOSTA_TIMEOUT_S = .20
RAMPA_SUBIDA_PWM = 120
RAMPA_DESCIDA_PWM = 50
RAMPA_SUBIDA_SPEED = RAMPA_SUBIDA_PWM / MAX_PWM
RAMPA_DESCIDA_SPEED = RAMPA_DESCIDA_PWM / MAX_PWM

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
LINE_CROP_INITIAL = .52
LINE_CROP_NORMAL = .52
LINE_CROP_GREEN = .45                     # durante curva verde

# O controle visual separa dois erros que antes eram misturados em um unico
# angulo: posicao da linha sob a camera e direcao da linha a frente. O primeiro
# mantem o robo centrado; o segundo antecipa curvas e cantos de 90 graus.
LINE_LATERAL_GAIN = .55
LINE_HEADING_GAIN = 1.0
# O termo derivativo produz uma correcao curta quando o erro muda rapidamente,
# sem acumular integral (que causaria sobrepassagem nas rodas omni).
LINE_DERIVATIVE_GAIN = .025
LINE_DERIVATIVE_FILTER = .35
LINE_DERIVATIVE_LIMIT = .25
LINE_CORRECTION_DEADBAND = .025
LINE_MAX_FRAME_AGE_S = .15

# Pure pursuit visual. O ponto futuro antecipa o rumo no contorno completo,
# enquanto o ponto inferior continua fechando o erro junto da base do robo.
# Assim um desvio que volta na imagem nao vira varios giros completos, mas a
# faixa tambem nunca deixa de ser puxada para o centro fisico do chassi.
LINE_PATH_SAMPLE_BANDS = 20
LINE_PATH_MIN_VERTICAL_SPAN_RATIO = .28
LINE_PATH_MIN_SAMPLES = 7
LINE_PATH_MAX_BAND_GAP = 2
LINE_PATH_MAX_LATERAL_JUMP_PX = camera_x * .30
# Reflexos das luzes abrem pequenos vazios dentro da faixa preta. Eles nao
# sao bifurcacoes: intervalos proximos sao reunidos e uma separacao so limita
# o ponto futuro quando for larga e persistir por varias bandas da imagem.
LINE_PATH_INTERVAL_MERGE_GAP_PX = camera_x * .055
LINE_PATH_BRANCH_MIN_GAP_PX = camera_x * .11
LINE_PATH_BRANCH_MIN_BANDS = 4
LINE_FUTURE_FRACTION = 1.0
LINE_FUTURE_SMOOTH_RADIUS = 1
LINE_FUTURE_GAIN = 1.05
LINE_FUTURE_FILTER = .45
# Quando o rumo local se desfaz antes de um ponto futuro realmente distante,
# essa diagonal nao pode armar a memoria de um canto de 90.
LINE_FUTURE_RETURN_RATIO = .65
LINE_FUTURE_RETURN_MAX_Y_RATIO = .65

# Curvas fechadas sao confirmadas em dois frames e ficam travadas para o mesmo
# lado ate que a nova reta esteja vertical e centralizada. Assim um canto nao
# volta a ser interpretado como reta no meio do giro.
LINE_CORNER_ENTRY_HEADING_DEG = 48.
LINE_CORNER_SIDE_HEADING_DEG = 35.
LINE_CORNER_TARGET_MIN = .35
LINE_CORNER_CONFIRM_FRAMES = 2
LINE_CORNER_MIN_CORRECTION = .72
LINE_CORNER_FINISH_CORRECTION = .35
LINE_CORNER_EXIT_HEADING_DEG = 18.
LINE_CORNER_EXIT_LATERAL = .18
LINE_CORNER_EXIT_TARGET = .28
LINE_CORNER_EXIT_FRAMES = 3
# Se a visao distante passa a pedir o sentido oposto, dois quadros coerentes
# cancelam a trava antes do timeout. Um quadro isolado ainda e tratado como
# ruido, preservando a firmeza dos cantos reais de 90 graus.
LINE_CORNER_RETURN_CANCEL_FRAMES = 2
LINE_CORNER_TIMEOUT_S = 1.6
LINE_CORNER_LOST_HOLD_S = .65
LINE_TRACK_LOST_HOLD_S = .25

# Mantidos somente para o angulo legado consumido por decisoes de alto nivel
# (verde, entrada e ferramentas antigas). Os motores do segue-linha normal nao
# usam mais esta mistura.
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
GREEN_MARKER_MIN_ASPECT = .5              # quadrado tolerando perspectiva
GREEN_MARKER_MAX_ASPECT = 2.0
GREEN_MARKER_MIN_RECT_FILL = .58          # rejeita reflexos/manchas alongadas
GREEN_BLACK_ROI_SCALE = .9                # alcance da busca ao redor do verde
GREEN_BLACK_MIN_RUN_RATIO = .42           # linha continua, nao media de pixels
GREEN_BLACK_MAX_GAP_RATIO = .28           # precisa estar logo junto do quadrado
GREEN_CONFIRM_FRAMES = 3                  # direcao igual em quadros consecutivos
GREEN_VOTE_WINDOW = .2                    # janela da media de votos
GREEN_VOTE_THRESHOLD = .1                 # |media| que arma memoria
GREEN_MARKER_MEMORY = .5                  # memoria do marcador (plano)
GREEN_RELEASE_MEMORY_S = .55              # neutraliza o voto residual ao sair
# A aproximacao nao e mais um avanco cego. A visao trava o ramo marcado e o
# segue-linha continua alinhando a base ate o alvo lateral chegar nesta regiao.
# O tempo existe apenas como limite de seguranca caso a geometria congele.
GREEN_APPROACH_TIME = .65                 # s — depois disso para, nunca gira
GREEN_APPROACH_BRANCH_MIN_Y_RATIO = .82   # camera no eixo frontal: gira bem abaixo
GREEN_APPROACH_MAX_CORRECTION = .32       # centraliza entrada sem antecipar giro
GREEN_APPROACH_SPEED = .5                 # base PWM 60, preserva a manobra
GREEN_BRANCH_TRANSVERSE_MIN_RUN_PX = 90   # preto horizontal continuo
GREEN_BRANCH_CENTER_TOLERANCE_PX = 45     # trecho deve tocar o eixo da camera
GREEN_BRANCH_CONFIRM_FRAMES = 2           # evita liberar tanque por um ruido
GREEN_TURN_BLIND_TIME = .30                # s — giro sem aceitar leitura da camera
GREEN_TURN_SIDE_MIN_ERROR_PX = 55          # linha alvo deve primeiro entrar pelo lado marcado
GREEN_TURN_CENTER_TOLERANCE_PX = 35        # px — ponto inferior aceito no centro da camera
GREEN_TURN_TIMEOUT = 2.0                   # s — para com seguranca se nao reencontrar o ramo
GREEN_TURN_SPEED = .5                     # base PWM 60, preserva o giro
GREEN_MPU_ENABLED = True                  # camera guia; MPU limita excesso de giro
GREEN_MPU_QUERY_INTERVAL_S = .04
GREEN_MPU_RESPONSE_TIMEOUT_S = .12
GREEN_MPU_SLOWDOWN_DEG = 70.
GREEN_MPU_SLOW_SPEED = .32
GREEN_MPU_TARGET_ARM_DEG = 40.           # evita confundir a linha de entrada
GREEN_MPU_HARD_LIMIT_DEG = 94.
GREEN_REVERSE_TIME = 0.                    # controlador continuo nao recua ao alinhar
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
# Calibracao do segue-linha: reduz 0,2 s do complemento cego do giro de 180.
T_180_BLIND_EXTRA = .10                   # s extras girando à direita sem procurar linha
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
VISION_MAX_FRAMES = 45                    # folga para não descartar captura de 40 FPS
VISION_OPENCV_THREADS = 1                 # evita picos curtos em todos os núcleos
VISION_READY_TIMEOUT = 15                 # s que o controle espera a visao no boot

# ----------------------------------------------------------------------------
# Entrada da sala de resgate por modelo ONNX — CÂMERA DE LINHA
# ----------------------------------------------------------------------------
# Este perfil pertence exclusivamente à câmera de linha (índice 1). Ele NÃO
# compartilha limites com a esfera prateada da vítima: aquela é vista pela
# câmera de resgate, de outro ângulo, com outra iluminação e outro tamanho
# aparente. Misturar os dois perfis foi explicitamente evitado.
#
# Teste temporario de entrada: nesta configuracao a entrada pelo YOLO prata
# fica completamente desligada. O controle entra no resgate pela ausencia de
# preto configurada logo abaixo. Para voltar ao modelo, ponha este valor em
# ``True`` e desligue ``ENTRY_NO_BLACK_RESCUE_TEST_ENABLED``.
ENTRY_SILVER_ENABLED = False
# `entrada.onnx` é um YOLO de uma classe, exportado em 640×640.
ENTRY_MODEL_PATH = "modelos/entrada.onnx"
# Export NCNN 416×416 do mesmo `entrada.pt`. No Pi, NCNN é mais adequado ao
# CPU ARM. O modo `auto` tenta NCNN primeiro e mantém o ONNX como contingência.
ENTRY_MODEL_BACKEND = "auto"
ENTRY_NCNN_MODEL_PATH = "modelos/entrada_416_ncnn_model"
ENTRY_MODEL_INPUT = 640
ENTRY_NCNN_MODEL_INPUT = 416
# A faixa prata pode cruzar poucos frames quando o robo esta rapido. Aceita o
# modelo a partir de 30%; a protecao real contra a rampa e o preto normal e o
# preto exclusivo da rampa alem/ao redor da caixa, no mesmo frame.
ENTRY_MODEL_MIN_CONFIDENCE = .30
# A faixa vista em velocidade pode aparecer pequena e inclinada. Ainda
# recusamos uma caixa quadrada, mas nao exigimos que ela ja ocupe a pista.
ENTRY_SILVER_MIN_WIDTH_RATIO = .05
ENTRY_SILVER_MIN_ASPECT_RATIO = 1.1
# O YOLO leva mais que um periodo de camera. Guardar esta janela curta evita
# perder os poucos frames da prata, mas o limite impede backlog indefinido.
# A faixa prata pode existir por poucos frames e a inferencia NCNN/ONNX leva
# mais que um periodo da camera. A janela de 24 preserva esse evento curto;
# cada resultado carrega a mascara preta do proprio frame, portanto nao abre
# a entrada na rampa quando a resposta chega depois.
ENTRY_MODEL_PENDING_FRAMES = 24
# Limita o runtime do modelo para não disputar todos os núcleos com a linha.
ENTRY_MODEL_THREADS = 2
# Um unico prata alinhado sem preto depois da caixa inicia o resgate logo. A
# rampa ainda e barrada antes deste voto pelos dois limiares de preto.
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
# A saída da rampa forma uma faixa preta transversal e larga; a linha que
# chega à entrada prata é estreita. Esta segunda medida usa somente a máscara
# preta mais rígida da rampa e procura uma barra larga perto da caixa do YOLO.
# Ela não acrescenta votos ou atraso à entrada verdadeira.
ENTRY_RAMP_BLACK_X_MIN = .15
ENTRY_RAMP_BLACK_X_MAX = .85
ENTRY_RAMP_BLACK_ROW_FILL = .60
ENTRY_RAMP_BLACK_MIN_ROWS_RATIO = .02
ENTRY_RAMP_BLACK_NEAR_BOX_MARGIN_RATIO = .08
# Com um voto a confirmação é imediata. Este valor só tem efeito quando a
# configuração exigir mais de um voto, para voltar à observação parada.
ENTRY_SILVER_VALIDATION_S = 0.0
# Se houver preto alem da prata, mantenha o segue-linha e bloqueie apenas uma
# nova candidatura prata por este periodo. Cada frame com preto renova o prazo.
ENTRY_BLACK_FOLLOW_TIMEOUT_S = 1.0
# ---------------------------------------------------------------------------
# Teste de entrada sem prata -- SOMENTE na missao completa
# ---------------------------------------------------------------------------
# Depois de ter seguido uma linha preta, se ela desaparecer enquanto o robo
# estiver reto por este tempo, entra no resgate. O temporizador nao conta no
# boot sem linha, em curva, em marcador ou em manobra verde.
ENTRY_NO_BLACK_RESCUE_TEST_ENABLED = True
ENTRY_NO_BLACK_RESCUE_DELAY_S = 3.0
# Ao confirmar, o controle apenas para e entrega a serial. O resgate preserva
# seu avanço reto normal de 1 s antes de iniciar os giros de busca.
# Além do modelo, o primeiro voto precisa ter a linha preta rastreada. A
# tolerância é propositalmente ampla: a faixa de prata deve poder iniciar o
# resgate mesmo com o robô levemente torto ou deslocado. O veto de preto
# (normal e de rampa) continua sendo a proteção contra uma falsa entrada.
ENTRY_LINE_MAX_ANGLE = 35
ENTRY_LINE_MAX_BOTTOM_ERROR_PX = 110
# A faixa pode cobrir o fim da linha no frame seguinte. Conserva o último
# alinhamento comprovado por este intervalo, sem aceitar uma linha antiga.
ENTRY_ALIGNMENT_HOLD_S = .70

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
# Marcador verde sob pouca luz perde saturacao e pode deslocar um pouco o
# matiz. A area/forma do triangulo ainda e validada depois desta mascara.
GREEN_MIN_DEFAULT = [15, 60, 12]                     # HSV (era H=58)
GREEN_MAX_DEFAULT = [75, 255, 255]                   # HSV (era H=98)
# Laranja ocupa o fim da banda baixa do HSV. Vermelho fica restrito aos tons
# proximos de 0 ou 180, com cor e brilho suficientes para nao pegar laranja.
RED_MIN_1_DEFAULT = [0, 125, 100]                    # HSV
RED_MAX_1_DEFAULT = [7, 255, 255]                    # HSV
RED_MIN_2_DEFAULT = [173, 125, 100]                  # HSV
RED_MAX_2_DEFAULT = [180, 255, 255]                 # HSV
