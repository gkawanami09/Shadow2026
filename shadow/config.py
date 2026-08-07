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
OBSTACLE_FORWARD_TIME_S = 2.0              # s avançando depois do lateral
OBSTACLE_TANK_RIGHT_PWM = 60
OBSTACLE_TANK_RIGHT_TIME_S = 1.3            # s girando tanque à direita
# Depois do desvio, aproxima até a linha realmente chegar perto da câmera.
OBSTACLE_LINE_SEARCH_PWM = 60
OBSTACLE_LINE_SEARCH_TIMEOUT_S = 4.0
OBSTACLE_LINE_NEAR_BOTTOM_RATIO = .85
OBSTACLE_LINE_CONFIRM_TIME_S = .10
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
RAMP_AHEAD_HOLD = 2                       # s segurando velocidade reduzida
RAMP_AHEAD_SPEED_PIVOT = .65
RAMP_AHEAD_SPEED_ARC = .4
RAMP_AHEAD_SPEED_STRAIGHT = .3

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
# O marcador verde so vale PERTO, na parte de baixo do quadro. Um contorno so
# conta se a BASE dele alcancar esta fracao da altura da imagem.
#
# Motivo medido, nao preferencia de estilo: a fita PRATA da entrada reflete o
# ambiente e cai dentro da faixa HSV do verde (H 30-60). Nas capturas de
# `captures/linha_prata` ela vira um contorno "verde" de ~7000-10000 px, bem
# acima de GREEN_MIN_AREA. Com o robo se aproximando, essa mancha aparece na
# parte de CIMA do quadro, o robo a trata como marcador, dispara o giro verde
# e nunca chega a confirmar a entrada da sala — foi exatamente o que os
# prints mostraram.
#
# O marcador de verdade fica no chao colado na interseccao: quando o robo
# esta perto o bastante para agir, ele ja desceu para a metade de baixo. Um
# marcador ainda alto no quadro esta longe demais para virar manobra, e
# ignora-lo nao perde nada — ele volta a ser visto alguns frames depois, ja
# na regiao valida. O veto de "marcador baixo demais" (base > 95% da altura,
# em `determine_turn_direction`) continua valendo do outro lado.
GREEN_ROI_TOP = .55
GREEN_VOTE_WINDOW = .2                    # janela da media de votos
GREEN_VOTE_THRESHOLD = .1                 # |media| que arma memoria
GREEN_MARKER_MEMORY = .5                  # memoria do marcador (plano)
GREEN_APPROACH_TIME = .7                  # s — avanca reto antes do giro verde
GREEN_APPROACH_SPEED = .5                 # base PWM 60, preserva a manobra
GREEN_TURN_MIN_TIME = .2                  # s — evita encerrar o tanque no primeiro frame
GREEN_TURN_EXIT_ANGLE = 35                # graus — linha realinhada apos o giro
GREEN_TURN_SPEED = .5                     # base PWM 60, preserva o giro
GREEN_REVERSE_TIME = .5
GREEN_REVERSE_SPEED = .4                  # PWM 48


# ----------------------------------------------------------------------------
# Vermelho
# ----------------------------------------------------------------------------
RED_MIN_CONTOUR = 15000                   # candidato vermelho em cada frame
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
# Faixa PRATA de entrada da sala de resgate — CÂMERA DE LINHA
# ----------------------------------------------------------------------------
# Este perfil pertence exclusivamente à câmera de linha (índice 1). Ele NÃO
# compartilha limites com a esfera prateada da vítima: aquela é vista pela
# câmera de resgate, de outro ângulo, com outra iluminação e outro tamanho
# aparente. Misturar os dois perfis foi explicitamente evitado.
#
# Os limiares abaixo são um ponto de partida conservador. Eles precisam ser
# medidos na arena real com `tools/calibrar_cores.py` (grupo 7). Nenhum deles
# foi validado com a fita de verdade sob a iluminação da competição.
ENTRY_SILVER_ENABLED = True
# HSV do OpenCV (H 0..180). Prata NÃO é "claro": medido nas capturas reais da
# câmera de linha (`captures/linha_prata`), a mesma fita aparece com V mediano
# entre 62 e 190 e S até ~87, enquanto o PISO branco fica em V 200-255 e S<30.
# Vista de raso, a fita reflete o ambiente e sai mais ESCURA e mais tingida que
# o piso. A janela antiga (V≥140, S≤70) era, na prática, uma máscara de piso:
# a fita ficava de fora e o detector morria em "sem_linha_cheia"/"fina" — que é
# exatamente o que os prints do robô mostram.
#
# Por isso a janela aqui é deliberadamente larga: ela só exclui o que é PRETO
# (linha e intersecção) e o que é COLORIDO (marcador verde/vermelho). Quem
# separa fita de piso é a TEXTURA da luz (ENTRY_SILVER_MIN_LOCAL_RANGE, ver
# abaixo), somada à geometria transversal, ao contraste e ao veto de escuro.
ENTRY_SILVER_MIN_DEFAULT = [0, 0, 55]
ENTRY_SILVER_MAX_DEFAULT = [180, 110, 255]

# A fita só é procurada na parte de baixo da imagem: acima disso aparecem
# público, sapatos, cadeiras e o resto do ginásio.
#
# 0.55 era restritivo DEMAIS e é a causa principal da detecção intermitente.
# Medido em `captures/linha_prata/...140318`: com o robô se aproximando, a
# fita ocupa de 19% a 46% da altura do quadro — inteiramente ACIMA do corte
# antigo. O detector nem chegava a ver a fita; via só o piso adiante dela.
#
# Varredura do corte contra 30 cenas positivas (as duas capturas reais mais
# borrão, escurecimento, ruído e distância) e 18 negativas: 0.55 acertava
# 19/30, 0.45 acerta 25/30 sem NENHUM falso positivo. Abaixo de 0.30 a
# plateia entra no quadro e o primeiro falso positivo aparece — por isso o
# corte para em 0.45, com margem folgada até esse ponto.
ENTRY_SILVER_ROI_TOP = .45
ENTRY_SILVER_ROI_BOTTOM = 1.0

# Forma. A fita tem ~250 mm e o campo inferior da câmera é ~80 mm: ela
# atravessa a imagem inteira. Exigir isso elimina de uma vez parafuso, reflexo
# de LED, fita brilhante pequena e a própria vítima prateada.
ENTRY_SILVER_MIN_ROW_FILL = .45
ENTRY_SILVER_MIN_SPAN_RATIO = .70
ENTRY_SILVER_MAX_SPAN_RATIO = 1.0
ENTRY_SILVER_MIN_THICKNESS_RATIO = .04
# Teto de espessura: é ele que separa a fita de um PISO BRANCO inteiro, e
# junto com o piso mínimo de largura elimina QUALQUER círculo. Para um disco
# de raio r as linhas com preenchimento ≥ .45 têm espessura 2·√(r²−(.225·W)²);
# atingir largura ≥ .60·W exige r ≥ .30·W, o que já força espessura ≥ .397·W.
# Em 448×252 isso são ~178 px, muito acima do teto abaixo. Ou seja: nenhuma
# esfera passa neste filtro, por maior que esteja no quadro.
ENTRY_SILVER_MAX_THICKNESS_RATIO = .30
# A fita real e texturizada/vazada pelo reflexo. 0.50 aceita as duas capturas
# reais proximas sem fazer nenhuma das curvas/faixas pretas passar pelos
# demais filtros de geometria e aparencia.
ENTRY_SILVER_MIN_FILL_RATIO = .50
ENTRY_SILVER_MIN_ASPECT = 3.5
# Com o robô colado na fita ela sobe acima do corte da ROI. Medido na captura
# `linha_prata/...140307`: a fita ocupa de 24% a 71% da altura e era rejeitada
# em "cortada_no_topo" — o veto matava a detecção verdadeira justamente no
# frame mais fácil de todos. A faixa PRETA de saída mantém o veto ligado; lá
# ele foi calibrado contra seis falsas soleiras. Aqui, quem cobre o risco é o
# teto de espessura, o veto de escuro e o contraste com a vizinhança. Uma
# faixa que toca o topo E a base da ROI continua rejeitada nos dois casos.
ENTRY_SILVER_ALLOW_TOP_TOUCH = True

# Separação por REFLEXO, aplicada na máscara antes da geometria.
#
# Medido na arena real: o piso cinza e a fita prata chegam ao mesmo brilho
# (V≈216-226) e à mesma neutralidade (S≈20-24). Nesse caso HSV sozinho não
# separa os dois, a máscara engole o piso inteiro e o candidato morre em
# "espessa" antes de qualquer teste de aparência.
#
# O que continua diferente é a TEXTURA da luz: metal amassado/refletivo
# concentra brilho em pontos e tem variação local alta; piso fosco é
# uniforme, por mais claro que seja. A faixa local (máx − mín numa janela
# pequena) mede exatamente isso e custa duas operações de morfologia.
#
# Ajuste `ENTRY_SILVER_MIN_LOCAL_RANGE` no calibrador (grupo 7): suba até o
# piso sair da máscara e só a fita continuar. Zero desliga o filtro.
ENTRY_SILVER_LOCAL_WINDOW_PX = 7
ENTRY_SILVER_MIN_LOCAL_RANGE = 18
# A fita real perde parte do reflexo quando chega muito perto da câmera. Se a
# máscara principal não formar uma faixa, o detector tenta uma segunda vez
# com este limite, mas exige confiança maior depois. O piso liso continua sem
# textura suficiente e não passa pela geometria transversal.
ENTRY_SILVER_FALLBACK_LOCAL_RANGE = 12
ENTRY_SILVER_FALLBACK_MIN_CONFIDENCE = .60

# Faixa INCLINADA — o robô chegando torto na entrada.
#
# A busca da faixa é feita por linhas horizontais: conta-se quanto de cada
# linha da imagem está na máscara. Uma faixa inclinada não preenche linha
# nenhuma — ela cruza muitas, com um pedaço em cada. Medido girando as duas
# capturas reais: até ~9° tudo passa; a partir de ~12° o detector cai em
# "sem_linha_cheia"/"espessa" mesmo com a fita inteira e nítida no quadro.
# É a mesma coisa que o robô vê quando chega de esguelha na soleira.
#
# A correção não mexe em nenhum limiar: quando a busca normal falha, a imagem
# é girada de volta ao horizontal e a MESMA busca roda outra vez. Todos os
# testes de forma, cor, reflexo, contraste e escuro rodam iguais — só que num
# quadro endireitado.
#
# Estimar o ângulo a partir da máscara foi tentado e descartado: com a faixa
# atravessando o quadro inteiro ela fica cortada nas laterais, e tanto o
# `minAreaRect` quanto os momentos da imagem devolvem quase zero justamente
# nos casos que mais precisam de correção (medido: 12° reais viravam 3.7°).
# Em vez de estimar, varre-se uma escada curta de ângulos e deixa-se o próprio
# detector dizer qual funcionou — não existe erro de estimativa possível.
#
# O custo é controlado sondando primeiro só a MÁSCARA (girar 1 canal + somar
# linhas). O quadro inteiro só é girado no ângulo em que a geometria fechou.
ENTRY_SILVER_DESKEW_ENABLED = True
# Passo e alcance da escada: ±8, ±16, ±24, ±32 graus. O passo é menor que a
# tolerância natural do detector (~9°), então nenhuma inclinação fica num vão
# entre dois degraus.
ENTRY_SILVER_TILT_STEP_DEG = 8.
ENTRY_SILVER_MAX_TILT_DEG = 32.
# Como no fallback de reflexo: o caminho girado é mais permissivo por
# construção, então paga com confiança maior.
ENTRY_SILVER_DESKEW_MIN_CONFIDENCE = .55

# Aparência. Neutralidade + assinatura reflexiva. O papel branco fosco é
# neutro mas quase não tem faixa dinâmica nem brilho especular concentrado.
# O teto de saturação acompanha a janela HSV: a fita real chegou a S≈87 nas
# capturas, e um marcador verde/vermelho fica bem acima de 110.
ENTRY_SILVER_MAX_SATURATION = 110.
# Veto de ESCURO — a contrapartida da janela HSV larga. Mede a fração de
# pixels realmente pretos DENTRO da caixa da faixa candidata, na imagem crua
# (não na máscara). Uma intersecção ou uma faixa preta transversal produz uma
# caixa dominada por preto: a máscara pega só a auréola clara da borda, mas a
# caixa continua majoritariamente escura e o candidato cai aqui. A fita prata,
# por mais escura que fique de raso, nunca é preta nesse grau.
# "Preto" é RELATIVO ao brilho da cena, não um número fixo. Um limiar
# absoluto foi tentado e reprovou a fita verdadeira assim que a luz caiu: sob
# iluminação fraca o quadro inteiro escurece, a fita boa desce junto e passa a
# ser contada como preta. O que não muda com a luz é a RAZÃO — a linha preta
# reflete uma fração pequena do que o piso reflete, sob qualquer lâmpada.
#
# O nível claro é o percentil 75 do brilho na ROI (o piso domina a ROI mesmo
# com a fita grande no quadro). Margem medida sobre as capturas reais, em
# giros de 0 a 32° e com a cena escurecida até a metade: a fração de escuro
# dentro da caixa da fita fica entre 0.00 e 0.06; a faixa PRETA salpicada de
# reflexo — o negativo mais parecido que existe — fica entre 0.18 e 0.42.
ENTRY_SILVER_DARK_V_RATIO = .33
# Grades de segurança para cena estourada ou quase apagada, onde o percentil
# perde sentido.
ENTRY_SILVER_DARK_V_MIN = 25
ENTRY_SILVER_DARK_V_MAX = 90
ENTRY_SILVER_MAX_DARK_FRACTION = .12
ENTRY_SILVER_MIN_DYNAMIC_RANGE = 26.
ENTRY_SILVER_HIGHLIGHT_V = 205
ENTRY_SILVER_MIN_HIGHLIGHT_FRACTION = .02
# Contraste contra a vizinhança imediata (acima e abaixo da faixa). O sinal
# pode ser de qualquer polaridade: dependendo do ângulo, a fita reflexiva fica
# mais clara OU mais escura que o piso. O que não pode é ser igual ao piso.
ENTRY_SILVER_SURROUND_MARGIN_RATIO = .06
ENTRY_SILVER_MIN_SURROUND_CONTRAST = 12.
ENTRY_SILVER_MIN_CONFIDENCE = .45

# Evidência de contexto: a linha preta termina antes da entrada. Exigir isso
# impede que um brilho sobre a linha, com a linha continuando à frente, seja
# lido como entrada da sala.
ENTRY_SILVER_REQUIRE_LINE_END = True

# Confirmação temporal rápida: dois frames distintos ainda eliminam um
# reflexo isolado, mas não deixam o robô preso diante da entrada verdadeira.
#
# A JANELA foi de 3 para 5 sem mexer nos votos. A evidência exigida é a mesma
# — duas detecções completas em frames distintos —, o que muda é quantas
# falhas cabem no meio: 1 antes, 3 agora. Com detecção intermitente (o robô
# em movimento borra alguns frames), 2-de-3 exige quase acerto seguido;
# 2-de-5 tolera a alternância. Um reflexo isolado continua incapaz de
# confirmar sozinho, que é a garantia que importa aqui.
ENTRY_SILVER_VOTES_NEEDED = 2
ENTRY_SILVER_VOTE_WINDOW = 5
# Evidência fraca ainda precisa repetir em dois frames. Ela aceita somente
# uma faixa transversal que falhou por estar distante/fina ou por confiança
# visual baixa; piso, esfera e reflexo pontual continuam fora.
ENTRY_SILVER_WEAK_VOTES_NEEDED = 2
ENTRY_SILVER_WEAK_VOTE_WINDOW = 3
ENTRY_SILVER_MAX_AGE_S = .35
# Depois de sair da sala o robô volta a ver prata; o cooldown impede
# reentrada imediata na mesma faixa.
ENTRY_SILVER_COOLDOWN_S = 8.

# Entrada na sala depois da confirmação. O tempo NÃO é a única evidência: o
# avanço termina quando a faixa deixa de ser vista (passou por baixo do robô)
# e o timeout é apenas o limite de segurança.
ENTRY_ADVANCE_SPEED = .40
ENTRY_ADVANCE_MIN_S = .60
ENTRY_ADVANCE_TIMEOUT_S = 3.5

# A fita prata aparece muito fina quando ainda está longe. Nesse instante o
# contorno preto da linha e os reflexos podem parecer uma curva ou um verde.
# Antes de aceitar qualquer correção, avance reto por um trecho curto para a
# faixa ganhar espessura e poder fechar a votação normal.
ENTRY_PRE_APPROACH_SPEED = .40
ENTRY_PRE_APPROACH_TIME_S = .50
ENTRY_PRE_APPROACH_SETTLE_S = .12
ENTRY_PRE_APPROACH_COOLDOWN_S = .80

# ----------------------------------------------------------------------------
# Cores usadas quando uma chave não existe no config.ini
# ----------------------------------------------------------------------------
BLACK_MIN_DEFAULT = [0, 0, 0]
BLACK_MAX_NORMAL_TOP_DEFAULT = [82, 83, 84]         # BGR
BLACK_MAX_NORMAL_BOTTOM_DEFAULT = [133, 133, 135]   # BGR
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
