"""Configuração isolada da primeira etapa do resgate do Shadow.

Este modulo nao e importado pelo segue-linha. Alterar valores aqui nao muda o
comportamento de ``shadow/main.py``.
"""

# Câmera de resgate. O visualizador de câmeras chama a câmera 0 de
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

# A soleira da arena e apenas um veto quando existe evidencia forte. Nas fotos
# reais a lente larga curva a parede/piso e portas interrompem a linha; exigir
# um modelo valido em todo frame deixava o detector completamente cego.
# A analise roda reduzida e em frames alternados para nao derrubar o FPS.
ARENA_GUARD_WORK_WIDTH = 160
ARENA_GUARD_WORK_HEIGHT = 120
ARENA_GUARD_INTERVAL_FRAMES = 2
ARENA_VETO_MIN_CONFIDENCE = 0.64
ARENA_VETO_MAX_FLOOR_SUPPORT = 0.12

# Durante a busca de bolinhas, os dois filtros cromaticos de triangulo tambem
# rodam em frames alternados. Como adquirir uma esfera exige tres resultados,
# um triangulo ainda e interceptado antes de poder obter LOCK.
MARKER_GUARD_INTERVAL_FRAMES = 2

# Melhoria de iluminação já experimentada no visualizador de câmeras.
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
# Pessoas, roupas e objetos externos aparecem acima da parede da arena. Uma
# esfera so pode entrar no detector/tracker quando o centro estiver na metade
# inferior; o raio ainda pode atravessar a linha quando a vitima estiver perto.
BALL_TARGET_MIN_CENTER_Y_RATIO = 0.50
# O circulo pode continuar valido ate a ultima linha. Uma tolerancia pequena
# deixa o lock sobreviver quando a base acabou de ser cortada, sem mover para
# cima o ponto fisico que dispara a coleta.
BALL_ROI_BOTTOM = 1.00
BALL_ROI_BOTTOM_OVERFLOW_RATIO = 0.10
BALL_MIN_RADIUS_PX = 9
BALL_MAX_RADIUS_PX = 135
BALL_MIN_CIRCULARITY = 0.56
BALL_MIN_FILL_RATIO = 0.50
BALL_MAX_ASPECT_RATIO = 1.32
# O perimetro precisa existir ao redor da proposta, e nao apenas em alguns
# pontos que por acaso formam um arco de Hough. A busca radial usa gradiente
# alinhado com a normal do circulo, divide a volta em oito setores e mede a
# dispersao da borda encontrada. O setor inferior pode faltar quando a esfera
# toca o piso ou comeca a sair do quadro; topo e laterais continuam obrigatorios.
BALL_RADIAL_SAMPLES = 72
BALL_RADIAL_SECTORS = 8
# Hough frequentemente encaixa um reflexo interno com raio cerca de 25% menor
# que o envelope real. A faixa larga permite encontrar o perimetro externo; a
# dispersao radial abaixo impede que lados de triangulos/elipses se aproveitem.
# A busca e ASSIMETRICA porque o erro do Hough e assimetrico: ele engancha em
# um reflexo/faceta INTERNA e quase nunca em algo maior que a esfera. Medido
# nas capturas reais da arena: com r proposto 38 e envelope real 67, a borda
# verdadeira esta a +76% — muito alem dos 40% originais, entao a busca nem
# alcancava o perimetro e so encontrava facetas internas do papel amassado.
BALL_RADIAL_SEARCH_BAND_RATIO = 0.40          # para dentro
BALL_RADIAL_SEARCH_OUTWARD_RATIO = 1.00       # para fora
BALL_RADIAL_MAX_STEPS = 41
BALL_RADIAL_MIN_GRADIENT = 20.0
# Fracao dos votos do pico que um raio precisa ter para disputar como
# envelope. Entre os que passam, vence o mais externo (ver a votacao em
# bola_resgate). Perto de 1.0 o comportamento volta a ser "so o pico".
BALL_RADIAL_VOTE_RATIO = 0.65
# Fracao minima do perimetro que precisa estar DENTRO da imagem para o teste
# valer alguma coisa. Uma esfera encostada na lateral do quadro continua
# sendo vitima e e julgada pela parte visivel; abaixo deste limite sobra
# pouco contorno observavel para decidir qualquer coisa.
BALL_RADIAL_MIN_MEASURABLE_FRACTION = 0.45
BALL_RADIAL_MIN_ALIGNMENT = 0.72
BALL_MIN_EDGE_SUPPORT = 0.58

# --- Rota de FOIL no teste de perimetro ------------------------------------
# Papel-aluminio amassado nao e um circulo dentro de 10% de tolerancia: sua
# silhueta e genuinamente irregular. Em vez de afrouxar a dispersao para todo
# mundo (o que deixava sombras e roupas passarem), existe uma rota separada
# que troca tolerancia de FORMA por evidencia de TEXTURA — a mesma filosofia
# ja usada no gate de proximidade (BALL_CRESCENT_FOIL_*).
#
# Medido nas capturas reais da arena, densidade de borda no interior do
# circulo: esferas amassadas 0.235 a 0.256; esferas lisas 0.000 a 0.063;
# a regiao do falso positivo (pessoa) 0.034. A separacao e limpa.
BALL_RADIAL_FOIL_MIN_SUPPORT = 0.65
BALL_RADIAL_FOIL_MIN_SECTORS = 7
BALL_RADIAL_FOIL_MAX_DISPERSION = 0.24
BALL_RADIAL_FOIL_MIN_TEXTURE = 0.15
BALL_RADIAL_FOIL_INNER_RATIO = 0.78
BALL_RADIAL_MIN_SECTOR_SUPPORT = 0.25
BALL_RADIAL_MIN_GOOD_SECTORS = 6
BALL_RADIAL_MAX_DISPERSION_RATIO = 0.10

# Hough + bordas. Os contornos de mascara cobrem a esfera preta; Hough e
# contraste local cobrem a esfera prateada/reflexiva.
BALL_MEDIAN_BLUR = 5
BALL_CANNY_SIGMA = 0.33
BALL_HOUGH_DP = 1.2
BALL_HOUGH_MIN_DIST_PX = 28
BALL_HOUGH_PARAM1 = 105
BALL_HOUGH_PARAM2 = 18
BALL_HOUGH_MIN_CONFIDENCE = 0.66
# Sem candidato nem track, o Hough pesado roda em frames alternados. Assim o
# fundo cheio de circulos falsos nao prende o Raspberry; ao primeiro candidato
# Hough, o tracker existe e os frames seguintes voltam a ser consecutivos.
BALL_HOUGH_IDLE_INTERVAL_FRAMES = 2
# Um contorno forte ja passou por circularidade, borda e aparencia. Nessa
# situacao, Hough redundante durante os 3 hits de aquisicao so adiciona atraso.
BALL_CONTOUR_FAST_CONFIDENCE = 0.68

# Aparencia no frame original (classificacao nunca usa o gamma).
BALL_BLACK_V_MAX = 105
BALL_BLACK_DARK_FRACTION_MIN = 0.52
BALL_BLACK_LOCAL_CONTRAST_MIN = 8.0
# Uma mancha escura uniforme nao vira esfera apenas por ser muito preta. O
# exterior deve ser mais claro que o interior em varios trechos independentes
# do perimetro.
BALL_APPEARANCE_SECTORS = 8
BALL_APPEARANCE_MIN_SECTOR_SAMPLES = 4
BALL_BLACK_MIN_USABLE_CONTRAST_SECTORS = 6
BALL_BLACK_MIN_CONTRAST_SECTORS = 5
BALL_SILVER_S_MAX = 88
# Referencia de bonus, nao gate: aluminio reflete a cor do iluminante e pode
# ficar ciano/verde com saturacao alta mesmo continuando metalico.
BALL_SILVER_LOW_SAT_FRACTION_MIN = 0.62
BALL_SILVER_DYNAMIC_RANGE_MIN = 20.0
BALL_SILVER_HIGHLIGHT_V = 195
BALL_SILVER_HIGHLIGHT_FRACTION_MIN = 0.015
# Madeira clara e sombras com uma linha podiam satisfazer um unico percentil
# global. O aluminio precisa distribuir textura e reflexos por varios setores.
BALL_SILVER_TEXTURE_SECTORS = 6
BALL_SILVER_MIN_TEXTURE_SECTORS = 4
BALL_SILVER_MIN_REFLECTIVE_SECTORS = 3
BALL_SILVER_SECTOR_DYNAMIC_RANGE_MIN = 18.0
BALL_SILVER_SECTOR_HIGHLIGHT_FRACTION_MIN = 0.01
# Segunda assinatura para a esfera prata lisa vista nas novas fotos. Ela pode
# ter pouco "amassado" por setor, mas conserva um contorno circular forte,
# neutralidade, um reflexo compacto e escurecimento esferico ate a borda.
BALL_SILVER_SMOOTH_LOW_SAT_FRACTION_MIN = 0.70
BALL_SILVER_SMOOTH_INNER_V_MIN = 115
BALL_SILVER_SMOOTH_DYNAMIC_RANGE_MIN = 7.0
BALL_SILVER_SMOOTH_DYNAMIC_RANGE_MAX = 45.0
BALL_SILVER_SMOOTH_HIGHLIGHT_MIN = 0.008
BALL_SILVER_SMOOTH_HIGHLIGHT_MAX = 0.32
BALL_SILVER_SMOOTH_CENTER_RIM_MIN = 5.0
BALL_SILVER_SMOOTH_QUADRANT_CONTRAST_MIN = 4.0
BALL_SILVER_SMOOTH_MIN_RADIAL_QUADRANTS = 3
BALL_SILVER_SMOOTH_MIN_REFLECTIVE_SECTORS = 1
BALL_SILVER_SMOOTH_GEOMETRY_MIN = 0.78
BALL_SILVER_BRIGHT_INNER_V_MIN = 175
BALL_SILVER_BRIGHT_DYNAMIC_RANGE_MIN = 4.0
BALL_SILVER_BRIGHT_HIGHLIGHT_MIN = 0.25
BALL_SILVER_BRIGHT_HIGHLIGHT_MAX = 0.92
BALL_SILVER_BRIGHT_CENTER_RIM_MIN = 3.0
BALL_SILVER_BRIGHT_MIN_REFLECTIVE_SECTORS = 4
BALL_SILVER_BRIGHT_GEOMETRY_MIN = 0.80
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
BALL_ACQUIRE_ASSOCIATION_RADIUS_FACTOR = 0.45
BALL_ACQUIRE_ASSOCIATION_MAX_PX = 32
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
BALL_PICKUP_FORWARD_S = 1.50
# As garras so fecham depois que a reta inteira termina. O alias e mantido
# para o sequenciador representar o intervalo entre iniciar o avanco e fechar.
BALL_PICKUP_FORWARD_LEAD_S = BALL_PICKUP_FORWARD_S
BALL_PICKUP_FORWARD_SPEED = BALL_APPROACH_SPEED_NEAR
BALL_PICKUP_GRIPPER_SETTLE_S = 0.50

# Depois de prender a esfera, o elevador sobe, aplica um pulso curto para
# descer e entao executa a liberacao correspondente a cor confirmada.
BALL_PICKUP_LIFT_POWER = 20
BALL_PICKUP_LIFT_MS = 2500
BALL_PICKUP_LIFT_GUARD_S = 0.10
BALL_PICKUP_LOWER_POWER = -20
BALL_PICKUP_LOWER_MS = 25
BALL_PICKUP_LOWER_GUARD_S = 0.05
BALL_PICKUP_RELEASE_DELTA = 50
BALL_PICKUP_WIGGLE_DELTA = 40
BALL_PICKUP_WIGGLE_REPETITIONS = 2
BALL_PICKUP_WIGGLE_STEP_S = 0.20

# Busca das proximas vitimas. O Shadow nao possui IMU: o 360 e temporizado a
# partir da calibracao existente de 0,70 s ~= 90 graus em velocidade 0,70.
# O giro tanque foi reduzido novamente para 0,22: nas imagens novas, a esfera
# atravessava o campo de visao antes de obter os tres resultados distintos.
BALL_SEARCH_TANK_ANGLE = 180
BALL_SEARCH_TANK_SPEED = 0.22
BALL_SEARCH_FULL_TURN_S = 8.93
BALL_SEARCH_BRAKE_MIN_CONFIDENCE = BALL_MIN_CONFIDENCE
BALL_SEARCH_VERIFY_TIMEOUT_S = 1.00

# --- Busca PULSADA ---------------------------------------------------------
# "Gira e observa" no lugar do giro continuo. O motivo e medido: girando sem
# parar, a esfera atravessa o campo de visao antes de acumular os tres
# resultados distintos exigidos para o lock, e os frames capturados em
# movimento saem borrados e com o autoexposure ainda corrigindo.
#
# O ciclo e PULSE_ROTATE -> BRAKE -> SETTLE -> OBSERVE -> PULSE_ROTATE.
# Somente frames capturados DEPOIS do fim do SETTLE podem confirmar.
BALL_SEARCH_PULSED = True
# Duracao ativa de cada pulso. Com BALL_SEARCH_FULL_TURN_S = 8.93 s para 360,
# 0.30 s equivalem a aproximadamente 12 graus por pulso. MEDIR no robo: este
# valor nao foi verificado fisicamente.
BALL_SEARCH_PULSE_S = 0.30
# Pausa mecanica antes de olhar: vibracao do chassi e autoexposure.
BALL_SEARCH_SETTLE_S = 0.12
# Frames novos e nitidos observados a cada parada (2 a 4).
BALL_SEARCH_OBSERVE_FRAMES = 3
# Teto de espera por esses frames; sem isso uma camera travada pararia a busca.
BALL_SEARCH_OBSERVE_TIMEOUT_S = 0.60
# Setores de cobertura do giro completo. Serve de referencia cruzada com
# BALL_SEARCH_PULSE_S: setores * pulso deve ficar proximo do 360 temporizado.
BALL_SEARCH_SECTORS = 30
# Teto global da busca, contando pulsos e pausas. Protege contra laco infinito
# quando o 360 temporizado nao fecha por escorregamento das rodas.
BALL_SEARCH_TOTAL_TIMEOUT_S = 75.0

# Transporte ate o ponto de evacuacao. O marcador correto e imutavel durante
# o ciclo: esfera prata -> verde; esfera preta -> vermelho. A navegacao usa
# apenas a camera frontal de resgate e sempre para antes de liberar a esfera.
DEPOSIT_MARKER_BY_BALL_KIND = {
    "silver": "green",
    "black": "red",
}
DEPOSIT_SEARCH_TANK_ANGLE = 180
DEPOSIT_SEARCH_TANK_SPEED = 0.22
# Mesmo chassi/calibracao do giro das bolas, compensado pela menor velocidade.
DEPOSIT_SEARCH_FULL_TURN_S = (
    BALL_SEARCH_FULL_TURN_S
    * BALL_SEARCH_TANK_SPEED
    / DEPOSIT_SEARCH_TANK_SPEED
)
DEPOSIT_SEARCH_VERIFY_TIMEOUT_S = 1.00
DEPOSIT_REACQUIRE_TIMEOUT_S = 0.60
# O transporte nunca pode comandar rodas indefinidamente. O limite global
# comeca somente depois da primeira escrita serial de movimento; o watchdog
# curto exige melhora de alinhamento, largura ou altura aparente do marcador.
DEPOSIT_MAX_ACTIVE_S = 45.0
DEPOSIT_PROGRESS_TIMEOUT_S = 6.0
DEPOSIT_PROGRESS_MIN_ERROR = 0.05
DEPOSIT_PROGRESS_MIN_WIDTH_RATIO = 0.025
DEPOSIT_PROGRESS_MIN_BOTTOM_RATIO = 0.030
DEPOSIT_ALIGN_ENTER_ERROR = 0.22
DEPOSIT_ALIGN_EXIT_ERROR = 0.12
DEPOSIT_ALIGN_ANGLE_MIN = 62
DEPOSIT_ALIGN_ANGLE_MAX = 76
DEPOSIT_ALIGN_SPEED_MIN = 0.25
DEPOSIT_ALIGN_SPEED_MAX = 0.29
DEPOSIT_APPROACH_CENTER_DEADBAND = 0.08
DEPOSIT_APPROACH_STEER_MAX_ANGLE = 45
DEPOSIT_APPROACH_SPEED_FAR = 0.30
DEPOSIT_APPROACH_SPEED_NEAR = 0.23
# Gate conservador de chegada; deve ser refinado com PNGs brutos dos dois
# triangulos na arena real. Largura e base baixa precisam ocorrer juntas.
DEPOSIT_NEAR_MIN_WIDTH_RATIO = 0.30
DEPOSIT_NEAR_MIN_BOTTOM_RATIO = 0.84
DEPOSIT_NEAR_MAX_CENTER_ERROR = 0.16
DEPOSIT_NEAR_CONFIRM_FRAMES = 3
DEPOSIT_NEAR_CONFIRM_WINDOW_S = 0.45

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

# ---------------------------------------------------------------------------
# Faixa PRETA de saida da sala — CAMERA DE RESGATE
# ---------------------------------------------------------------------------
# Este detector so pode ser consultado no estado FIND_BLACK_EXIT. Fora dele,
# a faixa preta nao existe para o robo: durante a busca de vitimas ela nao
# pode interromper nada, e a vitima preta nao pode ser lida como saida.
#
# A separacao entre faixa preta e vitima preta e GEOMETRICA, nao cromatica:
# as duas sao escuras. A vitima e compacta (proporcao ~1) e a faixa e
# alongada e transversal. O veto de proporcao abaixo e o que as distingue.
EXIT_BLACK_ENABLED = True
# HSV: qualquer matiz/saturacao, apenas escuro.
EXIT_BLACK_HSV_MIN = (0, 0, 0)
EXIT_BLACK_HSV_MAX = (180, 255, 70)

EXIT_BLACK_ROI_TOP = 0.50
EXIT_BLACK_ROI_BOTTOM = 1.00

EXIT_BLACK_MIN_ROW_FILL = 0.45
EXIT_BLACK_MIN_SPAN_RATIO = 0.60
EXIT_BLACK_MAX_SPAN_RATIO = 1.00
EXIT_BLACK_MIN_THICKNESS_RATIO = 0.03
# Teto que exclui a esfera preta por construcao: para um disco de raio r, as
# linhas com preenchimento >= 0.45 formam uma espessura 2*sqrt(r^2-(0.225W)^2),
# e alcancar largura >= 0.60W exige r >= 0.30W, o que ja impoe espessura
# >= 0.397W. Em 640x480 isso sao ~254 px (0.53 da altura), muito acima deste
# teto. Nenhum raio de esfera satisfaz largura e espessura ao mesmo tempo.
EXIT_BLACK_MAX_THICKNESS_RATIO = 0.30
EXIT_BLACK_MIN_FILL_RATIO = 0.55
# Mais exigente que a entrada: aqui existe uma esfera preta na mesma arena.
EXIT_BLACK_MIN_ASPECT = 4.0

EXIT_BLACK_MAX_INSIDE_VALUE = 80.0
# Contraste COM SINAL: o piso ao redor precisa ser mais claro que a faixa.
# Uma sombra grande sobre piso escuro nao satisfaz isso.
EXIT_BLACK_SURROUND_MARGIN_RATIO = 0.06
EXIT_BLACK_MIN_SURROUND_CONTRAST = 25.0
EXIT_BLACK_MIN_CONFIDENCE = 0.55

EXIT_BLACK_VOTES_NEEDED = 3
EXIT_BLACK_VOTE_WINDOW = 5
EXIT_BLACK_COOLDOWN_S = 0.0

# Travessia da soleira de saida. Igual a entrada: o tempo e apenas o limite
# de seguranca; o fim normal e a faixa deixar de ser vista.
EXIT_ADVANCE_SPEED = 0.35
EXIT_ADVANCE_MIN_S = 0.60
EXIT_ADVANCE_TIMEOUT_S = 3.5
# Giro pulsado de procura da saida quando nenhuma faixa esta no campo.
EXIT_SEARCH_TIMEOUT_S = 60.0
EXIT_SEARCH_PULSE_S = BALL_SEARCH_PULSE_S
EXIT_SEARCH_SETTLE_S = BALL_SEARCH_SETTLE_S
EXIT_SEARCH_OBSERVE_TIMEOUT_S = BALL_SEARCH_OBSERVE_TIMEOUT_S
EXIT_SEARCH_TANK_ANGLE = BALL_SEARCH_TANK_ANGLE
EXIT_SEARCH_TANK_SPEED = BALL_SEARCH_TANK_SPEED

# Alinhamento com a soleira antes de atravessar. Arco suave, nunca pivo.
EXIT_ALIGN_MAX_CENTER_ERROR = 0.12
EXIT_ALIGN_ANGLE = 55
EXIT_ALIGN_SPEED = 0.26

# Mapeamento final dos DOIS triangulos, so para diagnostico e para provar que
# a sala foi compreendida. Nenhum deles comanda o robo nesta fase.
FINAL_TRIANGLE_MAP_FRAMES = 6
FINAL_TRIANGLE_MAP_TIMEOUT_S = 4.0
# Cores do overlay em BGR do OpenCV. Verde=(0,255,0), Vermelho=(0,0,255).
# Trocar estas duas constantes inverte o diagnostico da equipe inteira; existe
# um teste dedicado para elas em tests/test_triangulos_finais.py.
FINAL_TRIANGLE_OVERLAY_BGR = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
}

# ---------------------------------------------------------------------------
# Marcadores triangulares de deposito
# ---------------------------------------------------------------------------
# Esta calibracao pertence exclusivamente a camera frontal de resgate. Os
# limites do segue-linha usam outro sensor, outra orientacao e outra iluminacao.
MARKER_BASE_WIDTH = 640
MARKER_BASE_HEIGHT = 480

# O codigo de referencia descarta aproximadamente os 45% superiores da camera
# da zona. Aqui o mesmo corte impede roupa/cadeiras acima da parede de virar
# destino; o contorno pode tocar a linha, mas seu centro util fica na arena.
MARKER_ROI_TOP = 0.45

# HSV do OpenCV (H em 0..180). Vermelho cruza a origem e, por isso, usa duas
# bandas. O contraste cromatico local abaixo continua obrigatorio: o HSV
# sozinho nao aceita um banho uniforme de luz ciano/verde.
MARKER_GREEN_HSV_MIN = (45, 80, 40)
MARKER_GREEN_HSV_MAX = (95, 255, 255)
MARKER_RED_HSV_MIN_1 = (0, 90, 50)
MARKER_RED_HSV_MAX_1 = (12, 255, 255)
MARKER_RED_HSV_MIN_2 = (168, 90, 50)
MARKER_RED_HSV_MAX_2 = (180, 255, 255)

# Limpeza da mascara. Os tamanhos sao definidos na base 640x480 e escalados
# automaticamente para o detector continuar equivalente em 320x240.
MARKER_MORPH_OPEN_PX = 3
MARKER_MORPH_CLOSE_PX = 5

# Geometria do triangulo. A razao principal e area(hull) dividida pela area do
# menor triangulo que envolve o hull: triangulos se aproximam de 1; circulos,
# retangulos e manchas difusas ficam muito abaixo.
MARKER_MIN_AREA_RATIO = 0.0015
MARKER_MAX_AREA_RATIO = 0.55
MARKER_MIN_SIDE_PX = 9

# Um blob que encosta na borda LATERAL do quadro esta INCOMPLETO: parte dele
# ficou fora da imagem e sua forma nao pode ser julgada. Medido nas capturas
# reais: com a camera nao mirada no triangulo, so a ponta dele aparece e a
# "triangularidade" desse pedaco (0.577) nao descreve triangulo nenhum.
#
# Duas situacoes bem diferentes produzem um blob encostado na borda:
#
#   fragmento  — pequeno, o robo ainda nao mirou. Rejeitar e o certo: o giro
#                de procura continua ate o marcador entrar inteiro no quadro.
#   chegada    — grande, o marcador ja ocupa o quadro porque o robo esta em
#                cima dele. Aqui julgar forma nao faz sentido e rejeitar seria
#                perder o alvo exatamente no momento de depositar.
#
# Medido: fragmento = 0.050 do quadro; chegada = 0.256. O corte separa os dois
# com folga. Na rota de chegada, forma e ignorada mas a exigencia cromatica
# (MARKER_MIN_INSIDE_CHROMA e o contraste com o anel) continua valendo.
MARKER_EDGE_MARGIN_PX = 2
MARKER_NEAR_AREA_RATIO = 0.15
# NAO afrouxar estes limites sem resolver antes o problema abaixo.
#
# Medicao nas capturas reais da arena (visao/marcador_resgate no pipeline):
#
#   marcador a distancia de navegacao  triangularidade 0.577  proporcao 1.03
#   marcador de perto                  triangularidade 0.623  proporcao 3.83
#   cadeira vermelha do laboratorio    triangularidade 0.677  proporcao 3.80
#   circulo perfeito                   triangularidade 0.605
#   quadrado perfeito                  triangularidade 0.500
#
# Ou seja: nesta perspectiva quase rente ao piso, a cadeira e MAIS triangular
# que o marcador, e o marcador cai entre quadrado e circulo. Nao existe
# limiar de triangularidade que aceite o marcador e rejeite um circulo — os
# testes test_colored_circles_are_not_triangles provaram isso na pratica.
#
# Quem separa de verdade e a CROMATICIDADE do blob: marcador 124-148 contra
# cadeira 63-79. Ver MARKER_MIN_INSIDE_CHROMA acima.
MARKER_MAX_ASPECT_RATIO = 3.0
MARKER_MIN_SOLIDITY = 0.82
MARKER_MIN_MASK_FILL = 0.78
MARKER_APPROX_EPSILON_RATIO = 0.055
MARKER_MAX_APPROX_VERTICES = 5
MARKER_MIN_TRIANGULARITY = 0.78

# Aparencia local. Para verde mede G-max(R,B); para vermelho mede
# R-max(G,B). O anel ao redor do contorno precisa ser cromaticamente mais
# neutro que o interior, rejeitando iluminacao colorida uniforme.
MARKER_RING_WIDTH_PX = 12
MARKER_RING_MIN_PIXELS = 24
# Dois limiares cromaticos distintos, que antes eram a mesma constante:
#
# MARKER_MASK_MIN_CHROMA age POR PIXEL, antes de achar contornos. Subi-lo
# corroi a borda do blob (onde a cromaticidade cai naturalmente) e MUDA A
# FORMA do candidato — foi assim que subir o limiar chegou a fazer o detector
# PERDER o marcador real. Por isso ele fica baixo: serve so para tirar fundo.
#
# MARKER_MIN_INSIDE_CHROMA age no BLOB inteiro, pela mediana interna, depois
# de a forma estar definida. Ele pode ser rigoroso sem deformar nada.
# Medido nas capturas reais da arena: marcador 147 e 123 de cromaticidade
# mediana, cadeira vermelha do laboratorio 62 e 65. A folga e enorme e e
# ela que permite afrouxar a geometria com seguranca.
MARKER_MASK_MIN_CHROMA = 45.0
MARKER_MIN_INSIDE_CHROMA = 90.0
MARKER_MIN_CHROMA_CONTRAST = 30.0
MARKER_MIN_CONFIDENCE = 0.58

# Rastreamento temporal e espacial de um unico destino.
MARKER_ACQUIRE_HITS = 3
MARKER_MAX_TRACK_MISSES = 2
MARKER_ASSOCIATION_MIN_PX = 28
MARKER_ASSOCIATION_SIZE_FACTOR = 0.85
MARKER_AREA_RATIO_MIN = 0.30
MARKER_AREA_RATIO_MAX = 3.50
