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


# ---------------------------------------------------------------------------
# Camada de plausibilidade fisica (visao/plausibilidade.py)
# ---------------------------------------------------------------------------
# Estas sao as UNICAS regras que sobrevivem a troca de arena, porque descrevem
# geometria de camera e nao aparencia. Elas dependem da ALTURA E DO ANGULO de
# montagem da camera de resgate: se a camera mudar de posicao, recalibre.
#
# Medido nas 18 fotos reais deste robo: vitimas com centro entre 0.75 e 0.96
# da altura, raio entre 0.060 e 0.115 da largura. Essas fotos cobrem apenas
# vitimas PROXIMAS, entao a envoltoria abaixo e propositalmente larga — nao
# tenho dado de vitima distante nesta camera e apertar cegamente rejeitaria
# justamente o que nunca vi. Ajuste com tools/ajustar_plausibilidade.py assim
# que houver dataset rotulado.
PLAUSIBLE_ENABLED = True
PLAUSIBLE_EDGE_MARGIN_PX = 2
# Acima disto so existe parede, publico, cadeira e mesa. Generoso de
# proposito: uma vitima distante aparece mais alta no quadro.
PLAUSIBLE_MIN_CENTER_Y_RATIO = 0.45
# Envoltoria tamanho x linha. Raio esperado, em fracao da LARGURA do quadro,
# nas duas linhas de referencia (fracao da ALTURA).
PLAUSIBLE_ROW_TOP = 0.45
PLAUSIBLE_ROW_BOTTOM = 1.00
PLAUSIBLE_RADIUS_AT_TOP = 0.020
PLAUSIBLE_RADIUS_AT_BOTTOM = 0.120
# Tolerancia larga ao redor do esperado enquanto o dataset nao existe.
PLAUSIBLE_RADIUS_TOLERANCE_LOW = 0.40
PLAUSIBLE_RADIUS_TOLERANCE_HIGH = 2.50

# ---------------------------------------------------------------------------
# Detector de vitimas por modelo treinado (visao/vitima_yolo.py)
# ---------------------------------------------------------------------------
# O modelo NAO acompanha o repositorio: ele depende de imagens da camera
# deste robo. Enquanto o arquivo nao existir, o detector falha alto e
# explica o que falta — nunca finge estar pronto.
VICTIM_MODEL_PATH = "modelos/vitimas.onnx"
VICTIM_MODEL_INPUT = 320
VICTIM_MODEL_MIN_CONFIDENCE = 0.45
VICTIM_MODEL_NMS_IOU = 0.45
# Ordem das classes no modelo. Precisa bater com o data.yaml do treino.
VICTIM_MODEL_CLASSES = ("black", "silver")

# Confirmacao temporal e rastreamento do alvo unico.
VICTIM_ACQUIRE_HITS = 3
VICTIM_MAX_TRACK_MISSES = 2
VICTIM_ASSOCIATION_MIN_PX = 34
VICTIM_ASSOCIATION_RADIUS_FACTOR = 1.05
VICTIM_RADIUS_RATIO_MIN = 0.55
VICTIM_RADIUS_RATIO_MAX = 1.80

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


BALL_SILVER_S_MAX = 88
BALL_SILVER_SMOOTH_INNER_V_MIN = 115
BALL_MIN_CONFIDENCE = 0.56

# Rastreamento e confirmacao temporal.
BALL_ACQUIRE_HITS = 3
# O segundo gate temporal do worker tambem tolera somente uma falha entre
# resultados novos do mesmo track bloqueado.
BALL_FRESH_GATE_MAX_MISSES = 1


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
# Primeiro o robo avanca por 1 s com o elevador levantado. Depois para, baixa
# o Futaba, avanca por mais 1 s e completa 200 ms antes de fechar as garras.
# Separar os avancos impede que a garra desca longe demais da esfera.
BALL_PICKUP_FUTABA_POWER = -20
BALL_PICKUP_FUTABA_MS = 1500
BALL_PICKUP_FUTABA_GUARD_S = 0.10
BALL_PICKUP_LEFT_DELTA = -70
BALL_PICKUP_RIGHT_DELTA = 70
BALL_PICKUP_PRE_FORWARD_S = 1.00
BALL_PICKUP_FORWARD_S = 1.00
BALL_PICKUP_FINAL_FORWARD_S = 0.20
# As garras so fecham depois do segundo avanco. O alias e mantido para os
# modulos que usam o nome antigo deste intervalo.
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
# A abertura compensa todo o fechamento de 70 graus para a vitima nao ficar
# presa durante a selecao esquerda/direita.
BALL_PICKUP_RELEASE_DELTA = 70
BALL_PICKUP_WIGGLE_DELTA = 40
BALL_PICKUP_WIGGLE_REPETITIONS = 2
BALL_PICKUP_WIGGLE_STEP_S = 0.20

# Busca das proximas vitimas. O Shadow nao possui IMU: o 360 e temporizado.
# steer() multiplica o pivot por 1,2, por isso 80 / (120 * 1,2) produz PWM 80
# real em cada lado, igual ao PWM normal do segue-linha.
BALL_SEARCH_TANK_ANGLE = 180
BALL_SEARCH_TANK_PWM = 80
BALL_SEARCH_TANK_SPEED = BALL_SEARCH_TANK_PWM / (120 * 1.2)
# A volta antiga media 8,93 s com speed 0,22. A proporcao entre as velocidades
# estima 3,54 s ativos com PWM 80. Conferir no piso real e ajustar so este
# valor caso a volta termine antes ou depois de 360 graus.
BALL_SEARCH_FULL_TURN_S = 3.54
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
# Duracao ativa de cada pulso. Com 3,54 s para 360 graus, 0,40 s equivalem a
# aproximadamente 41 graus: gira com forca, para e ainda deixa sobreposicao
# suficiente entre os campos observados da camera.
BALL_SEARCH_PULSE_S = 0.40
# Pausa mecanica antes de olhar: vibracao do chassi e autoexposure.
BALL_SEARCH_SETTLE_S = 0.12
# Frames novos e nitidos observados a cada parada (2 a 4).
BALL_SEARCH_OBSERVE_FRAMES = 3
# Teto de espera por esses frames; sem isso uma camera travada pararia a busca.
BALL_SEARCH_OBSERVE_TIMEOUT_S = 0.60
# Setores de cobertura do giro completo. Serve de referencia cruzada com
# BALL_SEARCH_PULSE_S: setores * pulso deve ficar proximo do 360 temporizado.
BALL_SEARCH_SECTORS = 9
# Teto global da busca, contando pulsos e pausas. Protege contra laco infinito
# quando o 360 temporizado nao fecha por escorregamento das rodas.
BALL_SEARCH_TOTAL_TIMEOUT_S = 75.0

# Fim da busca de vitimas. O verde so conta durante uma observacao parada da
# busca pulsada. Varios frames seguidos do mesmo marcador valem uma aparicao;
# ele precisa sumir por tres frames validos antes de poder contar novamente.
# Uma coleta e selecao concluidas zeram toda esta contagem.
RESCUE_GREEN_SIGHTINGS_REQUIRED = 2
RESCUE_GREEN_REARM_FRAMES = 3
# O rastreador ja exigiu tres aparicoes e a rota so comeca depois da segunda
# passagem verde. Para entregar o controle ao ultrassonico basta um frame NOVO
# depois da parada, desde que o painel esteja proximo e centralizado. Exigir
# mais tres frames aqui prendia o robo em "confirmando 1/3" na arena.
RESCUE_GREEN_CAMERA_NEAR_CONFIRM_FRAMES = 1
# Se o verde nao puder ser confirmado, o robo nao pode girar para sempre.
# Depois de tres voltas completas ele para em estado de falha, sem declarar
# falsamente que terminou o resgate.
RESCUE_SEARCH_MAX_EMPTY_SWEEPS = 3

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

# Fim do resgate: depois que a busca confirma que nao restaram vitimas, o
# robo procura o retangulo verde com o controlador de deposito acima. Quando
# chega perto, a geometria deixa de ser confiavel porque o verde ocupa quase
# todo o quadro. A ultima aproximacao usa diretamente a porcentagem de pixels
# verdes no centro da imagem e confirma em varios frames diferentes.
RESCUE_GREEN_FULL_FRAME_MIN_RATIO = 0.88
RESCUE_GREEN_FULL_FRAME_MARGIN_RATIO = 0.03
RESCUE_GREEN_FULL_FRAME_CONFIRM_FRAMES = 3
RESCUE_GREEN_FULL_FRAME_CONFIRM_WINDOW_S = 0.50
# Depois da segunda passagem verde, a camera confirma, centraliza e aproxima o
# robo do painel. So depois dessa etapa o avanco final usa o PWM 80.
RESCUE_GREEN_FINAL_PWM = 80
RESCUE_GREEN_FINAL_FORWARD_SPEED = RESCUE_GREEN_FINAL_PWM / 120.0
# O alinhamento antigo usava arco em speed 0,25: com erro +0,58 isso chegava
# ao chassi como aproximadamente PWM 32/12 e nao vencia o atrito. Na rota
# final, centralizar usa tanque com o mesmo PWM real 80 da busca; depois a
# aproximacao visual tambem conserva PWM 80 ate entregar ao ultrassonico.
RESCUE_GREEN_CAMERA_ALIGN_TANK_SPEED = BALL_SEARCH_TANK_SPEED
RESCUE_GREEN_CAMERA_APPROACH_SPEED = RESCUE_GREEN_FINAL_FORWARD_SPEED
# A camera frontal nao consegue ficar totalmente coberta pelo retangulo.
# A chegada real e confirmada pelo HC-SR04 a 5 cm (o sensor usa milimetros).
RESCUE_GREEN_ARRIVAL_DISTANCE_MM = 50
# O loop principal roda a cada 5 ms, mas consultar a USB nessa frequencia nao
# melhora o HC-SR04. Vinte ms ainda recolhe a resposta antes da proxima medida.
RESCUE_GREEN_ULTRASONIC_POLL_INTERVAL_S = 0.02
# Como o deposito e irreversivel, as tres leituras precisam concordar. O
# monitor nasce somente depois do alinhamento, portanto nao herda ecos do giro.
RESCUE_GREEN_ULTRASONIC_CONFIRM_READINGS = 3
# O robo nunca avanca sem uma leitura valida. Tres medidas sem eco encerram a
# aproximacao para impedir que os quatro motores fiquem presos contra a parede.
RESCUE_GREEN_ULTRASONIC_MAX_NO_ECHO = 3
RESCUE_GREEN_FINAL_CENTER_DEADBAND = 0.06
RESCUE_GREEN_FINAL_STEER_MAX_ANGLE = 30
RESCUE_GREEN_FINAL_MIN_VISIBLE_RATIO = 0.002
RESCUE_GREEN_FINAL_LOST_TIMEOUT_S = 0.75
RESCUE_GREEN_FINAL_MAX_ACTIVE_S = 5.0

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
MARKER_GREEN_HSV_MIN = (45, 60, 30)
MARKER_GREEN_HSV_MAX = (105, 255, 255)
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
# FORMA NAO DISCRIMINA MARCADOR NESTA CAMERA. Medido no pipeline real:
#
#   marcador a distancia de navegacao  triangularidade 0.577  proporcao 1.03
#   marcador de perto                  triangularidade 0.623  proporcao 3.83
#   cadeira vermelha do laboratorio    triangularidade 0.677  proporcao 3.80
#   circulo perfeito                   triangularidade 0.605
#   quadrado perfeito                  triangularidade 0.500
#
# A cadeira e MAIS triangular que o marcador, e o marcador cai entre quadrado
# e circulo. Nenhum limiar de triangularidade aceita o marcador e rejeita um
# circulo — a camera olha quase rente ao piso e o escorco destroi a forma.
#
# Por isso o gate de triangularidade FOI REMOVIDO e o rigor foi transferido
# para a CROMATICIDADE do blob, onde a separacao medida e enorme:
# marcador 124-148 contra cadeira 63-79 (ver MARKER_MIN_INSIDE_CHROMA).
#
# Consequencia honesta: um circulo verde ou vermelho MUITO saturado, no chao,
# dentro da ROI, seria aceito. Nao existe objeto assim na arena, mas isso e
# uma protecao que foi perdida de proposito porque ela era incompativel com
# detectar o marcador real.
MARKER_MAX_ASPECT_RATIO = 6.0
MARKER_MIN_SOLIDITY = 0.70
MARKER_MIN_MASK_FILL = 0.70
MARKER_APPROX_EPSILON_RATIO = 0.055
# Sanidade de forma apenas: exclui filamento e mancha difusa, nao julga
# triangulo contra retangulo contra circulo.
MARKER_MIN_COMPACTNESS = 0.35

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

# Detector complementar do RETANGULO verde. O detector acima continua sendo
# usado e o vermelho nao muda. Na camera real, o painel verde apareceu como
# B=131, G=110, R=31: o azul fica um pouco acima do verde por causa do balanco
# de branco, mas ambos continuam muito acima do vermelho. O detector aceita
# essa mudanca para ciano e o verde normal, sempre com area solida e tres
# frames consecutivos.
GREEN_RECTANGLE_ROI_TOP = 0.45
GREEN_RECTANGLE_MIN_AREA_RATIO = 0.0025
GREEN_RECTANGLE_MAX_AREA_RATIO = 0.45
GREEN_RECTANGLE_MIN_SIDE_PX = 10
GREEN_RECTANGLE_MIN_HORIZONTAL_ASPECT = 1.80
GREEN_RECTANGLE_MIN_SATURATION = 120.0
GREEN_RECTANGLE_MIN_CHROMA = 40.0
GREEN_RECTANGLE_MIN_SOLIDITY = 0.60
GREEN_RECTANGLE_MIN_COMPACTNESS = 0.25
GREEN_RECTANGLE_ACQUIRE_HITS = 3
GREEN_RECTANGLE_MAX_MISSES = 2
GREEN_RECTANGLE_ASSOCIATION_MIN_PX = 40
GREEN_RECTANGLE_ASSOCIATION_SIZE_FACTOR = 1.25

# Deposito final das vitimas prata/cinza guardadas no lado esquerdo.
# O giro completo medido leva 3,54 s a PWM 80; metade dele fornece o primeiro
# valor calibrado para 180 graus. Todos os prazos comecam somente depois que o
# comando correspondente e aceito pela serial.
SILVER_DEPOSIT_TURN_SPEED = BALL_SEARCH_TANK_SPEED
SILVER_DEPOSIT_TURN_S = BALL_SEARCH_FULL_TURN_S / 2.0
SILVER_DEPOSIT_REVERSE_SPEED = 0.35
SILVER_DEPOSIT_REVERSE_S = 3.0
SILVER_DEPOSIT_BUCKET_OPEN_DELTA = 90
SILVER_DEPOSIT_BUCKET_SETTLE_S = 0.60
SILVER_DEPOSIT_SHAKE_PWM = 80
SILVER_DEPOSIT_SHAKE_SPEED = SILVER_DEPOSIT_SHAKE_PWM / 120.0
SILVER_DEPOSIT_SHAKE_MOVE_S = 0.18
SILVER_DEPOSIT_SHAKE_STOP_S = 0.08
SILVER_DEPOSIT_SHAKE_REPETITIONS = 2
SILVER_DEPOSIT_BUCKET_RESTORE_DELTA = -90
SILVER_DEPOSIT_BUCKET_RESTORE_S = 0.60
