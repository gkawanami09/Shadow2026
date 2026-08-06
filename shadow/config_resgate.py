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

# Ao entrar na sala pela missao completa, o segue-linha entrega os motores
# parados ao resgate. O robo atravessa a faixa prata em linha reta antes de
# iniciar os pulsos de giro que procuram as vitimas. Este movimento nao e
# executado ao abrir ``resgate.py`` sozinho.
MISSION_ENTRY_FORWARD_S = 1.0
MISSION_ENTRY_FORWARD_PWM = 80
MISSION_ENTRY_FORWARD_SPEED = MISSION_ENTRY_FORWARD_PWM / 120.0


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
# Um pequeno aumento sobre os antigos 36..41 PWM vence melhor o atrito sem
# deixar o alinhamento agressivo. O angulo continua proporcional ao erro.
BALL_ALIGN_PWM = 50
BALL_ALIGN_SPEED_MIN = BALL_ALIGN_PWM / 120.0
BALL_ALIGN_SPEED_MAX = BALL_ALIGN_PWM / 120.0
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

# Coleta normal depois que a aproximacao visual termina. Nela nao existe re.
# Primeiro o robo baixa o Futaba, sem avancar. Depois percorre com ele baixo a
# soma dos dois avancos antigos (1 s + 1 s) e completa mais 200 ms antes de
# fechar as garras. Somente o modo de parede, configurado abaixo, acrescenta re.
BALL_PICKUP_FUTABA_POWER = -20
BALL_PICKUP_FUTABA_MS = 1500
BALL_PICKUP_FUTABA_GUARD_S = 0.10
BALL_PICKUP_LEFT_DELTA = -55
BALL_PICKUP_RIGHT_DELTA = 55
BALL_PICKUP_PRE_FORWARD_S = 1.00
BALL_PICKUP_FORWARD_S = 1.00
BALL_PICKUP_FINAL_FORWARD_S = 0.20
# Todo o avanco principal agora acontece com o Futaba embaixo.
BALL_PICKUP_FORWARD_LEAD_S = (
    BALL_PICKUP_PRE_FORWARD_S + BALL_PICKUP_FORWARD_S
)
BALL_PICKUP_FORWARD_SPEED = BALL_APPROACH_SPEED_NEAR
BALL_PICKUP_GRIPPER_SETTLE_S = 0.50
# Primeiro captura a esfera rapidamente e termina em um unico passo menor.
# Continua movendo uma garra por vez, separada por 40 ms, para limitar o pico
# de corrente, mas chega ao angulo final com 4 comandos em vez de 8.
BALL_PICKUP_GRIPPER_CAPTURE_DEGREES = 40
BALL_PICKUP_GRIPPER_CAPTURE_INTERVAL_S = 0.04
BALL_PICKUP_GRIPPER_STEP_DEGREES = 15
BALL_PICKUP_GRIPPER_STEP_INTERVAL_S = 0.05

# Depois de prender a esfera, o elevador sobe, aplica um pulso curto para
# descer e entao executa a liberacao correspondente a cor confirmada.
# A subida terminava em 2,5 s. Agora ela para 200 ms antes e desacelera nos
# 400 ms finais para nao bater no limite e voltar pela elasticidade/folga.
BALL_PICKUP_LIFT_POWER = 20
BALL_PICKUP_LIFT_MS = 1900
BALL_PICKUP_LIFT_SLOW_POWER = 10
BALL_PICKUP_LIFT_SLOW_MS = 400
# Sustentacao curta no alto. Em servo continuo isto e velocidade minima, nao
# controle real de torque; por isso o comando e fraco e limitado a 300 ms.
BALL_PICKUP_LIFT_HOLD_POWER = 1
BALL_PICKUP_LIFT_HOLD_MS = 300
BALL_PICKUP_LOWER_POWER = -20
BALL_PICKUP_LOWER_MS = 25
BALL_PICKUP_LOWER_GUARD_S = 0.05

# Se o Arduino reiniciar durante a coleta, o firmware volta as garras para a
# posicao aberta. O Raspberry espera a nova conexao, mantem as rodas zeradas,
# leva o Futaba novamente para cima e reinicia a mesma coleta normal. O limite
# impede insistencia infinita caso exista uma falha eletrica ou mecanica real.
BALL_PICKUP_SERIAL_RECOVERY_MAX_RETRIES = 2
BALL_PICKUP_SERIAL_RECOVERY_CONNECT_TIMEOUT_S = 8.0
BALL_PICKUP_SERIAL_RECOVERY_POLL_S = 0.05
# A abertura compensa todo o fechamento de 70 graus para a vitima nao ficar
# presa durante a selecao esquerda/direita.
BALL_PICKUP_RELEASE_DELTA = 70
BALL_PICKUP_WIGGLE_DELTA = 40
BALL_PICKUP_WIGGLE_REPETITIONS = 2
BALL_PICKUP_WIGGLE_STEP_S = 0.20

# --- Teste experimental de vitima encostada na parede ---------------------
# Quando a vitima chega ao ponto de coleta, o ultrassonico mede primeiro no
# centro. Se houver eco proximo, o robo mede uma vez de cada lado da posicao
# original e retorna ao centro. Distancias proximas e parecidas nos dois lados
# indicam parede reta provavel. Resultado misto ou muito diferente aciona uma
# correcao curta de yaw e o teste inteiro precisa convergir antes das garras.
# Se ambos parecem livres, a varredura angular abaixo elimina o ponto cego de
# uma parede inclinada antes de liberar a coleta normal.
# Desativado: toda vitima usa a mesma coleta normal. O ultrassonico nao pode
# mais trocar a sequencia apenas porque encontrou uma parede ou outro eco.
BALL_WALL_TEST_ENABLED = False
BALL_WALL_PROBE_DISTANCE_MM = 220
BALL_WALL_PROBE_SAMPLES = 3
BALL_WALL_PROBE_MIN_CLOSE_SAMPLES = 2
BALL_WALL_PROBE_SIMILARITY_MM = 45
BALL_WALL_PROBE_READ_TIMEOUT_S = 0.08
BALL_WALL_PROBE_SAMPLE_INTERVAL_S = 0.06
BALL_WALL_PROBE_MEASURE_TIMEOUT_S = 0.60
BALL_WALL_PROBE_LATERAL_PWM = 50
BALL_WALL_PROBE_LATERAL_S = 0.50
BALL_WALL_PROBE_SETTLE_S = 0.15
BALL_WALL_PROBE_FRAME_TIMEOUT_S = 0.60
BALL_WALL_PROBE_BALL_OUTSIDE_CENTER_ERROR = 0.20
BALL_WALL_PROBE_VISUAL_CONFIRM_FRAMES = 2
BALL_WALL_PROBE_RETURN_MAX_CENTER_ERROR = 0.22
BALL_WALL_PROBE_RADIUS_RATIO_MIN = 0.65
BALL_WALL_PROBE_RADIUS_RATIO_MAX = 1.45
BALL_WALL_PROBE_CENTER_Y_TOLERANCE_RATIO = 0.18
# O teste nasce somente depois do NEAR visual. Antes do handoff direto para o
# Futaba, a mesma esfera precisa voltar tambem a uma profundidade parecida com
# esse NEAR; o envelope largo acima serve apenas para acompanha-la nas manobras.
BALL_WALL_FINAL_RADIUS_RATIO_MIN = 0.85
BALL_WALL_FINAL_RADIUS_RATIO_MAX = 1.18
BALL_WALL_FINAL_BOTTOM_Y_TOLERANCE_RATIO = 0.06
# A correcao final e somente longitudinal: as quatro rodas recebem o mesmo
# comando curto, o robo freia e espera a imagem assentar antes de medir de
# novo. Ela nunca tenta compensar uma leitura ruim com um movimento longo.
BALL_WALL_DEPTH_PWM = 40
BALL_WALL_DEPTH_PULSE_S = 0.08
BALL_WALL_DEPTH_SETTLE_S = 0.15
BALL_WALL_DEPTH_MAX_PULSES = 6
BALL_WALL_DEPTH_MIN_PROGRESS = 0.01
BALL_WALL_DEPTH_MAX_NO_PROGRESS_PULSES = 2
# Se os dois lados medidos nao concordarem, o robo corrige o proprio angulo
# antes de repetir o teste. O pivo usa somente as rodas traseiras: assim a
# frente, que ja esta perto da vitima, desloca o minimo possivel.
BALL_WALL_ALIGN_PIVOT_PWM = 50
BALL_WALL_ALIGN_PIVOT_S = 0.12
BALL_WALL_ALIGN_CENTER_DEADBAND = 0.08
BALL_WALL_ALIGN_OMNI_PWM = 45
BALL_WALL_ALIGN_OMNI_PULSE_S = 0.08
BALL_WALL_ALIGN_SETTLE_S = 0.15
BALL_WALL_ALIGN_MAX_CORRECTIONS = 3
BALL_WALL_ALIGN_MAX_OMNI_PULSES = 6
# Duas tentativas sem reduzir o erro horizontal encerram o teste. Isso evita
# insistir contra uma quina ou perseguir outra esfera da mesma cor.
BALL_WALL_ALIGN_MIN_PROGRESS = 0.01
BALL_WALL_ALIGN_MAX_NO_PROGRESS_PULSES = 2
# Se os dois offsets laterais parecem livres, ainda pode existir uma parede
# inclinada fora do cone estreito do ultrassonico. Em cada offset o robo gira
# somente a traseira para varrer tres angulos, freia, confirma um frame novo e
# mede uma bateria completa. Se nao achar eco, restaura o angulo pela posicao
# visual que a mesma esfera tinha antes da varredura.
BALL_WALL_SCAN_PIVOT_PWM = 50
BALL_WALL_SCAN_PULSE_S = 0.12
BALL_WALL_SCAN_SETTLE_S = 0.15
BALL_WALL_SCAN_MAX_OUTWARD_PULSES_PER_SIDE = 3
BALL_WALL_SCAN_MAX_RESTORE_PULSES_PER_SIDE = 4
BALL_WALL_SCAN_TOTAL_PULSE_LIMIT = 12
BALL_WALL_SCAN_MIN_VISUAL_PROGRESS = 0.02
BALL_WALL_SCAN_RESTORE_DEADBAND = 0.04
BALL_WALL_SCAN_MAX_NO_PROGRESS_PULSES = 2
BALL_WALL_REAPPROACH_AUTH_S = 3.00

# Depois de confirmar a parede, a nova aproximacao usa a coleta especial:
# empurra um pouco mais com a garra aberta, da uma re curta e so entao fecha.
BALL_WALL_PICKUP_FORWARD_S = 2.50
BALL_WALL_PICKUP_REVERSE_SPEED = 0.65
BALL_WALL_PICKUP_REVERSE_S = 0.85
BALL_WALL_PICKUP_DIRECTION_CHANGE_PAUSE_S = 0.05
BALL_WALL_PICKUP_POST_REVERSE_PAUSE_S = 0.10

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
# busca pulsada. Varios frames seguidos do mesmo marcador valem uma aparicao e
# cada varredura completa pode somar no maximo uma vez, mesmo se a deteccao
# oscilar. Uma coleta e selecao concluidas zeram toda esta contagem.
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
# A busca do marcador vermelho usa o mesmo ritmo seguro da busca das bolas:
# gira um trecho, freia, espera a vibracao e somente entao observa a camera.
DEPOSIT_SEARCH_PULSE_S = BALL_SEARCH_PULSE_S
DEPOSIT_SEARCH_SETTLE_S = BALL_SEARCH_SETTLE_S
DEPOSIT_SEARCH_OBSERVE_FRAMES = BALL_SEARCH_OBSERVE_FRAMES
DEPOSIT_SEARCH_OBSERVE_TIMEOUT_S = BALL_SEARCH_OBSERVE_TIMEOUT_S
RED_DEPOSIT_SEARCH_TANK_SPEED = BALL_SEARCH_TANK_SPEED
RED_DEPOSIT_SEARCH_FULL_TURN_S = BALL_SEARCH_FULL_TURN_S
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
# Para compensar a inercia, a chegada e confirmada 1 cm antes: 7 cm do painel.
# O sensor e todo o protocolo continuam usando milimetros.
RESCUE_GREEN_ARRIVAL_DISTANCE_MM = 70
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

# A soleira de saída aparece rente ao chão. Os 78% superiores ficam fora da
# procura para o robô, o reflexo na parede e o retângulo verde não virarem
# candidatos. A confirmação final continua sendo feita pela câmera de linha.
EXIT_BLACK_ROI_TOP = 0.78
EXIT_BLACK_ROI_BOTTOM = 1.00

EXIT_BLACK_MIN_ROW_FILL = 0.30
EXIT_BLACK_MIN_SPAN_RATIO = 0.28
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
EXIT_BLACK_MIN_ASPECT = 3.0

EXIT_BLACK_MAX_INSIDE_VALUE = 80.0
# Contraste COM SINAL: o piso ao redor precisa ser mais claro que a faixa.
# Uma sombra grande sobre piso escuro nao satisfaz isso.
EXIT_BLACK_SURROUND_MARGIN_RATIO = 0.06
EXIT_BLACK_MIN_SURROUND_CONTRAST = 25.0
EXIT_BLACK_MIN_CONFIDENCE = 0.55

# Um retangulo verde escuro tambem pode cair abaixo de V=70 e entrar na
# mascara "preta". Antes de aceitar a geometria, vetamos candidatos cuja
# vizinhanca tenha uma quantidade relevante de verde saturado.
EXIT_BLACK_GREEN_VETO_HSV_MIN = (40, 80, 35)
EXIT_BLACK_GREEN_VETO_HSV_MAX = (110, 255, 255)
EXIT_BLACK_GREEN_VETO_MARGIN_RATIO = 0.06
EXIT_BLACK_GREEN_VETO_MAX_RATIO = 0.18

# A câmera de resgate enxerga a soleira rente ao chão por poucos quadros entre
# dois pulsos. Duas imagens novas ainda rejeitam um borrão isolado, e esta
# decisão NÃO libera a saída sozinha: ultrassônico e câmera de linha fazem as
# duas confirmações seguintes.
EXIT_BLACK_VOTES_NEEDED = 2
EXIT_BLACK_VOTE_WINDOW = 5
EXIT_BLACK_COOLDOWN_S = 0.0

# A camera de resgate fica quase horizontal. Quando a soleira esta longe ela
# aparece como uma linha fina e inclinada, nao como um retangulo perfeitamente
# horizontal. Um segmento isolado nao basta: riscos, manchas e quinas da arena
# tambem produzem linhas no Canny. O detector junta somente trechos colineares
# e exige uma borda escura continua ocupando quase metade da imagem.
EXIT_LINE_ROI_TOP = 0.78
EXIT_LINE_ROI_BOTTOM = 0.995
EXIT_LINE_CANNY_LOW = 25
EXIT_LINE_CANNY_HIGH = 90
EXIT_LINE_HOUGH_THRESHOLD_RATIO = 0.020
# Cada pedaco pode ser curto porque reflexos quebram visualmente a fita.
EXIT_LINE_MIN_SEGMENT_RATIO = 0.05
# Depois de unir os pedacos, a borda precisa atravessar boa parte da imagem.
EXIT_LINE_MIN_LENGTH_RATIO = 0.25
EXIT_LINE_MAX_GAP_RATIO = 0.04
EXIT_LINE_MAX_ANGLE_DEG = 22.0
EXIT_LINE_MAX_GROUP_ANGLE_DIFF_DEG = 5.0
EXIT_LINE_MAX_GROUP_Y_DISTANCE_RATIO = 0.025
EXIT_LINE_MAX_JOIN_GAP_RATIO = 0.08
EXIT_LINE_MAX_DARK_SIDE_VALUE = 125.0
EXIT_LINE_MIN_DARK_SUPPORT = 0.55
EXIT_LINE_MIN_SIDE_CONTRAST = 18.0
EXIT_LINE_MIN_CONTRAST_SUPPORT = 0.45

# Os votos sao feitos com o robo parado. Portanto, a mesma faixa deve aparecer
# praticamente no mesmo lugar nos frames usados na confirmacao. Isso impede
# que tres bordas/manchas diferentes completem a votacao 3-de-5.
EXIT_BLACK_MAX_VOTE_CENTER_X_DRIFT_RATIO = 0.14
EXIT_BLACK_MAX_VOTE_CENTER_Y_DRIFT_RATIO = 0.10
EXIT_BLACK_MAX_VOTE_SPAN_DRIFT_RATIO = 0.22

# Travessia da soleira de saida. Igual a entrada: o tempo e apenas o limite
# de seguranca; o fim normal e a faixa deixar de ser vista.
# Ao confirmar a faixa distante, o robo precisa vencer o atrito imediatamente.
# 0.35 equivalia a somente 42 PWM e, no chassi real, podia deixar os motores
# apenas fazendo ruido. A saida usa o mesmo PWM forte ja validado nos demais
# movimentos do resgate.
EXIT_ADVANCE_PWM = 80
EXIT_ADVANCE_SPEED = EXIT_ADVANCE_PWM / 120.0
EXIT_ADVANCE_MIN_S = 0.60
EXIT_ADVANCE_TIMEOUT_S = 3.5

# Antes de entregar a decisao para a camera do segue-linha, o robo para e
# confirma que nao esta apenas olhando uma mancha/objeto proximo. Tres medidas
# livres sao obrigatorias. Duas medidas proximas confirmam o bloqueio; medida
# sem eco ou resultado misturado nunca autoriza a troca de camera.
EXIT_CLEARANCE_DISTANCE_MM = 150
EXIT_CLEARANCE_VALID_READINGS = 5
EXIT_CLEARANCE_NEAR_CONFIRMATIONS = 2
EXIT_CLEARANCE_SETTLE_S = 0.20
EXIT_CLEARANCE_TIMEOUT_S = 1.20
EXIT_CLEARANCE_READ_TIMEOUT_S = 0.08
EXIT_CLEARANCE_SAMPLE_INTERVAL_S = 0.06
EXIT_CLEARANCE_MIN_VALID_MM = 1
EXIT_CLEARANCE_MAX_VALID_MM = 4000
EXIT_CLEARANCE_REVERSE_SPEED = EXIT_ADVANCE_SPEED
# Quando o ultrassonico veta um candidato visto pela camera de resgate, um
# recuo curto e um pequeno giro mudam o ponto de vista antes da nova busca.
EXIT_CLEARANCE_BLOCKED_REVERSE_S = 0.30
# Depois de um bloqueio, muda o enquadramento com exatamente um pulso igual ao
# usado na procura dos retangulos.
EXIT_CLEARANCE_ESCAPE_TURN_S = DEPOSIT_SEARCH_PULSE_S

# Giro pulsado de procura da saida quando nenhuma faixa esta no campo.
EXIT_SEARCH_TIMEOUT_S = 60.0
# Usa o mesmo pulso completo da procura dos retangulos. Um candidato visto
# durante o movimento nao encurta o giro: a confirmacao so comeca depois de
# terminar o pulso, frear e assentar o chassi.
EXIT_SEARCH_PULSE_S = DEPOSIT_SEARCH_PULSE_S
EXIT_SEARCH_SETTLE_S = BALL_SEARCH_SETTLE_S
# A faixa preta é muito mais fina que uma esfera. Dê tempo para dois frames
# nítidos chegarem depois de a vibração do pulso acabar.
EXIT_SEARCH_OBSERVE_TIMEOUT_S = DEPOSIT_SEARCH_VERIFY_TIMEOUT_S
EXIT_SEARCH_TANK_ANGLE = DEPOSIT_SEARCH_TANK_ANGLE
EXIT_SEARCH_TANK_SPEED = RED_DEPOSIT_SEARCH_TANK_SPEED

# Alinhamento com o centro da soleira antes de qualquer avanco. Tanque com
# PWM 50 gira o chassi sem se aproximar da faixa enquanto ainda esta torto.
EXIT_ALIGN_MAX_CENTER_ERROR = 0.10
EXIT_ALIGN_ANGLE = 180
EXIT_ALIGN_PWM = 50
EXIT_ALIGN_SPEED = EXIT_ALIGN_PWM / (120 * 1.2)
EXIT_ALIGN_SETTLE_S = EXIT_SEARCH_SETTLE_S
# Uma falha de segmentação isolada não pode mandar o robô voltar ao giro e
# ultrapassar uma faixa que já estava centralizada.
EXIT_ALIGN_LOST_TIMEOUT_S = 0.35

# Confirmacao final com a camera do segue-linha. Primeiro a camera de resgate
# aproxima ate deixar de enxergar a soleira. So entao a camera de linha abre e
# o robo avanca devagar. Ao primeiro sinal da faixa ele PARA, espera a
# autoexposicao assentar e zera os votos. Cinza/prata precisa de apenas dois
# votos para bloquear; preto precisa de quatro. Assim uma prata escura vista
# de longe nunca vence a votacao antes de seus reflexos aparecerem.
EXIT_LINE_VERIFY_SPEED = 0.35
EXIT_LINE_VERIFY_TIMEOUT_S = 5.0
EXIT_LINE_VERIFY_WINDOW = 5
EXIT_LINE_VERIFY_SETTLE_S = 0.25
# Antes de votar preto/prata, a faixa precisa chegar ao meio da imagem da
# camera de linha. Se passar do centro, o mesmo PWM baixo corrige em re.
EXIT_LINE_VERIFY_CENTER_Y_RATIO = 0.50
EXIT_LINE_VERIFY_CENTER_Y_TOLERANCE = 0.10
EXIT_LINE_VERIFY_CENTER_SPEED = 0.25
EXIT_LINE_VERIFY_BLACK_VOTES = 3
EXIT_LINE_VERIFY_SILVER_VOTES = 3
EXIT_LINE_VERIFY_MAX_AGE_S = 0.35
EXIT_LINE_VERIFY_EDGE_MIN = 18
EXIT_LINE_VERIFY_EDGE_FILL = 0.48
EXIT_LINE_VERIFY_DARK_VALUE_MAX = 100
EXIT_LINE_VERIFY_DARK_LOCAL_MAX = 12
EXIT_LINE_VERIFY_DARK_ROW_FILL = 0.60
EXIT_LINE_VERIFY_DARK_MIN_HEIGHT_RATIO = 0.05
# Medido nas quatro imagens reais de 05/08: preto ficou em 6-7 e prata em
# 13-21 depois da normalizacao para 448x252. A zona 9.5..11.5 permanece
# inconclusiva em vez de arriscar classificar prata como preta.
EXIT_LINE_VERIFY_BLACK_TEXTURE_MAX = 10.5
# A janela local acompanha a faixa e pode incluir a propria borda. Por isso
# recebe uma folga pequena; preto ainda exige que a janela global seja lisa.
EXIT_LINE_VERIFY_BLACK_LOCAL_TEXTURE_MAX = 13.5
EXIT_LINE_VERIFY_SILVER_TEXTURE_MIN = 11.5
EXIT_LINE_VERIFY_TEXTURE_ROI_TOP = 0.15
EXIT_LINE_VERIFY_TEXTURE_ROI_BOTTOM = 0.75
# Depois de centralizar, a textura e medida a partir da borda da propria
# faixa, nao mais em uma janela fixa que podia ficar acima dela.
EXIT_LINE_VERIFY_TEXTURE_BAND_HEIGHT_RATIO = 0.35
EXIT_LINE_VERIFY_REJECT_REVERSE_SPEED = 0.35
EXIT_LINE_VERIFY_REJECT_REVERSE_S = 1.0
EXIT_LINE_VERIFY_BLACK_FORWARD_SPEED = 0.40
EXIT_LINE_VERIFY_BLACK_FORWARD_S = 0.45

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
SILVER_DEPOSIT_PRE_TURN_PWM = 80
SILVER_DEPOSIT_PRE_TURN_SPEED = SILVER_DEPOSIT_PRE_TURN_PWM / 120.0
SILVER_DEPOSIT_PRE_TURN_FORWARD_S = 1.0
SILVER_DEPOSIT_PRE_TURN_REVERSE_S = 0.5
SILVER_DEPOSIT_TURN_SPEED = BALL_SEARCH_TANK_SPEED
SILVER_DEPOSIT_TURN_EXTRA_S = 0.50
SILVER_DEPOSIT_TURN_S = (
    BALL_SEARCH_FULL_TURN_S / 2.0 + SILVER_DEPOSIT_TURN_EXTRA_S
)
# A re antiga usava PWM 42 e uma das rodas nao vencia o atrito. As sacudidas
# ja provaram que as quatro rodas recuam corretamente em PWM 80, entao o
# alinhamento longo usa a mesma forca sem mudar sua duracao.
SILVER_DEPOSIT_REVERSE_PWM = 80
SILVER_DEPOSIT_REVERSE_SPEED = SILVER_DEPOSIT_REVERSE_PWM / 120.0
SILVER_DEPOSIT_REVERSE_S = 3.0
SILVER_DEPOSIT_BUCKET_OPEN_DELTA = -90
SILVER_DEPOSIT_BUCKET_SETTLE_S = 0.60
BLACK_DEPOSIT_BUCKET_OPEN_DELTA = 90
SILVER_DEPOSIT_SHAKE_PWM = 80
SILVER_DEPOSIT_SHAKE_SPEED = SILVER_DEPOSIT_SHAKE_PWM / 120.0
SILVER_DEPOSIT_SHAKE_MOVE_S = 0.18
SILVER_DEPOSIT_SHAKE_STOP_S = 0.08
SILVER_DEPOSIT_SHAKE_REPETITIONS = 2
SILVER_DEPOSIT_BUCKET_RESTORE_DELTA = 90
BLACK_DEPOSIT_BUCKET_RESTORE_DELTA = -90
SILVER_DEPOSIT_BUCKET_RESTORE_S = 0.60
SILVER_DEPOSIT_EXIT_FORWARD_PWM = 80
SILVER_DEPOSIT_EXIT_FORWARD_SPEED = SILVER_DEPOSIT_EXIT_FORWARD_PWM / 120.0
SILVER_DEPOSIT_EXIT_FORWARD_S = 1.5
