"""Configuracoes principais da Raspberry Pi."""

BAUD_RATE = 115200

# Use SERIAL_PORT = None para tentar detectar automaticamente.
# Se a deteccao falhar, definir manualmente, por exemplo:
# SERIAL_PORT = "/dev/ttyACM0"
SERIAL_PORT = None

VELOCIDADE_TESTE_BAIXA = 60
VELOCIDADE_TESTE_MEDIA = 80
VELOCIDADE_MAXIMA_SEGURA = 120
TIMEOUT_SERIAL = 2.0

MOTORES_ATIVADOS = True

CAMERA_HEIGHT_CM = 8
CAMERA_TILT_DEGREES = 35
CAMERA_FOV_DEGREES = 160

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
PASTA_CAPTURAS = "captures"

# ===============================
# CONFIGURACOES DA LINHA PRETA
# ===============================

# ROI em proporcao da imagem inteira.
ROI_Y_INICIO = 0.20
ROI_Y_FIM = 1.00
ROI_X_INICIO = 0.05
ROI_X_FIM = 0.95

# A parte superior esta mais distante da camera.
DIVISAO_TOPO_BAIXO = 0.40

# Preto em BGR: os tres canais da linha devem estar baixos.
PRETO_MIN_BGR = [0, 0, 0]
PRETO_MAX_TOPO_BGR = [90, 90, 90]
PRETO_MAX_BAIXO_BGR = [145, 145, 145]

# Area minima para aceitar um contorno como linha.
AREA_MINIMA_LINHA = 1500

# Limpeza simples da mascara.
KERNEL_LINHA = 3
ERODE_INICIAL = 2
DILATE_LINHA = 6
ERODE_FINAL = 2

# ===============================
# CONFIGURACOES DO COMANDO SUGERIDO
# ===============================

# Valores seguros para o primeiro teste real do robo.
# A ideia e comecar devagar e aumentar depois.
VELOCIDADE_BASE_SEGUE_LINHA = 60
VELOCIDADE_MINIMA_SEGUE_LINHA = 35
VELOCIDADE_MAXIMA_SEGUE_LINHA = 90

# Ganho proporcional inicial mais suave.
KP_SEGUE_LINHA = 0.30
CORRECAO_MAXIMA = 30

# Faixas proporcionais dentro da ROI.
FAIXA_BAIXA_INICIO = 0.65
FAIXA_BAIXA_FIM = 1.00
FAIXA_MEDIA_INICIO = 0.35
FAIXA_MEDIA_FIM = 0.65
FAIXA_ALTA_INICIO = 0.00
FAIXA_ALTA_FIM = 0.35

# A faixa baixa e a mais importante para a decisao.
PESO_FAIXA_BAIXA = 0.70
PESO_FAIXA_MEDIA = 0.25
PESO_FAIXA_ALTA = 0.05

# ===============================
# CONFIGURACOES DO PRIMEIRO SEGUE-LINHA REAL
# ===============================

DURACAO_TESTE_SEGUE_LINHA = 3.0
INTERVALO_COMANDO_SEGUNDOS = 0.10
PARAR_SE_PERDER_LINHA = True
MAX_FRAMES_SEM_LINHA = 2
