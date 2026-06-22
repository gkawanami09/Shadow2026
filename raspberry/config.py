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

# Configuracoes de visao da linha preta. Ajuste estes valores usando as imagens
# de debug salvas em captures/.
LIMIAR_PRETO = 80

# Regiao de interesse em proporcao da imagem.
ROI_Y_INICIO = 0.35
ROI_Y_FIM = 1.00
ROI_X_INICIO = 0.10
ROI_X_FIM = 0.90

AREA_MINIMA_LINHA = 800
IMAGEM_TESTE_PADRAO = None
