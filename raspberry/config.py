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
