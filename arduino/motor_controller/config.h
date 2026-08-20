#ifndef CONFIG_H
#define CONFIG_H

// ======================================================
// COMUNICACAO
// ======================================================

#define BAUD_RATE 115200

#define VELOCIDADE_MAXIMA_SEGURA 120

#define TIMEOUT_COMUNICACAO_MS 1000


// ======================================================
// LED
// ======================================================

// LED indicador controlado pela Raspberry Pi no CH4 do PCA9685.
#define LED_CANAL_PCA 4


// ======================================================
// SENSORES ULTRASSONICOS
// ======================================================

// Frontal.
#define ULTRASSOM_FRENTE_TRIG_PIN 8
#define ULTRASSOM_FRENTE_ECHO_PIN 11

// Lateral.
#define ULTRASSOM_LATERAL_TRIG_PIN 12
#define ULTRASSOM_LATERAL_ECHO_PIN 13

#define ULTRASSOM_TIMEOUT_US 30000UL


// ======================================================
// PCA9685
//
// Barramento I2C do Arduino Uno:
// SDA = A4
// SCL = A5
//
// O MPU6050 compartilha esse mesmo barramento.
// ======================================================

#define PCA9685_ENDERECO 0x40

#define PCA9685_FREQUENCIA_HZ 50

#define SERVO_PULSO_MIN_US 500

#define SERVO_PULSO_MAX_US 2500

#define SERVO_ANGULO_MIN 0

#define SERVO_ANGULO_MAX 180

#define SERVO_POSICAO_INICIAL_GARRA_ESQ 180

#define SERVO_POSICAO_INICIAL_GARRA_DIR 0

#define SERVO_POSICAO_INICIAL_CACAMBA 90


// ======================================================
// FUTABA
//
// Servo continuo de elevacao - CH3.
// ======================================================

#define FUTABA_PULSO_NEUTRO_US 1660

#define FUTABA_DESVIO_MIN_US 80

#define FUTABA_DESVIO_MAX_US 400

#define FUTABA_TEMPO_MAX_MS 3000UL


// ======================================================
// CANAIS DO PCA9685
//
// Olhando os conectores utilizados da esquerda
// para a direita:
//
// CH0 = Garra esquerda
// CH1 = Garra direita
// CH2 = Servo cacamba
// CH3 = Futaba
// CH4 = LED
// ======================================================

#define SERVO_GARRA_ESQUERDA 0

#define SERVO_GARRA_DIREITA 1

#define SERVO_CACAMBA 2

#define SERVO_FUTABA 3


// ======================================================
// MPU6050
// ======================================================

// MPU6050 com AD0 em LOW/GND.
#define MPU6050_ENDERECO 0x68


// Faz uma leitura a cada 10 ms.
// 100 Hz.
#define MPU_INTERVALO_LEITURA_US 10000UL


// Filtro complementar.
//
// 0.98:
// 98% giroscopio
// 2% acelerometro
#define MPU_FILTRO_COMPLEMENTAR 0.98f


// Escala configurada para +-2 g.
#define MPU_ACEL_LSB_POR_G 16384.0f


// Escala configurada para +-250 graus/s.
#define MPU_GIRO_LSB_POR_DPS 131.0f


// Amostras para calcular o erro/bias do giroscopio.
#define MPU_AMOSTRAS_CALIBRACAO 200


// Ao ligar, a posicao atual do robo e considerada
// a posicao de referencia/plano.
#define MPU_ZERO_AUTOMATICO_INICIAL 1


// ======================================================
// ORIENTACAO DO MPU NO SHADOW
//
// Considerando a montagem atual:
//
// Y = frente / tras
// X = esquerda / direita
// Z = cima / baixo
//
// PITCH = rampa.
// ROLL = inclinacao lateral.
//
// Se no primeiro teste levantar a frente resultar
// em PITCH negativo, mudar SOMENTE 1.0f para -1.0f.
// ======================================================

#define MPU_SINAL_RAMPA 1.0f


// ======================================================
// DETECCAO DE RAMPA
// ======================================================

// Passou de 8 graus -> entrou em rampa.
//
// O limite anterior de 7 graus tambem disparava no redutor.
#define RAMPA_ANGULO_ENTRADA_GRAUS 8.0f


// Voltou para menos de 5 graus -> piso plano.
//
// Isso cria uma histerese de 8/5 graus para evitar
// oscilacao provocada pela vibracao dos motores.
#define RAMPA_ANGULO_SAIDA_GRAUS 5.0f


// ======================================================
// DIRECAO DOS MOTORES
// ======================================================

#define DIRECAO_FE 1

#define DIRECAO_TE 1

#define DIRECAO_FD 1

#define DIRECAO_TD 1


// ======================================================
// MOTOR FRENTE ESQUERDA
//
// Driver esquerdo, canal B.
// ======================================================

#define FE_IN1 A1

#define FE_IN2 A0

#define FE_PWM 9


// ======================================================
// MOTOR TRASEIRO ESQUERDO
//
// Driver esquerdo, canal A.
// ======================================================

#define TE_IN1 A2

#define TE_IN2 A3

#define TE_PWM 10


// ======================================================
// MOTOR FRENTE DIREITA
//
// Driver direito, canal B.
// ======================================================

#define FD_IN1 3

#define FD_IN2 2

#define FD_PWM 5


// ======================================================
// MOTOR TRASEIRO DIREITO
//
// Driver direito, canal A.
// ======================================================

#define TD_IN1 4

#define TD_IN2 7

#define TD_PWM 6


#endif
