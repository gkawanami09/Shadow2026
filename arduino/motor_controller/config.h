#ifndef CONFIG_H
#define CONFIG_H

#define BAUD_RATE 115200
#define VELOCIDADE_MAXIMA_SEGURA 120
#define TIMEOUT_COMUNICACAO_MS 1000

// LED indicador controlado pela Raspberry Pi.
#define LED_PIN 12

// Sensor ultrassonico.
#define ULTRASSOM_TRIG_PIN 8
#define ULTRASSOM_ECHO_PIN 11
#define ULTRASSOM_TIMEOUT_US 30000UL

// PCA9685 no barramento I2C do Uno (SDA=A4, SCL=A5).
#define PCA9685_ENDERECO 0x40
#define PCA9685_FREQUENCIA_HZ 50
#define SERVO_PULSO_MIN_US 500
#define SERVO_PULSO_MAX_US 2500
#define SERVO_ANGULO_MIN 0
#define SERVO_ANGULO_MAX 180
#define SERVO_POSICAO_INICIAL_GARRA_ESQ 180
#define SERVO_POSICAO_INICIAL_GARRA_DIR 0
#define SERVO_POSICAO_INICIAL_CACAMBA 90

// Servo continuo de elevacao (CH3): potencia assinada e tempo limitado.
#define FUTABA_PULSO_NEUTRO_US 1660
#define FUTABA_DESVIO_MIN_US 80
#define FUTABA_DESVIO_MAX_US 400
#define FUTABA_TEMPO_MAX_MS 3000UL

// Canais dos servos no PCA9685.
#define SERVO_GARRA_ESQUERDA 0
#define SERVO_GARRA_DIREITA 1
#define SERVO_CACAMBA 2
#define SERVO_FUTABA 3

// Multiplicadores ajustaveis durante a calibracao.
#define DIRECAO_FE 1
#define DIRECAO_TE 1
#define DIRECAO_FD 1
#define DIRECAO_TD 1

// Motor frente esquerda (driver esquerdo, canal B).
#define FE_IN1 A1
#define FE_IN2 A0
#define FE_PWM 9

// Motor traseira esquerda (driver esquerdo, canal A).
#define TE_IN1 A2
#define TE_IN2 A3
#define TE_PWM 10

// Motor frente direita (driver direito, canal B).
#define FD_IN1 3
#define FD_IN2 2
#define FD_PWM 5

// Motor traseira direita (driver direito, canal A).
#define TD_IN1 4
#define TD_IN2 7
#define TD_PWM 6

#endif
