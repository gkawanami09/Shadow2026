#ifndef CONFIG_H
#define CONFIG_H

#define BAUD_RATE 115200
#define VELOCIDADE_MAXIMA_SEGURA 120
#define TIMEOUT_COMUNICACAO_MS 1000

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
