#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

const byte TAMANHO_COMANDO = 64;
char buffer_comando[TAMANHO_COMANDO];
byte tamanho_comando = 0;
unsigned long ultimo_comando_ms = 0;

// O Raspberry envia comandos de motor muitas vezes por segundo e, no loop de
// controle, nao le as respostas. Se o Arduino responder "OK LADO..." a cada
// frame, o buffer serial pode encher e os Serial.print/println podem bloquear.
// Mantemos respostas apenas para comandos de diagnostico/seguranca como PING,
// STATUS, PARAR e erros.
const bool RESPONDER_COMANDOS_CONTINUOS = false;

void configurar_pinos() {
  pinMode(FE_IN1, OUTPUT);
  pinMode(FE_IN2, OUTPUT);
  pinMode(FE_PWM, OUTPUT);
  pinMode(TE_IN1, OUTPUT);
  pinMode(TE_IN2, OUTPUT);
  pinMode(TE_PWM, OUTPUT);
  pinMode(FD_IN1, OUTPUT);
  pinMode(FD_IN2, OUTPUT);
  pinMode(FD_PWM, OUTPUT);
  pinMode(TD_IN1, OUTPUT);
  pinMode(TD_IN2, OUTPUT);
  pinMode(TD_PWM, OUTPUT);
}

int limitar_velocidade(int velocidade) {
  if (velocidade > VELOCIDADE_MAXIMA_SEGURA) return VELOCIDADE_MAXIMA_SEGURA;
  if (velocidade < -VELOCIDADE_MAXIMA_SEGURA) return -VELOCIDADE_MAXIMA_SEGURA;
  return velocidade;
}

void parar_motor(int pino_in1, int pino_in2, int pino_pwm) {
  digitalWrite(pino_in1, LOW);
  digitalWrite(pino_in2, LOW);
  analogWrite(pino_pwm, 0);
}

void controlar_motor(int pino_in1, int pino_in2, int pino_pwm, int velocidade, int multiplicador_direcao) {
  velocidade = limitar_velocidade(velocidade * multiplicador_direcao);

  if (velocidade > 0) {
    digitalWrite(pino_in1, HIGH);
    digitalWrite(pino_in2, LOW);
    analogWrite(pino_pwm, velocidade);
  } else if (velocidade < 0) {
    digitalWrite(pino_in1, LOW);
    digitalWrite(pino_in2, HIGH);
    analogWrite(pino_pwm, -velocidade);
  } else {
    parar_motor(pino_in1, pino_in2, pino_pwm);
  }
}

void parar_todos_motores() {
  parar_motor(FE_IN1, FE_IN2, FE_PWM);
  parar_motor(TE_IN1, TE_IN2, TE_PWM);
  parar_motor(FD_IN1, FD_IN2, FD_PWM);
  parar_motor(TD_IN1, TD_IN2, TD_PWM);
}

bool controlar_motor_por_nome(const char* nome_motor, int velocidade) {
  if (strcmp(nome_motor, "FE") == 0) {
    controlar_motor(FE_IN1, FE_IN2, FE_PWM, velocidade, DIRECAO_FE);
  } else if (strcmp(nome_motor, "TE") == 0) {
    controlar_motor(TE_IN1, TE_IN2, TE_PWM, velocidade, DIRECAO_TE);
  } else if (strcmp(nome_motor, "FD") == 0) {
    controlar_motor(FD_IN1, FD_IN2, FD_PWM, velocidade, DIRECAO_FD);
  } else if (strcmp(nome_motor, "TD") == 0) {
    controlar_motor(TD_IN1, TD_IN2, TD_PWM, velocidade, DIRECAO_TD);
  } else {
    return false;
  }
  return true;
}

void controlar_lados(int velocidade_esquerda, int velocidade_direita) {
  controlar_motor(FE_IN1, FE_IN2, FE_PWM, velocidade_esquerda, DIRECAO_FE);
  controlar_motor(TE_IN1, TE_IN2, TE_PWM, velocidade_esquerda, DIRECAO_TE);
  controlar_motor(FD_IN1, FD_IN2, FD_PWM, velocidade_direita, DIRECAO_FD);
  controlar_motor(TD_IN1, TD_IN2, TD_PWM, velocidade_direita, DIRECAO_TD);
}

void controlar_rodas(int vel_fe, int vel_te, int vel_fd, int vel_td) {
  controlar_motor(FE_IN1, FE_IN2, FE_PWM, vel_fe, DIRECAO_FE);
  controlar_motor(TE_IN1, TE_IN2, TE_PWM, vel_te, DIRECAO_TE);
  controlar_motor(FD_IN1, FD_IN2, FD_PWM, vel_fd, DIRECAO_FD);
  controlar_motor(TD_IN1, TD_IN2, TD_PWM, vel_td, DIRECAO_TD);
}

bool ler_inteiro(const char* texto, int* valor) {
  char* fim;
  long numero = strtol(texto, &fim, 10);
  if (*texto == '\0' || *fim != '\0' || numero < -255 || numero > 255) return false;
  *valor = (int)numero;
  return true;
}

void responder_ok_motor(const char* nome_motor, int velocidade) {
  if (!RESPONDER_COMANDOS_CONTINUOS) return;
  Serial.print("OK MOTOR ");
  Serial.print(nome_motor);
  Serial.print(" ");
  Serial.println(limitar_velocidade(velocidade));
}

void responder_ok_lado(int velocidade_esquerda, int velocidade_direita) {
  if (!RESPONDER_COMANDOS_CONTINUOS) return;
  Serial.print("OK LADO ");
  Serial.print(limitar_velocidade(velocidade_esquerda));
  Serial.print(" ");
  Serial.println(limitar_velocidade(velocidade_direita));
}

void responder_ok_rodas(int vel_fe, int vel_te, int vel_fd, int vel_td) {
  if (!RESPONDER_COMANDOS_CONTINUOS) return;
  Serial.print("OK RODAS ");
  Serial.print(limitar_velocidade(vel_fe));
  Serial.print(" ");
  Serial.print(limitar_velocidade(vel_te));
  Serial.print(" ");
  Serial.print(limitar_velocidade(vel_fd));
  Serial.print(" ");
  Serial.println(limitar_velocidade(vel_td));
}

void responder_ok_movimento(const char* tipo, int velocidade) {
  if (!RESPONDER_COMANDOS_CONTINUOS) return;
  Serial.print("OK ");
  Serial.print(tipo);
  Serial.print(" ");
  Serial.println(velocidade);
}

void processar_comando(char* comando) {
  char* tipo = strtok(comando, " \t");
  if (tipo == NULL) return;

  if (strcmp(tipo, "PING") == 0 && strtok(NULL, " \t") == NULL) {
    Serial.println("PONG");
    return;
  }
  if (strcmp(tipo, "PARAR") == 0 && strtok(NULL, " \t") == NULL) {
    parar_todos_motores();
    Serial.println("OK PARADO");
    return;
  }
  if (strcmp(tipo, "STATUS") == 0 && strtok(NULL, " \t") == NULL) {
    Serial.println("OK STATUS SPEC_01");
    return;
  }

  char* primeiro = strtok(NULL, " \t");
  char* segundo = strtok(NULL, " \t");
  char* terceiro = strtok(NULL, " \t");
  char* quarto = strtok(NULL, " \t");
  char* extra = strtok(NULL, " \t");
  int valor1, valor2, valor3, valor4;

  if (strcmp(tipo, "MOTOR") == 0) {
    if (primeiro == NULL || segundo == NULL || terceiro != NULL || !ler_inteiro(segundo, &valor1)) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else if (!controlar_motor_por_nome(primeiro, valor1)) {
      Serial.println("ERRO MOTOR_INVALIDO");
    } else {
      responder_ok_motor(primeiro, valor1);
    }
    return;
  }

  if (strcmp(tipo, "LADO") == 0) {
    if (primeiro == NULL || segundo == NULL || terceiro != NULL || !ler_inteiro(primeiro, &valor1) || !ler_inteiro(segundo, &valor2)) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else {
      controlar_lados(valor1, valor2);
      responder_ok_lado(valor1, valor2);
    }
    return;
  }

  if (strcmp(tipo, "RODAS") == 0) {
    if (primeiro == NULL || segundo == NULL || terceiro == NULL || quarto == NULL || extra != NULL ||
        !ler_inteiro(primeiro, &valor1) || !ler_inteiro(segundo, &valor2) || !ler_inteiro(terceiro, &valor3) || !ler_inteiro(quarto, &valor4)) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else {
      controlar_rodas(valor1, valor2, valor3, valor4);
      responder_ok_rodas(valor1, valor2, valor3, valor4);
    }
    return;
  }

  if (strcmp(tipo, "FRENTE") == 0 || strcmp(tipo, "TRAS") == 0 || strcmp(tipo, "GIRAR_ESQ") == 0 || strcmp(tipo, "GIRAR_DIR") == 0) {
    if (primeiro == NULL || segundo != NULL || !ler_inteiro(primeiro, &valor1)) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
      return;
    }
    valor1 = abs(limitar_velocidade(valor1));
    if (strcmp(tipo, "FRENTE") == 0) controlar_rodas(valor1, valor1, valor1, valor1);
    if (strcmp(tipo, "TRAS") == 0) controlar_rodas(-valor1, -valor1, -valor1, -valor1);
    if (strcmp(tipo, "GIRAR_ESQ") == 0) controlar_lados(-valor1, valor1);
    if (strcmp(tipo, "GIRAR_DIR") == 0) controlar_lados(valor1, -valor1);
    responder_ok_movimento(tipo, valor1);
    return;
  }

  Serial.println("ERRO COMANDO_INVALIDO");
}

void setup() {
  configurar_pinos();
  parar_todos_motores();
  Serial.begin(BAUD_RATE);
  ultimo_comando_ms = millis();
  Serial.println("Arduino pronto - SPEC 01");
}

void loop() {
  if (millis() - ultimo_comando_ms > TIMEOUT_COMUNICACAO_MS) {
    parar_todos_motores();
  }

  while (Serial.available() > 0) {
    char recebido = Serial.read();
    if (recebido == '\n') {
      buffer_comando[tamanho_comando] = '\0';
      if (tamanho_comando > 0) {
        ultimo_comando_ms = millis();
        processar_comando(buffer_comando);
      }
      tamanho_comando = 0;
    } else if (recebido != '\r' && tamanho_comando < TAMANHO_COMANDO - 1) {
      buffer_comando[tamanho_comando++] = recebido;
    }
  }
}
