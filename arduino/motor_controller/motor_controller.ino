#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <Wire.h>

#include "config.h"

const byte TAMANHO_COMANDO = 64;
char buffer_comando[TAMANHO_COMANDO];
byte tamanho_comando = 0;
unsigned long ultimo_comando_ms = 0;

enum ModoLed {
  LED_APAGADO,
  LED_ACESO
};

ModoLed modo_led = LED_ACESO;

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

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  pinMode(ULTRASSOM_TRIG_PIN, OUTPUT);
  pinMode(ULTRASSOM_ECHO_PIN, INPUT);
  digitalWrite(ULTRASSOM_TRIG_PIN, LOW);
}

void escrever_pca9685(byte registrador, byte valor) {
  Wire.beginTransmission(PCA9685_ENDERECO);
  Wire.write(registrador);
  Wire.write(valor);
  Wire.endTransmission();
}

void configurar_pca9685() {
  Wire.begin();

  // Coloca o PCA9685 em sleep para configurar 50 Hz e depois o reativa.
  escrever_pca9685(0x00, 0x10);
  byte prescale = (byte)(25000000UL / (4096UL * PCA9685_FREQUENCIA_HZ) - 1UL);
  escrever_pca9685(0xFE, prescale);
  escrever_pca9685(0x00, 0x20);
  delay(1);
  escrever_pca9685(0x00, 0xA0);

  // Nao movimenta os servos durante o boot. Cada canal e ativado somente
  // quando a Raspberry enviar o primeiro comando SERVO correspondente.
  for (byte canal = 0; canal < 4; canal++) {
    byte base = 0x06 + 4 * canal;
    escrever_pca9685(base, 0);
    escrever_pca9685(base + 1, 0);
    escrever_pca9685(base + 2, 0);
    escrever_pca9685(base + 3, 0x10);
  }
}

void definir_servo(byte canal, int angulo) {
  long pulso_us = map(angulo, 0, 180, SERVO_PULSO_MIN_US, SERVO_PULSO_MAX_US);
  unsigned int contador = (unsigned int)(pulso_us * 4096L / 20000L);
  byte base = 0x06 + 4 * canal;

  escrever_pca9685(base, 0);
  escrever_pca9685(base + 1, 0);
  escrever_pca9685(base + 2, contador & 0xFF);
  escrever_pca9685(base + 3, (contador >> 8) & 0x0F);
}

bool canal_servo_por_nome(const char* nome, byte* canal) {
  if (strcmp(nome, "GARRA_ESQ") == 0 || strcmp(nome, "CH0") == 0) {
    *canal = SERVO_GARRA_ESQUERDA;
  } else if (strcmp(nome, "GARRA_DIR") == 0 || strcmp(nome, "CH1") == 0) {
    *canal = SERVO_GARRA_DIREITA;
  } else if (strcmp(nome, "CACAMBA") == 0 || strcmp(nome, "CH2") == 0) {
    *canal = SERVO_CACAMBA;
  } else if (strcmp(nome, "FUTABA") == 0 || strcmp(nome, "CH3") == 0) {
    *canal = SERVO_FUTABA;
  } else {
    return false;
  }
  return true;
}

void definir_modo_led(ModoLed novo_modo) {
  modo_led = novo_modo;
  digitalWrite(LED_PIN, modo_led == LED_ACESO ? HIGH : LOW);
}

long medir_distancia_mm() {
  digitalWrite(ULTRASSOM_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASSOM_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASSOM_TRIG_PIN, LOW);

  unsigned long duracao = pulseIn(
    ULTRASSOM_ECHO_PIN,
    HIGH,
    ULTRASSOM_TIMEOUT_US
  );
  if (duracao == 0) return -1;

  // Velocidade do som: aproximadamente 0,343 mm/us; ida e volta.
  return (long)(duracao * 343UL / 2000UL);
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
  Serial.print("OK MOTOR ");
  Serial.print(nome_motor);
  Serial.print(" ");
  Serial.println(limitar_velocidade(velocidade));
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

  if (strcmp(tipo, "SERVO") == 0) {
    byte canal;
    if (primeiro == NULL || segundo == NULL || terceiro != NULL ||
        !ler_inteiro(segundo, &valor1) || valor1 < 0 || valor1 > 180) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else if (!canal_servo_por_nome(primeiro, &canal)) {
      Serial.println("ERRO SERVO_INVALIDO");
    } else {
      definir_servo(canal, valor1);
      Serial.print("OK SERVO "); Serial.print(primeiro); Serial.print(" "); Serial.println(valor1);
    }
    return;
  }

  if (strcmp(tipo, "LED") == 0) {
    if (primeiro == NULL || segundo != NULL) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else if (strcmp(primeiro, "APAGADO") == 0) {
      definir_modo_led(LED_APAGADO);
      Serial.println("OK LED APAGADO");
    } else if (strcmp(primeiro, "ACESO") == 0) {
      definir_modo_led(LED_ACESO);
      Serial.println("OK LED ACESO");
    } else {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    }
    return;
  }

  if (strcmp(tipo, "ULTRASSOM") == 0) {
    if (primeiro != NULL) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else {
      Serial.print("OK ULTRASSOM ");
      Serial.println(medir_distancia_mm());
    }
    return;
  }

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
      Serial.print("OK LADO "); Serial.print(limitar_velocidade(valor1)); Serial.print(" "); Serial.println(limitar_velocidade(valor2));
    }
    return;
  }

  if (strcmp(tipo, "RODAS") == 0) {
    if (primeiro == NULL || segundo == NULL || terceiro == NULL || quarto == NULL || extra != NULL ||
        !ler_inteiro(primeiro, &valor1) || !ler_inteiro(segundo, &valor2) || !ler_inteiro(terceiro, &valor3) || !ler_inteiro(quarto, &valor4)) {
      Serial.println("ERRO PARAMETROS_INVALIDOS");
    } else {
      controlar_rodas(valor1, valor2, valor3, valor4);
      Serial.print("OK RODAS "); Serial.print(limitar_velocidade(valor1)); Serial.print(" "); Serial.print(limitar_velocidade(valor2)); Serial.print(" "); Serial.print(limitar_velocidade(valor3)); Serial.print(" "); Serial.println(limitar_velocidade(valor4));
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
    Serial.print("OK "); Serial.print(tipo); Serial.print(" "); Serial.println(valor1);
    return;
  }

  Serial.println("ERRO COMANDO_INVALIDO");
}

void setup() {
  configurar_pinos();
  configurar_pca9685();
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
