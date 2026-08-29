#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include "config.h"
// ======================================================
// COMUNICACAO SERIAL
// ======================================================
const byte TAMANHO_COMANDO = 64;
char buffer_comando[TAMANHO_COMANDO];
byte tamanho_comando = 0;
unsigned long ultimo_comando_ms = 0;
// SERVOS
// ======================================================
int posicao_servo_atual[5] = {
  SERVO_POSICAO_INICIAL_GARRA_ESQ,
  SERVO_POSICAO_INICIAL_GARRA_DIR,
  SERVO_POSICAO_INICIAL_CACAMBA,
  0,
  0
};
// ======================================================
// FUTABA
// ======================================================
bool futaba_ativo = false;
int futaba_potencia_atual = 0;
unsigned long futaba_desligar_em_ms = 0;
// ======================================================
// MPU6050
// ======================================================
MPU6050 mpu(MPU6050_ENDERECO);
bool mpu_disponivel = false;
// Valores RAW do acelerometro.
int16_t mpu_ax_raw = 0;
int16_t mpu_ay_raw = 0;
int16_t mpu_az_raw = 0;
// Valores RAW do giroscopio.
int16_t mpu_gx_raw = 0;
int16_t mpu_gy_raw = 0;
int16_t mpu_gz_raw = 0;
// Bias/erro do giroscopio.
float mpu_bias_gx = 0.0f;
float mpu_bias_gy = 0.0f;
float mpu_bias_gz = 0.0f;
// Aceleracao convertida para g.
float mpu_ax_g = 0.0f;
float mpu_ay_g = 0.0f;
float mpu_az_g = 0.0f;
// Velocidade angular em graus/s.
float mpu_gx_dps = 0.0f;
float mpu_gy_dps = 0.0f;
float mpu_gz_dps = 0.0f;
// Angulos internos do filtro complementar.
float mpu_pitch_filtrado = 0.0f;
float mpu_roll_filtrado = 0.0f;
// Referencia correspondente ao robo no piso plano.
float mpu_referencia_pitch = 0.0f;
float mpu_referencia_roll = 0.0f;
// Angulos disponibilizados para Raspberry.
float mpu_pitch_graus = 0.0f;
float mpu_roll_graus = 0.0f;
float mpu_yaw_graus = 0.0f;
// Controle de tempo da leitura.
unsigned long mpu_ultima_leitura_us = 0;
// ======================================================
// ESTADO DA RAMPA
// ======================================================
enum EstadoRampa {
  RAMPA_PLANO,
  RAMPA_SUBINDO,
  RAMPA_DESCENDO
};
EstadoRampa estado_rampa = RAMPA_PLANO;
// ======================================================
// PROTOTIPOS
// ======================================================
void escrever_pca9685(byte registrador, byte valor);
void definir_servo(byte canal, int angulo);
void atualizar_mpu();
// ======================================================
// PCA9685
// ======================================================
void desligar_canal_pca9685(byte canal) {
  byte base = 0x06 + 4 * canal;
  escrever_pca9685(base, 0);
  escrever_pca9685(base + 1, 0);
  escrever_pca9685(base + 2, 0);
  escrever_pca9685(base + 3, 0x10);
}
// ======================================================
// CONFIGURACAO DOS PINOS
// ======================================================
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
  pinMode(
    ULTRASSOM_FRENTE_TRIG_PIN,
    OUTPUT
  );
  pinMode(
    ULTRASSOM_FRENTE_ECHO_PIN,
    INPUT
  );
  digitalWrite(
    ULTRASSOM_FRENTE_TRIG_PIN,
    LOW
  );
  pinMode(
    ULTRASSOM_LATERAL_TRIG_PIN,
    OUTPUT
  );
  pinMode(
    ULTRASSOM_LATERAL_ECHO_PIN,
    INPUT
  );
  digitalWrite(
    ULTRASSOM_LATERAL_TRIG_PIN,
    LOW
  );
}
// ======================================================
// ESCRITA NO PCA9685
// ======================================================
void escrever_pca9685(
  byte registrador,
  byte valor
) {
  Wire.beginTransmission(
    PCA9685_ENDERECO
  );
  Wire.write(
    registrador
  );
  Wire.write(
    valor
  );
  Wire.endTransmission();
}
// ======================================================
// CONFIGURAR PCA9685
// ======================================================
void configurar_pca9685() {
  // Inicializa o barramento I2C.
  //
  // A4 = SDA
  // A5 = SCL
  //
  // O MPU6050 utilizara exatamente esse mesmo
  // barramento.
  Wire.begin();
  // Coloca o PCA9685 em sleep para configurar 50 Hz.
  escrever_pca9685(
    0x00,
    0x10
  );
  byte prescale =
    (byte)(
      25000000UL /
      (
        4096UL *
        PCA9685_FREQUENCIA_HZ
      ) -
      1UL
    );
  escrever_pca9685(
    0xFE,
    prescale
  );
  escrever_pca9685(
    0x00,
    0x20
  );
  delay(1);
  escrever_pca9685(
    0x00,
    0xA0
  );
  // Primeiro desliga os cinco primeiros canais.
  //
  // Futaba CH4 permanece desligado ate receber um comando FUTABA.
  for (
    byte canal = 0;
    canal <= SERVO_FUTABA;
    canal++
  ) {
    desligar_canal_pca9685(
      canal
    );
  }
  // Posicoes iniciais originais.
  definir_servo(
    SERVO_GARRA_ESQUERDA,
    SERVO_POSICAO_INICIAL_GARRA_ESQ
  );
  definir_servo(
    SERVO_GARRA_DIREITA,
    SERVO_POSICAO_INICIAL_GARRA_DIR
  );
  definir_servo(
    SERVO_CACAMBA,
    SERVO_POSICAO_INICIAL_CACAMBA
  );
}
// ======================================================
// SERVO POR ANGULO
// ======================================================
void definir_servo(
  byte canal,
  int angulo
) {
  long pulso_us =
    map(
      angulo,
      0,
      180,
      SERVO_PULSO_MIN_US,
      SERVO_PULSO_MAX_US
    );
  unsigned int contador =
    (unsigned int)(
      pulso_us *
      4096L /
      20000L
    );
  byte base =
    0x06 +
    4 * canal;
  escrever_pca9685(
    base,
    0
  );
  escrever_pca9685(
    base + 1,
    0
  );
  escrever_pca9685(
    base + 2,
    contador & 0xFF
  );
  escrever_pca9685(
    base + 3,
    (contador >> 8) & 0x0F
  );
}
// ======================================================
// SERVO POR PULSO
// ======================================================
void definir_servo_pulso(
  byte canal,
  int pulso_us
) {
  unsigned int contador =
    (unsigned int)(
      (long)pulso_us *
      4096L /
      20000L
    );
  byte base =
    0x06 +
    4 * canal;
  escrever_pca9685(
    base,
    0
  );
  escrever_pca9685(
    base + 1,
    0
  );
  escrever_pca9685(
    base + 2,
    contador & 0xFF
  );
  escrever_pca9685(
    base + 3,
    (contador >> 8) & 0x0F
  );
}
// ======================================================
// FUTABA
// ======================================================
void parar_futaba() {
  desligar_canal_pca9685(
    SERVO_FUTABA
  );
  futaba_ativo = false;
  futaba_potencia_atual = 0;
  futaba_desligar_em_ms = 0;
}
void acionar_futaba(
  int potencia,
  unsigned long tempo_ms
) {
  potencia =
    constrain(
      potencia,
      -100,
      100
    );
  int magnitude =
    abs(
      potencia
    );
  int desvio_us =
    map(
      magnitude,
      1,
      100,
      FUTABA_DESVIO_MIN_US,
      FUTABA_DESVIO_MAX_US
    );
  int pulso_us =
    FUTABA_PULSO_NEUTRO_US +
    (
      potencia > 0 ?
      desvio_us :
      -desvio_us
    );
  definir_servo_pulso(
    SERVO_FUTABA,
    pulso_us
  );
  futaba_ativo = true;
  futaba_potencia_atual =
    potencia;
  futaba_desligar_em_ms =
    millis() +
    tempo_ms;
}
void atualizar_futaba() {
  if (
    futaba_ativo &&
    (long)(
      millis() -
      futaba_desligar_em_ms
    ) >= 0
  ) {
    parar_futaba();
  }
}
// ======================================================
// SERVO RELATIVO
// ======================================================
int mover_servo_relativo(
  byte canal,
  int deslocamento
) {
  int destino =
    posicao_servo_atual[canal] +
    deslocamento;
  destino =
    constrain(
      destino,
      SERVO_ANGULO_MIN,
      SERVO_ANGULO_MAX
    );
  definir_servo(
    canal,
    destino
  );
  posicao_servo_atual[canal] =
    destino;
  return destino;
}
// ======================================================
// IDENTIFICACAO DOS SERVOS
// ======================================================
bool canal_servo_por_nome(
  const char* nome,
  byte* canal
) {
  if (
    strcmp(
      nome,
      "GARRA_ESQ"
    ) == 0 ||
    strcmp(
      nome,
      "CH0"
    ) == 0
  ) {
    *canal =
      SERVO_GARRA_ESQUERDA;
  } else if (
    strcmp(
      nome,
      "GARRA_DIR"
    ) == 0 ||
    strcmp(
      nome,
      "CH1"
    ) == 0
  ) {
    *canal =
      SERVO_GARRA_DIREITA;
  } else if (
    strcmp(
      nome,
      "CACAMBA"
    ) == 0 ||
    strcmp(
      nome,
      "CH2"
    ) == 0
  ) {
    *canal =
      SERVO_CACAMBA;
  } else if (
    strcmp(
      nome,
      "FUTABA"
    ) == 0 ||
    strcmp(
      nome,
      "CH4"
    ) == 0
  ) {
    *canal =
      SERVO_FUTABA;
  } else {
    return false;
  }
  return true;
}
// ULTRASSOM
// ======================================================
long medir_distancia_mm(
  byte pino_trig,
  byte pino_echo
) {
  digitalWrite(
    pino_trig,
    LOW
  );
  delayMicroseconds(2);
  digitalWrite(
    pino_trig,
    HIGH
  );
  delayMicroseconds(10);
  digitalWrite(
    pino_trig,
    LOW
  );
  unsigned long duracao =
    pulseIn(
      pino_echo,
      HIGH,
      ULTRASSOM_TIMEOUT_US
    );
  if (
    duracao == 0
  ) {
    return -1;
  }
  // Velocidade do som:
  // aproximadamente 0,343 mm/us.
  //
  // Divide por 2 por causa da ida e volta.
  return
    (long)(
      duracao *
      343UL /
      2000UL
    );
}
// ======================================================
// MOTORES
// ======================================================
int velocidade_anterior_fe = 0;
int velocidade_anterior_te = 0;
int velocidade_anterior_fd = 0;
int velocidade_anterior_td = 0;

bool inverte_sentido(
  int velocidade_anterior,
  int velocidade_nova
) {
  return
    (velocidade_anterior > 0 && velocidade_nova < 0) ||
    (velocidade_anterior < 0 && velocidade_nova > 0);
}

int limitar_velocidade(
  int velocidade
) {
  if (
    velocidade >
    VELOCIDADE_MAXIMA_SEGURA
  ) {
    return
      VELOCIDADE_MAXIMA_SEGURA;
  }
  if (
    velocidade <
    -VELOCIDADE_MAXIMA_SEGURA
  ) {
    return
      -VELOCIDADE_MAXIMA_SEGURA;
  }
  return velocidade;
}
void parar_motor(
  int pino_in1,
  int pino_in2,
  int pino_pwm
) {
  // O PWM deve cair antes da mudança dos pinos da ponte H.
  analogWrite(
    pino_pwm,
    0
  );
  digitalWrite(
    pino_in1,
    LOW
  );
  digitalWrite(
    pino_in2,
    LOW
  );
}
void controlar_motor(
  int pino_in1,
  int pino_in2,
  int pino_pwm,
  int velocidade,
  int multiplicador_direcao
) {
  velocidade =
    limitar_velocidade(
      velocidade *
      multiplicador_direcao
    );
  // Retira o PWM antes de tocar nos pinos de direção. Sem isso, o valor
  // anterior continuava ativo durante os digitalWrite de uma reversão.
  analogWrite(
    pino_pwm,
    0
  );
  if (
    velocidade > 0
  ) {
    digitalWrite(
      pino_in1,
      HIGH
    );
    digitalWrite(
      pino_in2,
      LOW
    );
    analogWrite(
      pino_pwm,
      velocidade
    );
  } else if (
    velocidade < 0
  ) {
    digitalWrite(
      pino_in1,
      LOW
    );
    digitalWrite(
      pino_in2,
      HIGH
    );
    analogWrite(
      pino_pwm,
      -velocidade
    );
  } else {
    parar_motor(
      pino_in1,
      pino_in2,
      pino_pwm
    );
  }
}
void parar_todos_motores() {
  parar_motor(
    FE_IN1,
    FE_IN2,
    FE_PWM
  );
  parar_motor(
    TE_IN1,
    TE_IN2,
    TE_PWM
  );
  parar_motor(
    FD_IN1,
    FD_IN2,
    FD_PWM
  );
  parar_motor(
    TD_IN1,
    TD_IN2,
    TD_PWM
  );
  velocidade_anterior_fe = 0;
  velocidade_anterior_te = 0;
  velocidade_anterior_fd = 0;
  velocidade_anterior_td = 0;
}

void preparar_reversoes(
  int nova_fe,
  int nova_te,
  int nova_fd,
  int nova_td
) {
  bool houve_reversao = false;
  if (inverte_sentido(velocidade_anterior_fe, nova_fe)) {
    parar_motor(FE_IN1, FE_IN2, FE_PWM);
    houve_reversao = true;
  }
  if (inverte_sentido(velocidade_anterior_te, nova_te)) {
    parar_motor(TE_IN1, TE_IN2, TE_PWM);
    houve_reversao = true;
  }
  if (inverte_sentido(velocidade_anterior_fd, nova_fd)) {
    parar_motor(FD_IN1, FD_IN2, FD_PWM);
    houve_reversao = true;
  }
  if (inverte_sentido(velocidade_anterior_td, nova_td)) {
    parar_motor(TD_IN1, TD_IN2, TD_PWM);
    houve_reversao = true;
  }
  if (houve_reversao) {
    // Todas as rodas que inverterão esperam juntas: o comando inteiro ganha
    // somente 2 ms, em vez de somar um atraso para cada motor.
    delayMicroseconds(TEMPO_MORTO_REVERSAO_US);
  }
}

void registrar_velocidades(
  int nova_fe,
  int nova_te,
  int nova_fd,
  int nova_td
) {
  velocidade_anterior_fe = nova_fe;
  velocidade_anterior_te = nova_te;
  velocidade_anterior_fd = nova_fd;
  velocidade_anterior_td = nova_td;
}

void controlar_motor_com_reversao(
  int pino_in1,
  int pino_in2,
  int pino_pwm,
  int velocidade,
  int multiplicador_direcao,
  int* velocidade_anterior
) {
  int velocidade_nova = limitar_velocidade(
    velocidade * multiplicador_direcao
  );
  if (inverte_sentido(*velocidade_anterior, velocidade_nova)) {
    parar_motor(pino_in1, pino_in2, pino_pwm);
    delayMicroseconds(TEMPO_MORTO_REVERSAO_US);
  }
  controlar_motor(
    pino_in1,
    pino_in2,
    pino_pwm,
    velocidade,
    multiplicador_direcao
  );
  *velocidade_anterior = velocidade_nova;
}

bool controlar_motor_por_nome(
  const char* nome_motor,
  int velocidade
) {
  if (
    strcmp(
      nome_motor,
      "FE"
    ) == 0
  ) {
    controlar_motor_com_reversao(
      FE_IN1,
      FE_IN2,
      FE_PWM,
      velocidade,
      DIRECAO_FE,
      &velocidade_anterior_fe
    );
  } else if (
    strcmp(
      nome_motor,
      "TE"
    ) == 0
  ) {
    controlar_motor_com_reversao(
      TE_IN1,
      TE_IN2,
      TE_PWM,
      velocidade,
      DIRECAO_TE,
      &velocidade_anterior_te
    );
  } else if (
    strcmp(
      nome_motor,
      "FD"
    ) == 0
  ) {
    controlar_motor_com_reversao(
      FD_IN1,
      FD_IN2,
      FD_PWM,
      velocidade,
      DIRECAO_FD,
      &velocidade_anterior_fd
    );
  } else if (
    strcmp(
      nome_motor,
      "TD"
    ) == 0
  ) {
    controlar_motor_com_reversao(
      TD_IN1,
      TD_IN2,
      TD_PWM,
      velocidade,
      DIRECAO_TD,
      &velocidade_anterior_td
    );
  } else {
    return false;
  }
  return true;
}
void controlar_lados(
  int velocidade_esquerda,
  int velocidade_direita
) {
  int nova_fe = limitar_velocidade(velocidade_esquerda * DIRECAO_FE);
  int nova_te = limitar_velocidade(velocidade_esquerda * DIRECAO_TE);
  int nova_fd = limitar_velocidade(velocidade_direita * DIRECAO_FD);
  int nova_td = limitar_velocidade(velocidade_direita * DIRECAO_TD);
  preparar_reversoes(nova_fe, nova_te, nova_fd, nova_td);

  controlar_motor(
    FE_IN1,
    FE_IN2,
    FE_PWM,
    velocidade_esquerda,
    DIRECAO_FE
  );
  controlar_motor(
    TE_IN1,
    TE_IN2,
    TE_PWM,
    velocidade_esquerda,
    DIRECAO_TE
  );
  controlar_motor(
    FD_IN1,
    FD_IN2,
    FD_PWM,
    velocidade_direita,
    DIRECAO_FD
  );
  controlar_motor(
    TD_IN1,
    TD_IN2,
    TD_PWM,
    velocidade_direita,
    DIRECAO_TD
  );
  registrar_velocidades(nova_fe, nova_te, nova_fd, nova_td);
}
void controlar_rodas(
  int vel_fe,
  int vel_te,
  int vel_fd,
  int vel_td
) {
  int nova_fe = limitar_velocidade(vel_fe * DIRECAO_FE);
  int nova_te = limitar_velocidade(vel_te * DIRECAO_TE);
  int nova_fd = limitar_velocidade(vel_fd * DIRECAO_FD);
  int nova_td = limitar_velocidade(vel_td * DIRECAO_TD);
  preparar_reversoes(nova_fe, nova_te, nova_fd, nova_td);

  controlar_motor(
    FE_IN1,
    FE_IN2,
    FE_PWM,
    vel_fe,
    DIRECAO_FE
  );
  controlar_motor(
    TE_IN1,
    TE_IN2,
    TE_PWM,
    vel_te,
    DIRECAO_TE
  );
  controlar_motor(
    FD_IN1,
    FD_IN2,
    FD_PWM,
    vel_fd,
    DIRECAO_FD
  );
  controlar_motor(
    TD_IN1,
    TD_IN2,
    TD_PWM,
    vel_td,
    DIRECAO_TD
  );
  registrar_velocidades(nova_fe, nova_te, nova_fd, nova_td);
}
// ======================================================
// MPU6050 - CALIBRAR GIROSCOPIO
// ======================================================
void calibrar_giroscopio_mpu() {
  if (
    !mpu_disponivel
  ) {
    return;
  }
  long soma_gx = 0;
  long soma_gy = 0;
  long soma_gz = 0;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
  // Durante essa etapa o robo deve estar parado.
  for (
    int i = 0;
    i < MPU_AMOSTRAS_CALIBRACAO;
    i++
  ) {
    mpu.getMotion6(
      &ax,
      &ay,
      &az,
      &gx,
      &gy,
      &gz
    );
    soma_gx += gx;
    soma_gy += gy;
    soma_gz += gz;
    delay(3);
  }
  mpu_bias_gx =
    (float)soma_gx /
    MPU_AMOSTRAS_CALIBRACAO;
  mpu_bias_gy =
    (float)soma_gy /
    MPU_AMOSTRAS_CALIBRACAO;
  mpu_bias_gz =
    (float)soma_gz /
    MPU_AMOSTRAS_CALIBRACAO;
}
// ======================================================
// MPU6050 - ORIENTACAO INICIAL
// ======================================================
void inicializar_orientacao_mpu() {
  if (
    !mpu_disponivel
  ) {
    return;
  }
  const byte amostras = 20;
  float soma_pitch = 0.0f;
  float soma_roll = 0.0f;
  for (
    byte i = 0;
    i < amostras;
    i++
  ) {
    mpu.getMotion6(
      &mpu_ax_raw,
      &mpu_ay_raw,
      &mpu_az_raw,
      &mpu_gx_raw,
      &mpu_gy_raw,
      &mpu_gz_raw
    );
    float ax =
      (float)mpu_ax_raw;
    float ay =
      (float)mpu_ay_raw;
    float az =
      (float)mpu_az_raw;
    // ----------------------------------------------
    // Orientacao fisica adotada no Shadow:
    //
    // Y = frente / tras
    // X = esquerda / direita
    // Z = vertical
    //
    // Pitch = inclinacao frente/traseira.
    // Roll = inclinacao lateral.
    // ----------------------------------------------
    float pitch =
      atan2(
        ay,
        sqrt(
          ax * ax +
          az * az
        )
      ) *
      RAD_TO_DEG;
    float roll =
      atan2(
        -ax,
        sqrt(
          ay * ay +
          az * az
        )
      ) *
      RAD_TO_DEG;
    soma_pitch +=
      pitch;
    soma_roll +=
      roll;
    delay(3);
  }
  mpu_pitch_filtrado =
    soma_pitch /
    amostras;
  mpu_roll_filtrado =
    soma_roll /
    amostras;
#if MPU_ZERO_AUTOMATICO_INICIAL
  mpu_referencia_pitch =
    mpu_pitch_filtrado;
  mpu_referencia_roll =
    mpu_roll_filtrado;
#else
  mpu_referencia_pitch =
    0.0f;
  mpu_referencia_roll =
    0.0f;
#endif
  mpu_pitch_graus =
    0.0f;
  mpu_roll_graus =
    0.0f;
  mpu_yaw_graus =
    0.0f;
  estado_rampa =
    RAMPA_PLANO;
  mpu_ultima_leitura_us =
    micros();
}
// ======================================================
// MPU6050 - ZERAR REFERENCIA
// ======================================================
void zerar_referencia_mpu() {
  if (
    !mpu_disponivel
  ) {
    return;
  }
  mpu_referencia_pitch =
    mpu_pitch_filtrado;
  mpu_referencia_roll =
    mpu_roll_filtrado;
  mpu_pitch_graus =
    0.0f;
  mpu_roll_graus =
    0.0f;
  mpu_yaw_graus =
    0.0f;
  estado_rampa =
    RAMPA_PLANO;
}
// ======================================================
// MPU6050 - ESTADO DA RAMPA
// ======================================================
const char* nome_estado_rampa() {
  if (
    estado_rampa ==
    RAMPA_SUBINDO
  ) {
    return
      "SUBINDO";
  }
  if (
    estado_rampa ==
    RAMPA_DESCENDO
  ) {
    return
      "DESCENDO";
  }
  return
    "PLANO";
}
void atualizar_estado_rampa() {
  float pitch =
    mpu_pitch_graus;
  // Entrando em subida.
  if (
    pitch >=
    RAMPA_ANGULO_ENTRADA_GRAUS
  ) {
    estado_rampa =
      RAMPA_SUBINDO;
    return;
  }
  // Entrando em descida.
  if (
    pitch <=
    -RAMPA_ANGULO_ENTRADA_GRAUS
  ) {
    estado_rampa =
      RAMPA_DESCENDO;
    return;
  }
  // Para voltar a PLANO precisa ficar abaixo
  // do limite menor.
  if (
    fabs(
      pitch
    ) <=
    RAMPA_ANGULO_SAIDA_GRAUS
  ) {
    estado_rampa =
      RAMPA_PLANO;
  }
}
// ======================================================
// MPU6050 - ATUALIZACAO CONTINUA
// ======================================================
void atualizar_mpu() {
  if (
    !mpu_disponivel
  ) {
    return;
  }
  unsigned long agora_us =
    micros();
  unsigned long intervalo_us =
    agora_us -
    mpu_ultima_leitura_us;
  if (
    intervalo_us <
    MPU_INTERVALO_LEITURA_US
  ) {
    return;
  }
  float dt =
    intervalo_us /
    1000000.0f;
  // Evita integrar um intervalo muito grande
  // caso o programa tenha ficado bloqueado por algum
  // comando, pulseIn, servo etc.
  if (
    dt <= 0.0f ||
    dt > 0.1f
  ) {
    dt =
      MPU_INTERVALO_LEITURA_US /
      1000000.0f;
  }
  mpu_ultima_leitura_us =
    agora_us;
  // --------------------------------------------------
  // LEITURA RAW
  // --------------------------------------------------
  mpu.getMotion6(
    &mpu_ax_raw,
    &mpu_ay_raw,
    &mpu_az_raw,
    &mpu_gx_raw,
    &mpu_gy_raw,
    &mpu_gz_raw
  );
  // --------------------------------------------------
  // CONVERTER ACELEROMETRO PARA g
  // --------------------------------------------------
  mpu_ax_g =
    (float)mpu_ax_raw /
    MPU_ACEL_LSB_POR_G;
  mpu_ay_g =
    (float)mpu_ay_raw /
    MPU_ACEL_LSB_POR_G;
  mpu_az_g =
    (float)mpu_az_raw /
    MPU_ACEL_LSB_POR_G;
  // --------------------------------------------------
  // CONVERTER GIROSCOPIO PARA graus/s
  // --------------------------------------------------
  mpu_gx_dps =
    (
      (float)mpu_gx_raw -
      mpu_bias_gx
    ) /
    MPU_GIRO_LSB_POR_DPS;
  mpu_gy_dps =
    (
      (float)mpu_gy_raw -
      mpu_bias_gy
    ) /
    MPU_GIRO_LSB_POR_DPS;
  mpu_gz_dps =
    (
      (float)mpu_gz_raw -
      mpu_bias_gz
    ) /
    MPU_GIRO_LSB_POR_DPS;
  // --------------------------------------------------
  // ANGULOS CALCULADOS PELO ACELEROMETRO
  //
  // Montagem do Shadow:
  //
  // Y = frente / tras
  // X = esquerda / direita
  // Z = vertical
  //
  // Por isso:
  //
  // PITCH usa Y.
  // ROLL usa X.
  // --------------------------------------------------
  float pitch_acelerometro =
    atan2(
      mpu_ay_g,
      sqrt(
        mpu_ax_g *
        mpu_ax_g +
        mpu_az_g *
        mpu_az_g
      )
    ) *
    RAD_TO_DEG;
  float roll_acelerometro =
    atan2(
      -mpu_ax_g,
      sqrt(
        mpu_ay_g *
        mpu_ay_g +
        mpu_az_g *
        mpu_az_g
      )
    ) *
    RAD_TO_DEG;
  // --------------------------------------------------
  // FILTRO COMPLEMENTAR
  //
  // Pitch frente/tras:
  // rotacao no eixo X.
  //
  // Roll esquerda/direita:
  // rotacao no eixo Y.
  // --------------------------------------------------
  float alpha =
    MPU_FILTRO_COMPLEMENTAR;
  mpu_pitch_filtrado =
    alpha *
    (
      mpu_pitch_filtrado +
      mpu_gx_dps *
      dt
    ) +
    (
      1.0f -
      alpha
    ) *
    pitch_acelerometro;
  mpu_roll_filtrado =
    alpha *
    (
      mpu_roll_filtrado +
      mpu_gy_dps *
      dt
    ) +
    (
      1.0f -
      alpha
    ) *
    roll_acelerometro;
  // --------------------------------------------------
  // YAW
  //
  // MPU6050 NAO possui magnetometro.
  //
  // Portanto o YAW e relativo e acumulara drift.
  // --------------------------------------------------
  mpu_yaw_graus +=
    mpu_gz_dps *
    dt;
  // --------------------------------------------------
  // VALORES RELATIVOS AO PISO PLANO
  // --------------------------------------------------
  mpu_pitch_graus =
    (
      mpu_pitch_filtrado -
      mpu_referencia_pitch
    ) *
    MPU_SINAL_RAMPA;
  mpu_roll_graus =
    mpu_roll_filtrado -
    mpu_referencia_roll;
  atualizar_estado_rampa();
}
// ======================================================
// CONFIGURAR MPU6050
// ======================================================
void configurar_mpu() {
  // O Wire ja foi inicializado pelo PCA9685.
  mpu.initialize();
  mpu_disponivel =
    mpu.testConnection();
  // Se o sensor nao responder, simplesmente retorna.
  //
  // Isso permite que TODO o restante do robo continue
  // funcionando normalmente mesmo sem o MPU.
  if (
    !mpu_disponivel
  ) {
    return;
  }
  // Acelerometro +-2 g.
  mpu.setFullScaleAccelRange(
    MPU6050_ACCEL_FS_2
  );
  // Giroscopio +-250 graus/s.
  mpu.setFullScaleGyroRange(
    MPU6050_GYRO_FS_250
  );
  // Filtro digital interno de aproximadamente 20 Hz.
  //
  // Ajuda bastante com vibracao dos motores.
  mpu.setDLPFMode(
    MPU6050_DLPF_BW_20
  );
  // Com o DLPF ativo a base e 1 kHz.
  //
  // 1000 / (1 + 9)
  // = 100 Hz.
  mpu.setRate(9);
  delay(50);
  calibrar_giroscopio_mpu();
  inicializar_orientacao_mpu();
}
// ======================================================
// RESPOSTA COMPLETA DO MPU
// ======================================================
void responder_mpu() {
  if (
    !mpu_disponivel
  ) {
    Serial.println(
      "ERRO MPU_INDISPONIVEL"
    );
    return;
  }
  atualizar_mpu();
  int16_t temperatura_raw =
    mpu.getTemperature();
  float temperatura_c =
    temperatura_raw /
    340.0f +
    36.53f;
  Serial.print(
    "OK MPU"
  );
  Serial.print(
    " PITCH="
  );
  Serial.print(
    mpu_pitch_graus,
    2
  );
  Serial.print(
    " ROLL="
  );
  Serial.print(
    mpu_roll_graus,
    2
  );
  Serial.print(
    " YAW="
  );
  Serial.print(
    mpu_yaw_graus,
    2
  );
  Serial.print(
    " AX="
  );
  Serial.print(
    mpu_ax_g,
    3
  );
  Serial.print(
    " AY="
  );
  Serial.print(
    mpu_ay_g,
    3
  );
  Serial.print(
    " AZ="
  );
  Serial.print(
    mpu_az_g,
    3
  );
  Serial.print(
    " GX="
  );
  Serial.print(
    mpu_gx_dps,
    2
  );
  Serial.print(
    " GY="
  );
  Serial.print(
    mpu_gy_dps,
    2
  );
  Serial.print(
    " GZ="
  );
  Serial.print(
    mpu_gz_dps,
    2
  );
  Serial.print(
    " TEMP="
  );
  Serial.print(
    temperatura_c,
    1
  );
  Serial.print(
    " RAMPA="
  );
  Serial.println(
    nome_estado_rampa()
  );
}
// ======================================================
// RESPOSTA RAW DO MPU
// ======================================================
void responder_mpu_raw() {
  if (
    !mpu_disponivel
  ) {
    Serial.println(
      "ERRO MPU_INDISPONIVEL"
    );
    return;
  }
  // Faz uma leitura direta para o comando RAW.
  mpu.getMotion6(
    &mpu_ax_raw,
    &mpu_ay_raw,
    &mpu_az_raw,
    &mpu_gx_raw,
    &mpu_gy_raw,
    &mpu_gz_raw
  );
  Serial.print(
    "OK MPU_RAW"
  );
  Serial.print(
    " AX="
  );
  Serial.print(
    mpu_ax_raw
  );
  Serial.print(
    " AY="
  );
  Serial.print(
    mpu_ay_raw
  );
  Serial.print(
    " AZ="
  );
  Serial.print(
    mpu_az_raw
  );
  Serial.print(
    " GX="
  );
  Serial.print(
    mpu_gx_raw
  );
  Serial.print(
    " GY="
  );
  Serial.print(
    mpu_gy_raw
  );
  Serial.print(
    " GZ="
  );
  Serial.println(
    mpu_gz_raw
  );
}
// ======================================================
// RESPOSTA ESPECIFICA DE RAMPA
// ======================================================
void responder_rampa() {
  if (
    !mpu_disponivel
  ) {
    Serial.println(
      "ERRO MPU_INDISPONIVEL"
    );
    return;
  }
  atualizar_mpu();
  Serial.print(
    "OK RAMPA ESTADO="
  );
  Serial.print(
    nome_estado_rampa()
  );
  Serial.print(
    " ANGULO="
  );
  Serial.println(
    mpu_pitch_graus,
    2
  );
}
// ======================================================
// LEITURA DE NUMEROS
// ======================================================
bool ler_inteiro(
  const char* texto,
  int* valor
) {
  char* fim;
  long numero =
    strtol(
      texto,
      &fim,
      10
    );
  if (
    *texto == '\0' ||
    *fim != '\0' ||
    numero < -255 ||
    numero > 255
  ) {
    return false;
  }
  *valor =
    (int)numero;
  return true;
}
bool ler_tempo_futaba(
  const char* texto,
  unsigned long* valor
) {
  char* fim;
  long numero =
    strtol(
      texto,
      &fim,
      10
    );
  if (
    *texto == '\0' ||
    *fim != '\0' ||
    numero < 1 ||
    numero >
    FUTABA_TEMPO_MAX_MS
  ) {
    return false;
  }
  *valor =
    (unsigned long)numero;
  return true;
}
// ======================================================
// RESPOSTA MOTOR
// ======================================================
void responder_ok_motor(
  const char* nome_motor,
  int velocidade
) {
  Serial.print(
    "OK MOTOR "
  );
  Serial.print(
    nome_motor
  );
  Serial.print(
    " "
  );
  Serial.println(
    limitar_velocidade(
      velocidade
    )
  );
}
// ======================================================
// PROCESSAMENTO DOS COMANDOS
// ======================================================
void processar_comando(
  char* comando
) {
  char* tipo =
    strtok(
      comando,
      " \t"
    );
  if (
    tipo == NULL
  ) {
    return;
  }
  // ==================================================
  // COMANDOS ORIGINAIS QUE NAO POSSUEM PARAMETRO
  // ==================================================
  if (
    strcmp(
      tipo,
      "PING"
    ) == 0 &&
    strtok(
      NULL,
      " \t"
    ) == NULL
  ) {
    Serial.println(
      "PONG"
    );
    return;
  }
  if (
    strcmp(
      tipo,
      "PARAR"
    ) == 0 &&
    strtok(
      NULL,
      " \t"
    ) == NULL
  ) {
    parar_todos_motores();
    parar_futaba();
    Serial.println(
      "OK PARADO"
    );
    return;
  }
  // IMPORTANTE:
  //
  // Mantido EXATAMENTE igual ao protocolo antigo
  // para nao quebrar a Raspberry.
  if (
    strcmp(
      tipo,
      "STATUS"
    ) == 0 &&
    strtok(
      NULL,
      " \t"
    ) == NULL
  ) {
    Serial.println(
      "OK STATUS SPEC_01"
    );
    return;
  }
  // ==================================================
  // PARAMETROS
  // ==================================================
  char* primeiro =
    strtok(
      NULL,
      " \t"
    );
  char* segundo =
    strtok(
      NULL,
      " \t"
    );
  char* terceiro =
    strtok(
      NULL,
      " \t"
    );
  char* quarto =
    strtok(
      NULL,
      " \t"
    );
  char* extra =
    strtok(
      NULL,
      " \t"
    );
  int valor1;
  int valor2;
  int valor3;
  int valor4;
  // ==================================================
  // MPU
  //
  // NOVO.
  //
  // Comandos:
  //
  // MPU
  // MPU RAW
  // MPU STATUS
  // MPU ZERO
  // MPU CALIBRAR
  // ==================================================
  if (
    strcmp(
      tipo,
      "MPU"
    ) == 0
  ) {
    // MPU sem parametros.
    if (
      primeiro == NULL
    ) {
      responder_mpu();
      return;
    }
    // Os comandos MPU abaixo aceitam apenas
    // um parametro.
    if (
      segundo != NULL
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
      return;
    }
    // MPU RAW
    if (
      strcmp(
        primeiro,
        "RAW"
      ) == 0
    ) {
      responder_mpu_raw();
      return;
    }
    // MPU STATUS
    if (
      strcmp(
        primeiro,
        "STATUS"
      ) == 0
    ) {
      if (
        !mpu_disponivel
      ) {
        Serial.println(
          "ERRO MPU_INDISPONIVEL"
        );
        return;
      }
      bool conectado =
        mpu.testConnection();
      Serial.print(
        "OK MPU STATUS="
      );
      Serial.print(
        conectado ?
        "OK" :
        "ERRO"
      );
      Serial.print(
        " ENDERECO=0x"
      );
      Serial.println(
        MPU6050_ENDERECO,
        HEX
      );
      return;
    }
    // MPU ZERO
    //
    // Coloque o Shadow no piso plano antes.
    if (
      strcmp(
        primeiro,
        "ZERO"
      ) == 0
    ) {
      if (
        !mpu_disponivel
      ) {
        Serial.println(
          "ERRO MPU_INDISPONIVEL"
        );
        return;
      }
      atualizar_mpu();
      zerar_referencia_mpu();
      Serial.println(
        "OK MPU ZERO"
      );
      return;
    }
    // MPU CALIBRAR
    //
    // Para motores e Futaba por seguranca.
    //
    // O Shadow precisa ficar parado durante
    // a calibracao.
    if (
      strcmp(
        primeiro,
        "CALIBRAR"
      ) == 0
    ) {
      if (
        !mpu_disponivel
      ) {
        Serial.println(
          "ERRO MPU_INDISPONIVEL"
        );
        return;
      }
      parar_todos_motores();
      parar_futaba();
      calibrar_giroscopio_mpu();
      inicializar_orientacao_mpu();
      Serial.println(
        "OK MPU CALIBRADO"
      );
      return;
    }
    Serial.println(
      "ERRO PARAMETROS_INVALIDOS"
    );
    return;
  }
  // ==================================================
  // RAMPA
  //
  // NOVO.
  // ==================================================
  if (
    strcmp(
      tipo,
      "RAMPA"
    ) == 0
  ) {
    if (
      primeiro != NULL
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    } else {
      responder_rampa();
    }
    return;
  }
  // ==================================================
  // FUTABA
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "FUTABA"
    ) == 0
  ) {
    unsigned long tempo_futaba_ms;
    if (
      primeiro == NULL ||
      terceiro != NULL
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
      return;
    }
    if (
      (
        strcmp(
          primeiro,
          "PARAR"
        ) == 0 ||
        strcmp(
          primeiro,
          "DESATIVAR"
        ) == 0
      ) &&
      segundo == NULL
    ) {
      parar_futaba();
      Serial.println(
        "OK FUTABA PARADO"
      );
    } else if (
      strcmp(
        primeiro,
        "STATUS"
      ) == 0 &&
      segundo == NULL
    ) {
      Serial.print(
        "OK FUTABA "
      );
      Serial.print(
        futaba_ativo ?
        "ATIVO" :
        "DESATIVADO"
      );
      Serial.print(
        " POTENCIA "
      );
      Serial.println(
        futaba_potencia_atual
      );
    } else if (
      segundo != NULL &&
      ler_inteiro(
        primeiro,
        &valor1
      ) &&
      valor1 >= -100 &&
      valor1 <= 100 &&
      valor1 != 0 &&
      ler_tempo_futaba(
        segundo,
        &tempo_futaba_ms
      )
    ) {
      acionar_futaba(
        valor1,
        tempo_futaba_ms
      );
      Serial.print(
        "OK FUTABA POTENCIA "
      );
      Serial.print(
        valor1
      );
      Serial.print(
        " TEMPO_MS "
      );
      Serial.println(
        tempo_futaba_ms
      );
    } else {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    }
    return;
  }
  // ==================================================
  // SERVO
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "SERVO"
    ) == 0
  ) {
    byte canal;
    if (
      primeiro == NULL ||
      segundo == NULL ||
      terceiro != NULL ||
      !ler_inteiro(
        segundo,
        &valor1
      ) ||
      valor1 < -180 ||
      valor1 > 180
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    } else if (
      !canal_servo_por_nome(
        primeiro,
        &canal
      )
    ) {
      Serial.println(
        "ERRO SERVO_INVALIDO"
      );
    } else if (
      canal ==
      SERVO_FUTABA
    ) {
      Serial.println(
        "ERRO SERVO_DESATIVADO"
      );
    } else {
      int destino =
        mover_servo_relativo(
          canal,
          valor1
        );
      Serial.print(
        "OK SERVO "
      );
      Serial.print(
        primeiro
      );
      Serial.print(
        " DELTA "
      );
      Serial.print(
        valor1
      );
      Serial.print(
        " POS "
      );
      Serial.println(
        destino
      );
    }
    return;
  }
  // ULTRASSOM
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "ULTRASSOM"
    ) == 0
  ) {
    if (primeiro == NULL) {
      // Mantido: ULTRASSOM sem parametro le o frontal.
      Serial.print(
        "OK ULTRASSOM "
      );
      Serial.println(
        medir_distancia_mm(
          ULTRASSOM_FRENTE_TRIG_PIN,
          ULTRASSOM_FRENTE_ECHO_PIN
        )
      );
    } else if (
      segundo == NULL &&
      strcmp(primeiro, "FRENTE") == 0
    ) {
      Serial.print(
        "OK ULTRASSOM FRENTE "
      );
      Serial.println(
        medir_distancia_mm(
          ULTRASSOM_FRENTE_TRIG_PIN,
          ULTRASSOM_FRENTE_ECHO_PIN
        )
      );
    } else if (
      segundo == NULL &&
      strcmp(primeiro, "LATERAL") == 0
    ) {
      Serial.print(
        "OK ULTRASSOM LATERAL "
      );
      Serial.println(
        medir_distancia_mm(
          ULTRASSOM_LATERAL_TRIG_PIN,
          ULTRASSOM_LATERAL_ECHO_PIN
        )
      );
    } else {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    }
    return;
  }
  // ==================================================
  // MOTOR
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "MOTOR"
    ) == 0
  ) {
    if (
      primeiro == NULL ||
      segundo == NULL ||
      terceiro != NULL ||
      !ler_inteiro(
        segundo,
        &valor1
      )
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    } else if (
      !controlar_motor_por_nome(
        primeiro,
        valor1
      )
    ) {
      Serial.println(
        "ERRO MOTOR_INVALIDO"
      );
    } else {
      responder_ok_motor(
        primeiro,
        valor1
      );
    }
    return;
  }
  // ==================================================
  // LADO
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "LADO"
    ) == 0
  ) {
    if (
      primeiro == NULL ||
      segundo == NULL ||
      terceiro != NULL ||
      !ler_inteiro(
        primeiro,
        &valor1
      ) ||
      !ler_inteiro(
        segundo,
        &valor2
      )
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    } else {
      controlar_lados(
        valor1,
        valor2
      );
      Serial.print(
        "OK LADO "
      );
      Serial.print(
        limitar_velocidade(
          valor1
        )
      );
      Serial.print(
        " "
      );
      Serial.println(
        limitar_velocidade(
          valor2
        )
      );
    }
    return;
  }
  // ==================================================
  // RODAS
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "RODAS"
    ) == 0
  ) {
    if (
      primeiro == NULL ||
      segundo == NULL ||
      terceiro == NULL ||
      quarto == NULL ||
      extra != NULL ||
      !ler_inteiro(
        primeiro,
        &valor1
      ) ||
      !ler_inteiro(
        segundo,
        &valor2
      ) ||
      !ler_inteiro(
        terceiro,
        &valor3
      ) ||
      !ler_inteiro(
        quarto,
        &valor4
      )
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
    } else {
      controlar_rodas(
        valor1,
        valor2,
        valor3,
        valor4
      );
      Serial.print(
        "OK RODAS "
      );
      Serial.print(
        limitar_velocidade(
          valor1
        )
      );
      Serial.print(
        " "
      );
      Serial.print(
        limitar_velocidade(
          valor2
        )
      );
      Serial.print(
        " "
      );
      Serial.print(
        limitar_velocidade(
          valor3
        )
      );
      Serial.print(
        " "
      );
      Serial.println(
        limitar_velocidade(
          valor4
        )
      );
    }
    return;
  }
  // ==================================================
  // FRENTE / TRAS / GIROS
  //
  // CODIGO ORIGINAL.
  // ==================================================
  if (
    strcmp(
      tipo,
      "FRENTE"
    ) == 0 ||
    strcmp(
      tipo,
      "TRAS"
    ) == 0 ||
    strcmp(
      tipo,
      "GIRAR_ESQ"
    ) == 0 ||
    strcmp(
      tipo,
      "GIRAR_DIR"
    ) == 0
  ) {
    if (
      primeiro == NULL ||
      segundo != NULL ||
      !ler_inteiro(
        primeiro,
        &valor1
      )
    ) {
      Serial.println(
        "ERRO PARAMETROS_INVALIDOS"
      );
      return;
    }
    valor1 =
      abs(
        limitar_velocidade(
          valor1
        )
      );
    if (
      strcmp(
        tipo,
        "FRENTE"
      ) == 0
    ) {
      controlar_rodas(
        valor1,
        valor1,
        valor1,
        valor1
      );
    }
    if (
      strcmp(
        tipo,
        "TRAS"
      ) == 0
    ) {
      controlar_rodas(
        -valor1,
        -valor1,
        -valor1,
        -valor1
      );
    }
    if (
      strcmp(
        tipo,
        "GIRAR_ESQ"
      ) == 0
    ) {
      controlar_lados(
        -valor1,
        valor1
      );
    }
    if (
      strcmp(
        tipo,
        "GIRAR_DIR"
      ) == 0
    ) {
      controlar_lados(
        valor1,
        -valor1
      );
    }
    Serial.print(
      "OK "
    );
    Serial.print(
      tipo
    );
    Serial.print(
      " "
    );
    Serial.println(
      valor1
    );
    return;
  }
  Serial.println(
    "ERRO COMANDO_INVALIDO"
  );
}
// ======================================================
// SETUP
// ======================================================
void setup() {
  // Mantem a mesma ordem principal do controller antigo.
  configurar_pinos();
  // Wire.begin() acontece aqui.
  configurar_pca9685();
  parar_todos_motores();
  // NOVO:
  //
  // Inicializa o MPU no mesmo I2C do PCA.
  //
  // Se ele nao existir ou estiver desconectado,
  // mpu_disponivel fica false e o restante continua.
  configurar_mpu();
  // Mantido na mesma velocidade do sistema original.
  Serial.begin(
    BAUD_RATE
  );
  ultimo_comando_ms =
    millis();
  // IMPORTANTE:
  //
  // Mantido EXATAMENTE igual ao controller anterior
  // para nao quebrar a Raspberry.
  Serial.println(
    "Arduino pronto - SPEC 01"
  );
}
// ======================================================
// LOOP
// ======================================================
void loop() {
  // NOVO:
  //
  // MPU e atualizado continuamente a aproximadamente
  // 100 Hz, independentemente da Raspberry solicitar
  // ou nao seus dados.
  atualizar_mpu();
  // Original.
  atualizar_futaba();
  // Original:
  // failsafe de comunicacao.
  if (
    millis() -
    ultimo_comando_ms >
    TIMEOUT_COMUNICACAO_MS
  ) {
    parar_todos_motores();
  }
  // Original:
  // comunicacao Raspberry -> Arduino.
  while (
    Serial.available() > 0
  ) {
    char recebido =
      Serial.read();
    if (
      recebido == '\n'
    ) {
      buffer_comando[
        tamanho_comando
      ] = '\0';
      if (
        tamanho_comando > 0
      ) {
        ultimo_comando_ms =
          millis();
        processar_comando(
          buffer_comando
        );
      }
      tamanho_comando =
        0;
    } else if (
      recebido != '\r' &&
      tamanho_comando <
      TAMANHO_COMANDO - 1
    ) {
      buffer_comando[
        tamanho_comando++
      ] =
        recebido;
    }
  }
}
