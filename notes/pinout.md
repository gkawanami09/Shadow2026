# Pinout dos motores - Arduino Uno

| Motor | Lado | Driver | Canal | Pinos |
| --- | --- | --- | --- | --- |
| FE | frente esquerda | esquerdo | B | BIN1 A1, BIN2 A0, PWM D9 |
| TE | traseira esquerda | esquerdo | A | AIN1 A2, AIN2 A3, PWM D10 |
| FD | frente direita | direito | B | BIN1 D3, BIN2 D2, PWM D5 |
| TD | traseira direita | direito | A | AIN1 D4, AIN2 D7, PWM D6 |

- STBY dos drivers TB6612 esta ligado diretamente ao 5 V logico.
- D0 e D1 nao devem ser usados: eles sao usados pela Serial USB.
- O GND e comum entre bateria, Arduino, drivers e Raspberry Pi.

## Perifericos

| Dispositivo | Ligacao |
| --- | --- |
| LED indicador | D12 |
| Ultrassonico TRIG | D8 |
| Ultrassonico ECHO | D11 |
| PCA9685 SDA | A4 (I2C) |
| PCA9685 SCL | A5 (I2C) |
| PCA9685 CH0 | Garra esquerda |
| PCA9685 CH1 | Garra direita |
| PCA9685 CH2 | Cacamba |
| PCA9685 CH3 | Futaba |

O PCA9685 usa por padrao o endereco `0x40`. Os servos devem ter alimentacao
externa adequada; una o GND dessa fonte ao GND do Arduino. Nao alimente quatro
servos pelo pino 5 V do Arduino Uno.
