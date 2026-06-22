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
