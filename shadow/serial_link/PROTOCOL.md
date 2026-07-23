# Protocolo serial Pi ⇄ Arduino Uno — "SPEC 01"

Este documento descreve o protocolo **já gravado** no Uno do Shadow2026
(`../Shadow2026/arduino/motor_controller/motor_controller.ino` + `config.h`).
O `shadow/` **não** substitui esse firmware — apenas fala com ele
(decisão registrada em `MAPEAMENTO.md` §0). A lei de direção do OE²
(`steer(angle, speed)`, Hotspot 2) vive inteiramente em Python
(`control/steer.py`) e emite velocidades finais por lado.

## Parâmetros do link

| Parâmetro | Valor |
|---|---|
| Baud | 115200 |
| Framing | ASCII, linhas terminadas em `\n` (`\r` é ignorado pelo firmware) |
| Banner de boot | `Arduino pronto - SPEC 01` (uma vez; o Uno auto-reseta ao abrir a porta) |
| Watchdog | 1000 ms sem comando válido ⇒ todos os motores param |
| Teto de PWM | ±120 (`VELOCIDADE_MAXIMA_SEGURA`, clampado no firmware **e** no Python) |
| Faixa aceita | inteiros em [-255, 255] (clampados a ±120) |

## Comandos Pi → Uno

| Comando | Efeito | Resposta |
|---|---|---|
| `PING` | teste de vida | `PONG` |
| `PARAR` | para os 4 motores (IN1=IN2=LOW, PWM=0) | `OK PARADO` |
| `STATUS` | versão do protocolo | `OK STATUS SPEC_01` |
| `LADO <esq> <dir>` | FE+TE = `esq`, FD+TD = `dir` (com sinal; >0 = frente) | `OK LADO <e> <d>` |
| `RODAS <fe> <te> <fd> <td>` | velocidade individual por roda | `OK RODAS …` |
| `MOTOR <FE\|TE\|FD\|TD> <v>` | uma roda só (usado no teste de polaridade) | `OK MOTOR <nome> <v>` |
| `FRENTE <v>` / `TRAS <v>` | as 4 rodas para frente / trás (v = módulo) | `OK FRENTE <v>` … |
| `GIRAR_ESQ <v>` / `GIRAR_DIR <v>` | pivot no lugar | `OK GIRAR_… <v>` |

| `SERVO <nome> <angulo>` | move um canal do PCA9685; `GARRA_ESQ`, `GARRA_DIR`, `CACAMBA` ou `FUTABA`; angulo 0..180 | `OK SERVO <nome> <angulo>` |
| `LED APAGADO\|ACESO` | controla o LED indicador em D12; inicia aceso no boot | `OK LED <modo>` |
| `ULTRASSOM` | mede o sensor com TRIG D8 e ECHO D11 | `OK ULTRASSOM <mm>`; `-1` = sem eco |

Erros: `ERRO PARAMETROS_INVALIDOS`, `ERRO MOTOR_INVALIDO`, `ERRO SERVO_INVALIDO`, `ERRO COMANDO_INVALIDO`.

Os comandos antigos e o banner `SPEC_01` continuam iguais. Os comandos de
perifericos sao adicionais e nao modificam as velocidades dos motores. O PCA9685
usa o endereco I2C `0x40`, 50 Hz, com CH0=garra esquerda, CH1=garra direita,
CH2=cacamba e CH3=Futaba. No boot, os canais ficam desligados ate receberem o
primeiro comando `SERVO`.

## Como o Python usa o protocolo

`control/steer.py` traduz a semântica do OE² para `LADO`:

| `steer(angle, speed)` | Comando emitido |
|---|---|
| `angle == 190` (parada) | `PARAR` |
| `angle == 200` (ré) | `LADO -v -v` |
| `0 ≤ angle ≤ 110` (arco p/ direita) | `LADO v v·(110−angle)/109` |
| `angle > 110` (pivot p/ direita) | `LADO +1.2v −1.2v` (clampado a ±120) |
| `−110 ≤ angle < 0` (arco p/ esquerda) | `LADO v·(110+angle)/109 v` |
| `angle < −110` (pivot p/ esquerda) | `LADO −1.2v +1.2v` |

`v = round(speed × 120)`, `speed ∈ [0, 1]` como no OE².

## Regras do lado Python (`serial_link/arduino.py`)

- **ACKs nunca bloqueiam**: o buffer RX é drenado de forma não bloqueante;
  linhas `ERRO …` são impressas no log.
- **Dedupe**: comando idêntico ao anterior só é reenviado após 50 ms
  (protege o link contra loops `steer(...); sleep(.001)` do código portado).
- **Keepalive**: `refresh()` reenvia o último comando após 250 ms sem tráfego;
  `control/steer.py::sleep_steering()` chama isso durante qualquer espera com
  motor girando — sem isso o watchdog de 1 s abortaria manobras do OE² que
  dormem até 1.35 s.
- **Reconexão**: erro de escrita fecha a porta e tenta reabrir com backoff de
  0.5 s; enquanto desconectado os comandos são descartados (o watchdog do Uno
  mantém os motores parados).
- **Aviso**: não usar `dsrdtr`/`rtscts` (eram do Nano do OE²; no Uno podem
  travar a escrita).
