# Protocolo serial entre Raspberry Pi e Arduino Uno — SPEC 01

Este documento descreve o protocolo **já gravado** no Uno do Shadow2026
(`../../arduino/motor_controller/motor_controller.ino` + `config.h`).
O Python não substitui esse firmware: ele envia as velocidades finais de cada
lado por meio de `controle/direcao.py`.

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

| `SERVO <nome> <delta>` | move relativamente a ultima posicao; nomes ativos `GARRA_ESQ`, `GARRA_DIR` ou `CACAMBA`; delta -180..180 | `OK SERVO <nome> DELTA <d> POS <alvo>` |
| `LED APAGADO\|ACESO` | controla o LED indicador em D12; inicia aceso no boot | `OK LED <modo>` |
| `ULTRASSOM` | mede o sensor com TRIG D8 e ECHO D11 | `OK ULTRASSOM <mm>`; `-1` = sem eco |
| `FUTABA <potencia> <tempo_ms>` | aciona CH3 com potencia assinada -100..100 por 1..3000 ms e corta o sinal automaticamente | `OK FUTABA POTENCIA <p> TEMPO_MS <ms>` |
| `FUTABA STATUS` | informa se esta ativo e a potencia atual | `OK FUTABA <estado> POTENCIA <p>` |
| `FUTABA PARAR` | corta imediatamente a saida CH3 | `OK FUTABA PARADO` |

Erros: `ERRO PARAMETROS_INVALIDOS`, `ERRO MOTOR_INVALIDO`, `ERRO SERVO_INVALIDO`, `ERRO SERVO_DESATIVADO`, `ERRO COMANDO_INVALIDO`.

Os comandos antigos e o banner `SPEC_01` continuam iguais. Os comandos de
perifericos sao adicionais e nao modificam as velocidades dos motores. O PCA9685
usa o endereco I2C `0x40`, 50 Hz, com CH0=garra esquerda, CH1=garra direita,
CH2=cacamba e CH3=Futaba. No boot, a garra esquerda vai para 180 graus e a
direita para 0 graus (extremos abertos); a cacamba vai para 90 graus e o canal
do Futaba permanece totalmente desligado.

Potencia positiva e negativa giram em sentidos opostos. O firmware nunca mantem
o servo continuo ligado por mais de 3000 ms por comando. Para o servo instalado,
o neutro foi calibrado em 1660 us e a zona morta e compensada em 80 us: valores
positivos comandam subida e negativos comandam descida.

## Como o Python usa o protocolo

`controle/direcao.py` transforma ângulo e velocidade no comando `LADO`:

| `steer(angle, speed)` | Comando emitido |
|---|---|
| `angle == 190` (parada) | `PARAR` |
| `angle == 200` (ré) | `LADO -v -v` |
| `0 ≤ angle ≤ 110` (arco p/ direita) | `LADO v v·(110−angle)/109` |
| `angle > 110` (pivot p/ direita) | `LADO +1.2v −1.2v` (clampado a ±120) |
| `−110 ≤ angle < 0` (arco p/ esquerda) | `LADO v·(110+angle)/109 v` |
| `angle < −110` (pivot p/ esquerda) | `LADO −1.2v +1.2v` |

`v = round(speed × 120)`, com `speed` entre 0 e 1.

## Regras do lado Python (`comunicacao_serial/arduino.py`)

- **ACKs nunca bloqueiam**: o buffer RX é drenado de forma não bloqueante;
  linhas `ERRO …` são impressas no log.
- **Dedupe**: comando idêntico ao anterior só é reenviado após 50 ms
  (protege o link contra loops `steer(...); sleep(.001)` do código portado).
- **Keepalive**: `refresh()` reenvia o último comando após 250 ms sem tráfego;
  `controle/direcao.py::sleep_steering()` chama isso durante qualquer espera com
  motor girando — sem isso o watchdog de 1 s interromperia as esperas mais
  longas.
- **Reconexão**: erro de escrita fecha a porta e tenta reabrir com backoff de
  0.5 s; enquanto desconectado os comandos são descartados (o watchdog do Uno
  mantém os motores parados).
- **Garras juntas**: `Arduino.garras(esq, dir)` envia as duas linhas `SERVO`
  no mesmo pacote USB. O Uno ainda atualiza CH0 e CH1 sequencialmente, mas sem
  permitir outro comando intercalado.
- **Futaba com rodas paradas**: durante `FUTABA -20 1500`, use `LADO 0 0`
  como keepalive. Repetir `PARAR` cortaria CH3 antes dos 1500 ms.
- **Aviso**: não usar `dsrdtr`/`rtscts`, pois no Uno eles podem travar a
  escrita.
