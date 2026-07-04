# Diferenças em relação ao OE² — todas as alterações, com motivo

Referência: `OE2_READING_DOSSIER.md` (raiz do repo) e `shadow/MAPEAMENTO.md`.
Tudo que não está listado aqui foi portado **verbatim**.

## Hardware / atuação

| # | Diferença | Motivo |
|---|---|---|
| 1 | Motores via Arduino Uno + 2× TB6612 falando SPEC 01 (`LADO`/`PARAR`), em vez de L298N em GPIO direto | O Shadow2026 não tem os motores no GPIO do Pi; o firmware SPEC 01 já existente cobre tudo (MAPEAMENTO §0) |
| 2 | Lei `steer()` computa velocidades por lado **em Python** e envia o resultado; a inversão de nomes de pinos do OE² (dossiê §9.1) não existe aqui | O firmware mapeia sinal→direção; polaridade por roda é `DIRECAO_*` no `config.h` do Shadow2026 |
| 3 | `sleep_steering()` substitui `time.sleep()` em manobras com motor girando (reenvio a cada 250 ms) | O Uno para os motores após 1 s sem comando; o OE² dorme até 1.35 s com motor ligado — sem keepalive o watchdog abortaria a manobra. Os TEMPOS das manobras não mudaram |
| 4 | Dedupe de comandos idênticos (mínimo 50 ms entre reenvios) | Busy-loops do OE² chamam steer() a cada 1 ms; a 115200 baud isso saturaria o link |
| 5 | `speed_left/right` compartilhados (GUI) não existem | Sem GUI |
| 6 | Chave física de partida (GPIO 21) e `program_continue()` → flag `terminate` (Ctrl-C) | Não há botão no Shadow2026 |
| 7 | Servos, relé do LED, luzes → não portados | Fora de escopo |

## Substituições de IMU (marcadas `# IMU_REPLACEMENT` no código)

| # | OE² | Shadow2026 |
|---|---|---|
| 8 | `rotation_y` ("none"/"ramp_up"/"ramp_down") | Constante "none": ramos de rampa de `get_speed`, `avoid_stuck`, latch do verde e gatilho do gap removidos (mortos). Só o ramo `ramp_ahead` (câmera) permanece |
| 9 | Varredura do gap com `turn_to_angle` ±45°/90° stop-on-black | Pivots temporizados: direita 0.35 s, esquerda 0.70 s, retorno 0.35 s @ 0.6, abortando cedo se `line_detected` (config: `T_SWEEP_RIGHT`, `SWEEP_SPEED`) |
| 10 | `turn_around()` com giro por giroscópio (yaw+180 arredondado a 90°) | Pivot temporizado `T_180 = 0.9 s @ 0.7` na direção de `last_turn_dir`; pre-roll de 0.55 s e cauda de ré (0.3 s, +0.4 s se linha pequena) idênticos; alternância l/r mantida |
| 11 | Wiggle de rampa do turn_around (gated por `sensor_z`) | Removido — nunca dispararia sem IMU (dossiê §9.8) |
| 12 | Escape de giro travado dentro de `turn_to_angle` (yaw estático 1.5 s) | Não tem equivalente — o pivot temporizado não tem feedback; se o 180° travar fisicamente, o SSIM de "preso" recupera no ciclo seguinte |

## Visão

| # | Diferença | Motivo |
|---|---|---|
| 13 | Captura: Picamera2 640×480 RGB888 → resize 448×252 → BGR (padrão comprovado do Shadow2026), em vez de sensor mode 0 do CM3 Wide | Câmera diferente; resolução do ALGORITMO continua 448×252 |
| 14 | `LensPosition` desabilitado por padrão (`None`) | Módulo sem AF; o OE² usava 6.5 no CM3 Wide. Configurável em `config.py` |
| 15 | FPS: câmera trava em 40 fps e todo frame é processado (cap de 90 continua no código) | O OE² capturava a 50 fps com cap de processamento 90 (efetivo 35-40). Comportamento efetivo igual |
| 16 | Silver AI (YOLO), branches de zone-entry/`check_silver`/`position_entry*`, calibração via GUI, zone loop | Fora de escopo — removidos, sem stubs |
| 17 | Retângulos de obstáculo (`obstacle_avoid` etc.) | Sem sensores IR |
| 18 | `calculate_angle` perdeu o parâmetro `entry` | Só era usado no posicionamento de entrada da zona |
| 19 | `time_line_similarity` é amostrado no laço de controle a 60 Hz | No OE² isso vivia em `update_sensor_average()` (plumbing IR/IMU, removido); mesma taxa efetiva |
| 20 | Frame de debug via shm apenas com `--debug`; sem shm/GUI no modo normal | Missão §4.9/4.10; OE² sempre escrevia o shm para a GUI |
| 21 | `average_line_point`/`average_line_angle`/`last_bottom_point` publicados como escalares compartilhados | No OE² eram arrays locais do processo de visão; a missão §4.1 os lista na API compartilhada |
| 22 | `vision_ready`: o controle espera o primeiro frame processado | Sem isso, a câmera escura no boot dispararia busca de gap com o robô parado |

## Controle / estados

| # | Diferença | Motivo |
|---|---|---|
| 23 | Máquina de estados com exatamente 4 estados | obstacle/silver/zone fora de escopo |
| 24 | `stop_for_red`: SEM o empurrão final `steer(0, 55)` e sem reset do run-timer da GUI | Dossiê §9.2 (constante não migrada da escala 0-100) + missão Fase F: o robô simplesmente segura 9 s |
| 25 | `silver_detected()`/`obstacle_detected()` em `orientate_gap`/`gap_avoid` viraram `False` literal (condições simplificadas algebricamente) | Sem esses sensores; lógica restante intacta |
| 26 | `stuck_cooldown` fixo em 4 s (variante 8 s era de rampa) | Ramo morto sem IMU |
| 27 | `except UnicodeDecodeError or UnboundLocalError` do OE² não foi replicado | Bug conhecido (dossiê §9.5); o link novo trata erros corretamente |
| 28 | `Timer` reimplementado com dict (semântica idêntica, inclusive "timer inexistente = não expirado") | O original usava um array numpy de strings; comportamento preservado, legibilidade melhor |

## Estrutura

| # | Diferença | Motivo |
|---|---|---|
| 29 | 5 processos + GUI → 2 processos + pai supervisor | Missão §4.10; nada nos hotspots depende do split de 5 |
| 30 | `line_cam.py` monolítico → `vision/{capture,pipeline,line,gap,green,red}.py`; `control.py` → `control/*.py` | Estrutura exigida pela missão §5; cada arquivo documenta as linhas de origem |
| 31 | Constantes centralizadas em `config.py` + cores em `config.ini` | Missão §4.11. As frações geométricas profundas da cascata de POI ficaram inline em `vision/line.py` (paridade verbatim) — ver CALIBRATION_GUIDE.md |
