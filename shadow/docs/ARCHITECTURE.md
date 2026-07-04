# Arquitetura — Shadow2026 line follower

## Modelo de 2 processos

```
                    ┌──────────────────────────────────┐
                    │  main.py (processo pai)          │
                    │  - cria Manager() e shm (--debug)│
                    │  - imprime status; janela debug  │
                    └───────┬──────────────┬───────────┘
                       fork │         fork │
        ┌───────────────────▼───┐   ┌──────▼────────────────────┐
        │ vision (pipeline.py)  │   │ control (loop.py)         │
        │ Picamera2 640×480 →   │   │ máquina de estados        │
        │ 448×252 BGR           │   │ line_status + steer()     │
        │ máscaras/contornos/   │   │ ÚNICO dono da porta serial│
        │ POI/verde/vermelho    │   │ → Uno SPEC 01 (LADO/PARAR)│
        │ ~40 fps               │   │ ≤ 60 it/s                 │
        └───────────┬───────────┘   └──────▲────────────────────┘
                    │   Manager().Value    │
                    └──────────────────────┘
```

- **A visão decide, o controle age** (mesma divisão do OE²): a visão publica
  medições (`line_angle`, `gap_angle`…) E decisões semânticas (`turn_dir`
  com o latch do verde); o controle é o único escritor de comandos de motor.
- Partida: `python3 shadow/main.py` — o pai instancia o `Manager()` ao importar
  `shared/mp_manager.py` e faz fork dos dois filhos (start method `fork`,
  padrão no Linux/Pi — este projeto não roda em Windows).
- `--debug`: o frame anotado viaja por `multiprocessing.shared_memory`
  (`shadow_shm_cam`, 448·252·3 B) da visão para o pai, que mostra a janela.

## Variáveis compartilhadas (Manager().Value)

| Variável | Escritor | Semântica |
|---|---|---|
| `line_angle` | visão | erro de direção ∈ [−180, 180] (offset de pixel escalado; + = direita) |
| `line_angle_y` | visão | y do POI escolhido (−1 sem linha) |
| `line_detected` | visão | há contorno preto ≥ `min_line_size` |
| `line_size` | visão | área do contorno seguido |
| `line_crop` | visão | fração y do sub-contorno próximo (.48 normal / .45 verde) |
| `line_similarity` | visão | SSIM entre máscaras (detecção de preso), a cada 30 frames |
| `black_average` | visão | média da máscara preta inteira |
| `ramp_ahead` | visão | detector "escuro à frente" (Hotspot 1) |
| `turn_dir` | visão | straight / left / right / turn_around (latch do verde) |
| `red_detected` | visão | contorno vermelho > 15000 px² (frame único) |
| `gap_angle`, `gap_center_x/y` | visão | geometria do toco de linha (só em gap_detected; reset −181) |
| `last_bottom_point`, `average_line_point`, `average_line_angle` | visão | histórico/projeções do POI |
| `min_line_size` | controle | knob global de área mínima (3000/4000/4500/9000) |
| `line_status` | controle | estado da máquina (abaixo) |
| `status` | controle | string de status em português (impressa pelo pai) |
| `terminate`, `vision_ready` | pai / visão | ciclo de vida |

`Timer` (timers nomeados) e os arrays de janela temporal (`add_time_value` /
`get_time_average`) são **locais por processo**, como no OE².

## Máquina de estados (`line_status`)

```
                 ┌───────────────────────────────────────────┐
                 ▼                                           │
        ┌─────────────────┐  vermelho > 15000 px²   ┌────────┴──┐
        │ line_detected   │────────────────────────►│   stop    │ 9 s
        │ steer(P-only)   │                         └───────────┘
        └───┬─────────────┘
            │ linha sumiu (e não é ramp_ahead)
            ▼
        ┌─────────────────┐ validou + orientou (≤7 ciclos) ┌───────────┐
        │  gap_detected   │───────────────────────────────►│ gap_avoid │
        │ orientate_gap() │                                └─────┬─────┘
        └───┬─────────────┘   linha reapareceu OU timeout        │
            │ abortou (não era gap / linha grande / y_gap<10)    │
            └────────────────► line_detected ◄───────────────────┘
```

O giro de 180° (verde duplo) não é um estado: é despachado dentro de
`line_detected` quando `turn_dir == "turn_around"`. O giro de 90° não é nem
despachado — ele **emerge** do POI esquerdo/direito + bias do tracker +
regime de pivot do steer() (Hotspot 4 do dossiê).

## Cadência e watchdog

- Visão ~40 fps (câmera trava em 25000 µs por frame); controle ≤ 60 it/s.
- O firmware do Uno para os motores após 1 s sem comando. O laço de controle
  a 60 Hz alimenta o watchdog naturalmente; manobras bloqueantes usam
  `sleep_steering()` (reenvio do último comando a cada 250 ms).
- Comandos idênticos são deduplicados (reenvio mínimo de 50 ms) para não
  saturar o link de 115200 nos busy-loops de 1 ms do código portado.
