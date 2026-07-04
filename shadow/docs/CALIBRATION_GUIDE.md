# Guia de calibração — Shadow2026

Os valores do OE² em `config.ini`/`config.py` são **pontos de partida** afinados
para a câmera, iluminação (fita de LED) e altura (10 cm) do robô alemão. A sua
câmera é mais baixa (8 cm), mais inclinada (35°) e muito mais aberta (160°) —
espere retunar. Ordem recomendada:

## 1. Polaridade dos motores (uma vez, rodas suspensas)

```bash
python3 -m shadow.tools.serial_smoke
```

- "FRENTE": as 4 rodas giram levando o robô para frente? Para cada roda ao
  contrário, inverta o multiplicador dela (`+1` ↔ `-1`) em
  `Shadow2026/arduino/motor_controller/config.h` (`DIRECAO_FE/TE/FD/TD`) e
  regrave o firmware. **Não pule** — com polaridade errada o robô foge da linha.
- Depois: `python3 -m shadow.tools.serial_smoke --watchdog` — os motores devem
  parar sozinhos ~1 s após o silêncio.

## 2. Cores (`tools/color_slider.py`)

```bash
python3 -m shadow.tools.color_slider
```

Com o robô SOBRE a pista, na iluminação real da sala:

1. **Grupo 1 — `black_max_normal_top`** (faixa distante, 0-40 % da imagem):
   suba B/G/R até a linha aparecer branca sólida na máscara **na metade de
   cima** da imagem, sem o piso virar branco. Como não temos fita de LED, os
   tetos top/bottom tendem a ficar mais PRÓXIMOS entre si que os do OE²
   ([82,83,84] vs [133,133,135]).
2. **Grupo 2 — `black_max_normal_bottom`** (faixa próxima, 40-100 %): idem
   para a metade de baixo. É a região mais crítica — o POI vem daqui.
3. **Grupo 3 — `black_max_ramp_down_top`**: teto BEM escuro. Só é usado quando
   o detector "escuro à frente" dispara. Ajuste apontando a câmera para fora
   da pista (chão escuro): a máscara deve separar linha de piso.
4. **Grupo 4 — verde**: marcador verde sólido branco na máscara, resto preto.
   Valide nas 4 posições de marcador.
5. **Grupos 5/6 — vermelho**: as duas bandas de hue (0-10 e 170-180). A faixa
   vermelha da pista deve encher a máscara; um objeto vermelho pequeno pode
   aparecer — não é problema (o gatilho exige 15000 px²).

Salve cada grupo com `s`. Valide com `python3 shadow/main.py --vision-only --debug`.

## 3. O que provavelmente precisa de retune (fish-eye 160°, 8 cm, 35°)

| Constante | Onde | Sintoma se errada |
|---|---|---|
| `RAMP_SWAP_TRIGGER` (90) | config.py | Chão fora da pista no campo de visão dispara `ramp_ahead` → robô lento sem motivo. Suba para 110-130 se o `--debug` mostrar o círculo preto no canto sem rampa |
| `min_line_size` (3000) | config.py | Linha fina/distante ignorada (suba a câmera nos testes) ou ruído aceito. Com fish-eye a linha próxima fica GRANDE — se contornos de ruído passarem, suba |
| `GAP_NOT_A_STUB_SIZE` (17000) | config.py | Com a linha maior no near-field, um toco de gap pode passar de 17000 e abortar a orientação → suba proporcionalmente ao que `line_size` mostra no `--debug` |
| `RED_MIN_CONTOUR` (15000) | config.py | Vermelho nunca dispara (fish-eye encolhe a faixa no topo) → desça; especks disparam → suba |
| `GREEN_MIN_AREA` (2500) | config.py | Marcador ignorado de longe (desça) ou ruído verde aceito (suba) |
| `T_180` (0.9 s) | config.py | Giro de 180° passa/falta ângulo — cronometre e ajuste (depende de atrito e bateria) |
| `T_SWEEP_RIGHT` (0.35 s) | config.py | Varredura do gap curta/longa demais para ~45° |
| `max_turn_angle` (110) | config.py | Oscilação na reta → suba um pouco (ex.: 120); curvas moles → desça |
| `left/right_correction` (1/1) | config.py | Robô puxa para um lado em linha reta |
| `LENS_POSITION` (None) | config.py | Imagem desfocada a 8 cm em módulo com AF → tente 6-8 |

## 4. Frações geométricas (raramente precisam mudar)

A cascata de POI (`vision/line.py`) usa frações inline verbatim do OE²:
`0.1` (linha chega ao topo), `0.02/0.98` (linha encosta na borda), `0.75`
(contorno "chega ao fundo"), `0.5` (ponto lateral alto), `0.19` (crossbar),
gap de topo `1 px`, split de fundo `80 px`, bias verde `±150 px`. Elas são
proporcionais à resolução 448×252, que não mudou — só mexa se o `--debug`
mostrar o POI saltando errado em interseções, e anote o que mudou.

## 5. Validação final

1. `python3 shadow/main.py --debug` suspenso: linha reta sob a câmera →
   `ang≈0`; deslocar a linha para a direita → ângulo positivo → roda direita
   desacelera.
2. No chão, reta a velocidade padrão: oscilação ≤ ±3 cm.
3. Curva de 90°, curva arredondada, gap de 5/10/15 cm, verdes nos 4 casos,
   vermelho — na ordem dos gates das Fases C-F do RUNBOOK.
