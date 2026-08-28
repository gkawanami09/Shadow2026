# Guia de calibração do Shadow2026

Os valores atuais de `config.ini` e `config.py` foram ajustados para o
Shadow2026. Refaça a calibração sempre que mudar a câmera, a iluminação, a
altura ou a inclinação. Use esta ordem:

## 1. Polaridade dos motores (uma vez, rodas suspensas)

```bash
python3 -m shadow.tools.teste_serial
```

- "FRENTE": as 4 rodas giram levando o robô para frente? Para cada roda ao
  contrário, inverta o multiplicador dela (`+1` ↔ `-1`) em
  `Shadow2026/arduino/motor_controller/config.h` (`DIRECAO_FE/TE/FD/TD`) e
  regrave o firmware. **Não pule** — com polaridade errada o robô foge da linha.
- Depois: `python3 -m shadow.tools.teste_serial --watchdog` — os motores devem
  parar sozinhos ~1 s após o silêncio.

## 2. Cores (`tools/calibrar_cores.py`)

```bash
python3 -m shadow.tools.calibrar_cores
```

Com o robô SOBRE a pista, na iluminação real da sala:

1. **Grupo 1 — `black_max_normal_top`** (faixa distante, 0-40 % da imagem):
   suba B/G/R até a linha aparecer branca sólida na máscara **na metade de
   cima** da imagem, sem o piso virar branco.
2. **Grupo 2 — `black_max_normal_bottom`** (faixa próxima, 40-100 %): idem
   para a metade de baixo. É a região mais crítica — o POI vem daqui.
3. **Grupo 3 — `black_max_ramp_down_top`** (preto da rampa): posicione o
   robô onde a linha continua depois da descida. Ajuste B/G/R para a linha
   preta aparecer na máscara sem o piso da rampa ficar branco. Este perfil
   não dirige o robô; ele apenas veta o resgate quando ainda há preto à frente.
4. **Grupo 4 — verde**: marcador verde sólido branco na máscara, resto preto.
   Valide nas 4 posições de marcador.
5. **Grupos 5/6 — vermelho**: as duas bandas de hue (0-10 e 170-180). A faixa
   vermelha da pista deve encher a máscara; um objeto vermelho pequeno pode
   aparecer — não é problema (o gatilho exige 15000 px²).

Salve cada grupo com `s`. Valide com `python3 shadow/main.py --vision-only --debug`.

## 2.1 Câmera Wide e verde (sem tabuleiro)

A câmera wide usa o campo de visão máximo configurado na captura. O verde é
validado diretamente em pixels: quadrado verde antes do preto, com preto acima
e no lado interno da interseção. A topologia da linha escolhe o ramo futuro e
o controle acompanha esse caminho. Nenhum tabuleiro ou arquivo
\`calibracao_camera_wide.npz\` é necessário.

## 2.2 Entrada no resgate: prata + contexto preto

O modelo reconhece a prata da entrada. Ao primeiro frame alinhado e sem uma
linha preta à frente, o robô para e observa a prata por um segundo inteiro.
Essa linha é procurada tanto pelo perfil normal (grupos 1/2) como pelo perfil
específico da rampa (grupo 3). Portanto, teste os dois cenários depois de
calibrar:

1. no fim da rampa, o debug deve mostrar
   `linha_preta_depois_da_prata_seguindo_linha` ou
   `preto_rampa_depois_da_prata_seguindo_linha`; o robô continua o
   segue-linha por um segundo e não aceita nova prata nesse intervalo;
2. na entrada prata real, sem preto depois dela, o debug mostra
   `validando_prata_parado` por um segundo e depois `confirmada`; só então o
   resgate inicia.

O texto de debug ONNX PRATA mostra conf=atual/limiar: a confiança bruta do
YOLO aparece inclusive quando a faixa ainda não atingiu o limiar. O limiar
começa em .45; reduza no máximo até .40 somente se a prata real continuar
abaixo disso, pois valores menores tornam mais provável aceitar claridade da
rampa como candidata.

## 2.3 Faixa PRETA de saída e triângulos (câmera de resgate)

Os limiares estão em `config_resgate.py` (`EXIT_BLACK_*`, `MARKER_*`). Eles
não têm grupo no calibrador de linha de propósito: pertencem à outra câmera.
Valide-os por replay, com PNGs brutos capturados por
`python3 shadow/resgate.py --debug` (tecla `s`):

```bash
python3 shadow/tools/replay_visao.py --perfil saida --frames <fotos_da_soleira> --esperado positivo
```

```bash
python3 shadow/tools/replay_visao.py --perfil saida --frames <fotos_da_vitima_preta> --esperado negativo
```

O segundo comando é o teste que importa: a vítima preta e a soleira preta
têm a mesma cor, e é a geometria que as separa.

## 3. O que provavelmente precisa de retune (Wide 120° diagonal, ~4,5 cm)

| Constante | Onde | Sintoma se errada |
|---|---|---|
| `min_line_size` (3000) | config.py | Linha fina/distante ignorada (suba a câmera nos testes) ou ruído aceito. Com fish-eye a linha próxima fica GRANDE — se contornos de ruído passarem, suba |
| `GAP_NOT_A_STUB_SIZE` (17000) | config.py | Com a linha maior no near-field, um toco de gap pode passar de 17000 e abortar a orientação → suba proporcionalmente ao que `line_size` mostra no `--debug` |
| `RED_MIN_CONTOUR` (15000) | config.py | Vermelho nunca dispara (fish-eye encolhe a faixa no topo) → desça; especks disparam → suba |
| `GREEN_TOPOLOGY_MARKER_MIN_MM/MAX_MM` (18/35) | config.py | Marcador físico rejeitado depois da retificação; confirme primeiro impressão e homografia, não calibre em pixels |
| `T_180` (0.82 s) | config.py | Giro de 180° passa/falta ângulo — cronometre e ajuste (depende de atrito e bateria) |
| `T_SWEEP_RIGHT` (0.35 s) | config.py | Varredura do gap curta/longa demais para ~45° |
| `max_turn_angle` (110) | config.py | Oscilação na reta → suba um pouco (ex.: 120); curvas moles → desça |
| `left/right_correction` (1/1) | config.py | Robô puxa para um lado em linha reta |
| `LENS_POSITION` (`None`) | config.py | `None` faz autofocus durante a calibração; o artefato salva essa posição e o runtime precisa reaplicá-la e confirmá-la antes de armar o verde |

## 4. Frações geométricas (raramente precisam mudar)

A escolha do ponto da linha em `visao/linha.py` usa algumas frações internas:
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

Para registrar uma falha reproduzível, acrescente
`--record-vision /home/pi/inova/diagnosticos`. A pasta receberá uma subpasta
de PNGs lossless e um JSONL sincronizado por `frame_index`, `sequence` e
`decision_id`, com estado, direção travada, PWM e yaw. O caminho exato de
cada PNG aparece em `raw_frame`; duas sessões nunca reutilizam o mesmo nome.
