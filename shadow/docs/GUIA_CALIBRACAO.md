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
3. **Grupo 3 — `black_max_ramp_down_top`**: teto BEM escuro. Só é usado quando
   o detector "escuro à frente" dispara. Ajuste apontando a câmera para fora
   da pista (chão escuro): a máscara deve separar linha de piso.
4. **Grupo 4 — verde**: marcador verde sólido branco na máscara, resto preto.
   Valide nas 4 posições de marcador.
5. **Grupos 5/6 — vermelho**: as duas bandas de hue (0-10 e 170-180). A faixa
   vermelha da pista deve encher a máscara; um objeto vermelho pequeno pode
   aparecer — não é problema (o gatilho exige 15000 px²).

6. **Grupo 7 — prata da faixa de ENTRADA** (câmera de linha). Ver a seção
   2.1 abaixo: este grupo tem um procedimento próprio.

Salve cada grupo com `s`. Valide com `python3 shadow/main.py --vision-only --debug`.

## 2.1 Faixa PRATA de entrada (grupo 7)

Este perfil é **independente** do prata da vítima. A vítima é vista pela
câmera de resgate e seus limiares vivem em `config_resgate.py`; a faixa é
vista pela câmera de linha e seus limiares vivem em `config.py` +
`[color_values_line]`. Nunca copie um para o outro — são câmeras, alturas,
distâncias e iluminações diferentes.

Os valores em `config.py` (`ENTRY_SILVER_*`) são um ponto de partida
conservador e **ainda não foram medidos com a fita real**. A calibração
abaixo é obrigatória antes de confiar na entrada automática.

Procedimento, com o robô sobre a pista e a fita prata colada no chão:

1. abra `python3 -m shadow.tools.calibrar_cores` e tecle `7`;
2. a janela mostra o frame em cima e a máscara embaixo. No frame aparecem
   também a linha da ROI, a caixa do candidato e — o mais importante — o
   **motivo da rejeição** quando ele não passa;
3. ajuste `V min` até a fita ficar sólida na máscara e o piso não;
4. ajuste `S max` para baixo até o piso colorido sair da máscara sem perder a
   fita (prata é neutra: S baixo);
5. aproxime e afaste o robô. A caixa deve ficar verde (`ACEITA`) na faixa de
   distância em que o robô realmente chega à sala;
6. salve com `s`.

Leia o motivo quando a fita for rejeitada:

| Motivo | Significado | O que ajustar |
|---|---|---|
| `sem_linha_cheia` | nenhuma linha horizontal atingiu o preenchimento mínimo | `V min`/`S max`, ou `ENTRY_SILVER_MIN_ROW_FILL` |
| `estreita` | a fita não atravessa largura suficiente | aproxime; ou desça `ENTRY_SILVER_MIN_SPAN_RATIO` |
| `espessa` | a máscara preencheu a ROI — normalmente o piso inteiro entrou | suba `V min` |
| `compacta` | forma quase quadrada: é uma esfera, não uma fita | nada — é o veto funcionando |
| `saturada` | a região tem cor demais para ser metal | suba `S max` com cuidado |
| `sem_assinatura_reflexiva` | neutro e claro, mas sem brilho nem faixa dinâmica: papel branco | desça `ENTRY_SILVER_MIN_DYNAMIC_RANGE` só se a fita real for fosca |
| `sem_contraste` | a fita ficou idêntica ao piso, em brilho E em textura | mude o ângulo do LED; ou desça `ENTRY_SILVER_MIN_SURROUND_CONTRAST` |

### Quando o piso cinza tem o mesmo brilho da fita

Esse foi o caso medido na arena real: piso e fita chegando os dois a
V≈216-226 e S≈20-24. Nessa situação **HSV sozinho não separa** — a máscara
engole o piso inteiro e o candidato morre em `espessa`.

O que continua diferente é a **textura da luz**: metal concentra brilho em
pontos, piso fosco é uniforme por mais claro que seja. O slider
**`Reflexo min`** (só no grupo 7) filtra por isso, antes da geometria.

Como ajustar: suba `Reflexo min` até o piso sair da máscara e sobrar só a
fita. Se a fita sumir junto, desça. Zero desliga o filtro.

Esse mesmo sinal também vale como contraste: se a fita e o piso tiverem o
mesmo brilho mas texturas diferentes, o detector aceita pela textura. Só é
rejeitado o que for igual nas **duas** coisas.
| `linha_continua` | a linha preta segue à frente | correto: não é a entrada |

Depois de calibrar, valide **sem** transição, com o replay:

```bash
python3 shadow/tools/replay_visao.py --perfil entrada --frames <positivos> --esperado positivo
```

```bash
python3 shadow/tools/replay_visao.py --perfil entrada --frames <negativos> --esperado negativo
```

O conjunto de negativos precisa incluir piso branco, reflexo de LED, a
vítima prateada, a faixa preta e luz sobre a linha. **Zero falsos positivos
nesse conjunto** é o critério para liberar a entrada automática.

## 2.2 Faixa PRETA de saída e triângulos (câmera de resgate)

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

## 3. O que provavelmente precisa de retune (fish-eye 160°, 8 cm, 35°)

| Constante | Onde | Sintoma se errada |
|---|---|---|
| `RAMP_SWAP_TRIGGER` (90) | config.py | Chão fora da pista no campo de visão dispara `ramp_ahead` → robô lento sem motivo. Suba para 110-130 se o `--debug` mostrar o círculo preto no canto sem rampa |
| `min_line_size` (3000) | config.py | Linha fina/distante ignorada (suba a câmera nos testes) ou ruído aceito. Com fish-eye a linha próxima fica GRANDE — se contornos de ruído passarem, suba |
| `GAP_NOT_A_STUB_SIZE` (17000) | config.py | Com a linha maior no near-field, um toco de gap pode passar de 17000 e abortar a orientação → suba proporcionalmente ao que `line_size` mostra no `--debug` |
| `RED_MIN_CONTOUR` (15000) | config.py | Vermelho nunca dispara (fish-eye encolhe a faixa no topo) → desça; especks disparam → suba |
| `GREEN_MIN_AREA` (2500) | config.py | Marcador ignorado de longe (desça) ou ruído verde aceito (suba) |
| `T_180` (0.82 s) | config.py | Giro de 180° passa/falta ângulo — cronometre e ajuste (depende de atrito e bateria) |
| `T_SWEEP_RIGHT` (0.35 s) | config.py | Varredura do gap curta/longa demais para ~45° |
| `max_turn_angle` (110) | config.py | Oscilação na reta → suba um pouco (ex.: 120); curvas moles → desça |
| `left/right_correction` (1/1) | config.py | Robô puxa para um lado em linha reta |
| `LENS_POSITION` (None) | config.py | Imagem desfocada a 8 cm em módulo com AF → tente 6-8 |

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
