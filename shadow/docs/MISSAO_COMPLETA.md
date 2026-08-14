# Missão completa: percurso → sala de resgate → percurso

Este documento descreve a camada que coordena as duas metades do Shadow. Ele
complementa `ARQUITETURA.md`, que continua descrevendo cada metade isolada.

## 1. Por que existe um supervisor

`main.py` e `resgate.py` funcionam bem, mas **não podem coexistir**: disputam
a mesma serial, o mesmo Arduino e os mesmos motores. Até agora a troca entre
eles era manual. O supervisor `mission.py` faz essa troca sozinho, e o faz
numa ordem fixa e testada.

Nada foi substituído. `main.py` e `resgate.py` continuam sendo a forma
recomendada de depurar cada metade separadamente, e continuam se comportando
exatamente como antes quando executados sozinhos.

## 2. Máquina de estados

Os nomes abaixo são as constantes de `controle/missao.py::MissionState`.

```text
FOLLOW_LINE
  └─ prata alinhada sem preto após a caixa ──► ENTER_RESCUE_ZONE
  └─ preto depois da prata ──────────────────► FOLLOW_LINE (nova prata bloqueada por 1 s)
ENTER_RESCUE_ZONE ──► STOP_AND_HANDOFF_TO_RESCUE ──► RESCUE_SCAN
                                                        │
   ┌────────────────────────────────────────────────────┘
   ▼
RESCUE_SCAN ──► TARGET_BRAKE ──► TARGET_VERIFY ──► TARGET_LOCK
   │  ▲                                               │
   │  │ varredura vazia (1ª vez)                      ▼
   │  └── RESCUE_RECOVERY ◄──┐                     ALIGN ──► APPROACH ──► PICKUP
   │                          │                                              │
   │                          │                                              ▼
   │                          │                                        CARRY_READY
   │                          │                                              │
   │                          │                        FIND_CORRECT_TRIANGLE ┘
   │                          │                              │
   │                          │                              ▼
   │                          │                     APPROACH_TRIANGLE ──► DEPOSIT
   │                          │                                              │
   │                          │                              RESTORE_GRIPPERS ┘
   │                          │                                     │
   └──────────────────────────┴───── UPDATE_INVENTORY ◄─────────────┘
                                          │
                     inventário completo? │
                                          ▼
                              VERIFY_RESCUE_COMPLETE
                                          │
                              DETECT_BOTH_TRIANGLES_FINAL
                                          │
                                  FIND_BLACK_EXIT ──► ALIGN_EXIT
                                          │
                             HANDOFF_TO_LINE_AND_VERIFY
                                          │
                                    FOLLOW_LINE ──► RED_FINISH
```

Regras que a máquina garante, e que são testadas em `tests/test_missao.py`:

- **uma vítima por vez.** Com uma esfera presa, travar outro alvo é recusado
  com erro. Não existe modo "guardar três": não há cesto validado no robô.
- **o contador só sobe depois de restaurar as garras.** Chegar ao triângulo
  não conta; soltar sem restaurar não conta.
- **duas pratas e uma preta.** Uma terceira prata ou uma segunda preta é
  recusada pelo inventário — contar isso significaria que o robô pegou algo
  que não é vítima, ou contou a mesma duas vezes.
- **o destino vem da cor presa**, nunca de um lado fixo. Prata → verde,
  preta → vermelho, e sem vítima presa nenhum triângulo comanda o robô.
- **a faixa preta só existe em `FIND_BLACK_EXIT`.** Durante toda a busca de
  vítimas o detector de saída nem é consultado, o que impede a soleira (e a
  vítima preta) de interromperem o resgate.
- **uma varredura vazia não encerra a sala.** A primeira leva a
  `RESCUE_RECOVERY`; só a segunda encerra. Isso dá segunda chance a uma
  vítima perdida por iluminação sem criar laço infinito.

## 3. Handoff — o contrato de segurança

A ordem está declarada como dado em `controle/missao.py` e executada por
`HandoffExecutor`. Os testes em `tests/test_missao.py` verificam as
propriedades, não a lista literal.

**Percurso → resgate** (`HANDOFF_TO_RESCUE`):

| # | Passo | Onde acontece de verdade |
|---|---|---|
| 1 | `stop_motors` | `ciclo.py`, ao terminar a entrada |
| 2 | `led_off` | `ciclo.py`, **enquanto a serial do percurso ainda existe** |
| 3 | `terminate_line_children` | supervisor |
| 4 | `join_line_children` | supervisor |
| 5 | `assert_line_children_dead` | supervisor |
| 6 | `close_line_camera` | `finally` do processo de visão |
| 7 | `release_serial` | `finally` do processo de controle |
| 8 | `release_motor_lock` | supervisor |
| 9 | `acquire_rescue_motor_lock` | `resgate.py` |
| 10 | `open_rescue_serial` | `resgate.py` |
| 11 | `assert_led_off` | `resgate.py`, na serial nova |
| 12 | `open_rescue_camera` | `resgate.py` |
| 13 | `start_rescue` | supervisor |

Sobre o LED: o regulamento interno manda apagá-lo antes do resgate, mas
apagar o LED **exige a serial**. Por isso ele é apagado no passo 2, que é o
último instante em que a serial do percurso existe, e reafirmado no passo 11
pelo processo de resgate. Os dois estão explícitos na lista.

Onde o passo acontece dentro de um filho, o supervisor o implementa como
**verificação**: ele confirma que o filho realmente terminou (e portanto que
o recurso foi liberado) e se recusa a prosseguir caso contrário. A câmera de
resgate nunca abre com o processo de visão ainda vivo.

Se qualquer passo falhar, o executor para onde está, chama `stop_motors` e
propaga a exceção. Ele não continua abrindo a câmera seguinte.

**Resgate → percurso** (`HANDOFF_TO_LINE`) é o espelho: parar, fechar a
câmera 0, liberar serial e trava, readquirir a trava, abrir a câmera 1, abrir
a serial, LED ACESO, reacquirir a linha.

## 4. Faixa PRATA de entrada

Roda dentro do processo de visão (o único que possui a câmera 1) e publica o
resultado por memória compartilhada. Sem `mission_mode` o detector nem é
construído — rodar `main.py` sozinho tem custo zero.

Evidência conjunta exigida (`visao/entrada_missao.py`):

1. `entrada.onnx` encontra a faixa prata com confiança mínima configurada;
2. o frame precisa estar com a linha preta centralizada e com ângulo pequeno;
3. não pode haver preto além da caixa dela, medido tanto pelo limiar normal
   quanto pelo limiar da rampa;
4. satisfeitas essas condições, um resultado positivo já confirma a entrada.

Após a confirmação, o processo de percurso para e libera câmera/serial. O
processo `resgate.py` então faz o avanço reto já calibrado de 1 s e inicia os
giros de busca. O modelo de entrada não roda durante o resgate.

## 5. Busca pulsada

`controle/busca_pulsada.py` substitui o giro contínuo por
`PULSE_ROTATE → BRAKE → SETTLE → OBSERVE`. Motivo: girando sem parar, a
esfera atravessa o campo de visão antes de acumular os três resultados
distintos exigidos para o lock, e os frames saem borrados com o autoexposure
ainda corrigindo.

Regra central: **um frame só confirma se foi capturado depois do fim do
SETTLE** (`frame_allowed()`). Um candidato visto durante o giro freia
imediatamente, mas não confirma — a confirmação usa apenas frames posteriores
à parada.

Sem IMU, o ângulo continua estimado por **tempo ativo de giro**; as pausas não
contam. `BALL_SEARCH_SECTORS × BALL_SEARCH_PULSE_S` = 9 × 0,40 s = 3,60 s,
compatível com os 3,54 s estimados para 360° em PWM 80.

O controlador contínuo `BallSearchController` **não foi removido**: continua
existindo com seus testes e volta a ser usado se
`BALL_SEARCH_PULSED = False`. Os dois expõem a mesma interface.

## 6. Parâmetros novos

`config.py` (câmera de linha):

| Parâmetro | Padrão | Papel |
|---|---|---|
| `ENTRY_SILVER_ENABLED` | `True` | liga a detecção da entrada |
| `ENTRY_MODEL_PATH` | `modelos/entrada.onnx` | modelo da faixa prata |
| `ENTRY_MODEL_INPUT` | `640` | tamanho de entrada do modelo |
| `ENTRY_MODEL_TRIGGER_CONFIDENCE` | `.15` | sinal inicial: para o robô e abre uma confirmação curta |
| `ENTRY_MODEL_MIN_CONFIDENCE` | `.30` | confiança que confirma o resgate; o veto de preto/rampa continua obrigatório |
| `ENTRY_SILVER_HINT_VALIDATION_S` | `.80` | tempo parado para o sinal inicial virar confirmação ou ser descartado |
| `ENTRY_BLACK_AFTER_SILVER_MAX_DISTANCE_RATIO` | `.10` | só procura a continuação da linha perto da faixa, não no fundo do quadro |
| `ENTRY_BLACK_AFTER_SILVER_MAX_BRIGHTNESS/CHROMA` | `70` / `25` | preto exclusivo do veto: escuro e neutro, sem confundir prata/cinza |
| `ENTRY_SILVER_VOTES_NEEDED/VOTE_WINDOW` | `1` / `3` | confirma no primeiro prata alinhado sem preto após a caixa |
| `ENTRY_LINE_MAX_ANGLE` | `35` | ângulo máximo para entrar alinhado, tolerando o robô torto |
| `ENTRY_LINE_MAX_BOTTOM_ERROR_PX` | `110` | erro máximo do ponto inferior da linha, tolerando deslocamento |
| `ENTRY_ALIGNMENT_HOLD_S` | `.70` | conserva o último alinhamento quando a faixa cobre a linha |

`config_resgate.py` (câmera de resgate):

| Parâmetro | Padrão | Papel |
|---|---|---|
| `BALL_SEARCH_PULSED` | `True` | liga a busca pulsada |
| `BALL_SEARCH_PULSE_S` | `0.40` | duração ativa de cada pulso (~41°) |
| `BALL_SEARCH_SETTLE_S` | `0.12` | pausa mecânica antes de observar |
| `BALL_SEARCH_OBSERVE_FRAMES` | `3` | frames novos observados por parada |
| `BALL_SEARCH_OBSERVE_TIMEOUT_S` | `0.60` | teto da observação |
| `BALL_SEARCH_SECTORS` | `9` | setores de cobertura |
| `BALL_SEARCH_TOTAL_TIMEOUT_S` | `75.0` | teto global da busca |
| `EXIT_BLACK_*` | — | faixa preta de saída (ver `GUIA_CALIBRACAO.md`) |
| `EXIT_BLACK_MIN_ASPECT` | `4.0` | separa soleira de vítima preta |
| `EXIT_ALIGN_MAX_CENTER_ERROR` | `0.12` | alinhamento antes de atravessar |
| `FINAL_TRIANGLE_MAP_FRAMES` | `6` | frames do mapeamento final |
| `FINAL_TRIANGLE_OVERLAY_BGR` | verde `(0,255,0)`, vermelho `(0,0,255)` | cores do overlay |

## 7. Comandos

Testes:

```bash
python3 -m unittest discover -s shadow/tests -p "test_*.py" -v
```

Replay sem motores (o degrau antes de qualquer teste físico):

```bash
python3 shadow/tools/replay_visao.py --perfil entrada --frames <dir> --esperado positivo
```

Benchmark reprodutível:

```bash
python3 shadow/tools/benchmark_visao.py --sintetico --repeticoes 60
```

Benchmark ao vivo no Pi (mede captura e visão separadamente):

```bash
python3 shadow/tools/benchmark_visao.py --camera --segundos 20
```

Visão sem motores — só o segue-linha:

```bash
python3 shadow/main.py --vision-only --debug
```

Visão sem motores — só o resgate:

```bash
python3 shadow/resgate.py --debug
```

Calibração (grupo 7 = prata de entrada):

```bash
python3 -m shadow.tools.calibrar_cores
```

Missão completa:

```bash
python3 shadow/mission.py
```

Missão completa priorizando as vítimas vivas (mundial):

```bash
python3 shadow/mission.py --policy silver_first
```

## 8. Limitações reais

Declaradas para não serem descobertas em competição:

1. **Nada aqui foi testado no robô.** Todo o trabalho é offline: 356 testes
   automatizados e replay sintético. Nenhuma afirmação sobre comportamento
   físico foi verificada.
2. **Os limiares da faixa prata não foram medidos com a fita real.** São um
   ponto de partida conservador. `GUIA_CALIBRACAO.md` §2.1 é obrigatório.
3. **`BALL_SEARCH_PULSE_S = 0,40 s ≈ 41°` é uma conta, não uma medição.**
   Depende de atrito, bateria e piso. Precisa ser cronometrado.
4. **Não há modelo de aprendizado de máquina.** Não existe dataset rotulado
   do Shadow suficiente para treinar e validar um. A infraestrutura de coleta
   (`--debug` + `s` no resgate) e de replay existe; o detector clássico foi
   preservado. Ver §9.
5. **Câmera não prova condutividade.** Se um alvo falso for visualmente
   idêntico à vítima, nenhuma confirmação visual resolve. Nenhum hardware foi
   alterado para isso.
6. **Sem IMU, o 360° é temporizado.** Escorregamento de roda muda a cobertura
   real; por isso existe o teto global `BALL_SEARCH_TOTAL_TIMEOUT_S`.
7. **O mapeamento final dos triângulos não bloqueia a saída.** Se os dois não
   forem encontrados no prazo, o robô sai mesmo assim — ficar preso ali
   custaria a prova.

## 8.1 Medições nas capturas reais da arena

18 capturas da câmera de resgate (12 com vítima prateada, 6 sem, 2 com o
triângulo vermelho). Frames **independentes**, não vídeo — como o detector
exige 3 confirmações temporais, nenhuma imagem isolada confirma. A avaliação é
no nível de candidato: a recall por frame é piso, não teto.

| Detector | Resultado |
|---|---|
| Vítimas @ 640×480 | 4/12 detectadas, **0/6 falso positivo** |
| Vítimas @ 320×240 (produção) | 2/12 detectadas, 0/6 falso positivo |
| Faixa preta de saída | 16/18, 2 falsos positivos |
| Marcador vermelho | fragmento rejeitado corretamente; chegada detectada |

**Vítimas — o gargalo é o filtro de perímetro, não a proposta.** A esfera é
proposta pelo estágio geométrico nas 12 imagens, a 3–14 px do centro real.
Quem rejeita é a análise radial de borda (7 das 8 falhas); a oitava é a
esfera cortada pela borda do quadro, que é comportamento intencional.

**Afrouxar o perímetro é armadilha, não solução.** Com apoio 0.30/4 setores a
recall sobe para 10/12, mas: aparece falso positivo numa foto com pessoa, e
**4 esferas prateadas passam a ser classificadas como `black`** — o que
entregaria vítima viva no triângulo vermelho. Causa: o Hough propõe vários
raios para a mesma esfera (10, 41, 87 e 100 px na mesma imagem, pegando o
reflexo interno); com o filtro frouxo passa um círculo errado e a aparência é
amostrada fora da esfera. O filtro rígido está certo; o problema é a proposta.

**Geometria não discrimina marcador nesta perspectiva.** Medido no pipeline:

| | triangularidade | proporção | chroma |
|---|---|---|---|
| Marcador (navegação) | 0.577 | 1.03 | **148** |
| Marcador (chegada) | 0.623 | 3.83 | **124** |
| Cadeira vermelha | 0.677 | 3.80 | 63 |
| Cadeira vermelha | 0.627 | 2.81 | 79 |
| Círculo perfeito | 0.605 | 1.00 | — |
| Quadrado perfeito | 0.500 | 1.00 | — |

A cadeira do laboratório é **mais triangular** que o marcador, e o marcador
cai entre quadrado e círculo. Não existe limiar de triangularidade que aceite
um e rejeite o outro — afrouxá-lo fez círculos coloridos serem detectados como
marcador, o que os testes já existentes pegaram. Quem separa é a
**cromaticidade** (124–148 contra 63–79), e é nela que o rigor foi colocado.

## 9. Checklist de testes físicos ainda necessários

Nenhum item abaixo foi executado. Faça **na ordem** — cada degrau protege o
seguinte. Em toda fase, mantenha parada de emergência ao alcance.

| # | Teste | Como | Critério de aprovação |
|---|---|---|---|
| 1 | suíte automatizada | `python3 -m unittest discover -s shadow/tests -p "test_*.py"` | 360/360 |
| 2 | replay de frames reais | `replay_visao.py --perfil entrada/saida/vitima` | zero falsos positivos nos negativos |
| 3 | visão ao vivo sem motores | `main.py --vision-only --debug` e `resgate.py --debug` | imagem fluida, sem backlog |
| 4 | benchmark no Pi | `benchmark_visao.py --camera --segundos 20` | atraso p95 < `BALL_FRAME_STALE_S` |
| 5 | faixa prata sem transição | `main.py --vision-only --debug` andando sobre a fita | confirma na fita, nunca no piso/reflexo |
| 6 | handoff de câmeras e LED | `mission.py`, **rodas suspensas** | LED apaga na entrada, câmera 1 fecha antes da 0, sem erro de trava |
| 7 | busca pulsada | rodas suspensas | gira ~41°, PARA, observa, repete; 360° fecha e não repete |
| 8 | aproximação sem garra | motores ligados, garra desconectada | para na vítima sem tocar |
| 9 | coleta de uma prata | `resgate.py --drive --target silver` | sequência de garra/Futaba idêntica à calibrada |
| 10 | coleta de uma preta | `resgate.py --drive --target black` | idem |
| 11 | depósito de uma prata | — | vai ao triângulo VERDE |
| 12 | depósito de uma preta | — | vai ao triângulo VERMELHO |
| 13 | ciclo de três vítimas | `resgate.py --drive --no-exit-phase` | contador 2 prata + 1 preta, sem repetir vítima |
| 14 | dois triângulos juntos | fase final, `--debug` | verde desenhado verde, vermelho desenhado vermelho |
| 15 | saída e retorno | `resgate.py --drive` completo | atravessa a soleira e para |
| 16 | missão completa | `mission.py` | percurso → sala → percurso → vermelho final |

Itens 9 a 12 existem porque a sequência de garra e Futaba foi preservada
**exatamente** como estava calibrada, mas essa preservação só foi verificada
por leitura de código e pelos testes existentes de `coleta_resgate` — não com
o servo real.

## 10. Dataset ainda necessário

Para treinar qualquer classificador (e mesmo para confiar nos limiares
clássicos), faltam imagens reais do Shadow, capturadas pelas câmeras do
próprio robô, separadas **por sessão/arena** entre treino e validação — nunca
por frames vizinhos da mesma gravação.

Câmera de linha — faixa prata:

- 300+ positivos: a fita a várias distâncias, ângulos e iluminações;
- 300+ negativos difíceis: piso branco, reflexo de LED, parafuso, fita
  brilhante pequena, luz forte sobre a linha, a faixa preta, a vítima
  prateada dentro do campo.

Câmera de resgate — vítimas:

- 500+ com prata lisa, prata amassada, prata tingida e preta, incluindo
  junto à parede e parcialmente fora do quadro;
- 500+ negativos: madeira, sombras, participantes, roupas, sapatos, cadeiras,
  rodas, parafusos, bordas da arena, portas, triângulos, objetos circulares.

Câmera de resgate — soleira preta:

- 200+ positivos da soleira em vários ângulos;
- 200+ negativos com a vítima preta em todos os tamanhos aparentes.

Enquanto esses conjuntos não existirem, treinar um modelo produziria um
número de validação sem significado. O detector atual foi preservado
inteiro.
