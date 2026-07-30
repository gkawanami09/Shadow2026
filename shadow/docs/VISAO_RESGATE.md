# Visão do resgate — arquitetura nova

## Por que mudou

O detector anterior encadeava **dez portões de aparência** (cor, brilho,
textura, contraste, reflexo) e tinha 218 constantes de calibração. Medido em
121 imagens rotuladas: **recall de 45%**.

Isso é aritmética, não azar. Dez portões em série, cada um aprovando ~92% das
vítimas verdadeiras, dão 0.92¹⁰ ≈ 45%. E numa arena diferente, cada portão
piora um pouco:

| Cada portão aprova | Recall final |
|---|---|
| 95% | 60% |
| **92%** | **45%** ← medido |
| 85% | 20% |

Nenhum ajuste de limiar sai desse regime: para o produto dar 95%, cada portão
precisaria de 99,5%.

## A arquitetura agora

```text
modelo treinado   ->  APARÊNCIA   é vítima? prata ou preta?
plausibilidade    ->  GEOMETRIA   cabe fisicamente naquela posição?
rastreamento      ->  TEMPO       aparece de forma consistente?
```

A separação é o ponto. Trocar de arena afeta **só o modelo** — e o modelo é a
única peça retreinável. As outras duas camadas descrevem geometria de câmera
e consistência temporal, que não mudam quando o piso ou a luz mudam.

| Módulo | Papel |
|---|---|
| `visao/deteccao.py` | tipos compartilhados; o contrato entre visão e controle |
| `visao/plausibilidade.py` | regras físicas, independentes de arena |
| `visao/vitima_yolo.py` | modelo + rastreamento do alvo único |
| `visao/marcador_resgate.py` | marcadores verde/vermelho (clássico, por cor) |
| `visao/overlay_resgate.py` | preview; nunca decide nada |

### Por que os marcadores continuam clássicos

Porque a cor **funciona medida** para eles: marcador com cromaticidade 124–148
contra cadeira vermelha do laboratório com 63–79. Separação limpa, sem treino.

Já a **forma** não funciona: o triângulo real, visto quase rente ao piso, dá
triangularidade 0,577 — entre um quadrado (0,500) e um círculo (0,605), e
*abaixo* da cadeira (0,677). O gate de forma foi removido e o rigor foi para a
cromaticidade. Limitação assumida: um círculo muito saturado no chão seria
aceito. Está registrada em `tests/test_marcador_resgate.py`.

## Escopo atual

`shadow/resgate.py` faz hoje: **procurar uma vítima em giros pulsados,
aproximar, avançar 1 s, baixar o Futaba, avançar mais 1 s, fechar, elevar e
selecionar a vítima pelo lado**.

Prata abre a garra esquerda; preta abre a direita. Depois de restaurar as duas
garras, a busca pulsada recomeça. O mesmo verde em vários frames vale uma
aparição. Duas passagens separadas pelo verde sem uma coleta no meio encerram
a procura; uma coleta concluída zera essa contagem.

O transporte até o depósito e a saída da sala **não** estão no fluxo atual:

- `controle/deposito_resgate.py`
- `controle/saida_resgate.py`
- `controle/missao.py`, `shadow/mission.py`

O ciclo ativo reaproveita `controle/coleta_resgate.py`, inclusive a abertura
por lado e a restauração das garras.

## O que falta: o modelo

O arquivo do modelo **não acompanha o repositório** — ele depende de imagens
da câmera deste robô. Sem ele, `resgate.py` para e explica o que fazer. Nunca
produz detecção falsa.

### 1. Coletar

```bash
python3 shadow/tools/coletar_dataset.py --sessao prata_cozinha
```

Uma pasta por sessão, em `shadow/captures/dataset/`. Isso é o que permite
dividir treino/validação **por sessão** — dividir aleatoriamente frames de uma
mesma gravação coloca quase-cópias dos dois lados e infla o resultado.

Regra de ouro: **varie tudo, menos a câmera.**

| Deve variar | Não pode variar | Precisa ficar realista |
|---|---|---|
| fundo, piso, parede | a câmera | vítima no chão |
| iluminação, cor da luz | a altura de montagem | terço inferior do quadro |
| distância, ângulo | a lente | distâncias que o robô encontra |

Use o próprio robô como tripé. Fotografar com o celular ensina uma geometria
que o robô nunca vai ver.

Meta aproximada:

| Tipo | Quantidade |
|---|---|
| prata + preta, ambientes variados | ~800 |
| prata + preta na arena real | ~400 |
| duas vítimas no quadro | ~150 |
| junto à parede / parcialmente oclusa | ~150 |
| **sem vítima nenhuma** | ~400 |
| validação (sessão própria, só arena) | ~200 |

Os negativos valem tanto quanto os positivos — é neles que se mede falso
acionamento.

### 2. Rotular

Roboflow, duas classes, **nesta ordem**:

```
0 = black
1 = silver
```

A ordem tem que bater com `config_resgate.VICTIM_MODEL_CLASSES`.

### 3. Treinar e exportar

YOLOv8n, entrada 320. Exporte para ONNX e salve em:

```
shadow/modelos/vitimas.onnx
```

### 4. Medir antes de confiar

```bash
python3 shadow/tools/benchmark_visao.py --camera --segundos 20
```

O número que importa: **atraso p95 abaixo de 750 ms** (`BALL_FRAME_STALE_S`).
Acima disso o robô decide com imagem vencida.

```bash
python3 shadow/tools/replay_visao.py --perfil vitima --frames <validacao> --esperado positivo
```

```bash
python3 shadow/tools/replay_visao.py --perfil vitima --frames <negativos> --esperado negativo
```

**Zero falsos positivos nos negativos** é o critério para ligar os motores.

## Calibrar a plausibilidade

A envoltória tamanho×linha depende da **altura e do ângulo** da câmera. Os
valores atuais vieram das 18 fotos reais deste robô, que cobrem só vítimas
próximas, e por isso são propositalmente largos.

Se a câmera mudar de posição, recalibre `PLAUSIBLE_*` em `config_resgate.py`.

## Comandos

Só visão, sem motores:

```bash
python3 shadow/resgate.py --debug
```

Só marcadores (funciona **antes** de o modelo existir):

```bash
python3 shadow/resgate.py --sem-vitimas --debug
```

Com motores, depois de validar a visão:

```bash
python3 shadow/resgate.py --drive --camera-index 0 --debug
```

Testes:

```bash
python3 -m unittest discover -s shadow/tests -p "test_*.py"
```

## Limitações declaradas

1. **Nada foi testado no robô.** Todo o trabalho é offline.
2. **O modelo não existe ainda** — a visão de vítimas não funciona até o
   treino. É proposital que ela pare em vez de improvisar.
3. **Um círculo muito saturado no chão** seria aceito como marcador. Forma não
   discrimina nesta perspectiva; medido e documentado.
4. **A envoltória de plausibilidade** não tem dado de vítima distante nesta
   câmera. Está larga por isso.
5. **33% das fotos de vítima** deste robô têm a esfera cortada pela borda.
   Subir ou inclinar a câmera é o maior ganho isolado disponível, e não
   depende de qual detector se use.
