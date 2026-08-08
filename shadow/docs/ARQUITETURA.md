# Arquitetura do Shadow2026

O percurso e o resgate são programas separados. Eles nunca devem rodar ao
mesmo tempo porque compartilham a serial e os motores.

Desde a missão completa existe um terceiro programa, `shadow/mission.py`, que
alterna entre os dois automaticamente. Ele não substitui nenhum deles: o
percurso e o resgate continuam funcionando isolados, exatamente como antes, e
continuam sendo a forma recomendada de depurar cada metade.

A ordem da troca (qual câmera fecha antes de qual abrir, quando a serial muda
de dono, quando o LED apaga) é o contrato de segurança da missão e está
documentada em **`MISSAO_COMPLETA.md`**.

## Segue-linha

`shadow/main.py` inicia dois processos:

```text
main.py
├── visão: captura a câmera 1 e encontra linha, verde e vermelho
└── controle: decide o movimento e é o único dono da serial
```

A visão escreve os resultados em `shared/dados_compartilhados.py`. O controle
lê esses valores e envia os comandos ao Arduino usando
`comunicacao_serial/arduino.py`.

Quando `--debug` está ativo, a imagem anotada passa da visão para o processo
principal por memória compartilhada.

### Valores compartilhados principais

| Valor | Uso |
|---|---|
| `line_angle` | correção necessária para seguir a linha |
| `line_detected` | informa se existe uma linha válida |
| `line_size` | área do contorno seguido |
| `line_ahead` | informa se existe continuação à frente |
| `last_bottom_point` | posição da linha perto da base da imagem |
| `turn_dir` | decisão dos marcadores verdes |
| `red_detected` | faixa vermelha encontrada |
| `gap_angle`, `gap_center_x/y` | geometria usada na validação do gap |
| `line_status` | estado atual do segue-linha |
| `status` | texto mostrado no terminal e no debug |
| `terminate`, `vision_ready` | inicialização e encerramento dos processos |

### Estados do percurso

- `line_detected`: segue a linha e executa as decisões de verde;
- `gap_detected`: confirma se a perda de linha é realmente um gap;
- `gap_avoid`: atravessa o gap procurando a continuação;
- `stop`: permanece parado sobre a faixa vermelha.

## Resgate

`shadow/resgate.py` usa a câmera 0 e mantém três tarefas:

```text
câmera mais recente ──► detector mais recente ──► controle e coleta
```

As filas guardam somente a imagem mais nova. Isso evita que o robô tome uma
decisão usando uma imagem atrasada.

O resgate possui:

- `visao/captura_resgate.py`: abre somente a câmera frontal;
- `visao/bola_resgate.py`: encontra e acompanha as vítimas;
- `visao/resgate_assincrono.py`: descarta imagens antigas;
- `visao/faixa_saida.py`: soleira preta de saída (só na fase de saída);
- `visao/triangulos_finais.py`: mapeia os dois triângulos no fim;
- `controle/aproximacao_resgate.py`: alinha e aproxima;
- `controle/coleta_resgate.py`: comanda garras e elevador;
- `controle/busca_pulsada.py`: busca "gira e observa" em modo tanque;
- `controle/saida_resgate.py`: encontra e atravessa a soleira de saída;
- `controle/trava_motores.py`: impede dois programas de controlar os motores.

## Missão completa

`shadow/mission.py` coordena as duas metades:

```text
mission.py
├── fase percurso: sobe visão (câmera 1) + controle (serial, LED aceso)
├── handoff: encerra os filhos, confirma que morreram, libera a trava
└── fase resgate: subprocesso resgate.py (câmera 0, serial, LED apagado)
```

- `controle/missao.py`: máquina de estados, inventário das três vítimas e a
  ordem declarada do handoff;
- `visao/entrada_missao.py`: modelo `entrada.onnx` para a faixa prata,
  executado somente no processo de visão do percurso durante a missão;
- `visao/faixa_transversal.py`: geometria comum às duas faixas e a votação
  temporal com histerese e cooldown.

Sem `--drive`, o programa mantém os motores parados e serve apenas para
conferir a visão.

## Arduino

O arquivo usado continua sendo
`arduino/motor_controller/motor_controller.ino`. O firmware recebe comandos
como `LADO`, `PARAR`, `GARRAS`, `FUTABA` e `LED`. Se a comunicação parar, o
watchdog do Arduino corta os motores.
