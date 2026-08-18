# Arquitetura do Shadow2026

`shadow/mission.py` é o controlador central da execução autônoma. Ele é a
única autoridade sobre os estados globais, permanece ativo durante a prova e
chama diretamente a rotina de resgate. `main.py` e `resgate.py` continuam como
diagnósticos isolados e nunca devem rodar juntos nem ao lado de `mission.py`.

A ordem da troca (qual câmera fecha antes de qual abrir, quando a serial muda
de dono, quando o LED apaga) é o contrato de segurança da missão e está
documentada em **`MISSAO_COMPLETA.md`**.

## Segue-linha

No diagnóstico isolado, `shadow/main.py` inicia dois processos. Na missão,
`mission.py` inicia os mesmos alvos e supervisiona sua prontidão:

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
├── fase resgate: chamada direta (câmera 0, serial, LED apagado)
└── finalização: confirma saída, limpa estado e retoma o percurso
```

- `mission.py::EstadoMissao`: estados globais e transições permitidas;
- `controle/missao.py`: políticas e ordem declarada do handoff;
- `visao/entrada_missao.py`: modelo `entrada.onnx` para a faixa prata,
  executado somente no processo de visão do percurso quando habilitado (o
  perfil atual de bancada usa o gatilho temporário de ausência de preto);
- `visao/faixa_transversal.py`: geometria comum às duas faixas e a votação
  temporal com histerese e cooldown.

O estado `RESGATE` não pode transicionar diretamente para `SEGUE_LINHA`.
Somente `FINALIZANDO_RESGATE`, após depósito e saída confirmados, libera o
handoff normal. Falhas ficam paradas no resgate ou entram em `RECONECTANDO`.

## Arduino

O arquivo usado continua sendo
`arduino/motor_controller/motor_controller.ino`. O firmware recebe comandos
como `LADO`, `PARAR`, `GARRAS`, `FUTABA` e `LED`. Se a comunicação parar, o
watchdog do Arduino corta os motores.
