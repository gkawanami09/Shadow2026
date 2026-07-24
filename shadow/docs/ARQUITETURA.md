# Arquitetura do Shadow2026

O percurso e o resgate são programas separados. Eles nunca devem rodar ao
mesmo tempo porque compartilham a serial e os motores.

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
- `controle/aproximacao_resgate.py`: alinha e aproxima;
- `controle/coleta_resgate.py`: comanda garras e elevador;
- `controle/trava_motores.py`: impede dois programas de controlar os motores.

Sem `--drive`, o programa mantém os motores parados e serve apenas para
conferir a visão.

## Arduino

O arquivo usado continua sendo
`arduino/motor_controller/motor_controller.ino`. O firmware recebe comandos
como `LADO`, `PARAR`, `GARRAS`, `FUTABA` e `LED`. Se a comunicação parar, o
watchdog do Arduino corta os motores.
