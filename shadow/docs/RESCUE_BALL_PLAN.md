# Plano de resgate — detectar, aproximar e iniciar a coleta

Esta etapa é deliberadamente independente do segue-linha. A lógica de linha,
a máquina de estados existente e o firmware não foram alterados; a única
integração no controle de linha é enviar `LED ACESO` ao iniciar esse modo.

## Escopo

```text
WAIT_TARGET -> ALIGN -> APPROACH -> NEAR_CONFIRM -> NEAR
                    \-> LOST/FAULT (PARAR)
NEAR -> PICKUP_BACKUP -> PICKUP_FUTABA -> PICKUP_GRIPPERS -> PICKUP_COMPLETE
```

- `WAIT_TARGET`: motores parados até uma esfera aparecer de forma consistente.
- `ALIGN`: pivô lento, sem avanço, para centralizar a esfera.
- `APPROACH`: avanço em arco; a velocidade diminui à medida que a esfera cresce.
- `NEAR_CONFIRM`: para imediatamente e confirma por três frames que a esfera
  entrou na faixa visual da garra.
- `NEAR`: parada confirmada e transferência única para a coleta.
- `LOST`: qualquer perda ou imagem antiga produz `PARAR` imediatamente.
- `FAULT`: timeout ou falta de progresso produz `PARAR` travado.
- `PICKUP_BACKUP`: ré reta por 1,50 s a velocidade 0,35.
- `PICKUP_FUTABA`: rodas zeradas e `FUTABA -20 1500`; aguarda 1,50 s
  mais 0,10 s de margem.
- `PICKUP_GRIPPERS`: envia esquerda `-50` e direita `+50` no mesmo pacote
  USB e avança reto por 1,50 s a velocidade 0,35 enquanto elas fecham.
- `PICKUP_COMPLETE`: envia `PARAR` e encerra.

Ainda não há busca cega por rotação, transporte, depósito ou navegação completa
pela zona.

## Câmera

O `dual_camera_viewer.py` chama por convenção a câmera `0` de resgate. Ao mesmo
tempo, o segue-linha antigo abre `Picamera2()` sem índice. Por isso:

1. o resgate abre uma câmera por índice explícito;
2. o padrão é `0`, seguindo o viewer já existente;
3. `--camera-index` permite corrigir o mapeamento sem tocar no segue-linha;
4. o programa imprime `Picamera2.global_camera_info()` no início;
5. no OV5647 frontal, o modo full-FoV já medido (`1296x972`, 10-bit) é pedido
   diretamente, evitando a consulta lenta de todos os modos na partida;
6. sensores diferentes ainda usam descoberta automática do maior campo;
7. a saída preserva toda a proporção em até `640x480`, sem esticar círculos;
8. a câmera frontal é rotacionada em 180° por estar montada de ponta-cabeça;
9. somente esta câmera frontal participa do resgate; a câmera de segue-linha
   não é aberta por `rescue_main.py`;
10. `shadow/main.py` e `shadow/rescue_main.py` nunca devem rodar juntos.

Antes de liberar motores, execute:

```bash
python3 shadow/rescue_main.py --camera-index 0 --debug
```

Se a janela não mostrar a câmera frontal de resgate, encerre e teste índice `1`.

## Visão

O sensor continua usando o campo de visão completo, mas a saída de trabalho é
`640x480`. A resolução de saída menor não recorta a imagem; ela reduz somente
o número de pixels processados. Captura, controle/preview e detecção agora são
independentes: tanto a captura quanto o detector publicam apenas o frame mais
recente, portanto uma etapa lenta nunca forma uma fila atrasada. A visão roda
em `320x240` e suas coordenadas são remapeadas para o preview `640x480`. A
janela mostra FPS da câmera, FPS da visão, tempo de processamento, `C` para o
caminho rápido de contornos ou `H` para fallback Hough, número de candidatos
aceitos/propostas Hough brutas, motivo principal de rejeição, até quatro raios
aceitos e frames descartados. Por exemplo, `H1/5:ok r42` significa que cinco
círculos foram propostos, um passou pelos filtros e seu raio no preview é 42 px.

O preview também desenha a `FAIXA GARRA`: o corredor inferior fica amarelo
enquanto a esfera está fora, laranja em `1/3` e `2/3`, e verde em `3/3`.
Quando a esfera prateada está tão perto que sai parcialmente do quadro, o
tracker pode escolher um reflexo interno pequeno. Nesse caso, a faixa exige
também confiança, histórico e pelo menos dois círculos externos grandes entre
os candidatos do mesmo frame. Os dois círculos precisam envolver espacialmente
o reflexo rastreado, ser classificados como prateados e manter o centro do
envelope dentro do corredor da garra; círculos grandes de outro objeto ou uma
esfera deslocada para o lado não podem armar a coleta.
O watchdog também considera a descida da esfera na imagem, além do aumento do
raio, para não declarar falta de progresso durante essa aproximação final.

Todos os limites medidos em pixels usam uma escala isotrópica derivada de
`640x480`. O detector aplica:

1. CLAHE e gamma no canal de luminosidade LAB;
2. filtro mediano;
3. Canny automático e fechamento morfológico;
4. propostas por contornos de borda e máscara escura;
5. Hough somente como fallback no mesmo frame quando os contornos falham;
6. raio, proporção, circularidade e preenchimento;
7. suporte de borda ao redor da circunferência;
8. contraste entre interior e anel de piso próximo;
9. aparência escura para esfera preta;
10. faixa dinâmica e reflexo para esfera prateada; baixa saturação aumenta a
    confiança, mas iluminação ciano/verde possui uma rota metálica mais estrita;
11. classificação de todas as propostas Hough antes da deduplicação, para um
    halo inválido não apagar o perímetro verdadeiro;
12. preferência pelo envelope externo somente quando um círculo menor está
    realmente contido nele e as duas confianças são compatíveis;
13. associação espacial mais rígida durante os três hits de aquisição,
    suavização e confirmação em vários frames.

Uma detecção incerta não movimenta o robô.

## Sequência dos atuadores

O controlador de aproximação permanece independente. Somente depois da
proximidade confirmada ele arma um sequenciador monotônico separado. Cada ação
do Futaba e das garras possui latch one-shot, pois os comandos das garras são
deslocamentos relativos e não podem ser repetidos.

`PARAR` também corta o Futaba no firmware. Por isso, ao terminar a ré, o
programa usa `LADO 0 0` para zerar as quatro rodas. O keepalive repete esse
comando enquanto CH3 desce, sem interromper os 1500 ms. Depois do prazo, o
programa envia `FUTABA PARAR` por segurança e as duas linhas das garras em uma
única escrita serial. Confirmado o lote das garras, inicia o avanço reto; o
cronômetro de 1,50 s começa somente depois que todas essas escritas retornam.
Se as garras não receberem o comando, o avanço não começa.

O PCA9685 precisa de alimentação externa regulada adequada para os servos, com
GND comum ao Arduino. Não alimente Futaba e garras pelo pino 5 V do Uno.

### Dataset da câmera montada

No modo `--debug` sem `--drive`, pressione `s` para salvar o frame bruto atual.
O programa não salva o texto, as linhas ou os círculos do preview. A codificação
PNG e a escrita acontecem em um worker com uma única posição pendente, portanto
não formam uma fila capaz de atrasar o controle. Os pares ficam em:

```text
shadow/captures/rescue_dataset/session_.../
├── frame_....png
└── frame_....json
```

O JSON registra estado, diagnóstico, propostas, raios e a sequência temporal.
Como o detector e a câmera são assíncronos, `same_frame` informa explicitamente
se esses dados pertencem exatamente ao PNG ou se são apenas contexto do frame
anterior. A tecla é recusada com `--drive`; faça a calibração com os motores
desativados e o LED frontal apagado.

Para calibrar a variação espacial da iluminação, colete a esfera prateada em
esquerda/centro/direita, três distâncias e mais de uma iluminação. Inclua também
piso/parede sem esfera e reflexos que não podem ser confundidos com ela. Estes
arquivos formam um dataset de calibração do detector OpenCV; não há uma rede
neural sendo treinada nesta etapa.

## Parada e segurança

A chegada perto da bolinha é decidida exclusivamente pela câmera frontal:
posição inferior, centralização e evidências do envelope externo precisam ser
confirmadas em três frames. O primeiro indício já produz `PARAR`, antes de
confirmar e armar a coleta. O resgate não consulta o HC-SR04, porque uma vítima
esférica pode desviar o eco e uma parede pode gerar uma falsa proximidade.

Outras travas:

- ao usar a câmera real, a serial é aberta, os motores recebem `PARAR` e o
  comando `LED APAGADO` é enviado antes da abertura da câmera;
- se a USB reconectar e reiniciar o Uno, o modo do LED é reaplicado
  automaticamente;
- `--drive` continua obrigatório para permitir qualquer movimento; sem ele, a
  serial fica aberta apenas para manter `PARAR` e o LED apagado;
- um lock de sistema impede segue-linha e resgate de comandarem os motores ao
  mesmo tempo;
- contagem regressiva de 3 segundos com preview já funcionando e `PARAR`;
- perda da esfera = `PARAR`;
- nenhuma leitura ultrassônica pode alterar, pausar ou encerrar a aproximação;
- frame antigo = `PARAR`;
- o detector descarta backlog e nunca repete o mesmo resultado no controle;
- captura travada não bloqueia o watchdog de imagem nem o envio de `PARAR`;
- após imagem antiga, o controle exige três resultados distintos e frescos;
- uma busca Hough antiga pode apenas semear a posição interna; nunca gera
  movimento e ainda precisa das três verificações frescas;
- após reconexão serial, `PARAR` substitui qualquer movimento antigo;
- o timestamp é obtido depois da captura, evitando classificar como antigo um
  frame que apenas aguardou a câmera;
- aproximação sem aumento do raio = `FAULT`;
- timeout = `FAULT`;
- falha de escrita de ré, Futaba ou garras = `PICKUP_FAULT`, sem avançar para
  a próxima ação;
- reconexão serial durante a coleta cancela a sequência, pois o Uno pode ter
  reiniciado a posição relativa das garras;
- `finally` sempre envia `PARAR`, `FUTABA PARAR` e fecha a serial;
- PWM continua limitado pelo código existente e pelo firmware.

## Ordem de validação

1. Testes offline:

   ```bash
   python3 -m unittest discover -s shadow/tests -p "test_rescue_*.py" -v
   ```

2. Visão, rodas suspensas ou robô desligado:

   ```bash
   python3 shadow/rescue_main.py --camera-index 0 --debug
   ```

3. Pressionar `s` e testar cenas reais: piso vazio, sombra, parede, reflexo,
   esfera preta e prateada em 15, 20, 30, 50 e 80 cm.

4. Ajustar somente `shadow/rescue_config.py`, principalmente ROI, raios,
   confiança e parada.

5. Primeiro movimento com rodas suspensas:

   ```bash
   python3 shadow/rescue_main.py --camera-index 0 --drive --debug
   ```

   No modo com motores, `--camera-index` é obrigatório: o programa não aceita
   assumir silenciosamente qual câmera física deve comandar o robô.

6. Ainda com as rodas suspensas e sem bolinha presa, confirme no log a ordem:
   `PICKUP_BACKUP`, `PICKUP_FUTABA`, `PICKUP_GRIPPERS`,
   `PICKUP_COMPLETE`. Em `PICKUP_GRIPPERS`, as rodas devem avançar por 1,50 s
   enquanto os servos fecham. Mantenha acesso imediato à alimentação.

7. Teste no chão em velocidade baixa.

## Limite desta implementação

As capturas de tela já permitiram corrigir a dominante ciano e a competição
entre o perímetro da esfera, seus reflexos internos e halos. Elas, porém,
contêm a anotação do programa e não permitem reproduzir todos os estágios do
detector. A calibração de competição deve usar os PNGs brutos salvos com `s`,
especialmente para a esfera prateada no centro e nos cantos da imagem.
