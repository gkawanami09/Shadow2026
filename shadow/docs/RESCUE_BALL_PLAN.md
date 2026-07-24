# Plano de resgate — detectar, aproximar e iniciar a coleta

Esta etapa é deliberadamente independente do segue-linha. A lógica de linha,
a máquina de estados existente e o firmware não foram alterados; a única
integração no controle de linha é enviar `LED ACESO` ao iniciar esse modo.

## Escopo

```text
WAIT_TARGET -> ALIGN -> APPROACH -> NEAR_CONFIRM -> NEAR
                    \-> LOST/FAULT (PARAR)
NEAR -> PICKUP_FUTABA -> PICKUP_FORWARD -> PICKUP_GRIPPERS
     -> PICKUP_COMPLETE
```

- `WAIT_TARGET`: motores parados até uma esfera aparecer de forma consistente.
- `ALIGN`: curva curta para a frente, proporcional ao erro e com histerese,
  para centralizar sem ultrapassar a esfera de um lado para o outro.
- `APPROACH`: avanço em arco; a velocidade diminui à medida que a esfera cresce.
- `NEAR_CONFIRM`: para imediatamente no primeiro círculo que toca o ponto
  inferior e exige uma segunda medição nova; a segunda pode ser o mesmo
  círculo ou a meia-lua do frame seguinte, se o perímetro já foi cortado.
- `NEAR`: parada confirmada e transferência única para a coleta.
- `LOST`: qualquer perda ou imagem antiga produz `PARAR` imediatamente.
- `FAULT`: timeout ou falta de progresso produz `PARAR` travado.
- `PICKUP_FUTABA`: rodas zeradas e `FUTABA -20 1500`; aguarda 1,50 s
  mais 0,10 s de margem.
- `PICKUP_FORWARD`: mantém as garras abertas durante os 2,00 s de avanço reto.
- `PICKUP_GRIPPERS`: ao final da reta, envia `PARAR` e só então esquerda `-50`
  e direita `+50` no mesmo pacote USB.
- `PICKUP_COMPLETE`: confirma a coleta e encerra com as rodas já paradas.

Ainda não há busca cega por rotação, transporte, depósito ou navegação completa
pela zona.

## Câmera

O mapeamento físico atual possui duas câmeras e nenhum processo pode depender
da câmera padrão do Picamera2:

1. resgate = índice `0`, aberto explicitamente por `rescue_main.py`;
2. segue-linha no flat 2 = índice `1`, fixado por `LINE_CAMERA_INDEX`;
3. se a câmera `1` estiver ausente, o segue-linha falha com mensagem clara em
   vez de abrir silenciosamente a câmera de resgate;
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

Se a janela não mostrar a câmera frontal de resgate, encerre e confira os
flats com `dual_camera_viewer.py`; não use o índice `1` no resgate, pois ele
está reservado ao segue-linha.

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

O preview desenha uma cruz `PONTO GARRA` em `(0,50W, 0,95H)`. O círculo
suavizado da esfera recebe o rótulo `LOCK` depois de três associações
consistentes. Quando esse mesmo círculo cobre o ponto, está centralizado, tem
raio de pelo menos `0,085H` e possui histórico real de avanço, o robô para no
primeiro indício e exige uma segunda medição nova em até 0,35 s. Não existe
folga acima do ponto: a borda calculada do círculo precisa realmente
alcançá-lo. O ROI chega até a base da imagem e aceita até `0,03H` do perímetro
já cortado; isso preserva vários frames para confirmar em 240p sem subir o
ponto físico. No máximo uma perda isolada pode preservar a confirmação por
0,18 s com as rodas paradas; duas perdas liberam o lock e reiniciam a
confirmação. Uma perda nunca conta como nova medição nem aciona a garra por
previsão.

O preview também desenha a `MEIA-LUA GARRA`: duas curvas delimitam a faixa onde
deve aparecer a borda superior da esfera enorme. Ela fica amarela fora do
gate, laranja durante a confirmação e verde quando a coleta é autorizada. Os
brutos de calibração mostraram que essa borda é um arco circular entre `0,62H`
e `0,74H`, ocupando de 80% a 92% da largura. O detector exige apoio no ombro
esquerdo, centro e ombro direito, além de continuidade, contraste e
preenchimento abaixo da borda.

A rota estrita exige a mesma componente de Canny, coerência de polaridade,
ajuste circular e curvatura distribuída. Uma segunda rota é exclusiva para o
papel-alumínio amassado: ela tolera pequenas interrupções unidas pelo
fechamento morfológico apenas quando há reflexos distribuídos em pelo menos
quatro de cinco setores internos e o fundo acima da esfera permanece muito
menos texturizado. O histórico de aproximação continua obrigatório nas duas
rotas. No marcador, `p=` mostra a coerência de polaridade, `q=` a curvatura
estrita e `f=4/5` ou `f=5/5` os setores metálicos da rota de foil.

Assim, uma esfera pequena distante pode até estar alinhada com o ponto, mas o
raio e o histórico impedem que ela arme a coleta por perspectiva. O círculo
rastreado é a rota primária enquanto o perímetro ainda cabe no quadro. A
meia-lua nunca inicia a coleta sozinha: ela só pode completar `2/2` logo após
o círculo travado produzir `1/2` no ponto inferior. Por segurança, o contato
também depende de um token curto criado durante uma aproximação anterior, com
vários círculos centralizados, baixos e crescendo enquanto o robô avança. Um
círculo grande ou uma meia-lua presentes desde a partida mantêm o robô parado;
nunca iniciam a coleta a frio.
O watchdog também considera a descida da esfera na imagem, além do aumento do
raio, para não declarar falta de progresso durante essa aproximação final.

Todos os limites medidos em pixels usam uma escala isotrópica derivada de
`640x480`. O detector aplica:

1. CLAHE e gamma no canal de luminosidade LAB;
2. filtro mediano;
3. Canny automático e fechamento morfológico;
4. propostas por contornos de borda e máscara escura;
5. Hough somente como fallback quando os contornos falham; ele continua
   disponível no frame da meia-lua para vetar um círculo ainda distante;
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
13. associação espacial e de tamanho mais rígida, lock após três hits e
    suavização temporal; uma falha curta preserva a identidade com o robô
    parado, enquanto um brilho incompatível não pode roubar o track;
14. gate primário pelo círculo travado cobrindo o ponto inferior, com duas
    medições novas, centralização, raio mínimo e histórico obrigatório;
15. arco circular largo em `320x240`, com rota estrita e fallback metálico,
    usado somente como segunda confirmação depois do contato inferior.

Uma detecção incerta não movimenta o robô.

## Sequência dos atuadores

O controlador de aproximação permanece independente. Somente depois da
proximidade confirmada ele arma um sequenciador monotônico separado. Cada ação
do Futaba e das garras possui latch one-shot, pois os comandos das garras são
deslocamentos relativos e não podem ser repetidos.

`PARAR` também corta o Futaba no firmware. Por isso, depois do `PARAR` que
finaliza a aproximação, o primeiro passo da coleta usa `LADO 0 0` para manter
as quatro rodas zeradas. O keepalive repete esse comando enquanto CH3 desce,
sem interromper os 1500 ms. Depois do prazo, o programa envia `FUTABA PARAR`
por segurança e inicia o avanço reto. O cronômetro de 2,00 s começa quando o
avanço é entregue. Durante toda a reta as garras permanecem abertas; no fim do
prazo o programa envia `PARAR` e depois fecha as duas garras em uma única
escrita serial. Não existe comando de ré nessa sequência. Se o lote das garras
falhar, o programa mantém `PARAR` e entra em `PICKUP_FAULT`.

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

A chegada perto da bolinha é decidida exclusivamente pela câmera frontal. A
primeira confirmação exige que o círculo travado alcance `0,95H`, sem folga.
A segunda medição fresca pode repetir o círculo ou usar a meia-lua larga,
centralizada e contrastada caso o perímetro tenha sido cortado. Uma meia-lua
sozinha nunca arma a coleta. O primeiro contato já produz `PARAR`, antes de
confirmar. O resgate não consulta o HC-SR04, porque uma vítima esférica pode
desviar o eco e uma parede pode gerar uma falsa proximidade.

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
- falha de escrita de Futaba, avanço ou garras = `PICKUP_FAULT`; se o
  avanço já começou, `PARAR` é enviado imediatamente;
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

4. Ajustar somente `shadow/rescue_config.py`, principalmente os ratios,
   suportes e contraste `BALL_CRESCENT_*`.

5. Primeiro movimento com rodas suspensas:

   ```bash
   python3 shadow/rescue_main.py --camera-index 0 --drive --debug
   ```

   No modo com motores, `--camera-index` é obrigatório: o programa não aceita
   assumir silenciosamente qual câmera física deve comandar o robô.

6. Ainda com as rodas suspensas e sem bolinha presa, confirme no log a ordem:
   `PICKUP_FUTABA`, `PICKUP_FORWARD`, `PICKUP_GRIPPERS`,
   `PICKUP_COMPLETE`. Não pode aparecer ré. O avanço precisa aparecer antes
   das garras; elas só podem fechar depois que a reta completar 2,00 s e as
   rodas receberem `PARAR`. Mantenha acesso imediato à alimentação.

7. Teste no chão em velocidade baixa.

## Limite desta implementação

Os PNGs brutos de 23/07/2026 permitiram corrigir a forma circular do domo e a
fragmentação causada pelos reflexos do papel-alumínio. Quando o JSON salvo
indicar `"same_frame": false`, os números do detector descrevem um frame
anterior e servem apenas como contexto; a decisão visual sobre aquele PNG deve
usar a própria imagem. A calibração de competição deve continuar coletando
PNGs brutos com `s`, especialmente no instante exato em que o domo entra na
faixa da garra.
