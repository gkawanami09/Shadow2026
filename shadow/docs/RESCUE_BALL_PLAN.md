# Plano de resgate — etapa 1: detectar e aproximar da bolinha

Esta etapa é deliberadamente independente do segue-linha. A lógica de linha,
a máquina de estados existente e o firmware não foram alterados; a única
integração no controle de linha é enviar `LED ACESO` ao iniciar esse modo.

## Escopo

```text
WAIT_TARGET -> ALIGN -> APPROACH -> NEAR
                    \-> LOST/FAULT (PARAR)
```

- `WAIT_TARGET`: motores parados até uma esfera aparecer de forma consistente.
- `ALIGN`: pivô lento, sem avanço, para centralizar a esfera.
- `APPROACH`: avanço em arco; a velocidade diminui à medida que a esfera cresce.
- `NEAR`: parada travada, encerrando esta primeira etapa.
- `LOST`: qualquer perda ou imagem antiga produz `PARAR` imediatamente.
- `FAULT`: timeout ou falta de progresso produz `PARAR` travado.

Ainda não há busca cega, coleta, transporte, depósito ou navegação pela zona.

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
9. `shadow/main.py` e `shadow/rescue_main.py` nunca devem rodar juntos.

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
caminho rápido de contornos ou `H` para fallback Hough, número de candidatos,
motivo principal de rejeição e frames descartados.

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
11. associação espacial, suavização e confirmação em vários frames.

Uma detecção incerta não movimenta o robô.

## Parada e segurança

A parada principal usa simultaneamente raio aparente, posição inferior e
centralização, confirmados em vários frames. O HC-SR04 é apenas uma barreira
auxiliar e só pode parar quando a esfera ainda está confirmada e centralizada.

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
- o primeiro eco ultrassônico próximo já produz uma parada provisória; o
  segundo confirma a chegada ou acusa obstáculo fora do eixo;
- a consulta ultrassônica é assíncrona: o comando é enviado e a resposta é
  consultada em ciclos posteriores, sem pausar a janela durante o timeout;
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
- `finally` sempre envia `PARAR` e fecha a serial;
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

3. Gravar e testar cenas reais: piso vazio, sombra, parede, reflexo, esfera
   preta e prateada em 15, 20, 30, 50 e 80 cm.

4. Ajustar somente `shadow/rescue_config.py`, principalmente ROI, raios,
   confiança e parada.

5. Primeiro movimento com rodas suspensas:

   ```bash
   python3 shadow/rescue_main.py --camera-index 0 --drive --debug
   ```

   No modo com motores, `--camera-index` é obrigatório: o programa não aceita
   assumir silenciosamente qual câmera física deve comandar o robô.

6. Teste no chão em velocidade baixa, com acesso imediato à alimentação.

## Limite desta implementação

Sem imagens reais da câmera montada, os limiares são valores iniciais seguros,
não calibração de competição. O objetivo desta versão é oferecer a arquitetura,
os filtros, as travas e um caminho reproduzível para calibrar. Precisão real,
especialmente na esfera prateada, deve ser medida com capturas do próprio robô.
