# Plano de resgate — etapa 1: detectar e aproximar da bolinha

Esta etapa é deliberadamente independente do segue-linha. Nenhum arquivo do
pipeline atual de linha, da máquina de estados existente ou do firmware foi
alterado.

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
5. `shadow/main.py` e `shadow/rescue_main.py` nunca devem rodar juntos.

Antes de liberar motores, execute:

```bash
python3 shadow/rescue_main.py --camera-index 0 --debug
```

Se a janela não mostrar a câmera frontal de resgate, encerre e teste índice `1`.

## Visão

O detector conserva `640x480` (4:3), evitando transformar círculos em elipses.
Ele aplica:

1. CLAHE e gamma no canal de luminosidade LAB;
2. filtro mediano;
3. Canny automático e fechamento morfológico;
4. propostas por Hough, contornos de borda e máscara escura;
5. raio, proporção, circularidade e preenchimento;
6. suporte de borda ao redor da circunferência;
7. contraste entre interior e anel de piso próximo;
8. aparência escura para esfera preta;
9. baixa saturação, faixa dinâmica e reflexo para esfera prateada;
10. associação espacial, suavização e confirmação em vários frames.

Uma detecção incerta não movimenta o robô.

## Parada e segurança

A parada principal usa simultaneamente raio aparente, posição inferior e
centralização, confirmados em vários frames. O HC-SR04 é apenas uma barreira
auxiliar e só pode parar quando a esfera ainda está confirmada e centralizada.

Outras travas:

- `--drive` é obrigatório para abrir a serial;
- um lock de sistema impede segue-linha e resgate de comandarem os motores ao
  mesmo tempo;
- contagem regressiva de 3 segundos;
- perda da esfera = `PARAR`;
- o primeiro eco ultrassônico próximo já produz uma parada provisória; o
  segundo confirma a chegada ou acusa obstáculo fora do eixo;
- frame antigo = `PARAR`;
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
