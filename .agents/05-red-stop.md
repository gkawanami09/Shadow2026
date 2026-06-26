# Parada no vermelho

## Objetivo
Detectar vermelho na pista e parar conforme a regra definida.

## Detecção
- Usar HSV para vermelho.
- Lembrar que vermelho pode aparecer em duas faixas de H:
  - próximo de 0
  - próximo de 180
- Usar área mínima.
- Confirmar se o vermelho está na região da pista.
- Evitar falso positivo por reflexo, objeto externo ou iluminação.

## Comportamento esperado
Quando vermelho real for detectado:
- Parar o robô.
- Manter parado pelo tempo definido na regra/código.
- Não confundir vermelho com sombra, reflexo ou objeto fora da pista.

## Cuidados
- Vermelho não deve ser detectado em qualquer lugar da imagem.
- Deve haver confirmação por área, posição e contexto.
- Não parar por pequenos pixels vermelhos.