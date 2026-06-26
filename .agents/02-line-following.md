# Segue-linha

## Objetivo
Fazer o robô seguir a linha preta de forma estável usando câmera.

## Princípios
- Detectar a linha preta na região inferior da imagem.
- Calcular o centro da linha.
- Calcular o erro entre o centro da linha e o centro da imagem.
- Usar esse erro para ajustar os motores.
- Evitar movimentos bruscos e zigue-zague excessivo.

## Detecção
- Preferir ROI inferior da imagem.
- Usar máscara para preto ou threshold adaptativo/Otsu.
- Filtrar ruídos pequenos.
- Escolher o maior contorno ou o contorno mais provável da linha.
- Ignorar manchas pequenas que não parecem linha.

## Controle
- Se a linha estiver à esquerda, corrigir para a esquerda.
- Se a linha estiver à direita, corrigir para a direita.
- Se a linha estiver centralizada, seguir reto.
- Manter velocidades seguras para evitar perder a linha em curva.

## Quando a linha some
- Não parar imediatamente.
- Usar o último erro conhecido por um curto tempo.
- Tentar recuperar suavemente.
- Se continuar sem linha, passar para lógica de gap/perda de linha.

## Cuidados
- Não deixar a detecção de verde substituir a linha preta.
- Não fazer o robô girar agressivamente por qualquer perda momentânea.
- Não remover fallback de busca/recuperação.