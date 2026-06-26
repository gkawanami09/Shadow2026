# Máquina de estados do robô

## Objetivo
Organizar o comportamento do robô em estados claros.

## Estados principais

### LINE_FOLLOW
Estado normal.
- Segue a linha preta.
- Calcula erro lateral.
- Ajusta motores.
- Observa possíveis eventos: verde, vermelho, perda de linha ou prata.

### GREEN_CHECK
Estado de confirmação do verde.
- Verifica se o verde é real.
- Classifica esquerda, direita, duplo ou falso.
- Só toma decisão se houver confiança suficiente.

### INTERSECTION
Estado para decisões em cruzamentos/interseções.
- Usa informação do verde.
- Evita decisões precipitadas.
- Depois da manobra, volta para LINE_FOLLOW.

### GAP_RECOVERY
Estado para perda de linha/gap.
- Tenta recuperar a linha.
- Se for gap provável, segue reto controlado.
- Se encontrar a linha, volta para LINE_FOLLOW.
- Se não encontrar, entra em busca segura.

### RED_STOP
Estado de parada no vermelho.
- Para o robô.
- Aguarda tempo definido.
- Depois decide se continua ou encerra conforme regra.

### SILVER_DETECTED
Estado futuro.
- Usado para identificar possível entrada da sala de resgate.
- Ainda não deve ativar sala de resgate completa sem implementação específica.

### RESCUE_ROOM
Estado futuro.
- Lógica separada para sala de resgate.

## Regras
- Um estado não deve fazer tudo.
- Cada estado deve ter entrada, comportamento e saída claros.
- Evitar vários ifs soltos competindo entre si.
- Verde, vermelho, gap e prata devem ser eventos, não bagunça dentro do segue-linha.