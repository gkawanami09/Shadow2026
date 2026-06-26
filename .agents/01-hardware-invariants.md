# Hardware e regras fixas

## Robô
O projeto é um robô seguidor de linha para OBR/Resgate.

## Controle
- Raspberry Pi processa câmera e visão computacional.
- Arduino controla motores.
- Drivers usados: TB6612FNG.
- Motores: 4 motores DC, divididos entre lado esquerdo e lado direito.

## Câmera
- Câmera frontal usada para seguir linha e detectar cores.
- A câmera fica baixa, próxima do chão.
- Usar preferencialmente a região inferior da imagem para seguir linha.
- Considerar distorção, variação de luz e reflexos.

## Regras importantes
- Não mudar pinout sem confirmação.
- Não inverter lógica dos motores sem testar.
- Não assumir que os motores são perfeitamente iguais.
- Não assumir que o robô gira exatamente pelo tempo definido.
- Sempre considerar derrapagem, bateria fraca, diferença entre rodas e iluminação variável.

## Comunicação
Se existir comunicação serial entre Raspberry Pi e Arduino, preserve os comandos existentes. Não mude o protocolo sem explicar a razão.