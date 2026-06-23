# Fases do projeto

1. **Fase 0 - Estrutura simples:** criar a base do projeto.
2. **Fase 1 - Controle de motores e comunicacao serial:** validar que a Raspberry consegue enviar comandos ao Arduino e que o Arduino controla cada motor, cada lado e todas as rodas com seguranca.
3. **Fase 2 - Teste da camera:** validar captura de imagem na Raspberry Pi 5 antes de implementar visao computacional e segue-linha.
4. **Fase 3 - Deteccao basica da linha:** detectar a linha preta por faixa BGR, gerar debug visual e calcular seu erro, sem mover o robo.
5. **Fase 4 - Comando sugerido de segue-linha:** calcular e imprimir o comando por faixas, sem mover o robo.
6. **Fase 5 - Segue-linha real:** iniciar o controle pela linha em baixa velocidade.
7. **Fase 6 - Verde/vermelho:** detectar marcacoes da pista.
8. **Fase 7 - Obstaculos e ajustes OBR:** tratar situacoes especiais.
