# Fases do projeto

1. **Fase 0 - Estrutura simples:** criar a base do projeto.
2. **Fase 1 - Controle de motores e comunicacao serial:** validar que a Raspberry consegue enviar comandos ao Arduino e que o Arduino controla cada motor, cada lado e todas as rodas com seguranca.
3. **Fase 2 - Teste da camera:** validar captura de imagem na Raspberry Pi 5 antes de implementar visao computacional e segue-linha.
4. **Fase 3 - Deteccao basica da linha:** detectar a linha preta por faixa BGR, gerar debug visual e calcular seu erro, sem mover o robo.
5. **Fase 4 - Comando sugerido de segue-linha:** calcular e imprimir o comando por faixas, sem mover o robo. A Fase 4.5 usa parametros suaves antes do primeiro movimento real.
6. **Fase 5 - Segue-linha real:** seguir continuamente por vetor, sem duracao fixa; a parada normal e por CTRL+C.
7. **Fase 6 - Recuperacao e curvas fechadas:** usar curva forte com `LADO` e Tank Assist por pulsos para recuperar perda de linha, sem rotina fixa de 90 graus.
8. **Fase 7 - Segue-linha limpo:** usar `follow_clean.py` com centroline conectada, lookahead e `LADO` assinado; a calibracao 1.1 entra em extrema mais cedo e força uma roda interna negativa quando a curva extrema esta visivel. Recuperar somente quando a linha ou o guia estiver invalido.
9. **Fase 8 - Verde/vermelho:** detectar marcacoes da pista.
10. **Fase 9 - Obstaculos e ajustes OBR:** tratar situacoes especiais.
