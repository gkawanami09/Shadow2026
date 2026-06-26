Gap e recuperação de linha
Objetivo

Quando a linha preta desaparecer, o robô precisa diferenciar:

Perda momentânea da linha.
Gap real.
Curva muito fechada.
Beco/fim de caminho.
Erro de trajetória.
Regra OBR
Gap aparece em trecho reto.
Gap não deve ter comprimento superior a 10 cm.
Deve haver pelo menos 5 cm de linha reta antes do gap.
Pontua quando o robô supera o ladrilho e continua seguindo linha.
Regra principal

Não girar imediatamente quando perder a linha.

Estratégia esperada
1. Perda curta

Se a linha sumir por poucos frames:

Manter movimento suave.
Usar último erro conhecido.
Reduzir correção brusca.
Não entrar em giro agressivo.
2. Possível gap

Se a linha sumir em contexto de trecho reto:

Reduzir velocidade.
Seguir reto por tempo/distância controlada.
Procurar continuação da linha à frente.
Se encontrar linha, voltar para LINE_FOLLOW.
3. Possível curva fechada

Se a linha sumir depois de erro lateral alto:

Procurar para o lado do último erro.
Não assumir gap imediatamente.
Fazer busca limitada por tempo.
4. Linha não encontrada

Se não achar linha dentro do limite:

Entrar em busca segura.
Evitar giro infinito.
Evitar ficar parado por 10 segundos.
Gerar condição de recuperação/falha controlada.
Cuidados
Não tratar todo sumiço de linha como gap.
Não tratar todo sumiço de linha como beco.
Não acelerar durante gap.
Não buscar por tempo ilimitado.
Não seguir uma linha aleatória que não esteja na sequência correta.
O robô deve tentar encontrar a continuação no mesmo ladrilho ou no ladrilho seguinte.