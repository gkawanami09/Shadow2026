Detecção de verde, interseções e beco
Objetivo

Detectar marcações verdes oficiais da OBR e tomar a decisão correta sem atrapalhar o segue-linha.

Regra OBR
Marcação verde esperada: 2,5 cm x 2,5 cm.
O verde fica logo antes da interseção.
O verde fica na região interna da curva.
Verde antes da interseção indica o caminho correto.
Interseção sem verde antes dela significa seguir reto.
Verde depois da interseção não deve controlar a interseção atual.
Dois verdes antes da interseção indicam beco sem saída: seguir o sentido oposto indicado pelas marcações.
Princípios de visão
Detectar verde em HSV.
Usar área mínima e máxima.
Usar posição na imagem.
Confirmar se está na região da pista.
Não aceitar qualquer pixel verde como marcador.
Não usar verde sozinho; combinar cor + área + posição + contexto de interseção.
Classificações possíveis
NO_GREEN

Nenhum verde confiável.
Comportamento:

Continuar LINE_FOLLOW.
Em interseção sem verde confirmado, seguir reto.
GREEN_LEFT

Verde confirmado à esquerda antes da interseção.
Comportamento:

Preparar decisão para esquerda.
Não virar antes de confirmar interseção.
GREEN_RIGHT

Verde confirmado à direita antes da interseção.
Comportamento:

Preparar decisão para direita.
Não virar antes de confirmar interseção.
GREEN_DOUBLE / DEAD_END_MARKER

Dois verdes confirmados antes da interseção.
Comportamento:

Tratar como beco sem saída.
Seguir o sentido oposto indicado pelos marcadores.
A lógica deve fazer o robô retornar/seguir de volta para a linha correta.
FALSE_GREEN

Verde pequeno, distante, alto demais na imagem, fora da pista, reflexo ou objeto externo.
Comportamento:

Ignorar.
Continuar segue-linha.
Cuidados
Verde não pode fazer o robô abandonar a linha cedo demais.
Verde só deve gerar decisão quando estiver antes da interseção.
Verde precisa ter confirmação por alguns frames ou confiança suficiente.
Não confundir objeto verde no fundo com marcador de pista.
Não confundir área de resgate verde com marcador verde do percurso.
Quando estiver dentro da sala de resgate, desativar lógica de marcador verde do percurso.

## Responsabilidade correta do verde.py

O arquivo `shadow/visao/verde.py` deve detectar apenas marcadores verdes.

Decisões permitidas no detector de verde:

- NENHUM
- ESQUERDA
- DIREITA
- RETORNO
- INSEGURO

Por enquanto, `verdes.py` NÃO deve retornar `RETO`.

Mesmo que a regra da OBR diga que interseção sem verde significa seguir reto, essa decisão deve ficar para a lógica de navegação/interseção do `follow_destinos.py`, não para o detector de verde.

Motivo:
Se o detector de cruzamento errar em curva normal, ele pode gerar falso `RETO`. Então, no detector de verdes, cruzamento sem verde deve retornar `NENHUM`.

Resumo:
- verde esquerdo -> ESQUERDA
- verde direito -> DIREITA
- dois verdes -> RETORNO
- sem verde -> NENHUM
- dúvida -> INSEGURO
- nunca retornar RETO nesta fase

## Referência Overengineering

Usar como referência conceitual:
https://github.com/Overengineering-squared/Overengineering-squared-RoboCup

Estratégia a adaptar:
- detectar contorno verde;
- filtrar por tamanho/área;
- analisar preto ao redor do marcador verde;
- decidir lado pela relação entre verde e linha preta;
- usar confirmação temporal futuramente.

Não copiar coordenadas fixas, porque a câmera do Shadow é frontal, inclinada e vê o verde antes do robô chegar na interseção.
