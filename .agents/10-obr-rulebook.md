Regras OBR essenciais para o Shadow
Fonte de verdade

Este arquivo resume regras importantes da OBR para orientar agentes de programação. Ele não substitui o regulamento oficial. Antes da competição, conferir a versão mais recente do manual no site oficial da OBR.

Filosofia da prova

O robô deve reconhecer a arena sozinho usando sensores. Não pode depender de mapa prévio, posição conhecida de obstáculos, quantidade de ladrilhos, entrada da sala, posição das áreas de resgate ou qualquer informação fornecida antes/durante a rodada.

Autonomia
O robô deve atuar de forma autônoma.
Não usar controle remoto.
Não receber comandos externos durante a rodada.
Não usar pré-mapeamento da arena.
O robô deve ser iniciado manualmente pelo capitão.
Arena
Ladrilhos padrão: aproximadamente 300 mm x 300 mm, com tolerância.
Piso claro: branco ou próximo de branco.
Linha escura/preta no chão.
Pode haver desnível entre ladrilhos.
A linha pode fazer curvas grandes, pequenas, 90 graus, zigue-zague, círculos, retas, interseções e becos.
Linha preta
Largura esperada: 1 cm a 2 cm.
A linha pode estar em fita isolante, papel, adesivo ou outro material.
O robô deve seguir a linha até imediatamente antes de entrar na sala de resgate.
Depois de sair da sala de resgate, deve voltar a seguir a linha até o ladrilho de chegada.
Gap
Gap é uma falha na linha.
Gap aparece em trecho reto.
Comprimento máximo esperado: 10 cm.
Deve existir pelo menos 5 cm de linha reta antes do gap.
Estratégia: não girar imediatamente; seguir reto controlado e procurar a continuação.
Verde
Marcação verde oficial: 2,5 cm x 2,5 cm.
Fica logo antes da interseção, na região interna da curva.
Verde antes da interseção indica a direção correta.
Interseção com verde: seguir o lado indicado pelo marcador.
Interseção sem verde antes dela: seguir reto.
Verde depois da interseção não deve ser usado para decidir aquela interseção.
Dois verdes antes da interseção indicam beco sem saída: seguir o sentido oposto indicado pelos marcadores.
Vermelho
Ladrilho de chegada possui faixa vermelha no centro.
Faixa vermelha esperada: aproximadamente 25 mm x 300 mm.
Para pontuar chegada, o robô deve parar sobre a faixa vermelha por pelo menos 5 segundos.
Se passar direto ou não parar na faixa, pode ser Falha de Progresso.
Prata e sala de resgate
A linha escura termina na entrada da sala de resgate.
A entrada da sala é demarcada por fita prata ou prata reflexiva.
Fita prata esperada: aproximadamente 25 mm x 250 mm, preenchendo a porta de entrada.
A saída da sala é demarcada por fita preta.
Fita preta de saída esperada: aproximadamente 25 mm x 250 mm.
A sala mede aproximadamente 90 cm x 90 cm ou 120 cm x 90 cm.
A sala possui paredes claras com no mínimo 10 cm de altura.
A porta da sala, se existir, pode ter 25 cm de largura e altura.
No estágio atual do Shadow, detectar prata deve apenas confirmar entrada da sala e mudar para estado de RESCUE_ENTRY_DETECTED. Não implementar resgate completo sem hardware.
Falha de Progresso

Evitar qualquer comportamento que gere Falha de Progresso:

perder a linha escura;
ficar parado por 10 segundos;
seguir caminho errado em interseção ou beco;
não passar pela entrada da sala;
não retornar à linha após obstáculo;
atingir linha que não está na sequência correta;
não encontrar a linha no mesmo ladrilho ou no ladrilho da sequência.
Após Falha de Progresso
A equipe só pode executar o procedimento informado ao árbitro.
Não alterar programa.
Não reparar o robô.
Não fornecer informação da arena ao robô.
Não trocar modo de execução com base no mapa da arena.
Regra para os agentes

Ao alterar código, preservar essas regras. Se alguma mudança violar este arquivo, avisar claramente antes de sugerir.