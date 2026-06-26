AGENTS.md — Shadow Robot
Projeto

Este repositório controla o robô Shadow para OBR/Robótica de Resgate. O objetivo atual é completar com alta confiabilidade a parte de percurso: seguir linha preta, detectar verde, lidar com gap, parar no vermelho e identificar a entrada da sala de resgate pela fita prata.

Fonte de verdade

Antes de alterar lógica de comportamento, leia:

.agents/10-obr-rulebook.md

Esse arquivo resume regras importantes da OBR para programação. Ele não substitui o regulamento oficial.

Foco atual

O foco atual é:

Segue-linha estável.
Detecção de verde.
Decisão correta em interseções e becos.
Recuperação em gap.
Parada correta no vermelho.
Detecção de prata para saber que entrou/está entrando na sala de resgate.
Ainda não implementar

Não implementar resgate completo de vítimas sem pedido explícito e sem hardware adequado.

Não implementar desvio completo de obstáculo sem pedido explícito e sem sensor/hardware adequado.

Não transformar a sala de resgate em lógica principal agora. A prioridade atual é chegar até a entrada da sala com confiabilidade.

Contexto obrigatório

Antes de mexer no código, leia:

.agents/00-current-focus.md
.agents/01-hardware-invariants.md
.agents/09-do-not-break.md
.agents/10-obr-rulebook.md
Contexto por tarefa

Se a tarefa envolver segue-linha, leia:

.agents/02-line-following.md

Se envolver verde, interseção ou beco, leia:

.agents/03-green-detection.md
.agents/07-state-machine.md

Se envolver gap ou perda de linha, leia:

.agents/04-gap-recovery.md

Se envolver vermelho ou chegada, leia:

.agents/05-red-stop.md

Se envolver prata, entrada da sala ou saída futura da sala, leia:

.agents/06-silver-rescue-room.md

Se envolver lógica geral, estados ou decisões, leia:

.agents/07-state-machine.md
.agents/08-test-checklist.md
Regras de alteração
Não alterar pinout sem confirmação.
Não alterar protocolo Raspberry Pi ↔ Arduino sem explicar.
Não usar pré-mapeamento da arena.
Não criar comportamento que dependa de posições fixas da pista.
Não refatorar o projeto inteiro sem necessidade.
Fazer mudanças pequenas, testáveis e reversíveis.
Preservar o que já funciona.
Preferir parâmetros ajustáveis no topo do código.
Sempre manter fallback de segurança quando a linha sumir.
Sempre explicar quais arquivos, parâmetros e comportamentos foram alterados.