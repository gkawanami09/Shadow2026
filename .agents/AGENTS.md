AGENTS.md — Shadow Robot
Projeto

Este repositório controla o robô Shadow para OBR/Robótica de Resgate. O projeto
possui um programa de percurso e outro programa separado para o resgate.

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
Detecção, aproximação e coleta das vítimas na área de resgate.

Não implementar desvio completo de obstáculo sem pedido explícito e sem sensor/hardware adequado.

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
