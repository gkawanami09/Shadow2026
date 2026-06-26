# Não quebrar

## Coisas que não devem ser quebradas

### Hardware
- Não mudar pinout sem confirmação.
- Não inverter motores sem testar.
- Não mudar protocolo Raspberry Pi ↔ Arduino sem explicar.
- Não assumir hardware diferente do robô Shadow.

### Segue-linha
- Não remover o comportamento atual de seguir linha.
- Não deixar o verde dominar a lógica principal.
- Não fazer o robô parar por qualquer perda pequena de linha.
- Não substituir o controle por uma lógica mais instável.

### Verde
- Não aceitar qualquer pixel verde como marcação.
- Não ignorar posição e área.
- Não tomar decisão antes de confirmar se o verde é real.

### Gap
- Não girar imediatamente quando a linha sumir.
- Não tratar todo sumiço como beco.
- Não tratar todo sumiço como gap.
- Sempre ter limite de tempo/tentativas.

### Código
- Não fazer refatoração gigante sem necessidade.
- Não apagar comentários úteis.
- Não esconder parâmetros mágicos no meio do código.
- Não deixar valores importantes sem nome.
- Não remover logs/debugs úteis durante fase de teste.

## Regra final
Se uma mudança pode quebrar o robô na pista, explique antes e faça de forma pequena, reversível e testável.