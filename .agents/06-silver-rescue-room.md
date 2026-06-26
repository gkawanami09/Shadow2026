Prata e entrada da sala de resgate
Status atual

O Shadow ainda não tem hardware completo para resgatar vítimas. Portanto, por enquanto, este arquivo serve para detectar a entrada da sala de resgate e mudar o estado do robô corretamente.

Regra OBR
A linha escura termina na entrada da sala de resgate.
A entrada é demarcada por fita prata ou prata reflexiva.
A fita prata tem aproximadamente 25 mm x 250 mm.
A saída da sala é demarcada por fita preta de aproximadamente 25 mm x 250 mm.
A sala pode medir aproximadamente 90 cm x 90 cm ou 120 cm x 90 cm.
A sala possui paredes claras de no mínimo 10 cm de altura.
A porta, se existir, pode ter 25 cm de largura e altura.
Objetivo atual

Detectar que o robô está entrando na sala de resgate.

Detecção de prata

A prata é difícil porque depende de:

iluminação;
reflexo;
brilho;
exposição automática da câmera;
ângulo da câmera;
material usado na arena.
Estratégia recomendada

Não depender apenas de uma cor fixa.

Usar combinação de:

brilho alto;
baixa saturação;
região horizontal larga;
posição próxima ao fim da linha;
desaparecimento/terminação da linha preta;
largura aproximada da faixa;
persistência por alguns frames.
Estado esperado

Quando a prata for confirmada:

Sair de LINE_FOLLOW.
Entrar em RESCUE_ENTRY_DETECTED.
Parar ou avançar lentamente conforme estratégia definida.
Não executar resgate completo ainda sem hardware.
Saída da sala

No futuro, a saída será detectada pela faixa preta de saída.
A lógica de saída deve ser separada da lógica normal de segue-linha, porque dentro da sala a linha preta do percurso não existe do mesmo jeito.

Cuidados
Não confundir reflexo branco com prata.
Não confundir linha clara/brilho da pista com entrada da sala.
Não acionar sala de resgate por uma mancha pequena.
Não implementar busca de vítimas sem hardware.
Não confundir área de resgate verde/vermelha com marcadores de percurso.