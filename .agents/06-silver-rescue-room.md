Prata e entrada da sala de resgate
Status atual

O Shadow possui uma câmera frontal, garras e elevador para o resgate. O
segue-linha continua em `shadow/main.py`, enquanto a busca e a coleta ficam no
programa separado `shadow/resgate.py`.

Regra OBR
A linha escura termina na entrada da sala de resgate.
A entrada é demarcada por fita prata ou prata reflexiva.
A fita prata tem aproximadamente 25 mm x 250 mm.
A saída da sala é demarcada por fita preta de aproximadamente 25 mm x 250 mm.
A sala pode medir aproximadamente 90 cm x 90 cm ou 120 cm x 90 cm.
A sala possui paredes claras de no mínimo 10 cm de altura.
A porta, se existir, pode ter 25 cm de largura e altura.
Objetivo atual

Manter a chegada à sala separada da procura e coleta das vítimas. Dentro da
sala, o resgate usa somente a câmera frontal e não usa a lógica de verde do
percurso.

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
Encerrar o segue-linha antes de iniciar o programa de resgate.
Saída da sala

No futuro, a saída será detectada pela faixa preta de saída.
A lógica de saída deve ser separada da lógica normal de segue-linha, porque dentro da sala a linha preta do percurso não existe do mesmo jeito.

Cuidados
Não confundir reflexo branco com prata.
Não confundir linha clara/brilho da pista com entrada da sala.
Não acionar sala de resgate por uma mancha pequena.
Não misturar o detector de vítimas com o detector de verde do percurso.
Não confundir área de resgate verde/vermelha com marcadores de percurso.
