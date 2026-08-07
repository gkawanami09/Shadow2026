"""Confirma pela CAMERA DE RESGATE que a sala existe mesmo.

A faixa prata responde uma pergunta: "vi prata na frente?". Ela nao responde
a pergunta que importa: "estou dentro da sala de resgate?". Fita refletiva,
rodape metalico e reflexo forte de piso continuam sendo prata legitima aos
olhos do detector de entrada, por mais apertado que ele fique.

A segunda camera responde a segunda pergunta de graca. Se a sala e de
verdade, ela contem uma vitima (esfera prata ou preta) ou um dos triangulos
de deposito (verde ou vermelho). Se a camera de resgate abre e nao ve nenhum
desses, a entrada foi falsa.

Este modulo e deliberadamente PURO: recebe deteccoes ja prontas e devolve uma
decisao. Nao abre camera, nao toca motor e nao le configuracao de hardware,
entao roda inteiro nos testes sem Raspberry.

A evidencia aceita e sempre uma deteccao CONFIRMADA — a mesma confirmacao
temporal que o resgate ja exige para agir. Um candidato solto de um frame nao
autoriza o resgate: seria trocar um filtro fraco (a faixa) por outro.
"""

import time


#: Marcadores que valem como prova de que a sala existe.
MARCADORES_VALIDOS = ("green", "red")


class ConfirmacaoEntradaResgate:
    """Janela temporal curta que procura vitima ou triangulo na sala.

    Uso::

        confirmacao = ConfirmacaoEntradaResgate(janela_s=4.0, inicio=agora)
        while True:
            if confirmacao.observar(vitima=det, marcadores=marc, agora=agora):
                break                      # sala confirmada
            if confirmacao.expirou(agora):
                break                      # entrada falsa
    """

    def __init__(self, janela_s, inicio=None, relogio=time.monotonic):
        self.janela_s = max(float(janela_s), 0.0)
        self.relogio = relogio
        self.inicio = float(relogio() if inicio is None else inicio)
        self.confirmado = False
        #: O que confirmou a sala: ``"vitima:silver"``, ``"marcador:green"``…
        self.motivo = ""

    @property
    def prazo(self):
        return self.inicio + self.janela_s

    def restante(self, agora=None):
        agora = self.relogio() if agora is None else float(agora)
        return max(self.prazo - agora, 0.0)

    def expirou(self, agora=None):
        """Só expira quando ainda não houve confirmação (ela é definitiva)."""
        if self.confirmado:
            return False
        agora = self.relogio() if agora is None else float(agora)
        return agora >= self.prazo

    def observar(self, vitima=None, marcadores=None, agora=None):
        """Registra um frame e devolve ``True`` quando a sala é confirmada.

        ``vitima`` é a detecção já passada pelo portão de frescor do resgate;
        ``marcadores`` é o dicionário ``{"green": det, "red": det}`` do
        ``MarkerPair``. Qualquer um dos dois confirma — a sala pode ser
        avistada primeiro pelo triângulo, se as vítimas estiverem atrás do
        robô.
        """
        if self.confirmado:
            return True

        if vitima is not None and getattr(vitima, "confirmed", False):
            self.confirmado = True
            tipo = getattr(vitima, "kind", "?")
            self.motivo = f"vitima:{tipo}"
            return True

        for tipo in MARCADORES_VALIDOS:
            deteccao = (marcadores or {}).get(tipo)
            if deteccao is not None and getattr(deteccao, "confirmed", False):
                self.confirmado = True
                self.motivo = f"marcador:{tipo}"
                return True

        return False

    def resumo(self, agora=None):
        if self.confirmado:
            return f"sala confirmada por {self.motivo}"
        return f"procurando vitima/triangulo ({self.restante(agora):.1f} s)"
