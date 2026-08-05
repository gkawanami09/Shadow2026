"""Faixa PRATA de entrada, decidida pela mesma assinatura da saída.

Por que este módulo substituiu o gate HSV
-----------------------------------------
O detector anterior exigia que a fita fosse **clara e neutra** (V ≥ 140,
S ≤ 70). Medido nas duas fotos reais da fita (``captures/linha_prata``, as
close-ups 140307 e 140318):

===========  =========  =========
região       V          S
===========  =========  =========
fita         50 a 140   36 a 70
piso branco  199 a 228  16 a 24
===========  =========  =========

Ou seja: **o piso passava no gate e a fita não**. Dependendo do ângulo, a
mesma fita refletiva aparece clara e neutra ou escura e amarelada — é a
natureza de uma superfície especular. Brilho não é uma assinatura estável.

O que É estável é a TEXTURA. A fita tem trama ranhurada e produz variação
local alta (amplitude 25 a 101 nas fotos reais); piso e linha preta são
lisos (7 a 13). Essa é exatamente a mesma medida que
``ClassificadorFaixaSaidaLinha`` já usa para separar preta de prata na
saída — e que acerta 4/4 nas quatro referências reais da arena.

Reusar o classificador dá três vantagens: uma só física para calibrar, uma
só coisa para testar, e a garantia de que entrada e saída nunca discordam
sobre o que é prata e o que é preto.

Limitação declarada
-------------------
Não existe imagem da câmera de linha durante o segue-linha normal neste
repositório — só as duas close-ups da fita. Portanto a taxa de FALSO
POSITIVO durante o percurso **não foi medida**. O risco conhecido é piso
muito brilhante ou reflexo forte gerar textura alta. Antes de confiar,
colete negativos com::

    python3 shadow/tools/coletar_dataset.py --linha --sessao percurso_normal
"""

from collections import deque

import config
import config_resgate as cfg
from visao.confirmacao_saida_linha import (
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
)


class PortaoEntradaPrata:
    """Votação temporal sobre a assinatura de prata da câmera de linha.

    Compatível com o que ``entrada_missao`` espera de um portão: ``update``
    devolvendo ``(confirmado, deteccao)``, mais ``votes`` e ``detector``.
    """

    def __init__(self, classificador=None, votos_necessarios=None,
                 janela=None):
        self.detector = (
            ClassificadorFaixaSaidaLinha()
            if classificador is None else classificador)
        self.votos_necessarios = int(
            config.ENTRY_SILVER_VOTES_NEEDED
            if votos_necessarios is None else votos_necessarios)
        self._votos = deque(
            maxlen=int(
                config.ENTRY_SILVER_VOTE_WINDOW if janela is None
                else janela))
        self._confirmado = False
        self._ultimo_timestamp = None
        self._bloqueado_ate = None
        self.ultimo_resultado = None

    @property
    def confirmed(self):
        return self._confirmado

    @property
    def votes(self):
        return sum(1 for voto in self._votos if voto)

    def reset(self, now=None):
        """Desarma e arma o cooldown — usado ao voltar do resgate."""
        self._votos.clear()
        self._confirmado = False
        self._ultimo_timestamp = None
        self.ultimo_resultado = None
        if now is not None and config.ENTRY_SILVER_COOLDOWN_S > 0:
            self._bloqueado_ate = (
                float(now) + config.ENTRY_SILVER_COOLDOWN_S)

    def update(self, frame_bgr, line_ahead=None, timestamp=None, now=None,
               hsv_image=None):
        timestamp = 0.0 if timestamp is None else float(timestamp)
        now = timestamp if now is None else float(now)

        resultado = self.detector.classificar(
            frame_bgr, timestamp=timestamp)
        self.ultimo_resultado = resultado

        if self._confirmado:
            # Histerese: quem desarma é a missão, via reset().
            return True, resultado
        if (
            self._bloqueado_ate is not None
            and now < self._bloqueado_ate
        ):
            return False, None
        if (
            self._ultimo_timestamp is not None
            and timestamp <= self._ultimo_timestamp + 1e-9
        ):
            # Frame repetido não gera um segundo voto.
            return False, resultado
        if now - timestamp > config.ENTRY_SILVER_MAX_AGE_S:
            # Resultado velho não descreve o presente.
            return False, resultado

        self._ultimo_timestamp = timestamp

        # A linha preta continuando à frente significa pista, não soleira.
        linha_continua = bool(
            config.ENTRY_SILVER_REQUIRE_LINE_END and line_ahead)
        # PRETA aqui é a própria linha do percurso: nunca é a entrada.
        e_prata = bool(
            resultado.classificacao == NAO_PRETA
            and resultado.faixa_presente
            and not linha_continua
        )
        self._votos.append(e_prata)
        if self.votes >= self.votos_necessarios:
            self._confirmado = True
        return self._confirmado, (resultado if e_prata else None)


def resumo(resultado):
    """Texto curto do estado, para o log do processo de visão."""
    if resultado is None:
        return "sem faixa"
    return (
        f"{resultado.classificacao} textura={resultado.textura:.1f} "
        f"borda={resultado.preenchimento_borda:.2f}")
