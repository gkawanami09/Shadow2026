"""Ponte entre a faixa prata de entrada e o processo de visão do percurso.

Vive num módulo próprio, e não dentro de ``visao/processamento.py``, por dois
motivos:

* o pipeline do segue-linha depende de ``numba``, que só existe no Raspberry;
  isolando a ponte aqui, ela pode ser testada em qualquer máquina;
* a detecção da entrada é uma responsabilidade da missão, não do segue-linha.
  Mantê-la separada deixa claro que ``main.py`` sozinho não a executa.

Regra de ouro deste módulo: sem ``mission_mode`` ligado, nada é construído e
nada é calculado. Rodar ``shadow/main.py`` isolado custa zero.
"""

import config
from shared.dados_compartilhados import (config_manager, entry_armed,
                                         entry_silver_confirmed,
                                         entry_silver_detected,
                                         entry_silver_reason,
                                         entry_silver_votes, mission_mode)


def _deve_pre_avancar_entrada(
    *,
    modo_missao,
    armada,
    confirmada,
    detectada,
    votos,
    motivo,
    linha_adiante,
):
    """Reconhece a faixa ainda distante sem transformá-la numa curva.

    ``fina`` só é publicado quando a máscara prata já formou várias linhas
    horizontais largas, mas ainda não possui espessura suficiente para ser
    aceita. É justamente a perspectiva mostrada quando a soleira está longe.
    O estado ``fina`` tem prioridade sobre ``linha_adiante``: quando a entrada
    ainda está longe, o próprio trecho preto que termina nela ocupa boa parte
    do corredor central e mantém essa flag verdadeira. Para os demais casos,
    a linha continuar à frente ainda veta o avanço.
    """
    if not modo_missao or not armada or confirmada:
        return False
    if str(motivo) == "fina":
        return True
    if linha_adiante:
        return False
    return bool(detectada or int(votos) > 0)


def build_entry_gate():
    """Cria o portão da faixa prata apenas no modo de missão completa."""
    if not mission_mode.value or not config.ENTRY_SILVER_ENABLED:
        return None
    from visao.faixa_entrada import (EntrySilverDetector, EntrySilverGate,
                                     load_bounds)
    # A faixa prata pertence ao perfil da CÂMERA DE LINHA; os limites da
    # vítima prateada vivem em config_resgate e nunca são lidos aqui.
    hsv_min, hsv_max = load_bounds(config_manager)
    print(
        f"[visão] faixa prata de entrada armada — HSV {hsv_min}..{hsv_max} "
        "(perfil da câmera de linha)")
    return EntrySilverGate(
        detector=EntrySilverDetector(hsv_min=hsv_min, hsv_max=hsv_max))


def update_entry_silver(entry_gate, frame, captured_at, line_ahead=False,
                        hsv_image=None):
    """Publica o estado da faixa prata para o processo de controle."""
    if entry_gate is None:
        return
    if not entry_armed.value:
        # Já entramos na sala nesta volta: não reprocessar nem reconfirmar.
        entry_silver_detected.value = False
        return
    confirmed, detection = entry_gate.update(
        frame,
        line_ahead=bool(line_ahead),
        timestamp=captured_at,
        now=captured_at,
        hsv_image=hsv_image,
    )
    entry_silver_detected.value = detection is not None
    entry_silver_votes.value = int(entry_gate.votes)
    entry_silver_reason.value = str(entry_gate.detector.last_reason)
    if confirmed and not entry_silver_confirmed.value:
        print(
            "[visão] faixa PRATA de entrada confirmada "
            f"({entry_gate.votes}/{config.ENTRY_SILVER_VOTE_WINDOW} votos)")
    entry_silver_confirmed.value = bool(confirmed)
