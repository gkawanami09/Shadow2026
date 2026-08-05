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


def build_entry_gate():
    """Cria o portão da faixa prata apenas no modo de missão completa."""
    if not mission_mode.value or not config.ENTRY_SILVER_ENABLED:
        return None
    # A entrada usa a MESMA assinatura da confirmação de saída: textura.
    # O gate HSV anterior exigia fita clara e neutra, mas medido nas fotos
    # reais da fita ela aparece com V 50..140 e S 36..70 — enquanto o piso
    # branco dá V 199..228 e S 16..24. O piso passava e a fita não.
    # Ver o docstring de visao/entrada_prata.py.
    from visao.entrada_prata import PortaoEntradaPrata
    print(
        "[visão] faixa prata de entrada armada — assinatura de TEXTURA "
        "(mesma da confirmação de saída)")
    return PortaoEntradaPrata()


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
