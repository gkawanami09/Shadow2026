"""Conta passagens separadas pelo marcador verde durante a busca."""

import config_resgate as cfg


BUSCA_REINICIAR = "reiniciar"
BUSCA_CONCLUIR = "concluir"
BUSCA_FALHAR = "falhar"


def decidir_apos_varredura(contador, varreduras_sem_vitima):
    """Decide o fim de uma volta completa sem vítima."""
    if contador.completo:
        return BUSCA_CONCLUIR
    if int(varreduras_sem_vitima) < cfg.RESCUE_SEARCH_MAX_EMPTY_SWEEPS:
        return BUSCA_REINICIAR
    return BUSCA_FALHAR


class ContadorVerdeBusca:
    """Evita que varios frames do mesmo verde sejam contados varias vezes."""

    def __init__(self, necessario=None, frames_para_rearmar=None):
        self.necessario = int(
            cfg.RESCUE_GREEN_SIGHTINGS_REQUIRED
            if necessario is None else necessario
        )
        self.frames_para_rearmar = int(
            cfg.RESCUE_GREEN_REARM_FRAMES
            if frames_para_rearmar is None else frames_para_rearmar
        )
        if self.necessario < 1:
            raise ValueError("a quantidade de verdes deve ser positiva")
        if self.frames_para_rearmar < 1:
            raise ValueError("o rearme exige pelo menos um frame")
        self.reset()

    @property
    def quantidade(self):
        return self._quantidade

    @property
    def completo(self):
        return self._quantidade >= self.necessario

    def reset(self):
        self._quantidade = 0
        self._verde_armado = False
        self._frames_ausente = 0
        self._ultimo_timestamp = None
        self._ultima_varredura_contada = None

    def observar(self, deteccao, permitido=True, varredura=None):
        """Recebe um frame parado e devolve ``True`` se somou uma passagem.

        Frames capturados durante o giro nao contam e tambem nao rearmam o
        contador. Uma deteccao ainda em confirmacao mantem a passagem atual,
        mas somente uma deteccao confirmada consegue somar. Quando o numero da
        varredura e informado, o mesmo giro completo pode somar no maximo uma
        passagem, mesmo que o detector oscile ou o verde saia e volte ao quadro.
        """
        if not permitido:
            return False

        timestamp = getattr(deteccao, "timestamp", None)
        if timestamp is not None:
            timestamp = float(timestamp)
            if (
                self._ultimo_timestamp is not None
                and timestamp <= self._ultimo_timestamp + 1e-9
            ):
                return False
            self._ultimo_timestamp = timestamp

        if deteccao is None:
            self._frames_ausente += 1
            if self._frames_ausente >= self.frames_para_rearmar:
                self._verde_armado = False
            return False

        self._frames_ausente = 0
        if not bool(getattr(deteccao, "confirmed", False)):
            return False
        if (
            varredura is not None
            and self._ultima_varredura_contada == int(varredura)
        ):
            return False
        if self._verde_armado:
            return False

        self._verde_armado = True
        self._quantidade += 1
        if varredura is not None:
            self._ultima_varredura_contada = int(varredura)
        return True
