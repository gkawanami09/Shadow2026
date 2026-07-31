"""Sequencia final para liberar as vitimas prata pela saida configurada."""

from dataclasses import dataclass
import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


@dataclass(frozen=True)
class PassoDepositoCinza:
    """Comando de um tick; o delta da cacamba aparece uma unica vez."""

    state: str
    detail: str
    angle: int = 190
    speed: float = 0.0
    bucket_delta: object = None
    terminal: bool = False

    def motion_command(self):
        return MotionCommand(
            self.state,
            angle=self.angle,
            speed=self.speed,
            detail=self.detail,
            terminal=self.terminal,
        )


@dataclass(frozen=True)
class _Etapa:
    nome: str
    detalhe: str
    angulo: int
    velocidade: float
    duracao: float
    delta_cacamba: object = None


class SequenciadorDepositoCinza:
    """Gira, encosta, abre a saida da prata, sacode e restaura a cacamba."""

    INICIO = "SILVER_DEPOSIT_START"
    CONCLUIDO = "SILVER_DEPOSIT_COMPLETE"
    FALHA = "SILVER_DEPOSIT_FAULT"

    def __init__(self):
        self._etapas = self._montar_etapas()
        self._indice = 0
        self._ativa = False
        self._prazo = None
        self._detalhe_falha = ""

    @property
    def terminal(self):
        return bool(self._detalhe_falha) or self._indice >= len(
            self._etapas)

    @property
    def cacamba_aberta(self):
        return (
            self._indice > self._indice_abertura
            and self._indice <= self._indice_restauracao
        )

    def update(self, now=None):
        agora = time.monotonic() if now is None else float(now)
        if self._detalhe_falha:
            return PassoDepositoCinza(
                self.FALHA,
                self._detalhe_falha,
                terminal=True,
            )
        if self._indice >= len(self._etapas):
            return PassoDepositoCinza(
                self.CONCLUIDO,
                "vitimas prata liberadas e cacamba restaurada para 90 graus",
                terminal=True,
            )

        etapa = self._etapas[self._indice]
        if not self._ativa:
            return PassoDepositoCinza(
                self._estado_pendente(etapa),
                etapa.detalhe,
                angle=etapa.angulo,
                speed=etapa.velocidade,
                bucket_delta=etapa.delta_cacamba,
            )
        if agora < self._prazo:
            return PassoDepositoCinza(
                self._estado_ativo(etapa),
                etapa.detalhe,
                angle=etapa.angulo,
                speed=etapa.velocidade,
            )

        self._indice += 1
        self._ativa = False
        self._prazo = None
        return self.update(now=agora)

    def notify_command_written(self, state, now=None):
        """Inicia o prazo apenas depois de motor e servo serem aceitos."""
        if self.terminal or self._ativa:
            return False
        etapa = self._etapas[self._indice]
        if state != self._estado_pendente(etapa):
            return False
        agora = time.monotonic() if now is None else float(now)
        self._ativa = True
        self._prazo = agora + etapa.duracao
        return True

    def fail(self, detail):
        self._detalhe_falha = str(detail)
        return PassoDepositoCinza(
            self.FALHA,
            self._detalhe_falha,
            terminal=True,
        )

    @staticmethod
    def _estado_pendente(etapa):
        return f"SILVER_DEPOSIT_{etapa.nome}_PENDING"

    @staticmethod
    def _estado_ativo(etapa):
        return f"SILVER_DEPOSIT_{etapa.nome}"

    def _montar_etapas(self):
        etapas = [
            _Etapa(
                "TURN_180",
                "girando 180 graus em movimento de tanque",
                180,
                cfg.SILVER_DEPOSIT_TURN_SPEED,
                cfg.SILVER_DEPOSIT_TURN_S,
            ),
            _Etapa(
                "TURN_STOP",
                "giro concluido; estabilizando antes da re",
                190,
                0.0,
                0.10,
            ),
            _Etapa(
                "REVERSE_ALIGN",
                "dando re por 3 segundos para alinhar no deposito",
                200,
                cfg.SILVER_DEPOSIT_REVERSE_SPEED,
                cfg.SILVER_DEPOSIT_REVERSE_S,
            ),
            _Etapa(
                "REVERSE_STOP",
                "alinhamento concluido; parando antes de abrir a cacamba",
                190,
                0.0,
                0.12,
            ),
            _Etapa(
                "BUCKET_OPEN_RIGHT",
                "abrindo cacamba da prata: 90 para 0 graus",
                190,
                0.0,
                cfg.SILVER_DEPOSIT_BUCKET_SETTLE_S,
                cfg.SILVER_DEPOSIT_BUCKET_OPEN_DELTA,
            ),
        ]
        self._indice_abertura = len(etapas) - 1

        for repeticao in range(cfg.SILVER_DEPOSIT_SHAKE_REPETITIONS):
            numero = repeticao + 1
            etapas.extend((
                _Etapa(
                    f"SHAKE_FORWARD_{numero}",
                    f"sacudida {numero}: avanco rapido",
                    0,
                    cfg.SILVER_DEPOSIT_SHAKE_SPEED,
                    cfg.SILVER_DEPOSIT_SHAKE_MOVE_S,
                ),
                _Etapa(
                    f"SHAKE_FRONT_STOP_{numero}",
                    "pausa curta antes de inverter os motores",
                    190,
                    0.0,
                    cfg.SILVER_DEPOSIT_SHAKE_STOP_S,
                ),
                _Etapa(
                    f"SHAKE_REVERSE_{numero}",
                    f"sacudida {numero}: re rapida",
                    200,
                    cfg.SILVER_DEPOSIT_SHAKE_SPEED,
                    cfg.SILVER_DEPOSIT_SHAKE_MOVE_S,
                ),
                _Etapa(
                    f"SHAKE_REAR_STOP_{numero}",
                    "pausa curta depois da sacudida",
                    190,
                    0.0,
                    cfg.SILVER_DEPOSIT_SHAKE_STOP_S,
                ),
            ))

        etapas.append(_Etapa(
            "BUCKET_RESTORE",
            "restaurando cacamba de 0 para 90 graus",
            190,
            0.0,
            cfg.SILVER_DEPOSIT_BUCKET_RESTORE_S,
            cfg.SILVER_DEPOSIT_BUCKET_RESTORE_DELTA,
        ))
        self._indice_restauracao = len(etapas) - 1
        return tuple(etapas)
