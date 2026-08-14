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
    """Gira, encosta, abre o lado escolhido, sacode e restaura a cacamba."""

    INICIO = "SILVER_DEPOSIT_START"
    CONCLUIDO = "SILVER_DEPOSIT_COMPLETE"
    FALHA = "SILVER_DEPOSIT_FAULT"

    def __init__(self, marcador_destino="green"):
        if marcador_destino not in ("green", "red"):
            raise ValueError("marcador_destino deve ser green ou red")
        self.marcador_destino = marcador_destino
        if marcador_destino == "red":
            self.INICIO = "BLACK_DEPOSIT_START"
            self.CONCLUIDO = "BLACK_DEPOSIT_COMPLETE"
            self.FALHA = "BLACK_DEPOSIT_FAULT"
        self._etapas = self._montar_etapas()
        self._indice = 0
        self._ativa = False
        self._prazo = None
        self._detalhe_falha = ""
        # Estes dois reconhecimentos formam a permissao para a proxima fase.
        # Em especial, a busca da saida jamais pode ser armada apenas porque
        # os temporizadores terminaram: no deposito vermelho a caçamba precisa
        # ter recebido tanto a abertura quanto a restauracao.
        self._cacamba_aberta_comandada = False
        self._cacamba_restaurada_comandada = False

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

    @property
    def deposito_fisico_comandado(self):
        """Indica que abrir e restaurar a caçamba foram aceitos pela serial."""
        return (
            self._cacamba_aberta_comandada
            and self._cacamba_restaurada_comandada
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
            destino = (
                "verde" if self.marcador_destino == "green" else "vermelho")
            return PassoDepositoCinza(
                self.CONCLUIDO,
                f"deposito {destino} concluido e cacamba restaurada "
                "para 90 graus",
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
        if etapa.nome.startswith("BUCKET_OPEN_"):
            self._cacamba_aberta_comandada = True
        elif etapa.nome == "BUCKET_RESTORE":
            self._cacamba_restaurada_comandada = True
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

    def _estado_pendente(self, etapa):
        prefixo = (
            "SILVER_DEPOSIT"
            if self.marcador_destino == "green"
            else "BLACK_DEPOSIT"
        )
        return f"{prefixo}_{etapa.nome}_PENDING"

    def _estado_ativo(self, etapa):
        prefixo = (
            "SILVER_DEPOSIT"
            if self.marcador_destino == "green"
            else "BLACK_DEPOSIT"
        )
        return f"{prefixo}_{etapa.nome}"

    def _montar_etapas(self):
        deposito_verde = self.marcador_destino == "green"
        delta_abertura = (
            cfg.SILVER_DEPOSIT_BUCKET_OPEN_DELTA
            if deposito_verde
            else cfg.BLACK_DEPOSIT_BUCKET_OPEN_DELTA
        )
        delta_restauracao = (
            cfg.SILVER_DEPOSIT_BUCKET_RESTORE_DELTA
            if deposito_verde
            else cfg.BLACK_DEPOSIT_BUCKET_RESTORE_DELTA
        )
        destino = "verde" if deposito_verde else "vermelho"
        posicao_aberta = 0 if deposito_verde else 180
        etapas = [
            _Etapa(
                "PRE_TURN_FORWARD",
                "avancando reto por 1 segundo antes do giro",
                0,
                cfg.SILVER_DEPOSIT_PRE_TURN_SPEED,
                cfg.SILVER_DEPOSIT_PRE_TURN_FORWARD_S,
            ),
            _Etapa(
                "PRE_TURN_FORWARD_STOP",
                "avanco inicial concluido; parando antes da re",
                190,
                0.0,
                0.10,
            ),
            _Etapa(
                "PRE_TURN_REVERSE",
                "dando re por 0,5 segundo antes do giro",
                200,
                cfg.SILVER_DEPOSIT_PRE_TURN_SPEED,
                cfg.SILVER_DEPOSIT_PRE_TURN_REVERSE_S,
            ),
            _Etapa(
                "PRE_TURN_REVERSE_STOP",
                "re inicial concluida; parando antes do giro de 180 graus",
                190,
                0.0,
                0.10,
            ),
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
                f"BUCKET_OPEN_{self.marcador_destino.upper()}",
                f"abrindo cacamba para o deposito {destino}: "
                f"90 para {posicao_aberta} graus",
                190,
                0.0,
                cfg.SILVER_DEPOSIT_BUCKET_SETTLE_S,
                delta_abertura,
            ),
            _Etapa(
                "BUCKET_OPEN_WAIT",
                "cacamba aberta; aguardando a vitima assentar antes da sacudida",
                190,
                0.0,
                cfg.SILVER_DEPOSIT_BUCKET_OPEN_EXTRA_WAIT_S,
            ),
        ]
        self._indice_abertura = len(etapas) - 2

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
            f"restaurando cacamba de {posicao_aberta} para 90 graus",
            190,
            0.0,
            cfg.SILVER_DEPOSIT_BUCKET_RESTORE_S,
            delta_restauracao,
        ))
        self._indice_restauracao = len(etapas) - 1
        etapas.extend((
            _Etapa(
                "EXIT_FORWARD",
                "cacamba fechada; avancando reto por 1,5 segundo",
                0,
                cfg.SILVER_DEPOSIT_EXIT_FORWARD_SPEED,
                cfg.SILVER_DEPOSIT_EXIT_FORWARD_S,
            ),
            _Etapa(
                "EXIT_STOP",
                "avanco final concluido; parando o robo",
                190,
                0.0,
                0.10,
            ),
        ))
        return tuple(etapas)
