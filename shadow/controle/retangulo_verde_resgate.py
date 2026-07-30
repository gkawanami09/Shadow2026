"""Leva o robo ao retangulo verde no encerramento do resgate."""

import time

import cv2
import numpy as np

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand
from controle.deposito_resgate import DepositMarkerController


class ConfirmadorTelaVerde:
    """Confirma que praticamente todo o campo util da camera ficou verde."""

    def __init__(self):
        self.quantidade = 0
        self.proporcao = 0.0
        self._primeiro_timestamp = None
        self._ultimo_timestamp = None

    @property
    def confirmado(self):
        return (
            self.quantidade
            >= cfg.RESCUE_GREEN_FULL_FRAME_CONFIRM_FRAMES
        )

    def reset(self):
        self.quantidade = 0
        self.proporcao = 0.0
        self._primeiro_timestamp = None
        self._ultimo_timestamp = None

    def observar(self, mascara_verde, timestamp):
        """Conta somente frames novos e consecutivos com cobertura suficiente."""
        proporcao, _erro = medir_verde(mascara_verde)
        self.proporcao = proporcao
        timestamp = float(timestamp)

        if (
            self._ultimo_timestamp is not None
            and timestamp <= self._ultimo_timestamp + 1e-9
        ):
            return self.confirmado
        self._ultimo_timestamp = timestamp

        if proporcao < cfg.RESCUE_GREEN_FULL_FRAME_MIN_RATIO:
            self.quantidade = 0
            self._primeiro_timestamp = None
            return False

        if (
            self._primeiro_timestamp is None
            or timestamp - self._primeiro_timestamp
            > cfg.RESCUE_GREEN_FULL_FRAME_CONFIRM_WINDOW_S
        ):
            self.quantidade = 1
            self._primeiro_timestamp = timestamp
        else:
            self.quantidade += 1
        return self.confirmado


def medir_verde(mascara_verde):
    """Retorna ``(cobertura, erro_horizontal)`` da mascara verde."""
    if (
        mascara_verde is None
        or not isinstance(mascara_verde, np.ndarray)
        or mascara_verde.ndim != 2
        or mascara_verde.size == 0
    ):
        return 0.0, 0.0

    altura, largura = mascara_verde.shape
    margem = cfg.RESCUE_GREEN_FULL_FRAME_MARGIN_RATIO
    margem_x = min(int(round(largura * margem)), max(largura // 3, 0))
    margem_y = min(int(round(altura * margem)), max(altura // 3, 0))
    x1, x2 = margem_x, largura - margem_x
    y1, y2 = margem_y, altura - margem_y
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0

    regiao = np.asarray(mascara_verde[y1:y2, x1:x2], dtype=np.uint8)
    pixels = regiao > 0
    quantidade = int(np.count_nonzero(pixels))
    proporcao = quantidade / float(max(pixels.size, 1))
    if quantidade <= 0:
        return proporcao, 0.0

    momentos = cv2.moments(pixels.astype(np.uint8), binaryImage=True)
    centro_x = (
        float(momentos["m10"]) / float(momentos["m00"])
        if momentos["m00"] > 0.0
        else regiao.shape[1] / 2.0
    )
    erro = float(np.clip(
        (centro_x - regiao.shape[1] / 2.0)
        / max(regiao.shape[1] / 2.0, 1.0),
        -1.0,
        1.0,
    ))
    return proporcao, erro


class ControladorRetanguloVerde:
    """Procura o verde, aproxima e avanca ate o quadro ficar verde."""

    APROXIMACAO_FINAL = "GREEN_FINAL_APPROACH"
    CONFIRMANDO_TELA = "GREEN_FULL_VERIFY"
    CONCLUIDO = "GREEN_FULL_FRAME"
    FALHA = "GREEN_FINAL_FAULT"

    def __init__(self, start_time=None):
        self.navegacao = DepositMarkerController(
            "green", start_time=start_time)
        self.confirmador = ConfirmadorTelaVerde()
        self.aproximacao_final = False
        self._aproximacao_iniciada_em = None
        self._ultimo_frame_em = None
        self._ultimo_verde_em = None
        self._detalhe_falha = ""

    @property
    def terminal(self):
        return self.navegacao.state == self.navegacao.FAULT or bool(
            self._detalhe_falha) or self.confirmador.confirmado

    def update(
        self,
        deteccao_verde,
        formato_frame,
        mascara_verde=None,
        timestamp_frame=None,
        now=None,
    ):
        agora = time.monotonic() if now is None else float(now)

        if self.confirmador.confirmado:
            return self._parar(
                self.CONCLUIDO,
                "retangulo verde ocupa toda a camera; resgate encerrado",
                terminal=True,
            )
        if self._detalhe_falha:
            return self._parar(
                self.FALHA, self._detalhe_falha, terminal=True)

        if not self.aproximacao_final:
            return self.navegacao.update(
                deteccao_verde, formato_frame, now=agora)

        if (
            self._aproximacao_iniciada_em is not None
            and agora - self._aproximacao_iniciada_em
            >= cfg.RESCUE_GREEN_FINAL_MAX_ACTIVE_S
        ):
            return self._falhar(
                "tempo maximo procurando preencher a camera de verde")

        frame_novo = (
            mascara_verde is not None
            and timestamp_frame is not None
            and (
                self._ultimo_frame_em is None
                or float(timestamp_frame) > self._ultimo_frame_em + 1e-9
            )
        )
        if frame_novo:
            timestamp_frame = float(timestamp_frame)
            self._ultimo_frame_em = timestamp_frame
            if self._aproximacao_iniciada_em is None:
                self._aproximacao_iniciada_em = agora

            proporcao, erro = medir_verde(mascara_verde)
            if proporcao >= cfg.RESCUE_GREEN_FINAL_MIN_VISIBLE_RATIO:
                self._ultimo_verde_em = agora
            confirmado = self.confirmador.observar(
                mascara_verde, timestamp_frame)
            if confirmado:
                return self._parar(
                    self.CONCLUIDO,
                    "retangulo verde ocupa toda a camera; "
                    f"cobertura={self.confirmador.proporcao:.0%}",
                    terminal=True,
                )
            if proporcao >= cfg.RESCUE_GREEN_FULL_FRAME_MIN_RATIO:
                return self._parar(
                    self.CONFIRMANDO_TELA,
                    "confirmando camera verde "
                    f"{self.confirmador.quantidade}/"
                    f"{cfg.RESCUE_GREEN_FULL_FRAME_CONFIRM_FRAMES}; "
                    f"cobertura={proporcao:.0%}",
                )
            if proporcao < cfg.RESCUE_GREEN_FINAL_MIN_VISIBLE_RATIO:
                return self._sem_verde(agora, proporcao)
            return self._avancar(erro, proporcao)

        if self._aproximacao_iniciada_em is None:
            return self._parar(
                self.APROXIMACAO_FINAL,
                "chegada confirmada; aguardando frame novo para avancar",
            )
        if (
            self._ultimo_frame_em is None
            or agora - self._ultimo_frame_em > cfg.BALL_FRAME_STALE_S
        ):
            return self._falhar(
                "camera sem frame novo durante a aproximacao final")
        return MotionCommand(
            self.APROXIMACAO_FINAL,
            angle=0,
            speed=cfg.RESCUE_GREEN_FINAL_FORWARD_SPEED,
            detail=(
                "avancando sobre o retangulo verde; "
                f"cobertura={self.confirmador.proporcao:.0%}"
            ),
        )

    def notify_command_written(self, state, now=None):
        """Confirma ao navegador que o comando chegou ao Arduino."""
        agora = time.monotonic() if now is None else float(now)
        if self.aproximacao_final:
            return False
        if (
            state == self.navegacao.START
            and self.navegacao.state == self.navegacao.START
        ):
            self.navegacao.mark_rotation_started(now=agora)
            return True
        if (
            state == self.navegacao.TARGET_STOP
            and self.navegacao.state == self.navegacao.TARGET_STOP
        ):
            self.navegacao.mark_target_stopped(now=agora)
            return True
        if (
            state == self.navegacao.TURN_STOP
            and self.navegacao.state == self.navegacao.TURN_STOP
        ):
            self.navegacao.mark_full_turn_stopped(now=agora)
            return True
        if (
            state == self.navegacao.LOST_STOP
            and self.navegacao.state == self.navegacao.LOST_STOP
        ):
            self.navegacao.mark_lost_stopped(now=agora)
            return True
        if (
            state == self.navegacao.ARRIVAL_STOP
            and self.navegacao.state == self.navegacao.ARRIVAL_STOP
        ):
            self.navegacao.mark_arrival_stopped(now=agora)
            self.aproximacao_final = True
            self.confirmador.reset()
            return True
        return False

    def frame_allowed(self, captured_at):
        if self.aproximacao_final:
            return True
        return self.navegacao.frame_allowed(captured_at)

    def consume_tracking_reset(self):
        if self.aproximacao_final:
            return False
        return self.navegacao.consume_tracking_reset()

    def _sem_verde(self, agora, proporcao):
        if self._ultimo_verde_em is None:
            self._ultimo_verde_em = agora
        if (
            agora - self._ultimo_verde_em
            >= cfg.RESCUE_GREEN_FINAL_LOST_TIMEOUT_S
        ):
            return self._falhar(
                "verde sumiu durante a aproximacao final")
        return self._parar(
            self.APROXIMACAO_FINAL,
            f"verde insuficiente ({proporcao:.0%}); robo parado",
        )

    def _avancar(self, erro, proporcao):
        if abs(erro) <= cfg.RESCUE_GREEN_FINAL_CENTER_DEADBAND:
            angulo = 0
        else:
            angulo = int(round(np.clip(
                erro * cfg.RESCUE_GREEN_FINAL_STEER_MAX_ANGLE,
                -cfg.RESCUE_GREEN_FINAL_STEER_MAX_ANGLE,
                cfg.RESCUE_GREEN_FINAL_STEER_MAX_ANGLE,
            )))
        return MotionCommand(
            self.APROXIMACAO_FINAL,
            angle=angulo,
            speed=cfg.RESCUE_GREEN_FINAL_FORWARD_SPEED,
            detail=(
                "avancando sobre o retangulo verde; "
                f"cobertura={proporcao:.0%}, erro={erro:+.2f}"
            ),
        )

    def _falhar(self, detalhe):
        self._detalhe_falha = str(detalhe)
        return self._parar(
            self.FALHA, self._detalhe_falha, terminal=True)

    @staticmethod
    def _parar(state, detail, terminal=False):
        return MotionCommand(
            state,
            angle=190,
            speed=0.0,
            detail=detail,
            terminal=terminal,
        )
