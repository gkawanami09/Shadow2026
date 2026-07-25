"""Controla a busca temporizada por uma esfera usando giro tanque."""

import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


class BallSearchController:
    """Gira, para ao travar um alvo e confirma a esfera ja parado."""

    START = "SEARCH_START"
    ROTATING = "SEARCH"
    TARGET_STOP = "SEARCH_TARGET_STOP"
    VERIFY = "SEARCH_VERIFY"
    TURN_STOP = "SEARCH_TURN_STOP"
    FINAL_VERIFY = "SEARCH_FINAL_VERIFY"
    ACQUIRED = "SEARCH_ACQUIRED"
    COMPLETE = "SEARCH_COMPLETE"

    def __init__(self, start_time=None):
        # O relogio do 360 nao comeca aqui. Ele so e armado depois que o
        # comando de giro foi realmente escrito na serial.
        self.state = self.START
        self._created_at = (
            time.monotonic()
            if start_time is None
            else float(start_time)
        )
        self._rotation_started_at = None
        self._rotation_elapsed_s = 0.0
        self._target_stopped_at = None
        self._target_kind = None
        self._tentative_target = False
        self._tracking_reset_requested = False

    @property
    def target_acquired(self):
        return self.state == self.ACQUIRED

    @property
    def terminal(self):
        return self.state == self.COMPLETE

    @property
    def target_kind(self):
        return self._target_kind

    def update(self, detection, now=None):
        now = time.monotonic() if now is None else float(now)

        if self.state == self.COMPLETE:
            return self._stop_command(
                self.COMPLETE,
                "giro de 360 graus concluido sem encontrar outra esfera",
                terminal=True,
            )
        if self.state == self.ACQUIRED:
            return self._stop_command(
                self.ACQUIRED,
                "alvo unico reconfirmado com o robo parado",
                target_kind=self._target_kind,
            )

        if self.state == self.START:
            if self._valid_target(detection, now):
                return self._request_target_stop(detection)
            if self._plausible_target(detection, now):
                return self._request_target_stop(
                    detection, tentative=True)
            return self._tank_command(
                self.START,
                "iniciando giro tanque para procurar outra esfera",
            )

        if self.state == self.ROTATING:
            # Uma deteccao valida vence o prazo inclusive no instante exato
            # do fim do giro; assim uma esfera no fechamento do 360 nao e
            # descartada por ordem de avaliacao.
            if self._valid_target(detection, now):
                return self._request_target_stop(detection)
            if self._plausible_target(detection, now):
                return self._request_target_stop(
                    detection, tentative=True)
            if (
                self._rotation_started_at is not None
                and self._rotation_elapsed(now)
                >= cfg.BALL_SEARCH_FULL_TURN_S - 1e-9
            ):
                self.state = self.TURN_STOP
                return self._stop_command(
                    self.TURN_STOP,
                    "360 concluido; parando para a verificacao final",
                )
            return self._tank_command(
                self.ROTATING,
                "girando em modo tanque; procurando uma esfera",
            )

        if self.state == self.TARGET_STOP:
            return self._stop_command(
                self.TARGET_STOP,
                "alvo travado; aguardando confirmacao de PARAR",
                target_kind=self._target_kind,
            )

        if self.state == self.TURN_STOP:
            return self._stop_command(
                self.TURN_STOP,
                "aguardando confirmacao de PARAR apos o 360",
            )

        if self.state == self.VERIFY:
            if self._valid_target(
                detection,
                now,
                expected_kind=(
                    None
                    if self._tentative_target
                    else self._target_kind
                ),
                captured_after=self._target_stopped_at,
            ):
                self._target_kind = detection.kind
                self._tentative_target = False
                self.state = self.ACQUIRED
                return self._stop_command(
                    self.ACQUIRED,
                    "alvo unico reconfirmado com o robo parado",
                    target_kind=self._target_kind,
                )
            if (
                self._target_stopped_at is not None
                and now - self._target_stopped_at
                >= cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
            ):
                self.state = (
                    self.TURN_STOP
                    if self._rotation_elapsed_s
                    >= cfg.BALL_SEARCH_FULL_TURN_S - 1e-9
                    else self.START
                )
                self._target_stopped_at = None
                self._target_kind = None
                self._tentative_target = False
                self._tracking_reset_requested = True
                if self.state == self.TURN_STOP:
                    return self._stop_command(
                        self.TURN_STOP,
                        "falso alvo descartado no fim do 360",
                    )
                return self._tank_command(
                    self.START,
                    "falso alvo descartado; retomando o restante do 360",
                )
            return self._stop_command(
                self.VERIFY,
                "robo parado; reconfirmando o mesmo alvo",
                target_kind=self._target_kind,
            )

        if self.state == self.FINAL_VERIFY:
            if self._valid_target(
                detection,
                now,
                captured_after=self._target_stopped_at,
            ):
                self._target_kind = detection.kind
                self.state = self.ACQUIRED
                return self._stop_command(
                    self.ACQUIRED,
                    "alvo encontrado na verificacao final do 360",
                    target_kind=self._target_kind,
                )
            if (
                self._target_stopped_at is not None
                and now - self._target_stopped_at
                >= cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
            ):
                self.state = self.COMPLETE
                return self._stop_command(
                    self.COMPLETE,
                    "360 e verificacao final concluidos sem outra esfera",
                    terminal=True,
                )
            return self._stop_command(
                self.FINAL_VERIFY,
                "robo parado; verificando os ultimos frames do 360",
            )

        raise RuntimeError(f"estado de busca desconhecido: {self.state}")

    def mark_rotation_started(self, now=None):
        """Inicia o prazo do 360 depois da escrita serial bem-sucedida."""
        if self.state != self.START:
            raise RuntimeError(
                "confirmacao do giro fora do estado de partida")
        now = time.monotonic() if now is None else float(now)
        self.state = self.ROTATING
        self._rotation_started_at = now

    def mark_target_stopped(self, now=None):
        """Passa a exigir frames capturados depois de o robo parar."""
        if self.state != self.TARGET_STOP:
            raise RuntimeError(
                "confirmacao de PARAR fora do estado de alvo")
        now = time.monotonic() if now is None else float(now)
        self._finish_rotation_segment(now)
        self.state = self.VERIFY
        self._target_stopped_at = now

    def mark_full_turn_stopped(self, now=None):
        """Confirma a parada do 360 antes de declarar a busca vazia."""
        if self.state != self.TURN_STOP:
            raise RuntimeError(
                "confirmacao de PARAR fora do fim do 360")
        now = time.monotonic() if now is None else float(now)
        self._finish_rotation_segment(now)
        self.state = self.FINAL_VERIFY
        self._target_stopped_at = now

    def notify_command_written(self, state, now=None):
        """Interface unificada com a busca pulsada.

        Devolve ``True`` quando o robo acabou de parar e a memoria visual
        anterior precisa ser descartada. Existe para o chamador tratar os dois
        controladores de busca da mesma forma, sem uma cadeia de ``if`` por
        nome de estado.
        """
        if state == self.START:
            self.mark_rotation_started(now)
            return False
        if state == self.TARGET_STOP:
            self.mark_target_stopped(now)
            return True
        if state == self.TURN_STOP:
            self.mark_full_turn_stopped(now)
            return True
        return False

    def frame_allowed(self, captured_at):
        """Rejeita durante a verificacao imagens feitas antes da parada."""
        if (
            self.state not in (self.VERIFY, self.FINAL_VERIFY)
            or self._target_stopped_at is None
        ):
            return True
        return float(captured_at) > self._target_stopped_at + 1e-9

    def consume_tracking_reset(self):
        requested = self._tracking_reset_requested
        self._tracking_reset_requested = False
        return requested

    def _rotation_elapsed(self, now):
        elapsed = self._rotation_elapsed_s
        if self._rotation_started_at is not None:
            elapsed += max(float(now) - self._rotation_started_at, 0.0)
        return elapsed

    def _finish_rotation_segment(self, now):
        if self._rotation_started_at is None:
            return
        self._rotation_elapsed_s = self._rotation_elapsed(now)
        self._rotation_started_at = None

    def _request_target_stop(self, detection, tentative=False):
        self.state = self.TARGET_STOP
        self._target_kind = detection.kind
        self._tentative_target = bool(tentative)
        return self._stop_command(
            self.TARGET_STOP,
            (
                "candidato visual encontrado; freando para confirmar"
                if tentative
                else "alvo unico travado; parando antes de confirmar"
            ),
            target_kind=self._target_kind,
        )

    @staticmethod
    def _plausible_target(detection, now):
        """Freia cedo; a confirmacao forte continua obrigatoria ja parado."""
        if (
            detection is None
            or detection.kind not in ("silver", "black")
            or float(detection.confidence)
            < cfg.BALL_SEARCH_BRAKE_MIN_CONFIDENCE
        ):
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    @staticmethod
    def _valid_target(
        detection,
        now,
        expected_kind=None,
        captured_after=None,
    ):
        if (
            detection is None
            or detection.kind not in ("silver", "black")
            or not detection.confirmed
            or not bool(getattr(detection, "track_locked", False))
        ):
            return False
        if expected_kind is not None and detection.kind != expected_kind:
            return False
        if (
            captured_after is not None
            and float(detection.timestamp) <= float(captured_after) + 1e-9
        ):
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    @staticmethod
    def _tank_command(state, detail):
        return MotionCommand(
            state,
            angle=cfg.BALL_SEARCH_TANK_ANGLE,
            speed=cfg.BALL_SEARCH_TANK_SPEED,
            detail=detail,
        )

    @staticmethod
    def _stop_command(state, detail, terminal=False, target_kind=None):
        return MotionCommand(
            state,
            angle=190,
            speed=0.0,
            detail=detail,
            terminal=terminal,
            target_kind=target_kind,
        )
