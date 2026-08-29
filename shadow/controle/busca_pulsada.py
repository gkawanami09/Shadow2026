"""Busca pulsada em tanque: gira um trecho curto, para, e só então observa.

Por que trocar o giro contínuo
------------------------------
Girando sem parar, dois problemas aparecem juntos:

1. a esfera atravessa o campo de visão antes de acumular os três resultados
   distintos que o detector exige para travar o alvo;
2. os frames capturados em movimento saem borrados e com o autoexposure ainda
   corrigindo, o que é exatamente a imagem que não pode confirmar nada.

O ciclo implementado aqui é::

    PULSE_ROTATE → BRAKE → SETTLE → OBSERVE → PULSE_ROTATE …

Regra central, e a razão de o controlador existir: **um frame só pode
confirmar se foi capturado depois do fim do SETTLE**. Isso está em
``frame_allowed()`` e é a mesma proteção que o controlador contínuo já tinha
para a verificação do alvo, agora aplicada a todos os pulsos.

Compatibilidade
---------------
A classe expõe exatamente a mesma interface pública de
``BallSearchController`` (``update``, ``frame_allowed``,
``consume_tracking_reset``, ``target_acquired``, ``terminal``,
``target_kind``) mais ``notify_command_written``, que substitui a cadeia de
``mark_*`` do chamador. Assim ela é substituível sem tocar no resto do
resgate, e o controlador contínuo continua existindo, intacto, com seus
testes.

Sem IMU, o ângulo percorrido continua sendo estimado por TEMPO ATIVO de giro:
as pausas não contam. ``BALL_SEARCH_FULL_TURN_S`` é a calibração de 360°.
"""

import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


class PulsedBallSearchController:
    """Gira em pulsos, observa parado e trava um único alvo por vez."""

    # Estados visíveis ao chamador. START/TARGET_STOP/TURN_STOP mantêm os
    # mesmos nomes do controlador contínuo para o log e o overlay não mudarem
    # de vocabulário no meio da competição.
    START = "SEARCH_START"
    INITIAL_OBSERVE = "SEARCH_INITIAL_OBSERVE"
    ROTATING = "SEARCH_PULSE"
    BRAKE = "SEARCH_BRAKE"
    SETTLE = "SEARCH_SETTLE"
    OBSERVE = "SEARCH_OBSERVE"
    TARGET_STOP = "SEARCH_TARGET_STOP"
    VERIFY = "SEARCH_VERIFY"
    TURN_STOP = "SEARCH_TURN_STOP"
    FINAL_VERIFY = "SEARCH_FINAL_VERIFY"
    ACQUIRED = "SEARCH_ACQUIRED"
    COMPLETE = "SEARCH_COMPLETE"

    #: Estados em que o robô está parado por construção.
    STOPPED_STATES = (
        INITIAL_OBSERVE, BRAKE, SETTLE, OBSERVE, TARGET_STOP, VERIFY, TURN_STOP,
        FINAL_VERIFY, ACQUIRED, COMPLETE,
    )

    def __init__(self, start_time=None, accepts_kind=None, initial_observe_s=0.0):
        now = time.monotonic() if start_time is None else float(start_time)
        self.state = (
            self.INITIAL_OBSERVE
            if float(initial_observe_s) > 0.0 else self.START)
        self._created_at = now
        self._initial_observe_until = now + max(float(initial_observe_s), 0.0)
        #: Filtro opcional de cor para diagnósticos e futuras estratégias.
        self._accepts_kind = accepts_kind
        self._rotation_started_at = None
        self._rotation_elapsed_s = 0.0
        self._pulse_started_at = None
        self._stopped_at = None
        self._settled_at = None
        self._observed_frames = 0
        self._last_observed_timestamp = None
        self._target_stopped_at = None
        self._target_kind = None
        self._tentative_target = False
        self._tracking_reset_requested = False
        # A entrada da sala já termina com PARAR. O primeiro pulso começa
        # logo depois desse ponto; a confirmação continua protegida nas pausas
        # entre pulsos, sempre com frame novo após o assentamento.
        self.pulses = 0

    # -- estado ----------------------------------------------------------
    @property
    def target_acquired(self):
        return self.state == self.ACQUIRED

    @property
    def terminal(self):
        return self.state == self.COMPLETE

    @property
    def target_kind(self):
        return self._target_kind

    @property
    def stopped(self):
        return self.state in self.STOPPED_STATES

    @property
    def rotation_elapsed_s(self):
        """Tempo ATIVO de giro. Pausas nunca entram nesta conta."""
        return self._rotation_elapsed(time.monotonic())

    def consume_tracking_reset(self):
        requested = self._tracking_reset_requested
        self._tracking_reset_requested = False
        return requested

    def frame_allowed(self, captured_at):
        """Só aceita imagens capturadas depois de o chassi assentar.

        Enquanto gira, freia ou assenta, nenhum frame vale — é o borrão do
        movimento. Após o SETTLE, exige timestamp estritamente posterior ao
        instante em que o assentamento terminou.
        """
        if self.state in (self.START, self.ROTATING, self.BRAKE, self.SETTLE):
            return False
        reference = None
        if self.state in (self.VERIFY, self.FINAL_VERIFY):
            reference = self._target_stopped_at
        elif self.state == self.OBSERVE:
            reference = self._settled_at
        if reference is None:
            return True
        return float(captured_at) > reference + 1e-9

    # -- ciclo principal -------------------------------------------------
    def update(self, detection, now=None):
        now = time.monotonic() if now is None else float(now)

        if self.state == self.COMPLETE:
            return self._stop(
                self.COMPLETE,
                "cobertura de 360 graus concluida sem encontrar esfera",
                terminal=True)
        if self.state == self.ACQUIRED:
            return self._stop(
                self.ACQUIRED,
                "alvo unico reconfirmado com o robo parado",
                target_kind=self._target_kind)

        if now - self._created_at >= cfg.BALL_SEARCH_TOTAL_TIMEOUT_S:
            # Teto global: escorregamento de roda pode impedir o 360
            # temporizado de fechar. A busca não pode virar laço infinito.
            self.state = self.COMPLETE
            return self._stop(
                self.COMPLETE,
                "tempo total de busca esgotado; encerrando a varredura",
                terminal=True)

        handler = {
            self.START: self._on_start,
            self.INITIAL_OBSERVE: self._on_initial_observe,
            self.ROTATING: self._on_rotating,
            self.BRAKE: self._on_brake,
            self.SETTLE: self._on_settle,
            self.OBSERVE: self._on_observe,
            self.TARGET_STOP: self._on_target_stop,
            self.VERIFY: self._on_verify,
            self.TURN_STOP: self._on_turn_stop,
            self.FINAL_VERIFY: self._on_final_verify,
        }.get(self.state)
        if handler is None:
            raise RuntimeError(f"estado de busca desconhecido: {self.state}")
        return handler(detection, now)

    def _on_start(self, detection, now):
        return self._tank(
            self.START, "iniciando pulso de busca em modo tanque")

    def _on_initial_observe(self, detection, now):
        """Readquire a bola recém-vista antes de começar uma varredura."""
        if self._valid(detection, now):
            return self._request_target_stop(detection)
        if self._plausible(detection, now):
            return self._request_target_stop(detection, tentative=True)
        if now >= self._initial_observe_until:
            self.state = self.START
            return self._tank(
                self.START, "alvo inicial nao reapareceu; iniciando busca")
        return self._stop(
            self.INITIAL_OBSERVE,
            "parado; readquirindo a bola vista antes do handoff")

    def _on_rotating(self, detection, now):
        # Um candidato visto durante o giro FREIA imediatamente, mas não
        # confirma: a confirmação exige frames capturados depois da parada.
        if self._plausible(detection, now):
            return self._request_target_stop(detection, tentative=True)
        if self._pulse_finished(now):
            self.state = self.BRAKE
            return self._stop(
                self.BRAKE, "fim do pulso; freando para observar")
        return self._tank(
            self.ROTATING, "girando um trecho curto em modo tanque")

    def _on_brake(self, detection, now):
        return self._stop(
            self.BRAKE, "aguardando confirmacao de PARAR do pulso")

    def _on_settle(self, detection, now):
        if (
            self._stopped_at is not None
            and now - self._stopped_at
            >= cfg.BALL_SEARCH_SETTLE_S - 1e-9
        ):
            self.state = self.OBSERVE
            self._settled_at = now
            self._observed_frames = 0
            self._last_observed_timestamp = None
            return self._stop(
                self.OBSERVE, "chassi assentado; observando frames novos")
        return self._stop(
            self.SETTLE, "aguardando vibracao e autoexposure assentarem")

    def _on_observe(self, detection, now):
        if self._valid(detection, now, captured_after=self._settled_at):
            return self._request_target_stop(detection)
        if self._plausible(detection, now, captured_after=self._settled_at):
            return self._request_target_stop(detection, tentative=True)

        if (
            detection is not None
            and self.frame_allowed(detection.timestamp)
            and (
                self._last_observed_timestamp is None
                or detection.timestamp > self._last_observed_timestamp + 1e-9
            )
        ):
            self._observed_frames += 1
            self._last_observed_timestamp = float(detection.timestamp)

        observed_enough = (
            self._observed_frames >= cfg.BALL_SEARCH_OBSERVE_FRAMES)
        timed_out = (
            self._settled_at is not None
            and now - self._settled_at
            >= cfg.BALL_SEARCH_OBSERVE_TIMEOUT_S - 1e-9
        )
        if observed_enough or timed_out:
            if self._turn_finished():
                self.state = self.FINAL_VERIFY
                self._target_stopped_at = self._settled_at
                return self._stop(
                    self.FINAL_VERIFY,
                    "cobertura completa; verificando os ultimos frames")
            self.state = self.START
            return self._tank(
                self.START,
                "nada no setor; proximo pulso da varredura")
        return self._stop(
            self.OBSERVE,
            f"observando parado ({self._observed_frames}"
            f"/{cfg.BALL_SEARCH_OBSERVE_FRAMES} frames novos)")

    def _on_target_stop(self, detection, now):
        return self._stop(
            self.TARGET_STOP,
            "alvo travado; aguardando confirmacao de PARAR",
            target_kind=self._target_kind)

    def _on_verify(self, detection, now):
        expected = None if self._tentative_target else self._target_kind
        if self._valid(
            detection, now,
            expected_kind=expected,
            captured_after=self._target_stopped_at,
        ):
            self._target_kind = detection.kind
            self._tentative_target = False
            self.state = self.ACQUIRED
            return self._stop(
                self.ACQUIRED,
                "alvo unico reconfirmado com o robo parado",
                target_kind=self._target_kind)
        if (
            self._target_stopped_at is not None
            and now - self._target_stopped_at
            >= cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
        ):
            # Falso alvo: descarta e RETOMA a varredura de onde parou. O tempo
            # de giro já acumulado é preservado, então a cobertura restante
            # continua correta.
            self._target_stopped_at = None
            self._target_kind = None
            self._tentative_target = False
            self._tracking_reset_requested = True
            if self._turn_finished():
                self.state = self.TURN_STOP
                return self._stop(
                    self.TURN_STOP, "falso alvo descartado no fim da varredura")
            self.state = self.START
            return self._tank(
                self.START,
                "falso alvo descartado; retomando a cobertura restante")
        return self._stop(
            self.VERIFY,
            "robo parado; reconfirmando o mesmo alvo",
            target_kind=self._target_kind)

    def _on_turn_stop(self, detection, now):
        return self._stop(
            self.TURN_STOP, "aguardando confirmacao de PARAR final")

    def _on_final_verify(self, detection, now):
        if self._valid(
            detection, now, captured_after=self._target_stopped_at
        ):
            self._target_kind = detection.kind
            self.state = self.ACQUIRED
            return self._stop(
                self.ACQUIRED,
                "alvo encontrado na verificacao final",
                target_kind=self._target_kind)
        if (
            self._target_stopped_at is not None
            and now - self._target_stopped_at
            >= cfg.BALL_SEARCH_VERIFY_TIMEOUT_S
        ):
            self.state = self.COMPLETE
            return self._stop(
                self.COMPLETE,
                "varredura e verificacao final concluidas sem outra esfera",
                terminal=True)
        return self._stop(
            self.FINAL_VERIFY,
            "robo parado; verificando os ultimos frames da varredura")

    # -- confirmações de escrita serial ----------------------------------
    def notify_command_written(self, state, now=None):
        """Confirma que o comando daquele estado foi escrito na serial.

        Devolve ``True`` quando o robô acabou de parar e a memória visual
        anterior precisa ser descartada pelo chamador. Concentrar isso em um
        método só evita a cadeia de ``if`` que o chamador mantinha para cada
        transição.
        """
        now = time.monotonic() if now is None else float(now)
        if state == self.START:
            self._begin_pulse(now)
            return False
        if state == self.BRAKE:
            self._finish_rotation_segment(now)
            self._stopped_at = now
            self.state = self.SETTLE
            return True
        if state == self.TARGET_STOP:
            self._finish_rotation_segment(now)
            self.state = self.VERIFY
            self._target_stopped_at = now
            return True
        if state == self.TURN_STOP:
            self._finish_rotation_segment(now)
            self.state = self.FINAL_VERIFY
            self._target_stopped_at = now
            return True
        return False

    # Nomes do controlador contínuo, mantidos para quem já os chamava.
    def mark_rotation_started(self, now=None):
        self.notify_command_written(self.START, now)

    def mark_target_stopped(self, now=None):
        self.notify_command_written(self.TARGET_STOP, now)

    def mark_full_turn_stopped(self, now=None):
        self.notify_command_written(self.TURN_STOP, now)

    # -- internos --------------------------------------------------------
    def _begin_pulse(self, now):
        self.state = self.ROTATING
        self._rotation_started_at = now
        self._pulse_started_at = now
        self._settled_at = None
        self._observed_frames = 0
        self._last_observed_timestamp = None
        self.pulses += 1

    def _pulse_finished(self, now):
        # A margem de 1e-9 é a mesma convenção usada no giro de 360: sem ela,
        # um instante calculado como `inicio + PULSE_S` pode ficar alguns
        # ULPs abaixo do limite e o pulso nunca terminar naquele tick.
        return (
            self._pulse_started_at is not None
            and now - self._pulse_started_at
            >= cfg.BALL_SEARCH_PULSE_S - 1e-9
        )

    def _turn_finished(self):
        return (
            self._rotation_elapsed_s
            >= cfg.BALL_SEARCH_FULL_TURN_S - 1e-9
        )

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
        self._pulse_started_at = None

    def _request_target_stop(self, detection, tentative=False):
        self.state = self.TARGET_STOP
        self._target_kind = detection.kind
        self._tentative_target = bool(tentative)
        return self._stop(
            self.TARGET_STOP,
            (
                "candidato visual encontrado; freando para confirmar"
                if tentative
                else "alvo unico travado; parando antes de confirmar"
            ),
            target_kind=self._target_kind)

    def _kind_allowed(self, kind):
        if kind not in ("silver", "black"):
            return False
        if self._accepts_kind is None:
            return True
        return bool(self._accepts_kind(kind))

    def _plausible(self, detection, now, captured_after=None):
        """Freia cedo; a confirmação forte continua obrigatória já parado."""
        if (
            detection is None
            or not self._kind_allowed(detection.kind)
            or float(detection.confidence)
            < cfg.BALL_SEARCH_BRAKE_MIN_CONFIDENCE
        ):
            return False
        if (
            captured_after is not None
            and float(detection.timestamp) <= float(captured_after) + 1e-9
        ):
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    def _valid(self, detection, now, expected_kind=None, captured_after=None):
        if (
            detection is None
            or not self._kind_allowed(detection.kind)
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
    def _tank(state, detail):
        return MotionCommand(
            state,
            angle=cfg.BALL_SEARCH_TANK_ANGLE,
            speed=cfg.BALL_SEARCH_TANK_SPEED,
            detail=detail)

    @staticmethod
    def _stop(state, detail, terminal=False, target_kind=None):
        return MotionCommand(
            state,
            angle=190,
            speed=0.0,
            detail=detail,
            terminal=terminal,
            target_kind=target_kind)


def make_search_controller(
    start_time=None,
    accepts_kind=None,
    initial_observe_s=0.0,
):
    """Fábrica usada pelo resgate: pulsada por padrão, contínua se desligada."""
    if cfg.BALL_SEARCH_PULSED:
        return PulsedBallSearchController(
            start_time=start_time,
            accepts_kind=accepts_kind,
            initial_observe_s=initial_observe_s,
        )
    from controle.busca_resgate import BallSearchController
    return BallSearchController(start_time=start_time)
