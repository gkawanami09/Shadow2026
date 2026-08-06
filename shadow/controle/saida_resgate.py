"""Fase de saída: mapear os dois triângulos, achar a soleira preta e sair.

Esta fase só começa depois de as três vítimas estarem depositadas. A partir
daqui o detector de vítimas está desligado — quem diz isso é o coordenador da
missão, via ``victim_detector_enabled``. A faixa preta, que durante toda a
busca era invisível para o robô, passa a ser a única coisa que importa.

Sequência::

    MAP_TRIANGLES → SEARCH(pulsado) → CROSS → DONE

A procura reaproveita o mesmo princípio da busca pulsada de vítimas: gira um
trecho curto, para, e só confirma com frames capturados depois da parada. A
faixa confirmada de longe inicia imediatamente o avanço reto, sem ficar presa
num alinhamento de baixa potência. A travessia termina quando a faixa deixa de
ser vista, com o tempo servindo apenas de limite de segurança.
"""

import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


class ExitPhaseController:
    """Conduz o robô do fim do resgate até fora da sala, parado e seguro."""

    MAP_TRIANGLES = "EXIT_MAP_TRIANGLES"
    SEARCH_START = "EXIT_SEARCH_START"
    SEARCH_ROTATE = "EXIT_SEARCH_ROTATE"
    SEARCH_BRAKE = "EXIT_SEARCH_BRAKE"
    SEARCH_SETTLE = "EXIT_SEARCH_SETTLE"
    SEARCH_OBSERVE = "EXIT_SEARCH_OBSERVE"
    ALIGN = "EXIT_ALIGN"
    CROSS = "EXIT_CROSS"
    DONE = "EXIT_DONE"
    FAILED = "EXIT_FAILED"

    def __init__(self, start_time=None):
        now = time.monotonic() if start_time is None else float(start_time)
        self.state = self.MAP_TRIANGLES
        self._created_at = now
        self._map_started_at = now
        self._rotate_started_at = None
        self._stopped_at = None
        self._settled_at = None
        self._cross_started_at = None
        self._cross_finished_at = None
        self._tracking_reset_requested = False
        self.mapped_triangles = {"green": False, "red": False}

    @property
    def terminal(self):
        return self.state in (self.DONE, self.FAILED)

    @property
    def succeeded(self):
        return self.state == self.DONE

    @property
    def cross_elapsed_s(self):
        """Tempo real em que o comando reto da tentativa ficou ativo."""
        if self._cross_started_at is None:
            return 0.0
        end = (
            self._cross_finished_at
            if self._cross_finished_at is not None
            else time.monotonic()
        )
        return max(float(end) - float(self._cross_started_at), 0.0)

    @property
    def stopped(self):
        return self.state in (
            self.MAP_TRIANGLES, self.SEARCH_BRAKE, self.SEARCH_SETTLE,
            self.SEARCH_OBSERVE, self.DONE, self.FAILED)

    def consume_tracking_reset(self):
        requested = self._tracking_reset_requested
        self._tracking_reset_requested = False
        return requested

    def frame_allowed(self, captured_at):
        """Imagens feitas durante o giro não confirmam a soleira."""
        if self.state in (
            self.SEARCH_START, self.SEARCH_ROTATE, self.SEARCH_BRAKE,
            self.SEARCH_SETTLE,
        ):
            return False
        if self.state == self.SEARCH_OBSERVE and self._settled_at is not None:
            return float(captured_at) > self._settled_at + 1e-9
        return True

    def update(self, exit_detection, frame_shape, mapper=None, now=None):
        now = time.monotonic() if now is None else float(now)

        if self.state == self.DONE:
            return self._stop(self.DONE, "fora da sala; robo parado",
                              terminal=True)
        if self.state == self.FAILED:
            return self._stop(
                self.FAILED, "saida nao encontrada no prazo", terminal=True)

        if (
            self.state != self.CROSS
            and now - self._created_at >= cfg.EXIT_SEARCH_TIMEOUT_S
        ):
            self.state = self.FAILED
            return self._stop(
                self.FAILED,
                "tempo de procura da saida esgotado; motores parados",
                terminal=True)

        if self.state == self.MAP_TRIANGLES:
            return self._on_map(mapper, now)
        if self.state == self.SEARCH_START:
            return self._tank(
                self.SEARCH_START, "girando para procurar a soleira preta")
        if self.state == self.SEARCH_ROTATE:
            return self._on_rotate(exit_detection, now)
        if self.state == self.SEARCH_BRAKE:
            return self._stop(
                self.SEARCH_BRAKE, "freando para observar a soleira")
        if self.state == self.SEARCH_SETTLE:
            return self._on_settle(now)
        if self.state == self.SEARCH_OBSERVE:
            return self._on_observe(exit_detection, frame_shape, now)
        if self.state == self.ALIGN:
            return self._on_align(exit_detection, frame_shape, now)
        if self.state == self.CROSS:
            return self._on_cross(exit_detection, now)
        raise RuntimeError(f"estado de saida desconhecido: {self.state}")

    # -- estados ---------------------------------------------------------
    def _on_map(self, mapper, now):
        """Mapeia os dois triângulos; nenhum deles comanda o robô aqui."""
        if mapper is not None:
            self.mapped_triangles = dict(mapper.confirmed)
            enough = (
                mapper.both_found
                or mapper.frames >= cfg.FINAL_TRIANGLE_MAP_FRAMES)
        else:
            enough = True
        timed_out = (
            now - self._map_started_at
            >= cfg.FINAL_TRIANGLE_MAP_TIMEOUT_S - 1e-9)
        if enough or timed_out:
            # Não encontrar os dois triângulos não impede a saída: o
            # mapeamento é diagnóstico, e ficar preso aqui custaria a prova.
            self.state = self.SEARCH_START
            return self._tank(
                self.SEARCH_START,
                "triangulos mapeados; procurando a soleira preta")
        return self._stop(
            self.MAP_TRIANGLES,
            f"mapeando os dois triangulos ({mapper.frames if mapper else 0}"
            f"/{cfg.FINAL_TRIANGLE_MAP_FRAMES} frames)")

    def _on_rotate(self, exit_detection, now):
        if (
            self._stopped_at is None
            and self._rotate_started_at is not None
            and now - self._rotate_started_at
            >= cfg.EXIT_SEARCH_PULSE_S - 1e-9
        ):
            self.state = self.SEARCH_BRAKE
            return self._stop(
                self.SEARCH_BRAKE, "fim do pulso; freando para observar")
        return self._tank(
            self.SEARCH_ROTATE, "girando em pulso para achar a saida")

    def _on_settle(self, now):
        if (
            self._stopped_at is not None
            and now - self._stopped_at
            >= cfg.EXIT_SEARCH_SETTLE_S - 1e-9
        ):
            self.state = self.SEARCH_OBSERVE
            self._settled_at = now
            return self._stop(
                self.SEARCH_OBSERVE, "assentado; procurando a soleira")
        return self._stop(
            self.SEARCH_SETTLE, "aguardando o chassi assentar")

    def _on_observe(self, exit_detection, frame_shape, now):
        if self._usable(exit_detection, now):
            self.state = self.ALIGN
            self._stopped_at = None
            return self._align_command(exit_detection, frame_shape)
        if (
            self._settled_at is not None
            and now - self._settled_at
            >= cfg.EXIT_SEARCH_OBSERVE_TIMEOUT_S - 1e-9
        ):
            self._tracking_reset_requested = True
            self.state = self.SEARCH_START
            return self._tank(
                self.SEARCH_START, "nada aqui; proximo pulso de procura")
        return self._stop(
            self.SEARCH_OBSERVE, "parado, observando a procura da saida")

    def _on_align(self, exit_detection, frame_shape, now):
        if not self._usable(exit_detection, now):
            # Perdeu a soleira: parar e voltar a procurar, nunca seguir cego.
            self._tracking_reset_requested = True
            self.state = self.SEARCH_START
            return self._tank(
                self.SEARCH_START, "soleira perdida; retomando a procura")
        if not self.aligned(exit_detection, frame_shape):
            self._stopped_at = None
            return self._align_command(exit_detection, frame_shape)
        if self._stopped_at is None:
            self._stopped_at = now
            return self._stop(
                self.ALIGN,
                "soleira centralizada; freando antes de avancar",
            )
        if now - self._stopped_at >= cfg.EXIT_ALIGN_SETTLE_S - 1e-9:
            return self.begin_cross(now=now)
        return self._stop(
            self.ALIGN,
            "soleira centralizada; aguardando o chassi assentar",
        )

    def _on_cross(self, exit_detection, now):
        if self._cross_started_at is None:
            return self._forward(
                self.CROSS, "atravessando a soleira de saida")
        elapsed = now - self._cross_started_at
        if elapsed >= cfg.EXIT_ADVANCE_TIMEOUT_S - 1e-9:
            self._cross_finished_at = now
            self.state = self.DONE
            return self._stop(
                self.DONE,
                f"travessia encerrada por timeout ({elapsed:.2f} s)",
                terminal=True)
        if (
            elapsed >= cfg.EXIT_ADVANCE_MIN_S - 1e-9
            and exit_detection is None
        ):
            # Evidência visual: a faixa passou para trás do robô.
            self._cross_finished_at = now
            self.state = self.DONE
            return self._stop(
                self.DONE,
                f"soleira passou para tras ({elapsed:.2f} s); fora da sala",
                terminal=True)
        return self._forward(self.CROSS, "atravessando a soleira de saida")

    # -- confirmações de escrita serial ----------------------------------
    def notify_command_written(self, state, now=None):
        """Confirma a escrita serial e devolve se a visão deve ser zerada."""
        now = time.monotonic() if now is None else float(now)
        if state == self.SEARCH_START:
            self.state = self.SEARCH_ROTATE
            self._rotate_started_at = now
            self._stopped_at = None
            self._settled_at = None
            return False
        if state == self.SEARCH_BRAKE:
            self.state = self.SEARCH_SETTLE
            self._stopped_at = now
            self._rotate_started_at = None
            return True
        if state == self.ALIGN:
            return False
        if state == self.CROSS and self._cross_started_at is None:
            self._cross_started_at = now
            return False
        return False

    def begin_cross(self, now=None):
        """Inicia o avanco reto assim que a soleira distante e confirmada."""
        if self.state not in (self.SEARCH_OBSERVE, self.ALIGN):
            raise RuntimeError(
                "travessia exige soleira confirmada durante a observacao")
        self.state = self.CROSS
        self._cross_started_at = None
        self._cross_finished_at = None
        return self._forward(
            self.CROSS,
            "soleira confirmada de longe; avancando reto imediatamente",
        )

    # -- auxiliares ------------------------------------------------------
    def _align_command(self, detection, frame_shape):
        width = float(frame_shape[1]) if frame_shape is not None else 1.0
        half = max(width / 2.0, 1.0)
        error = (float(detection.center_x) - half) / half
        if abs(error) <= cfg.EXIT_ALIGN_MAX_CENTER_ERROR:
            return self._stop(
                self.ALIGN,
                f"soleira centralizada (erro {error:+.2f}); pronta para cruzar")
        angle = (
            cfg.EXIT_ALIGN_ANGLE if error > 0 else -cfg.EXIT_ALIGN_ANGLE)
        return MotionCommand(
            self.ALIGN,
            angle=angle,
            speed=cfg.EXIT_ALIGN_SPEED,
            detail=f"alinhando com a soleira (erro {error:+.2f})")

    def aligned(self, detection, frame_shape):
        """A soleira está centralizada o bastante para atravessar?"""
        if detection is None or frame_shape is None:
            return False
        half = max(float(frame_shape[1]) / 2.0, 1.0)
        error = (float(detection.center_x) - half) / half
        return abs(error) <= cfg.EXIT_ALIGN_MAX_CENTER_ERROR

    def _usable(self, detection, now):
        if detection is None:
            return False
        if not self.frame_allowed(detection.timestamp):
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    @staticmethod
    def _fresh_preview(detection, now):
        """Aceita a prévia apenas para frear, nunca para confirmar a saída."""
        if detection is None:
            return False
        age = now - float(detection.timestamp)
        return -0.05 <= age <= cfg.BALL_FRAME_STALE_S

    @staticmethod
    def _tank(state, detail):
        return MotionCommand(
            state,
            angle=cfg.EXIT_SEARCH_TANK_ANGLE,
            speed=cfg.EXIT_SEARCH_TANK_SPEED,
            detail=detail)

    @staticmethod
    def _forward(state, detail):
        return MotionCommand(
            state, angle=0, speed=cfg.EXIT_ADVANCE_SPEED, detail=detail)

    @staticmethod
    def _stop(state, detail, terminal=False):
        return MotionCommand(
            state, angle=190, speed=0.0, detail=detail, terminal=terminal)
