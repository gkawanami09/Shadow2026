"""Fase de saída: mapear os dois triângulos, achar a soleira preta e sair.

Esta fase só começa depois de as três vítimas estarem depositadas. A partir
daqui o detector de vítimas está desligado — quem diz isso é o coordenador da
missão, via ``victim_detector_enabled``. A faixa preta, que durante toda a
busca era invisível para o robô, passa a ser a única coisa que importa.

Sequência::

    MAP_TRIANGLES → SEARCH(pulsado) → ALIGN(curva) → CROSS → DONE

A procura reaproveita o mesmo princípio da busca pulsada de vítimas: gira um
trecho curto, para, e só confirma com frames capturados depois da parada. A
faixa confirmada de longe é centralizada com a mesma curva contínua usada na
aproximação da bolinha. A travessia termina quando a faixa deixa de ser vista,
com o tempo servindo apenas de limite de segurança.
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
    ALIGN_YAW = "EXIT_ALIGN_YAW"
    ALIGN_ARC = "EXIT_ALIGN_ARC"
    ALIGN_BRAKE = "EXIT_ALIGN_BRAKE"
    ALIGN_SETTLE = "EXIT_ALIGN_SETTLE"
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
        self._align_last_seen_at = None
        self._align_frame_after = None
        self._align_motion_started_at = None
        self._align_motion_duration_s = 0.0
        self._align_corrections = 0
        self._align_wheel_speeds = None
        self._align_angle = 190
        self._align_previous_errors = None
        self._align_target = None
        self._target_committed = False
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
            self.SEARCH_OBSERVE, self.ALIGN_BRAKE, self.ALIGN_SETTLE,
            self.DONE, self.FAILED)

    def consume_tracking_reset(self):
        requested = self._tracking_reset_requested
        self._tracking_reset_requested = False
        return requested

    def frame_allowed(self, captured_at):
        """Imagens feitas durante o giro não confirmam a soleira."""
        if self.state in (
            self.SEARCH_START, self.SEARCH_ROTATE, self.SEARCH_BRAKE,
            self.SEARCH_SETTLE, self.ALIGN_YAW,
            self.ALIGN_BRAKE, self.ALIGN_SETTLE,
        ):
            return False
        if self.state == self.SEARCH_OBSERVE and self._settled_at is not None:
            return float(captured_at) > self._settled_at + 1e-9
        if self.state == self.ALIGN and self._align_frame_after is not None:
            return float(captured_at) > self._align_frame_after + 1e-9
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
        if self.state == self.ALIGN_YAW:
            return self._on_align_motion(now)
        if self.state == self.ALIGN_ARC:
            return self._on_align(exit_detection, frame_shape, now)
        if self.state == self.ALIGN_BRAKE:
            return self._stop(
                self.ALIGN_BRAKE,
                "fim da correcao; freando antes de medir novamente")
        if self.state == self.ALIGN_SETTLE:
            return self._on_align_settle(now)
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
            self._fresh_preview(exit_detection, now)
            and float(getattr(exit_detection, "confidence", 0.0))
            >= cfg.EXIT_MODEL_FAST_LOCK_CONFIDENCE
        ):
            # Uma resposta forte do modelo ja e suficiente para guardar o
            # centro. Depois da frenagem, esse alvo nao pode ser esquecido so
            # porque a faixa ficou borrada ou saiu parcialmente do quadro.
            self._remember_alignment_target(exit_detection, now)
            self.state = self.SEARCH_BRAKE
            return self._stop(
                self.SEARCH_BRAKE,
                "modelo viu a saida com alta confianca; freando para confirmar",
            )
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
            self._remember_alignment_target(exit_detection, now)
            self._align_frame_after = None
            self._align_corrections = 0
            self._align_previous_errors = None
            return self._align_command(exit_detection, frame_shape)
        if self._target_committed and self._align_target is not None:
            # O modelo ja confirmou com alta confianca durante o giro. Use o
            # centro salvo para apontar o robo, sem voltar aos picos de busca.
            self.state = self.ALIGN
            self._stopped_at = None
            self._align_frame_after = None
            self._align_corrections = 0
            self._align_previous_errors = None
            return self._align_command(self._align_target, frame_shape)
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
            # O modelo pode falhar em um frame durante a curva. Não recomece
            # o giro, pois isso abandona justamente a saída já confirmada.
            perdida_por = now - (
                self._align_last_seen_at
                if self._align_last_seen_at is not None
                else now
            )
            if perdida_por < cfg.EXIT_ALIGN_LOST_TIMEOUT_S - 1e-9:
                if self.state == self.ALIGN_ARC:
                    return self._current_alignment_motion(
                        f"soleira oculta por {perdida_por:.2f}s; "
                        "mantendo a curva do alvo travado",
                    )
                return self._stop(
                    self.ALIGN,
                    f"soleira oculta por {perdida_por:.2f}s; "
                    "parado aguardando reaparecer",
                )
            if self._target_committed and self._align_target is not None:
                # O centro ja foi confirmado. Depois de uma curva curta na
                # ultima direcao conhecida, perder a faixa normalmente quer
                # dizer que ela ficou perto/embaixo do robo: siga reto.
                return self.begin_cross(now=now)
            self._tracking_reset_requested = True
            self.state = self.SEARCH_START
            self._align_last_seen_at = None
            return self._tank(
                self.SEARCH_START, "soleira perdida; retomando a procura")
        self._remember_alignment_target(exit_detection, now)
        erro_centro, erro_angulo = self._alignment_errors(
            exit_detection, frame_shape)
        if abs(erro_angulo) > cfg.EXIT_ALIGN_MAX_ANGLE_DEG:
            self._align_previous_errors = (erro_centro, erro_angulo)
            return self._start_yaw_correction(erro_angulo)
        if abs(erro_centro) > self._center_threshold():
            self._align_previous_errors = (erro_centro, erro_angulo)
            return self._start_arc_correction(erro_centro)
        # A histerese permite começar o avanço com um pequeno erro lateral.
        return self.begin_cross(now=now)

    def _on_align_motion(self, now):
        if self._align_motion_started_at is None:
            return self._current_alignment_motion()
        if (
            now - self._align_motion_started_at
            < self._align_motion_duration_s - 1e-9
        ):
            return self._current_alignment_motion()
        self.state = self.ALIGN_BRAKE
        return self._stop(
            self.ALIGN_BRAKE,
            "pulso de alinhamento concluido; freando")

    def _on_align_settle(self, now):
        if (
            self._stopped_at is not None
            and now - self._stopped_at
            >= cfg.EXIT_ALIGN_SETTLE_S - 1e-9
        ):
            self.state = self.ALIGN
            self._align_frame_after = now
            self._align_last_seen_at = now
            self._align_motion_started_at = None
            self._align_wheel_speeds = None
            self._align_angle = 190
            self._stopped_at = None
            return self._stop(
                self.ALIGN,
                "chassi assentado; medindo centro e inclinacao novamente")
        return self._stop(
            self.ALIGN_SETTLE,
            "aguardando vibracao do alinhamento terminar")

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
        if exit_detection is None and (
            self._target_committed
            or elapsed >= cfg.EXIT_ADVANCE_MIN_S - 1e-9
        ):
            # O alvo ja foi confirmado. De perto, desaparecer significa que a
            # faixa pode ter passado sob a camera; a camera de linha assume o
            # avanco sem exigir outra confirmacao da camera de resgate.
            self._cross_finished_at = now
            self.state = self.DONE
            return self._stop(
                self.DONE,
                f"soleira confirmada saiu da camera ({elapsed:.2f} s); "
                "trocando para a camera de linha",
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
        if state == self.ALIGN_YAW:
            if self._align_motion_started_at is None:
                self._align_motion_started_at = now
            return False
        if state == self.ALIGN_ARC:
            return False
        if state == self.ALIGN_BRAKE:
            self.state = self.ALIGN_SETTLE
            self._stopped_at = now
            self._align_motion_started_at = None
            self._align_frame_after = None
            # Diferente da busca inicial, o alinhamento já possui uma faixa
            # confirmada. Não apague seu gate: mantenha o LOCK entre pulsos.
            return False
        if state == self.CROSS and self._cross_started_at is None:
            self._cross_started_at = now
            return False
        return False

    def begin_cross(self, now=None):
        """Inicia o avanco reto assim que a soleira distante e confirmada."""
        if self.state not in (
            self.SEARCH_OBSERVE, self.ALIGN, self.ALIGN_ARC
        ):
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
        erro_centro, erro_angulo = self._alignment_errors(
            detection, frame_shape)
        if abs(erro_angulo) > cfg.EXIT_ALIGN_MAX_ANGLE_DEG:
            self._align_previous_errors = (erro_centro, erro_angulo)
            return self._start_yaw_correction(erro_angulo)
        if abs(erro_centro) > self._center_threshold():
            self._align_previous_errors = (erro_centro, erro_angulo)
            return self._start_arc_correction(erro_centro)
        return self.begin_cross()

    def aligned(self, detection, frame_shape):
        """A soleira está centralizada o bastante para atravessar?"""
        if detection is None or frame_shape is None:
            return False
        erro_centro, erro_angulo = self._alignment_errors(
            detection, frame_shape)
        return (
            abs(erro_centro) <= cfg.EXIT_ALIGN_EXIT_CENTER_ERROR
            and abs(erro_angulo) <= cfg.EXIT_ALIGN_MAX_ANGLE_DEG
        )

    @staticmethod
    def _alignment_errors(detection, frame_shape):
        half = max(float(frame_shape[1]) / 2.0, 1.0)
        erro_centro = (float(detection.center_x) - half) / half
        erro_angulo = float(getattr(detection, "angle_deg", 0.0))
        return erro_centro, erro_angulo

    def _start_yaw_correction(self, erro_angulo, detalhe=None):
        if not self._correction_allowed():
            return self._restart_search_after_alignment()
        proporcao = min(
            abs(float(erro_angulo))
            / max(cfg.EXIT_ALIGN_YAW_FULL_ERROR_DEG, 1e-6),
            1.0,
        )
        self._align_motion_duration_s = self._interpolate(
            cfg.EXIT_ALIGN_YAW_MIN_PULSE_S,
            cfg.EXIT_ALIGN_YAW_MAX_PULSE_S,
            proporcao,
        )
        self._align_angle = (
            cfg.EXIT_ALIGN_ANGLE
            if erro_angulo > 0.0
            else -cfg.EXIT_ALIGN_ANGLE
        )
        self._align_wheel_speeds = None
        self._align_motion_started_at = None
        self._stopped_at = None
        self.state = self.ALIGN_YAW
        return self._current_alignment_motion(
            detalhe or f"corrigindo inclinacao {erro_angulo:+.1f} graus")

    def _start_arc_correction(self, erro_centro, detalhe=None):
        severidade = min(max(
            (
                abs(float(erro_centro))
                - cfg.EXIT_ALIGN_EXIT_CENTER_ERROR
            ) / max(1.0 - cfg.EXIT_ALIGN_EXIT_CENTER_ERROR, 1e-6),
            0.0,
        ), 1.0)
        magnitude = int(round(
            cfg.EXIT_ALIGN_ARC_MIN_ANGLE
            + severidade
            * (cfg.EXIT_ALIGN_ARC_MAX_ANGLE - cfg.EXIT_ALIGN_ARC_MIN_ANGLE)
        ))
        self._align_angle = magnitude if erro_centro > 0.0 else -magnitude
        self._align_wheel_speeds = None
        self._align_motion_started_at = None
        self._stopped_at = None
        self.state = self.ALIGN_ARC
        return self._current_alignment_motion(
            detalhe or (
                f"curvando como na bolinha; erro={erro_centro:+.2f}"))

    def _remember_alignment_target(self, detection, now):
        self._align_target = detection
        self._align_last_seen_at = now
        self._target_committed = True

    def _correction_allowed(self):
        self._align_corrections += 1
        return self._align_corrections <= cfg.EXIT_ALIGN_MAX_CORRECTIONS

    def _restart_search_after_alignment(self):
        self._tracking_reset_requested = True
        self._align_last_seen_at = None
        self._align_frame_after = None
        self._align_motion_started_at = None
        self._align_wheel_speeds = None
        self._align_previous_errors = None
        self.state = self.SEARCH_START
        return self._tank(
            self.SEARCH_START,
            "alinhamento nao convergiu; retomando a procura da soleira")

    def _current_alignment_motion(self, detail=None):
        if self.state == self.ALIGN_ARC:
            severidade = min(max(
                (
                    abs(float(self._align_angle))
                    - cfg.EXIT_ALIGN_ARC_MIN_ANGLE
                ) / max(
                    cfg.EXIT_ALIGN_ARC_MAX_ANGLE
                    - cfg.EXIT_ALIGN_ARC_MIN_ANGLE,
                    1e-6,
                ),
                0.0,
            ), 1.0)
            velocidade = (
                cfg.EXIT_ALIGN_ARC_SPEED_MIN
                + severidade
                * (
                    cfg.EXIT_ALIGN_ARC_SPEED_MAX
                    - cfg.EXIT_ALIGN_ARC_SPEED_MIN
                )
            )
            return MotionCommand(
                self.ALIGN_ARC,
                angle=self._align_angle,
                speed=float(velocidade),
                detail=(
                    detail
                    or "curvando continuamente para centralizar a saida"),
            )
        return MotionCommand(
            self.ALIGN_YAW,
            angle=self._align_angle,
            speed=cfg.EXIT_ALIGN_SPEED,
            detail=detail or "executando pulso tanque de alinhamento angular",
        )

    def _center_threshold(self):
        if self.state == self.ALIGN_ARC:
            return cfg.EXIT_ALIGN_EXIT_CENTER_ERROR
        return cfg.EXIT_ALIGN_ENTER_CENTER_ERROR

    @staticmethod
    def _interpolate(minimo, maximo, proporcao):
        proporcao = min(max(float(proporcao), 0.0), 1.0)
        return float(minimo + (maximo - minimo) * proporcao)

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
