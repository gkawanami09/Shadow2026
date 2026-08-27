"""Executa o retorno de 180 graus indicado por dois verdes."""

import math
import time

from config import (GREEN_LOCKED_BRANCH_MIN_Y_RATIO,
                    GREEN_MPU_QUERY_INTERVAL_S, GREEN_MPU_RESPONSE_TIMEOUT_S,
                    GREEN_TURN_SIDE_MIN_ERROR_PX,
                    LINE_MAX_FRAME_AGE_S, T_180, T_180_BLIND_EXTRA,
                    T_180_CONFIRM_FRAMES,
                    T_180_EXIT_BOTTOM_PX, T_180_MPU_HARD_LIMIT_DEG,
                    T_180_MPU_MIN_COMPLETION_DEG, T_180_MPU_SLOW_SPEED,
                    T_180_MPU_SLOWDOWN_DEG, T_180_POST_REVERSE_TIMEOUT,
                    T_180_SEARCH_SPEED,
                    T_180_SEARCH_TIMEOUT, T_180_SPEED, T_180_TEST_STOP,
                    TURN_AROUND_PREROLL, TURN_AROUND_REVERSE,
                    TURN_AROUND_REVERSE_EXTRA, TURN_AROUND_SMALL_LINE,
                    camera_x, camera_y)
from controle.direcao import sleep_steering, steer
from shared.dados_compartilhados import (ler_resultado_visao_rapida, line_size,
                                         status, terminate)


class _ConfirmadorSaida180:
    """Exige alinhamento em frames novos, nunca em telemetria congelada."""

    def __init__(self, required_frames=T_180_CONFIRM_FRAMES, *,
                 require_side_entry=False, expected_side=1,
                 expected_branch_token=0):
        self.required_frames = max(int(required_frames), 1)
        if int(expected_side) not in (-1, 1):
            raise ValueError("expected_side precisa ser -1 ou 1")
        self.require_side_entry = bool(require_side_entry)
        self.expected_side = int(expected_side)
        self.expected_branch_token = max(int(expected_branch_token), 0)
        self.side_seen = False
        self.last_sequence = -1
        self.aligned_frames = 0

    def _locked_branch_x(self, resultado):
        if not self.require_side_entry:
            return float(resultado.ponto_inferior_x)
        if (
            self.expected_branch_token <= 0
            or not bool(getattr(resultado, "locked_branch_valid", False))
            or int(getattr(resultado, "locked_branch_token", 0))
            != self.expected_branch_token
        ):
            return None
        try:
            value = float(resultado.locked_branch_bottom_x)
            value_y = float(resultado.locked_branch_bottom_y)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            not math.isfinite(value)
            or not math.isfinite(value_y)
            or value_y < camera_y * GREEN_LOCKED_BRANCH_MIN_Y_RATIO
        ):
            return None
        return value

    def note_branch_side(self, resultado, *, now):
        """Trava a identidade do ramo antes que ele chegue ao centro."""

        recent = bool(
            0. <= now - float(resultado.publicado_em)
            <= LINE_MAX_FRAME_AGE_S
        )
        branch_x = self._locked_branch_x(resultado)
        if (
            recent
            and resultado.linha_detectada
            and branch_x is not None
            and self.expected_side * (branch_x - camera_x / 2)
            >= GREEN_TURN_SIDE_MIN_ERROR_PX
        ):
            self.side_seen = True
        return self.side_seen

    def update(self, resultado, *, now, mpu_allows, junction_clear=None):
        sequence = int(resultado.sequencia)
        if sequence == self.last_sequence:
            if now - float(resultado.publicado_em) > LINE_MAX_FRAME_AGE_S:
                self.aligned_frames = 0
            return self.aligned_frames >= self.required_frames

        self.last_sequence = sequence
        self.note_branch_side(resultado, now=now)
        branch_x = self._locked_branch_x(resultado)
        recent = bool(
            0. <= now - float(resultado.publicado_em)
            <= LINE_MAX_FRAME_AGE_S
        )
        if junction_clear is None:
            # Falha fechado para objetos legados: só o frame atômico que
            # carrega a visibilidade da mesma sequência pode liberar a saída.
            junction_clear = not bool(getattr(
                resultado, "juncao_topologica_visivel", True))
        aligned = bool(
            recent
            and resultado.linha_detectada
            and branch_x is not None
            and abs(branch_x - camera_x / 2)
            <= T_180_EXIT_BOTTOM_PX
            and mpu_allows
            and junction_clear
            and (self.side_seen or not self.require_side_entry)
        )
        self.aligned_frames = self.aligned_frames + 1 if aligned else 0
        return self.aligned_frames >= self.required_frames


class _MonitorMpu180:
    """Consulta yaw sem tornar o MPU obrigatorio para o retorno."""

    def __init__(self, arduino, tracker, callback=None):
        self.arduino = arduino
        self.tracker = tracker
        self.callback = callback
        self.next_query = 0.0
        self.last_progress = None
        self.last_valid_at = None
        self.last_generation = 0
        self.active = bool(
            arduino is not None
            and tracker is not None
            and hasattr(arduino, "poll_mpu")
            and hasattr(arduino, "iniciar_mpu")
        )

    def poll(self, now=None):
        if not self.active:
            return None
        now = time.monotonic() if now is None else float(now)
        concluido, leitura = self.arduino.poll_mpu()
        if concluido and leitura is not None:
            generation = int(getattr(leitura, "request_generation", 0))
            received_at = float(getattr(leitura, "received_at", 0.0))
            generation_is_new = bool(
                generation > 0
                and generation > self.last_generation
                and math.isfinite(received_at)
                and 0. < received_at <= now + .01
            )
            if generation_is_new:
                self.last_generation = max(self.last_generation, generation)
                self.last_progress = self.tracker.update(
                    leitura.yaw_graus,
                    received_at,
                    now=now,
                )
                if self.last_progress.valid:
                    self.last_valid_at = received_at
                    if self.callback is not None:
                        self.callback(float(leitura.yaw_graus))
        if now >= self.next_query:
            self.arduino.iniciar_mpu(timeout=GREEN_MPU_RESPONSE_TIMEOUT_S)
            self.next_query = now + GREEN_MPU_QUERY_INTERVAL_S
        return self.last_progress

    def fresh(self, now=None):
        if self.last_valid_at is None:
            return False
        now = time.monotonic() if now is None else float(now)
        return (
            0. <= now - self.last_valid_at
            <= max(GREEN_MPU_RESPONSE_TIMEOUT_S * 2., .20)
        )

    def must_abort(self, now=None):
        return bool(
            self.fresh(now)
            and self.last_progress is not None
            and (
                self.last_progress.wrong_direction
                or self.last_progress.progress_deg
                >= T_180_MPU_HARD_LIMIT_DEG
            )
        )

    def turn_speed(self, normal_speed, now=None):
        if (
            self.fresh(now)
            and self.last_progress is not None
            and self.last_progress.progress_deg >= T_180_MPU_SLOWDOWN_DEG
        ):
            return min(float(normal_speed), T_180_MPU_SLOW_SPEED)
        return float(normal_speed)

    def camera_may_finish(self, now=None):
        if not self.fresh(now) or self.last_progress is None:
            return True
        return (
            not self.last_progress.wrong_direction
            and self.last_progress.progress_deg
            >= T_180_MPU_MIN_COMPLETION_DEG
        )


def _abort_requested(should_abort=None):
    if terminate.value:
        return True
    if should_abort is None:
        return False
    try:
        return bool(should_abort())
    except Exception:
        return True


def _wait_interruptible(duration, should_abort=None):
    end = time.monotonic() + max(float(duration), 0.)
    while True:
        now = time.monotonic()
        if _abort_requested(should_abort):
            steer()
            return False
        if now >= end:
            return True
        if sleep_steering(min(.02, end - now)) is False:
            steer()
            return False


def _turn_for(duration, speed, monitor, should_abort=None,
              vision_observer=None):
    """Gira para a direita, consultando yaw quando ele estiver disponivel."""
    duration = max(float(duration), 0.)
    if not monitor.active and vision_observer is None:
        if _abort_requested(should_abort):
            steer()
            return False
        steer(180, speed)
        return _wait_interruptible(duration, should_abort)

    end = time.monotonic() + duration
    while True:
        now = time.monotonic()
        if _abort_requested(should_abort):
            steer()
            return False
        if now >= end:
            return True
        if monitor.active:
            monitor.poll(now)
            if monitor.must_abort(now):
                steer()
                return False
        if vision_observer is not None:
            try:
                vision_observer(now)
            except Exception:
                steer()
                return False
        steer(180, monitor.turn_speed(speed, now))
        if sleep_steering(min(.02, end - now)) is False:
            steer()
            return False


def turn_around(
    _last_turn_dir,
    *,
    require_alignment=False,
    arduino=None,
    yaw_tracker=None,
    yaw_callback=None,
    should_abort=None,
    expected_branch_token=0,
):
    """Executa o retorno de 180° sempre pelo lado direito."""
    monitor_mpu = _MonitorMpu180(arduino, yaw_tracker, yaw_callback)
    # O retorno e sempre horario. Depois de ~105 graus, o ramo de onde o robo
    # veio deve entrar pelo lado direito antes de poder chegar ao centro.
    confirmador_saida = _ConfirmadorSaida180(
        require_side_entry=True,
        expected_side=1,
        expected_branch_token=expected_branch_token,
    )

    # avanca por cima do marcador duplo
    if _abort_requested(should_abort):
        steer()
        return None
    steer(0, .7)
    if not _wait_interruptible(TURN_AROUND_PREROLL, should_abort):
        steer()
        return None

    # Pivô temporizado, pois o robô não possui giroscópio.
    if not _turn_for(T_180, T_180_SPEED, monitor_mpu, should_abort):
        steer()
        return None
    # Completa mais 0,3 s no mesmo giro, ainda sem consultar a câmera.
    if not _turn_for(
        T_180_BLIND_EXTRA,
        T_180_SPEED,
        monitor_mpu,
        should_abort,
        vision_observer=lambda agora: confirmador_saida.note_branch_side(
            ler_resultado_visao_rapida(),
            now=agora,
        ),
    ):
        steer()
        return None
    steer()

    # Modo temporario de afericao: isola somente o giro cronometrado. Mantem
    # PARAR ate o operador encerrar o programa, sem busca visual, re ou
    # retomada automatica do segue-linha mascararem o angulo obtido.
    if T_180_TEST_STOP:
        status.value = 'Teste 180 concluido — parado apos o giro'
        while not terminate.value:
            sleep_steering(.05)
        return "r"

    # Depois da parte cega, reduz a velocidade e continua no mesmo sentido ate
    # a camera confirmar a linha centralizada. Somente a posicao inferior pode
    # concluir o giro, pois ela representa diretamente a bolinha azul.
    steer(180, T_180_SEARCH_SPEED)
    status.value = 'Completando 180 — procurando linha no centro'
    search_end = time.monotonic() + T_180_SEARCH_TIMEOUT
    saida_alinhada = False
    while time.monotonic() < search_end:
        agora = time.monotonic()
        if _abort_requested(should_abort):
            steer()
            return None
        monitor_mpu.poll(agora)
        if monitor_mpu.must_abort(agora):
            steer()
            return None
        steer(180, monitor_mpu.turn_speed(T_180_SEARCH_SPEED, agora))
        resultado_visao = ler_resultado_visao_rapida()
        if confirmador_saida.update(
            resultado_visao,
            now=agora,
            mpu_allows=monitor_mpu.camera_may_finish(agora),
        ):
            saida_alinhada = True
            break
        if sleep_steering(.01) is False:
            steer()
            return None

    steer()

    if require_alignment and not saida_alinhada:
        return None

    # Dá ré até a câmera voltar a encontrar a linha.
    if _abort_requested(should_abort):
        steer()
        return None
    steer(200, .7)
    if not _wait_interruptible(TURN_AROUND_REVERSE, should_abort):
        steer()
        return None
    steer()

    if line_size.value < TURN_AROUND_SMALL_LINE:
        steer(200, .7)
        if not _wait_interruptible(
            TURN_AROUND_REVERSE_EXTRA, should_abort,
        ):
            steer()
            return None
        steer()

    if require_alignment:
        # A re altera novamente a pose. O frame que encerrou o giro nao pode
        # autorizar a retomada: aguardamos tres sequencias capturadas depois
        # de o robo estar parado na pose final.
        confirmador_pos_re = _ConfirmadorSaida180()
        confirmador_pos_re.last_sequence = int(
            ler_resultado_visao_rapida().sequencia)
        limite_pos_re = time.monotonic() + T_180_POST_REVERSE_TIMEOUT
        alinhado_pos_re = False
        while time.monotonic() < limite_pos_re:
            agora = time.monotonic()
            if _abort_requested(should_abort):
                steer()
                return None
            resultado_pos_re = ler_resultado_visao_rapida()
            if confirmador_pos_re.update(
                resultado_pos_re,
                now=agora,
                mpu_allows=True,
            ):
                alinhado_pos_re = True
                break
            if sleep_steering(.01) is False:
                steer()
                return None
        if not alinhado_pos_re:
            steer()
            return None

    return "r"
