"""Contrato de troca de hardware da missão completa.

O estado de alto nível pertence exclusivamente a ``mission.py``. Este módulo
mantém apenas a ordem testável do handoff entre o segue-linha e a rotina de
resgate.
"""


# ---------------------------------------------------------------------------
# Handoff entre segue-linha e resgate
# ---------------------------------------------------------------------------
# A ordem abaixo é o contrato de segurança da missão. Cada passo é o nome de
# um método do objeto "sistema" injetado no executor.
#
# Observação importante sobre o LED: o regulamento interno do Shadow manda
# apagá-lo antes do resgate, mas apagar o LED exige a serial. Por isso ele é
# apagado enquanto a serial do segue-linha ainda existe (único momento em que
# isso é possível) e a rotina de resgate REAFIRMA o LED apagado assim que
# abre a própria serial. Os dois passos estão explícitos na lista.
HANDOFF_TO_RESCUE = (
    "stop_motors",              # PARAR antes de qualquer desmontagem
    "led_off",                  # ainda na serial do segue-linha
    "terminate_line_children",  # sinaliza visão e controle
    "join_line_children",       # espera de fato terminarem
    "assert_line_children_dead",
    "close_line_camera",        # câmera 1 liberada
    "release_serial",           # serial do segue-linha liberada
    "release_motor_lock",       # MotorOwnerLock liberado
    "acquire_rescue_motor_lock",
    "open_rescue_serial",
    "assert_led_off",           # reafirma na serial nova
    "open_rescue_camera",       # somente agora a câmera 0 abre
    "start_rescue",
)

# A ordem da volta espelha a ida. `open_line_camera` vem antes de
# `open_line_serial` porque é essa a ordem real de partida dos filhos do
# segue-linha (visão primeiro, controle depois) — e a ordem entre câmera e
# serial não é uma propriedade de segurança: as duas pertencem a donos
# diferentes. O que É propriedade de segurança, e está garantido, é a câmera
# de resgate fechar antes de a de linha abrir e a serial ter um dono só.
HANDOFF_TO_LINE = (
    "stop_motors",
    "close_rescue_camera",      # câmera 0 liberada
    "release_serial",
    "release_motor_lock",
    "acquire_line_motor_lock",
    "open_line_camera",         # câmera 1 reaberta
    "open_line_serial",
    "led_on",                   # volta ao percurso com LED aceso
    "reacquire_line",
)


class HandoffError(RuntimeError):
    pass


class HandoffExecutor:
    """Executa uma sequência de handoff contra um objeto de sistema.

    O executor é intencionalmente burro: ele apenas chama os passos na ordem
    e registra o que foi chamado. Toda a inteligência está na ordem declarada
    acima, que é o que os testes verificam. Se um passo falhar, ele para,
    tenta ``stop_motors`` e propaga — nunca continua abrindo a câmera
    seguinte com o processo anterior ainda vivo.
    """

    def __init__(self, system, steps):
        self.system = system
        self.steps = tuple(steps)
        self.log = []
        self.failed_step = None

    def run(self):
        for step in self.steps:
            action = getattr(self.system, step, None)
            if action is None:
                raise HandoffError(
                    f"o sistema não implementa o passo de handoff: {step}")
            try:
                action()
            except Exception as err:
                self.failed_step = step
                self._emergency_stop()
                raise HandoffError(
                    f"handoff abortado em '{step}': {err}") from err
            self.log.append(step)
        return tuple(self.log)

    def _emergency_stop(self):
        stop = getattr(self.system, "stop_motors", None)
        if stop is None:
            return
        try:
            stop()
            self.log.append("stop_motors")
        except Exception:
            # O watchdog de 1 s do Uno ainda corta os motores sozinho.
            pass


def index_of(log, step):
    """Posição de um passo no registro do handoff (-1 se não ocorreu)."""
    try:
        return list(log).index(step)
    except ValueError:
        return -1
