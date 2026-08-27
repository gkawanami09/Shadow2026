"""Eventos persistentes e estados monotônicos para manobras verdes.

Este módulo não conhece câmera, motores ou variáveis compartilhadas.  Ele
recebe observações geométricas prontas e oferece duas garantias importantes:

* uma decisão só nasce depois de confirmação temporal da mesma junção e dos
  mesmos marcadores;
* depois do compromisso, nenhum frame posterior consegue trocar ou cancelar
  a direção até que o evento seja consumido.

Os enums são ``IntEnum`` para que a integração possa transportá-los em memória
compartilhada sem serialização Python.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import IntEnum
import math
from typing import Deque, Iterable, Optional, Tuple


Point = Tuple[float, float]
GREEN_OBSERVATION_ATOMIC_SIZE = 19


class GreenDecision(IntEnum):
    """Resultado topológico de uma junção."""

    NONE = 0
    PENDING = 1
    STRAIGHT = 2
    LEFT = 3
    RIGHT = 4
    UTURN = 5


class GreenManeuverState(IntEnum):
    """Estados de uma manobra; a ordem numérica também é a ordem válida."""

    FOLLOW = 0
    OBSERVE = 1
    COMMITTED = 2
    APPROACH = 3
    TURNING = 4
    REACQUIRE = 5
    COOLDOWN = 6
    FAULT_STOP = 7


TURN_DECISIONS = frozenset((
    GreenDecision.LEFT,
    GreenDecision.RIGHT,
    GreenDecision.UTURN,
))
COMMITTABLE_DECISIONS = frozenset((
    GreenDecision.STRAIGHT,
    GreenDecision.LEFT,
    GreenDecision.RIGHT,
    GreenDecision.UTURN,
))


def calibracao_permite_motores(*, obrigatoria: bool, pronta: bool) -> bool:
    """Gate competitivo: calibração obrigatória ausente nunca arma motores."""
    return bool(pronta or not obrigatoria)


def _point(value: Iterable[float]) -> Point:
    x, y = value
    return float(x), float(y)


def _same_committed_identity(
    committed: "GreenObservation",
    observation: "GreenObservation",
) -> bool:
    """Confere a identidade autorizada a atualizar uma geometria travada.

    O marcador POST de uma decisao STRAIGHT pode aparecer ou sair da imagem
    durante a travessia. Se um dos lados ainda nao possui ids, a geometria
    continua sendo da mesma intersecao, mas os ids comprometidos permanecem
    imutaveis. Dois conjuntos nao vazios e diferentes continuam conflitantes.
    """

    if (
        observation.junction_id != committed.junction_id
        or observation.decision != committed.decision
    ):
        return False
    if observation.marker_ids == committed.marker_ids:
        return True
    return bool(
        committed.decision == GreenDecision.STRAIGHT
        and (not committed.marker_ids or not observation.marker_ids)
    )


@dataclass(frozen=True, slots=True)
class GreenObservation:
    """Observação atômica publicada pela visão.

    ``marker_ids`` contém identidades geométricas estáveis dentro de uma
    junção. Para um retorno são esperadas duas identidades, ordenadas.  Uma
    observação bruta usa ``decision_id=0``; o confirmador atribui um id
    estritamente crescente ao compromisso.

    Contrato de coordenadas (mantido explícito porque os campos atravessam
    processos): ``entry_tangent`` está no plano retificado do chão, com X
    positivo para a direita e Y positivo para a frente;
    ``junction_center`` está em pixels da imagem retificada;
    ``target_branch`` está em pixels do frame cru enviado ao controle.
    """

    sequence: int
    junction_id: int
    decision_id: int
    timestamp: float
    decision: GreenDecision
    confidence: float
    entry_tangent: Point = (0.0, 1.0)
    junction_center: Point = (-1.0, -1.0)
    target_branch: Point = (-1.0, -1.0)
    target_branch_token: int = 0
    ready_to_turn: bool = False
    junction_visible: bool = False
    geometry_predicted: bool = False
    marker_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "junction_id", int(self.junction_id))
        object.__setattr__(self, "decision_id", int(self.decision_id))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "decision", GreenDecision(self.decision))
        confidence = float(self.confidence)
        if not math.isfinite(confidence):
            confidence = 0.0
        object.__setattr__(self, "confidence", min(max(confidence, 0.0), 1.0))
        object.__setattr__(self, "entry_tangent", _point(self.entry_tangent))
        object.__setattr__(self, "junction_center", _point(self.junction_center))
        object.__setattr__(self, "target_branch", _point(self.target_branch))
        object.__setattr__(
            self, "target_branch_token", max(int(self.target_branch_token), 0))
        object.__setattr__(self, "ready_to_turn", bool(self.ready_to_turn))
        object.__setattr__(self, "junction_visible", bool(self.junction_visible))
        object.__setattr__(self, "geometry_predicted", bool(self.geometry_predicted))
        object.__setattr__(
            self,
            "marker_ids",
            tuple(sorted({int(marker_id) for marker_id in self.marker_ids})),
        )

    @property
    def committed(self) -> bool:
        return (
            self.decision_id > 0
            and self.decision in COMMITTABLE_DECISIONS
        )

    @property
    def is_turn(self) -> bool:
        return self.decision in TURN_DECISIONS

    @property
    def entry_tangent_ground(self) -> Point:
        """Tangente no plano do chão (X direita, Y frente)."""

        return self.entry_tangent

    @property
    def junction_rectified(self) -> Point:
        """Centro da junção em pixels da imagem retificada."""

        return self.junction_center

    @property
    def target_branch_raw(self) -> Point:
        """Alvo do ramo em pixels do frame cru."""

        return self.target_branch

    def as_atomic_values(self) -> Tuple[float, ...]:
        """Codifica em doubles, sem pickle nem Manager intermediário."""

        marker_ids = self.marker_ids[:2]
        marker_a = marker_ids[0] if marker_ids else 0
        marker_b = marker_ids[1] if len(marker_ids) > 1 else 0
        return (
            float(self.sequence),
            float(self.junction_id),
            float(self.decision_id),
            self.timestamp,
            float(int(self.decision)),
            self.confidence,
            self.entry_tangent[0],
            self.entry_tangent[1],
            self.junction_center[0],
            self.junction_center[1],
            self.target_branch[0],
            self.target_branch[1],
            float(self.target_branch_token),
            float(self.ready_to_turn),
            float(self.junction_visible),
            float(self.geometry_predicted),
            float(len(marker_ids)),
            float(marker_a),
            float(marker_b),
        )

    @classmethod
    def from_atomic_values(cls, values: Iterable[float]) -> "GreenObservation":
        values = tuple(values)
        if len(values) != GREEN_OBSERVATION_ATOMIC_SIZE:
            raise ValueError(
                f"observação atômica precisa de {GREEN_OBSERVATION_ATOMIC_SIZE} valores"
            )
        marker_count = min(max(int(values[16]), 0), 2)
        marker_ids = tuple(int(values[17 + index]) for index in range(marker_count))
        return cls(
            sequence=int(values[0]),
            junction_id=int(values[1]),
            decision_id=int(values[2]),
            timestamp=float(values[3]),
            decision=GreenDecision(int(values[4])),
            confidence=float(values[5]),
            entry_tangent=(values[6], values[7]),
            junction_center=(values[8], values[9]),
            target_branch=(values[10], values[11]),
            target_branch_token=int(values[12]),
            ready_to_turn=bool(values[13]),
            junction_visible=bool(values[14]),
            geometry_predicted=bool(values[15]),
            marker_ids=marker_ids,
        )


def empty_observation(sequence: int = 0, timestamp: float = 0.0) -> GreenObservation:
    """Cria o valor neutro usado na memória compartilhada."""

    return GreenObservation(
        sequence=sequence,
        junction_id=0,
        decision_id=0,
        timestamp=timestamp,
        decision=GreenDecision.NONE,
        confidence=0.0,
    )


class GreenDecisionTracker:
    """Confirma e trava uma decisão verde até seu consumo explícito.

    LEFT, RIGHT e UTURN exigem ``confirm_frames`` dentro dos últimos
    ``window_frames`` para a mesma tupla (junção, marcadores, decisão). Depois
    da confirmação de um marcador único, LEFT/RIGHT permanece PENDING por até
    ``second_marker_wait_s`` para permitir que o segundo marcador produza um
    UTURN confirmado. STRAIGHT também é filtrado temporalmente por segurança,
    mas não abre a janela de espera pelo segundo marcador.
    """

    def __init__(
        self,
        *,
        confirm_frames: int = 3,
        window_frames: int = 5,
        second_marker_wait_s: float = 0.35,
        prediction_max_s: float = 0.20,
        rearm_frames: int = 3,
        rearm_min_s: float = 0.0,
        first_decision_id: int = 1,
    ) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames precisa ser positivo")
        if window_frames < confirm_frames:
            raise ValueError("window_frames precisa comportar confirm_frames")
        if second_marker_wait_s < 0.0:
            raise ValueError("second_marker_wait_s não pode ser negativo")
        if prediction_max_s < 0.0:
            raise ValueError("prediction_max_s não pode ser negativo")
        if rearm_frames < 1:
            raise ValueError("rearm_frames precisa ser positivo")
        if rearm_min_s < 0.0:
            raise ValueError("rearm_min_s nao pode ser negativo")

        self.confirm_frames = int(confirm_frames)
        self.window_frames = int(window_frames)
        self.second_marker_wait_s = float(second_marker_wait_s)
        self.prediction_max_s = float(prediction_max_s)
        self.rearm_frames = int(rearm_frames)
        self.rearm_min_s = float(rearm_min_s)
        self._history: Deque[GreenObservation] = deque(maxlen=self.window_frames)
        self._next_decision_id = max(int(first_decision_id), 1)
        self._committed: Optional[GreenObservation] = None
        self._single_candidate: Optional[GreenObservation] = None
        self._single_deadline: Optional[float] = None
        self._last_input = empty_observation()
        self._new_commit = False
        self._cooldown = False
        self._rearm_count = 0
        self._consumed_ids: set[int] = set()
        self._consumed_at: Optional[float] = None
        self._last_geometry_timestamp: Optional[float] = None

    @property
    def committed(self) -> Optional[GreenObservation]:
        return self._committed

    @property
    def new_commit(self) -> bool:
        """Verdadeiro somente na chamada que criou o compromisso."""

        return self._new_commit

    @property
    def waiting_second_marker(self) -> bool:
        return self._single_candidate is not None and self._committed is None

    @property
    def second_marker_deadline(self) -> Optional[float]:
        return self._single_deadline

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown

    @property
    def last_consumed_id(self) -> int:
        return max(self._consumed_ids, default=0)

    def _same_scene(self, a: GreenObservation, b: GreenObservation) -> bool:
        return (
            a.junction_id > 0
            and a.junction_id == b.junction_id
            and a.marker_ids == b.marker_ids
            and a.decision == b.decision
        )

    def _matching_frames(self, observation: GreenObservation) -> list[GreenObservation]:
        return [
            frame for frame in self._history
            if (
                frame.junction_visible
                and not frame.geometry_predicted
                and self._same_scene(frame, observation)
            )
        ]

    def _confirmed_version(
        self,
        observation: GreenObservation,
    ) -> Optional[GreenObservation]:
        matches = self._matching_frames(observation)
        if len(matches) < self.confirm_frames:
            return None
        # Geometria do frame mais recente; confiança média evita que um único
        # pico de confiança esconda frames fracos dentro da confirmação.
        confidence = sum(frame.confidence for frame in matches) / len(matches)
        return replace(observation, confidence=confidence)

    def _pending_from(self, observation: GreenObservation) -> GreenObservation:
        return replace(
            observation,
            decision_id=0,
            decision=GreenDecision.PENDING,
            ready_to_turn=False,
        )

    def _commit(
        self,
        observation: GreenObservation,
        *,
        timestamp: Optional[float] = None,
        sequence: Optional[int] = None,
        geometry_timestamp: Optional[float] = None,
    ) -> GreenObservation:
        if self._committed is not None:
            return self._committed
        event = replace(
            observation,
            sequence=(observation.sequence if sequence is None else int(sequence)),
            decision_id=self._next_decision_id,
            timestamp=(observation.timestamp if timestamp is None else float(timestamp)),
        )
        self._next_decision_id += 1
        self._committed = event
        self._last_geometry_timestamp = (
            event.timestamp
            if geometry_timestamp is None
            else float(geometry_timestamp)
        )
        self._single_candidate = None
        self._single_deadline = None
        self._new_commit = True
        return event

    def _single_geometry_age(self, timestamp: float) -> float:
        if self._single_candidate is None:
            return math.inf
        return float(timestamp) - self._single_candidate.timestamp

    def _pending_single_geometry(
        self,
        *,
        timestamp: float,
        sequence: int,
    ) -> GreenObservation:
        """Publica PENDING atual sem fingir que a geometria antiga e visivel."""

        candidate = self._single_candidate
        if candidate is None:
            return empty_observation(sequence, timestamp)
        age = self._single_geometry_age(timestamp)
        predicted = 0. <= age <= self.prediction_max_s
        return replace(
            candidate,
            sequence=int(sequence),
            timestamp=float(timestamp),
            decision_id=0,
            decision=GreenDecision.PENDING,
            ready_to_turn=False,
            junction_visible=False,
            geometry_predicted=predicted,
        )

    def _commit_single_if_fresh(
        self,
        *,
        timestamp: float,
        sequence: int,
    ) -> GreenObservation:
        """Confirma a direcao sem ultrapassar a previsao geometrica maxima."""

        candidate = self._single_candidate
        if candidate is None:
            return empty_observation(sequence, timestamp)
        age = self._single_geometry_age(timestamp)
        if not 0. <= age <= self.prediction_max_s:
            return self._pending_single_geometry(
                timestamp=timestamp,
                sequence=sequence,
            )
        if age > 1e-9:
            candidate = replace(
                candidate,
                junction_visible=False,
                geometry_predicted=True,
            )
        return self._commit(
            candidate,
            timestamp=timestamp,
            sequence=sequence,
            geometry_timestamp=self._single_candidate.timestamp,
        )

    def _refresh_single_candidate(self, observation: GreenObservation) -> bool:
        """Atualiza somente geometria real da identidade unica confirmada."""

        candidate = self._single_candidate
        if candidate is None:
            return False
        if not self._same_scene(candidate, observation):
            return False
        if not observation.junction_visible or observation.geometry_predicted:
            return False
        if observation.timestamp < candidate.timestamp:
            return False
        # O prazo pertence a primeira confirmacao e nunca e prorrogado.
        self._single_candidate = replace(observation, decision_id=0)
        return True

    def update(self, observation: GreenObservation) -> GreenObservation:
        """Ingere um frame e devolve NONE, PENDING ou o evento travado."""

        if not isinstance(observation, GreenObservation):
            raise TypeError("observation precisa ser GreenObservation")
        self._new_commit = False
        self._last_input = observation

        if self._committed is not None:
            # ID, direção e marcadores são imutáveis. A posição da mesma
            # junção, entretanto, continua avançando até o gatilho de giro.
            identidade_valida = _same_committed_identity(
                self._committed, observation)
            if (identidade_valida
                    and observation.junction_visible
                    and not observation.geometry_predicted):
                self._committed = replace(
                    self._committed,
                    sequence=observation.sequence,
                    timestamp=observation.timestamp,
                    junction_center=observation.junction_center,
                    target_branch=observation.target_branch,
                    target_branch_token=observation.target_branch_token,
                    ready_to_turn=observation.ready_to_turn,
                    junction_visible=True,
                    geometry_predicted=False,
                )
                self._last_geometry_timestamp = observation.timestamp
            else:
                idade = (
                    math.inf
                    if self._last_geometry_timestamp is None
                    else observation.timestamp - self._last_geometry_timestamp
                )
                self._committed = replace(
                    self._committed,
                    sequence=observation.sequence,
                    timestamp=observation.timestamp,
                    junction_visible=False,
                    geometry_predicted=(0. <= idade <= self.prediction_max_s),
                    ready_to_turn=(
                        self._committed.ready_to_turn
                        if 0. <= idade <= self.prediction_max_s
                        else False
                    ),
                )
            return self._committed
        if self._cooldown:
            return empty_observation(observation.sequence, observation.timestamp)

        if self._history and (
            observation.junction_id > 0
            and self._history[-1].junction_id > 0
            and observation.junction_id != self._history[-1].junction_id
        ):
            self._history.clear()
            self._single_candidate = None
            self._single_deadline = None

        self._history.append(observation)

        if observation.decision == GreenDecision.UTURN:
            confirmed = self._confirmed_version(observation)
            if confirmed is not None:
                return self._commit(confirmed)

        if observation.decision in (GreenDecision.LEFT, GreenDecision.RIGHT):
            confirmed = self._confirmed_version(observation)
            if confirmed is not None and self._single_candidate is None:
                self._single_candidate = confirmed
                self._single_deadline = (
                    observation.timestamp + self.second_marker_wait_s
                )
            elif self._single_candidate is not None:
                self._refresh_single_candidate(observation)
            if self._single_candidate is not None:
                if observation.timestamp >= float(self._single_deadline):
                    return self._commit_single_if_fresh(
                        timestamp=observation.timestamp,
                        sequence=observation.sequence,
                    )
                return self._pending_single_geometry(
                    timestamp=observation.timestamp,
                    sequence=observation.sequence,
                )

        if self._single_candidate is not None:
            if observation.timestamp >= float(self._single_deadline):
                return self._commit_single_if_fresh(
                    timestamp=observation.timestamp,
                    sequence=observation.sequence,
                )
            return self._pending_single_geometry(
                timestamp=observation.timestamp,
                sequence=observation.sequence,
            )

        if observation.decision == GreenDecision.STRAIGHT:
            confirmed = self._confirmed_version(observation)
            if confirmed is not None:
                return self._commit(confirmed)

        if observation.decision in COMMITTABLE_DECISIONS or (
            observation.decision == GreenDecision.PENDING
        ):
            return self._pending_from(observation)
        return empty_observation(observation.sequence, observation.timestamp)

    def tick(
        self,
        timestamp: float,
        *,
        sequence: Optional[int] = None,
    ) -> GreenObservation:
        """Avança somente o relógio, útil quando o candidato some da imagem."""

        self._new_commit = False
        if self._committed is not None:
            return self._committed
        if self._cooldown:
            return empty_observation(
                self._last_input.sequence if sequence is None else sequence,
                timestamp,
            )
        if (
            self._single_candidate is not None
            and timestamp >= float(self._single_deadline)
        ):
            return self._commit_single_if_fresh(
                timestamp=timestamp,
                sequence=(
                    self._last_input.sequence if sequence is None else sequence
                ),
            )
        if self._single_candidate is not None:
            return self._pending_single_geometry(
                timestamp=timestamp,
                sequence=(
                    self._last_input.sequence if sequence is None else sequence
                ),
            )
        return empty_observation(
            self._last_input.sequence if sequence is None else sequence,
            timestamp,
        )

    def consume(
        self,
        decision_id: int,
        *,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Consome exatamente uma vez e bloqueia rearme até a saída estabilizar."""

        decision_id = int(decision_id)
        if decision_id in self._consumed_ids:
            return False
        if self._committed is None or self._committed.decision_id != decision_id:
            return False
        # Com cooldown temporal positivo, omitir o relogio deve falhar fechado
        # em note_rearm_frame, nunca inferir um instante antigo do evento.
        self._consumed_at = None if timestamp is None else float(timestamp)
        self._consumed_ids.add(decision_id)
        self._committed = None
        self._history.clear()
        self._single_candidate = None
        self._single_deadline = None
        self._last_geometry_timestamp = None
        self._cooldown = True
        self._rearm_count = 0
        return True

    def note_rearm_frame(
        self,
        *,
        junction_visible: bool,
        exit_line_stable: bool,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Rearma após junção ausente e linha de saída estável por N frames."""

        if not self._cooldown:
            return True
        if self.rearm_min_s > 0.0:
            if timestamp is None or self._consumed_at is None:
                self._rearm_count = 0
                return False
            if float(timestamp) - self._consumed_at < self.rearm_min_s:
                self._rearm_count = 0
                return False
        if (not junction_visible) and exit_line_stable:
            self._rearm_count += 1
        else:
            self._rearm_count = 0
        if self._rearm_count < self.rearm_frames:
            return False
        self._cooldown = False
        self._rearm_count = 0
        self._consumed_at = None
        self._history.clear()
        return True


class GreenManeuverFSM:
    """Máquina de estados monotônica de uma manobra já confirmada."""

    def __init__(self) -> None:
        self.state = GreenManeuverState.FOLLOW
        self.event: Optional[GreenObservation] = None
        self.state_since = 0.0
        self.deadline: Optional[float] = None
        self.fault_reason = ""

    @property
    def decision_id(self) -> int:
        return 0 if self.event is None else self.event.decision_id

    @property
    def locked_direction(self) -> GreenDecision:
        return GreenDecision.NONE if self.event is None else self.event.decision

    @property
    def stopped(self) -> bool:
        return self.state == GreenManeuverState.FAULT_STOP

    def _fault_if_deadline_expired(self, *, now: float) -> bool:
        """Falha fechado antes que qualquer chamada possa pular um timeout."""

        if self.stopped:
            return True
        if self.deadline is None or float(now) < self.deadline:
            return False
        self.fault("timeout", now=now)
        return True

    def observe(
        self,
        observation: GreenObservation,
        *,
        now: Optional[float] = None,
        observe_timeout_s: Optional[float] = None,
    ) -> bool:
        """Aceita PENDING ou um compromisso; conflitos nunca alteram o evento."""

        timestamp = observation.timestamp if now is None else float(now)
        if self._fault_if_deadline_expired(now=timestamp):
            return False
        if observation.decision == GreenDecision.PENDING:
            if self.state == GreenManeuverState.FOLLOW:
                self.state = GreenManeuverState.OBSERVE
                self.state_since = timestamp
                self.deadline = (
                    None
                    if observe_timeout_s is None
                    else timestamp + max(float(observe_timeout_s), 0.0)
                )
            return self.event is None
        if not observation.committed:
            return False
        if self.event is not None:
            mesmo_evento = (
                observation.decision_id == self.event.decision_id
                and _same_committed_identity(self.event, observation)
            )
            if mesmo_evento:
                self.event = replace(
                    self.event,
                    sequence=observation.sequence,
                    timestamp=observation.timestamp,
                    junction_center=observation.junction_center,
                    target_branch=observation.target_branch,
                    target_branch_token=observation.target_branch_token,
                    ready_to_turn=observation.ready_to_turn,
                    junction_visible=observation.junction_visible,
                    geometry_predicted=observation.geometry_predicted,
                )
            return mesmo_evento
        if self.state not in (
            GreenManeuverState.FOLLOW,
            GreenManeuverState.OBSERVE,
        ):
            return False
        self.event = observation
        self.state = GreenManeuverState.COMMITTED
        self.state_since = timestamp
        self.deadline = None
        return True

    def cancel_observation(self, *, now: float) -> bool:
        """Cancela apenas OBSERVE; um compromisso nunca pode ser cancelado."""

        if self._fault_if_deadline_expired(now=now):
            return False
        if self.state != GreenManeuverState.OBSERVE or self.event is not None:
            return False
        self.state = GreenManeuverState.FOLLOW
        self.state_since = float(now)
        self.deadline = None
        return True

    def _advance(
        self,
        expected: GreenManeuverState,
        target: GreenManeuverState,
        *,
        now: float,
        timeout_s: Optional[float] = None,
    ) -> bool:
        if self._fault_if_deadline_expired(now=now):
            return False
        if self.state != expected or self.event is None:
            return False
        if target <= expected:
            raise ValueError("transição de manobra precisa ser monotônica")
        self.state = target
        self.state_since = float(now)
        self.deadline = (
            None if timeout_s is None else float(now) + max(float(timeout_s), 0.0)
        )
        return True

    def begin_approach(self, *, now: float, timeout_s: Optional[float] = None) -> bool:
        return self._advance(
            GreenManeuverState.COMMITTED,
            GreenManeuverState.APPROACH,
            now=now,
            timeout_s=timeout_s,
        )

    def begin_turn(self, *, now: float, timeout_s: Optional[float] = None) -> bool:
        return self._advance(
            GreenManeuverState.APPROACH,
            GreenManeuverState.TURNING,
            now=now,
            timeout_s=timeout_s,
        )

    def begin_reacquire(self, *, now: float, timeout_s: Optional[float] = None) -> bool:
        return self._advance(
            GreenManeuverState.TURNING,
            GreenManeuverState.REACQUIRE,
            now=now,
            timeout_s=timeout_s,
        )

    def complete(
        self,
        *,
        now: float,
        timeout_s: Optional[float] = None,
    ) -> int:
        """Entra em COOLDOWN e devolve o decision_id a ser consumido."""

        if not self._advance(
            GreenManeuverState.REACQUIRE,
            GreenManeuverState.COOLDOWN,
            now=now,
            timeout_s=timeout_s,
        ):
            return 0
        return self.decision_id

    def release_cooldown(self, *, now: float) -> bool:
        if self.stopped or self.state != GreenManeuverState.COOLDOWN:
            return False
        self.state = GreenManeuverState.FOLLOW
        self.state_since = float(now)
        self.deadline = None
        self.event = None
        return True

    def check_timeout(self, *, now: float) -> bool:
        """Qualquer timeout armado termina em parada persistente."""

        return self._fault_if_deadline_expired(now=now)

    def fault(self, reason: str, *, now: float) -> None:
        self.state = GreenManeuverState.FAULT_STOP
        self.state_since = float(now)
        self.deadline = None
        self.fault_reason = str(reason) or "fault"

    def manual_reset(self, *, now: float) -> None:
        """Única saída de FAULT_STOP; deve ser chamada por ação externa explícita."""

        self.state = GreenManeuverState.FOLLOW
        self.event = None
        self.state_since = float(now)
        self.deadline = None
        self.fault_reason = ""

    def locked_turn_angle(self, magnitude: float = 180.0) -> float:
        """Comando de tanque cujo sinal não depende de frames posteriores."""

        value = abs(float(magnitude))
        if self.locked_direction == GreenDecision.LEFT:
            return -value
        if self.locked_direction in (GreenDecision.RIGHT, GreenDecision.UTURN):
            return value
        return 0.0


def normalize_yaw_degrees(angle: float) -> float:
    """Normaliza para [-180, 180), preservando diferenças no wrap."""

    angle = float(angle)
    if not math.isfinite(angle):
        return math.nan
    return (angle + 180.0) % 360.0 - 180.0


def signed_yaw_delta(previous: float, current: float) -> float:
    """Delta modular assinado de ``previous`` para ``current``."""

    previous = float(previous)
    current = float(current)
    if not math.isfinite(previous) or not math.isfinite(current):
        return math.nan
    return normalize_yaw_degrees(current - previous)


def yaw_is_fresh(
    angle: Optional[float],
    timestamp: Optional[float],
    *,
    now: float,
    max_age_s: float,
) -> bool:
    if angle is None or timestamp is None:
        return False
    try:
        angle = float(angle)
        timestamp = float(timestamp)
        age = float(now) - timestamp
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(angle)
        and math.isfinite(timestamp)
        and -1e-6 <= age <= max(float(max_age_s), 0.0)
    )


def expected_yaw_sign(
    decision: GreenDecision,
    *,
    positive_is_right: bool = True,
) -> int:
    decision = GreenDecision(decision)
    if decision == GreenDecision.LEFT:
        sign = -1
    elif decision in (GreenDecision.RIGHT, GreenDecision.UTURN):
        sign = 1
    else:
        return 0
    return sign if positive_is_right else -sign


@dataclass(frozen=True, slots=True)
class YawProgress:
    valid: bool
    signed_delta_deg: float = 0.0
    progress_deg: float = 0.0
    wrong_direction: bool = False


class SignedYawTracker:
    """Acumula yaw modular e rejeita leitura velha, ausente ou no sentido oposto."""

    def __init__(
        self,
        decision: GreenDecision,
        *,
        positive_is_right: bool = True,
        max_age_s: float = 0.25,
        wrong_direction_tolerance_deg: float = 3.0,
    ) -> None:
        self.expected_sign = expected_yaw_sign(
            decision,
            positive_is_right=positive_is_right,
        )
        if self.expected_sign == 0:
            raise ValueError("yaw só é aplicável a LEFT, RIGHT ou UTURN")
        self.max_age_s = max(float(max_age_s), 0.0)
        self.wrong_direction_tolerance_deg = max(
            float(wrong_direction_tolerance_deg), 0.0,
        )
        self._last_angle: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._cumulative = 0.0
        self._continuity_lost = False

    @property
    def cumulative_signed_deg(self) -> float:
        return self._cumulative

    def update(
        self,
        angle: Optional[float],
        timestamp: Optional[float],
        *,
        now: float,
    ) -> YawProgress:
        if not yaw_is_fresh(
            angle,
            timestamp,
            now=now,
            max_age_s=self.max_age_s,
        ):
            return YawProgress(valid=False)
        angle = float(angle)
        timestamp = float(timestamp)
        if self._continuity_lost:
            return YawProgress(valid=False)
        if (
            self._last_timestamp is not None
            and timestamp <= self._last_timestamp
        ):
            return YawProgress(valid=False)
        if (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > self.max_age_s
        ):
            # O robô pode ter girado uma quantidade desconhecida durante a
            # lacuna. O MPU perde autoridade até uma nova manobra criar outro
            # tracker; a câmera permanece como fallback seguro.
            self._continuity_lost = True
            return YawProgress(valid=False)
        if self._last_angle is None:
            self._last_angle = angle
            self._last_timestamp = timestamp
            return YawProgress(valid=True)

        step = signed_yaw_delta(self._last_angle, angle)
        if not math.isfinite(step):
            return YawProgress(valid=False)
        self._cumulative += step
        self._last_angle = angle
        self._last_timestamp = timestamp
        directed = self.expected_sign * self._cumulative
        return YawProgress(
            valid=True,
            signed_delta_deg=self._cumulative,
            progress_deg=max(directed, 0.0),
            wrong_direction=(directed < -self.wrong_direction_tolerance_deg),
        )


# Aliases em português para facilitar a migração do código existente.
DecisaoVerde = GreenDecision
EstadoManobraVerde = GreenManeuverState
ObservacaoVerde = GreenObservation
ConfirmadorEventoVerde = GreenDecisionTracker
MaquinaEstadoVerde = GreenManeuverFSM
normalizar_yaw = normalize_yaw_degrees
delta_yaw_assinado = signed_yaw_delta
yaw_fresco = yaw_is_fresh


__all__ = [
    "COMMITTABLE_DECISIONS",
    "TURN_DECISIONS",
    "ConfirmadorEventoVerde",
    "DecisaoVerde",
    "EstadoManobraVerde",
    "GreenDecision",
    "GreenDecisionTracker",
    "GreenManeuverFSM",
    "GreenManeuverState",
    "GreenObservation",
    "GREEN_OBSERVATION_ATOMIC_SIZE",
    "MaquinaEstadoVerde",
    "ObservacaoVerde",
    "SignedYawTracker",
    "YawProgress",
    "delta_yaw_assinado",
    "empty_observation",
    "expected_yaw_sign",
    "normalize_yaw_degrees",
    "normalizar_yaw",
    "signed_yaw_delta",
    "yaw_fresco",
    "yaw_is_fresh",
]
