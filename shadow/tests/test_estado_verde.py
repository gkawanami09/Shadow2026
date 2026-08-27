"""Regressões do evento persistente e da máquina de estados verde."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.estado_verde import (  # noqa: E402
    GreenDecision,
    GreenDecisionTracker,
    GreenManeuverFSM,
    GreenManeuverState,
    GreenObservation,
    SignedYawTracker,
    calibracao_permite_motores,
    signed_yaw_delta,
    yaw_is_fresh,
)


def observation(
    sequence,
    timestamp,
    decision,
    *,
    junction_id=7,
    marker_ids=(101,),
    confidence=.9,
    junction_center=(100., 120.),
    target_branch=(180., 120.),
    ready_to_turn=False,
    junction_visible=True,
    geometry_predicted=False,
    decision_id=0,
):
    return GreenObservation(
        sequence=sequence,
        junction_id=junction_id,
        decision_id=decision_id,
        timestamp=timestamp,
        decision=decision,
        confidence=confidence,
        entry_tangent=(0., 1.),
        junction_center=junction_center,
        target_branch=target_branch,
        ready_to_turn=ready_to_turn,
        junction_visible=junction_visible,
        geometry_predicted=geometry_predicted,
        marker_ids=marker_ids,
    )


def confirm_single(tracker, direction=GreenDecision.RIGHT, *, start=0.):
    result = None
    for index in range(3):
        result = tracker.update(observation(
            index + 1,
            start + index * .02,
            direction,
        ))
    return result


class CalibrationGateTests(unittest.TestCase):
    def test_calibracao_obrigatoria_ausente_bloqueia_motores(self):
        self.assertFalse(calibracao_permite_motores(
            obrigatoria=True, pronta=False))

    def test_calibracao_valida_arma_e_modo_opcional_preserva_diagnostico(self):
        self.assertTrue(calibracao_permite_motores(
            obrigatoria=True, pronta=True))
        self.assertTrue(calibracao_permite_motores(
            obrigatoria=False, pronta=False))


class GreenDecisionTrackerTests(unittest.TestCase):
    def test_contrato_de_coordenadas_possui_aliases_explicitos(self):
        evento = observation(
            1,
            1.,
            GreenDecision.RIGHT,
            junction_center=(220., 104.),
            target_branch=(380., 105.),
        )

        self.assertEqual(evento.entry_tangent_ground, (0., 1.))
        self.assertEqual(evento.junction_rectified, (220., 104.))
        self.assertEqual(evento.target_branch_raw, (380., 105.))

    def test_observacao_tem_roundtrip_atomico_sem_pickle(self):
        original = observation(
            12,
            1.25,
            GreenDecision.UTURN,
            marker_ids=(202, 101),
        )

        restored = GreenObservation.from_atomic_values(original.as_atomic_values())

        self.assertEqual(restored, original)

    def test_um_verde_confirmado_espera_035s_pelo_segundo(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=.35)

        result = confirm_single(tracker)

        self.assertEqual(result.decision, GreenDecision.PENDING)
        self.assertTrue(tracker.waiting_second_marker)
        self.assertAlmostEqual(tracker.second_marker_deadline, .39)
        self.assertEqual(tracker.tick(.38).decision, GreenDecision.PENDING)

        stale = tracker.tick(.40, sequence=20)
        self.assertEqual(stale.decision, GreenDecision.PENDING)
        self.assertFalse(stale.committed)
        self.assertFalse(stale.junction_visible)
        self.assertFalse(stale.geometry_predicted)
        self.assertFalse(stale.ready_to_turn)

        committed = tracker.update(observation(
            21,
            .41,
            GreenDecision.RIGHT,
            junction_center=(105., 210.),
            target_branch=(205., 210.),
            ready_to_turn=True,
        ))
        self.assertEqual(committed.decision, GreenDecision.RIGHT)
        self.assertGreater(committed.decision_id, 0)
        self.assertTrue(committed.junction_visible)
        self.assertFalse(committed.geometry_predicted)
        self.assertTrue(committed.ready_to_turn)
        self.assertEqual(committed.target_branch, (205., 210.))
        self.assertTrue(tracker.new_commit)

    def test_um_unico_frame_nunca_executa_180(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=.35)
        confirm_single(tracker)

        result = tracker.update(observation(
            4,
            .10,
            GreenDecision.UTURN,
            marker_ids=(101, 202),
        ))
        self.assertEqual(result.decision, GreenDecision.PENDING)

        stale = tracker.tick(.40, sequence=20)
        self.assertEqual(stale.decision, GreenDecision.PENDING)
        self.assertFalse(stale.committed)

        committed = tracker.update(observation(
            21, .41, GreenDecision.RIGHT))
        self.assertEqual(committed.decision, GreenDecision.RIGHT)

    def test_dois_verdes_tres_de_cinco_tem_prioridade_e_nao_degradam(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=.35)
        confirm_single(tracker)
        for index in range(3):
            result = tracker.update(observation(
                4 + index,
                .10 + index * .02,
                GreenDecision.UTURN,
                marker_ids=(101, 202),
            ))

        self.assertEqual(result.decision, GreenDecision.UTURN)
        decision_id = result.decision_id

        # Perda de um verde e até direção oposta não podem degradar o retorno.
        for index, direction in enumerate((
            GreenDecision.RIGHT,
            GreenDecision.LEFT,
            GreenDecision.STRAIGHT,
        )):
            later = tracker.update(observation(20 + index, .20 + index * .02, direction))
            self.assertEqual(later.decision, GreenDecision.UTURN)
            self.assertEqual(later.decision_id, decision_id)

    def test_right_confirmado_ignora_frames_left_e_straight(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=.01)
        confirm_single(tracker)
        committed = tracker.tick(.10, sequence=10)

        for sequence, direction in enumerate((
            GreenDecision.LEFT,
            GreenDecision.STRAIGHT,
            GreenDecision.NONE,
        ), start=11):
            result = tracker.update(observation(sequence, sequence / 100., direction))
            self.assertEqual(result.decision, GreenDecision.RIGHT)
            self.assertEqual(result.decision_id, committed.decision_id)

    def test_ready_de_um_frame_nao_fica_or_latched(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=0.)
        committed = confirm_single(tracker)

        ready = tracker.update(observation(
            20,
            .20,
            GreenDecision.RIGHT,
            marker_ids=committed.marker_ids,
            junction_center=(100., 220.),
            target_branch=(200., 220.),
            ready_to_turn=True,
        ))
        not_ready = tracker.update(observation(
            21,
            .22,
            GreenDecision.RIGHT,
            marker_ids=committed.marker_ids,
            junction_center=(100., 180.),
            target_branch=(200., 180.),
            ready_to_turn=False,
        ))

        self.assertTrue(ready.ready_to_turn)
        self.assertFalse(not_ready.ready_to_turn)
        self.assertEqual(not_ready.junction_center, (100., 180.))
        self.assertFalse(not_ready.geometry_predicted)

    def test_conflito_right_left_preserva_alvo_e_so_prediz(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=0., prediction_max_s=.20)
        committed = confirm_single(tracker)
        original_target = committed.target_branch
        original_center = committed.junction_center

        conflict = tracker.update(observation(
            20,
            .20,
            GreenDecision.LEFT,
            marker_ids=(999,),
            junction_center=(20., 220.),
            target_branch=(20., 20.),
            ready_to_turn=True,
        ))

        self.assertEqual(conflict.decision, GreenDecision.RIGHT)
        self.assertEqual(conflict.decision_id, committed.decision_id)
        self.assertEqual(conflict.target_branch, original_target)
        self.assertEqual(conflict.junction_center, original_center)
        self.assertFalse(conflict.junction_visible)
        self.assertTrue(conflict.geometry_predicted)
        self.assertFalse(conflict.ready_to_turn)

    def test_mesma_decisao_com_marker_mismatch_nao_troca_geometria(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=0., prediction_max_s=.20)
        committed = confirm_single(tracker)

        mismatch = tracker.update(observation(
            20,
            .20,
            GreenDecision.RIGHT,
            marker_ids=(202,),
            junction_center=(300., 220.),
            target_branch=(400., 220.),
            ready_to_turn=True,
        ))

        self.assertEqual(mismatch.marker_ids, committed.marker_ids)
        self.assertEqual(mismatch.junction_center, committed.junction_center)
        self.assertEqual(mismatch.target_branch, committed.target_branch)
        self.assertTrue(mismatch.geometry_predicted)
        self.assertFalse(mismatch.ready_to_turn)

    def test_straight_aceita_sumiço_do_marker_sem_trocar_ids(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=0.)
        committed = None
        for index in range(3):
            committed = tracker.update(observation(
                index + 1,
                index * .02,
                GreenDecision.STRAIGHT,
                marker_ids=(303,),
            ))

        updated = tracker.update(observation(
            10,
            .10,
            GreenDecision.STRAIGHT,
            marker_ids=(),
            junction_center=(110., 190.),
            target_branch=(110., 20.),
            ready_to_turn=True,
        ))

        self.assertEqual(updated.decision, GreenDecision.STRAIGHT)
        self.assertEqual(updated.marker_ids, (303,))
        self.assertEqual(updated.junction_center, (110., 190.))
        self.assertEqual(updated.target_branch, (110., 20.))
        self.assertTrue(updated.ready_to_turn)
        self.assertFalse(updated.geometry_predicted)

    def test_straight_sem_marker_aceita_post_sem_trocar_ids(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=0.)
        committed = None
        for index in range(3):
            committed = tracker.update(observation(
                index + 1,
                index * .02,
                GreenDecision.STRAIGHT,
                marker_ids=(),
            ))

        self.assertEqual(committed.marker_ids, ())
        updated = tracker.update(observation(
            10,
            .10,
            GreenDecision.STRAIGHT,
            marker_ids=(303,),
            junction_center=(110., 190.),
            target_branch=(110., 20.),
            ready_to_turn=True,
        ))

        self.assertEqual(updated.decision, GreenDecision.STRAIGHT)
        self.assertEqual(updated.marker_ids, ())
        self.assertEqual(updated.junction_center, (110., 190.))
        self.assertEqual(updated.target_branch, (110., 20.))
        self.assertTrue(updated.junction_visible)
        self.assertTrue(updated.ready_to_turn)

    def test_frames_previstos_ou_invisiveis_nao_votam_no_tres_de_cinco(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=0.)
        frames = (
            observation(1, .01, GreenDecision.RIGHT),
            observation(
                2, .02, GreenDecision.RIGHT,
                junction_visible=False, geometry_predicted=True),
            observation(3, .03, GreenDecision.RIGHT),
            observation(
                4, .04, GreenDecision.RIGHT,
                junction_visible=False, geometry_predicted=False),
            observation(
                5, .05, GreenDecision.RIGHT,
                junction_visible=False, geometry_predicted=True),
            observation(6, .06, GreenDecision.RIGHT),
        )
        for frame in frames:
            result = tracker.update(frame)
            self.assertFalse(result.committed)

        result = tracker.update(observation(
            7, .07, GreenDecision.RIGHT))
        self.assertTrue(result.committed)
        self.assertEqual(result.decision, GreenDecision.RIGHT)

    def test_candidato_unico_nao_renova_com_direcao_ou_marker_diferente(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=.35, prediction_max_s=.20)
        pending = confirm_single(tracker)
        original_center = pending.junction_center
        original_target = pending.target_branch

        marker_mismatch = tracker.update(observation(
            10,
            .20,
            GreenDecision.RIGHT,
            marker_ids=(202,),
            junction_center=(300., 220.),
            target_branch=(400., 220.),
            ready_to_turn=True,
        ))
        direction_mismatch = tracker.update(observation(
            11,
            .40,
            GreenDecision.LEFT,
            marker_ids=(101,),
            junction_center=(20., 220.),
            target_branch=(20., 20.),
            ready_to_turn=True,
        ))

        self.assertEqual(marker_mismatch.target_branch, original_target)
        self.assertEqual(direction_mismatch.decision, GreenDecision.PENDING)
        self.assertFalse(direction_mismatch.committed)
        self.assertEqual(direction_mismatch.junction_center, original_center)
        self.assertEqual(direction_mismatch.target_branch, original_target)
        self.assertFalse(direction_mismatch.junction_visible)
        self.assertFalse(direction_mismatch.geometry_predicted)
        self.assertFalse(direction_mismatch.ready_to_turn)

    def test_ready_do_candidato_nao_congela_apos_frame_false(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=.35, prediction_max_s=.20)
        for index in range(3):
            tracker.update(observation(
                index + 1,
                index * .02,
                GreenDecision.RIGHT,
                ready_to_turn=(index == 2),
            ))

        current = tracker.update(observation(
            10,
            .25,
            GreenDecision.RIGHT,
            ready_to_turn=False,
            junction_center=(100., 180.),
        ))
        committed = tracker.tick(.40, sequence=11)

        self.assertEqual(current.decision, GreenDecision.PENDING)
        self.assertFalse(current.ready_to_turn)
        self.assertTrue(committed.committed)
        self.assertTrue(committed.geometry_predicted)
        self.assertFalse(committed.junction_visible)
        self.assertFalse(committed.ready_to_turn)
        self.assertEqual(committed.junction_center, (100., 180.))

    def test_candidato_ready_stale_fica_pending_e_nao_executavel(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=.35, prediction_max_s=.20)
        for index in range(3):
            tracker.update(observation(
                index + 1,
                index * .02,
                GreenDecision.RIGHT,
                ready_to_turn=(index == 2),
            ))

        stale = tracker.tick(.40, sequence=10)

        self.assertEqual(stale.decision, GreenDecision.PENDING)
        self.assertFalse(stale.committed)
        self.assertFalse(stale.junction_visible)
        self.assertFalse(stale.geometry_predicted)
        self.assertFalse(stale.ready_to_turn)
        self.assertIsNone(tracker.committed)

    def test_commit_previsto_nao_reinicia_validade_da_geometria(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=.15, prediction_max_s=.20)
        confirm_single(tracker)

        predicted = tracker.tick(.19, sequence=10)
        expired = tracker.update(observation(
            11,
            .25,
            GreenDecision.NONE,
            junction_id=0,
            marker_ids=(),
            junction_visible=False,
        ))

        self.assertTrue(predicted.committed)
        self.assertTrue(predicted.geometry_predicted)
        self.assertFalse(predicted.junction_visible)
        self.assertEqual(expired.decision, GreenDecision.RIGHT)
        self.assertFalse(expired.geometry_predicted)
        self.assertFalse(expired.ready_to_turn)

    def test_perda_geometrica_so_e_predita_por_020s(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=0., prediction_max_s=.20)
        committed = confirm_single(tracker)
        committed = tracker.update(observation(
            9,
            .08,
            GreenDecision.RIGHT,
            marker_ids=committed.marker_ids,
            junction_center=(100., 220.),
            target_branch=(200., 220.),
            ready_to_turn=True,
        ))
        ausente_curto = GreenObservation(
            sequence=10,
            junction_id=0,
            decision_id=0,
            timestamp=.20,
            decision=GreenDecision.NONE,
            confidence=0.,
        )
        curto = tracker.update(ausente_curto)
        longo = tracker.update(GreenObservation(
            sequence=11,
            junction_id=0,
            decision_id=0,
            timestamp=.30,
            decision=GreenDecision.NONE,
            confidence=0.,
        ))

        self.assertEqual(curto.decision, committed.decision)
        self.assertTrue(curto.geometry_predicted)
        self.assertFalse(curto.junction_visible)
        self.assertTrue(curto.ready_to_turn)
        self.assertEqual(curto.target_branch, (200., 220.))
        self.assertEqual(longo.decision, committed.decision)
        self.assertFalse(longo.geometry_predicted)
        self.assertFalse(longo.ready_to_turn)

    def test_confirmacao_exige_mesma_juncao_e_mesmos_marcadores(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=0.)
        frames = (
            observation(1, .01, GreenDecision.RIGHT, marker_ids=(1,)),
            observation(2, .02, GreenDecision.RIGHT, marker_ids=(2,)),
            observation(3, .03, GreenDecision.RIGHT, marker_ids=(1,)),
            observation(4, .04, GreenDecision.RIGHT, marker_ids=(2,)),
        )
        for frame in frames:
            result = tracker.update(frame)
            self.assertFalse(result.committed)

        result = tracker.update(observation(
            5, .05, GreenDecision.RIGHT, marker_ids=(1,)))
        self.assertTrue(result.committed)

        other = GreenDecisionTracker(second_marker_wait_s=0.)
        other.update(observation(1, .01, GreenDecision.RIGHT, junction_id=1))
        other.update(observation(2, .02, GreenDecision.RIGHT, junction_id=2))
        result = other.update(observation(3, .03, GreenDecision.RIGHT, junction_id=1))
        self.assertFalse(result.committed)

    def test_decision_id_executa_exatamente_uma_vez_e_exige_rearme(self):
        tracker = GreenDecisionTracker(second_marker_wait_s=0., rearm_frames=3)
        committed = confirm_single(tracker)
        decision_id = committed.decision_id

        self.assertTrue(tracker.consume(decision_id))
        self.assertFalse(tracker.consume(decision_id))
        self.assertTrue(tracker.in_cooldown)

        # A mesma junção continua visível: o evento não rearma nem repete.
        ignored = tracker.update(observation(10, .2, GreenDecision.RIGHT))
        self.assertEqual(ignored.decision, GreenDecision.NONE)
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=True, exit_line_stable=True))
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=False, exit_line_stable=True))
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=False, exit_line_stable=True))
        self.assertTrue(tracker.note_rearm_frame(
            junction_visible=False, exit_line_stable=True))

        new = confirm_single(tracker, start=.5)
        self.assertGreater(new.decision_id, decision_id)

    def test_rearme_exige_tempo_minimo_mais_frames_limpos(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=0.,
            rearm_frames=2,
            rearm_min_s=.50,
        )
        committed = confirm_single(tracker, start=1.)
        self.assertTrue(tracker.consume(
            committed.decision_id, timestamp=2.))

        # Frames limpos antes do tempo minimo nao sao acumulados.
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=False,
            exit_line_stable=True,
            timestamp=2.20,
        ))
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=False,
            exit_line_stable=True,
            timestamp=2.49,
        ))
        # O frame no limite e o primeiro voto; ainda falta o segundo.
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=False,
            exit_line_stable=True,
            timestamp=2.50,
        ))
        self.assertTrue(tracker.note_rearm_frame(
            junction_visible=False,
            exit_line_stable=True,
            timestamp=2.52,
        ))
        self.assertFalse(tracker.in_cooldown)

    def test_rearme_temporal_sem_timestamp_falha_fechado(self):
        tracker = GreenDecisionTracker(
            second_marker_wait_s=0.,
            rearm_frames=1,
            rearm_min_s=.25,
        )
        committed = confirm_single(tracker)
        self.assertTrue(tracker.consume(
            committed.decision_id, timestamp=1.))
        self.assertFalse(tracker.note_rearm_frame(
            junction_visible=False,
            exit_line_stable=True,
        ))
        self.assertTrue(tracker.in_cooldown)

        no_consume_clock = GreenDecisionTracker(
            second_marker_wait_s=0.,
            rearm_frames=1,
            rearm_min_s=.25,
        )
        committed = confirm_single(no_consume_clock)
        self.assertTrue(no_consume_clock.consume(committed.decision_id))
        self.assertFalse(no_consume_clock.note_rearm_frame(
            junction_visible=False,
            exit_line_stable=True,
            timestamp=10.,
        ))
        self.assertTrue(no_consume_clock.in_cooldown)


class GreenManeuverFSMTests(unittest.TestCase):
    def _committed(self, direction=GreenDecision.RIGHT, decision_id=5):
        return GreenObservation(
            sequence=10,
            junction_id=7,
            decision_id=decision_id,
            timestamp=1.,
            decision=direction,
            confidence=.9,
            marker_ids=(101,),
        )

    def test_estados_avancam_na_ordem_e_comando_mantem_sinal(self):
        fsm = GreenManeuverFSM()
        self.assertTrue(fsm.observe(self._committed(), now=1.))
        self.assertEqual(fsm.state, GreenManeuverState.COMMITTED)
        self.assertEqual(fsm.locked_turn_angle(), 180.)

        self.assertTrue(fsm.begin_approach(now=1.1, timeout_s=1.))
        self.assertTrue(fsm.begin_turn(now=1.2, timeout_s=1.))
        self.assertTrue(fsm.begin_reacquire(now=1.3, timeout_s=1.))
        self.assertEqual(fsm.complete(now=1.4), 5)
        self.assertEqual(fsm.state, GreenManeuverState.COOLDOWN)
        self.assertTrue(fsm.release_cooldown(now=2.))
        self.assertEqual(fsm.state, GreenManeuverState.FOLLOW)

    def test_cooldown_sem_ack_expira_em_fault_stop(self):
        fsm = GreenManeuverFSM()
        self.assertTrue(fsm.observe(self._committed(), now=1.))
        self.assertTrue(fsm.begin_approach(now=1.1, timeout_s=1.))
        self.assertTrue(fsm.begin_turn(now=1.2, timeout_s=1.))
        self.assertTrue(fsm.begin_reacquire(now=1.3, timeout_s=1.))
        self.assertEqual(fsm.complete(now=1.4, timeout_s=.5), 5)

        self.assertFalse(fsm.check_timeout(now=1.899))
        self.assertTrue(fsm.check_timeout(now=1.9))
        self.assertEqual(fsm.state, GreenManeuverState.FAULT_STOP)
        self.assertEqual(fsm.fault_reason, "timeout")

    def test_evento_oposto_nao_inverte_compromisso(self):
        fsm = GreenManeuverFSM()
        right = self._committed(GreenDecision.RIGHT, 5)
        left = self._committed(GreenDecision.LEFT, 5)
        other = self._committed(GreenDecision.LEFT, 6)

        self.assertTrue(fsm.observe(right))
        self.assertFalse(fsm.observe(left))
        self.assertFalse(fsm.observe(other))
        self.assertEqual(fsm.locked_direction, GreenDecision.RIGHT)
        self.assertEqual(fsm.locked_turn_angle(), 180.)

    def test_fsm_reflete_ready_atual_sem_or_latch(self):
        fsm = GreenManeuverFSM()
        ready = observation(
            10,
            1.,
            GreenDecision.RIGHT,
            decision_id=5,
            ready_to_turn=True,
        )
        not_ready = observation(
            11,
            1.02,
            GreenDecision.RIGHT,
            decision_id=5,
            ready_to_turn=False,
            junction_center=(100., 180.),
        )

        self.assertTrue(fsm.observe(ready))
        self.assertTrue(fsm.event.ready_to_turn)
        self.assertTrue(fsm.observe(not_ready))
        self.assertFalse(fsm.event.ready_to_turn)
        self.assertEqual(fsm.event.junction_center, (100., 180.))

    def test_fsm_rejeita_marker_mismatch_do_mesmo_decision_id(self):
        fsm = GreenManeuverFSM()
        committed = observation(
            10,
            1.,
            GreenDecision.RIGHT,
            decision_id=5,
            marker_ids=(101,),
            target_branch=(180., 120.),
        )
        mismatch = observation(
            11,
            1.02,
            GreenDecision.RIGHT,
            decision_id=5,
            marker_ids=(202,),
            target_branch=(20., 120.),
            ready_to_turn=True,
        )

        self.assertTrue(fsm.observe(committed))
        self.assertFalse(fsm.observe(mismatch))
        self.assertEqual(fsm.event.target_branch, (180., 120.))
        self.assertFalse(fsm.event.ready_to_turn)

    def test_esquerda_e_retorno_usam_sinais_obrigatorios(self):
        left = GreenManeuverFSM()
        left.observe(self._committed(GreenDecision.LEFT))
        self.assertEqual(left.locked_turn_angle(), -180.)

        uturn = GreenManeuverFSM()
        uturn.observe(self._committed(GreenDecision.UTURN))
        self.assertEqual(uturn.locked_turn_angle(), 180.)

    def test_timeout_entra_em_fault_stop_e_permanece_parado(self):
        fsm = GreenManeuverFSM()
        fsm.observe(self._committed(), now=1.)
        fsm.begin_approach(now=1., timeout_s=.5)

        self.assertTrue(fsm.check_timeout(now=1.5))
        self.assertEqual(fsm.state, GreenManeuverState.FAULT_STOP)
        self.assertTrue(fsm.check_timeout(now=10.))
        self.assertFalse(fsm.observe(self._committed(decision_id=6), now=10.))

        fsm.manual_reset(now=11.)
        self.assertEqual(fsm.state, GreenManeuverState.FOLLOW)

    def test_commit_nao_pode_pular_deadline_vencido_de_observe(self):
        fsm = GreenManeuverFSM()
        pending = observation(
            1, 1., GreenDecision.PENDING, decision_id=0)
        self.assertTrue(fsm.observe(
            pending, now=1., observe_timeout_s=.50))

        accepted = fsm.observe(
            self._committed(),
            now=1.50,
        )

        self.assertFalse(accepted)
        self.assertEqual(fsm.state, GreenManeuverState.FAULT_STOP)
        self.assertEqual(fsm.fault_reason, "timeout")
        self.assertIsNone(fsm.event)

    def test_transicao_nao_pode_substituir_deadline_vencido(self):
        fsm = GreenManeuverFSM()
        self.assertTrue(fsm.observe(self._committed(), now=1.))
        self.assertTrue(fsm.begin_approach(now=1., timeout_s=.50))

        accepted = fsm.begin_turn(now=1.50, timeout_s=1.)

        self.assertFalse(accepted)
        self.assertEqual(fsm.state, GreenManeuverState.FAULT_STOP)
        self.assertEqual(fsm.fault_reason, "timeout")

    def test_cancelamento_tardio_de_observe_tambem_falha_fechado(self):
        fsm = GreenManeuverFSM()
        pending = observation(
            1, 1., GreenDecision.PENDING, decision_id=0)
        fsm.observe(pending, now=1., observe_timeout_s=.50)

        self.assertFalse(fsm.cancel_observation(now=1.50))
        self.assertEqual(fsm.state, GreenManeuverState.FAULT_STOP)

    def test_pending_persistente_tambem_falha_fechado(self):
        fsm = GreenManeuverFSM()
        pending = observation(
            1, 1., GreenDecision.PENDING, decision_id=0)
        self.assertTrue(fsm.observe(
            pending, now=1., observe_timeout_s=.5))
        self.assertEqual(fsm.state, GreenManeuverState.OBSERVE)
        self.assertFalse(fsm.check_timeout(now=1.49))
        self.assertTrue(fsm.check_timeout(now=1.5))
        self.assertEqual(fsm.state, GreenManeuverState.FAULT_STOP)

    def test_pending_que_some_cancela_e_limpa_deadline(self):
        fsm = GreenManeuverFSM()
        pending = observation(
            1, 1., GreenDecision.PENDING, decision_id=0)
        fsm.observe(pending, now=1., observe_timeout_s=.5)
        self.assertTrue(fsm.cancel_observation(now=1.1))
        self.assertIsNone(fsm.deadline)
        self.assertFalse(fsm.check_timeout(now=10.))

    def test_nao_pula_estado_da_sequencia(self):
        fsm = GreenManeuverFSM()
        fsm.observe(self._committed())
        self.assertFalse(fsm.begin_turn(now=1.))
        self.assertEqual(fsm.state, GreenManeuverState.COMMITTED)


class SignedYawTests(unittest.TestCase):
    def test_delta_modular_respeita_wrap_e_sinal(self):
        self.assertEqual(signed_yaw_delta(350., 10.), 20.)
        self.assertEqual(signed_yaw_delta(10., 350.), -20.)

    def test_yaw_direita_acumula_ao_cruzar_wrap(self):
        tracker = SignedYawTracker(GreenDecision.RIGHT, max_age_s=.2)
        tracker.update(350., 1., now=1.)
        first = tracker.update(10., 1.05, now=1.05)
        second = tracker.update(45., 1.10, now=1.10)

        self.assertTrue(first.valid)
        self.assertEqual(first.progress_deg, 20.)
        self.assertEqual(second.progress_deg, 55.)
        self.assertFalse(second.wrong_direction)

    def test_yaw_esquerda_e_sentido_oposto(self):
        left = SignedYawTracker(GreenDecision.LEFT)
        left.update(10., 1., now=1.)
        result = left.update(350., 1.1, now=1.1)
        self.assertEqual(result.progress_deg, 20.)
        self.assertFalse(result.wrong_direction)

        wrong = SignedYawTracker(GreenDecision.RIGHT)
        wrong.update(10., 1., now=1.)
        result = wrong.update(350., 1.1, now=1.1)
        self.assertEqual(result.progress_deg, 0.)
        self.assertTrue(result.wrong_direction)

    def test_polaridade_fisica_do_yaw_e_configuravel(self):
        tracker = SignedYawTracker(
            GreenDecision.RIGHT,
            positive_is_right=False,
        )
        tracker.update(10., 1., now=1.)
        result = tracker.update(350., 1.1, now=1.1)

        self.assertEqual(result.progress_deg, 20.)
        self.assertFalse(result.wrong_direction)

    def test_yaw_ausente_velho_ou_repetido_e_invalido(self):
        self.assertFalse(yaw_is_fresh(None, 1., now=1., max_age_s=.2))
        self.assertFalse(yaw_is_fresh(10., .5, now=1., max_age_s=.2))

        tracker = SignedYawTracker(GreenDecision.RIGHT, max_age_s=.2)
        self.assertFalse(tracker.update(None, 1., now=1.).valid)
        self.assertFalse(tracker.update(10., .5, now=1.).valid)
        self.assertTrue(tracker.update(10., 1., now=1.).valid)
        self.assertFalse(tracker.update(20., 1., now=1.).valid)

    def test_lacuna_de_amostras_remove_autoridade_ate_nova_manobra(self):
        tracker = SignedYawTracker(GreenDecision.RIGHT, max_age_s=.2)
        self.assertTrue(tracker.update(0., 1., now=1.).valid)
        self.assertFalse(tracker.update(80., 1.25, now=1.25).valid)
        self.assertFalse(tracker.update(90., 1.30, now=1.30).valid)


if __name__ == "__main__":
    unittest.main()
