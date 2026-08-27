"""Regressoes da topologia verde no referencial da linha de entrada."""

import sys
from pathlib import Path
import time
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.intersecao_verde import (  # noqa: E402
    BranchKind,
    GreenDecision,
    GreenTopologyTracker,
    MarkerObservation,
    MarkerPhase,
    PathSide,
    TopologyObservation,
    analyze_green_intersection,
    draw_topology_debug,
)


SIZE = 420
CENTER = (210, 210)
ENTRY = (210.0, 400.0)


def _cross_scene(*, right_pre=False, left_pre=False,
                 right_post=False, left_post=False):
    black = np.zeros((SIZE, SIZE), dtype=np.uint8)
    green = np.zeros_like(black)
    cv2.rectangle(black, (199, 20), (221, 400), 255, -1)
    cv2.rectangle(black, (20, 199), (400, 221), 255, -1)
    if right_pre:
        cv2.rectangle(green, (223, 225), (243, 245), 255, -1)
    if left_pre:
        cv2.rectangle(green, (177, 225), (197, 245), 255, -1)
    if right_post:
        cv2.rectangle(green, (223, 175), (243, 195), 255, -1)
    if left_post:
        cv2.rectangle(green, (177, 175), (197, 195), 255, -1)
    black[green > 0] = 0
    return black, green


def _rotate_scene(black, green, angle):
    matrix = cv2.getRotationMatrix2D(CENTER, angle, 1.0)
    rotated_black = cv2.warpAffine(
        black, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    rotated_green = cv2.warpAffine(
        green, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    entry = cv2.transform(
        np.array([[[ENTRY[0], ENTRY[1]]]], dtype=np.float32), matrix,
    )[0, 0]
    # A tangente usa X-direita/Y-frente, por isso inverte o delta vertical.
    tangent = np.array(
        (CENTER[0] - entry[0], -(CENTER[1] - entry[1])),
        dtype=np.float64,
    )
    tangent /= np.linalg.norm(tangent)
    return rotated_black, rotated_green, tuple(entry), tangent


class TopologiaVerdeOrientadaTests(unittest.TestCase):
    def test_debug_mostra_marcador_mesmo_sem_juncao_localizada(self):
        marker = MarkerObservation(
            center_image=(100.0, 80.0),
            center_ground=(0.0, 0.0),
            side_length=20.0,
            phase=MarkerPhase.AMBIGUOUS,
            side=PathSide.UNKNOWN,
            plausible=True,
            associated=False,
            black_to_junction=False,
            black_inward=False,
            clear_outward=True,
            clear_behind=True,
            valid=False,
            confidence=.4,
        )
        observation = TopologyObservation(
            decision=GreenDecision.PENDING,
            junction_image=None,
            markers=(marker,),
        )
        image = np.zeros((120, 180, 3), dtype=np.uint8)

        returned = draw_topology_debug(image, observation)

        self.assertIs(returned, image)
        self.assertGreater(int(np.count_nonzero(image)), 0)

    def test_direita_permanece_direita_em_todas_as_rotacoes(self):
        black, green = _cross_scene(right_pre=True)
        for angle in range(0, 360, 5):
            with self.subTest(angle=angle):
                b_rot, g_rot, entry, tangent = _rotate_scene(
                    black, green, angle)
                observation = analyze_green_intersection(
                    b_rot,
                    g_rot,
                    entry_point=entry,
                    entry_tangent=tangent,
                )
                self.assertEqual(observation.decision, GreenDecision.RIGHT)
                self.assertEqual(observation.markers[0].phase, MarkerPhase.PRE)
                self.assertEqual(observation.markers[0].side, PathSide.RIGHT)

    def test_espelho_troca_exatamente_direita_por_esquerda(self):
        black, green = _cross_scene(right_pre=True)
        right = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        mirrored_black = cv2.flip(black, 1)
        mirrored_green = cv2.flip(green, 1)
        left = analyze_green_intersection(
            mirrored_black,
            mirrored_green,
            entry_point=(SIZE - 1 - ENTRY[0], ENTRY[1]),
            entry_tangent=(0.0, 1.0),
        )
        self.assertEqual(right.decision, GreenDecision.RIGHT)
        self.assertEqual(left.decision, GreenDecision.LEFT)

    def test_todas_as_decisoes_sao_invariantes_de_0_a_355_graus(self):
        cases = (
            ("esquerda", {"left_pre": True}, GreenDecision.LEFT),
            ("retorno", {
                "left_pre": True,
                "right_pre": True,
            }, GreenDecision.UTURN),
            ("posterior", {"right_post": True}, GreenDecision.STRAIGHT),
            ("sem_verde", {}, GreenDecision.STRAIGHT),
        )
        for name, scene, expected in cases:
            black, green = _cross_scene(**scene)
            for angle in range(0, 360, 5):
                with self.subTest(scene=name, angle=angle):
                    b_rot, g_rot, entry, tangent = _rotate_scene(
                        black, green, angle)
                    observation = analyze_green_intersection(
                        b_rot,
                        g_rot,
                        entry_point=entry,
                        entry_tangent=tangent,
                    )
                    self.assertEqual(observation.decision, expected)

    def test_espelho_preserva_reto_e_180_e_troca_so_lados(self):
        cases = (
            ({"right_pre": True}, GreenDecision.RIGHT, GreenDecision.LEFT),
            ({"left_pre": True}, GreenDecision.LEFT, GreenDecision.RIGHT),
            ({
                "left_pre": True,
                "right_pre": True,
            }, GreenDecision.UTURN, GreenDecision.UTURN),
            ({"right_post": True}, GreenDecision.STRAIGHT,
             GreenDecision.STRAIGHT),
            ({}, GreenDecision.STRAIGHT, GreenDecision.STRAIGHT),
        )
        for scene, expected, mirrored_expected in cases:
            black, green = _cross_scene(**scene)
            original = analyze_green_intersection(
                black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
            mirrored = analyze_green_intersection(
                cv2.flip(black, 1),
                cv2.flip(green, 1),
                entry_point=(SIZE - 1 - ENTRY[0], ENTRY[1]),
                entry_tangent=(0.0, 1.0),
            )
            with self.subTest(scene=scene):
                self.assertEqual(original.decision, expected)
                self.assertEqual(mirrored.decision, mirrored_expected)

    def test_marcador_depois_da_juncao_manda_reto_mesmo_diagonal(self):
        black, green = _cross_scene(right_post=True)
        for angle in (-40, -25, 0, 25, 40):
            with self.subTest(angle=angle):
                b_rot, g_rot, entry, tangent = _rotate_scene(
                    black, green, angle)
                observation = analyze_green_intersection(
                    b_rot, g_rot, entry_point=entry, entry_tangent=tangent)
                self.assertEqual(observation.decision, GreenDecision.STRAIGHT)
                self.assertTrue(observation.post_markers)

    def test_um_pre_e_um_post_nunca_formam_180(self):
        black, green = _cross_scene(right_pre=True, left_post=True)
        for angle in (-40, -20, 0, 20, 40):
            b_rot, g_rot, entry, tangent = _rotate_scene(
                black, green, angle)
            observation = analyze_green_intersection(
                b_rot, g_rot, entry_point=entry, entry_tangent=tangent)
            with self.subTest(angle=angle):
                self.assertEqual(observation.decision, GreenDecision.RIGHT)
                self.assertNotEqual(observation.decision, GreenDecision.UTURN)

    def test_dois_pre_opostos_formam_180(self):
        black, green = _cross_scene(right_pre=True, left_pre=True)
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.UTURN)
        self.assertEqual(len([m for m in observation.pre_markers if m.valid]), 2)
        self.assertIsNotNone(observation.target_branch)
        self.assertEqual(observation.target_branch.kind, BranchKind.INCOMING)

    def test_tracker_da_token_ao_incoming_do_retorno(self):
        tracker = GreenTopologyTracker()
        black, green = _cross_scene(right_pre=True, left_pre=True)

        observation = tracker.update(
            black,
            green,
            entry_point=ENTRY,
            entry_tangent=(0.0, 1.0),
        )

        self.assertEqual(observation.decision, GreenDecision.UTURN)
        self.assertGreater(observation.target_branch.branch_token, 0)
        self.assertEqual(
            observation.target_branch.kind,
            BranchKind.INCOMING,
        )

    def test_intersecao_sem_verde_trava_ramo_reto(self):
        black, green = _cross_scene()
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.STRAIGHT)
        self.assertIsNotNone(observation.target_branch)
        self.assertEqual(observation.target_branch.kind, BranchKind.STRAIGHT)

    def test_t_sem_ramo_reto_permanece_pending(self):
        black = np.zeros((SIZE, SIZE), dtype=np.uint8)
        green = np.zeros_like(black)
        cv2.rectangle(black, (199, 199), (221, 400), 255, -1)
        cv2.rectangle(black, (20, 199), (400, 221), 255, -1)
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.PENDING)
        self.assertIsNone(observation.target_branch)

    def test_post_sem_ramo_reto_nao_inventa_straight(self):
        black = np.zeros((SIZE, SIZE), dtype=np.uint8)
        green = np.zeros_like(black)
        cv2.rectangle(black, (199, 199), (221, 400), 255, -1)
        cv2.rectangle(black, (20, 199), (400, 221), 255, -1)
        cv2.rectangle(green, (223, 175), (243, 195), 255, -1)
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.PENDING)
        self.assertIsNone(observation.target_branch)

    def test_pre_falso_com_reta_completa_trava_reto(self):
        black = np.zeros((SIZE, SIZE), dtype=np.uint8)
        green = np.zeros_like(black)
        cv2.rectangle(black, (199, 20), (221, 400), 255, -1)
        cv2.rectangle(black, (20, 199), (221, 221), 255, -1)  # so esquerda
        cv2.rectangle(green, (223, 225), (243, 245), 255, -1)  # pede direita
        black[green > 0] = 0
        for angle in (-40, -20, 0, 20, 40):
            b_rot, g_rot, entry, tangent = _rotate_scene(
                black, green, angle)
            observation = analyze_green_intersection(
                b_rot, g_rot, entry_point=entry, entry_tangent=tangent)
            with self.subTest(angle=angle):
                self.assertEqual(observation.decision, GreenDecision.STRAIGHT)
                self.assertIsNotNone(observation.target_branch)
                self.assertEqual(
                    observation.target_branch.kind,
                    BranchKind.STRAIGHT,
                )
                self.assertIn(
                    "ramo indicado ausente",
                    observation.markers[0].reason,
                )

    def test_dropout_de_um_frame_preserva_id_do_marcador(self):
        tracker = GreenTopologyTracker()
        black, green = _cross_scene(right_pre=True)
        black_without_marker, no_green = _cross_scene()
        first = tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        second = tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        missing = tracker.update(
            black_without_marker,
            no_green,
            entry_point=ENTRY,
            entry_tangent=(0.0, 1.0),
        )
        recovered = tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(first.marker_ids, second.marker_ids)
        self.assertEqual(missing.marker_ids, ())
        self.assertEqual(first.marker_ids, recovered.marker_ids)
        votes = [
            observation.marker_ids
            for observation in (first, second, missing, recovered)
        ]
        self.assertEqual(votes.count(first.marker_ids), 3)

    def test_componente_distante_da_base_nao_autoriza_curva(self):
        height, width = 252, 448
        black = np.zeros((height, width), dtype=np.uint8)
        green = np.zeros_like(black)
        cv2.rectangle(black, (215, 20), (233, 112), 255, -1)
        cv2.rectangle(black, (100, 71), (350, 89), 255, -1)
        cv2.rectangle(green, (235, 94), (253, 112), 255, -1)
        black[green > 0] = 0
        observation = analyze_green_intersection(
            black, green, entry_point=(224.0, 251.0))
        self.assertNotEqual(observation.decision, GreenDecision.RIGHT)
        self.assertIsNone(observation.entry_image)
        self.assertIsNone(observation.junction_image)

    def test_tracker_propaga_entrada_por_somente_dois_frames(self):
        tracker = GreenTopologyTracker()
        black, green = _cross_scene(right_pre=True)
        initial = tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        truncated = black.copy()
        truncated[321:, :] = 0
        propagated_1 = tracker.update(
            truncated, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        propagated_2 = tracker.update(
            truncated, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        expired = tracker.update(
            truncated, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertFalse(initial.entry_propagated)
        self.assertTrue(propagated_1.entry_propagated)
        self.assertTrue(propagated_2.entry_propagated)
        self.assertFalse(expired.entry_propagated)
        self.assertIsNone(expired.junction_image)

    def test_frame_pending_nao_contamina_historico_de_tangente(self):
        tracker = GreenTopologyTracker()
        black, green = _cross_scene(right_pre=True)
        tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        stable_before = tracker.stable_tangent

        ambiguous_black, ambiguous_green = _cross_scene()
        cv2.rectangle(ambiguous_green, (223, 200), (243, 220), 255, -1)
        ambiguous_black[ambiguous_green > 0] = 0
        rotated_black, rotated_green, entry, tangent = _rotate_scene(
            ambiguous_black, ambiguous_green, 50)
        pending = tracker.update(
            rotated_black,
            rotated_green,
            entry_point=entry,
            entry_tangent=tangent,
        )
        self.assertEqual(pending.decision, GreenDecision.PENDING)
        self.assertEqual(tracker.stable_tangent, stable_before)

    def test_ids_de_juncao_e_marcador_sao_estaveis_entre_frames(self):
        tracker = GreenTopologyTracker()
        black, green = _cross_scene(right_pre=True)
        first = tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        matrix = np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]])
        shifted_black = cv2.warpAffine(
            black, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        shifted_green = cv2.warpAffine(
            green, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        second = tracker.update(
            shifted_black,
            shifted_green,
            entry_point=(ENTRY[0] + 3.0, ENTRY[1] + 2.0),
            entry_tangent=(0.0, 1.0),
        )
        self.assertGreater(first.junction_id, 0)
        self.assertEqual(first.junction_id, second.junction_id)
        self.assertEqual(first.marker_ids, second.marker_ids)
        self.assertEqual(len(first.marker_ids), 1)

    def test_duas_identidades_de_marcador_sao_distintas(self):
        tracker = GreenTopologyTracker()
        black, green = _cross_scene(right_pre=True, left_pre=True)
        observation = tracker.update(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(len(observation.marker_ids), 2)
        self.assertNotEqual(observation.marker_ids[0], observation.marker_ids[1])

    def test_homografia_aplica_limite_fisico_de_18_a_35_mm(self):
        black, green = _cross_scene(right_pre=True)
        image_to_ground = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        valid = analyze_green_intersection(
            black,
            green,
            image_to_ground=image_to_ground,
            entry_point=ENTRY,
            entry_tangent=(0.0, 1.0),
        )
        too_large = image_to_ground.copy()
        too_large[0, 0] = 2.0
        too_large[1, 1] = -2.0
        rejected = analyze_green_intersection(
            black,
            green,
            image_to_ground=too_large,
            entry_point=ENTRY,
            entry_tangent=(0.0, 1.0),
        )
        self.assertEqual(valid.decision, GreenDecision.RIGHT)
        self.assertFalse(rejected.markers[0].plausible)
        self.assertNotEqual(rejected.decision, GreenDecision.RIGHT)

    def test_quadrado_fisico_alongado_na_imagem_e_validado_no_chao(self):
        black, green = _cross_scene(right_pre=True)
        ground_to_image = np.array(
            [[1.7, 0.0, 0.0], [0.0, 0.65, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        output_size = (714, 273)
        warped_black = cv2.warpPerspective(
            black, ground_to_image, output_size, flags=cv2.INTER_NEAREST)
        warped_green = cv2.warpPerspective(
            green, ground_to_image, output_size, flags=cv2.INTER_NEAREST)
        entry = cv2.perspectiveTransform(
            np.float32([[[ENTRY[0], ENTRY[1]]]]), ground_to_image)[0, 0]
        y_forward = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        image_to_ground = y_forward @ np.linalg.inv(ground_to_image)
        observation = analyze_green_intersection(
            warped_black,
            warped_green,
            image_to_ground=image_to_ground,
            entry_point=tuple(entry),
            entry_tangent=(0.0, 1.0),
        )
        self.assertEqual(observation.decision, GreenDecision.RIGHT)
        self.assertTrue(observation.markers[0].plausible)
        self.assertAlmostEqual(
            observation.markers[0].side_length, 20.0, delta=1.0)

    def test_perspectiva_e_desfeita_no_referencial_do_chao(self):
        black, green = _cross_scene(right_pre=True)
        source = np.float32([[20, 20], [400, 20], [400, 400], [20, 400]])
        destination = np.float32(
            [[105, 45], [315, 45], [405, 405], [15, 405]])
        ground_to_image = cv2.getPerspectiveTransform(source, destination)
        warped_black = cv2.warpPerspective(
            black, ground_to_image, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        warped_green = cv2.warpPerspective(
            green, ground_to_image, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        entry = cv2.perspectiveTransform(
            np.float32([[[ENTRY[0], ENTRY[1]]]]), ground_to_image)[0, 0]
        y_forward = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        image_to_ground = y_forward @ np.linalg.inv(ground_to_image)
        observation = analyze_green_intersection(
            warped_black,
            warped_green,
            image_to_ground=image_to_ground,
            entry_point=tuple(entry),
            entry_tangent=(0.0, 1.0),
        )
        self.assertEqual(observation.decision, GreenDecision.RIGHT)
        self.assertAlmostEqual(observation.target_branch.angle_deg, -90.0, delta=3.0)

    def test_contorno_verde_dividido_por_um_pixel_e_reparado(self):
        black, green = _cross_scene(right_pre=True)
        green[225:246, 233] = 0
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.RIGHT)
        self.assertEqual(len(observation.markers), 1)

    def test_fendas_de_tres_a_oito_pixels_preservam_90_e_180(self):
        for gap in range(3, 9):
            for angle in range(0, 360, 45):
                for uturn, expected in (
                    (False, GreenDecision.RIGHT),
                    (True, GreenDecision.UTURN),
                ):
                    black, green = _cross_scene(
                        right_pre=True,
                        left_pre=uturn,
                    )
                    start = 233 - gap // 2
                    green[225:246, start:start + gap] = 0
                    b_rot, g_rot, entry, tangent = _rotate_scene(
                        black, green, angle)
                    observation = analyze_green_intersection(
                        b_rot,
                        g_rot,
                        entry_point=entry,
                        entry_tangent=tangent,
                    )
                    with self.subTest(gap=gap, angle=angle, uturn=uturn):
                        self.assertEqual(observation.decision, expected)

    def test_oclusao_parcial_do_canto_nao_inverte_marcador_valido(self):
        black, green = _cross_scene(right_pre=True)
        green[225:231, 223:229] = 0
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.RIGHT)

    def test_reflexo_verde_alongado_nunca_autoriza_curva(self):
        black, green = _cross_scene()
        cv2.rectangle(green, (225, 225), (231, 255), 255, -1)
        black[green > 0] = 0
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.STRAIGHT)
        self.assertFalse(any(marker.valid for marker in observation.markers))

    def test_dois_reflexos_quadrados_distantes_nao_formam_180(self):
        black, green = _cross_scene()
        cv2.rectangle(green, (170, 290), (190, 310), 255, -1)
        cv2.rectangle(green, (230, 290), (250, 310), 255, -1)
        black[green > 0] = 0
        observation = analyze_green_intersection(
            black, green, entry_point=ENTRY, entry_tangent=(0.0, 1.0))
        self.assertEqual(observation.decision, GreenDecision.STRAIGHT)
        self.assertFalse(any(
            marker.valid or marker.associated
            for marker in observation.markers
        ))

    def test_marcador_proximo_da_borda_mantem_o_lado_do_trajeto(self):
        for offset, scene, expected in (
            (110.0, {"right_pre": True}, GreenDecision.RIGHT),
            (-110.0, {"left_pre": True}, GreenDecision.LEFT),
        ):
            black, green = _cross_scene(**scene)
            matrix = np.float32([[1.0, 0.0, offset], [0.0, 1.0, 0.0]])
            moved_black = cv2.warpAffine(
                black, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            moved_green = cv2.warpAffine(
                green, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            observation = analyze_green_intersection(
                moved_black,
                moved_green,
                entry_point=(ENTRY[0] + offset, ENTRY[1]),
                entry_tangent=(0.0, 1.0),
            )
            with self.subTest(offset=offset):
                self.assertEqual(observation.decision, expected)

    def test_verde_e_ramo_cortados_na_borda_ficam_pending(self):
        for offset in (185.0, 190.0):
            black, green = _cross_scene(right_pre=True)
            matrix = np.float32([[1.0, 0.0, offset], [0.0, 1.0, 0.0]])
            moved_black = cv2.warpAffine(
                black, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            moved_green = cv2.warpAffine(
                green, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            observation = analyze_green_intersection(
                moved_black,
                moved_green,
                entry_point=(ENTRY[0] + offset, ENTRY[1]),
                entry_tangent=(0.0, 1.0),
            )
            with self.subTest(offset=offset):
                self.assertEqual(observation.decision, GreenDecision.PENDING)
                self.assertTrue(observation.geometry_truncated)
                self.assertIn("truncada pela borda", observation.reason)

    def test_verde_cortado_na_borda_inferior_nunca_autoriza_curva(self):
        for offset_y in (175.0, 185.0, 190.0):
            black, green = _cross_scene(right_pre=True)
            matrix = np.float32([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, offset_y],
            ])
            moved_black = cv2.warpAffine(
                black, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            moved_green = cv2.warpAffine(
                green, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            observation = analyze_green_intersection(
                moved_black,
                moved_green,
                entry_point=(ENTRY[0], SIZE - 1.0),
                entry_tangent=(0.0, 1.0),
            )
            with self.subTest(offset_y=offset_y):
                self.assertEqual(observation.decision, GreenDecision.PENDING)
                self.assertTrue(observation.geometry_truncated)
                self.assertIn("truncada pela borda", observation.reason)

    def test_verde_com_margem_inferior_real_continua_valido(self):
        black, green = _cross_scene(right_pre=True)
        matrix = np.float32([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 170.0],
        ])
        observation = analyze_green_intersection(
            cv2.warpAffine(
                black, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST),
            cv2.warpAffine(
                green, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST),
            entry_point=(ENTRY[0], SIZE - 1.0),
            entry_tangent=(0.0, 1.0),
        )

        self.assertEqual(observation.decision, GreenDecision.RIGHT)
        self.assertFalse(any(marker.touches_border
                             for marker in observation.markers))

    def test_ruido_de_um_pixel_na_borda_nao_bloqueia_verde_central(self):
        black, green = _cross_scene(right_pre=True)
        green[0, 0] = 255
        observation = analyze_green_intersection(
            black,
            green,
            entry_point=ENTRY,
            entry_tangent=(0.0, 1.0),
        )

        self.assertEqual(observation.decision, GreenDecision.RIGHT)

    def test_muitos_speckles_verdes_sao_descartados_antes_da_geometria(self):
        height, width = 252, 448
        black = np.zeros((height, width), dtype=np.uint8)
        green = np.zeros_like(black)
        cv2.rectangle(black, (215, 0), (233, 251), 255, -1)
        cv2.rectangle(black, (70, 105), (380, 123), 255, -1)
        cv2.rectangle(green, (235, 128), (253, 146), 255, -1)
        black[green > 0] = 0
        for y in range(8, height - 8, 24):
            for x in range(8, width - 8, 28):
                if black[y, x] == 0 and green[y, x] == 0:
                    green[y, x] = 255

        samples = []
        for _ in range(15):
            start = time.perf_counter()
            observation = analyze_green_intersection(
                black,
                green,
                entry_point=(224.0, 251.0),
                entry_tangent=(0.0, 1.0),
            )
            samples.append((time.perf_counter() - start) * 1000.0)

        self.assertEqual(observation.decision, GreenDecision.RIGHT)
        self.assertLess(float(np.percentile(samples, 95)), 25.0)

    def test_custo_da_cena_448x252_fica_abaixo_do_orcamento(self):
        height, width = 252, 448
        black = np.zeros((height, width), dtype=np.uint8)
        green = np.zeros_like(black)
        cv2.rectangle(black, (215, 0), (233, 251), 255, -1)
        cv2.rectangle(black, (70, 105), (380, 123), 255, -1)
        cv2.rectangle(green, (235, 128), (253, 146), 255, -1)
        black[green > 0] = 0
        samples = []
        for _ in range(15):
            start = time.perf_counter()
            observation = analyze_green_intersection(
                black,
                green,
                entry_point=(224.0, 251.0),
                entry_tangent=(0.0, 1.0),
            )
            samples.append((time.perf_counter() - start) * 1000.0)
        self.assertEqual(observation.decision, GreenDecision.RIGHT)
        # Folga para runners compartilhados; a medicao real da Pi permanece
        # parte da liberacao fisica.
        self.assertLess(float(np.percentile(samples, 95)), 25.0)


if __name__ == "__main__":
    unittest.main()
