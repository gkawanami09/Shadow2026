"""Identidade visual fail-closed do ramo topologico."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.rastreamento_ramo import LockedBranchTracker  # noqa: E402


def _scene(shift_x=0, junction_y=126):
    image = np.full((252, 448, 3), 235, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(
        mask,
        (224 + shift_x, junction_y),
        (224 + shift_x, 251),
        255,
        22,
    )
    image[mask > 0] = (15, 15, 15)
    # Textura fixa sobre o ramo fornece cantos que o LK consegue verificar.
    for y in range(junction_y + 19, 241, 16):
        cv2.rectangle(
            image,
            (218 + shift_x, y),
            (223 + shift_x, y + 5),
            (80, 80, 80),
            -1,
        )
    return image, mask


def _textured_cross(junction_y=55):
    image = np.full((252, 448, 3), 235, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, (224, 0), (224, 251), 255, 22)
    cv2.line(mask, (35, junction_y), (413, junction_y), 255, 22)
    image[mask > 0] = (15, 15, 15)
    for y in range(12, 246, 18):
        cv2.rectangle(image, (218, y), (223, y + 5), (75, 75, 75), -1)
    for x in range(45, 405, 24):
        cv2.rectangle(
            image, (x, junction_y - 6), (x + 5, junction_y - 1),
            (75, 75, 75), -1)
    # Textura fraca do piso, fixa no plano, distribui os inliers do SE(2).
    for y in range(20, 240, 44):
        for x in range(25, 430, 52):
            cv2.circle(image, (x, y), 2, (205, 205, 205), -1)
    return image, mask


class LockedBranchTrackerTests(unittest.TestCase):
    def test_mesmo_ramo_e_token_sobrevivem_a_movimento_curto(self):
        tracker = LockedBranchTracker(min_points=4)
        first, mask_first = _scene()
        armed = tracker.arm(
            first,
            mask_first,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 126),
            target=(224, 185),
            line_width_px=22,
        )
        second, mask_second = _scene(shift_x=4)
        tracked = tracker.update(
            second,
            mask_second,
            sequence=2,
            decision_id=3,
            branch_token=29,
        )

        self.assertTrue(armed.valid)
        self.assertTrue(tracked.valid)
        self.assertEqual(tracked.token, 29)
        self.assertAlmostEqual(tracked.bottom_x, 228, delta=5)

    def test_perda_nao_reacopla_automaticamente_em_outra_faixa(self):
        tracker = LockedBranchTracker(min_points=4)
        first, mask_first = _scene()
        tracker.arm(
            first,
            mask_first,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 126),
            target=(224, 185),
            line_width_px=22,
        )
        blank = np.full_like(first, 235)
        no_black = np.zeros_like(mask_first)
        lost = tracker.update(
            blank,
            no_black,
            sequence=2,
            decision_id=3,
            branch_token=29,
        )
        other, other_mask = _scene(shift_x=90)
        still_lost = tracker.update(
            other,
            other_mask,
            sequence=3,
            decision_id=3,
            branch_token=29,
        )

        self.assertFalse(lost.valid)
        self.assertFalse(still_lost.valid)
        self.assertEqual(still_lost.token, 29)

    def test_token_diferente_reseta_sem_transferir_identidade(self):
        tracker = LockedBranchTracker(min_points=4)
        frame, mask = _scene()
        tracker.arm(
            frame,
            mask,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 126),
            target=(224, 185),
            line_width_px=22,
        )

        result = tracker.update(
            frame,
            mask,
            sequence=2,
            decision_id=3,
            branch_token=30,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.token, 0)

    def test_reseed_topologico_sobrevive_a_150px_de_aproximacao(self):
        tracker = LockedBranchTracker(min_points=4)
        frame, mask = _scene(junction_y=55)
        tracker.arm(
            frame,
            mask,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 55),
            target=(224, 82),
            line_width_px=22,
        )

        # Dez passos representam a aproximacao; a geometria real da mesma
        # juncao/token renova pontos visiveis a cada frame.
        for step in range(1, 11):
            offset = 15 * step
            matrix = np.float32(((1, 0, 0), (0, 1, offset)))
            moved_frame = cv2.warpAffine(
                frame, matrix, (448, 252), borderValue=(235, 235, 235))
            moved_mask = cv2.warpAffine(
                mask, matrix, (448, 252), flags=cv2.INTER_NEAREST)
            result = tracker.update(
                moved_frame,
                moved_mask,
                sequence=step * 2,
                decision_id=3,
                branch_token=29,
            )
            result = tracker.refresh_from_verified_geometry(
                moved_frame,
                moved_mask,
                sequence=step * 2 + 1,
                decision_id=3,
                branch_token=29,
                junction=(224, 55 + offset),
                target=(224, 82 + offset),
                line_width_px=22,
            )

        self.assertTrue(result.valid)
        self.assertEqual(result.token, 29)

    def test_salto_de_pose_grande_nao_transfere_token_para_concorrente(self):
        tracker = LockedBranchTracker(min_points=4)
        frame, mask = _scene()
        tracker.arm(
            frame,
            mask,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 126),
            target=(224, 185),
            line_width_px=22,
        )
        tracker._points = None
        other, other_mask = _scene(shift_x=90)
        tracker._estimate_frame_motion = lambda *_args: np.array((
            (1.0, 0.0, 90.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ))

        result = tracker.update(
            other,
            other_mask,
            sequence=2,
            decision_id=3,
            branch_token=29,
        )

        # Mesmo uma matriz injetada nao deve ser aceita por fora dos limites
        # fisicos da estimativa real; sem lineage visual, fica fail-closed.
        self.assertFalse(result.valid)

    def test_sem_geometria_real_nao_pode_reseed_em_faixa_nova(self):
        tracker = LockedBranchTracker(min_points=4)
        frame, mask = _scene()
        tracker.arm(
            frame,
            mask,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 126),
            target=(224, 185),
            line_width_px=22,
        )
        blank = np.full_like(frame, 235)
        tracker.update(
            blank,
            np.zeros_like(mask),
            sequence=2,
            decision_id=3,
            branch_token=29,
        )
        other, other_mask = _scene(shift_x=90)

        result = tracker.update(
            other,
            other_mask,
            sequence=3,
            decision_id=3,
            branch_token=29,
        )

        self.assertFalse(result.valid)

    def test_token_sobrevive_ao_giro_em_passos_de_cinco_graus(self):
        tracker = LockedBranchTracker(min_points=4)
        frame, mask = _scene()
        tracker.arm(
            frame,
            mask,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 126),
            target=(224, 185),
            line_width_px=22,
        )

        result = None
        for sequence, angle in enumerate(range(5, 181, 5), start=2):
            matrix = cv2.getRotationMatrix2D((224, 126), -angle, 1.0)
            rotated_frame = cv2.warpAffine(
                frame, matrix, (448, 252), borderValue=(235, 235, 235))
            rotated_mask = cv2.warpAffine(
                mask, matrix, (448, 252), flags=cv2.INTER_NEAREST)
            result = tracker.update(
                rotated_frame,
                rotated_mask,
                sequence=sequence,
                decision_id=3,
                branch_token=29,
            )
            self.assertTrue(result.valid, f"token perdido em {angle} graus")

        self.assertEqual(result.token, 29)

    def test_fluxo_completo_aproximacao_preroll_e_giro_nao_troca_ramo(self):
        tracker = LockedBranchTracker(min_points=4)
        base, base_mask = _textured_cross()
        tracker.arm(
            base,
            base_mask,
            sequence=1,
            decision_id=3,
            branch_token=29,
            junction=(224, 55),
            target=(224, 82),
            line_width_px=22,
        )
        sequence = 1
        # Aproximacao: a junção desce 150 px e ainda permite reseed real.
        for offset in range(15, 151, 15):
            sequence += 1
            matrix = np.float32(((1, 0, 0), (0, 1, offset)))
            frame = cv2.warpAffine(
                base, matrix, (448, 252), borderValue=(235, 235, 235))
            mask = cv2.warpAffine(
                base_mask, matrix, (448, 252), flags=cv2.INTER_NEAREST)
            tracker.update(
                frame, mask, sequence=sequence,
                decision_id=3, branch_token=29)
            sequence += 1
            tracker.refresh_from_verified_geometry(
                frame,
                mask,
                sequence=sequence,
                decision_id=3,
                branch_token=29,
                junction=(224, 55 + offset),
                target=(224, 82 + offset),
                line_width_px=22,
            )

        # Pre-roll sem nova geometria topologica.
        total_offset = 150
        for extra in (8, 16, 24, 32):
            sequence += 1
            total_offset = 150 + extra
            matrix = np.float32(((1, 0, 0), (0, 1, total_offset)))
            frame = cv2.warpAffine(
                base, matrix, (448, 252), borderValue=(235, 235, 235))
            mask = cv2.warpAffine(
                base_mask, matrix, (448, 252), flags=cv2.INTER_NEAREST)
            tracker.update(
                frame, mask, sequence=sequence,
                decision_id=3, branch_token=29)

        translated = np.float32(((1, 0, 0), (0, 1, total_offset)))
        base_moved = cv2.warpAffine(
            base, translated, (448, 252), borderValue=(235, 235, 235))
        mask_moved = cv2.warpAffine(
            base_mask, translated, (448, 252), flags=cv2.INTER_NEAREST)
        seen_after_side = False
        for angle in range(5, 181, 5):
            sequence += 1
            rotation = cv2.getRotationMatrix2D((224, 126), -angle, 1.0)
            frame = cv2.warpAffine(
                base_moved, rotation, (448, 252),
                borderValue=(235, 235, 235))
            mask = cv2.warpAffine(
                mask_moved, rotation, (448, 252),
                flags=cv2.INTER_NEAREST)
            result = tracker.update(
                frame, mask, sequence=sequence,
                decision_id=3, branch_token=29)
            if angle >= 100 and result.valid:
                seen_after_side = True
                self.assertEqual(result.token, 29)

        self.assertTrue(seen_after_side)


if __name__ == "__main__":
    unittest.main()
