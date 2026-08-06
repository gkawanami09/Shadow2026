"""Testes da faixa PRETA de saída (câmera de resgate)."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from tests import cenas_sinteticas as cs  # noqa: E402
from visao.faixa_saida import (  # noqa: E402
    BlackExitDetector,
    BlackExitGate,
)
from visao.faixa_transversal import StripeConfirmer  # noqa: E402


class BlackExitDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = BlackExitDetector()

    def test_faixa_preta_verdadeira_e_aceita(self):
        detection = self.detector.detect(cs.faixa_preta(), timestamp=1.0)
        self.assertIsNotNone(detection)
        self.assertEqual(self.detector.last_reason, "")
        self.assertGreaterEqual(
            detection.confidence, cfg.EXIT_BLACK_MIN_CONFIDENCE)

    def test_vitima_preta_nunca_e_lida_como_saida(self):
        """O risco real da sala: a esfera preta é escura como a soleira."""
        for raio in (40, 60, 100, 160, 200, 260):
            frame = cs.esfera(cs.RESCUE_FRAME, raio, 15, 185)
            self.assertIsNone(
                self.detector.detect(frame, timestamp=1.0),
                f"esfera preta de raio {raio} foi lida como faixa de saída")

    def test_sombra_ampla_sem_contraste_e_rejeitada(self):
        self.assertIsNone(self.detector.detect(cs.sombra_ampla(),
                                               timestamp=1.0))

    def test_sombra_fina_sobre_piso_escuro_reprova_no_contraste(self):
        """Piso logo acima do limiar de escuro: a forma passa, o contraste não.

        Isto isola o teste de contraste — a faixa tem geometria perfeita e é
        reprovada apenas por não se destacar do piso ao redor.
        """
        frame = cs.faixa_preta(
            topo=0.82, espessura=0.08, piso=85, valor=65)
        self.assertIsNone(self.detector.detect(frame, timestamp=1.0))
        self.assertEqual(self.detector.last_reason, "sem_contraste")

    def test_madeira_nao_aciona_saida(self):
        self.assertIsNone(self.detector.detect(cs.madeira(), timestamp=1.0))

    def test_piso_claro_vazio_nao_aciona_saida(self):
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 185)
        self.assertIsNone(self.detector.detect(frame, timestamp=1.0))

    def test_reflexo_acima_da_regiao_do_piso_nao_e_faixa(self):
        """O robô refletido na parede prata fica acima da soleira real."""
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        height, width = frame.shape[:2]
        top = int(height * 0.53)
        bottom = int(height * 0.67)
        cv2.rectangle(
            frame,
            (0, top),
            (width - 1, bottom),
            (20, 20, 20),
            -1,
        )
        self.assertIsNone(self.detector.detect(frame, timestamp=1.0))

    def test_faixa_curta_e_fina_rente_ao_chao_e_aceita(self):
        """De longe, a soleira ocupa só parte da largura e poucos pixels."""
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        height, width = frame.shape[:2]
        y = int(round(height * 0.88))
        x1 = int(round(width * 0.34))
        x2 = int(round(width * 0.66))
        cv2.rectangle(
            frame,
            (x1, y),
            (x2, y + max(int(round(height * 0.025)), 4)),
            (18, 18, 18),
            -1,
        )

        detection = self.detector.detect(frame, timestamp=1.0)

        self.assertIsNotNone(detection)
        self.assertGreaterEqual(
            detection.center_y / height, cfg.EXIT_BLACK_ROI_TOP)

    def test_faixa_de_tres_pixels_rente_ao_chao_e_aceita(self):
        """A câmera quase horizontal reduz a soleira a uma linha muito fina."""
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        height, width = frame.shape[:2]
        cv2.line(
            frame,
            (int(width * 0.27), int(height * 0.89)),
            (int(width * 0.73), int(height * 0.91)),
            (18, 18, 18),
            3,
        )

        detection = self.detector.detect(frame, timestamp=1.0)

        self.assertIsNotNone(detection)
        self.assertGreaterEqual(
            detection.span_ratio, cfg.EXIT_LINE_MIN_LENGTH_RATIO)

    def test_triangulos_coloridos_nao_acionam_saida(self):
        """Verde e vermelho saturados não são escuros e não viram soleira."""
        for cor in ((0, 190, 0), (0, 0, 210)):
            frame = cs.piso_neutro(cs.RESCUE_FRAME, 185)
            frame[330:420, 200:440] = cor
            self.assertIsNone(self.detector.detect(frame, timestamp=1.0))

    def test_retangulo_verde_escuro_nao_vira_faixa_preta(self):
        """Mesmo com V baixo, matiz verde veta a falsa saída."""
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 185)
        frame[360:420, 100:540] = (0, 60, 0)

        self.assertIsNone(self.detector.detect(frame, timestamp=1.0))
        self.assertEqual(self.detector.last_reason, "verde")

    def test_manchas_e_risco_curto_nao_acionam_fallback(self):
        """Uma região escura local não é uma faixa transversal."""
        frames = []

        mancha = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        cv2.ellipse(mancha, (320, 370), (100, 45), 12, 0, 360,
                    (25, 25, 25), -1)
        frames.append(mancha)

        risco = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        cv2.line(risco, (210, 350), (390, 370), (20, 20, 20), 3)
        frames.append(risco)

        irregular = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        pontos = np.asarray([
            [130, 350], [210, 325], [290, 345], [370, 320], [460, 360],
            [420, 405], [330, 385], [240, 410], [150, 390],
        ], dtype=np.int32)
        cv2.fillPoly(irregular, [pontos], (25, 25, 25))
        frames.append(irregular)

        for frame in frames:
            self.assertIsNone(self.detector.detect(frame, timestamp=1.0))

    def test_faixa_fina_com_reflexos_ainda_e_aceita(self):
        """Trechos quebrados da mesma fita são unidos pelo fallback."""
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 190)
        faixa = np.asarray(
            [[90, 430], [560, 395], [560, 409], [90, 444]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, faixa, (25, 25, 25))
        for x in (220, 350, 480):
            cv2.rectangle(frame, (x, 390), (x + 12, 460),
                          (190, 190, 190), -1)

        detection = self.detector.detect(frame, timestamp=1.0)

        self.assertIsNotNone(detection)
        self.assertGreaterEqual(
            detection.span_ratio, cfg.EXIT_LINE_MIN_LENGTH_RATIO)
        self.assertGreaterEqual(
            detection.dark_support, cfg.EXIT_LINE_MIN_DARK_SUPPORT)


class BlackExitGateTests(unittest.TestCase):
    def _gate(self):
        return BlackExitGate(
            confirmer=StripeConfirmer(
                votes_needed=cfg.EXIT_BLACK_VOTES_NEEDED,
                window=cfg.EXIT_BLACK_VOTE_WINDOW,
                max_age_s=cfg.BALL_FRAME_STALE_S,
            )
        )

    def test_um_frame_nao_confirma_a_saida(self):
        gate = self._gate()
        confirmed, detection = gate.update(
            cs.faixa_preta(), timestamp=1.0, now=1.0)
        self.assertFalse(confirmed)
        self.assertIsNotNone(detection)

    def test_previa_do_giro_detecta_mas_nao_soma_voto(self):
        gate = self._gate()
        frame = cs.faixa_preta()
        for index in range(5):
            detection = gate.preview(
                frame, timestamp=1.0 + index * 0.05)
            self.assertIsNotNone(detection)
        self.assertEqual(gate.votes, 0)
        self.assertFalse(gate.confirmed)

    def test_tres_frames_distintos_confirmam_a_saida(self):
        gate = self._gate()
        frame = cs.faixa_preta()
        confirmed = False
        for index in range(cfg.EXIT_BLACK_VOTES_NEEDED):
            timestamp = 1.0 + index * 0.05
            confirmed, _ = gate.update(
                frame, timestamp=timestamp, now=timestamp)
        self.assertTrue(confirmed)

    def test_resultado_stale_nao_confirma(self):
        gate = self._gate()
        frame = cs.faixa_preta()
        confirmed = False
        for index in range(5):
            timestamp = 1.0 + index * 0.05
            # Cada resultado chega muito depois da captura.
            confirmed, _ = gate.update(
                frame, timestamp=timestamp,
                now=timestamp + cfg.BALL_FRAME_STALE_S + 0.5)
        self.assertFalse(confirmed)

    def test_sequencia_de_esferas_pretas_nunca_confirma(self):
        gate = self._gate()
        confirmed = False
        for index in range(10):
            frame = cs.esfera(cs.RESCUE_FRAME, 70 + index * 12, 15, 185)
            timestamp = 1.0 + index * 0.05
            confirmed, _ = gate.update(
                frame, timestamp=timestamp, now=timestamp)
        self.assertFalse(confirmed)

    def test_candidatos_em_lugares_diferentes_nao_somam_votos(self):
        class DetectorQuePula:
            def __init__(self):
                self.index = 0

            def detect(self, _frame, timestamp=None):
                centros = (90.0, 320.0, 550.0)
                centro = centros[self.index]
                self.index = min(self.index + 1, len(centros) - 1)
                return SimpleNamespace(
                    center_x=centro,
                    center_y=360.0,
                    span_ratio=0.50,
                    timestamp=timestamp,
                )

        gate = BlackExitGate(
            detector=DetectorQuePula(),
            confirmer=StripeConfirmer(
                votes_needed=cfg.EXIT_BLACK_VOTES_NEEDED,
                window=cfg.EXIT_BLACK_VOTE_WINDOW,
                max_age_s=cfg.BALL_FRAME_STALE_S,
            ),
        )
        frame = cs.piso_neutro(cs.RESCUE_FRAME, 185)
        confirmed = False
        for index in range(3):
            timestamp = 1.0 + index * 0.05
            confirmed, _ = gate.update(
                frame, timestamp=timestamp, now=timestamp)

        self.assertFalse(confirmed)
        self.assertEqual(gate.votes, 1)


if __name__ == "__main__":
    unittest.main()
