"""Testes da fronteira entre a topologia retificada e o controle."""

import sys
from pathlib import Path
import types
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

# O Windows de desenvolvimento nao possui Numba. A fronteira testada aqui nao
# usa JIT, mas processamento importa o modulo legado de linha ao inicializar.
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    numba_falso = types.ModuleType("numba")

    def njit_falso(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]
        return lambda funcao: funcao

    numba_falso.njit = njit_falso
    sys.modules["numba"] = numba_falso

import config  # noqa: E402
from controle.estado_verde import (  # noqa: E402
    GreenDecision,
    GreenDecisionTracker,
    GreenObservation,
    empty_observation,
)
from visao.intersecao_verde import (  # noqa: E402
    BranchKind,
    BranchObservation,
    GreenDecision as TopologyDecision,
    TopologyObservation,
)
from visao.processamento import (  # noqa: E402
    _evento_bruto_topologia,
    _juncao_presente_para_saida,
    _preferencia_esquerda_permitida,
)


def observacao_topologica(*, propagada=False, target=(280.0, 90.0)):
    ramo = BranchObservation(
        kind=BranchKind.STRAIGHT,
        angle_deg=0.0,
        direction=(0.0, 1.0),
        target_image=target,
        length_widths=1.5,
        branch_token=77,
    )
    return TopologyObservation(
        decision=TopologyDecision.STRAIGHT,
        confidence=.9,
        entry_tangent=(0.0, 1.0),
        junction_image=(250.0, config.camera_y * .90),
        junction_ground=(10.0, 80.0),
        branches=(ramo,),
        target_branch=ramo,
        junction_id=7,
        entry_propagated=propagada,
    )


class AdaptadorEventoTopologicoTests(unittest.TestCase):
    def test_entrada_propagada_nao_e_publicada_como_geometria_visivel(self):
        evento = _evento_bruto_topologia(
            observacao_topologica(propagada=True),
            sequencia=10,
            timestamp=1.0,
        )

        self.assertFalse(evento.junction_visible)
        self.assertTrue(evento.geometry_predicted)
        self.assertFalse(evento.ready_to_turn)

    def test_propagacao_bloqueia_saida_e_rearme_como_falha_fechada(self):
        # Isola o segundo operando: mesmo sem centro de juncao mensurado,
        # a janela curta de propagacao ainda nao prova que ela desapareceu.
        cena = TopologyObservation(
            decision=TopologyDecision.PENDING,
            junction_image=None,
            entry_propagated=True,
        )

        self.assertTrue(_juncao_presente_para_saida(cena))

    def test_so_ausencia_real_libera_saida_e_rearme(self):
        cena = TopologyObservation(
            decision=TopologyDecision.NONE,
            entry_propagated=False,
        )

        self.assertFalse(_juncao_presente_para_saida(cena))

    def test_propagacao_nao_completa_tres_de_cinco(self):
        tracker = GreenDecisionTracker(
            confirm_frames=3,
            window_frames=5,
            second_marker_wait_s=0.0,
        )
        cenas = (
            observacao_topologica(propagada=False),
            observacao_topologica(propagada=True),
            observacao_topologica(propagada=True),
        )

        resultado = None
        for indice, cena in enumerate(cenas, start=1):
            resultado = tracker.update(_evento_bruto_topologia(
                cena,
                sequencia=indice,
                timestamp=indice * .02,
            ))

        self.assertIsNotNone(resultado)
        self.assertFalse(resultado.committed)

    def test_alvo_retificado_e_convertido_para_o_frame_cru(self):
        chamadas = []

        def conversor_falso(pontos):
            pontos = np.asarray(pontos, dtype=np.float64)
            chamadas.append(pontos.copy())
            return pontos + np.array((12.0, -7.0), dtype=np.float64)

        evento = _evento_bruto_topologia(
            observacao_topologica(target=(280.0, 90.0)),
            sequencia=3,
            timestamp=.3,
            target_to_raw=conversor_falso,
        )

        self.assertEqual(len(chamadas), 1)
        np.testing.assert_allclose(chamadas[0], ((280.0, 90.0),))
        self.assertEqual(evento.target_branch, (292.0, 83.0))
        self.assertEqual(evento.target_branch_token, 77)
        # O centro da juncao e o gatilho de 82% permanecem retificados.
        self.assertEqual(
            evento.junction_center,
            (250.0, config.camera_y * .90),
        )
        self.assertTrue(evento.ready_to_turn)

    def test_preferencia_residual_nao_supera_straight_comprometido(self):
        comprometido = GreenObservation(
            sequence=1,
            junction_id=7,
            decision_id=11,
            timestamp=.1,
            decision=GreenDecision.STRAIGHT,
            confidence=.9,
            target_branch=(224.0, 90.0),
            junction_visible=True,
        )

        self.assertFalse(_preferencia_esquerda_permitida(
            True, comprometido))
        self.assertTrue(_preferencia_esquerda_permitida(
            True, empty_observation()))
        self.assertFalse(_preferencia_esquerda_permitida(
            False, empty_observation()))

    def test_preferencia_residual_tambem_cede_para_pending(self):
        pending = GreenObservation(
            sequence=2,
            junction_id=7,
            decision_id=0,
            timestamp=.2,
            decision=GreenDecision.PENDING,
            confidence=.5,
        )

        self.assertFalse(_preferencia_esquerda_permitida(True, pending))


if __name__ == "__main__":
    unittest.main()
