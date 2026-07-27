"""Testes da nova visão de vítimas: tipos, plausibilidade física e tracking.

A arquitetura separa três julgamentos que o detector anterior misturava:

    modelo        -> aparência   (é vítima? prata ou preta?)
    plausibilidade-> geometria   (cabe fisicamente ali?)
    tracking      -> tempo       (aparece de forma consistente?)

Estes testes cobrem as duas últimas — que não dependem de modelo treinado e
por isso podem ser verificadas hoje. A camada de aparência é o modelo, e só
poderá ser medida com o dataset da câmera deste robô.
"""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from visao.deteccao import (  # noqa: E402
    CORES_VALIDAS,
    VictimCandidate,
    VictimDetection,
)
from visao.plausibilidade import (  # noqa: E402
    PlausibilityGuard,
    envelope_de_raio,
)
from visao.vitima_yolo import (  # noqa: E402
    ModeloAusenteError,
    VictimDetector,
    VictimModel,
)


FORMA = (480, 640, 3)


def candidato(x=320, y=400, raio=40, kind="silver", confianca=0.9):
    return VictimCandidate(kind, float(x), float(y), float(raio), confianca)


class ModeloFalso:
    """Devolve candidatos fixos. Substitui a inferência nos testes."""

    def __init__(self, por_frame=None):
        self.por_frame = por_frame or []
        self.chamadas = 0

    def inferir(self, frame):
        indice = min(self.chamadas, len(self.por_frame) - 1)
        self.chamadas += 1
        return list(self.por_frame[indice]) if self.por_frame else []


class TiposTests(unittest.TestCase):
    def test_cor_invalida_e_recusada_na_construcao(self):
        with self.assertRaises(ValueError):
            VictimDetection(
                "azul", 1.0, 2.0, 3.0, 0.9, True, 3, 0.0)

    def test_cores_validas_sao_apenas_prata_e_preta(self):
        self.assertEqual(set(CORES_VALIDAS), {"silver", "black"})

    def test_erro_lateral_normalizado(self):
        deteccao = VictimDetection(
            "silver", 640.0, 400.0, 30.0, 0.9, True, 3, 0.0)
        self.assertAlmostEqual(deteccao.horizontal_error(640), 1.0)
        centrada = VictimDetection(
            "silver", 320.0, 400.0, 30.0, 0.9, True, 3, 0.0)
        self.assertAlmostEqual(centrada.horizontal_error(640), 0.0)

    def test_caixa_do_yolo_vira_circulo(self):
        cand = VictimCandidate.from_xyxy("black", 100, 200, 160, 260, 0.8)
        self.assertAlmostEqual(cand.center_x, 130.0)
        self.assertAlmostEqual(cand.center_y, 230.0)
        self.assertAlmostEqual(cand.radius, 30.0)


class PlausibilidadeTests(unittest.TestCase):
    """A camada que NÃO depende da arena."""

    def setUp(self):
        self.guard = PlausibilityGuard()

    def test_vitima_no_chao_e_aceita(self):
        resultado = self.guard.check(candidato(y=400, raio=45), FORMA)
        self.assertTrue(resultado.accepted)
        self.assertFalse(resultado.truncated)

    def test_acima_do_horizonte_e_rejeitada(self):
        """Ali só existe parede, público, cadeira e mesa."""
        alto = int(FORMA[0] * cfg.PLAUSIBLE_MIN_CENTER_Y_RATIO) - 20
        resultado = self.guard.check(candidato(y=alto), FORMA)
        self.assertFalse(resultado.accepted)
        self.assertEqual(resultado.reason, "acima_do_horizonte")

    def test_centro_fora_do_quadro_e_rejeitado(self):
        resultado = self.guard.check(candidato(x=-5), FORMA)
        self.assertFalse(resultado.accepted)
        self.assertEqual(resultado.reason, "centro_fora")

    def test_pequena_demais_para_a_linha_e_rejeitada(self):
        """Perspectiva: perto do robô a vítima não pode ser minúscula."""
        minimo, _ = envelope_de_raio(460, FORMA[0], FORMA[1])
        resultado = self.guard.check(
            candidato(y=460, raio=minimo * 0.5), FORMA)
        self.assertFalse(resultado.accepted)
        self.assertEqual(
            resultado.reason, "pequena_demais_para_a_linha")

    def test_grande_demais_para_a_linha_e_rejeitada(self):
        _, maximo = envelope_de_raio(300, FORMA[0], FORMA[1])
        resultado = self.guard.check(
            candidato(y=300, raio=maximo * 1.6), FORMA)
        self.assertFalse(resultado.accepted)
        self.assertEqual(
            resultado.reason, "grande_demais_para_a_linha")

    def test_envelope_cresce_conforme_desce_no_quadro(self):
        """Mais baixo = mais perto = maior. É a própria perspectiva."""
        alto_min, alto_max = envelope_de_raio(250, FORMA[0], FORMA[1])
        baixo_min, baixo_max = envelope_de_raio(460, FORMA[0], FORMA[1])
        self.assertGreater(baixo_min, alto_min)
        self.assertGreater(baixo_max, alto_max)

    def test_encostar_na_lateral_marca_mas_nao_reprova(self):
        """Continua sendo vítima: o robô precisa girar na direção dela."""
        resultado = self.guard.check(
            candidato(x=25, y=430, raio=45), FORMA)
        self.assertTrue(resultado.accepted)
        self.assertTrue(resultado.truncated)

    def test_desligado_aceita_tudo(self):
        solto = PlausibilityGuard(enabled=False)
        self.assertTrue(solto.check(candidato(y=10, raio=1), FORMA))


class ModeloAusenteTests(unittest.TestCase):
    """Sem modelo treinado o sistema PARA e explica — nunca finge."""

    def test_carregar_sem_arquivo_falha_com_instrucao(self):
        modelo = VictimModel(caminho=SHADOW_ROOT / "modelos" / "nao_existe.onnx")
        with self.assertRaises(ModeloAusenteError) as contexto:
            modelo.carregar()
        mensagem = str(contexto.exception)
        self.assertIn("coletar_dataset", mensagem)
        self.assertIn("Roboflow", mensagem)

    def test_inferir_sem_carregar_falha(self):
        with self.assertRaises(ModeloAusenteError):
            VictimModel().inferir(np.zeros(FORMA, dtype=np.uint8))


class RastreamentoTests(unittest.TestCase):
    """A camada temporal: consistência antes de virar alvo."""

    @staticmethod
    def _detector(sequencia, target_kind="any"):
        return VictimDetector(
            model=ModeloFalso(sequencia), target_kind=target_kind)

    @staticmethod
    def _frame():
        return np.zeros(FORMA, dtype=np.uint8)

    def test_um_frame_nao_confirma(self):
        detector = self._detector([[candidato()]])
        deteccao = detector.detect(self._frame(), timestamp=1.0)
        self.assertIsNotNone(deteccao)
        self.assertFalse(deteccao.confirmed)
        self.assertEqual(deteccao.hits, 1)

    def test_confirma_apos_o_numero_de_hits(self):
        sequencia = [[candidato()]] * (cfg.VICTIM_ACQUIRE_HITS + 1)
        detector = self._detector(sequencia)
        deteccao = None
        for indice in range(cfg.VICTIM_ACQUIRE_HITS):
            deteccao = detector.detect(
                self._frame(), timestamp=1.0 + indice * 0.1)
        self.assertTrue(deteccao.confirmed)
        self.assertTrue(deteccao.track_locked)

    def test_frame_repetido_nao_fabrica_confirmacao(self):
        """Mesmo timestamp = mesmo frame: não pode virar hit novo."""
        sequencia = [[candidato()]] * 6
        detector = self._detector(sequencia)
        deteccao = None
        for _ in range(6):
            deteccao = detector.detect(self._frame(), timestamp=1.0)
        self.assertEqual(deteccao.hits, 1)
        self.assertFalse(deteccao.confirmed)

    def test_alvo_travado_nao_e_roubado_por_outro_candidato(self):
        perto = candidato(x=320, y=400, raio=40)
        longe = candidato(x=100, y=430, raio=45, confianca=0.99)
        sequencia = (
            [[perto]] * cfg.VICTIM_ACQUIRE_HITS + [[longe]] * 2)
        detector = self._detector(sequencia)
        for indice in range(cfg.VICTIM_ACQUIRE_HITS):
            detector.detect(self._frame(), timestamp=1.0 + indice * 0.1)
        self.assertTrue(detector._track_locked)
        # Um candidato distante, mesmo com confiança maior, não assume.
        seguinte = detector.detect(self._frame(), timestamp=2.0)
        self.assertIsNone(seguinte)

    def test_alvo_some_por_ausencia_e_nao_por_concorrencia(self):
        sequencia = (
            [[candidato()]] * cfg.VICTIM_ACQUIRE_HITS
            + [[]] * (cfg.VICTIM_MAX_TRACK_MISSES + 2))
        detector = self._detector(sequencia)
        for indice in range(cfg.VICTIM_ACQUIRE_HITS):
            detector.detect(self._frame(), timestamp=1.0 + indice * 0.1)
        for indice in range(cfg.VICTIM_MAX_TRACK_MISSES + 1):
            detector.detect(self._frame(), timestamp=2.0 + indice * 0.1)
        self.assertIsNone(detector._tracked)

    def test_filtro_de_cor_recusa_o_tipo_errado(self):
        detector = self._detector(
            [[candidato(kind="black")]], target_kind="silver")
        self.assertIsNone(detector.detect(self._frame(), timestamp=1.0))
        self.assertIn("tipo", detector.last_rejections)

    def test_candidato_implausivel_nao_vira_alvo(self):
        """A geometria veta antes de o tempo sequer começar a contar."""
        detector = self._detector([[candidato(y=20, raio=40)]])
        self.assertIsNone(detector.detect(self._frame(), timestamp=1.0))
        self.assertIn("acima_do_horizonte", detector.last_rejections)

    def test_marca_truncada_chega_na_deteccao(self):
        detector = self._detector([[candidato(x=25, y=430, raio=45)]])
        deteccao = detector.detect(self._frame(), timestamp=1.0)
        self.assertIsNotNone(deteccao)
        self.assertTrue(deteccao.truncated)

    def test_sem_track_prefere_a_vitima_mais_proxima(self):
        """Mais baixa no quadro = mais perto do robô."""
        longe = candidato(x=200, y=300, raio=22, confianca=0.99)
        perto = candidato(x=420, y=440, raio=50, confianca=0.60)
        detector = self._detector([[longe, perto]])
        deteccao = detector.detect(self._frame(), timestamp=1.0)
        self.assertAlmostEqual(deteccao.center_y, 440.0)


if __name__ == "__main__":
    unittest.main()
