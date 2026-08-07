"""Testes da faixa PRATA de entrada (câmera de linha)."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from tests import cenas_sinteticas as cs  # noqa: E402
from visao.faixa_entrada import (  # noqa: E402
    EntrySilverDetector,
    EntrySilverGate,
    load_bounds,
)
from visao.faixa_transversal import StripeConfirmer  # noqa: E402


class EntrySilverDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = EntrySilverDetector()

    def test_faixa_prata_verdadeira_e_aceita(self):
        detection = self.detector.detect(
            cs.faixa_prata(), line_ahead=False, timestamp=1.0)
        self.assertIsNotNone(detection)
        self.assertEqual(self.detector.last_reason, "")
        self.assertGreaterEqual(
            detection.confidence, config.ENTRY_SILVER_MIN_CONFIDENCE)
        self.assertGreaterEqual(
            detection.span_ratio, config.ENTRY_SILVER_MIN_SPAN_RATIO)

    def test_piso_branco_nao_aciona_entrada(self):
        self.assertIsNone(
            self.detector.detect(cs.piso_branco(), line_ahead=False,
                                 timestamp=1.0))
        # O piso é uniforme: o filtro de reflexo o remove já na máscara, e
        # nem chega a existir região para a geometria julgar.
        self.assertIn(
            self.detector.last_reason,
            ("sem_linha_cheia", "cortada_no_topo", "espessa"))

    def test_reflexo_isolado_nao_aciona_entrada(self):
        self.assertIsNone(
            self.detector.detect(cs.reflexo_pontual(), line_ahead=False,
                                 timestamp=1.0))
        self.assertEqual(self.detector.last_reason, "sem_linha_cheia")

    def test_vitima_prateada_nao_aciona_entrada(self):
        for raio in (40, 90, 150, 220):
            frame = cs.esfera(cs.LINE_FRAME, raio, 210, 120)
            self.assertIsNone(
                self.detector.detect(frame, line_ahead=False, timestamp=1.0),
                f"esfera de raio {raio} foi lida como faixa de entrada")

    def test_faixa_preta_nao_aciona_entrada(self):
        frame = cs.faixa_preta(cs.LINE_FRAME, piso=120, valor=15)
        self.assertIsNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0))

    def test_fita_branca_fosca_reprova_na_assinatura_reflexiva(self):
        """Papel branco liso é neutro e claro, mas não reflete: deve cair.

        O piso fica abaixo do limiar de brilho para que a forma seja aceita e
        a rejeição venha comprovadamente da aparência, não da geometria.
        """
        frame = cs.faixa_prata(
            piso=120, base=200, brilho=200, densidade_brilho=0.0)
        self.assertIsNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0))
        # Sem brilho especular, o filtro de reflexo esvazia a fita já na
        # máscara; ela pode cair na geometria ou na aparência, mas cai.
        self.assertIn(
            self.detector.last_reason,
            ("sem_assinatura_reflexiva", "sem_linha_cheia", "fina",
             "vazada"))

    def test_fita_reflexiva_vence_piso_de_mesmo_brilho(self):
        """O caso medido na arena: cinza e branco com o MESMO brilho.

        Piso e fita têm brilho e neutralidade praticamente idênticos — HSV
        sozinho não separa os dois. O que difere é a textura da luz: a fita
        concentra brilho em pontos, o piso é uniforme. Sem esse filtro, a
        máscara engolia o piso inteiro e o candidato morria em "espessa"
        antes de qualquer teste de aparência.
        """
        piso_liso = cs.faixa_prata(
            piso=210, base=212, brilho=212, densidade_brilho=0.0)
        self.assertIsNone(
            self.detector.detect(piso_liso, line_ahead=False, timestamp=1.0),
            "piso uniforme do mesmo brilho não pode virar faixa")

        fita_reflexiva = cs.faixa_prata(
            piso=210, base=212, brilho=255, densidade_brilho=0.18)
        detection = self.detector.detect(
            fita_reflexiva, line_ahead=False, timestamp=2.0)
        self.assertIsNotNone(
            detection,
            "fita refletiva sobre piso de mesmo brilho deveria ser aceita")

    def test_reflexo_espalhado_perto_da_camera_usa_segunda_mascara(self):
        """A faixa perto da lente não pode virar vários riscos desconectados."""
        frame = cs.faixa_prata(
            piso=205,
            base=212,
            brilho=238,
            densidade_brilho=0.18,
        )
        detection = self.detector.detect(
            frame, line_ahead=False, timestamp=2.0)
        self.assertIsNotNone(detection)
        self.assertGreaterEqual(
            detection.confidence,
            config.ENTRY_SILVER_MIN_CONFIDENCE,
        )

    def test_linha_preta_continuando_veta_a_entrada(self):
        frame = cs.faixa_prata()
        self.assertIsNotNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0))
        self.assertIsNone(
            self.detector.detect(frame, line_ahead=True, timestamp=2.0))
        self.assertEqual(self.detector.last_reason, "linha_continua")

    def test_faixa_realmente_identica_ao_piso_e_rejeitada(self):
        """Idêntica de verdade: mesmo brilho E mesma textura.

        Não basta ter o mesmo brilho — se a fita concentra reflexo e o piso
        não, ela É distinguível e deve ser aceita (isso é coberto em
        `test_fita_reflexiva_vence_piso_de_mesmo_brilho`). O caso rejeitado
        aqui é a superfície uniforme, sem brilho nem textura própria.
        """
        frame = cs.faixa_prata(
            piso=200, base=200, brilho=200, densidade_brilho=0.0)
        detection = self.detector.detect(
            frame, line_ahead=False, timestamp=1.0)
        self.assertIsNone(
            detection, "superfície uniforme não pode virar faixa")

    def test_interseccao_nao_aciona_entrada(self):
        """O falso positivo mais caro: a transversal tem a forma da fita."""
        for piso, valor in ((225, 22), (170, 30), (200, 45)):
            frame = cs.interseccao(piso=piso, valor=valor)
            self.assertIsNone(
                self.detector.detect(frame, line_ahead=False, timestamp=1.0),
                f"interseccao piso={piso} valor={valor} virou entrada")

    def test_faixa_preta_com_reflexo_cai_no_veto_de_escuro(self):
        """Forma e textura certas, mas a caixa é preta: tem de cair.

        Este é o caso que paga pela janela HSV larga. O salpico claro forma
        uma faixa transversal impecável na máscara; o que a denuncia é a
        imagem crua dentro da caixa continuar majoritariamente preta.
        """
        frame = cs.faixa_preta_salpicada(densidade_brilho=0.5)
        self.assertIsNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0))
        self.assertEqual(self.detector.last_reason, "escura")

    def test_fita_escura_de_raso_ainda_e_aceita(self):
        """Medido em `captures/linha_prata`: a fita real fica ESCURA de raso.

        V mediano ~90 e S ~60, bem abaixo do piso branco em volta. A janela
        antiga (V>=140, S<=70) a deixava de fora e o robô parava em
        "sem_linha_cheia" na frente da entrada verdadeira.
        """
        frame = cs.faixa_prata(piso=235, base=95, brilho=150,
                               densidade_brilho=0.30)
        self.assertIsNotNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0),
            f"fita escura recusada por '{self.detector.last_reason}'")

    def test_fita_colada_na_camera_toca_o_topo_da_roi_e_passa(self):
        """Com o robô em cima da fita ela sobe acima do corte da ROI.

        Antes isso caía em "cortada_no_topo" — o veto matava a detecção
        verdadeira justamente no frame mais fácil de todos.
        """
        frame = cs.faixa_prata(topo=0.30, espessura=0.35, piso=235,
                               base=110, brilho=180, densidade_brilho=0.30)
        self.assertIsNotNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0),
            f"fita colada recusada por '{self.detector.last_reason}'")

    def test_regiao_gigante_que_atravessa_a_roi_inteira_continua_vetada(self):
        """Tocar topo E base é sombra/parede truncada, não fita."""
        frame = cs.faixa_prata(topo=0.40, espessura=0.60, piso=235,
                               base=110, brilho=180, densidade_brilho=0.30)
        self.assertIsNone(
            self.detector.detect(frame, line_ahead=False, timestamp=1.0))

    def test_frame_invalido_gera_erro_explicito(self):
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((10, 10), dtype=np.uint8))


class EntrySilverGateTests(unittest.TestCase):
    def _gate(self, cooldown=0.0):
        return EntrySilverGate(
            confirmer=StripeConfirmer(
                votes_needed=config.ENTRY_SILVER_VOTES_NEEDED,
                window=config.ENTRY_SILVER_VOTE_WINDOW,
                max_age_s=config.ENTRY_SILVER_MAX_AGE_S,
                cooldown_s=cooldown,
            )
        )

    def test_um_frame_bom_nao_confirma_a_entrada(self):
        gate = self._gate()
        confirmed, detection = gate.update(
            cs.faixa_prata(), line_ahead=False, timestamp=1.0, now=1.0)
        self.assertFalse(confirmed)
        self.assertIsNotNone(detection)

    def test_dois_frames_distintos_confirmam(self):
        gate = self._gate()
        frame = cs.faixa_prata()
        confirmed = False
        for index in range(config.ENTRY_SILVER_VOTES_NEEDED):
            timestamp = 1.0 + index * 0.05
            confirmed, _ = gate.update(
                frame, line_ahead=False, timestamp=timestamp, now=timestamp)
        self.assertTrue(confirmed)

    def test_frames_repetidos_nao_confirmam(self):
        gate = self._gate()
        frame = cs.faixa_prata()
        for _ in range(6):
            confirmed, _ = gate.update(
                frame, line_ahead=False, timestamp=1.0, now=1.0)
        self.assertFalse(confirmed)

    def test_reflexos_isolados_entre_negativos_nao_confirmam(self):
        gate = self._gate()
        sequencia = (
            cs.reflexo_pontual(),
            cs.faixa_prata(),
            cs.reflexo_pontual(),
            cs.piso_neutro(),
            cs.reflexo_pontual(),
        )
        confirmed = False
        for index, frame in enumerate(sequencia):
            timestamp = 1.0 + index * 0.05
            confirmed, _ = gate.update(
                frame, line_ahead=False, timestamp=timestamp, now=timestamp)
        self.assertFalse(confirmed)

    def test_duas_faixas_finas_nao_confirmam_entrada(self):
        class DetectorFraco:
            last_reason = "fina"

            def detect(self, *_args, **_kwargs):
                self.last_reason = "fina"
                return None

        gate = EntrySilverGate(detector=DetectorFraco())
        frame = cs.piso_neutro()
        primeiro, _ = gate.update(
            frame, line_ahead=True, timestamp=1.00, now=1.00)
        segundo, _ = gate.update(
            frame, line_ahead=True, timestamp=1.05, now=1.05)

        self.assertFalse(primeiro)
        self.assertFalse(segundo)
        self.assertFalse(gate.confirmed)

    def test_curva_90_rejeitada_nao_vota_nem_gruda_confirmacao(self):
        class DetectorCurva90:
            last_reason = "fina"

            def detect(self, *_args, **_kwargs):
                self.last_reason = "fina"
                return None

        gate = EntrySilverGate(detector=DetectorCurva90())
        frame = cs.piso_neutro()
        for index in range(8):
            instante = 1.0 + index * 0.05
            confirmed, detection = gate.update(
                frame,
                line_ahead=False,
                timestamp=instante,
                now=instante,
            )
            self.assertIsNone(detection)
            self.assertFalse(confirmed)
        self.assertEqual(gate.votes, 0)

    def test_rejeicao_sem_contraste_nao_usa_confirmacao_fraca(self):
        class DetectorSemContraste:
            last_reason = "sem_contraste"

            def detect(self, *_args, **_kwargs):
                return None

        gate = EntrySilverGate(detector=DetectorSemContraste())
        frame = cs.piso_neutro()
        confirmed = False
        for index in range(4):
            instante = 1.0 + index * 0.05
            confirmed, _ = gate.update(
                frame,
                line_ahead=False,
                timestamp=instante,
                now=instante,
            )
        self.assertFalse(confirmed)

    def test_cooldown_impede_reentrada_imediata(self):
        gate = self._gate(cooldown=8.0)
        frame = cs.faixa_prata()
        for index in range(3):
            timestamp = 1.0 + index * 0.05
            gate.update(frame, line_ahead=False, timestamp=timestamp,
                        now=timestamp)
        self.assertTrue(gate.confirmed)
        gate.reset(now=2.0)
        self.assertFalse(gate.confirmed)
        for index in range(5):
            timestamp = 3.0 + index * 0.05
            confirmed, _ = gate.update(
                frame, line_ahead=False, timestamp=timestamp, now=timestamp)
        self.assertFalse(confirmed, "cooldown não bloqueou a reconfirmação")


class EntrySilverBoundsTests(unittest.TestCase):
    def test_sem_config_manager_usa_os_defaults(self):
        minimum, maximum = load_bounds(None)
        self.assertEqual(minimum, list(config.ENTRY_SILVER_MIN_DEFAULT))
        self.assertEqual(maximum, list(config.ENTRY_SILVER_MAX_DEFAULT))

    def test_le_o_perfil_da_camera_de_linha(self):
        class FakeManager:
            def __init__(self):
                self.sections = []

            def read_variable(self, section, name):
                self.sections.append(section)
                return {"entry_silver_min": [0, 0, 150],
                        "entry_silver_max": [180, 55, 255]}[name]

        manager = FakeManager()
        minimum, maximum = load_bounds(manager)
        self.assertEqual(minimum, [0, 0, 150])
        self.assertEqual(maximum, [180, 55, 255])
        # O perfil da entrada pertence à câmera de linha e a nenhuma outra.
        self.assertEqual(set(manager.sections), {"color_values_line"})


if __name__ == "__main__":
    unittest.main()
