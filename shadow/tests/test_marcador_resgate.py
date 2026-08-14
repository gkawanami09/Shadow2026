"""Testes do detector isolado dos marcadores de deposito."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from visao.marcador_resgate import (  # noqa: E402
    GreenRectangleDetector,
    MarkerDetector,
    color_masks,
)


HEIGHT = 480
WIDTH = 640
BACKGROUND = (145, 145, 145)
GREEN = (25, 220, 25)
RED = (25, 25, 220)


def base_frame(color=BACKGROUND):
    return np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)


def triangle_frame(
    kind,
    points=((320, 190), (230, 400), (410, 400)),
    background=BACKGROUND,
):
    frame = base_frame(background)
    color = GREEN if kind == "green" else RED
    cv2.fillPoly(
        frame,
        [np.asarray(points, dtype=np.int32)],
        color,
        lineType=cv2.LINE_AA,
    )
    return frame


def confirmed(detector, frame, start=0.0):
    result = None
    for index in range(cfg.MARKER_ACQUIRE_HITS):
        result = detector.detect(frame, timestamp=start + index * 0.1)
    return result


class MarkerColorMaskTests(unittest.TestCase):
    def test_color_masks_expose_green_and_both_red_hue_bands(self):
        hsv = np.zeros((80, 240, 3), dtype=np.uint8)
        hsv[:, 0:80] = (60, 230, 220)
        hsv[:, 80:160] = (3, 230, 220)
        hsv[:, 160:240] = (176, 230, 220)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        masks = color_masks(frame)

        self.assertEqual(set(masks), {"green", "red"})
        self.assertGreater(np.mean(masks["green"][:, 0:80]), 250)
        self.assertEqual(np.count_nonzero(masks["green"][:, 80:]), 0)
        self.assertGreater(np.mean(masks["red"][:, 80:160]), 250)
        self.assertGreater(np.mean(masks["red"][:, 160:240]), 250)
        self.assertEqual(np.count_nonzero(masks["red"][:, 0:80]), 0)

    def test_color_masks_are_raw_and_do_not_apply_lower_roi(self):
        frame = base_frame()
        cv2.rectangle(frame, (20, 5), (100, 80), GREEN, -1)

        masks = color_masks(frame)

        self.assertGreater(np.count_nonzero(masks["green"][:100]), 0)

    def test_laranja_nao_entra_na_mascara_vermelha(self):
        hsv = np.zeros((40, 80, 3), dtype=np.uint8)
        hsv[:, :40] = (3, 220, 180)
        hsv[:, 40:] = (14, 220, 180)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        red = color_masks(frame)["red"]

        self.assertGreater(np.mean(red[:, :40]), 250)
        self.assertEqual(np.count_nonzero(red[:, 40:]), 0)

    def test_invalid_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            color_masks(np.zeros((40, 40), dtype=np.uint8))


class MarkerDetectorTests(unittest.TestCase):
    def test_requires_green_or_red_target(self):
        for invalid in (None, "silver", "any", "GREEN"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    MarkerDetector(invalid)

    def test_green_triangle_requires_three_distinct_timestamps(self):
        detector = MarkerDetector("green")
        frame = triangle_frame("green")

        first = detector.detect(frame, timestamp=1.0)
        repeated = detector.detect(frame, timestamp=1.0)
        second = detector.detect(frame, timestamp=1.1)
        third = detector.detect(frame, timestamp=1.2)

        self.assertEqual(first.kind, "green")
        self.assertEqual(first.hits, 1)
        self.assertFalse(first.confirmed)
        self.assertEqual(repeated.hits, 1)
        self.assertFalse(repeated.confirmed)
        self.assertEqual(second.hits, 2)
        self.assertTrue(third.confirmed)
        self.assertTrue(third.track_locked)
        self.assertEqual(third.hits, cfg.MARKER_ACQUIRE_HITS)
        self.assertAlmostEqual(third.center_x, 320, delta=3)
        self.assertAlmostEqual(third.bottom_y, 400, delta=3)
        self.assertEqual(len(third.bbox), 4)
        self.assertGreater(third.width, 0)
        self.assertGreater(third.height, 0)
        self.assertGreater(third.area, 0)
        self.assertAlmostEqual(third.horizontal_error(WIDTH), 0.0, delta=0.02)
        self.assertGreater(third.confidence, cfg.MARKER_MIN_CONFIDENCE)

    def test_red_triangle_is_confirmed_in_low_hue_band(self):
        result = confirmed(
            MarkerDetector("red"),
            triangle_frame("red"),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "red")
        self.assertTrue(result.confirmed)
        self.assertTrue(result.track_locked)

    def test_red_triangle_is_confirmed_in_high_hue_band(self):
        hsv_color = np.uint8([[[176, 230, 220]]])
        bgr_color = tuple(
            int(value)
            for value in cv2.cvtColor(
                hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        )
        frame = base_frame()
        cv2.fillPoly(
            frame,
            [np.asarray(
                ((320, 190), (230, 400), (410, 400)),
                dtype=np.int32,
            )],
            bgr_color,
            lineType=cv2.LINE_AA,
        )

        result = confirmed(MarkerDetector("red"), frame)

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)

    def test_red_triangle_with_less_saturation_is_confirmed(self):
        frame = base_frame()
        cv2.fillPoly(
            frame,
            [np.asarray(((320, 190), (230, 400), (410, 400)),
                        dtype=np.int32)],
            (75, 75, 145),
            lineType=cv2.LINE_AA,
        )

        result = confirmed(MarkerDetector("red"), frame)

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)

    def test_red_triangle_faded_and_distant_is_confirmed(self):
        """O vermelho pequeno nao pode exigir a saturacao do alvo proximo."""
        frame = base_frame()
        cv2.fillPoly(
            frame,
            [np.asarray(((320, 225), (313, 255), (327, 255)),
                        dtype=np.int32)],
            (85, 85, 140),
            lineType=cv2.LINE_AA,
        )

        result = confirmed(MarkerDetector("red"), frame)

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertTrue(result.track_locked)

    def test_red_triangle_distant_above_generic_roi_is_confirmed(self):
        """O vermelho distante nao pode depender da janela do verde."""
        frame = triangle_frame(
            "red",
            # Todo o alvo fica acima dos 45% usados pelo verde e a area fica
            # abaixo do minimo normal. Ainda e vermelho muito saturado e deve
            # atravessar as tres confirmacoes temporais.
            points=((320, 146), (312, 191), (328, 191)),
        )

        red = confirmed(MarkerDetector("red"), frame)
        green = confirmed(MarkerDetector("green"), frame)

        self.assertIsNotNone(red)
        self.assertTrue(red.confirmed)
        self.assertLess(
            red.area / float(WIDTH * HEIGHT),
            0.0015,
        )
        self.assertIsNone(green)

    def test_green_triangle_distant_is_confirmed_before_middle_of_frame(self):
        """O verde distante entra antes da metade sem perder os tres frames."""
        frame = triangle_frame(
            "green",
            points=((320, 146), (312, 191), (328, 191)),
        )

        result = confirmed(MarkerDetector("green"), frame)

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertLess(
            result.area / float(WIDTH * HEIGHT),
            0.0015,
        )

    def test_triangles_small_and_distant_are_confirmed_after_three_frames(self):
        """A deteccao distante nao pode exigir que o alvo ja esteja perto.

        O tamanho fica abaixo do antigo minimo verde. A cor ainda e intensa,
        o entorno e neutro e a confirmacao temporal continua obrigatoria.
        """
        for kind, points in (
            ("red", ((320, 170), (312, 202), (328, 202))),
        ):
            with self.subTest(kind=kind):
                detector = MarkerDetector(kind)
                frame = triangle_frame(kind, points=points)
                first = detector.detect(frame, timestamp=0.0)
                result = confirmed(detector, frame, start=0.1)

                self.assertIsNotNone(first)
                self.assertFalse(first.confirmed)
                self.assertIsNotNone(result)
                self.assertTrue(result.confirmed)
                self.assertLess(
                    result.area / float(WIDTH * HEIGHT),
                    0.0015,
                )

    def test_wrong_target_color_is_never_returned(self):
        cases = (
            ("green", triangle_frame("red")),
            ("red", triangle_frame("green")),
        )
        for target, frame in cases:
            with self.subTest(target=target):
                detector = MarkerDetector(target)
                for index in range(4):
                    self.assertIsNone(
                        detector.detect(frame, timestamp=index * 0.1))
                self.assertFalse(detector.last_candidates)

    def test_banho_de_cor_uniforme_e_rejeitado(self):
        """Quadro inteiro da cor: sem contraste com o entorno, não é marcador."""
        for target, color in (("green", GREEN), ("red", RED)):
            with self.subTest(target=target):
                detector = MarkerDetector(target)
                self.assertIsNone(
                    detector.detect(base_frame(color), timestamp=0.0))
                self.assertFalse(detector.last_candidates)

    def test_retangulo_saturado_e_aceito_agora(self):
        """É assim que o marcador REAL aparece nesta câmera.

        Medido: o triângulo de depósito, visto quase rente ao piso, tem
        silhueta retangular. Aceitar isto é o objetivo da mudança — antes
        ele era rejeitado por triangularidade e o depósito era impossível.
        """
        for target, color in (("green", GREEN), ("red", RED)):
            with self.subTest(target=target):
                frame = base_frame()
                cv2.rectangle(frame, (180, 260), (460, 410), color, -1)
                detector = MarkerDetector(target)
                self.assertIsNotNone(
                    detector.detect(frame, timestamp=0.0),
                    "o marcador real do robô tem esta silhueta")

    def test_uniform_cyan_wash_is_not_a_green_marker(self):
        # O fundo inteiro tem componente verde/ciano e saturacao suficiente
        # para atingir parte da banda HSV, mas nao possui contraste nem forma.
        frame = base_frame((190, 215, 135))
        detector = MarkerDetector("green")

        result = detector.detect(frame, timestamp=0.0)

        self.assertIsNone(result)
        self.assertFalse(detector.last_candidates)


class GreenRectangleDetectorTests(unittest.TestCase):
    def test_verde_ciano_medido_na_camera_real_e_confirmado(self):
        # Mediana medida no painel verde das capturas da camera de resgate.
        # O azul fica acima do verde, portanto G-max(B,R) seria negativo.
        frame = base_frame()
        cv2.rectangle(frame, (80, 300), (560, 430), (131, 110, 31), -1)
        detector = GreenRectangleDetector()

        resultado = None
        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS):
            resultado = detector.detect(frame, timestamp=indice * 0.1)

        self.assertIsNotNone(resultado)
        self.assertTrue(resultado.confirmed)
        self.assertTrue(resultado.track_locked)

    def test_parede_com_banho_verde_fraco_nao_vira_retangulo(self):
        # Mediana do falso componente que aparecia na parede: B=98, G=139,
        # R=109. A cor e verde, mas fraca demais para ser o painel.
        frame = base_frame((98, 139, 109))
        detector = GreenRectangleDetector()

        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS + 1):
            self.assertIsNone(
                detector.detect(frame, timestamp=indice * 0.1))

    def test_objeto_ciano_vertical_nao_vira_retangulo(self):
        frame = base_frame()
        cv2.rectangle(frame, (290, 250), (340, 450), (131, 110, 31), -1)
        detector = GreenRectangleDetector()

        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS + 1):
            self.assertIsNone(
                detector.detect(frame, timestamp=indice * 0.1))

    def test_objeto_ciano_largo_acima_do_piso_nao_vira_deposito(self):
        """Replica o falso verde largo visto no debug da busca."""
        frame = base_frame()
        # E' horizontal, saturado e ciano como o painel real, mas termina
        # acima do chao. Nao pode travar a navegacao para o deposito.
        cv2.rectangle(frame, (0, 215), (340, 340), (131, 110, 31), -1)
        detector = GreenRectangleDetector()

        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS + 1):
            self.assertIsNone(
                detector.detect(frame, timestamp=indice * 0.1))

    def test_verde_ciano_moderado_e_confirmado_pelo_retangulo(self):
        hsv = np.uint8([[[85, 130, 180]]])
        cor_lavada = tuple(
            int(valor)
            for valor in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        )
        frame = base_frame()
        cv2.rectangle(frame, (120, 250), (520, 420), cor_lavada, -1)

        antigo = MarkerDetector("green")
        self.assertIsNone(antigo.detect(frame, timestamp=0.0))

        novo = GreenRectangleDetector()
        resultado = None
        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS):
            resultado = novo.detect(frame, timestamp=indice * 0.1)

        self.assertIsNotNone(resultado)
        self.assertTrue(resultado.confirmed)
        self.assertTrue(resultado.track_locked)
        self.assertEqual(resultado.kind, "green")

    def test_painel_verde_ciano_escuro_e_confirmado(self):
        # Painel escuro visto na arena: ainda e verde-ciano, mas sua
        # cromaticidade e menor que o painel iluminado usado antigamente.
        frame = base_frame()
        cv2.rectangle(frame, (100, 270), (540, 440), (65, 75, 45), -1)
        detector = GreenRectangleDetector()

        resultado = None
        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS):
            resultado = detector.detect(frame, timestamp=indice * 0.1)

        self.assertIsNotNone(resultado)
        self.assertTrue(resultado.confirmed)

    def test_placa_clara_azulada_da_captura_nao_vira_verde(self):
        frame = base_frame()
        cv2.rectangle(
            frame, (150, 250), (520, 430), (192, 198, 143), -1)
        detector = GreenRectangleDetector()

        for indice in range(cfg.GREEN_RECTANGLE_ACQUIRE_HITS + 1):
            self.assertIsNone(
                detector.detect(frame, timestamp=indice * 0.1))

    def test_um_frame_verde_nao_confirma_retangulo(self):
        frame = base_frame()
        cv2.rectangle(frame, (180, 260), (460, 410), GREEN, -1)

        resultado = GreenRectangleDetector().detect(
            frame, timestamp=0.0)

        self.assertIsNotNone(resultado)
        self.assertFalse(resultado.confirmed)
        self.assertFalse(resultado.track_locked)

    def test_timestamp_repetido_nao_aumenta_hits(self):
        frame = base_frame()
        cv2.rectangle(frame, (180, 260), (460, 410), GREEN, -1)
        detector = GreenRectangleDetector()

        primeiro = detector.detect(frame, timestamp=1.0)
        repetido = detector.detect(frame, timestamp=1.0)

        self.assertEqual(primeiro.hits, 1)
        self.assertEqual(repetido.hits, 1)

    def test_vermelho_nao_passa_no_detector_verde(self):
        frame = base_frame()
        cv2.rectangle(frame, (180, 260), (460, 410), RED, -1)
        detector = GreenRectangleDetector()

        for indice in range(4):
            self.assertIsNone(
                detector.detect(frame, timestamp=indice * 0.1))

    def test_banho_ciano_uniforme_nao_vira_retangulo_verde(self):
        frame = base_frame((190, 215, 135))
        detector = GreenRectangleDetector()

        for indice in range(4):
            self.assertIsNone(
                detector.detect(frame, timestamp=indice * 0.1))

    def test_objeto_verde_acima_da_roi_nao_comanda(self):
        frame = base_frame()
        cv2.rectangle(frame, (180, 130), (460, 200), GREEN, -1)

        resultado = GreenRectangleDetector().detect(
            frame, timestamp=0.0)

        self.assertIsNone(resultado)


class MarkerDetectorRegressionTests(unittest.TestCase):
    def test_green_triangle_survives_cyan_illumination_wash(self):
        frame = triangle_frame(
            "green",
            background=(190, 215, 135),
        )

        result = confirmed(MarkerDetector("green"), frame)

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)

    def test_circulo_saturado_e_uma_limitacao_conhecida(self):
        """LIMITAÇÃO ASSUMIDA: forma não separa marcador de círculo aqui.

        Medido no pipeline, nesta câmera quase rente ao piso: o marcador
        real dá triangularidade 0.577, um círculo dá 0.605 e a cadeira
        vermelha do laboratório dá 0.677 — a cadeira é MAIS triangular que
        o marcador. Nenhum limiar aceita o marcador e rejeita o círculo.

        O gate de forma foi removido de propósito e o rigor passou para a
        cromaticidade. A consequência é esta: um círculo MUITO saturado, no
        chão, dentro da ROI, seria aceito como marcador.

        Este teste existe para que a limitação fique registrada e visível —
        se um dia a câmera subir e a forma voltar a discriminar, ele é o
        lembrete de que dá para reapertar o gate.
        """
        for target, color in (("green", GREEN), ("red", RED)):
            with self.subTest(target=target):
                frame = base_frame()
                cv2.circle(
                    frame, (320, 320), 90, color, -1, cv2.LINE_AA)
                detector = MarkerDetector(target)
                resultado = detector.detect(frame, timestamp=0.0)
                self.assertIsNotNone(
                    resultado,
                    "se isto voltar a rejeitar, o gate de forma retornou e "
                    "o marcador real do robô será perdido de novo")

    def test_cor_fraca_continua_sendo_rejeitada(self):
        """O que AINDA protege: cromaticidade, não forma.

        Mesmo círculo, mas com cor lavada — como a cadeira vermelha do
        laboratório (cromaticidade 63-79 contra 124-148 do marcador).
        """
        for target, color in (("green", (150, 178, 150)),
                              ("red", (150, 150, 178))):
            with self.subTest(target=target):
                frame = base_frame()
                cv2.circle(
                    frame, (320, 320), 90, color, -1, cv2.LINE_AA)
                detector = MarkerDetector(target)
                self.assertIsNone(
                    detector.detect(frame, timestamp=0.0),
                    "objeto de cor fraca não pode virar marcador")

    def test_triangle_above_lower_roi_is_rejected(self):
        frame = triangle_frame(
            "green",
            points=((320, 5), (270, 90), (370, 90)),
        )
        detector = MarkerDetector("green")

        result = detector.detect(frame, timestamp=0.0)

        self.assertIsNone(result)

    def test_detection_scales_between_640_and_320(self):
        full = triangle_frame("green")
        half = cv2.resize(
            full, (320, 240), interpolation=cv2.INTER_AREA)
        full_result = confirmed(MarkerDetector("green"), full)
        half_result = confirmed(MarkerDetector("green"), half)

        self.assertIsNotNone(full_result)
        self.assertIsNotNone(half_result)
        self.assertTrue(full_result.confirmed)
        self.assertTrue(half_result.confirmed)
        self.assertAlmostEqual(
            full_result.center_x / WIDTH,
            half_result.center_x / 320,
            delta=0.012,
        )
        self.assertAlmostEqual(
            full_result.center_y / HEIGHT,
            half_result.center_y / 240,
            delta=0.012,
        )
        self.assertAlmostEqual(
            full_result.bottom_y / HEIGHT,
            half_result.bottom_y / 240,
            delta=0.012,
        )
        self.assertAlmostEqual(
            full_result.confidence,
            half_result.confidence,
            delta=0.10,
        )

    def test_two_targets_do_not_replace_locked_spatial_target(self):
        left_points = ((170, 220), (110, 390), (230, 390))
        left_only = triangle_frame("green", points=left_points)
        detector = MarkerDetector("green")
        locked = confirmed(detector, left_only)
        self.assertTrue(locked.track_locked)
        self.assertLess(locked.center_x, WIDTH / 2)

        both = base_frame()
        cv2.fillPoly(
            both,
            [np.asarray(left_points, dtype=np.int32)],
            GREEN,
            lineType=cv2.LINE_AA,
        )
        # O triangulo da direita e maior e mais facil, mas esta fora do gate
        # do track esquerdo ja confirmado.
        cv2.fillPoly(
            both,
            [np.asarray(
                ((480, 150), (370, 430), (600, 430)),
                dtype=np.int32,
            )],
            GREEN,
            lineType=cv2.LINE_AA,
        )

        result = detector.detect(both, timestamp=0.4)

        self.assertIsNotNone(result)
        self.assertTrue(result.track_locked)
        self.assertLess(result.center_x, WIDTH / 2)
        self.assertGreaterEqual(len(detector.last_candidates), 2)

    def test_locked_target_is_not_stolen_by_distant_candidate(self):
        left_points = ((170, 220), (110, 390), (230, 390))
        detector = MarkerDetector("red")
        locked = confirmed(
            detector,
            triangle_frame("red", points=left_points),
        )
        self.assertTrue(locked.track_locked)

        right_only = triangle_frame(
            "red",
            points=((500, 190), (410, 420), (590, 420)),
        )
        first_miss = detector.detect(right_only, timestamp=0.4)
        second_miss = detector.detect(right_only, timestamp=0.5)

        self.assertIsNone(first_miss)
        self.assertIsNone(second_miss)


class MarkerEdgeAndArrivalTests(unittest.TestCase):
    """Blob encostado na borda lateral do quadro.

    Regras medidas nas capturas reais da arena. Com a câmera não mirada, só a
    ponta do triângulo aparece e a forma desse pedaço não descreve triângulo
    nenhum; já quando o robô chega, o marcador ocupa o quadro inteiro e julgar
    forma perderia o alvo justamente na hora de depositar.
    """

    def test_fragmento_na_borda_e_rejeitado_como_incompleto(self):
        for kind in ("green", "red"):
            frame = base_frame()
            color = GREEN if kind == "green" else RED
            # Pedaço pequeno cortado pela borda esquerda.
            cv2.rectangle(frame, (0, 300), (110, 430), color, -1)
            detector = MarkerDetector(kind)
            self.assertIsNone(detector.detect(frame, timestamp=0.0))
            self.assertIn("incompleto", detector.last_rejections)

    def test_marcador_de_chegada_nao_e_perdido_por_forma(self):
        """Ocupa o quadro e encosta nas duas bordas: é a chegada."""
        for kind in ("green", "red"):
            frame = base_frame()
            color = GREEN if kind == "green" else RED
            cv2.rectangle(frame, (0, 300), (WIDTH, 430), color, -1)
            detection = MarkerDetector(kind).detect(frame, timestamp=0.0)
            self.assertIsNotNone(
                detection,
                f"marcador {kind} de chegada foi perdido pela geometria")
            self.assertEqual(detection.kind, kind)

    def test_chegada_ainda_exige_cromaticidade(self):
        """A rota de chegada dispensa forma, nunca a cor."""
        frame = base_frame()
        # Cinza levemente esverdeado: forma de chegada, cromaticidade baixa.
        cv2.rectangle(frame, (0, 300), (WIDTH, 430), (150, 175, 150), -1)
        detector = MarkerDetector("green")
        self.assertIsNone(detector.detect(frame, timestamp=0.0))

    def test_marcador_inteiro_no_quadro_nao_e_fragmento(self):
        """Sem tocar a borda, ele é julgado normalmente — e aceito."""
        frame = base_frame()
        cv2.rectangle(frame, (200, 300), (440, 430), RED, -1)
        detector = MarkerDetector("red")
        self.assertIsNotNone(detector.detect(frame, timestamp=0.0))
        self.assertNotIn("incompleto", detector.last_rejections)

    def test_filamento_fino_e_rejeitado_por_compacidade(self):
        """A sanidade de forma que SOBROU: exclui filamento e borda."""
        frame = base_frame()
        cv2.line(frame, (120, 380), (520, 300), RED, 6)
        detector = MarkerDetector("red")
        self.assertIsNone(detector.detect(frame, timestamp=0.0))


if __name__ == "__main__":
    unittest.main()
