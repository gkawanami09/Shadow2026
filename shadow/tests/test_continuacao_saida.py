"""Regressoes da geometria da saida preta da area de resgate."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
import config_resgate as cfg  # noqa: E402
from visao.continuacao_saida import (  # noqa: E402
    DIREITA_BAIXA,
    ESQUERDA_BAIXA,
    NIVEL,
    AnalisadorSaidaPreta,
    detectar_continuacao_saida,
    detectar_soleira,
)


CAPTURA_PRETA_LIMPA = (
    SHADOW_ROOT
    / "captures"
    / "linha_preta"
    / "saida_preta"
    / "Captura de tela 2026-08-05 140225.png"
)


def mascara_soleira(
    inclinacao=0.0,
    com_ramo=False,
    alvo_x=None,
    base_central=175,
    espessura=28,
):
    """Cria uma faixa transversal; inclinacao positiva baixa a direita."""
    altura = config.camera_y
    largura = config.camera_x
    centro_x = largura // 2
    alvo_x = centro_x if alvo_x is None else int(alvo_x)
    mascara = np.zeros((altura, largura), dtype=np.uint8)

    for x in range(largura):
        base = int(round(
            base_central + inclinacao * (x - centro_x)
        ))
        topo = base - espessura
        mascara[max(topo, 0):min(base + 1, altura), x] = 255

    if com_ramo:
        topo_soleira = int(round(
            base_central + inclinacao * (alvo_x - centro_x)
        )) - espessura
        # O ramo toca a borda frontal da faixa e aponta para o horizonte.
        cv2.line(
            mascara,
            (alvo_x, topo_soleira + 2),
            (alvo_x + 15, 20),
            255,
            18,
        )
    return mascara


class DeteccaoSoleiraTests(unittest.TestCase):
    def test_classifica_as_tres_orientacoes(self):
        casos = (
            (0.15, DIREITA_BAIXA),
            (-0.15, ESQUERDA_BAIXA),
            (0.03, NIVEL),
        )

        for inclinacao, esperado in casos:
            mascara = mascara_soleira(
                inclinacao=inclinacao, com_ramo=True
            )
            deteccao = detectar_soleira(mascara)

            with self.subTest(inclinacao=inclinacao, esperado=esperado):
                self.assertIsNotNone(deteccao)
                self.assertEqual(deteccao.orientacao, esperado)
                self.assertGreater(deteccao.cobertura_esquerda, 0.95)
                self.assertGreater(deteccao.cobertura_direita, 0.95)
                if esperado == DIREITA_BAIXA:
                    self.assertGreater(
                        deteccao.delta_y_ratio,
                        cfg.EXIT_POST_LEVEL_DELTA_RATIO,
                    )
                elif esperado == ESQUERDA_BAIXA:
                    self.assertLess(
                        deteccao.delta_y_ratio,
                        -cfg.EXIT_POST_LEVEL_DELTA_RATIO,
                    )
                else:
                    self.assertLessEqual(
                        abs(deteccao.delta_y_ratio),
                        cfg.EXIT_POST_LEVEL_DELTA_RATIO,
                    )

    def test_manchas_centrais_sem_cobertura_lateral_nao_sao_soleira(self):
        mascara = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8
        )
        cv2.rectangle(mascara, (180, 120), (268, 190), 255, -1)

        self.assertIsNone(detectar_soleira(mascara))


class DeteccaoContinuacaoTests(unittest.TestCase):
    def test_separa_ramo_da_soleira_em_todas_as_orientacoes(self):
        alvo_x = config.camera_x // 2
        for inclinacao in (0.15, -0.15, 0.0):
            mascara = mascara_soleira(
                inclinacao=inclinacao,
                com_ramo=True,
                alvo_x=alvo_x,
            )
            soleira = detectar_soleira(mascara)
            continuacao = detectar_continuacao_saida(
                mascara, soleira=soleira
            )

            with self.subTest(inclinacao=inclinacao):
                self.assertIsNotNone(soleira)
                self.assertIsNotNone(continuacao)
                self.assertAlmostEqual(
                    continuacao.alvo_x, alvo_x + 15, delta=25
                )
                self.assertLess(continuacao.alvo_y, config.camera_y * 0.20)
                # A faixa ocupa toda a largura; o componente entregue precisa
                # ser a haste estreita que segue para a frente.
                self.assertLess(
                    continuacao.bbox[2], config.camera_x * 0.20
                )
                self.assertLess(
                    continuacao.bbox[1], soleira.bbox[1]
                )

    def test_soleira_diagonal_isolada_nao_vira_falsa_continuacao(self):
        mascara = mascara_soleira(inclinacao=0.15, com_ramo=False)
        soleira = detectar_soleira(mascara)

        self.assertIsNotNone(soleira)
        self.assertEqual(soleira.orientacao, DIREITA_BAIXA)
        self.assertIsNone(
            detectar_continuacao_saida(mascara, soleira=soleira)
        )

    def test_soleira_em_perspectiva_nao_deixa_cunha_como_ramo(self):
        casos = (
            (200.0, 0.30, 40.0, 60.0),
            (170.0, 0.45, 30.0, 45.0),
        )
        largura = config.camera_x
        altura = config.camera_y
        centro = largura / 2.0
        for base_central, inclinacao, esp_esq, esp_dir in casos:
            mascara = np.zeros((altura, largura), dtype=np.uint8)
            for x in range(largura):
                fracao = x / max(float(largura - 1), 1.0)
                espessura = esp_esq + (esp_dir - esp_esq) * fracao
                base = base_central + inclinacao * (x - centro)
                topo = int(round(base - espessura))
                base = int(round(base))
                mascara[max(topo, 0):min(base + 1, altura), x] = 255

            soleira = detectar_soleira(mascara)
            with self.subTest(
                base=base_central,
                inclinacao=inclinacao,
                espessuras=(esp_esq, esp_dir),
            ):
                self.assertIsNotNone(soleira)
                self.assertIsNone(
                    detectar_continuacao_saida(mascara, soleira=soleira)
                )

    def test_linha_desconectada_da_soleira_nao_e_o_ramo_dela(self):
        mascara = mascara_soleira(inclinacao=0.0, com_ramo=False)
        cv2.line(
            mascara,
            (config.camera_x // 2, 70),
            (config.camera_x // 2 + 10, 20),
            255,
            18,
        )
        soleira = detectar_soleira(mascara)

        self.assertIsNotNone(soleira)
        self.assertIsNone(
            detectar_continuacao_saida(mascara, soleira=soleira)
        )

    def test_linha_longitudinal_isolada_e_aceita_apos_sumir_a_soleira(self):
        mascara = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8
        )
        centro = config.camera_x // 2
        cv2.line(
            mascara,
            (centro, config.camera_y - 1),
            (centro + 18, 25),
            255,
            18,
        )

        continuacao = detectar_continuacao_saida(mascara, soleira=None)

        self.assertIsNotNone(continuacao)
        self.assertAlmostEqual(continuacao.alvo_x, centro + 18, delta=22)
        self.assertLess(continuacao.alvo_y, config.camera_y * 0.20)

    def test_faixa_horizontal_isolada_nao_e_linha_longitudinal(self):
        mascara = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8
        )
        cv2.line(
            mascara,
            (25, 175),
            (config.camera_x - 25, 175),
            255,
            22,
        )

        self.assertIsNone(
            detectar_continuacao_saida(mascara, soleira=None)
        )

    def test_soleira_diagonal_parcial_nao_vira_linha_longitudinal(self):
        casos = (
            ((0, 100), (config.camera_x - 1, 360)),
            ((0, 170), (config.camera_x - 1, 330)),
        )
        for inicio, fim in casos:
            mascara = np.zeros(
                (config.camera_y, config.camera_x), dtype=np.uint8
            )
            cv2.line(mascara, inicio, fim, 255, 28)

            with self.subTest(inicio=inicio, fim=fim):
                self.assertIsNone(detectar_soleira(mascara))
                self.assertIsNone(
                    detectar_continuacao_saida(mascara, soleira=None)
                )

    def test_captura_preta_real_tem_soleira_nivel_e_terceira_linha(self):
        self.assertTrue(
            CAPTURA_PRETA_LIMPA.is_file(),
            f"captura limpa ausente: {CAPTURA_PRETA_LIMPA}",
        )
        frame = cv2.imread(str(CAPTURA_PRETA_LIMPA), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)

        analise = AnalisadorSaidaPreta().analisar(frame)

        self.assertIsNotNone(analise.soleira)
        self.assertEqual(analise.soleira.orientacao, NIVEL)
        self.assertLess(
            abs(analise.soleira.delta_y_ratio),
            cfg.EXIT_POST_LEVEL_DELTA_RATIO,
        )
        self.assertIsNotNone(analise.continuacao)
        self.assertAlmostEqual(
            analise.continuacao.alvo_x,
            config.camera_x * 0.54,
            delta=config.camera_x * 0.08,
        )
        self.assertLess(
            analise.continuacao.alvo_y, config.camera_y * 0.10
        )


if __name__ == "__main__":
    unittest.main()
