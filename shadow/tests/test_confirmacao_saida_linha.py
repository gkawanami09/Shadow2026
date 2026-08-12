"""Regressoes da confirmacao preto/prata feita pela camera de linha."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
import config_resgate as cfg  # noqa: E402
from visao.confirmacao_saida_linha import (  # noqa: E402
    INCONCLUSIVA,
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
    ConfirmadorFaixaSaidaLinha,
    faixa_centralizada,
    posicao_vertical_faixa,
)


# Somente estas duas imagens nao possuem contornos/texto de debug. Manter os
# caminhos explicitos impede que um glob passe a usar, sem querer, uma imagem
# que ja foi anotada pelo visualizador.
CAPTURA_PRETA_LIMPA = (
    SHADOW_ROOT
    / "captures"
    / "linha_preta"
    / "saida_preta"
    / "Captura de tela 2026-08-05 140225.png"
)
CAPTURA_PRATA_LIMPA = (
    SHADOW_ROOT / "captures" / "saida_prata" / "prata4.png"
)


def cena_preta(topo=95, base=165, brilho_piso=205):
    """Soleira preta com uma terceira linha no centro da imagem."""
    frame = np.full(
        (config.camera_y, config.camera_x, 3),
        brilho_piso,
        dtype=np.uint8,
    )
    frame[:topo, 200:248] = 35
    frame[topo:base, :] = 35
    return frame


def cena_prata(topo=85, base=165, brilho_piso=205):
    """Soleira prata texturizada e uma linha preta longitudinal central."""
    frame = np.full(
        (config.camera_y, config.camera_x, 3),
        brilho_piso,
        dtype=np.uint8,
    )
    yy, xx = np.indices((base - topo, config.camera_x))
    textura = 95 + ((xx // 4 + yy // 3) % 2) * 65
    frame[topo:base, :, 0] = textura
    frame[topo:base, :, 1] = textura
    frame[topo:base, :, 2] = textura
    frame[:topo, 200:248] = 35
    return frame


def mudar_luz(frame, fator):
    """Simula uma mudanca uniforme de iluminacao sem alterar a geometria."""
    return np.clip(
        frame.astype(np.float32) * float(fator), 0, 255
    ).astype(np.uint8)


def mudar_luz_lateral(frame, fator_esquerda, fator_direita):
    gradiente = np.linspace(
        fator_esquerda,
        fator_direita,
        frame.shape[1],
        dtype=np.float32,
    )[None, :, None]
    return np.clip(
        frame.astype(np.float32) * gradiente, 0, 255
    ).astype(np.uint8)


def aplicar_gamma(frame, gamma):
    tabela = np.clip(
        (np.arange(256, dtype=np.float32) / 255.0) ** float(gamma)
        * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return cv2.LUT(frame, tabela)


def aplicar_sombra_local(frame, profundidade=0.65):
    frame = cv2.resize(
        frame,
        (config.camera_x, config.camera_y),
        interpolation=cv2.INTER_AREA,
    )
    y = np.arange(config.camera_y, dtype=np.float32)
    ganho = 1.0 - float(profundidade) * np.exp(
        -0.5 * ((y - 125.0) / 50.0) ** 2
    )
    return np.clip(
        frame.astype(np.float32) * ganho[:, None, None],
        0,
        255,
    ).astype(np.uint8)


def inclinar(frame, delta_y):
    """Inclina a faixa em ``delta_y`` pixels entre os lados da imagem."""
    inclinacao = float(delta_y) / float(config.camera_x - 1)
    matriz = np.asarray([
        [1.0, 0.0, 0.0],
        [inclinacao, 1.0, -inclinacao * config.camera_x / 2.0],
    ], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matriz,
        (config.camera_x, config.camera_y),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(205, 205, 205),
    )


def ler_captura_limpa(caminho):
    if not caminho.is_file():
        raise AssertionError(f"captura limpa ausente: {caminho}")
    frame = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if frame is None:
        raise AssertionError(f"OpenCV nao conseguiu ler: {caminho}")
    return frame


class ClassificadorTests(unittest.TestCase):
    def test_cenas_sinteticas_separam_preto_e_prata(self):
        classificador = ClassificadorFaixaSaidaLinha()

        preta = classificador.classificar(cena_preta())
        prata = classificador.classificar(cena_prata())

        self.assertEqual(preta.classificacao, PRETA)
        self.assertEqual(prata.classificacao, NAO_PRETA)
        self.assertTrue(preta.faixa_presente)
        self.assertTrue(prata.faixa_presente)
        self.assertLess(preta.brilho_relativo, prata.brilho_relativo)
        self.assertGreater(preta.preenchimento_escuro, 0.90)
        self.assertLess(prata.preenchimento_escuro, 0.10)

    def test_classificacao_relativa_resiste_a_luz_uniforme(self):
        classificador = ClassificadorFaixaSaidaLinha()
        for cena, esperado in (
            (cena_preta(), PRETA),
            (cena_prata(), NAO_PRETA),
        ):
            for fator in (0.50, 0.75, 1.00, 1.15):
                with self.subTest(esperado=esperado, fator=fator):
                    resultado = classificador.classificar(
                        mudar_luz(cena, fator)
                    )
                    self.assertEqual(resultado.classificacao, esperado)

    def test_classificacao_resiste_a_luz_muito_desigual_entre_os_lados(self):
        classificador = ClassificadorFaixaSaidaLinha()
        for cena, esperado in (
            (cena_preta(), PRETA),
            (cena_prata(), NAO_PRETA),
        ):
            for fatores in ((0.45, 1.20), (1.20, 0.45)):
                with self.subTest(esperado=esperado, fatores=fatores):
                    resultado = classificador.classificar(
                        mudar_luz_lateral(cena, *fatores))
                    self.assertEqual(resultado.classificacao, esperado)

    def test_cor_nao_muda_com_soleira_fortemente_diagonal(self):
        classificador = ClassificadorFaixaSaidaLinha()
        for cena, esperado in (
            (cena_preta(), PRETA),
            (cena_prata(), NAO_PRETA),
        ):
            for delta_y in (-100, -70, 70, 100):
                with self.subTest(esperado=esperado, delta_y=delta_y):
                    resultado = classificador.classificar(
                        inclinar(cena, delta_y))
                    self.assertEqual(resultado.classificacao, esperado)
                    self.assertTrue(faixa_centralizada(resultado))

    def test_prata_texturizada_escura_nao_vira_preta(self):
        frame = cena_prata()
        frame[85:165] = np.clip(
            frame[85:165].astype(np.float32) * 0.42,
            0,
            255,
        ).astype(np.uint8)

        resultado = ClassificadorFaixaSaidaLinha().classificar(frame)

        self.assertLess(
            resultado.brilho_relativo,
            cfg.EXIT_LINE_VERIFY_BLACK_BRIGHTNESS_RATIO_MAX,
        )
        self.assertGreater(
            resultado.textura_relativa,
            cfg.EXIT_LINE_VERIFY_SILVER_TEXTURE_RATIO_MIN,
        )
        self.assertEqual(resultado.classificacao, NAO_PRETA)

    def test_piso_vazio_e_inconclusivo(self):
        vazio = np.full(
            (config.camera_y, config.camera_x, 3), 205, dtype=np.uint8
        )

        resultado = ClassificadorFaixaSaidaLinha().classificar(vazio)

        self.assertEqual(resultado.classificacao, INCONCLUSIVA)
        self.assertFalse(resultado.faixa_presente)
        self.assertIsNone(posicao_vertical_faixa(resultado))

    def test_centralizacao_depende_da_posicao_vertical_da_faixa(self):
        classificador = ClassificadorFaixaSaidaLinha()
        central = classificador.classificar(cena_preta())
        distante = classificador.classificar(cena_preta(topo=28, base=78))

        self.assertAlmostEqual(
            posicao_vertical_faixa(central), 0.51, delta=0.03
        )
        self.assertTrue(faixa_centralizada(central))
        self.assertLess(posicao_vertical_faixa(distante), 0.30)
        self.assertFalse(faixa_centralizada(distante))

    def test_duas_capturas_reais_limpas_ficam_separadas(self):
        classificador = ClassificadorFaixaSaidaLinha()
        casos = (
            (CAPTURA_PRETA_LIMPA, PRETA),
            (CAPTURA_PRATA_LIMPA, NAO_PRETA),
        )

        for caminho, esperado in casos:
            frame = ler_captura_limpa(caminho)
            with self.subTest(captura=caminho.name, esperado=esperado):
                resultado = classificador.classificar(frame)
                self.assertEqual(resultado.classificacao, esperado)
                self.assertTrue(resultado.faixa_presente)
                self.assertTrue(faixa_centralizada(resultado))

    def test_capturas_reais_preservam_cor_com_variacao_uniforme_de_luz(self):
        classificador = ClassificadorFaixaSaidaLinha()
        casos = (
            (ler_captura_limpa(CAPTURA_PRETA_LIMPA), PRETA),
            (ler_captura_limpa(CAPTURA_PRATA_LIMPA), NAO_PRETA),
        )

        for frame, esperado in casos:
            for fator in (0.50, 0.70, 1.00, 1.15):
                with self.subTest(esperado=esperado, fator=fator):
                    resultado = classificador.classificar(
                        mudar_luz(frame, fator)
                    )
                    self.assertEqual(resultado.classificacao, esperado)

    def test_prata_real_resiste_a_gradiente_lateral_extremo(self):
        prata = ler_captura_limpa(CAPTURA_PRATA_LIMPA)
        for fatores in ((0.25, 1.40), (1.40, 0.25)):
            resultado = ClassificadorFaixaSaidaLinha().classificar(
                mudar_luz_lateral(prata, *fatores)
            )
            with self.subTest(fatores=fatores):
                self.assertEqual(resultado.classificacao, NAO_PRETA)

    def test_prata_real_em_sombra_extrema_falha_fechada(self):
        prata = ler_captura_limpa(CAPTURA_PRATA_LIMPA)
        casos = (
            aplicar_sombra_local(prata, profundidade=0.65),
            aplicar_gamma(prata, 3.0),
        )

        for indice, frame in enumerate(casos):
            resultado = ClassificadorFaixaSaidaLinha().classificar(frame)
            with self.subTest(caso=indice):
                # Com pouca informacao optica, recuar e tentar outra saida e
                # seguro; transformar prata em PRETA nao e.
                self.assertNotEqual(resultado.classificacao, PRETA)

    def test_preta_real_tolera_ruido_pequeno_da_camera(self):
        preta = cv2.resize(
            ler_captura_limpa(CAPTURA_PRETA_LIMPA),
            (config.camera_x, config.camera_y),
            interpolation=cv2.INTER_AREA,
        )
        gerador = np.random.default_rng(20260811)
        for indice in range(12):
            ruido = gerador.normal(0.0, 5.0, preta.shape)
            frame = np.clip(
                preta.astype(np.float32) + ruido, 0, 255
            ).astype(np.uint8)
            resultado = ClassificadorFaixaSaidaLinha().classificar(frame)
            with self.subTest(frame=indice):
                self.assertEqual(resultado.classificacao, PRETA)

    def test_ruido_forte_na_preta_fica_inconclusivo_nao_prata(self):
        preta = cv2.resize(
            ler_captura_limpa(CAPTURA_PRETA_LIMPA),
            (config.camera_x, config.camera_y),
            interpolation=cv2.INTER_AREA,
        )
        gerador = np.random.default_rng(20260812)
        ruido = gerador.normal(0.0, 12.0, preta.shape)
        frame = np.clip(
            preta.astype(np.float32) + ruido, 0, 255
        ).astype(np.uint8)

        resultado = ClassificadorFaixaSaidaLinha().classificar(frame)

        self.assertEqual(resultado.classificacao, INCONCLUSIVA)


class ConfirmadorTests(unittest.TestCase):
    def test_preto_e_prata_exigem_exposicao_estavel_e_tres_votos(self):
        for cena, esperado in (
            (cena_preta(), PRETA),
            (cena_prata(), NAO_PRETA),
        ):
            confirmador = ConfirmadorFaixaSaidaLinha()
            decisoes = []
            for indice in range(4):
                instante = 1.0 + indice * 0.03
                decisao, _ = confirmador.update(
                    cena, timestamp=instante, now=instante
                )
                decisoes.append(decisao)

            with self.subTest(esperado=esperado):
                # O primeiro frame serve apenas para estabilizar a exposicao;
                # os tres seguintes sao votos em timestamps distintos.
                self.assertEqual(decisoes, [None, None, None, esperado])

    def test_timestamp_repetido_nunca_acumula_voto(self):
        confirmador = ConfirmadorFaixaSaidaLinha()

        for _ in range(8):
            decisao, _ = confirmador.update(
                cena_preta(), timestamp=1.0, now=1.0
            )

        self.assertIsNone(decisao)
        self.assertEqual(confirmador.votos_pretos, 0)
        self.assertFalse(confirmador.exposicao_estavel)

    def test_frame_antigo_nao_estabiliza_exposicao_nem_vota(self):
        confirmador = ConfirmadorFaixaSaidaLinha()

        decisao, resultado = confirmador.update(
            cena_preta(), timestamp=1.0, now=2.0
        )

        self.assertEqual(resultado.classificacao, PRETA)
        self.assertIsNone(decisao)
        self.assertEqual(confirmador.votos_pretos, 0)
        self.assertFalse(confirmador.exposicao_estavel)

    def test_faixa_fora_do_centro_nao_vota(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        distante = cena_preta(topo=28, base=78)

        for indice in range(6):
            instante = 1.0 + indice * 0.03
            decisao, resultado = confirmador.update(
                distante, timestamp=instante, now=instante
            )

        self.assertEqual(resultado.classificacao, PRETA)
        self.assertFalse(faixa_centralizada(resultado))
        self.assertIsNone(decisao)
        self.assertEqual(confirmador.votos_pretos, 0)
        self.assertFalse(confirmador.exposicao_estavel)

    def test_salto_de_exposicao_apaga_votos_anteriores(self):
        confirmador = ConfirmadorFaixaSaidaLinha()
        normal = cena_preta()
        escura = mudar_luz(normal, 0.65)

        # Dois votos pretos depois do frame inicial de estabilizacao.
        for indice in range(3):
            instante = 1.0 + indice * 0.03
            decisao, _ = confirmador.update(
                normal, timestamp=instante, now=instante
            )
        self.assertIsNone(decisao)
        self.assertEqual(confirmador.votos_pretos, 2)

        instante = 1.09
        decisao, _ = confirmador.update(
            escura, timestamp=instante, now=instante
        )
        self.assertIsNone(decisao)
        self.assertEqual(confirmador.votos_pretos, 0)
        self.assertFalse(confirmador.exposicao_estavel)

        # A nova exposicao tambem precisa estabilizar e gerar tres votos novos.
        for indice in range(1, 4):
            instante = 1.09 + indice * 0.03
            decisao, _ = confirmador.update(
                escura, timestamp=instante, now=instante
            )
        self.assertEqual(decisao, PRETA)
        self.assertEqual(confirmador.votos_pretos, 3)


if __name__ == "__main__":
    unittest.main()
