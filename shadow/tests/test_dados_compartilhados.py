"""Testes dos dados rápidos compartilhados entre visão e controle."""

import sys
from pathlib import Path
import unittest
from unittest import mock


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from shared.dados_compartilhados import (  # noqa: E402
    add_time_value,
    empty_time_arr,
    get_time_average,
    ler_resultado_visao_rapida,
    ler_resultado_trajetoria_linha,
    publicar_resultado_trajetoria_linha,
    publicar_resultado_visao_rapida,
)


class HistoricoTemporalTests(unittest.TestCase):
    def test_historico_e_circular_sem_copiar_o_array_inteiro(self):
        historico = empty_time_arr(3)
        with mock.patch(
            "shared.dados_compartilhados.time.perf_counter",
            side_effect=(1., 2., 3., 4.),
        ):
            for valor in (10, 20, 30, 40):
                mesmo_historico = add_time_value(historico, valor)
                self.assertIs(mesmo_historico, historico)

        self.assertEqual(list(historico), [(2., 20), (3., 30), (4., 40)])

    def test_media_usa_apenas_valores_recentes(self):
        historico = empty_time_arr(4)
        historico.extend(((1., 10), (2., 20), (3., 30), (4., 40)))

        with mock.patch(
            "shared.dados_compartilhados.time.perf_counter",
            return_value=4.5,
        ):
            media = get_time_average(historico, 2.)

        self.assertEqual(media, 35.)


class ResultadoVisaoRapidaTests(unittest.TestCase):
    def test_publicacao_entrega_um_frame_completo_e_incrementa_sequencia(self):
        sequencia_antes = ler_resultado_visao_rapida().sequencia

        publicar_resultado_visao_rapida(
            publicado_em=12.5,
            processamento_ms=7.2,
            linha_detectada=True,
            linha_a_frente=True,
            angulo=-4,
            ponto_inferior_x=220,
            ponto_inferior_y=250,
            area_linha=8123,
            candidato_verde=False,
            candidato_vermelho=True,
        )
        resultado = ler_resultado_visao_rapida()

        self.assertEqual(resultado.sequencia, sequencia_antes + 1)
        self.assertEqual(resultado.publicado_em, 12.5)
        self.assertEqual(resultado.processamento_ms, 7.2)
        self.assertTrue(resultado.linha_detectada)
        self.assertTrue(resultado.linha_a_frente)
        self.assertEqual(resultado.angulo, -4)
        self.assertEqual(resultado.ponto_inferior_x, 220)
        self.assertEqual(resultado.ponto_inferior_y, 250)
        self.assertEqual(resultado.area_linha, 8123)
        self.assertFalse(resultado.candidato_verde)
        self.assertTrue(resultado.candidato_vermelho)

    def test_publicacao_da_trajetoria_e_atomica(self):
        sequencia_antes = ler_resultado_trajetoria_linha().sequencia
        publicar_resultado_trajetoria_linha(
            publicado_em=15.,
            valida=True,
            lateral=.2,
            orientacao=-.1,
            curvatura=.3,
            confianca=.9,
            largura_normalizada=.08,
            amostras=9,
        )
        resultado = ler_resultado_trajetoria_linha()

        self.assertEqual(resultado.sequencia, sequencia_antes + 1)
        self.assertEqual(resultado.publicado_em, 15.)
        self.assertTrue(resultado.valida)
        self.assertEqual(resultado.lateral, .2)
        self.assertEqual(resultado.orientacao, -.1)
        self.assertEqual(resultado.curvatura, .3)
        self.assertEqual(resultado.confianca, .9)
        self.assertEqual(resultado.amostras, 9)


if __name__ == "__main__":
    unittest.main()
