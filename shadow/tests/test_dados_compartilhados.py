"""Testes dos dados rápidos compartilhados entre visão e controle."""

import sys
from pathlib import Path
import unittest
from unittest import mock


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.estado_verde import GreenDecision, GreenObservation  # noqa: E402
from shared.dados_compartilhados import (  # noqa: E402
    add_time_value,
    empty_time_arr,
    get_time_average,
    ler_observacao_intersecao,
    ler_comando_motores,
    ler_resultado_visao_rapida,
    publicar_observacao_intersecao,
    publicar_comando_motores,
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
    def test_comando_dos_dois_lados_e_publicado_atomicamente(self):
        anterior = ler_comando_motores().command_id

        publicar_comando_motores(72, -72, publicado_em=4.5)
        comando = ler_comando_motores()

        self.assertEqual(comando.command_id, anterior + 1)
        self.assertEqual(comando.publicado_em, 4.5)
        self.assertEqual((comando.esquerda, comando.direita), (72, -72))

    @staticmethod
    def _publicar_minimo(**overrides):
        valores = dict(
            publicado_em=1.0,
            processamento_ms=1.0,
            linha_detectada=True,
            linha_a_frente=True,
            angulo=0.0,
            ponto_inferior_x=224.0,
            ponto_inferior_y=251.0,
            ponto_alvo_x=224.0,
            ponto_alvo_y=20.0,
            area_linha=1000.0,
            candidato_verde=False,
            candidato_vermelho=False,
            ponto_futuro_x=224.0,
            ponto_futuro_y=20.0,
            ponto_futuro_valido=True,
            faixa_transversal_y=-1.0,
            juncao_topologica_visivel=False,
        )
        valores.update(overrides)
        publicar_resultado_visao_rapida(**valores)

    def test_visao_pode_publicar_a_mesma_sequencia_do_evento_verde(self):
        self._publicar_minimo(sequencia=731)

        self.assertEqual(ler_resultado_visao_rapida().sequencia, 731)

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
            ponto_alvo_x=180,
            ponto_alvo_y=40,
            area_linha=8123,
            candidato_verde=False,
            candidato_vermelho=True,
            ponto_futuro_x=230,
            ponto_futuro_y=30,
            ponto_futuro_valido=True,
            faixa_transversal_y=126,
            juncao_topologica_visivel=True,
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
        self.assertEqual(resultado.ponto_alvo_x, 180)
        self.assertEqual(resultado.ponto_alvo_y, 40)
        self.assertEqual(resultado.area_linha, 8123)
        self.assertFalse(resultado.candidato_verde)
        self.assertTrue(resultado.candidato_vermelho)
        self.assertEqual(resultado.ponto_futuro_x, 230)
        self.assertEqual(resultado.ponto_futuro_y, 30)
        self.assertTrue(resultado.ponto_futuro_valido)
        self.assertEqual(resultado.faixa_transversal_y, 126)
        self.assertTrue(resultado.juncao_topologica_visivel)
        self.assertEqual(resultado.locked_branch_token, 0)
        self.assertFalse(resultado.locked_branch_valid)
        self.assertEqual(resultado.locked_branch_bottom_x, -1.0)
        self.assertEqual(resultado.locked_branch_bottom_y, -1.0)

    def test_observacao_verde_e_publicada_atomicamente(self):
        evento = GreenObservation(
            sequence=81,
            junction_id=12,
            decision_id=7,
            timestamp=42.25,
            decision=GreenDecision.RIGHT,
            confidence=.93,
            entry_tangent=(.1, .99),
            junction_center=(220., 104.),
            target_branch=(380., 105.),
            target_branch_token=97,
            ready_to_turn=True,
            marker_ids=(901,),
        )

        publicar_observacao_intersecao(evento)

        self.assertEqual(ler_observacao_intersecao(), evento)


if __name__ == "__main__":
    unittest.main()
