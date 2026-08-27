"""Testes deterministicos do controlador geometrico do segue-linha."""

import math
import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.seguidor_linha import (  # noqa: E402
    CORNER,
    LOST,
    TRACK,
    ControladorSegueLinha,
    angulo_para_ponto_futuro,
    erros_da_geometria,
)


class SeguidorLinhaTests(unittest.TestCase):
    def setUp(self):
        self.controlador = ControladorSegueLinha()
        self.sequencia = 0
        self.agora = 10.

    def quadro(
        self,
        *,
        detectada=True,
        a_frente=True,
        inferior=(224., 251.),
        alvo=(224., 20.),
        futuro=None,
        dt=.02,
    ):
        self.sequencia += 1
        self.agora += dt
        return self.controlador.atualizar(
            sequencia=self.sequencia,
            publicado_em=self.agora,
            linha_detectada=detectada,
            linha_a_frente=a_frente,
            ponto_inferior_x=inferior[0],
            ponto_inferior_y=inferior[1],
            ponto_alvo_x=alvo[0],
            ponto_alvo_y=alvo[1],
            ponto_futuro_x=None if futuro is None else futuro[0],
            ponto_futuro_y=None if futuro is None else futuro[1],
            ponto_futuro_valido=futuro is not None,
            agora=self.agora,
        )

    def test_reta_central_nao_corrige(self):
        saida = self.quadro()

        self.assertEqual(saida.estado, TRACK)
        self.assertAlmostEqual(saida.correcao, 0.)
        self.assertTrue(saida.comando_valido)

    def test_deslocamento_lateral_corrige_para_o_lado_da_linha(self):
        saida = self.quadro(inferior=(336., 251.), alvo=(336., 20.))

        self.assertAlmostEqual(saida.erro_lateral, .5)
        self.assertAlmostEqual(
            saida.correcao,
            config.LINE_LATERAL_GAIN * .5,
        )

    def test_rumo_antecipa_curva_antes_do_ponto_inferior_sair_do_centro(self):
        saida = self.quadro(inferior=(224., 251.), alvo=(324., 51.))

        self.assertGreater(saida.angulo_linha, 20.)
        self.assertGreater(saida.correcao, .4)

    def test_geometria_e_simetrica(self):
        erro_d, angulo_d = erros_da_geometria(300, 250, 380, 80)
        erro_e, angulo_e = erros_da_geometria(148, 250, 68, 80)

        self.assertAlmostEqual(erro_d, -erro_e)
        self.assertAlmostEqual(angulo_d, -angulo_e)

    def test_rumo_futuro_parte_do_centro_fisico_do_robo(self):
        direita = angulo_para_ponto_futuro(336, 20)
        esquerda = angulo_para_ponto_futuro(112, 20)

        self.assertGreater(direita, 0.)
        self.assertAlmostEqual(direita, -esquerda)

    def test_ponto_futuro_central_ainda_alinha_a_base_do_robo(self):
        saidas = [
            self.quadro(
                a_frente=True,
                inferior=(300., 251.),
                alvo=(420., 126.),
                futuro=(224., 10.),
            )
            for _ in range(3)
        ]

        self.assertTrue(all(saida.estado == TRACK for saida in saidas))
        self.assertTrue(all(saida.correcao > 0. for saida in saidas))
        self.assertAlmostEqual(
            saidas[-1].correcao,
            config.LINE_LATERAL_GAIN * (76. / 224.),
        )

    def test_retorno_futuro_suprime_canto_mas_conserva_alinhamento_da_base(self):
        saidas = [
            self.quadro(
                a_frente=False,
                inferior=(300., 251.),
                alvo=(420., 126.),
                futuro=(224., 10.),
            )
            for _ in range(3)
        ]

        self.assertTrue(all(saida.estado == TRACK for saida in saidas))
        self.assertTrue(all(saida.correcao > 0. for saida in saidas))

    def test_controle_segue_o_futuro_mesmo_se_o_local_aponta_ao_contrario(self):
        saida = self.quadro(
            a_frente=True,
            inferior=(300., 251.),
            alvo=(400., 126.),
            futuro=(100., 20.),
        )

        self.assertEqual(saida.estado, TRACK)
        self.assertLess(saida.correcao, 0.)

    def test_ponto_futuro_deslocado_produz_correcao_proporcional(self):
        saida = self.quadro(futuro=(336., 20.))
        angulo = angulo_para_ponto_futuro(336., 20.)
        esperado = config.LINE_FUTURE_GAIN * math.sin(math.radians(angulo))

        self.assertEqual(saida.estado, TRACK)
        self.assertAlmostEqual(saida.correcao, esperado)

    def test_90_real_continua_armando_canto(self):
        primeiro = self.quadro(
            a_frente=False,
            alvo=(447., 126.),
            futuro=(360., 120.),
        )
        segundo = self.quadro(
            a_frente=False,
            alvo=(447., 126.),
            futuro=(360., 120.),
        )

        self.assertEqual(primeiro.estado, TRACK)
        self.assertEqual(segundo.estado, CORNER)

    def test_canto_so_arma_apos_dois_frames_e_faz_pivo(self):
        primeiro = self.quadro(
            a_frente=False, inferior=(224., 251.), alvo=(447., 126.))
        segundo = self.quadro(
            a_frente=False, inferior=(224., 251.), alvo=(447., 126.))

        self.assertEqual(primeiro.estado, TRACK)
        self.assertEqual(segundo.estado, CORNER)
        self.assertGreaterEqual(
            segundo.correcao, config.LINE_CORNER_MIN_CORRECTION)

    def test_canto_esquerdo_e_direito_geram_comandos_espelhados(self):
        direita = ControladorSegueLinha()
        esquerda = ControladorSegueLinha()
        saida_d = saida_e = None
        for sequencia in (1, 2):
            instante = 10. + sequencia * .02
            saida_d = direita.atualizar(
                sequencia=sequencia, publicado_em=instante,
                linha_detectada=True, linha_a_frente=False,
                ponto_inferior_x=224, ponto_inferior_y=251,
                ponto_alvo_x=447, ponto_alvo_y=126, agora=instante)
            saida_e = esquerda.atualizar(
                sequencia=sequencia, publicado_em=instante,
                linha_detectada=True, linha_a_frente=False,
                ponto_inferior_x=224, ponto_inferior_y=251,
                ponto_alvo_x=1, ponto_alvo_y=126, agora=instante)

        self.assertAlmostEqual(saida_d.correcao, -saida_e.correcao)
        self.assertEqual(saida_d.estado, CORNER)
        self.assertEqual(saida_e.estado, CORNER)

    def test_canto_fica_travado_durante_perda_curta(self):
        self.quadro(a_frente=False, alvo=(447., 126.))
        self.quadro(a_frente=False, alvo=(447., 126.))
        perdida = self.quadro(detectada=False, dt=.10)

        self.assertEqual(perdida.estado, LOST)
        self.assertTrue(perdida.comando_valido)
        self.assertGreaterEqual(
            perdida.correcao, config.LINE_CORNER_MIN_CORRECTION)

    def test_ponto_futuro_oposto_cancela_canto_ja_armado(self):
        self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))
        armado = self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))

        retorno_1 = self.quadro(
            a_frente=False, alvo=(1., 126.), futuro=(70., 20.))
        retorno_2 = self.quadro(
            a_frente=False, alvo=(1., 126.), futuro=(70., 20.))

        self.assertEqual(armado.estado, CORNER)
        self.assertEqual(retorno_1.estado, CORNER)
        self.assertGreaterEqual(
            retorno_1.correcao, config.LINE_CORNER_MIN_CORRECTION)
        self.assertEqual(retorno_2.estado, TRACK)
        self.assertLess(retorno_2.correcao, 0.)

    def test_um_quadro_futuro_oposto_nao_desarma_90_real(self):
        self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))
        self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))

        ruido = self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(70., 20.))
        recuperado = self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))

        self.assertEqual(ruido.estado, CORNER)
        self.assertEqual(recuperado.estado, CORNER)
        self.assertGreaterEqual(
            recuperado.correcao, config.LINE_CORNER_MIN_CORRECTION)

    def test_futuro_central_sozinho_nao_abandona_90_real(self):
        self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))
        armado = self.quadro(
            a_frente=False, alvo=(447., 126.), futuro=(380., 80.))

        saidas = [
            self.quadro(
                a_frente=False,
                inferior=(300., 251.),
                alvo=(447., 126.),
                futuro=(224., 20.),
            )
            for _ in range(3)
        ]

        self.assertEqual(armado.estado, CORNER)
        self.assertTrue(all(saida.estado == CORNER for saida in saidas))
        self.assertTrue(all(
            saida.correcao >= config.LINE_CORNER_MIN_CORRECTION
            for saida in saidas
        ))

    def test_base_que_cruza_o_centro_interrompe_excesso_de_giro(self):
        self.quadro(a_frente=False, alvo=(447., 126.))
        armado = self.quadro(a_frente=False, alvo=(447., 126.))

        cruzou_1 = self.quadro(
            a_frente=True,
            inferior=(170., 251.),
            alvo=(224., 20.),
        )
        cruzou_2 = self.quadro(
            a_frente=True,
            inferior=(170., 251.),
            alvo=(224., 20.),
        )

        self.assertEqual(armado.estado, CORNER)
        self.assertEqual(cruzou_1.estado, CORNER)
        self.assertEqual(cruzou_2.estado, TRACK)
        self.assertLess(cruzou_2.correcao, 0.)

    def test_canto_sai_so_depois_de_tres_frames_alinhados(self):
        self.quadro(a_frente=False, alvo=(447., 126.))
        self.quadro(a_frente=False, alvo=(447., 126.))

        saidas = [self.quadro() for _ in range(3)]

        self.assertEqual(saidas[0].estado, CORNER)
        self.assertEqual(saidas[1].estado, CORNER)
        self.assertEqual(saidas[2].estado, TRACK)
        self.assertEqual(saidas[2].correcao, 0.)

    def test_perda_em_reta_para_apos_janela_curta(self):
        self.quadro(inferior=(250., 251.), alvo=(260., 20.))
        curta = self.quadro(detectada=False, dt=.10)
        longa = self.quadro(detectada=False, dt=.20)

        self.assertTrue(curta.comando_valido)
        self.assertEqual(curta.estado, LOST)
        self.assertFalse(longa.comando_valido)
        self.assertEqual(longa.correcao, 0.)

    def test_frame_congelado_e_tratado_como_linha_perdida(self):
        saida = self.quadro()
        congelada = self.controlador.atualizar(
            sequencia=self.sequencia,
            publicado_em=self.agora,
            linha_detectada=True,
            linha_a_frente=True,
            ponto_inferior_x=224,
            ponto_inferior_y=251,
            ponto_alvo_x=224,
            ponto_alvo_y=20,
            agora=self.agora + config.LINE_MAX_FRAME_AGE_S + .01,
        )

        self.assertEqual(saida.estado, TRACK)
        self.assertEqual(congelada.estado, LOST)


if __name__ == "__main__":
    unittest.main()
