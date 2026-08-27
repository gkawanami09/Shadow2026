"""Testes da transicao geometrica para o giro verde."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.manobra_verde import (  # noqa: E402
    alinhamento_verde_pode_concluir,
    controle_visual_verde_liberado,
    deve_iniciar_giro_verde,
    correcao_aproximacao,
    correcao_reaquisicao_verde,
    correcao_ramo_reto,
    juncao_topologica_realmente_ausente,
    progresso_giro_mpu,
    ramo_chegou_ao_centro,
    ramo_marcado_visto_pela_camera,
    ramo_pronto_para_giro,
    ramo_travado_recente,
    saida_topologica_real_estavel,
)


class ManobraVerdeTests(unittest.TestCase):
    @staticmethod
    def _resultado_saida(*, junction_visible, sequence=10, bottom_x=None,
                         published_at=1.0, detected=True):
        return SimpleNamespace(
            sequencia=sequence,
            publicado_em=published_at,
            linha_detectada=detected,
            ponto_inferior_x=(
                config.camera_x / 2 if bottom_x is None else bottom_x),
            juncao_topologica_visivel=junction_visible,
        )

    def test_saida_reta_exige_ausencia_topologica_crua(self):
        self.assertFalse(saida_topologica_real_estavel(
            self._resultado_saida(junction_visible=True),
            agora=1.05,
        ))

    def test_ramo_travado_exige_o_mesmo_token_atomico(self):
        resultado = SimpleNamespace(
            publicado_em=1.0,
            locked_branch_token=77,
            locked_branch_valid=True,
            locked_branch_bottom_x=310.0,
            locked_branch_bottom_y=config.camera_y - 2,
        )

        self.assertTrue(ramo_travado_recente(
            resultado, 77, agora=1.02))
        self.assertFalse(ramo_travado_recente(
            resultado, 78, agora=1.02))
        self.assertFalse(ramo_travado_recente(
            resultado, 77, agora=2.0))

    def test_ramo_perdido_nao_e_substituido_por_linha_generica(self):
        resultado = SimpleNamespace(
            publicado_em=1.0,
            locked_branch_token=77,
            locked_branch_valid=False,
            locked_branch_bottom_x=config.camera_x / 2,
            locked_branch_bottom_y=config.camera_y - 2,
            linha_detectada=True,
            ponto_inferior_x=config.camera_x / 2,
        )

        self.assertFalse(ramo_travado_recente(
            resultado, 77, agora=1.02))

    def test_token_no_topo_nao_pode_concluir_com_outra_linha_na_base(self):
        resultado = SimpleNamespace(
            publicado_em=1.0,
            locked_branch_token=77,
            locked_branch_valid=True,
            locked_branch_bottom_x=config.camera_x / 2,
            locked_branch_bottom_y=20.0,
            linha_detectada=True,
            ponto_inferior_x=config.camera_x / 2,
        )

        self.assertFalse(ramo_travado_recente(
            resultado, 77, agora=1.02))
        self.assertTrue(saida_topologica_real_estavel(
            self._resultado_saida(junction_visible=False),
            agora=1.05,
        ))

    def test_ausencia_topologica_nao_depende_da_linha_estar_central(self):
        resultado = self._resultado_saida(
            junction_visible=False,
            bottom_x=config.camera_x,
            detected=False,
        )

        self.assertTrue(juncao_topologica_realmente_ausente(
            resultado,
            agora=1.05,
        ))

    def test_saida_reta_rejeita_frame_velho_ou_descentralizado(self):
        self.assertFalse(saida_topologica_real_estavel(
            self._resultado_saida(
                junction_visible=False,
                published_at=0.0,
            ),
            agora=1.0,
        ))
        self.assertFalse(saida_topologica_real_estavel(
            self._resultado_saida(
                junction_visible=False,
                bottom_x=config.camera_x,
            ),
            agora=1.05,
        ))

    def test_aproximacao_reta_nao_antecipa_ramo_travado(self):
        self.assertEqual(correcao_aproximacao(config.camera_x / 2), 0.)

    def test_aproximacao_corrige_apenas_deslocamento_da_base(self):
        direita = correcao_aproximacao(config.camera_x * .75)
        esquerda = correcao_aproximacao(config.camera_x * .25)

        self.assertGreater(direita, 0.)
        self.assertAlmostEqual(direita, -esquerda)

    def test_aproximacao_limita_correcao_para_continuar_avancando(self):
        self.assertEqual(
            correcao_aproximacao(config.camera_x),
            config.GREEN_APPROACH_MAX_CORRECTION,
        )

    def test_ramo_reto_topologico_domina_topo_lateral_legacy(self):
        correcao = correcao_ramo_reto(
            config.camera_x * .75,
            config.camera_x * .25,
        )
        self.assertGreater(correcao, 0.)

    def test_ramo_reto_central_preserva_centralizacao_da_entrada(self):
        correcao = correcao_ramo_reto(
            config.camera_x / 2,
            config.camera_x * .75,
        )
        self.assertGreater(correcao, 0.)
        self.assertLess(correcao, config.GREEN_APPROACH_MAX_CORRECTION)

    def test_ramo_reto_sem_alvo_valido_falha_fechado(self):
        self.assertIsNone(correcao_ramo_reto(-1, config.camera_x / 2))

    def test_ramo_distante_ainda_nao_inicia_tanque(self):
        self.assertFalse(ramo_pronto_para_giro(
            "right",
            faixa_transversal_y=config.camera_y * .30,
        ))

    def test_ramo_direito_proximo_inicia_tanque(self):
        self.assertTrue(ramo_pronto_para_giro(
            "right",
            faixa_transversal_y=config.camera_y * .90,
        ))

    def test_ramo_esquerdo_proximo_inicia_tanque(self):
        self.assertTrue(ramo_pronto_para_giro(
            "left",
            faixa_transversal_y=config.camera_y * .90,
        ))

    def test_faixa_ausente_nao_inicia_tanque(self):
        self.assertFalse(ramo_pronto_para_giro(
            "right",
            faixa_transversal_y=-1,
        ))

    def test_ramo_dentro_da_tolerancia_conclui_giro(self):
        self.assertTrue(ramo_chegou_ao_centro(20, 80, 1))

    def test_tempo_sozinho_nunca_inicia_giro(self):
        self.assertFalse(deve_iniciar_giro_verde(
            0, agora=2., limite_aproximacao=1., linha_recente=True))

    def test_aproximacao_expirada_sem_linha_nao_gira_as_cegas(self):
        self.assertFalse(deve_iniciar_giro_verde(
            0, agora=2., limite_aproximacao=1., linha_recente=False))

    def test_transversal_confirmada_libera_antes_do_timeout(self):
        self.assertTrue(deve_iniciar_giro_verde(
            config.GREEN_BRANCH_CONFIRM_FRAMES,
            agora=.5,
            limite_aproximacao=1.,
            linha_recente=True,
        ))

    def test_linha_de_entrada_nao_libera_controle_visual_do_verde(self):
        self.assertFalse(ramo_marcado_visto_pela_camera(
            True, 0., 1))
        self.assertFalse(controle_visual_verde_liberado(False, True))

    def test_ramo_direito_na_base_libera_controle_visual(self):
        self.assertTrue(ramo_marcado_visto_pela_camera(
            True, config.GREEN_TURN_SIDE_MIN_ERROR_PX + 1, 1))
        self.assertTrue(controle_visual_verde_liberado(True, True))

    def test_reaquisicao_mantem_sinal_imutavel_ate_o_centro(self):
        direita = correcao_reaquisicao_verde(
            config.camera_x / 2 + 80, 1)
        esquerda = correcao_reaquisicao_verde(
            config.camera_x / 2 - 80, -1)
        self.assertGreater(direita, 0.)
        self.assertLess(esquerda, 0.)
        self.assertAlmostEqual(direita, -esquerda)

    def test_reaquisicao_para_de_girar_na_zona_central(self):
        self.assertEqual(correcao_reaquisicao_verde(
            config.camera_x / 2 + config.GREEN_TURN_CENTER_TOLERANCE_PX,
            1,
        ), 0.)

    def test_reaquisicao_nunca_inverte_apos_cruzar_o_centro(self):
        self.assertEqual(correcao_reaquisicao_verde(
            config.camera_x / 2 - 60,
            1,
        ), 0.)

    def test_reaquisicao_rejeita_lado_ou_ponto_invalidos(self):
        self.assertIsNone(correcao_reaquisicao_verde(float("nan"), 1))
        self.assertIsNone(correcao_reaquisicao_verde(100, 0))

    def test_ramo_oposto_nao_pode_cancelar_giro_verde(self):
        self.assertFalse(ramo_marcado_visto_pela_camera(
            True, -(config.GREEN_TURN_SIDE_MIN_ERROR_PX + 1), 1))

    def test_frame_antigo_nao_declara_que_encontrou_ramo(self):
        self.assertFalse(ramo_marcado_visto_pela_camera(
            False, config.camera_x, 1))

    def test_ramo_que_salta_sobre_centro_tambem_conclui(self):
        self.assertTrue(ramo_chegou_ao_centro(-60, 70, 1))

    def test_ramo_ainda_no_mesmo_lado_continua_girando(self):
        self.assertFalse(ramo_chegou_ao_centro(60, 80, 1))

    def test_mpu_nao_deixa_concluir_verde_com_apenas_40_graus(self):
        self.assertFalse(alinhamento_verde_pode_concluir(
            0, 60, 1, 40.))

    def test_mpu_e_centro_persistente_podem_concluir_verde(self):
        self.assertTrue(alinhamento_verde_pode_concluir(
            0, 60, 1, config.GREEN_MPU_COMPLETION_MIN_DEG))

    def test_mpu_mede_giro_independente_do_sentido(self):
        self.assertEqual(progresso_giro_mpu(12., 102.), 90.)
        self.assertEqual(progresso_giro_mpu(12., -78.), 90.)

    def test_mpu_legado_respeita_wrap(self):
        self.assertEqual(progresso_giro_mpu(350., 10.), 20.)

    def test_mpu_ausente_nao_inventa_progresso(self):
        self.assertIsNone(progresso_giro_mpu(None, 90.))


if __name__ == "__main__":
    unittest.main()
