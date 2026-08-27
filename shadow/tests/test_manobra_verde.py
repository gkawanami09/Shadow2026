"""Testes da transicao geometrica para o giro verde."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.manobra_verde import (  # noqa: E402
    alinhamento_verde_pode_concluir,
    deve_iniciar_giro_verde,
    correcao_aproximacao,
    progresso_giro_mpu,
    ramo_chegou_ao_centro,
    ramo_pronto_para_giro,
)


class ManobraVerdeTests(unittest.TestCase):
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

    def test_aproximacao_expirada_com_linha_valida_inicia_giro(self):
        self.assertTrue(deve_iniciar_giro_verde(
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

    def test_ramo_que_salta_sobre_centro_tambem_conclui(self):
        self.assertTrue(ramo_chegou_ao_centro(-60, 70, 1))

    def test_ramo_ainda_no_mesmo_lado_continua_girando(self):
        self.assertFalse(ramo_chegou_ao_centro(60, 80, 1))

    def test_mpu_nao_deixa_concluir_verde_com_apenas_40_graus(self):
        self.assertFalse(alinhamento_verde_pode_concluir(
            0, 60, 1, config.GREEN_MPU_TARGET_ARM_DEG))

    def test_mpu_e_centro_persistente_podem_concluir_verde(self):
        self.assertTrue(alinhamento_verde_pode_concluir(
            0, 60, 1, config.GREEN_MPU_COMPLETION_MIN_DEG))

    def test_mpu_mede_giro_independente_do_sentido(self):
        self.assertEqual(progresso_giro_mpu(12., 102.), 90.)
        self.assertEqual(progresso_giro_mpu(12., -78.), 90.)

    def test_mpu_ausente_nao_inventa_progresso(self):
        self.assertIsNone(progresso_giro_mpu(None, 90.))


if __name__ == "__main__":
    unittest.main()
