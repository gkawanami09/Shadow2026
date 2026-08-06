"""Testes dos pontos de integração da missão dentro do resgate.

Aqui não há câmera, serial nem motores: apenas as decisões que ``resgate.py``
e ``mission.py`` tomam com base no inventário e na fase de saída.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
import config_resgate as cfg  # noqa: E402
import resgate  # noqa: E402
from visao.entrada_missao import _deve_pre_avancar_entrada  # noqa: E402
from mission import (  # noqa: E402
    MissionSystem,
    _tecla_fecha_debug,
    iniciar_debug_linha,
)


class MissionDebugWindowTests(unittest.TestCase):
    def test_q_e_escape_fecham_o_debug(self):
        self.assertTrue(_tecla_fecha_debug(ord("q")))
        self.assertTrue(_tecla_fecha_debug(27))

    def test_outras_teclas_mantem_o_debug(self):
        self.assertFalse(_tecla_fecha_debug(-1))
        self.assertFalse(_tecla_fecha_debug(ord("s")))

    def test_modo_debug_inicia_visualizador_separado(self):
        processos = []

        class ProcessoFalso:
            def __init__(self, target, args=(), name=None):
                self.target = target
                self.args = args
                self.name = name
                processos.append(self)

            def start(self):
                return None

        valor = lambda inicial: SimpleNamespace(value=inicial)
        compartilhado = SimpleNamespace(
            terminate=valor(False),
            rescue_requested=valor(False),
            red_finished=valor(False),
            mission_mode=valor(False),
        )
        sistema = MissionSystem(
            compartilhado,
            motor_lock=None,
            args=SimpleNamespace(debug=True),
        )

        with (
            patch("mission.Process", ProcessoFalso),
            patch("mission.time.sleep"),
        ):
            sistema.start_line_phase()

        self.assertEqual(len(processos), 3)
        self.assertIs(processos[-1].target, iniciar_debug_linha)
        self.assertEqual(processos[-1].name, "shadow-debug-linha")


class MissionEntryAdvanceTests(unittest.TestCase):
    def test_entrada_prata_confirma_em_dois_de_tres_frames(self):
        self.assertEqual(config.ENTRY_SILVER_VOTES_NEEDED, 2)
        self.assertEqual(config.ENTRY_SILVER_VOTE_WINDOW, 3)

    def test_faixa_fina_distante_pede_avanco_antes_da_correcao(self):
        self.assertTrue(_deve_pre_avancar_entrada(
            modo_missao=True,
            armada=True,
            confirmada=False,
            detectada=False,
            votos=0,
            motivo="fina",
            linha_adiante=False,
        ))

    def test_faixa_fina_avanca_mesmo_com_trecho_preto_ainda_visivel(self):
        self.assertTrue(_deve_pre_avancar_entrada(
            modo_missao=True,
            armada=True,
            confirmada=False,
            detectada=False,
            votos=0,
            motivo="fina",
            linha_adiante=True,
        ))

    def test_linha_continuando_veta_candidato_que_nao_e_faixa_fina(self):
        self.assertFalse(_deve_pre_avancar_entrada(
            modo_missao=True,
            armada=True,
            confirmada=False,
            detectada=True,
            votos=1,
            motivo="",
            linha_adiante=True,
        ))

    def test_candidato_aceito_continua_reto_ate_confirmar(self):
        self.assertTrue(_deve_pre_avancar_entrada(
            modo_missao=True,
            armada=True,
            confirmada=False,
            detectada=True,
            votos=1,
            motivo="",
            linha_adiante=False,
        ))

    def test_main_isolado_nao_muda_comportamento(self):
        self.assertFalse(_deve_pre_avancar_entrada(
            modo_missao=False,
            armada=True,
            confirmada=False,
            detectada=True,
            votos=1,
            motivo="fina",
            linha_adiante=False,
        ))

    def test_avanco_da_entrada_tem_um_segundo_e_pwm_80(self):
        self.assertEqual(cfg.MISSION_ENTRY_FORWARD_S, 1.0)
        self.assertEqual(cfg.MISSION_ENTRY_FORWARD_PWM, 80)
        self.assertAlmostEqual(
            cfg.MISSION_ENTRY_FORWARD_SPEED * 120,
            cfg.MISSION_ENTRY_FORWARD_PWM,
        )

    def test_missao_avanca_e_para_antes_da_busca(self):
        args = SimpleNamespace(
            drive=True,
            gerenciado_pela_missao=True,
        )
        arduino = SimpleNamespace(connection_epoch=7)
        comandos = []

        def direcao(*argumentos):
            comandos.append(argumentos)
            return True

        with patch.object(
            resgate,
            "_mover_saida_por_tempo",
        ) as mover:
            executou = resgate._avancar_entrada_da_missao(
                args,
                arduino,
                direcao,
            )

        self.assertTrue(executou)
        mover.assert_called_once_with(
            arduino,
            direcao,
            0,
            cfg.MISSION_ENTRY_FORWARD_SPEED,
            cfg.MISSION_ENTRY_FORWARD_S,
            7,
        )
        self.assertEqual(comandos, [()])

    def test_resgate_aberto_sozinho_nao_faz_o_avanco(self):
        args = SimpleNamespace(
            drive=True,
            gerenciado_pela_missao=False,
        )
        arduino = SimpleNamespace(connection_epoch=7)

        with patch.object(
            resgate,
            "_mover_saida_por_tempo",
        ) as mover:
            executou = resgate._avancar_entrada_da_missao(
                args,
                arduino,
                lambda *_: True,
            )

        self.assertFalse(executou)
        mover.assert_not_called()


class ConfigProfileSeparationTests(unittest.TestCase):
    """Perfis de câmera diferentes não podem se sobrepor."""

    def test_faixa_prata_pertence_a_camera_de_linha(self):
        self.assertTrue(hasattr(config, "ENTRY_SILVER_MIN_DEFAULT"))
        self.assertTrue(hasattr(config, "ENTRY_SILVER_MAX_DEFAULT"))
        # E não vaza para o módulo do resgate.
        self.assertFalse(hasattr(cfg, "ENTRY_SILVER_MIN_DEFAULT"))

    def test_faixa_preta_pertence_a_camera_de_resgate(self):
        self.assertTrue(hasattr(cfg, "EXIT_BLACK_HSV_MIN"))
        self.assertFalse(hasattr(config, "EXIT_BLACK_HSV_MIN"))

    def test_vitima_e_entrada_tem_limiares_independentes(self):
        """Prata da vítima e prata da entrada não compartilham constantes."""
        self.assertNotEqual(
            id(config.ENTRY_SILVER_MAX_SATURATION),
            id(cfg.BALL_SILVER_S_MAX))
        # A entrada é calibrada em HSV; a vítima usa uma bateria própria de
        # assinaturas (lisa, amassada, clara, tingida). São modelos distintos.
        self.assertTrue(hasattr(cfg, "BALL_SILVER_SMOOTH_INNER_V_MIN"))
        self.assertFalse(hasattr(config, "BALL_SILVER_SMOOTH_INNER_V_MIN"))


class LineFollowerUnchangedTests(unittest.TestCase):
    """Rodar `main.py` sozinho não pode mudar de comportamento."""

    def test_sem_mission_mode_o_detector_de_entrada_nao_e_construido(self):
        from shared.dados_compartilhados import mission_mode
        from visao import entrada_missao

        anterior = mission_mode.value
        try:
            mission_mode.value = False
            self.assertIsNone(entrada_missao.build_entry_gate())
        finally:
            mission_mode.value = anterior

    def test_atualizacao_sem_portao_e_inofensiva(self):
        from visao import entrada_missao
        # Sem portão (modo main.py) a função retorna sem tocar em nada.
        self.assertIsNone(
            entrada_missao.update_entry_silver(None, None, 0.0))

    def test_valores_da_missao_comecam_desligados(self):
        import shared.dados_compartilhados as shared
        self.assertFalse(shared.rescue_requested.value)
        self.assertFalse(shared.red_finished.value)
        self.assertFalse(shared.entry_silver_confirmed.value)

    def test_entrada_desarmada_nao_reprocessa(self):
        """Depois de entrar na sala, a faixa prata deixa de ser avaliada."""
        from shared.dados_compartilhados import (entry_armed,
                                                 entry_silver_detected)
        from visao import entrada_missao

        class GatePlaceholder:
            def update(self, *args, **kwargs):
                raise AssertionError(
                    "o portão não pode ser consultado com a entrada desarmada")

        anterior = entry_armed.value
        try:
            entry_armed.value = False
            entry_silver_detected.value = True
            entrada_missao.update_entry_silver(GatePlaceholder(), None, 0.0)
            self.assertFalse(entry_silver_detected.value)
        finally:
            entry_armed.value = anterior


class PulsedSearchConfigTests(unittest.TestCase):
    def test_pulso_e_setores_batem_com_o_360_calibrado(self):
        """setores × pulso deve ficar próximo do 360 temporizado."""
        cobertura = cfg.BALL_SEARCH_SECTORS * cfg.BALL_SEARCH_PULSE_S
        self.assertAlmostEqual(
            cobertura, cfg.BALL_SEARCH_FULL_TURN_S, delta=1.0)

    def test_timeout_total_cobre_a_varredura_com_pausas(self):
        pausas = cfg.BALL_SEARCH_SECTORS * (
            cfg.BALL_SEARCH_SETTLE_S + cfg.BALL_SEARCH_OBSERVE_TIMEOUT_S)
        minimo = cfg.BALL_SEARCH_FULL_TURN_S + pausas
        self.assertGreaterEqual(cfg.BALL_SEARCH_TOTAL_TIMEOUT_S, minimo)


    # O prefixo da coleta voltou ao resgate.py. Deposito, busca da proxima
    # vitima e saida continuam fora do fluxo atual.


if __name__ == "__main__":
    unittest.main()
