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


class MissionLineResumeTests(unittest.TestCase):
    def test_retomada_apaga_memoria_lateral_e_prefere_ramo_reto(self):
        valor = lambda inicial: SimpleNamespace(value=inicial)
        compartilhado = SimpleNamespace(
            vision_ready=valor(True),
            line_detected=valor(True),
            line_ahead=valor(True),
            line_angle=valor(90),
            line_angle_y=valor(200),
            line_size=valor(9000.0),
            last_bottom_point=valor(30),
            last_bottom_point_y=valor(250),
            line_status=valor("gap_avoid"),
            turn_dir=valor("left"),
            green_turn_target=valor(-1),
            preferencia_linha_esquerda=valor(True),
            line_crop=valor(config.LINE_CROP_GREEN),
            min_line_size=valor(9000),
        )
        sistema = MissionSystem(
            compartilhado,
            motor_lock=None,
            args=SimpleNamespace(debug=False),
        )

        with patch.object(sistema, "start_line_phase") as iniciar:
            sistema.reacquire_line()

        iniciar.assert_called_once_with()
        self.assertFalse(compartilhado.vision_ready.value)
        self.assertFalse(compartilhado.line_detected.value)
        self.assertFalse(compartilhado.line_ahead.value)
        self.assertEqual(compartilhado.line_angle.value, 0)
        self.assertEqual(compartilhado.last_bottom_point.value, 224)
        self.assertEqual(compartilhado.line_status.value, "line_detected")
        self.assertEqual(compartilhado.turn_dir.value, "straight")
        self.assertEqual(compartilhado.green_turn_target.value, 0)
        self.assertFalse(compartilhado.preferencia_linha_esquerda.value)


class MissionRecoveryTests(unittest.TestCase):
    @staticmethod
    def _shared():
        valor = lambda inicial: SimpleNamespace(value=inicial)
        return SimpleNamespace(
            terminate=valor(False),
            vision_ready=valor(True),
            line_detected=valor(True),
            line_ahead=valor(True),
            line_angle=valor(90),
            line_angle_y=valor(200),
            line_size=valor(9000.0),
            last_bottom_point=valor(30),
            last_bottom_point_y=valor(250),
            line_status=valor("gap_avoid"),
            turn_dir=valor("right"),
            green_turn_target=valor(1),
            preferencia_linha_esquerda=valor(True),
            line_crop=valor(config.LINE_CROP_GREEN),
            min_line_size=valor(9000),
            entry_armed=valor(False),
            entry_silver_detected=valor(True),
            entry_silver_confirmed=valor(True),
            entry_silver_votes=valor(3),
            entry_silver_reason=valor("confirmada"),
            entry_silver_state=valor(1),
            rescue_requested=valor(True),
            red_finished=valor(True),
            mission_mode=valor(True),
            status=valor("Resgate"),
        )

    def test_recuperacao_rearma_prata_e_reinicia_do_percurso(self):
        class TravaFalsa:
            def __init__(self):
                self.aquisicoes = 0

            def acquire(self):
                self.aquisicoes += 1

        compartilhado = self._shared()
        trava = TravaFalsa()
        sistema = MissionSystem(
            compartilhado,
            motor_lock=trava,
            args=SimpleNamespace(debug=False),
        )
        sistema._lock_held = False

        with (
            patch.object(sistema, "start_line_phase") as iniciar,
            patch("mission.time.sleep"),
        ):
            sistema.reiniciar_missao_do_percurso("Arduino desconectado")

        iniciar.assert_called_once_with()
        self.assertEqual(trava.aquisicoes, 1)
        self.assertTrue(compartilhado.entry_armed.value)
        self.assertFalse(compartilhado.entry_silver_detected.value)
        self.assertFalse(compartilhado.entry_silver_confirmed.value)
        self.assertEqual(compartilhado.entry_silver_votes.value, 0)
        self.assertEqual(compartilhado.entry_silver_state.value, 0)
        self.assertFalse(compartilhado.rescue_requested.value)
        self.assertFalse(compartilhado.red_finished.value)
        self.assertEqual(compartilhado.turn_dir.value, "straight")
        self.assertEqual(compartilhado.line_status.value, "line_detected")


class MissionEntryAdvanceTests(unittest.TestCase):
    def test_entrada_prata_exige_dois_votos_para_evitar_falso_resgate(self):
        self.assertEqual(config.ENTRY_SILVER_VOTES_NEEDED, 2)
        self.assertEqual(config.ENTRY_SILVER_VOTE_WINDOW, 3)
        self.assertEqual(config.ENTRY_SILVER_VALIDATION_S, 0.0)

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

    def test_modelo_de_entrada_pertence_a_camera_de_linha(self):
        self.assertTrue(hasattr(config, "ENTRY_MODEL_PATH"))
        self.assertTrue(hasattr(config, "ENTRY_MODEL_MIN_CONFIDENCE"))
        # E não vaza para o módulo do resgate.
        self.assertFalse(hasattr(cfg, "ENTRY_MODEL_PATH"))

    def test_faixa_preta_pertence_a_camera_de_resgate(self):
        self.assertTrue(hasattr(cfg, "EXIT_BLACK_HSV_MIN"))
        self.assertFalse(hasattr(config, "EXIT_BLACK_HSV_MIN"))

    def test_vitima_e_entrada_usam_modelos_independentes(self):
        self.assertNotEqual(config.ENTRY_MODEL_PATH, cfg.VICTIM_MODEL_PATH)
        self.assertTrue(hasattr(cfg, "BALL_SILVER_SMOOTH_INNER_V_MIN"))
        self.assertFalse(hasattr(config, "VICTIM_MODEL_PATH"))


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
        from shared.dados_compartilhados import (
            entry_armed,
            entry_silver_confirmed,
            entry_silver_detected,
            entry_silver_reason,
            entry_silver_state,
            entry_silver_votes,
        )
        from visao import entrada_missao

        class GatePlaceholder:
            def __init__(self):
                self.arm_states = []

            def set_armed(self, armed):
                self.arm_states.append(bool(armed))

            def update(self, *args, **kwargs):
                raise AssertionError(
                    "o portão não pode ser consultado com a entrada desarmada")

        anterior = entry_armed.value
        anterior_detectada = entry_silver_detected.value
        anterior_confirmada = entry_silver_confirmed.value
        anterior_votos = entry_silver_votes.value
        anterior_motivo = entry_silver_reason.value
        anterior_estado = entry_silver_state.value
        try:
            entry_armed.value = False
            entry_silver_detected.value = True
            entry_silver_confirmed.value = True
            entry_silver_votes.value = 2
            entry_silver_reason.value = "confirmada"
            entry_silver_state.value = 1
            gate = GatePlaceholder()
            entrada_missao.update_entry_silver(gate, None, 0.0)
            self.assertFalse(entry_silver_detected.value)
            self.assertFalse(entry_silver_confirmed.value)
            self.assertEqual(entry_silver_votes.value, 0)
            self.assertEqual(entry_silver_reason.value, "entrada desarmada")
            self.assertEqual(entry_silver_state.value, 0)
            self.assertEqual(gate.arm_states, [False])
        finally:
            entry_armed.value = anterior
            entry_silver_detected.value = anterior_detectada
            entry_silver_confirmed.value = anterior_confirmada
            entry_silver_votes.value = anterior_votos
            entry_silver_reason.value = anterior_motivo
            entry_silver_state.value = anterior_estado


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
