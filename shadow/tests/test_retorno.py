"""Contrato da manobra de 180° acionada por dois verdes."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle import retorno  # noqa: E402


class RetornoTests(unittest.TestCase):
    @staticmethod
    def _resultado_visao(
        sequence,
        timestamp,
        *,
        detected=True,
        bottom_x=config.camera_x / 2,
        junction_visible=False,
        branch_token=0,
        branch_valid=False,
        branch_x=None,
        branch_y=config.camera_y - 2,
    ):
        if branch_x is None:
            branch_x = bottom_x
        return SimpleNamespace(
            sequencia=sequence,
            publicado_em=timestamp,
            linha_detectada=detected,
            ponto_inferior_x=bottom_x,
            juncao_topologica_visivel=junction_visible,
            locked_branch_token=branch_token,
            locked_branch_valid=branch_valid,
            locked_branch_bottom_x=branch_x,
            locked_branch_bottom_y=branch_y,
        )

    @staticmethod
    def _monitor_com_progresso(progress, *, wrong=False, timestamp=10.0):
        monitor = retorno._MonitorMpu180(None, None)
        monitor.last_valid_at = timestamp
        monitor.last_progress = SimpleNamespace(
            valid=True,
            wrong_direction=wrong,
            progress_deg=float(progress),
        )
        return monitor

    def test_mpu_180_desacelera_sem_escolher_o_lado(self):
        monitor = self._monitor_com_progresso(
            config.T_180_MPU_SLOWDOWN_DEG,
        )

        self.assertEqual(
            monitor.turn_speed(config.T_180_SEARCH_SPEED, now=10.0),
            min(config.T_180_SEARCH_SPEED, config.T_180_MPU_SLOW_SPEED),
        )
        self.assertFalse(monitor.must_abort(now=10.0))

    def test_mpu_180_exige_progresso_minimo_para_aceitar_camera(self):
        monitor = self._monitor_com_progresso(
            config.T_180_MPU_MIN_COMPLETION_DEG - 1,
        )
        self.assertFalse(monitor.camera_may_finish(now=10.0))

        monitor.last_progress.progress_deg = (
            config.T_180_MPU_MIN_COMPLETION_DEG
        )
        self.assertTrue(monitor.camera_may_finish(now=10.0))

    def test_mpu_180_aborta_sentido_errado_ou_excesso(self):
        wrong = self._monitor_com_progresso(4.0, wrong=True)
        excess = self._monitor_com_progresso(
            config.T_180_MPU_HARD_LIMIT_DEG,
        )

        self.assertTrue(wrong.must_abort(now=10.0))
        self.assertTrue(excess.must_abort(now=10.0))

    def test_mpu_180_velho_nao_bloqueia_a_camera(self):
        monitor = self._monitor_com_progresso(20.0)

        self.assertTrue(monitor.camera_may_finish(now=11.0))
        self.assertFalse(monitor.must_abort(now=11.0))

    def test_mpu_sem_proveniencia_nao_ganha_timestamp_fabricado(self):
        arduino = SimpleNamespace(
            poll_mpu=lambda: (
                True,
                SimpleNamespace(
                    yaw_graus=45., received_at=0., request_generation=0,
                ),
            ),
            iniciar_mpu=lambda **_kwargs: True,
        )
        tracker = SimpleNamespace(update=Mock())
        monitor = retorno._MonitorMpu180(arduino, tracker)

        self.assertIsNone(monitor.poll(now=1.))
        tracker.update.assert_not_called()
        self.assertFalse(monitor.fresh(now=1.))

    def test_saida_180_exige_tres_sequencias_novas(self):
        confirmador = retorno._ConfirmadorSaida180(required_frames=3)
        primeiro = self._resultado_visao(10, 1.0)

        self.assertFalse(confirmador.update(
            primeiro, now=1.0, mpu_allows=True))
        self.assertFalse(confirmador.update(
            primeiro, now=1.01, mpu_allows=True))
        self.assertFalse(confirmador.update(
            self._resultado_visao(11, 1.02),
            now=1.02,
            mpu_allows=True,
        ))
        self.assertTrue(confirmador.update(
            self._resultado_visao(12, 1.04),
            now=1.04,
            mpu_allows=True,
        ))

    def test_camera_only_nao_aceita_linha_central_sem_identidade_lateral(self):
        confirmador = retorno._ConfirmadorSaida180(
            required_frames=3,
            require_side_entry=True,
            expected_side=1,
            expected_branch_token=7,
        )

        for sequence in (1, 2, 3, 4):
            self.assertFalse(confirmador.update(
                self._resultado_visao(
                    sequence,
                    1. + sequence * .02,
                    branch_token=7,
                    branch_valid=True,
                ),
                now=1. + sequence * .02,
                mpu_allows=True,
            ))

    def test_camera_only_exige_direita_e_depois_tres_centrais(self):
        confirmador = retorno._ConfirmadorSaida180(
            required_frames=3,
            require_side_entry=True,
            expected_side=1,
            expected_branch_token=7,
        )
        side_frame = self._resultado_visao(
            1,
            1.0,
            bottom_x=(
                config.camera_x / 2
                + config.GREEN_TURN_SIDE_MIN_ERROR_PX
                + 1
            ),
            junction_visible=True,
            branch_token=7,
            branch_valid=True,
        )
        self.assertFalse(confirmador.update(
            side_frame,
            now=1.0,
            mpu_allows=True,
        ))
        self.assertTrue(confirmador.side_seen)

        self.assertFalse(confirmador.update(
            self._resultado_visao(
                2, 1.02, branch_token=7, branch_valid=True),
            now=1.02,
            mpu_allows=True,
        ))
        self.assertFalse(confirmador.update(
            self._resultado_visao(
                3, 1.04, branch_token=7, branch_valid=True),
            now=1.04,
            mpu_allows=True,
        ))
        self.assertTrue(confirmador.update(
            self._resultado_visao(
                4, 1.06, branch_token=7, branch_valid=True),
            now=1.06,
            mpu_allows=True,
        ))

    def test_180_nao_mistura_ramos_de_tokens_diferentes(self):
        confirmador = retorno._ConfirmadorSaida180(
            required_frames=3,
            require_side_entry=True,
            expected_side=1,
            expected_branch_token=7,
        )
        self.assertFalse(confirmador.update(
            self._resultado_visao(
                1,
                1.0,
                bottom_x=(config.camera_x / 2 + 80),
                branch_token=7,
                branch_valid=True,
            ),
            now=1.0,
            mpu_allows=True,
        ))
        self.assertTrue(confirmador.side_seen)

        for sequence in (2, 3, 4):
            self.assertFalse(confirmador.update(
                self._resultado_visao(
                    sequence,
                    1. + sequence * .02,
                    branch_token=8,
                    branch_valid=True,
                ),
                now=1. + sequence * .02,
                mpu_allows=True,
            ))
        self.assertEqual(confirmador.aligned_frames, 0)

    def test_saida_180_reseta_com_frame_velho_desalinhado_ou_mpu(self):
        confirmador = retorno._ConfirmadorSaida180(required_frames=3)
        self.assertFalse(confirmador.update(
            self._resultado_visao(1, 1.0), now=1.0, mpu_allows=True))
        self.assertFalse(confirmador.update(
            self._resultado_visao(2, 1.02), now=1.02, mpu_allows=True))
        self.assertFalse(confirmador.update(
            self._resultado_visao(3, 1.04, detected=False),
            now=1.04,
            mpu_allows=True,
        ))
        self.assertEqual(confirmador.aligned_frames, 0)

        self.assertFalse(confirmador.update(
            self._resultado_visao(4, 1.06), now=1.06, mpu_allows=False))
        self.assertEqual(confirmador.aligned_frames, 0)

        self.assertFalse(confirmador.update(
            self._resultado_visao(5, 1.08), now=2.0, mpu_allows=True))
        self.assertEqual(confirmador.aligned_frames, 0)

    def test_saida_180_nao_confirma_enquanto_juncao_ainda_aparece(self):
        confirmador = retorno._ConfirmadorSaida180(required_frames=3)
        for sequence in (1, 2, 3):
            self.assertFalse(confirmador.update(
                self._resultado_visao(
                    sequence,
                    1. + sequence * .02,
                    junction_visible=True,
                ),
                now=1. + sequence * .02,
                mpu_allows=True,
            ))
        self.assertEqual(confirmador.aligned_frames, 0)

    def test_retorno_sempre_gira_direita_com_trecho_cego_e_re_curta(self):
        previous_size = retorno.line_size.value
        previous_timeout = retorno.T_180_SEARCH_TIMEOUT
        try:
            retorno.line_size.value = config.TURN_AROUND_SMALL_LINE
            # Não é necessário esperar a busca para validar os comandos
            # temporizados; zera o timeout somente neste teste.
            retorno.T_180_SEARCH_TIMEOUT = 0
            with patch.object(retorno, "steer") as steer, \
                    patch.object(
                        retorno, "_turn_for", wraps=retorno._turn_for,
                    ) as turn_for, \
                    patch.object(
                        retorno, "_wait_interruptible", return_value=True,
                    ) as wait:
                next_direction = retorno.turn_around("l")

            self.assertEqual(steer.call_args_list[0].args, (0, .7))
            durations = [call.args[0] for call in wait.call_args_list]
            self.assertEqual(durations[0], config.TURN_AROUND_PREROLL)
            self.assertEqual(config.TURN_AROUND_PREROLL, .275)
            pivots = [call.args for call in steer.call_args_list
                      if call.args and abs(call.args[0]) == 180]
            self.assertTrue(pivots)
            self.assertTrue(all(args[0] == 180 for args in pivots))
            self.assertEqual(next_direction, "r")
            turn_durations = [
                chamada.args[0] for chamada in turn_for.call_args_list
            ]
            self.assertIn(config.T_180_BLIND_EXTRA, turn_durations)
            self.assertEqual(config.T_180_BLIND_EXTRA, .10)
            self.assertIn(config.TURN_AROUND_REVERSE, durations)
            self.assertEqual(config.TURN_AROUND_REVERSE, .15)
        finally:
            retorno.line_size.value = previous_size
            retorno.T_180_SEARCH_TIMEOUT = previous_timeout

    def test_retorno_competitivo_falha_fechado_sem_linha_de_saida(self):
        previous_timeout = retorno.T_180_SEARCH_TIMEOUT
        try:
            retorno.T_180_SEARCH_TIMEOUT = 0
            with patch.object(retorno, "steer") as steer, \
                    patch.object(retorno, "sleep_steering"):
                resultado = retorno.turn_around(
                    "l", require_alignment=True)

            self.assertIsNone(resultado)
            self.assertFalse(any(
                chamada.args and chamada.args[0] == 200
                for chamada in steer.call_args_list
            ))
        finally:
            retorno.T_180_SEARCH_TIMEOUT = previous_timeout

    def test_retorno_aborta_se_a_trava_cortar_uma_espera(self):
        with patch.object(retorno, "steer") as steer, \
                patch.object(retorno, "sleep_steering", return_value=False):
            resultado = retorno.turn_around("l", require_alignment=True)

        self.assertIsNone(resultado)
        self.assertEqual(steer.call_args_list[0].args, (0, .7))
        self.assertEqual(steer.call_args_list[-1].args, ())
        self.assertFalse(any(
            chamada.args and chamada.args[0] == 180
            for chamada in steer.call_args_list
        ))

    def test_retorno_respeita_aborto_antes_do_primeiro_movimento(self):
        with patch.object(retorno, "steer") as steer, \
                patch.object(retorno, "sleep_steering") as sleep:
            resultado = retorno.turn_around(
                "l",
                require_alignment=True,
                should_abort=lambda: True,
            )

        self.assertIsNone(resultado)
        self.assertEqual(steer.call_args_list, [call()])
        sleep.assert_not_called()
