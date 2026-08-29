"""Testes da ordem segura do handoff da missão."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from controle.missao import (  # noqa: E402
    HANDOFF_TO_LINE,
    HANDOFF_TO_RESCUE,
    HandoffError,
    HandoffExecutor,
    index_of,
)
import mission  # noqa: E402


class FakeSystem:
    """Sistema instrumentado: registra chamadas e nunca toca em hardware."""

    def __init__(self, falha_em=None):
        self.calls = []
        self.falha_em = falha_em

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def step():
            self.calls.append(name)
            if name == self.falha_em:
                raise RuntimeError(f"falha simulada em {name}")
        return step


class HandoffToRescueTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem()
        self.log = HandoffExecutor(self.system, HANDOFF_TO_RESCUE).run()

    def test_parar_e_o_primeiro_passo(self):
        self.assertEqual(self.log[0], "stop_motors")

    def test_filhos_encerrados_antes_de_fechar_a_camera_de_linha(self):
        self.assertLess(
            index_of(self.log, "join_line_children"),
            index_of(self.log, "close_line_camera"))
        self.assertLess(
            index_of(self.log, "assert_line_children_dead"),
            index_of(self.log, "close_line_camera"))

    def test_camera_de_linha_fecha_antes_de_abrir_a_de_resgate(self):
        """Requisito absoluto: nunca as duas câmeras abertas ao mesmo tempo."""
        self.assertLess(
            index_of(self.log, "close_line_camera"),
            index_of(self.log, "open_rescue_camera"))

    def test_serial_nunca_tem_dois_donos(self):
        self.assertLess(
            index_of(self.log, "release_serial"),
            index_of(self.log, "open_rescue_serial"))

    def test_lock_liberado_antes_de_ser_readquirido(self):
        self.assertLess(
            index_of(self.log, "release_motor_lock"),
            index_of(self.log, "acquire_rescue_motor_lock"))

    def test_resgate_inicia_por_ultimo(self):
        self.assertEqual(self.log[-1], "start_rescue")


class HandoffToLineTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem()
        self.log = HandoffExecutor(self.system, HANDOFF_TO_LINE).run()

    def test_parar_antes_de_fechar_a_camera_de_resgate(self):
        self.assertLess(
            index_of(self.log, "stop_motors"),
            index_of(self.log, "close_rescue_camera"))

    def test_camera_de_resgate_fecha_antes_de_abrir_a_de_linha(self):
        self.assertLess(
            index_of(self.log, "close_rescue_camera"),
            index_of(self.log, "open_line_camera"))

    def test_linha_e_reacquirida_no_fim(self):
        self.assertEqual(self.log[-1], "reacquire_line")


class HandoffFailureTests(unittest.TestCase):
    def test_falha_interrompe_e_para_os_motores(self):
        system = FakeSystem(falha_em="close_line_camera")
        executor = HandoffExecutor(system, HANDOFF_TO_RESCUE)
        with self.assertRaises(HandoffError):
            executor.run()
        self.assertEqual(executor.failed_step, "close_line_camera")
        # A câmera de resgate nunca chegou a abrir.
        self.assertNotIn("open_rescue_camera", system.calls)
        # E os motores foram parados de novo depois da falha.
        self.assertEqual(system.calls[-1], "stop_motors")

    def test_falha_ao_encerrar_filho_nao_abre_a_camera_de_resgate(self):
        system = FakeSystem(falha_em="join_line_children")
        with self.assertRaises(HandoffError):
            HandoffExecutor(system, HANDOFF_TO_RESCUE).run()
        self.assertNotIn("open_rescue_camera", system.calls)
        self.assertNotIn("start_rescue", system.calls)

    def test_passo_ausente_no_sistema_e_erro_explicito(self):
        class Incompleto:
            def stop_motors(self):
                pass

        with self.assertRaises(HandoffError):
            HandoffExecutor(Incompleto(), HANDOFF_TO_RESCUE).run()


class MissionRedRestartTests(unittest.TestCase):
    class SistemaFalso:
        def __init__(self):
            self.inicios = 0
            self.reinicios = []
            self.esperas_pronto = 0
            self.esperas_fase = 0
            self.encerrado = False

        def start_line_phase(self):
            self.inicios += 1

        def reiniciar_missao_do_percurso(self, motivo):
            self.reinicios.append(motivo)

        def wait_line_ready(self):
            self.esperas_pronto += 1
            return True

        def wait_line_phase(self):
            self.esperas_fase += 1
            if self.esperas_fase == 1:
                return "finished"
            raise KeyboardInterrupt

        def shutdown(self):
            self.encerrado = True

    def test_faixa_vermelha_reinicia_o_percurso_sem_encerrar_mission(self):
        sistema = self.SistemaFalso()
        args = SimpleNamespace(debug=False, rescue_camera_index=None)

        with (
            patch.object(mission, "parse_args", return_value=args),
            patch.object(mission, "MissionSystem", return_value=sistema),
        ):
            codigo = mission.main()

        self.assertEqual(codigo, 130)
        self.assertEqual(sistema.inicios, 1)
        self.assertEqual(sistema.reinicios, ["faixa vermelha final alcancada"])
        self.assertEqual(sistema.esperas_pronto, 2)
        self.assertTrue(sistema.encerrado)


if __name__ == "__main__":
    unittest.main()
