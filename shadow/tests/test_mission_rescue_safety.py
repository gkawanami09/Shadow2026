"""Garante que uma falha de resgate nunca seja confundida com a volta normal."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from mission import (  # noqa: E402
    EstadoMissao,
    MissionSystem,
    RESCUE_EXIT_ARDUINO_DESCONECTADO,
    RESCUE_EXIT_OK,
    RESCUE_RETURN_COMPLETED,
    RESCUE_RETURN_STOPPED,
    mudar_estado,
    rescue_return_action,
)
from resgate import (  # noqa: E402
    EXIT_ARDUINO_DESCONECTADO,
    EXIT_OK,
    _iniciar_busca_segura,
    _saida_concluida_libera_missao,
    _validar_comando_da_busca,
    runtime_error_exit_code,
)
from controle.aproximacao_resgate import MotionCommand  # noqa: E402


class RescueReturnSafetyTests(unittest.TestCase):
    def test_apenas_resgate_concluido_pode_voltar_normalmente_ao_percurso(self):
        self.assertEqual(
            rescue_return_action(RESCUE_EXIT_OK),
            RESCUE_RETURN_COMPLETED,
        )

    def test_desconexao_do_arduino_bloqueia_reinicio_do_percurso(self):
        self.assertEqual(
            rescue_return_action(RESCUE_EXIT_ARDUINO_DESCONECTADO),
            RESCUE_RETURN_STOPPED,
        )
        self.assertEqual(rescue_return_action(3), RESCUE_RETURN_STOPPED)
        self.assertEqual(rescue_return_action(4), RESCUE_RETURN_STOPPED)
        self.assertEqual(rescue_return_action(99), RESCUE_RETURN_STOPPED)

    def test_ciclo_fisico_sustentado_do_arduino_libera_nova_tentativa(self):
        sistema = MissionSystem(None, None, SimpleNamespace())
        portas = iter((set(), {"/dev/ttyACM0"}, set(), {"/dev/ttyACM0"}))

        with (
            patch.object(
                sistema, "_portas_arduino_presentes", side_effect=portas),
            patch("mission.time.sleep"),
            patch("mission.time.monotonic", side_effect=(0.0, 1.0, 2.0, 6.0)),
        ):
            sistema.aguardar_ciclo_do_arduino("teste")

    def test_fault_stop_do_percurso_exige_rearme_fisico_antes_de_limpar(self):
        shared = SimpleNamespace(
            green_fault_stop=SimpleNamespace(value=True),
        )
        sistema = MissionSystem(shared, None, SimpleNamespace())
        sistema.aguardar_ciclo_do_arduino = Mock()

        self.assertTrue(sistema.exigir_rearme_fisico_se_fault_stop("serial"))
        sistema.aguardar_ciclo_do_arduino.assert_called_once_with("serial")
        self.assertTrue(shared.green_fault_stop.value)

    def test_filho_sem_fault_stop_nao_exige_ciclo_fisico(self):
        shared = SimpleNamespace(
            green_fault_stop=SimpleNamespace(value=False),
        )
        sistema = MissionSystem(shared, None, SimpleNamespace())
        sistema.aguardar_ciclo_do_arduino = Mock()

        self.assertFalse(sistema.exigir_rearme_fisico_se_fault_stop("fim"))
        sistema.aguardar_ciclo_do_arduino.assert_not_called()

    def test_reinicio_checa_fault_stop_antes_de_limpar_compartilhados(self):
        shared = SimpleNamespace(
            terminate=SimpleNamespace(value=False),
            green_fault_stop=SimpleNamespace(value=True),
        )
        sistema = MissionSystem(shared, None, SimpleNamespace())
        ordem = []
        sistema._encerrar_resgate_para_recuperacao = Mock()
        sistema._preparar_nova_tentativa = Mock(
            side_effect=lambda: ordem.append("clear"))
        sistema.start_line_phase = Mock()
        sistema.exigir_rearme_fisico_se_fault_stop = Mock(
            side_effect=lambda _motivo: ordem.append("gate") or True)
        sistema.children = []

        with patch("mission.time.sleep"):
            sistema.reiniciar_missao_do_percurso("serial")

        sistema.exigir_rearme_fisico_se_fault_stop.assert_called_once_with(
            "serial")
        sistema._preparar_nova_tentativa.assert_called_once_with()
        self.assertEqual(ordem, ["gate", "clear"])

    def test_morte_da_visao_durante_manobra_promove_fault_stop(self):
        shared = SimpleNamespace(
            rescue_requested=SimpleNamespace(value=False),
            red_finished=SimpleNamespace(value=False),
            terminate=SimpleNamespace(value=False),
            status=SimpleNamespace(value="Girando"),
            green_control_state=SimpleNamespace(value=3),
            green_locked_decision=SimpleNamespace(value=4),
            green_fault_stop=SimpleNamespace(value=False),
        )
        sistema = MissionSystem(shared, None, SimpleNamespace())
        sistema.children = [SimpleNamespace(is_alive=lambda: False)]

        self.assertEqual(sistema.wait_line_phase(), "child_died")
        self.assertTrue(shared.green_fault_stop.value)

    def test_falha_serial_tardia_do_resgate_reinicia_o_percurso(self):
        args = type("Args", (), {"gerenciado_pela_missao": True})()
        arduino = type("Arduino", (), {"connected": False})()

        self.assertEqual(
            runtime_error_exit_code(args, arduino, 3),
            EXIT_ARDUINO_DESCONECTADO,
        )

    def test_resgate_nao_pode_ir_direto_para_segue_linha(self):
        with self.assertRaisesRegex(RuntimeError, "proibida"):
            mudar_estado(
                EstadoMissao.RESGATE,
                EstadoMissao.SEGUE_LINHA,
            )

    def test_so_finalizacao_concluida_libera_segue_linha(self):
        estado = mudar_estado(
            EstadoMissao.RESGATE,
            EstadoMissao.FINALIZANDO_RESGATE,
        )
        self.assertEqual(
            mudar_estado(estado, EstadoMissao.SEGUE_LINHA),
            EstadoMissao.SEGUE_LINHA,
        )

    def test_terminal_normal_exige_saida_preta_confirmada(self):
        args = SimpleNamespace(drive=True)
        concluido = MotionCommand("EXIT_BLACK_CONFIRMED", terminal=True)
        falha = MotionCommand("SEARCH_FAULT", terminal=True)

        self.assertTrue(
            _saida_concluida_libera_missao(args, EXIT_OK, concluido))
        self.assertFalse(
            _saida_concluida_libera_missao(args, 3, concluido))
        self.assertFalse(
            _saida_concluida_libera_missao(args, EXIT_OK, falha))


class RescueSearchMotorSafetyTests(unittest.TestCase):
    class ArduinoFalso:
        def __init__(self):
            self.paradas = 0

        def parar(self):
            self.paradas += 1
            return True

    def test_nova_busca_substitui_comando_anterior_por_parar(self):
        arduino = self.ArduinoFalso()

        busca = _iniciar_busca_segura(arduino, start_time=0.0)

        self.assertEqual(arduino.paradas, 1)
        self.assertIsNotNone(busca)

    def test_busca_aceita_somente_giro_ou_parada(self):
        _validar_comando_da_busca(MotionCommand("SEARCH", angle=180))
        _validar_comando_da_busca(MotionCommand("SEARCH_STOP", angle=190))
        with self.assertRaisesRegex(RuntimeError, "proibido"):
            _validar_comando_da_busca(
                MotionCommand("SEARCH_BUG", angle=0, speed=.5))


class CentralMissionOwnershipTests(unittest.TestCase):
    def test_mission_chama_resgate_diretamente(self):
        args = SimpleNamespace(
            rescue_camera_index=0,
            debug=False,
        )
        sistema = MissionSystem(None, None, args)
        sistema.start_rescue()

        with patch("resgate.executar_resgate", return_value=EXIT_OK) as executar:
            self.assertEqual(sistema.wait_rescue(), EXIT_OK)

        executar.assert_called_once_with(
            unittest.mock.ANY)
        argumentos = executar.call_args.args[0]
        self.assertTrue(argumentos.gerenciado_pela_missao)
        self.assertTrue(argumentos.drive)
        self.assertFalse(sistema.resgate_ativo)


if __name__ == "__main__":
    unittest.main()
