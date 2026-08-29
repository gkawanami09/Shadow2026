"""Garante que uma falha de resgate nunca seja confundida com a volta normal."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


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

    def test_desconexao_do_arduino_encerra_resgate_para_reiniciar_percurso(self):
        # ``mission.py`` trata este resultado como resgate interrompido. Ele
        # não faz o handoff normal de volta, mas reinicia uma sessão de linha
        # que aguarda a serial — sem fechar o supervisor.
        self.assertEqual(
            rescue_return_action(RESCUE_EXIT_ARDUINO_DESCONECTADO),
            RESCUE_RETURN_STOPPED,
        )
        self.assertEqual(rescue_return_action(3), RESCUE_RETURN_STOPPED)
        self.assertEqual(rescue_return_action(4), RESCUE_RETURN_STOPPED)
        self.assertEqual(rescue_return_action(99), RESCUE_RETURN_STOPPED)

    def test_fluxo_da_missao_reinicia_sem_esperar_outra_porta_usb(self):
        fonte = (SHADOW_ROOT / "mission.py").read_text(encoding="utf-8")

        self.assertIn(
            "if returncode == RESCUE_EXIT_ARDUINO_DESCONECTADO:", fonte)
        self.assertIn("reiniciando imediatamente pelo segue-linha", fonte)
        self.assertNotIn("system.aguardar_ciclo_do_arduino(motivo)", fonte)

    def test_falha_serial_tardia_do_resgate_reinicia_o_percurso(self):
        args = type("Args", (), {"gerenciado_pela_missao": True})()
        arduino = type("Arduino", (), {"connected": False})()

        self.assertEqual(
            runtime_error_exit_code(args, arduino, 3),
            EXIT_ARDUINO_DESCONECTADO,
        )

    def test_resgate_precisa_passar_por_reconectando_antes_da_linha(self):
        with self.assertRaisesRegex(RuntimeError, "proibida"):
            mudar_estado(
                EstadoMissao.RESGATE,
                EstadoMissao.SEGUE_LINHA,
            )
        estado = mudar_estado(
            EstadoMissao.RESGATE,
            EstadoMissao.RECONECTANDO,
        )
        self.assertEqual(
            mudar_estado(estado, EstadoMissao.SEGUE_LINHA),
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
