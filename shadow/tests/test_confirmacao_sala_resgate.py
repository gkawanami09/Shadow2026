"""Laço de confirmação da sala dentro de ``resgate.py``, com dublês.

O avaliador puro é testado em ``test_confirmacao_entrada``. Aqui o que está
sob teste é a integração: quem alimenta o avaliador, o que acontece com o
portão e os marcadores depois da janela, e se o robô recua quando a sala não
aparece.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import types
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
import resgate  # noqa: E402


@dataclass(frozen=True)
class FakeDeteccao:
    confirmed: bool
    kind: str = "silver"
    hits: int = 3


@dataclass
class FakePacote:
    sequence: int
    frame: object
    captured_at: float


class FakeCaptura:
    """Entrega frames numerados; nunca termina sozinha."""

    def __init__(self, frames=200):
        self.restantes = frames
        self.sequencia = 0
        self.ended = False

    def poll(self, after_sequence=0):
        if self.restantes <= 0:
            self.ended = True
            return None
        self.restantes -= 1
        self.sequencia += 1
        return FakePacote(self.sequencia, object(), float(self.sequencia))


class FakeTrabalhador:
    """Devolve sempre a mesma detecção, com sequência crescente."""

    def __init__(self, deteccao=None):
        self.deteccao = deteccao
        self.is_alive = True
        self.enviados = 0
        self.sequencia = 0
        self.resets = 0

    def submit(self, frame, captured_at=None, source_sequence=None):
        self.enviados += 1

    def poll(self, after_sequence=0):
        self.sequencia += 1
        return types.SimpleNamespace(
            sequence=self.sequencia, detection=self.deteccao)

    def reset_tracking(self):
        self.resets += 1


class FakePortao:
    """Repassa a detecção sem alterar; conta os resets."""

    def __init__(self):
        self.resets = 0

    def accept(self, detection):
        return detection

    def reset(self):
        self.resets += 1


class FakeMarcadores:
    def __init__(self, deteccoes=None):
        self.deteccoes = deteccoes or {"green": None, "red": None}
        self.resets = 0

    def update(self, frame, timestamp):
        return dict(self.deteccoes)

    def reset(self):
        self.resets += 1


class FakeArduino:
    def __init__(self):
        self.connected = True
        self.connection_epoch = 1
        self.refreshes = 0

    def refresh(self, fail_closed=False):
        self.refreshes += 1


class FakeDirecao:
    """Registra cada comando enviado aos motores."""

    def __init__(self):
        self.comandos = []

    def __call__(self, angle=190., speed=.8, **kwargs):
        self.comandos.append((angle, speed))
        return True

    @property
    def deu_re(self):
        return any(angulo == 200 for angulo, _ in self.comandos)


def _args(drive=True, missao=True):
    return types.SimpleNamespace(
        drive=drive, gerenciado_pela_missao=missao)


class ConfirmacaoSalaTests(unittest.TestCase):
    def setUp(self):
        self.janela = cfg.MISSION_ENTRY_CONFIRM_S
        self.recuo = cfg.MISSION_ENTRY_RETREAT_S
        # Janelas curtas para o teste não gastar segundos de relógio real.
        cfg.MISSION_ENTRY_CONFIRM_S = 0.20
        cfg.MISSION_ENTRY_RETREAT_S = 0.02

    def tearDown(self):
        cfg.MISSION_ENTRY_CONFIRM_S = self.janela
        cfg.MISSION_ENTRY_RETREAT_S = self.recuo

    def _rodar(self, deteccao=None, marcadores=None, args=None):
        captura = FakeCaptura()
        trabalhador = FakeTrabalhador(deteccao)
        portao = FakePortao()
        marc = FakeMarcadores(marcadores)
        arduino = FakeArduino()
        direcao = FakeDirecao()
        ok = resgate._confirmar_sala_de_resgate(
            args or _args(), captura, trabalhador, portao, marc,
            arduino, direcao)
        return ok, trabalhador, portao, marc, direcao

    def test_vitima_confirmada_libera_o_resgate(self):
        ok, _t, _p, _m, direcao = self._rodar(
            deteccao=FakeDeteccao(True, "silver"))
        self.assertTrue(ok)
        self.assertFalse(direcao.deu_re, "não deveria recuar com a sala boa")

    def test_triangulo_confirmado_libera_o_resgate(self):
        ok, _t, _p, _m, direcao = self._rodar(
            marcadores={"green": FakeDeteccao(True), "red": None})
        self.assertTrue(ok)
        self.assertFalse(direcao.deu_re)

    def test_sala_vazia_reprova_e_recua(self):
        ok, _t, _p, _m, direcao = self._rodar()
        self.assertFalse(ok)
        self.assertTrue(direcao.deu_re, "o robô precisa dar ré na recusa")
        # Depois do recuo os motores param; nunca terminamos andando.
        self.assertEqual(direcao.comandos[-1][0], 190)

    def test_candidato_nao_confirmado_reprova(self):
        ok, _t, _p, _m, _d = self._rodar(
            deteccao=FakeDeteccao(False),
            marcadores={"green": FakeDeteccao(False), "red": None})
        self.assertFalse(ok)

    def test_janela_deixa_o_resgate_com_o_estado_limpo(self):
        """O resgate começa como começaria sem esta etapa."""
        _ok, trabalhador, portao, marc, _d = self._rodar(
            deteccao=FakeDeteccao(True))
        self.assertEqual(portao.resets, 1)
        self.assertEqual(marc.resets, 1)
        self.assertEqual(trabalhador.resets, 1)

    def test_sem_missao_a_etapa_nao_roda(self):
        """`resgate.py` aberto sozinho não pode ganhar um passo novo."""
        ok, trabalhador, _p, _m, direcao = self._rodar(
            args=_args(missao=False))
        self.assertTrue(ok)
        self.assertEqual(trabalhador.enviados, 0)
        self.assertEqual(direcao.comandos, [])

    def test_sem_drive_a_etapa_nao_roda(self):
        ok, trabalhador, _p, _m, _d = self._rodar(args=_args(drive=False))
        self.assertTrue(ok)
        self.assertEqual(trabalhador.enviados, 0)

    def test_desligada_por_configuracao(self):
        cfg.MISSION_ENTRY_CONFIRM_ENABLED = False
        try:
            ok, trabalhador, _p, _m, _d = self._rodar()
        finally:
            cfg.MISSION_ENTRY_CONFIRM_ENABLED = True
        self.assertTrue(ok)
        self.assertEqual(trabalhador.enviados, 0)

    def test_serial_trocada_durante_a_janela_e_erro(self):
        """Perder a serial no meio não pode virar 'sala confirmada'."""
        class ArduinoInstavel(FakeArduino):
            def refresh(self, fail_closed=False):
                self.refreshes += 1
                if self.refreshes > 2:
                    self.connection_epoch = 99

        with self.assertRaises(RuntimeError):
            resgate._confirmar_sala_de_resgate(
                _args(), FakeCaptura(), FakeTrabalhador(), FakePortao(),
                FakeMarcadores(), ArduinoInstavel(), FakeDirecao())

    def test_detector_morto_durante_a_janela_e_erro(self):
        trabalhador = FakeTrabalhador()
        trabalhador.is_alive = False
        with self.assertRaises(RuntimeError):
            resgate._confirmar_sala_de_resgate(
                _args(), FakeCaptura(), trabalhador, FakePortao(),
                FakeMarcadores(), FakeArduino(), FakeDirecao())


class CodigoDeSaidaTests(unittest.TestCase):
    def test_codigo_de_entrada_falsa_e_unico(self):
        codigos = {
            resgate.EXIT_OK,
            resgate.EXIT_INCOMPLETE,
            resgate.EXIT_SEM_MODELO,
            resgate.EXIT_ENTRADA_FALSA,
        }
        self.assertEqual(len(codigos), 4)

    def test_supervisor_le_o_mesmo_codigo(self):
        import mission
        self.assertEqual(
            mission.RESCUE_EXIT_FALSE_ENTRY, resgate.EXIT_ENTRADA_FALSA)


if __name__ == "__main__":
    unittest.main()
