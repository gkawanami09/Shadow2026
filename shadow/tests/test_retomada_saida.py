"""Testes deterministas da retomada da terceira linha na saida do resgate."""

from collections import deque
from types import SimpleNamespace
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.retomada_saida import (  # noqa: E402
    ControladorRetomadaSaida,
    ErroRetomadaSaida,
)
from visao.continuacao_saida import (  # noqa: E402
    DIREITA_BAIXA,
    ESQUERDA_BAIXA,
    NIVEL,
)


class RelogioFalso:
    def __init__(self):
        self.tempo = 0.0

    def monotonic(self):
        return self.tempo

    def sleep(self, duracao):
        self.tempo += max(float(duracao), 0.0)


class ChassiFalso:
    """Arduino e funcao steer apoiados no mesmo estado de movimento."""

    def __init__(self, relogio, falhar_em=None):
        self.relogio = relogio
        self.falhar_em = falhar_em
        self.connected = True
        self.connection_epoch = 1
        self.modo = "parado"
        self.modo_desde = relogio.monotonic()
        self.ativacoes = {}
        self.comandos = []

    def _registrar(self, nome, *argumentos):
        self.comandos.append(
            (self.relogio.monotonic(), nome, *argumentos))

    def _mover(self, modo):
        self.modo = modo
        self.modo_desde = self.relogio.monotonic()
        self.ativacoes[modo] = self.ativacoes.get(modo, 0) + 1

    def acao_direcao(self, *argumentos):
        if not argumentos:
            self._registrar("parar")
            self.modo = "parado"
            self.modo_desde = self.relogio.monotonic()
            return True

        angulo, velocidade = argumentos
        if angulo == 0:
            modo = "avanco"
        elif angulo > 0:
            modo = "tanque_direita"
        else:
            modo = "tanque_esquerda"
        self._registrar("direcao", angulo, velocidade)
        if self.falhar_em == modo:
            return False
        self._mover(modo)
        return True

    def rodas(self, fe, te, fd, td):
        vetor = (fe, te, fd, td)
        if vetor == (-60, 60, 60, -60):
            modo = "omni_esquerda"
        elif vetor == (60, -60, -60, 60):
            modo = "omni_direita"
        else:
            modo = "rodas_desconhecidas"
        self._registrar("rodas", *vetor)
        if self.falhar_em == modo:
            return False
        self._mover(modo)
        return True

    def parar(self):
        self._registrar("parar_arduino")
        self.modo = "parado"
        self.modo_desde = self.relogio.monotonic()
        return True

    def refresh(self, fail_closed=False):
        return self.connected

    def tempo_no_modo(self):
        return self.relogio.monotonic() - self.modo_desde


class EventoDeteccao:
    def __init__(self, modo, ativacao, depois_s, confirmacoes=(True, True)):
        self.modo = modo
        self.ativacao = ativacao
        self.depois_s = float(depois_s)
        self.confirmacoes = tuple(confirmacoes)


class FonteAnalises:
    """Produz pose primeiro e deteccoes ligadas a fases fisicas depois."""

    def __init__(self, chassi, delta_pose, eventos=()):
        self.chassi = chassi
        self.delta_pose = float(delta_pose)
        self.eventos = deque(eventos)
        self.frames = 0
        self.confirmacoes_pendentes = None
        self.instantes_candidato = []

    @staticmethod
    def _continuacao():
        return SimpleNamespace(
            alvo_x=224.0,
            alvo_y=32.0,
            confianca=.95,
            area=1200.0,
            altura_ratio=.55,
            bbox=(205, 10, 38, 130),
        )

    def proxima(self):
        self.frames += 1
        if self.frames <= cfg.EXIT_POST_POSE_WINDOW:
            return SimpleNamespace(
                soleira=SimpleNamespace(delta_y_ratio=self.delta_pose),
                continuacao=None,
            )

        if (
            self.confirmacoes_pendentes is not None
            and self.chassi.modo == "parado"
        ):
            presente = self.confirmacoes_pendentes.popleft()
            if not self.confirmacoes_pendentes:
                self.confirmacoes_pendentes = None
            return SimpleNamespace(
                soleira=None,
                continuacao=(self._continuacao() if presente else None),
            )

        if self.eventos:
            evento = self.eventos[0]
            ativacao = self.chassi.ativacoes.get(evento.modo, 0)
            if (
                self.chassi.modo == evento.modo
                and ativacao == evento.ativacao
                and self.chassi.tempo_no_modo() >= evento.depois_s
            ):
                self.eventos.popleft()
                self.confirmacoes_pendentes = deque(evento.confirmacoes)
                self.instantes_candidato.append(
                    self.chassi.relogio.monotonic())
                return SimpleNamespace(
                    soleira=None,
                    continuacao=self._continuacao(),
                )

        return SimpleNamespace(soleira=None, continuacao=None)


class CameraFalsa:
    def __init__(self, fonte):
        self.fonte = fonte

    def get_frame(self):
        # Captura instantanea: apenas dormir() faz o tempo fisico avancar.
        return self.fonte.proxima()


class AnalisadorFalso:
    def analisar(self, frame):
        return frame


def criar_controlador(delta_pose, eventos=(), falhar_em=None):
    relogio = RelogioFalso()
    chassi = ChassiFalso(relogio, falhar_em=falhar_em)
    fonte = FonteAnalises(chassi, delta_pose, eventos=eventos)
    controlador = ControladorRetomadaSaida(
        camera=CameraFalsa(fonte),
        arduino=chassi,
        acao_direcao=chassi.acao_direcao,
        analisador=AnalisadorFalso(),
        relogio=relogio.monotonic,
        dormir=relogio.sleep,
    )
    return controlador, chassi, relogio, fonte


def primeiro_intervalo(chassi, nome, predicado=lambda _cmd: True):
    for indice, comando in enumerate(chassi.comandos):
        if comando[1] != nome or not predicado(comando):
            continue
        for posterior in chassi.comandos[indice + 1:]:
            if posterior[1] in ("parar", "parar_arduino"):
                return comando[0], posterior[0]
    raise AssertionError(f"nenhum intervalo encontrado para {nome}")


class RetomadaSaidaTests(unittest.TestCase):
    def test_avanco_reto_dura_os_030_segundos_completos(self):
        evento = EventoDeteccao("tanque_esquerda", 1, .04)
        controlador, chassi, _relogio, _fonte = criar_controlador(
            delta_pose=.12, eventos=(evento,))

        resultado = controlador.executar()

        inicio, fim = primeiro_intervalo(
            chassi,
            "direcao",
            lambda comando: comando[2] == 0,
        )
        self.assertAlmostEqual(
            fim - inicio, cfg.EXIT_POST_FORWARD_S, places=9)
        self.assertEqual(resultado.orientacao_soleira, DIREITA_BAIXA)

    def test_diagonal_direita_baixa_gira_tanque_para_esquerda(self):
        evento = EventoDeteccao("tanque_esquerda", 1, .04)
        controlador, chassi, _relogio, _fonte = criar_controlador(
            delta_pose=.12, eventos=(evento,))

        resultado = controlador.executar()

        giros = [
            comando for comando in chassi.comandos
            if comando[1] == "direcao" and comando[2] != 0
        ]
        self.assertTrue(giros)
        self.assertTrue(all(comando[2] == -cfg.EXIT_POST_TANK_ANGLE
                            for comando in giros))
        self.assertTrue(all(comando[3] == cfg.EXIT_POST_TANK_SPEED
                            for comando in giros))
        self.assertEqual(resultado.fase_encontro, "tanque_esquerda")

    def test_diagonal_esquerda_baixa_gira_tanque_para_direita(self):
        evento = EventoDeteccao("tanque_direita", 1, .04)
        controlador, chassi, _relogio, _fonte = criar_controlador(
            delta_pose=-.12, eventos=(evento,))

        resultado = controlador.executar()

        giros = [
            comando for comando in chassi.comandos
            if comando[1] == "direcao" and comando[2] != 0
        ]
        self.assertTrue(giros)
        self.assertTrue(all(comando[2] == cfg.EXIT_POST_TANK_ANGLE
                            for comando in giros))
        self.assertEqual(resultado.fase_encontro, "tanque_direita")

    def test_nivelado_varre_esquerda_1s_e_direita_ate_2s(self):
        controlador, chassi, _relogio, _fonte = criar_controlador(
            delta_pose=0.0)

        with self.assertRaises(ErroRetomadaSaida):
            controlador.executar()

        esquerda = primeiro_intervalo(
            chassi,
            "rodas",
            lambda comando: comando[2:] == (-60, 60, 60, -60),
        )
        direita = primeiro_intervalo(
            chassi,
            "rodas",
            lambda comando: comando[2:] == (60, -60, -60, 60),
        )
        self.assertAlmostEqual(
            esquerda[1] - esquerda[0], cfg.EXIT_POST_OMNI_LEFT_S, places=9)
        self.assertAlmostEqual(
            direita[1] - direita[0], cfg.EXIT_POST_OMNI_RIGHT_S, places=9)
        self.assertEqual(chassi.comandos[-1][1], "parar")

    def test_encontro_durante_esquerda_para_imediatamente_e_nao_vai_direita(self):
        evento = EventoDeteccao("omni_esquerda", 1, .23)
        controlador, chassi, _relogio, fonte = criar_controlador(
            delta_pose=0.0, eventos=(evento,))

        resultado = controlador.executar()

        inicio, fim = primeiro_intervalo(
            chassi,
            "rodas",
            lambda comando: comando[2:] == (-60, 60, 60, -60),
        )
        self.assertEqual(fim, fonte.instantes_candidato[0])
        self.assertLess(fim - inicio, cfg.EXIT_POST_OMNI_LEFT_S)
        self.assertFalse(any(
            comando[1] == "rodas"
            and comando[2:] == (60, -60, -60, 60)
            for comando in chassi.comandos
        ))
        self.assertEqual(resultado.fase_encontro, "omni_esquerda")

    def test_encontro_durante_direita_interrompe_antes_dos_2s(self):
        evento = EventoDeteccao("omni_direita", 1, .37)
        controlador, chassi, _relogio, fonte = criar_controlador(
            delta_pose=0.0, eventos=(evento,))

        resultado = controlador.executar()

        inicio, fim = primeiro_intervalo(
            chassi,
            "rodas",
            lambda comando: comando[2:] == (60, -60, -60, 60),
        )
        self.assertEqual(fim, fonte.instantes_candidato[0])
        self.assertLess(fim - inicio, cfg.EXIT_POST_OMNI_RIGHT_S)
        self.assertEqual(resultado.fase_encontro, "omni_direita")

    def test_um_frame_parado_nao_confirma_e_a_varredura_recomeca(self):
        eventos = (
            EventoDeteccao(
                "omni_esquerda", 1, .10,
                confirmacoes=(True, False, False),
            ),
            EventoDeteccao(
                "omni_esquerda", 2, .10,
                confirmacoes=(True, True),
            ),
        )
        controlador, chassi, _relogio, fonte = criar_controlador(
            delta_pose=0.0, eventos=eventos)

        resultado = controlador.executar()

        comandos_esquerda = [
            comando for comando in chassi.comandos
            if comando[1] == "rodas"
            and comando[2:] == (-60, 60, 60, -60)
        ]
        self.assertEqual(len(comandos_esquerda), 2)
        self.assertEqual(len(fonte.instantes_candidato), 2)
        self.assertEqual(resultado.fase_encontro, "omni_esquerda")

    def test_falha_de_comando_termina_parado(self):
        controlador, chassi, _relogio, _fonte = criar_controlador(
            delta_pose=0.0, falhar_em="omni_esquerda")

        with self.assertRaises(ErroRetomadaSaida):
            controlador.executar()

        self.assertEqual(chassi.comandos[-1][1], "parar")
        self.assertEqual(chassi.modo, "parado")

    def test_timeout_total_interrompe_movimento_e_termina_parado(self):
        controlador, chassi, relogio, _fonte = criar_controlador(
            delta_pose=.12)

        with patch.object(cfg, "EXIT_POST_TOTAL_TIMEOUT_S", .15):
            with self.assertRaisesRegex(
                ErroRetomadaSaida, "timeout total"):
                controlador.executar()

        self.assertGreaterEqual(relogio.tempo, .15)
        self.assertLess(relogio.tempo, cfg.EXIT_POST_FORWARD_S)
        self.assertEqual(chassi.comandos[-1][1], "parar")
        self.assertEqual(chassi.modo, "parado")


if __name__ == "__main__":
    unittest.main()
