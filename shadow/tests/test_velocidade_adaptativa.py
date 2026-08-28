"""Testes das travas que permitem acelerar somente em reta segura."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from controle.velocidade_adaptativa import (  # noqa: E402
    ControladorVelocidadeAdaptativa,
)


class RelogioFalso:
    def __init__(self):
        self.tempo = 0.

    def __call__(self):
        return self.tempo


def resultado_reto(sequencia, publicado_em, **alteracoes):
    dados = {
        "sequencia": sequencia,
        "publicado_em": publicado_em,
        "processamento_ms": 8.,
        "linha_detectada": True,
        "linha_a_frente": True,
        "angulo": 0.,
        "ponto_inferior_x": config.camera_x / 2,
        "ponto_inferior_y": config.camera_y - 1,
        "area_linha": 9000.,
        "candidato_verde": False,
        "candidato_vermelho": False,
    }
    dados.update(alteracoes)
    return SimpleNamespace(**dados)


def alimentar_reta(controlador, relogio, quantidade, fps=60, inicio=1):
    velocidades = []
    for sequencia in range(inicio, inicio + quantidade):
        relogio.tempo = sequencia / fps
        velocidades.append(controlador.atualizar(
            resultado_reto(sequencia, relogio.tempo),
            velocidade_base=config.LINE_FOLLOW_SPEED,
            direcao="straight",
            permitir_rapido=True,
        ))
    return velocidades


class VelocidadeAdaptativaTests(unittest.TestCase):
    def setUp(self):
        # O robô usa PWM 80 fixo. Para continuar testando o controlador
        # opcional isoladamente, simulamos aqui um teto maior que a base.
        self.teto_teste = .75
        patcher = mock.patch(
            "controle.velocidade_adaptativa.VELOCIDADE_RETA_RAPIDA",
            self.teto_teste,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_configuracao_principal_usa_pwm_70_sem_modo_adaptativo(self):
        self.assertEqual(config.LINE_FOLLOW_PWM, 70)
        self.assertAlmostEqual(
            config.LINE_FOLLOW_SPEED * config.MAX_PWM,
            70.,
        )
        self.assertFalse(config.RETA_RAPIDA_HABILITADA)

    def test_verde_mantem_as_velocidades_ja_calibradas(self):
        self.assertEqual(
            round(config.GREEN_APPROACH_SPEED * config.MAX_PWM),
            60,
        )
        self.assertEqual(
            round(config.GREEN_TURN_SPEED * config.MAX_PWM),
            60,
        )

    def test_seis_frames_a_sessenta_fps_liberam_primeiro_passo(self):
        relogio = RelogioFalso()
        controlador = ControladorVelocidadeAdaptativa(relogio)

        velocidades = alimentar_reta(controlador, relogio, 6)

        self.assertEqual(
            velocidades[:5],
            [config.LINE_FOLLOW_SPEED] * 5,
        )
        self.assertAlmostEqual(
            velocidades[5],
            config.LINE_FOLLOW_SPEED
            + config.PASSO_VELOCIDADE_RETA_RAPIDA,
        )
        self.assertTrue(controlador.modo_rapido)
        self.assertAlmostEqual(controlador.fps_visao, 60., delta=.01)

    def test_sobe_gradualmente_e_nunca_ultrapassa_o_teto(self):
        relogio = RelogioFalso()
        controlador = ControladorVelocidadeAdaptativa(relogio)

        velocidades = alimentar_reta(controlador, relogio, 25)

        self.assertAlmostEqual(
            velocidades[5],
            config.LINE_FOLLOW_SPEED
            + config.PASSO_VELOCIDADE_RETA_RAPIDA,
        )
        self.assertLess(velocidades[14], self.teto_teste)
        self.assertAlmostEqual(velocidades[-1],
                               self.teto_teste)
        self.assertLessEqual(max(velocidades),
                             self.teto_teste)

    def test_quadro_repetido_nao_confirma_nem_aumenta_velocidade(self):
        relogio = RelogioFalso()
        controlador = ControladorVelocidadeAdaptativa(relogio)
        alimentar_reta(controlador, relogio, 6)
        velocidade_antes = controlador.velocidade

        relogio.tempo += .005
        velocidade_depois = controlador.atualizar(
            resultado_reto(6, relogio.tempo - .005),
            velocidade_base=config.LINE_FOLLOW_SPEED,
            direcao="straight",
            permitir_rapido=True,
        )

        self.assertEqual(velocidade_depois, velocidade_antes)

    def test_quarenta_fps_nunca_libera_velocidade_extra(self):
        relogio = RelogioFalso()
        controlador = ControladorVelocidadeAdaptativa(relogio)

        velocidades = alimentar_reta(controlador, relogio, 30, fps=40)

        self.assertEqual(
            velocidades,
            [config.LINE_FOLLOW_SPEED] * len(velocidades),
        )
        self.assertFalse(controlador.modo_rapido)

    def test_frame_velho_cancela_imediatamente(self):
        relogio = RelogioFalso()
        controlador = ControladorVelocidadeAdaptativa(relogio)
        alimentar_reta(controlador, relogio, 8)
        self.assertTrue(controlador.modo_rapido)

        relogio.tempo += config.IDADE_MAXIMA_VISAO_RAPIDA_S + .001
        velocidade = controlador.atualizar(
            resultado_reto(8, relogio.tempo - .1),
            velocidade_base=config.LINE_FOLLOW_SPEED,
            direcao="straight",
            permitir_rapido=True,
        )

        self.assertEqual(velocidade, config.LINE_FOLLOW_SPEED)
        self.assertFalse(controlador.modo_rapido)

    def test_cada_condicao_de_seguranca_cancela_o_modo_rapido(self):
        casos = {
            "direcao": ({"direcao": "left"}, {}),
            "verde": ({}, {"candidato_verde": True}),
            "vermelho": ({}, {"candidato_vermelho": True}),
            "sem_linha": ({}, {"linha_detectada": False}),
            "sem_continuacao": ({}, {"linha_a_frente": False}),
            "angulo": ({}, {
                "angulo": config.ANGULO_MAXIMO_RETA_RAPIDA + 1,
            }),
            "fora_do_centro": ({}, {
                "ponto_inferior_x": (
                    config.camera_x / 2
                    + config.ERRO_INFERIOR_RETA_RAPIDA_PX + 1
                ),
            }),
            "ponto_alto": ({}, {
                "ponto_inferior_y": (
                    config.camera_y
                    * config.ALTURA_MINIMA_PONTO_INFERIOR_RAPIDA - 1
                ),
            }),
            "area_pequena": ({}, {
                "area_linha": config.AREA_MINIMA_LINHA_RAPIDA - 1,
            }),
            "nao_permitido": ({"permitir_rapido": False}, {}),
            "velocidade_especial": ({"velocidade_base": .4}, {}),
        }

        for nome, (argumentos, alteracoes) in casos.items():
            with self.subTest(nome=nome):
                relogio = RelogioFalso()
                controlador = ControladorVelocidadeAdaptativa(relogio)
                alimentar_reta(controlador, relogio, 8)
                self.assertTrue(controlador.modo_rapido)
                relogio.tempo += 1 / 60
                velocidade = controlador.atualizar(
                    resultado_reto(9, relogio.tempo, **alteracoes),
                    velocidade_base=argumentos.get(
                        "velocidade_base", config.LINE_FOLLOW_SPEED),
                    direcao=argumentos.get("direcao", "straight"),
                    permitir_rapido=argumentos.get(
                        "permitir_rapido", True),
                )
                self.assertEqual(
                    velocidade,
                    argumentos.get(
                        "velocidade_base", config.LINE_FOLLOW_SPEED),
                )
                self.assertFalse(controlador.modo_rapido)

    def test_salto_de_angulo_ou_ponto_reinicia_confirmacao(self):
        for alteracao in (
            {"angulo": config.VARIACAO_ANGULO_RETA_RAPIDA + 1},
            {
                "ponto_inferior_x": (
                    config.camera_x / 2
                    + config.VARIACAO_INFERIOR_RETA_RAPIDA_PX + 1
                ),
            },
        ):
            with self.subTest(alteracao=alteracao):
                relogio = RelogioFalso()
                controlador = ControladorVelocidadeAdaptativa(relogio)
                alimentar_reta(controlador, relogio, 5)
                relogio.tempo = 6 / 60
                velocidade = controlador.atualizar(
                    resultado_reto(6, relogio.tempo, **alteracao),
                    velocidade_base=config.LINE_FOLLOW_SPEED,
                    direcao="straight",
                    permitir_rapido=True,
                )
                self.assertEqual(velocidade, config.LINE_FOLLOW_SPEED)
                self.assertFalse(controlador.modo_rapido)

    def test_timestamp_futuro_ou_parado_nao_autoriza(self):
        relogio = RelogioFalso()
        controlador = ControladorVelocidadeAdaptativa(relogio)
        velocidades = []
        for sequencia in range(1, 15):
            relogio.tempo = 1.
            velocidades.append(controlador.atualizar(
                resultado_reto(sequencia, 2.),
                velocidade_base=config.LINE_FOLLOW_SPEED,
                direcao="straight",
                permitir_rapido=True,
            ))

        self.assertEqual(
            velocidades,
            [config.LINE_FOLLOW_SPEED] * len(velocidades),
        )
        self.assertFalse(controlador.modo_rapido)


if __name__ == "__main__":
    unittest.main()
