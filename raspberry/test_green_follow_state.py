"""Testes deterministas da maquina de estados verde no follow destinos."""

import unittest

import follow_destinos as follow


def memoria():
    return {"correcao_anterior": 12, "vel_anterior": (30, 31)}


def destino(ok=True):
    return {"ok": ok}


def acao(valor="PREPARAR_DIREITA", verde="DIREITA"):
    return {"acao_visual": valor, "verde_acionavel": verde}


class GreenFollowStateTests(unittest.TestCase):
    def confirmar(self, estado, move, inicio=0.0):
        dados = memoria()
        for indice in range(follow.GREEN_FOLLOW_CONFIRM_FRAMES):
            follow.atualizar_estado_verde(estado, acao(), destino(), "FOLLOW_DESTINO", move, inicio + indice, dados)
        return dados

    def test_sem_green_move_nao_gera_override(self):
        estado = follow.criar_estado_verde(0.0)
        self.confirmar(estado, False)
        self.assertEqual(estado["estado"], "GREEN_COOLDOWN")
        self.assertIsNone(follow.comando_override_verde(estado, destino(), False))

    def test_green_move_confirmado_entra_approach(self):
        estado = follow.criar_estado_verde(0.0)
        self.confirmar(estado, True)
        esperado = "GREEN_APPROACH" if follow.GREEN_APPROACH_BEFORE_TURN_SEC > 0 else "GREEN_TURNING"
        self.assertEqual(estado["estado"], esperado)

    def test_acao_instavel_nao_confirma(self):
        estado = follow.criar_estado_verde(0.0)
        dados = memoria()
        follow.atualizar_estado_verde(estado, acao(), destino(), "FOLLOW_DESTINO", True, 0.0, dados)
        follow.atualizar_estado_verde(estado, acao("SEGUIR_RETO", "NENHUM"), destino(), "FOLLOW_DESTINO", True, 1.0, dados)
        follow.atualizar_estado_verde(estado, acao("SEGUIR_RETO", "NENHUM"), destino(), "FOLLOW_DESTINO", True, 2.0, dados)
        self.assertEqual(estado["estado"], "GREEN_IDLE")

    def test_seguir_reto_e_recuperacao_nao_iniciam(self):
        for estado_follow, dado in (("FOLLOW_DESTINO", acao("SEGUIR_RETO", "NENHUM")), ("RECUPERAR_GIRO", acao())):
            with self.subTest(estado_follow=estado_follow):
                estado = follow.criar_estado_verde(0.0)
                follow.atualizar_estado_verde(estado, dado, destino(), estado_follow, True, 0.0, memoria())
                self.assertEqual(estado["estado"], "GREEN_IDLE")

    def test_giro_respeita_minimo_e_sai_por_timeout(self):
        estado = follow.criar_estado_verde(0.0)
        estado.update({"estado": "GREEN_TURNING", "acao_confirmada": "PREPARAR_DIREITA", "inicio_estado": 0.0})
        follow.atualizar_estado_verde(estado, acao(), destino(), "FOLLOW_DESTINO", True, follow.GREEN_TURN_MIN_SEC - 0.01, memoria())
        self.assertEqual(estado["estado"], "GREEN_TURNING")
        follow.atualizar_estado_verde(estado, acao(), destino(), "FOLLOW_DESTINO", True, follow.GREEN_TURN_MIN_SEC + 0.01, memoria())
        self.assertEqual(estado["estado"], "GREEN_REACQUIRE")

        estado.update({"estado": "GREEN_TURNING", "inicio_estado": 0.0})
        follow.atualizar_estado_verde(estado, acao(), destino(False), "FOLLOW_DESTINO", True, follow.GREEN_TURN_MAX_SEC + 0.01, memoria())
        self.assertEqual(estado["estado"], "GREEN_REACQUIRE")

    def test_reacquire_exige_frames_e_reseta_suavizacao(self):
        estado = follow.criar_estado_verde(0.0)
        estado.update({"estado": "GREEN_REACQUIRE", "inicio_estado": 0.0, "acao_confirmada": "PREPARAR_ESQUERDA"})
        dados = memoria()
        for indice in range(follow.GREEN_REACQUIRE_CONFIRM_FRAMES):
            follow.atualizar_estado_verde(estado, acao(), destino(), "FOLLOW_DESTINO", True, indice, dados)
        self.assertEqual(estado["estado"], "GREEN_COOLDOWN")
        self.assertEqual(dados["correcao_anterior"], 0)
        self.assertEqual(dados["vel_anterior"], (0, 0))

    def test_cooldown_exige_tempo_e_verde_limpo(self):
        estado = follow.criar_estado_verde(0.0)
        estado.update({"estado": "GREEN_COOLDOWN", "inicio_cooldown": 0.0})
        dados = memoria()
        follow.atualizar_estado_verde(estado, acao(), destino(), "FOLLOW_DESTINO", True, follow.GREEN_FOLLOW_COOLDOWN_SEC + 1, dados)
        self.assertEqual(estado["estado"], "GREEN_COOLDOWN")
        for indice in range(follow.GREEN_FOLLOW_CLEAR_FRAMES_TO_RELEASE):
            follow.atualizar_estado_verde(estado, acao("SEGUIR_RETO", "NENHUM"), destino(), "FOLLOW_DESTINO", True, follow.GREEN_FOLLOW_COOLDOWN_SEC + 2 + indice, dados)
        self.assertEqual(estado["estado"], "GREEN_IDLE")


if __name__ == "__main__":
    unittest.main()
