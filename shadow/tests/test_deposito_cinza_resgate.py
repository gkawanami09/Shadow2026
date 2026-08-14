"""Testes do deposito final das vitimas prata."""

import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config_resgate as cfg  # noqa: E402
from controle.deposito_cinza_resgate import (  # noqa: E402
    PassoDepositoCinza,
    SequenciadorDepositoCinza,
)
from resgate import (  # noqa: E402
    _aplicar_acoes_deposito_cinza,
    _preparar_deposito_cinza,
    _preparar_deposito_final,
    _deposito_vermelho_libera_saida,
    _proximo_marcador_deposito,
)


def executar_sequencia():
    sequenciador = SequenciadorDepositoCinza()
    agora = 10.0
    pendentes = []

    while True:
        passo = sequenciador.update(now=agora)
        if passo.terminal:
            return sequenciador, pendentes, passo
        if not passo.state.endswith("_PENDING"):
            raise AssertionError(
                f"teste esperava etapa pendente, recebeu {passo.state}")
        pendentes.append(passo)
        indice = sequenciador._indice
        duracao = sequenciador._etapas[indice].duracao
        if not sequenciador.notify_command_written(
            passo.state, now=agora
        ):
            raise AssertionError(f"ack recusado para {passo.state}")
        agora += duracao


class SequenciadorDepositoCinzaTests(unittest.TestCase):
    def test_chegada_verde_inicia_deposito_mesmo_com_contador_zero(self):
        sequenciador, comando = _preparar_deposito_cinza(0)

        self.assertIsInstance(sequenciador, SequenciadorDepositoCinza)
        self.assertEqual(comando.state, SequenciadorDepositoCinza.INICIO)
        self.assertFalse(comando.terminal)
        self.assertIn("iniciando giro", comando.detail)

    def test_giro_de_180_usa_metade_do_360_mais_quinhentos_ms(self):
        self.assertAlmostEqual(
            cfg.SILVER_DEPOSIT_TURN_S,
            (
                cfg.BALL_SEARCH_FULL_TURN_S / 2.0
                + cfg.SILVER_DEPOSIT_TURN_EXTRA_S
            ),
            places=6,
        )
        self.assertAlmostEqual(cfg.SILVER_DEPOSIT_TURN_S, 2.27, places=2)

        sequenciador = SequenciadorDepositoCinza()
        giro = next(
            etapa
            for etapa in sequenciador._etapas
            if etapa.nome == "TURN_180"
        )
        self.assertEqual(giro.angulo, 180)
        self.assertEqual(giro.velocidade, cfg.SILVER_DEPOSIT_TURN_SPEED)

    def test_temporizador_so_comeca_depois_da_escrita_serial(self):
        sequenciador = SequenciadorDepositoCinza()
        primeiro = sequenciador.update(now=0.0)
        ainda_pendente = sequenciador.update(now=50.0)

        self.assertEqual(primeiro.state, ainda_pendente.state)
        self.assertTrue(primeiro.state.endswith("_PENDING"))

        duracao_primeira = sequenciador._etapas[0].duracao
        sequenciador.notify_command_written(primeiro.state, now=50.0)
        antes = sequenciador.update(
            now=50.0 + duracao_primeira - 0.001)
        seguinte = sequenciador.update(
            now=50.0 + duracao_primeira)

        self.assertFalse(antes.state.endswith("_PENDING"))
        self.assertIn("PRE_TURN_FORWARD_STOP", seguinte.state)

    def test_sequencia_completa_respeita_ordem_e_restaura_cacamba(self):
        sequenciador, passos, final = executar_sequencia()
        nomes = [passo.state for passo in passos]

        indice_giro = next(
            i for i, nome in enumerate(nomes) if "TURN_180" in nome)
        indice_pre_frente = next(
            i for i, nome in enumerate(nomes)
            if "PRE_TURN_FORWARD_PENDING" in nome)
        indice_pre_re = next(
            i for i, nome in enumerate(nomes)
            if "PRE_TURN_REVERSE_PENDING" in nome)
        indice_re = next(
            i for i, nome in enumerate(nomes) if "REVERSE_ALIGN" in nome)
        indice_abertura = next(
            i for i, nome in enumerate(nomes) if "BUCKET_OPEN_GREEN" in nome)
        indice_restauracao = next(
            i for i, nome in enumerate(nomes) if "BUCKET_RESTORE" in nome)
        indice_saida = next(
            i for i, nome in enumerate(nomes) if "EXIT_FORWARD" in nome)
        self.assertLess(indice_pre_frente, indice_pre_re)
        self.assertLess(indice_pre_re, indice_giro)
        self.assertLess(indice_giro, indice_re)
        self.assertLess(indice_re, indice_abertura)
        self.assertLess(indice_abertura, indice_restauracao)
        self.assertLess(indice_restauracao, indice_saida)

        deltas = [
            passo.bucket_delta
            for passo in passos
            if passo.bucket_delta is not None
        ]
        self.assertEqual(deltas, [-90, 90])
        self.assertEqual(final.state, SequenciadorDepositoCinza.CONCLUIDO)
        self.assertTrue(final.terminal)
        self.assertEqual(final.angle, 190)
        self.assertTrue(sequenciador.deposito_fisico_comandado)

    def test_vermelho_abre_e_restaura_a_cacamba_no_lado_oposto(self):
        sequenciador = SequenciadorDepositoCinza("red")
        agora = 10.0
        deltas = []

        while True:
            passo = sequenciador.update(now=agora)
            if passo.terminal:
                final = passo
                break
            if passo.bucket_delta is not None:
                deltas.append(passo.bucket_delta)
            duracao = sequenciador._etapas[sequenciador._indice].duracao
            self.assertTrue(sequenciador.notify_command_written(
                passo.state, now=agora))
            agora += duracao

        self.assertEqual(deltas, [90, -90])
        self.assertEqual(final.state, "BLACK_DEPOSIT_COMPLETE")
        self.assertTrue(final.terminal)
        self.assertTrue(sequenciador.deposito_fisico_comandado)

    def test_saida_exige_deposito_vermelho_com_cacamba_comandada(self):
        sequenciador = SequenciadorDepositoCinza("red")
        final_antes_da_serial = sequenciador.update(now=0.0)

        self.assertFalse(_deposito_vermelho_libera_saida(
            sequenciador, final_antes_da_serial))

        agora = 0.0
        while True:
            passo = sequenciador.update(now=agora)
            if passo.terminal:
                final = passo
                break
            duracao = sequenciador._etapas[sequenciador._indice].duracao
            self.assertTrue(sequenciador.notify_command_written(
                passo.state, now=agora))
            agora += duracao

        self.assertTrue(_deposito_vermelho_libera_saida(sequenciador, final))

    def test_deposito_verde_nunca_libera_a_saida(self):
        sequenciador, _passos, final = executar_sequencia()

        self.assertTrue(sequenciador.deposito_fisico_comandado)
        self.assertFalse(_deposito_vermelho_libera_saida(
            sequenciador, final))

    def test_depois_do_verde_procura_vermelho_e_so_entao_encerra(self):
        self.assertEqual(_proximo_marcador_deposito("green"), "red")
        self.assertIsNone(_proximo_marcador_deposito("red"))

        sequenciador, comando = _preparar_deposito_final("red", 1)
        self.assertEqual(sequenciador.marcador_destino, "red")
        self.assertEqual(comando.state, "BLACK_DEPOSIT_START")
        self.assertIn("preta", comando.detail)
        self.assertIn("vermelho", comando.detail)

    def test_avanco_e_re_antes_do_giro_usam_os_tempos_calibrados(self):
        sequenciador = SequenciadorDepositoCinza()
        etapas = {etapa.nome: etapa for etapa in sequenciador._etapas}

        frente = etapas["PRE_TURN_FORWARD"]
        re = etapas["PRE_TURN_REVERSE"]
        self.assertEqual(frente.angulo, 0)
        self.assertEqual(re.angulo, 200)
        self.assertEqual(frente.duracao, 1.0)
        self.assertEqual(re.duracao, 0.5)
        self.assertEqual(
            round(frente.velocidade * 120),
            cfg.SILVER_DEPOSIT_PRE_TURN_PWM,
        )
        self.assertEqual(frente.velocidade, re.velocidade)

    def test_depois_de_fechar_avanca_reto_por_um_segundo_e_meio(self):
        sequenciador = SequenciadorDepositoCinza()
        etapas = {etapa.nome: etapa for etapa in sequenciador._etapas}

        saida = etapas["EXIT_FORWARD"]
        self.assertEqual(saida.angulo, 0)
        self.assertEqual(saida.duracao, 1.5)
        self.assertEqual(
            round(saida.velocidade * 120),
            cfg.SILVER_DEPOSIT_EXIT_FORWARD_PWM,
        )

    def test_re_de_alinhamento_dura_tres_segundos(self):
        sequenciador = SequenciadorDepositoCinza()
        etapa = next(
            etapa
            for etapa in sequenciador._etapas
            if etapa.nome == "REVERSE_ALIGN"
        )

        self.assertEqual(etapa.angulo, 200)
        self.assertEqual(etapa.duracao, 3.0)
        self.assertEqual(
            etapa.velocidade, cfg.SILVER_DEPOSIT_REVERSE_SPEED)
        self.assertEqual(
            round(etapa.velocidade * 120),
            cfg.SILVER_DEPOSIT_REVERSE_PWM,
        )

    def test_sacudida_e_rapida_e_tem_parada_antes_de_inverter(self):
        sequenciador = SequenciadorDepositoCinza()
        nomes = [etapa.nome for etapa in sequenciador._etapas]

        self.assertEqual(
            round(cfg.SILVER_DEPOSIT_SHAKE_SPEED * 120),
            cfg.SILVER_DEPOSIT_SHAKE_PWM,
        )

        for numero in range(1, cfg.SILVER_DEPOSIT_SHAKE_REPETITIONS + 1):
            frente = nomes.index(f"SHAKE_FORWARD_{numero}")
            parada = nomes.index(f"SHAKE_FRONT_STOP_{numero}")
            re = nomes.index(f"SHAKE_REVERSE_{numero}")
            self.assertEqual(parada, frente + 1)
            self.assertEqual(re, parada + 1)
            self.assertEqual(
                sequenciador._etapas[frente].velocidade,
                cfg.SILVER_DEPOSIT_SHAKE_SPEED,
            )
            self.assertEqual(
                sequenciador._etapas[frente].duracao,
                cfg.SILVER_DEPOSIT_SHAKE_MOVE_S,
            )

    def test_falha_sempre_termina_com_parar(self):
        sequenciador = SequenciadorDepositoCinza()

        falha = sequenciador.fail("serial caiu")

        self.assertEqual(falha.state, sequenciador.FALHA)
        self.assertEqual(falha.angle, 190)
        self.assertEqual(falha.speed, 0.0)
        self.assertTrue(falha.terminal)


class AplicacaoDepositoCinzaTests(unittest.TestCase):
    class ArduinoFalso:
        def __init__(self):
            self.connected = True
            self.connection_epoch = 4
            self.chamadas = []

        def servo(self, nome, delta):
            self.chamadas.append(("servo", nome, delta))
            return True

    @staticmethod
    def direcao(chamadas, resultado=True):
        def aplicar(angulo=190, velocidade=0.8):
            chamadas.append(("direcao", angulo, velocidade))
            return resultado
        return aplicar

    def test_parar_e_enviado_antes_de_abrir_a_cacamba(self):
        arduino = self.ArduinoFalso()
        passo = PassoDepositoCinza(
            "ABRIR",
            "abrindo",
            bucket_delta=-90,
        )

        erro = _aplicar_acoes_deposito_cinza(
            passo,
            arduino,
            self.direcao(arduino.chamadas),
            epoca_serial_esperada=4,
        )

        self.assertIsNone(erro)
        self.assertEqual(
            arduino.chamadas,
            [
                ("direcao", 190, 0.0),
                ("servo", "CACAMBA", -90),
            ],
        )

    def test_falha_do_motor_impede_comando_da_cacamba(self):
        arduino = self.ArduinoFalso()
        passo = PassoDepositoCinza(
            "ABRIR",
            "abrindo",
            bucket_delta=-90,
        )

        erro = _aplicar_acoes_deposito_cinza(
            passo,
            arduino,
            self.direcao(arduino.chamadas, resultado=False),
            epoca_serial_esperada=4,
        )

        self.assertIn("motores", erro)
        self.assertFalse(any(
            chamada[0] == "servo" for chamada in arduino.chamadas))

    def test_reconexao_antes_da_etapa_bloqueia_tudo(self):
        arduino = self.ArduinoFalso()
        arduino.connection_epoch = 5
        passo = PassoDepositoCinza(
            "ABRIR",
            "abrindo",
            bucket_delta=-90,
        )

        erro = _aplicar_acoes_deposito_cinza(
            passo,
            arduino,
            self.direcao(arduino.chamadas),
            epoca_serial_esperada=4,
        )

        self.assertIn("serial mudou", erro)
        self.assertEqual(arduino.chamadas, [])


if __name__ == "__main__":
    unittest.main()
