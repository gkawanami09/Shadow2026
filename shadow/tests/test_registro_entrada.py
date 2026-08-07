"""Gravador de diagnóstico da aproximação da faixa prata.

Ele existe para parar de ajustar limiar no escuro: grava o que o robô viu de
verdade, num formato que o replay lê direto. Os testes abaixo cobrem o que
torna seguro deixá-lo ligado num teste de pista — teto, intervalo e, acima de
tudo, jamais derrubar o laço de visão.
"""

from pathlib import Path
import sys
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.registro_entrada import RegistradorEntrada  # noqa: E402


class Escritor:
    """Dublê de ``cv2.imwrite``; nunca toca no disco."""

    def __init__(self, falha=False, excecao=None):
        self.caminhos = []
        self.falha = falha
        self.excecao = excecao

    def __call__(self, caminho, quadro):
        if self.excecao is not None:
            raise self.excecao
        self.caminhos.append(caminho)
        return not self.falha


class Relogio:
    def __init__(self):
        self.agora = 0.0

    def __call__(self):
        return self.agora


class RegistradorTests(unittest.TestCase):
    def _registrador(self, max_quadros=10, intervalo=0.1, escritor=None):
        self.escritor = Escritor() if escritor is None else escritor
        self.relogio = Relogio()
        return RegistradorEntrada(
            pasta="/tmp/nao-existe", max_quadros=max_quadros,
            intervalo_min_s=intervalo, escritor=self.escritor,
            relogio=self.relogio)

    def test_grava_quadro_promissor(self):
        reg = self._registrador()
        caminho = reg.registrar(
            "quadro", "fina", votos=1, promissor=True, agora=0.0)
        self.assertIsNotNone(caminho)
        self.assertIn("fina", caminho)
        self.assertIn("v1", caminho)

    def test_deteccao_boa_tambem_e_gravada(self):
        """O positivo é o contraexemplo: prova que funciona naquela luz."""
        reg = self._registrador()
        caminho = reg.registrar(
            "quadro", "", votos=2, detectou=True, promissor=False, agora=0.0)
        self.assertIsNotNone(caminho)
        self.assertIn("ACHOU", caminho)

    def test_piso_vazio_nao_e_gravado(self):
        """Sem isso o cartão enche de piso e o sinal se perde no ruído."""
        reg = self._registrador()
        self.assertIsNone(
            reg.registrar("quadro", "sem_linha_cheia", agora=0.0))
        self.assertEqual(reg.gravados, 0)

    def test_intervalo_minimo_e_respeitado(self):
        reg = self._registrador(intervalo=0.1)
        self.assertIsNotNone(
            reg.registrar("q", "fina", promissor=True, agora=1.00))
        self.assertIsNone(
            reg.registrar("q", "fina", promissor=True, agora=1.05))
        self.assertIsNotNone(
            reg.registrar("q", "fina", promissor=True, agora=1.10))
        self.assertEqual(reg.gravados, 2)

    def test_teto_de_quadros_para_a_gravacao(self):
        reg = self._registrador(max_quadros=3, intervalo=0.0)
        for i in range(10):
            reg.registrar("q", "fina", promissor=True, agora=float(i))
        self.assertEqual(reg.gravados, 3)

    def test_nomes_sao_ordenados_e_unicos(self):
        reg = self._registrador(intervalo=0.0)
        for i in range(3):
            reg.registrar("q", "fina", promissor=True, agora=float(i))
        self.assertEqual(len(set(self.escritor.caminhos)), 3)
        self.assertEqual(self.escritor.caminhos,
                         sorted(self.escritor.caminhos))

    def test_falha_de_escrita_desliga_sem_derrubar_a_visao(self):
        """Cartão cheio não pode virar exceção no meio da prova."""
        reg = self._registrador(escritor=Escritor(falha=True))
        self.assertIsNone(reg.registrar("q", "fina", promissor=True,
                                        agora=0.0))
        self.assertTrue(reg.desligado)
        # E não tenta de novo a cada quadro.
        self.assertFalse(reg.deve_gravar(True, False, agora=99.0))

    def test_excecao_do_disco_e_engolida(self):
        reg = self._registrador(
            escritor=Escritor(excecao=OSError("disco cheio")))
        self.assertIsNone(
            reg.registrar("q", "fina", promissor=True, agora=0.0))
        self.assertTrue(reg.desligado)
        self.assertIn("disco cheio", reg.ultimo_erro)

    def test_quadro_ausente_nao_quebra(self):
        reg = self._registrador()
        self.assertIsNone(reg.registrar(None, "fina", promissor=True))


class GatilhoDoDetectorTests(unittest.TestCase):
    """`last_promising` é o que decide o que vale a pena guardar."""

    def test_detector_marca_quadro_com_prata_plausivel(self):
        from tests import cenas_sinteticas as cs
        from visao.faixa_entrada import EntrySilverDetector

        detector = EntrySilverDetector()
        detector.detect(cs.faixa_prata(), line_ahead=False, timestamp=1.0)
        self.assertTrue(detector.last_promising)

    def test_detector_nao_marca_piso_vazio(self):
        from tests import cenas_sinteticas as cs
        from visao.faixa_entrada import EntrySilverDetector

        detector = EntrySilverDetector()
        detector.detect(cs.piso_branco(), line_ahead=False, timestamp=1.0)
        self.assertFalse(detector.last_promising)


if __name__ == "__main__":
    unittest.main()
