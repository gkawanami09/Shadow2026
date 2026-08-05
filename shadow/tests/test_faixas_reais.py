"""Testes contra as FOTOS REAIS da arena, em captures/linha_prata e linha_preta.

Estes testes existem porque os dois bugs relatados só apareciam em imagem
real:

* a câmera 0 disparava a verificação de saída em 14 de 19 fotos, inclusive
  em fotos sem faixa preta nenhuma — ela agarrava a sombra sob a barreira
  branca e a borda do marcador;
* o gate HSV da entrada REJEITAVA a fita prata verdadeira, porque exigia
  fita clara e neutra e a fita real aparece escura e amarelada.

Se as pastas de captura não existirem (checkout limpo), os testes são
pulados em vez de falhar — eles dependem de dado que não é versionado.
"""

import sys
from pathlib import Path
import unittest

import cv2


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.confirmacao_saida_linha import (  # noqa: E402
    NAO_PRETA,
    PRETA,
    ClassificadorFaixaSaidaLinha,
)
from visao.entrada_prata import PortaoEntradaPrata  # noqa: E402
from visao.faixa_saida import (  # noqa: E402
    BlackExitDetector,
    BlackExitGate,
)


CAPTURAS = SHADOW_ROOT / "captures"
PRATA = CAPTURAS / "linha_prata"
PRETA_DIR = CAPTURAS / "linha_preta"

#: Referências da CÂMERA DE LINHA, identificadas visualmente.
FITA_PRATA = ("140307", "140318")
FAIXA_PRETA = ("140225", "140238")


def _carregar(pasta, chave):
    candidatos = sorted(pasta.glob(f"*{chave}.png"))
    if not candidatos:
        return None
    return cv2.imread(str(candidatos[0]))


def _tem_capturas():
    return PRATA.is_dir() and PRETA_DIR.is_dir()


@unittest.skipUnless(_tem_capturas(), "capturas reais não estão presentes")
class ConfirmacaoCamera1Tests(unittest.TestCase):
    """A câmera 1 é quem decide. Ela tem de acertar as quatro."""

    def test_faixa_preta_real_e_classificada_preta(self):
        for chave in FAIXA_PRETA:
            frame = _carregar(PRETA_DIR, chave)
            if frame is None:
                self.skipTest(f"{chave} ausente")
            resultado = ClassificadorFaixaSaidaLinha().classificar(
                frame, timestamp=1.0)
            self.assertEqual(
                resultado.classificacao, PRETA,
                f"{chave} é a soleira preta e precisa liberar a saída")

    def test_fita_prata_real_nao_e_classificada_preta(self):
        """O erro caro: sair da sala achando que a prata é a saída."""
        for chave in FITA_PRATA:
            frame = _carregar(PRATA, chave)
            if frame is None:
                self.skipTest(f"{chave} ausente")
            resultado = ClassificadorFaixaSaidaLinha().classificar(
                frame, timestamp=1.0)
            self.assertEqual(
                resultado.classificacao, NAO_PRETA,
                f"{chave} é a fita prata e NÃO pode liberar a saída")


@unittest.skipUnless(_tem_capturas(), "capturas reais não estão presentes")
class GatilhoCamera0Tests(unittest.TestCase):
    """A câmera 0 só chama a verificação quando vale a pena parar."""

    def test_sala_sem_soleira_nao_dispara(self):
        """Era o bug: 14 de 19 fotos disparavam, quase todas na sombra."""
        disparos = []
        for pasta in (PRETA_DIR, PRATA):
            for caminho in sorted(pasta.glob("*.png")):
                chave = caminho.stem[-6:]
                if chave in FAIXA_PRETA:
                    continue          # esta é a soleira de verdade
                frame = cv2.imread(str(caminho))
                if frame is None:
                    continue
                detector = BlackExitDetector()
                if detector.detect(frame, timestamp=1.0) is not None:
                    disparos.append(f"{pasta.name}/{chave}")
        self.assertEqual(
            disparos, [],
            f"disparou sem soleira em: {disparos}")

    def test_soleira_real_ainda_dispara(self):
        """Cortar falso positivo não pode ter cegado o detector."""
        frame = _carregar(PRETA_DIR, "140225")
        if frame is None:
            self.skipTest("140225 ausente")
        detector = BlackExitDetector()
        self.assertIsNotNone(
            detector.detect(frame, timestamp=1.0),
            "a soleira real precisa continuar disparando a verificação")

    def test_gate_confirma_soleira_e_recusa_sala(self):
        cenarios = (
            (PRETA_DIR, "140225", True),
            (PRETA_DIR, "133405", False),
            (PRATA, "133604", False),
        )
        for pasta, chave, esperado in cenarios:
            frame = _carregar(pasta, chave)
            if frame is None:
                continue
            gate = BlackExitGate()
            confirmado = False
            for indice in range(8):
                instante = 1.0 + indice * 0.1
                confirmado, _ = gate.update(
                    frame, timestamp=instante, now=instante)
            self.assertEqual(
                confirmado, esperado,
                f"{pasta.name}/{chave} deveria confirmar={esperado}")


@unittest.skipUnless(_tem_capturas(), "capturas reais não estão presentes")
class EntradaPrataTests(unittest.TestCase):
    """A entrada usa a mesma assinatura de textura da saída."""

    @staticmethod
    def _rodar(frame, quadros=6):
        gate = PortaoEntradaPrata()
        confirmado = False
        for indice in range(quadros):
            instante = 1.0 + indice * 0.1
            confirmado, _ = gate.update(
                frame, line_ahead=False,
                timestamp=instante, now=instante)
        return confirmado, gate

    def test_fita_prata_real_confirma_a_entrada(self):
        """O gate HSV anterior rejeitava estas duas."""
        for chave in FITA_PRATA:
            frame = _carregar(PRATA, chave)
            if frame is None:
                self.skipTest(f"{chave} ausente")
            confirmado, gate = self._rodar(frame)
            self.assertTrue(
                confirmado,
                f"{chave} é a fita prata e precisa confirmar a entrada")
            self.assertGreaterEqual(gate.votes, 3)

    def test_faixa_preta_nao_confirma_a_entrada(self):
        """Entrar na sala ao ver a soleira de saída seria grave."""
        for chave in FAIXA_PRETA:
            frame = _carregar(PRETA_DIR, chave)
            if frame is None:
                self.skipTest(f"{chave} ausente")
            confirmado, _ = self._rodar(frame)
            self.assertFalse(
                confirmado,
                f"{chave} é a faixa PRETA e não pode virar entrada")

    def test_linha_continuando_veta_a_entrada(self):
        """Brilho sobre a pista, com a linha seguindo, não é a soleira."""
        frame = _carregar(PRATA, "140307")
        if frame is None:
            self.skipTest("140307 ausente")
        gate = PortaoEntradaPrata()
        confirmado = False
        for indice in range(6):
            instante = 1.0 + indice * 0.1
            confirmado, _ = gate.update(
                frame, line_ahead=True,
                timestamp=instante, now=instante)
        self.assertFalse(confirmado)

    def test_frame_repetido_nao_confirma(self):
        frame = _carregar(PRATA, "140307")
        if frame is None:
            self.skipTest("140307 ausente")
        gate = PortaoEntradaPrata()
        confirmado = False
        for _ in range(8):
            confirmado, _ = gate.update(
                frame, line_ahead=False, timestamp=1.0, now=1.0)
        self.assertFalse(
            confirmado, "o mesmo frame não pode gerar vários votos")


class EvidenciaFracaTests(unittest.TestCase):
    """Regra que não depende de imagem: Hough sozinho exige mais votos."""

    def test_config_exige_mais_votos_da_evidencia_fraca(self):
        import config_resgate as cfg
        self.assertGreater(
            cfg.EXIT_BLACK_WEAK_VOTES_NEEDED,
            cfg.EXIT_BLACK_VOTES_NEEDED,
            "a linha fina vista de longe é evidência fraca e precisa "
            "persistir mais que a faixa larga da máscara")

    def test_comprimento_minimo_ficou_acima_do_clutter_medido(self):
        import config_resgate as cfg
        # Medido nas fotos reais: sombra da barreira e borda de mesa
        # produzem segmentos de até 0.34 da largura.
        self.assertGreater(cfg.EXIT_LINE_MIN_LENGTH_RATIO, 0.34)


if __name__ == "__main__":
    unittest.main()
