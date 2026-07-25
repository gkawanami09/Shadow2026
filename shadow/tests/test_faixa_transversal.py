"""Testes da geometria de faixa transversal e da votação temporal."""

import sys
from pathlib import Path
import unittest

import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.faixa_transversal import (  # noqa: E402
    BandGeometry,
    StripeConfirmer,
    find_transversal_band,
)


def _mask(shape, rows=None, cols=None):
    mask = np.zeros(shape, dtype=np.uint8)
    if rows is not None:
        row_slice = slice(*rows)
        col_slice = slice(*cols) if cols is not None else slice(None)
        mask[row_slice, col_slice] = 255
    return mask


class TransversalBandTests(unittest.TestCase):
    shape = (252, 448)

    def test_faixa_larga_e_fina_e_aceita(self):
        mask = _mask(self.shape, rows=(200, 225))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertEqual(reason, "")
        self.assertIsNotNone(band)
        self.assertEqual(band.top_y, 200)
        self.assertEqual(band.bottom_y, 224)
        self.assertAlmostEqual(band.span_ratio, 1.0)
        self.assertGreater(band.aspect, 15.0)

    def test_mancha_estreita_nao_forma_faixa(self):
        mask = _mask(self.shape, rows=(200, 225), cols=(0, 120))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNone(band)
        self.assertEqual(reason, "sem_linha_cheia")

    def test_regiao_que_preenche_a_roi_e_rejeitada(self):
        """Um piso inteiro claro/escuro encosta no topo da ROI e cai lá."""
        mask = _mask(self.shape, rows=(139, 252))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNone(band)
        self.assertEqual(reason, "cortada_no_topo")

    def test_banda_grossa_dentro_da_roi_e_rejeitada_como_espessa(self):
        """Sem tocar o topo da ROI, o veto que age é o de espessura."""
        mask = _mask(self.shape, rows=(160, 252))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNone(band)
        self.assertEqual(reason, "espessa")

    def test_faixa_cortada_pelo_topo_da_roi_e_rejeitada(self):
        """Regra medida nas capturas reais da arena.

        Uma região que encosta no topo da ROI continua acima do corte: sua
        extensão real não pode ser medida e ela quase sempre é roupa escura,
        sombra ou parede truncada — não uma fita. Nas 18 capturas reais, as
        seis falsas soleiras de saída começavam todas exatamente nesta linha.
        """
        mask = _mask(self.shape, rows=(139, 165))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNone(band)
        self.assertEqual(reason, "cortada_no_topo")

    def test_faixa_que_encosta_embaixo_continua_valida(self):
        """Encostar na base é normal: é assim que a fita fica de perto."""
        mask = _mask(self.shape, rows=(225, 252))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNotNone(band)
        self.assertEqual(reason, "")

    def test_objeto_acima_da_roi_e_ignorado(self):
        mask = _mask(self.shape, rows=(20, 60))
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNone(band)
        self.assertEqual(reason, "sem_linha_cheia")

    def test_faixa_vazada_reprova_no_preenchimento(self):
        mask = _mask(self.shape, rows=(200, 225))
        mask[200:225, ::2] = 0  # metade dos pixels apagados em xadrez
        band, reason = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNone(band)
        self.assertIn(reason, ("sem_linha_cheia", "vazada"))

    def test_maior_sequencia_contigua_vence_linhas_isoladas(self):
        mask = _mask(self.shape, rows=(200, 225))
        mask[150, :] = 255   # linha solta de brilho
        mask[170, :] = 255
        band, _ = find_transversal_band(
            mask, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
        self.assertIsNotNone(band)
        self.assertEqual(band.top_y, 200)

    def test_nenhum_disco_satisfaz_largura_e_espessura(self):
        """Propriedade estrutural: círculo algum passa nos dois limites.

        Este é o veto que impede uma vítima esférica (prata ou preta) de ser
        lida como faixa, independentemente de quão perto ela esteja.
        """
        height, width = self.shape
        geometry = BandGeometry(
            min_row_fill=0.45,
            min_span_ratio=0.60,
            min_thickness_ratio=0.03,
            max_thickness_ratio=0.30,
            min_fill_ratio=0.55,
            min_aspect=3.5,
        )
        ys, xs = np.ogrid[:height, :width]
        for raio in range(30, 420, 10):
            mask = np.zeros(self.shape, dtype=np.uint8)
            disk = (
                (xs - width // 2) ** 2 + (ys - int(height * 0.8)) ** 2
                <= raio * raio
            )
            mask[disk] = 255
            band, reason = find_transversal_band(
                mask, geometry, roi_top_ratio=0.55, roi_bottom_ratio=1.0)
            self.assertIsNone(
                band, f"disco de raio {raio} foi aceito como faixa")
            self.assertIn(
                reason,
                ("sem_linha_cheia", "estreita", "espessa", "compacta",
                 "vazada", "cortada_no_topo"))


class StripeConfirmerTests(unittest.TestCase):
    def test_um_frame_positivo_nao_confirma(self):
        confirmer = StripeConfirmer(votes_needed=3, window=5, max_age_s=1.0)
        self.assertFalse(confirmer.update(True, timestamp=1.0, now=1.0))
        self.assertEqual(confirmer.votes, 1)

    def test_tres_de_cinco_confirma(self):
        confirmer = StripeConfirmer(votes_needed=3, window=5, max_age_s=1.0)
        for index, positive in enumerate((True, False, True, False, True)):
            confirmed = confirmer.update(
                positive, timestamp=1.0 + index * 0.1,
                now=1.0 + index * 0.1)
        self.assertTrue(confirmed)

    def test_frame_repetido_nao_gera_segundo_voto(self):
        confirmer = StripeConfirmer(votes_needed=2, window=5, max_age_s=1.0)
        confirmer.update(True, timestamp=2.0, now=2.0)
        self.assertFalse(confirmer.update(True, timestamp=2.0, now=2.0))
        self.assertEqual(confirmer.votes, 1)

    def test_resultado_stale_nao_vota(self):
        confirmer = StripeConfirmer(votes_needed=1, window=3, max_age_s=0.30)
        self.assertFalse(confirmer.update(True, timestamp=1.0, now=1.9))
        self.assertEqual(confirmer.samples, 0)

    def test_histerese_mantem_confirmado_apos_frame_ruim(self):
        confirmer = StripeConfirmer(votes_needed=2, window=5, max_age_s=1.0)
        confirmer.update(True, timestamp=1.0, now=1.0)
        self.assertTrue(confirmer.update(True, timestamp=1.1, now=1.1))
        self.assertTrue(confirmer.update(False, timestamp=1.2, now=1.2))

    def test_cooldown_bloqueia_reconfirmacao(self):
        confirmer = StripeConfirmer(
            votes_needed=1, window=3, max_age_s=1.0, cooldown_s=5.0)
        confirmer.update(True, timestamp=1.0, now=1.0)
        confirmer.reset(now=2.0)
        self.assertTrue(confirmer.blocked(3.0))
        self.assertFalse(confirmer.update(True, timestamp=3.0, now=3.0))
        self.assertFalse(confirmer.blocked(7.5))
        self.assertTrue(confirmer.update(True, timestamp=7.5, now=7.5))


if __name__ == "__main__":
    unittest.main()
