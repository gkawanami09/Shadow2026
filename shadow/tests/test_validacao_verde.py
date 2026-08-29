"""Regressoes da geometria obrigatoria dos marcadores verdes."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from shared.dados_compartilhados import (empty_time_arr, timer, turn_dir)  # noqa: E402
from visao.verde import (ConfirmadorVerde, check_green,
                         has_plausible_green, latch_turn_direction)  # noqa: E402


def _quadrado(x=160, y=130, lado=60):
    return np.array(
        [[[x, y]], [[x + lado, y]], [[x + lado, y + lado]],
         [[x, y + lado]]],
        dtype=np.int32,
    )


def _mascara_ao_redor(*, topo=True, esquerda=False, direita=False,
                       baixo=False, x=160, y=130, lado=60):
    mascara = np.zeros(
        (config.camera_y, config.camera_x), dtype=np.uint8)
    if topo:
        cv2.rectangle(mascara, (x + 3, y - 14),
                      (x + lado - 3, y - 1), 255, -1)
    if baixo:
        cv2.rectangle(mascara, (x + 3, y + lado + 1),
                      (x + lado - 3, y + lado + 14), 255, -1)
    if esquerda:
        cv2.rectangle(mascara, (x - 14, y + 3),
                      (x - 1, y + lado - 3), 255, -1)
    if direita:
        cv2.rectangle(mascara, (x + lado + 1, y + 3),
                      (x + lado + 14, y + lado - 3), 255, -1)
    return mascara


def _cena_intersecao_direita(angulo=0):
    """Reproduz a cruz da imagem: reto + ramo direito + verde no quadrante."""
    preto = np.zeros((config.camera_y, config.camera_x), dtype=np.uint8)
    verde = np.zeros_like(preto)
    cv2.rectangle(preto, (100, 0), (154, config.camera_y - 1), 255, -1)
    cv2.rectangle(preto, (100, 94), (300, 140), 255, -1)
    cv2.rectangle(verde, (154, 141), (209, 192), 255, -1)
    if angulo:
        matriz = cv2.getRotationMatrix2D((154, 141), angulo, 1.)
        tamanho = (config.camera_x, config.camera_y)
        preto = cv2.warpAffine(
            preto, matriz, tamanho, flags=cv2.INTER_NEAREST)
        verde = cv2.warpAffine(
            verde, matriz, tamanho, flags=cv2.INTER_NEAREST)
    preto[verde > 0] = 0
    contornos, _ = cv2.findContours(
        verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos, preto


def _dois_verdes_validos():
    esquerdo = _quadrado(x=70, y=130, lado=52)
    direito = _quadrado(x=300, y=130, lado=52)
    preto = np.zeros((config.camera_y, config.camera_x), dtype=np.uint8)
    preto |= _mascara_ao_redor(
        topo=True, direita=True, x=70, y=130, lado=52)
    preto |= _mascara_ao_redor(
        topo=True, esquerda=True, x=300, y=130, lado=52)
    return [esquerdo, direito], preto


class ValidacaoVerdeTests(unittest.TestCase):
    def setUp(self):
        timer.remove_timer("left_marker")
        timer.remove_timer("right_marker")
        turn_dir.value = "straight"

    def test_sem_marcador_nao_inventa_direcao_de_curva(self):
        latch_turn_direction("straight", empty_time_arr())

        self.assertEqual(turn_dir.value, "straight")

    def test_memoria_de_marcador_ainda_conserva_direcao(self):
        timer.set_timer("right_marker", 1.0)
        latch_turn_direction("straight", empty_time_arr())

        self.assertEqual(turn_dir.value, "right")

    def test_circulo_com_preto_acima_e_ao_lado_autoriza_curva(self):
        preto = np.zeros((config.camera_y, config.camera_x), dtype=np.uint8)
        verde = np.zeros_like(preto)
        cv2.circle(preto, (175, 88), 68, 255, 22)
        cv2.rectangle(preto, (164, 145), (186, 251), 255, -1)
        cv2.rectangle(preto, (230, 77), (340, 99), 255, -1)
        cv2.rectangle(verde, (190, 164), (245, 219), 255, -1)
        cv2.rectangle(verde, (286, 101), (341, 156), 255, -1)
        preto[verde > 0] = 0
        contornos, _ = cv2.findContours(
            verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.assertEqual(check_green(contornos, preto), "right")

    def test_verde_plausivel_e_guardado_mesmo_antes_de_validar(self):
        confirmador = ConfirmadorVerde(frames=3, window=5)
        self.assertTrue(has_plausible_green([_quadrado()]))
        confirmador.atualizar("straight", candidato=True)
        self.assertTrue(confirmador.candidato_ativo)
        for _ in range(5):
            confirmador.atualizar("straight", candidato=False)
        self.assertFalse(confirmador.candidato_ativo)

    def test_dois_verdes_validos_tem_prioridade_de_180(self):
        contornos, preto = _dois_verdes_validos()
        self.assertEqual(check_green(contornos, preto), "turn_around")

    def test_180_cancela_confirmacao_parcial_de_90(self):
        confirmador = ConfirmadorVerde(frames=3, frames_180=1)
        self.assertEqual(confirmador.atualizar("left"), "straight")
        self.assertEqual(confirmador.atualizar("left"), "straight")
        self.assertEqual(
            confirmador.atualizar("turn_around"), "turn_around")
        self.assertEqual(confirmador.atualizar("left"), "straight")

    def test_intersecao_que_continua_reto_obedece_ao_verde(self):
        contornos, preto = _cena_intersecao_direita()
        self.assertEqual(check_green(contornos, preto), "right")

    def test_verde_torto_positivo_continua_valido(self):
        contornos, preto = _cena_intersecao_direita(25)
        self.assertEqual(check_green(contornos, preto), "right")

    def test_verde_torto_negativo_continua_valido(self):
        contornos, preto = _cena_intersecao_direita(-25)
        self.assertEqual(check_green(contornos, preto), "right")

    def test_verde_para_esquerda_exige_preto_acima_e_a_direita(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(topo=True, direita=True))
        self.assertEqual(direcao, "left")

    def test_verde_para_direita_exige_preto_acima_e_a_esquerda(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(topo=True, esquerda=True))
        self.assertEqual(direcao, "right")

    def test_sem_preto_acima_nao_autoriza_curva(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=False, direita=True))
        self.assertEqual(direcao, "straight")

    def test_preto_dos_dois_lados_e_ambiguo(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=True, esquerda=True, direita=True))
        self.assertEqual(direcao, "straight")

    def test_preto_em_baixo_invalida_verde_como_na_logica_antiga(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=True, direita=True, baixo=True))
        self.assertEqual(direcao, "straight")

    def test_verde_cortado_na_borda_direita_continua_valido(self):
        verde = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8)
        x, y = config.camera_x - 58, 130
        cv2.rectangle(
            verde, (x, y), (config.camera_x - 1, y + 68), 255, -1)
        contornos, _ = cv2.findContours(
            verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preto = _mascara_ao_redor(
            topo=True,
            esquerda=True,
            x=x,
            y=y,
            lado=58,
        )

        self.assertEqual(check_green(contornos, preto), "right")

    def test_ruido_preto_descontinuo_nao_passa(self):
        mascara = np.zeros(
            (config.camera_y, config.camera_x), dtype=np.uint8)
        for x in range(163, 218, 8):
            cv2.rectangle(mascara, (x, 118), (x + 2, 128), 255, -1)
        for y in range(133, 188, 8):
            cv2.rectangle(mascara, (222, y), (232, y + 2), 255, -1)
        self.assertEqual(check_green([_quadrado()], mascara), "straight")

    def test_mancha_verde_alongada_nao_e_quadrado(self):
        mancha = np.array(
            [[[120, 130]], [[240, 130]], [[240, 160]], [[120, 160]]],
            dtype=np.int32,
        )
        self.assertEqual(
            check_green(
                [mancha], _mascara_ao_redor(topo=True, direita=True)),
            "straight",
        )

    def test_direcao_precisa_de_tres_quadros_consecutivos(self):
        confirmador = ConfirmadorVerde(frames=3)
        self.assertEqual(confirmador.atualizar("left"), "straight")
        self.assertEqual(confirmador.atualizar("left"), "straight")
        self.assertEqual(confirmador.atualizar("left"), "left")

    def test_um_quadro_invalido_nao_apaga_verde_coerente(self):
        confirmador = ConfirmadorVerde(frames=3)
        confirmador.atualizar("right")
        confirmador.atualizar("right")
        self.assertEqual(confirmador.atualizar("straight"), "straight")
        self.assertEqual(confirmador.atualizar("right"), "right")

    def test_votos_de_lados_opostos_nao_autorizam_curva(self):
        confirmador = ConfirmadorVerde(frames=3)
        confirmador.atualizar("right")
        confirmador.atualizar("left")
        confirmador.atualizar("right")
        self.assertEqual(confirmador.atualizar("left"), "straight")


if __name__ == "__main__":
    unittest.main()
