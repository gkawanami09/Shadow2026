"""Regressoes da geometria obrigatoria dos marcadores verdes."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

import config  # noqa: E402
from visao.verde import ConfirmadorVerde, check_green  # noqa: E402


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


class ValidacaoVerdeTests(unittest.TestCase):
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

    def test_preto_em_baixo_tambem_invalida_o_marcador(self):
        direcao = check_green(
            [_quadrado()], _mascara_ao_redor(
                topo=True, direita=True, baixo=True))
        self.assertEqual(direcao, "straight")

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

    def test_quadro_invalido_zerar_confirmacao(self):
        confirmador = ConfirmadorVerde(frames=3)
        confirmador.atualizar("right")
        confirmador.atualizar("right")
        self.assertEqual(confirmador.atualizar("straight"), "straight")
        self.assertEqual(confirmador.atualizar("right"), "straight")

    def test_votos_de_lados_opostos_nao_autorizam_curva(self):
        confirmador = ConfirmadorVerde(frames=3)
        confirmador.atualizar("right")
        confirmador.atualizar("left")
        confirmador.atualizar("right")
        self.assertEqual(confirmador.atualizar("left"), "straight")


if __name__ == "__main__":
    unittest.main()
