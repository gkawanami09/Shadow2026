import pickle
import sys
from pathlib import Path
import unittest


SHADOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW_ROOT))

from visao.cache_numba import aquecer_com_cache_recuperavel


class _CacheFalso:
    def __init__(self):
        self.limpezas = 0

    def flush(self):
        self.limpezas += 1


class _FuncaoFalsa:
    def __init__(self, erro=None):
        self._cache = _CacheFalso()
        self.erro = erro
        self.chamadas = 0

    def __call__(self, valor):
        self.chamadas += 1
        if self.chamadas == 1 and self.erro is not None:
            raise self.erro
        return valor * 2


class CacheNumbaTests(unittest.TestCase):
    def test_refaz_cache_numba_corrompido_e_tenta_novamente(self):
        for erro in (EOFError(), pickle.UnpicklingError()):
            with self.subTest(tipo=type(erro).__name__):
                funcao = _FuncaoFalsa(erro)

                self.assertEqual(
                    aquecer_com_cache_recuperavel(funcao, 21), 42)
                self.assertEqual(funcao.chamadas, 2)
                self.assertEqual(funcao._cache.limpezas, 1)

    def test_nao_limpa_cache_valido(self):
        funcao = _FuncaoFalsa()

        self.assertEqual(aquecer_com_cache_recuperavel(funcao, 21), 42)
        self.assertEqual(funcao.chamadas, 1)
        self.assertEqual(funcao._cache.limpezas, 0)

    def test_nao_esconde_erro_da_funcao(self):
        funcao = _FuncaoFalsa(RuntimeError("erro real"))

        with self.assertRaisesRegex(RuntimeError, "erro real"):
            aquecer_com_cache_recuperavel(funcao, 21)

        self.assertEqual(funcao._cache.limpezas, 0)


if __name__ == "__main__":
    unittest.main()
