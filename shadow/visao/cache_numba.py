"""Aquecimento de funcoes Numba com recuperacao do cache em disco."""

import pickle


ERROS_CACHE_CORROMPIDO = (EOFError, pickle.UnpicklingError)


def aquecer_com_cache_recuperavel(funcao, *args):
    """Executa uma funcao Numba e refaz somente um cache ilegivel.

    Um corte de energia pode interromper a gravacao do indice do Numba. Nesse
    caso, a primeira tentativa falha antes mesmo da compilacao. O dispatcher
    sabe apagar seus proprios arquivos; depois disso, uma unica nova chamada
    recompila e grava um cache valido para os proximos boots.
    """
    try:
        return funcao(*args)
    except ERROS_CACHE_CORROMPIDO:
        cache = getattr(funcao, "_cache", None)
        limpar = getattr(cache, "flush", None)
        if not callable(limpar):
            raise

        nome = getattr(getattr(funcao, "py_func", None), "__name__", None)
        nome = nome or getattr(funcao, "__name__", "funcao")
        print(f"[visao] cache Numba corrompido em {nome}; reconstruindo...")
        limpar()
        return funcao(*args)
