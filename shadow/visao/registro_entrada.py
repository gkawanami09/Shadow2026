"""Grava no cartão os quadros da aproximação da faixa prata.

Existe por um motivo específico: os limiares da entrada foram todos ajustados
contra DUAS capturas reais da fita, ambas coladas na câmera e sob a mesma luz.
Tudo além disso é cena sintética. Quando o robô falha na arena, ninguém sabe o
que ele viu — e ajustar limiar no escuro tem tanta chance de estragar o que
funciona quanto de consertar o que falha.

Este módulo fecha esse buraco. Ele guarda o quadro CRU (sem nenhuma anotação)
com o motivo da rejeição no nome do arquivo, e a pasta resultante é lida
direto por ``tools/replay_visao.py --perfil entrada``. Ou seja: o que o robô
viu na pista vira, sem conversão nenhuma, um caso de teste reproduzível na
bancada.

Três cuidados que o tornam seguro deixar ligado numa prova:

* **só grava quadro interessante** — a máscara precisa ter prata plausível, o
  que na prática é a aproximação da entrada e nada mais;
* **tem teto** de quadros e intervalo mínimo, então não enche o cartão nem
  rouba tempo do laço de visão;
* **nunca levanta exceção** para o pipeline. Um cartão cheio ou uma pasta sem
  permissão desliga o gravador e o robô segue correndo a prova.
"""

import os
import time


class RegistradorEntrada:
    """Salva quadros da aproximação, com o motivo no nome do arquivo.

    ``escritor`` e ``relogio`` são injetáveis para o teste não tocar em disco.
    """

    def __init__(self, pasta, max_quadros, intervalo_min_s,
                 escritor=None, relogio=time.monotonic):
        self.pasta = str(pasta)
        self.max_quadros = int(max_quadros)
        self.intervalo_min_s = float(intervalo_min_s)
        self.relogio = relogio
        self._escritor = escritor
        self.gravados = 0
        self.desligado = False
        self.ultimo_erro = ""
        self._ultimo_em = None
        self._pasta_pronta = False

    # -- infraestrutura ---------------------------------------------------
    def _escrever(self, caminho, quadro):
        if self._escritor is not None:
            return self._escritor(caminho, quadro)
        import cv2

        return cv2.imwrite(caminho, quadro)

    def _preparar_pasta(self):
        if self._pasta_pronta or self._escritor is not None:
            self._pasta_pronta = True
            return True
        os.makedirs(self.pasta, exist_ok=True)
        self._pasta_pronta = True
        return True

    def _desligar(self, motivo):
        """Um gravador de diagnóstico jamais derruba a prova."""
        self.desligado = True
        self.ultimo_erro = str(motivo)
        print(f"[visão] gravador da entrada desligado: {motivo}")

    # -- uso --------------------------------------------------------------
    def deve_gravar(self, promissor, detectou, agora=None):
        """Vale guardar este quadro?

        Uma detecção boa sempre vale — é o contraexemplo que prova que o
        detector funciona naquela luz. Fora isso, só quadro com prata
        plausível na máscara; o resto é piso vazio e não ensina nada.
        """
        if self.desligado or self.gravados >= self.max_quadros:
            return False
        if not (promissor or detectou):
            return False
        agora = self.relogio() if agora is None else float(agora)
        if (
            self._ultimo_em is not None
            and agora - self._ultimo_em < self.intervalo_min_s
        ):
            return False
        return True

    def registrar(self, quadro, motivo, votos=0, detectou=False,
                  promissor=False, agora=None):
        """Grava o quadro cru se ele valer a pena. Devolve o caminho ou ``None``."""
        if quadro is None:
            return None
        if not self.deve_gravar(promissor, detectou, agora=agora):
            return None

        agora = self.relogio() if agora is None else float(agora)
        rotulo = "ACHOU" if detectou else (str(motivo) or "sem_motivo")
        # O nome carrega o diagnóstico: um `ls` da pasta já conta a história
        # da aproximação inteira, sem abrir uma imagem sequer.
        nome = f"{self.gravados:04d}_{rotulo}_v{int(votos)}.png"
        caminho = os.path.join(self.pasta, nome)
        try:
            self._preparar_pasta()
            if not self._escrever(caminho, quadro):
                self._desligar(f"não consegui escrever em {caminho}")
                return None
        except Exception as erro:               # noqa: BLE001
            self._desligar(erro)
            return None

        self.gravados += 1
        self._ultimo_em = agora
        if self.gravados == self.max_quadros:
            print(
                f"[visão] gravador da entrada atingiu o teto de "
                f"{self.max_quadros} quadros em {self.pasta}")
        return caminho
