#!/usr/bin/env python3
"""Reexecuta os detectores sobre frames ou vídeos gravados, SEM motores.

Este é o segundo degrau da ordem segura de validação, logo depois da suíte
automatizada e antes de qualquer teste com o robô ligado. Ele responde a
pergunta que os testes sintéticos não respondem: *o detector acerta nas
imagens reais da nossa arena?*

Nada aqui abre a serial, adquire a trava dos motores ou toca no Arduino. O
programa só lê imagens do disco.

Perfis disponíveis
------------------
``entrada``  faixa PRATA de entrada — imagens da CÂMERA DE LINHA
``saida``    faixa PRETA de saída  — imagens da CÂMERA DE RESGATE
``vitima``   esferas prata/preta   — imagens da CÂMERA DE RESGATE
``triangulos`` os dois triângulos juntos, com as cores do overlay

Rotulagem
---------
Com ``--esperado positivo`` ou ``--esperado negativo`` o programa compara a
decisão do detector com o rótulo e imprime acertos, falsos positivos e falsos
negativos. Um conjunto de negativos difíceis (madeira, sombra, roupas, faixa
preta, reflexos) com **zero falsos positivos** é o critério que interessa
antes de ligar os motores.

Exemplos::

    python3 shadow/tools/replay_visao.py --perfil entrada \\
        --frames shadow/captures/entrada_positivos --esperado positivo
    python3 shadow/tools/replay_visao.py --perfil saida \\
        --video shadow/captures/saida.mp4 --salvar /tmp/anotados
    python3 shadow/tools/replay_visao.py --perfil vitima \\
        --frames shadow/captures/negativos --esperado negativo
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp")


def carregar_frames(args):
    """Gera ``(nome, frame)`` a partir de um diretório ou de um vídeo."""
    if args.frames is not None:
        caminhos = sorted(
            caminho for caminho in Path(args.frames).iterdir()
            if caminho.suffix.lower() in EXTENSOES
        )
        if not caminhos:
            raise SystemExit(f"nenhuma imagem encontrada em {args.frames}")
        for caminho in caminhos:
            frame = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[replay] ignorando arquivo ilegível: {caminho.name}")
                continue
            yield caminho.name, frame
        return

    captura = cv2.VideoCapture(str(args.video))
    if not captura.isOpened():
        raise SystemExit(f"não foi possível abrir o vídeo: {args.video}")
    indice = 0
    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            yield f"frame_{indice:05d}", frame
            indice += 1
    finally:
        captura.release()


class PerfilEntrada:
    """Faixa prata de entrada, com a votação temporal real."""

    nome = "entrada"
    camera = "linha"

    def __init__(self):
        from visao.entrada_missao import EntryGate, EntryInference, EntryModel
        self.gate = EntryGate()
        self.model = EntryModel().load()

    def processar(self, frame, timestamp):
        deteccao = self.model.detect(frame)
        confirmado, deteccao = self.gate.update(EntryInference(
            timestamp, True, deteccao, 0.0))
        motivo = self.gate.last_reason or "aceita"
        return deteccao is not None, {
            "confirmado": confirmado,
            "motivo": motivo,
            "votos": self.gate.votes,
            "confianca": (
                deteccao.confidence if deteccao is not None else 0.0),
        }

    def anotar(self, frame, deteccao):
        return frame


class PerfilSaida:
    """Faixa preta de saída."""

    nome = "saida"
    camera = "resgate"

    def __init__(self):
        from visao.faixa_saida import BlackExitGate
        self.gate = BlackExitGate()

    def processar(self, frame, timestamp):
        confirmado, deteccao = self.gate.update(
            frame, timestamp=timestamp, now=timestamp)
        motivo = self.gate.detector.last_reason or "aceita"
        return deteccao is not None, {
            "confirmado": confirmado,
            "motivo": motivo,
            "votos": self.gate.votes,
            "confianca": (
                deteccao.confidence if deteccao is not None else 0.0),
        }

    def anotar(self, frame, deteccao):
        return frame


class PerfilVitima:
    """Vítimas pelo modelo treinado + plausibilidade física.

    Exige o modelo em ``config_resgate.VICTIM_MODEL_PATH``. Sem ele o perfil
    falha na hora, com a instrução do que fazer — nunca produz detecção falsa.
    """

    nome = "vitima"
    camera = "resgate"

    def __init__(self, alvo="any"):
        from visao.vitima_yolo import (ModeloAusenteError, VictimDetector,
                                       VictimModel)
        try:
            modelo = VictimModel().carregar()
        except ModeloAusenteError as err:
            raise SystemExit(f"\n[replay] {err}\n")
        self.detector = VictimDetector(model=modelo, target_kind=alvo)

    def processar(self, frame, timestamp):
        deteccao = self.detector.detect(frame, timestamp=timestamp)
        return deteccao is not None, {
            "confirmado": bool(
                deteccao is not None and deteccao.confirmed),
            "motivo": (
                "aceita" if deteccao is not None
                else self.detector.last_diagnostic),
            "tipo": deteccao.kind if deteccao is not None else "-",
            "confianca": (
                deteccao.confidence if deteccao is not None else 0.0),
        }

    def anotar(self, frame, deteccao):
        return frame


class PerfilTriangulos:
    """Os dois triângulos ao mesmo tempo, com as cores corretas do overlay."""

    nome = "triangulos"
    camera = "resgate"

    def __init__(self):
        from visao.triangulos_finais import (FinalTriangleMapper,
                                             annotate_final_triangles)
        self.mapper = FinalTriangleMapper()
        self._anotar = annotate_final_triangles

    def processar(self, frame, timestamp):
        deteccoes = self.mapper.update(frame, timestamp=timestamp)
        self._ultimas = deteccoes
        achou = any(valor is not None for valor in deteccoes.values())
        return achou, {
            "confirmado": self.mapper.both_found,
            "verde": deteccoes["green"] is not None,
            "vermelho": deteccoes["red"] is not None,
            "motivo": "aceita" if achou else "sem_triangulo",
        }

    def anotar(self, frame, deteccao):
        return self._anotar(frame, getattr(self, "_ultimas", {}))


PERFIS = {
    "entrada": PerfilEntrada,
    "saida": PerfilSaida,
    "vitima": PerfilVitima,
    "triangulos": PerfilTriangulos,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay dos detectores do Shadow, sem motores")
    parser.add_argument(
        "--perfil", choices=sorted(PERFIS), required=True,
        help="qual detector executar")
    fonte = parser.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--frames", type=Path, help="diretório de imagens")
    fonte.add_argument("--video", type=Path, help="arquivo de vídeo")
    parser.add_argument(
        "--esperado", choices=("positivo", "negativo"),
        help="rótulo de todo o conjunto, para medir acertos e falsos")
    parser.add_argument(
        "--salvar", type=Path,
        help="diretório onde gravar os frames anotados")
    parser.add_argument(
        "--alvo", choices=("any", "black", "silver"), default="any",
        help="tipo de esfera aceito no perfil vitima")
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="cadência simulada dos timestamps (padrão: 30)")
    parser.add_argument(
        "--silencioso", action="store_true",
        help="imprime só o resumo final")
    return parser.parse_args()


def main():
    args = parse_args()
    classe = PERFIS[args.perfil]
    perfil = (
        classe(alvo=args.alvo) if args.perfil == "vitima" else classe())

    if args.salvar is not None:
        args.salvar.mkdir(parents=True, exist_ok=True)

    print(f"[replay] perfil '{perfil.nome}' — imagens da câmera de "
          f"{perfil.camera}; motores nunca são acionados")

    periodo = 1.0 / max(args.fps, 1e-6)
    total = 0
    aceitos = 0
    confirmados = 0
    primeiro_confirmado = None
    motivos = {}
    duracoes = []

    for indice, (nome, frame) in enumerate(carregar_frames(args)):
        timestamp = indice * periodo
        inicio = time.perf_counter()
        aceito, info = perfil.processar(frame, timestamp)
        duracoes.append((time.perf_counter() - inicio) * 1000.0)

        total += 1
        aceitos += int(aceito)
        if info.get("confirmado"):
            confirmados += 1
            if primeiro_confirmado is None:
                primeiro_confirmado = indice + 1
        motivo = info.get("motivo", "?")
        motivos[motivo] = motivos.get(motivo, 0) + 1

        if not args.silencioso:
            extras = " ".join(
                f"{chave}={valor}" for chave, valor in info.items()
                if chave != "motivo")
            print(f"  {nome:28s} {'ACEITA ' if aceito else 'rejeita'} "
                  f"{motivo:24s} {extras}")

        if args.salvar is not None:
            anotado = perfil.anotar(frame.copy(), info)
            cv2.imwrite(str(args.salvar / f"{Path(nome).stem}.png"), anotado)

    if total == 0:
        raise SystemExit("nenhum frame processado")

    duracoes = np.asarray(duracoes)
    print("\n=== resumo ===")
    print(f"frames processados : {total}")
    print(f"candidatos aceitos : {aceitos} ({aceitos / total:.1%})")
    print(f"confirmados        : {confirmados} ({confirmados / total:.1%})")
    print(f"tempo mediano      : {np.median(duracoes):.1f} ms")
    print(f"tempo p95          : {np.percentile(duracoes, 95):.1f} ms")
    print("motivos:")
    for motivo, quantidade in sorted(
        motivos.items(), key=lambda item: -item[1]
    ):
        print(f"  {motivo:26s} {quantidade}")

    codigo = 0
    if args.esperado == "positivo":
        # Os primeiros frames de uma sequência positiva NÃO são falhas: a
        # votação temporal precisa acumular votos antes de confirmar. O que
        # mede o detector é a taxa de aceitação por frame; o que mede a
        # votação é em qual frame veio a primeira confirmação.
        perdidos = total - aceitos
        print(f"\nfalsos negativos por frame : {perdidos} de {total}")
        if primeiro_confirmado is None:
            print(
                "FALHA: a sequência inteira não gerou nenhuma confirmação.")
            codigo = 1
        else:
            print(
                f"primeira confirmação       : frame {primeiro_confirmado} "
                f"(votação exige acumular votos)")
        if perdidos:
            print(
                "AVISO: alguns frames positivos não produziram candidato; "
                "verifique iluminação e limiares no calibrador.")
    elif args.esperado == "negativo":
        print(f"\nfalsos positivos   : {confirmados} de {total}")
        if confirmados:
            print(
                "FALHA: um negativo difícil foi CONFIRMADO. Não prossiga "
                "para os testes com motores antes de resolver isto.")
            codigo = 1
        else:
            print("Nenhum falso acionamento neste conjunto.")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
