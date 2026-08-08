#!/usr/bin/env python3
"""Mede o custo real de cada detector. Reprodutível, sem motores.

Existe porque decisão de visão sem número é chute. Antes de trocar um
detector clássico por um modelo, ou de subir a resolução, é preciso saber
quanto custa o que já existe — no Raspberry Pi 5, não no notebook.

Modos
-----
``--sintetico``  gera as cenas do próprio repositório; roda em qualquer
                 máquina e serve de linha de base comparável entre PCs;
``--frames``     usa imagens reais da arena;
``--camera``     abre a câmera de resgate no Pi e mede captura e visão
                 SEPARADAMENTE, que é a única forma de ver backlog.

O modo ``--camera`` adquire a trava dos motores como "benchmark-visao" para
garantir que nenhum outro processo esteja usando a serial ou a câmera ao mesmo
tempo. Ele nunca envia comando de movimento.

Exemplos::

    python3 shadow/tools/benchmark_visao.py --sintetico --repeticoes 60
    python3 shadow/tools/benchmark_visao.py --frames shadow/captures/arena
    python3 shadow/tools/benchmark_visao.py --camera --segundos 20
"""

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import config_resgate as cfg  # noqa: E402


EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp")


def _estatisticas(amostras_ms):
    dados = np.asarray(amostras_ms, dtype=np.float64)
    if dados.size == 0:
        return None
    return {
        "n": int(dados.size),
        "mediana": float(np.median(dados)),
        "media": float(np.mean(dados)),
        "p95": float(np.percentile(dados, 95)),
        "max": float(np.max(dados)),
        "fps": float(1000.0 / max(np.median(dados), 1e-6)),
    }


def _imprimir(nome, stats):
    if stats is None:
        print(f"  {nome:26s} sem amostras")
        return
    print(
        f"  {nome:26s} mediana {stats['mediana']:7.1f} ms | "
        f"p95 {stats['p95']:7.1f} ms | max {stats['max']:7.1f} ms | "
        f"~{stats['fps']:5.1f} fps | n={stats['n']}")


def _cpu_percent():
    """Uso aproximado de CPU do processo, sem depender de psutil."""
    try:
        uso = os.times()
        return uso.user + uso.system
    except Exception:
        return None


def cenas_sinteticas_linha(repeticoes):
    from tests import cenas_sinteticas as cs
    base = [
        cs.faixa_prata(),
        cs.piso_branco(),
        cs.reflexo_pontual(),
        cs.esfera(cs.LINE_FRAME, 120, 210, 120),
        cs.piso_neutro(),
    ]
    return [base[i % len(base)] for i in range(repeticoes)]


def cenas_sinteticas_resgate(repeticoes):
    from tests import cenas_sinteticas as cs
    base = [
        cs.faixa_preta(),
        cs.esfera(cs.RESCUE_FRAME, 70, 210, 120),
        cs.esfera(cs.RESCUE_FRAME, 70, 20, 185),
        cs.madeira(),
        cs.piso_neutro(cs.RESCUE_FRAME, 150),
    ]
    return [base[i % len(base)] for i in range(repeticoes)]


def carregar_frames(caminho, limite):
    arquivos = sorted(
        item for item in Path(caminho).iterdir()
        if item.suffix.lower() in EXTENSOES)
    frames = []
    for item in arquivos[:limite]:
        frame = cv2.imread(str(item), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise SystemExit(f"nenhuma imagem utilizável em {caminho}")
    return frames


def medir(detector_fn, frames, aquecimento=3):
    """Roda o detector sobre os frames e devolve os tempos em ms."""
    for frame in frames[:aquecimento]:
        detector_fn(frame, 0.0)
    tempos = []
    for indice, frame in enumerate(frames):
        inicio = time.perf_counter()
        detector_fn(frame, indice / 30.0)
        tempos.append((time.perf_counter() - inicio) * 1000.0)
    return tempos


def benchmark_offline(frames_linha, frames_resgate):
    from visao.vitima_yolo import VictimDetector as BallDetector
    from visao.entrada_missao import EntryModel
    from visao.faixa_saida import BlackExitDetector
    from visao.marcador_resgate import MarkerDetector
    from visao.triangulos_finais import FinalTriangleMapper

    resultados = {}

    entrada = EntryModel().load()
    resultados["faixa prata (linha)"] = _estatisticas(medir(
        lambda frame, ts: entrada.detect(frame),
        frames_linha))

    saida = BlackExitDetector()
    resultados["faixa preta (resgate)"] = _estatisticas(medir(
        lambda frame, ts: saida.detect(frame, timestamp=ts),
        frames_resgate))

    marcador = MarkerDetector("green")
    resultados["triangulo verde"] = _estatisticas(medir(
        lambda frame, ts: marcador.detect(frame, timestamp=ts),
        frames_resgate))

    mapper = FinalTriangleMapper()
    resultados["dois triangulos"] = _estatisticas(medir(
        lambda frame, ts: mapper.update(frame, timestamp=ts),
        frames_resgate))

    # O detector de esferas é o mais caro e o que decide a resolução de
    # trabalho. Medido na resolução real do detector (320x240 por padrão).
    reduzidos = [
        cv2.resize(
            frame,
            (cfg.RESCUE_DETECTOR_MAX_WIDTH, cfg.RESCUE_DETECTOR_MAX_HEIGHT))
        for frame in frames_resgate
    ]
    # O modelo de vítimas só entra no benchmark quando existir. Antes disso
    # não há o que medir, e inventar um número seria pior que omitir.
    from visao.vitima_yolo import (ModeloAusenteError, VictimDetector,
                                   VictimModel)
    try:
        modelo = VictimModel().carregar()
    except ModeloAusenteError:
        print(
            "  (modelo de vitimas ausente — o custo da inferencia so pode "
            "ser medido depois do treino)")
        return resultados

    vitima = VictimDetector(model=modelo)
    resultados[
        f"vitima {cfg.RESCUE_DETECTOR_MAX_WIDTH}x"
        f"{cfg.RESCUE_DETECTOR_MAX_HEIGHT}"
    ] = _estatisticas(medir(
        lambda frame, ts: vitima.detect(frame, timestamp=ts), reduzidos))

    vitima_cheia = VictimDetector(model=modelo)
    resultados["vitima resolucao cheia"] = _estatisticas(medir(
        lambda frame, ts: vitima_cheia.detect(frame, timestamp=ts),
        frames_resgate))

    return resultados


def benchmark_camera(segundos):
    """Mede captura e visão separadamente, com o modelo latest-frame real."""
    from controle.trava_motores import MotorLockError, MotorOwnerLock
    from visao.vitima_yolo import VictimDetector as BallDetector
    from visao.captura_resgate import RescueCamera
    from visao.resgate_assincrono import (LatestFrameBallDetector,
                                          LatestFrameSource)

    trava = MotorOwnerLock("benchmark-visao")
    try:
        trava.acquire()
    except MotorLockError as err:
        raise SystemExit(f"benchmark recusado: {err}")

    captura = None
    detector = None
    try:
        captura = LatestFrameSource(RescueCamera(cfg.RESCUE_CAMERA_INDEX))
        from visao.vitima_yolo import VictimModel
        detector = LatestFrameBallDetector(
            BallDetector(model=VictimModel().carregar()),
            max_width=cfg.RESCUE_DETECTOR_MAX_WIDTH,
            max_height=cfg.RESCUE_DETECTOR_MAX_HEIGHT,
        )

        fim = time.monotonic() + segundos
        sequencia_frame = 0
        sequencia_resultado = 0
        capturas = []
        visoes = []
        processamentos = []
        descartados = 0
        idades = []
        cpu_inicio = _cpu_percent()
        inicio = time.monotonic()

        while time.monotonic() < fim:
            pacote = captura.poll(sequencia_frame)
            if pacote is not None:
                sequencia_frame = pacote.sequence
                capturas.append(pacote.captured_at)
                detector.submit(
                    pacote.frame,
                    captured_at=pacote.captured_at,
                    source_sequence=pacote.sequence)
            resultado = detector.poll(sequencia_resultado)
            if resultado is not None:
                sequencia_resultado = resultado.sequence
                visoes.append(resultado.completed_at)
                processamentos.append(resultado.processing_s * 1000.0)
                descartados = resultado.dropped_frames
                idades.append(
                    (time.monotonic() - resultado.captured_at) * 1000.0)
            time.sleep(0.002)

        decorrido = time.monotonic() - inicio
        cpu_fim = _cpu_percent()

        print("\n=== câmera de resgate ao vivo ===")
        print(f"duração                    : {decorrido:.1f} s")
        print(
            f"FPS de captura             : "
            f"{len(capturas) / max(decorrido, 1e-6):.1f}")
        print(
            f"FPS efetivo da visão       : "
            f"{len(visoes) / max(decorrido, 1e-6):.1f}")
        print(f"frames descartados         : {descartados}")
        _imprimir("processamento", _estatisticas(processamentos))
        atraso = _estatisticas(idades)
        if atraso is not None:
            print(
                f"  atraso ate comandar motor  mediana "
                f"{atraso['mediana']:.1f} ms | p95 {atraso['p95']:.1f} ms | "
                f"max {atraso['max']:.1f} ms")
            print(
                f"  limite de frame stale      "
                f"{cfg.BALL_FRAME_STALE_S * 1000:.0f} ms")
        if cpu_inicio is not None and cpu_fim is not None:
            print(
                f"CPU do processo            : "
                f"{(cpu_fim - cpu_inicio) / max(decorrido, 1e-6) * 100:.0f}% "
                "de um núcleo")
    finally:
        if detector is not None:
            detector.close(timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
        if captura is not None:
            captura.close(timeout=cfg.RESCUE_WORKER_JOIN_TIMEOUT_S)
        trava.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark reprodutível da visão do Shadow")
    fonte = parser.add_mutually_exclusive_group(required=True)
    fonte.add_argument(
        "--sintetico", action="store_true",
        help="usa as cenas geradas do repositório (roda em qualquer máquina)")
    fonte.add_argument(
        "--frames", type=Path, help="diretório de imagens reais da arena")
    fonte.add_argument(
        "--camera", action="store_true",
        help="abre a câmera de resgate no Pi e mede captura vs visão")
    parser.add_argument(
        "--repeticoes", type=int, default=40,
        help="frames processados nos modos offline (padrão: 40)")
    parser.add_argument(
        "--segundos", type=float, default=15.0,
        help="duração do modo --camera (padrão: 15)")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Python {sys.version.split()[0]} | OpenCV {cv2.__version__}")
    print(f"threads do OpenCV: {cv2.getNumThreads()}")
    print(
        f"detector de esfera: {cfg.RESCUE_DETECTOR_MAX_WIDTH}x"
        f"{cfg.RESCUE_DETECTOR_MAX_HEIGHT} | "
        f"linha: {config.camera_x}x{config.camera_y}")

    if args.camera:
        benchmark_camera(args.segundos)
        return 0

    if args.frames is not None:
        frames = carregar_frames(args.frames, args.repeticoes)
        frames_linha = frames
        frames_resgate = frames
        print(f"\nfonte: {len(frames)} imagens de {args.frames}")
    else:
        frames_linha = cenas_sinteticas_linha(args.repeticoes)
        frames_resgate = cenas_sinteticas_resgate(args.repeticoes)
        print(f"\nfonte: cenas sintéticas ({args.repeticoes} frames)")

    print("\n=== tempo por detector ===")
    for nome, stats in benchmark_offline(frames_linha, frames_resgate).items():
        _imprimir(nome, stats)

    print(
        "\nObservação: números medidos NESTA máquina. Para decidir qualquer "
        "coisa sobre resolução ou trocar de detector, rode o mesmo comando "
        "no Raspberry Pi 5 do robô.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
