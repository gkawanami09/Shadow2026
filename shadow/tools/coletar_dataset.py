#!/usr/bin/env python3
"""Captura em massa de imagens para treinar a visão do resgate. SEM motores.

Este programa existe porque apertar `s` centenas de vezes não é viável. Você
liga, posiciona o robô, e ele grava frames brutos continuamente enquanto você
muda o cenário ao redor.

Regras que o programa impõe por construção
------------------------------------------
* **Nunca abre a serial e nunca move o robô.** Só a câmera.
* **Uma pasta por sessão.** É isso que permite dividir treino/validação por
  sessão depois. Dividir aleatoriamente frames de uma mesma gravação coloca
  quase-cópias dos dois lados e infla o resultado de forma absurda.
* **Frames sem anotação nenhuma.** O PNG é exatamente o que a câmera viu.
* **Intervalo mínimo entre frames.** Sem isso você grava 500 cópias do mesmo
  instante e desperdiça rotulagem.

O que você deve variar entre sessões
------------------------------------
Fundo, piso, parede, iluminação — quanto mais, melhor. O que NÃO pode variar
é a câmera e a montagem: use o próprio robô como tripé, na altura de
trabalho. E mantenha a esfera no chão, no terço inferior do quadro, que é
onde ela realmente aparece para este robô.

Exemplos::

    # sessão de vítima prateada na cozinha, 1 frame a cada 0,4 s
    python3 shadow/tools/coletar_dataset.py --sessao prata_cozinha

    # sessão de negativos (sem vítima), mais rápida
    python3 shadow/tools/coletar_dataset.py --sessao negativos_sala \\
        --intervalo 0.25 --max-frames 400

    # sem preview (mais leve, para gravar andando)
    python3 shadow/tools/coletar_dataset.py --sessao arena_01 --sem-preview
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

import config_resgate as cfg  # noqa: E402


RAIZ = Path(__file__).resolve().parents[1] / "captures" / "dataset"
JANELA = "Shadow2026 - coleta de dataset (ESPACO pausa, q sai)"


def criar_sessao(nome, notas):
    """Cria a pasta da sessão. Falha se já existir, para não misturar."""
    carimbo = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta = RAIZ / f"{carimbo}_{nome}"
    pasta.mkdir(parents=True, exist_ok=False)
    (pasta / "sessao.json").write_text(
        json.dumps(
            {
                "sessao": nome,
                "criada_em": dt.datetime.now().isoformat(timespec="seconds"),
                "notas": notas,
                "camera": "resgate",
                "aviso": (
                    "Frames desta pasta pertencem a UMA sessao. Ao dividir "
                    "treino/validacao, mantenha a pasta inteira de um lado "
                    "so — frames vizinhos sao quase identicos."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pasta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coleta frames brutos para treinar a visão do resgate")
    parser.add_argument(
        "--sessao", required=True,
        help=(
            "nome curto da sessão, ex: prata_cozinha, negativos_sala, "
            "preta_arena. Vira o nome da pasta"))
    parser.add_argument(
        "--intervalo", type=float, default=0.40,
        help=(
            "segundos entre frames gravados (padrão: 0.40). Abaixo de 0.2 "
            "você grava quase-cópias e desperdiça rotulagem"))
    parser.add_argument(
        "--max-frames", type=int, default=600,
        help="para sozinho depois deste número (padrão: 600)")
    parser.add_argument(
        "--camera-index", type=int, default=None,
        help=f"índice da câmera (padrão: {cfg.RESCUE_CAMERA_INDEX}, resgate)")
    parser.add_argument(
        "--linha", action="store_true",
        help=(
            "usa a CÂMERA DE LINHA em vez da de resgate; para coletar a "
            "faixa prata de entrada"))
    parser.add_argument(
        "--sem-preview", action="store_true",
        help="não abre janela (mais leve; use por SSH ou gravando andando)")
    parser.add_argument(
        "--notas", default="",
        help="observações da sessão: iluminação, piso, o que está no quadro")
    return parser.parse_args()


def abrir_camera(args):
    if args.linha:
        from visao.captura import LineCamera
        print("[coleta] abrindo a CÂMERA DE LINHA")
        return LineCamera(), "linha"
    from visao.captura_resgate import RescueCamera
    indice = (
        cfg.RESCUE_CAMERA_INDEX
        if args.camera_index is None else args.camera_index)
    print(f"[coleta] abrindo a câmera de RESGATE (índice {indice})")
    return RescueCamera(indice), "resgate"


def main():
    args = parse_args()
    if args.intervalo < 0.15:
        print(
            "[coleta] AVISO: intervalo muito curto; frames vizinhos vão sair "
            "quase idênticos e só dão trabalho de rotular.")

    camera = None
    pasta = None
    gravados = 0
    pausado = False

    try:
        camera, qual = abrir_camera(args)
        pasta = criar_sessao(args.sessao, args.notas)
        print(f"[coleta] sessão: {pasta}")
        print(
            "[coleta] motores NUNCA são acionados — a serial nem é aberta.")
        if args.sem_preview:
            print("[coleta] sem preview. Ctrl-C para encerrar.")
        else:
            print("[coleta] ESPAÇO pausa/retoma, q encerra.")

        proximo = time.monotonic()
        while gravados < args.max_frames:
            frame = camera.get_frame()
            if frame is None:
                continue
            agora = time.monotonic()

            if not pausado and agora >= proximo:
                nome = f"{args.sessao}_{gravados:05d}.png"
                # Frame BRUTO, sem nenhuma anotação.
                cv2.imwrite(str(pasta / nome), frame)
                gravados += 1
                proximo = agora + args.intervalo
                if gravados % 25 == 0:
                    print(f"[coleta] {gravados}/{args.max_frames}")

            if not args.sem_preview:
                preview = frame.copy()
                altura, largura = preview.shape[:2]
                # Guias do enquadramento correto: a esfera deve ficar no
                # terço inferior, que é onde este robô realmente a vê.
                cv2.line(
                    preview, (0, int(altura * 0.66)),
                    (largura, int(altura * 0.66)), (0, 200, 255), 1)
                cv2.putText(
                    preview, "mantenha a esfera ABAIXO desta linha",
                    (8, int(altura * 0.66) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                estado = "PAUSADO" if pausado else "GRAVANDO"
                cor = (0, 200, 255) if pausado else (0, 255, 0)
                cv2.putText(
                    preview, f"{estado}  {gravados}/{args.max_frames}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, cor, 2)
                cv2.imshow(JANELA, preview)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord(" "):
                    pausado = not pausado
                    print(f"[coleta] {'pausado' if pausado else 'gravando'}")
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[coleta] Ctrl-C")
    except Exception as err:                       # noqa: BLE001
        print(f"[coleta] ERRO: {err}")
        return 1
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if pasta is not None:
            print(f"\n[coleta] {gravados} frames em {pasta}")
            print(
                "[coleta] Suba ESTA PASTA inteira no Roboflow como um lote "
                "próprio. Não misture com outras sessões antes de dividir "
                "treino/validação.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
