"""Analyzer visual-only para marcadores verdes OBR por adjacencia da linha preta."""

import argparse
import pprint
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import CAMERA_HEIGHT, CAMERA_WIDTH, GREEN_INTERVALO_DEBUG, PASTA_CAPTURAS
from raspberry.backup.verde.green_action import analisar_intersecao_preta
from raspberry.backup.verde.green_detector import criar_mascara_linha_global, detectar_verde
from line_test import detectar_linha


GREEN_ADJ_MARGIN_MULT = 0.8
GREEN_ADJ_MIN_MARGIN_PX = 8
GREEN_ADJ_MIN_BLACK_PIXELS = 25
GREEN_ADJ_MIN_BLACK_RATIO = 0.08
GREEN_ADJ_MIN_AREA = 350

INTERSECOES_OBR_VALIDAS = {"CRUZ", "LATERAL_ESQ", "LATERAL_DIR"}


def avaliar_confianca_intersecao(tipo_intersecao):
    """Retorna a confianca contextual da intersecao, sem bloquear adjacencia."""
    if tipo_intersecao in INTERSECOES_OBR_VALIDAS:
        return "OK", "intersecao_valida"
    if tipo_intersecao in {"RETA", "NENHUMA"}:
        return "BAIXA", "sem_intersecao_confirmada"
    return "BAIXA", "intersecao_ambigua"


def _clamp_roi(x1, y1, x2, y2, largura_frame, altura_frame):
    return (
        max(0, min(largura_frame, int(x1))),
        max(0, min(altura_frame, int(y1))),
        max(0, min(largura_frame, int(x2))),
        max(0, min(altura_frame, int(y2))),
    )


def _medir_roi(mascara_linha_global, roi):
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return {"pixels": 0, "ratio": 0.0, "area": 0}
    recorte = mascara_linha_global[y1:y2, x1:x2]
    area = int(recorte.size)
    pixels = int(cv2.countNonZero(recorte))
    return {"pixels": pixels, "ratio": pixels / area if area else 0.0, "area": area}


def _tem_preto(medicao):
    return (
        medicao["pixels"] >= GREEN_ADJ_MIN_BLACK_PIXELS
        or medicao["ratio"] >= GREEN_ADJ_MIN_BLACK_RATIO
    )


def analisar_adjacencia_preta_verde(
    contorno, mascara_linha_global, largura_frame, altura_frame
):
    """Mede preto imediatamente acima, abaixo, a esquerda e a direita do verde."""
    x, y, w, h = contorno["bbox"]
    margem_x = max(GREEN_ADJ_MIN_MARGIN_PX, int(w * GREEN_ADJ_MARGIN_MULT))
    margem_y = max(GREEN_ADJ_MIN_MARGIN_PX, int(h * GREEN_ADJ_MARGIN_MULT))
    rois = {
        "top": _clamp_roi(
            x - margem_x, y - margem_y, x + w + margem_x, y, largura_frame, altura_frame
        ),
        "bottom": _clamp_roi(
            x - margem_x,
            y + h,
            x + w + margem_x,
            y + h + margem_y,
            largura_frame,
            altura_frame,
        ),
        "left": _clamp_roi(x - margem_x, y, x, y + h, largura_frame, altura_frame),
        "right": _clamp_roi(
            x + w, y, x + w + margem_x, y + h, largura_frame, altura_frame
        ),
    }
    medicoes = {nome: _medir_roi(mascara_linha_global, roi) for nome, roi in rois.items()}
    contorno.update(
        {
            "adj_rois": rois,
            "black_top_pixels": medicoes["top"]["pixels"],
            "black_bottom_pixels": medicoes["bottom"]["pixels"],
            "black_left_pixels": medicoes["left"]["pixels"],
            "black_right_pixels": medicoes["right"]["pixels"],
            "black_top_ratio": medicoes["top"]["ratio"],
            "black_bottom_ratio": medicoes["bottom"]["ratio"],
            "black_left_ratio": medicoes["left"]["ratio"],
            "black_right_ratio": medicoes["right"]["ratio"],
            "black_top": _tem_preto(medicoes["top"]),
            "black_bottom": _tem_preto(medicoes["bottom"]),
            "black_left": _tem_preto(medicoes["left"]),
            "black_right": _tem_preto(medicoes["right"]),
        }
    )
    contorno["adj"] = adjacencia_compacta(contorno)
    return contorno


def adjacencia_compacta(contorno):
    """Formata T/B/L/R de um contorno ja analisado."""
    return (
        f"T{1 if contorno.get('black_top') else 0}"
        f"B{1 if contorno.get('black_bottom') else 0}"
        f"L{1 if contorno.get('black_left') else 0}"
        f"R{1 if contorno.get('black_right') else 0}"
    )


def classificar_marcador_verde_por_adjacencia(contorno):
    """Classifica um marcador verde confirmado usando o padrao de preto ao redor."""
    if not contorno.get("confirmado", False):
        return "MARCADOR_INVALIDO", "verde_nao_confirmado"
    if contorno.get("area", 0.0) < GREEN_ADJ_MIN_AREA:
        return "MARCADOR_INVALIDO", "verde_area_pequena"
    if contorno.get("lado") == "CENTRO":
        return "MARCADOR_AMBIGUO", "verde_central_ambiguo"

    top = contorno.get("black_top", False)
    bottom = contorno.get("black_bottom", False)
    left = contorno.get("black_left", False)
    right = contorno.get("black_right", False)

    if not top and bottom:
        return "MARCADOR_INVALIDO", "verde_depois_ou_falso"
    if not top:
        return "MARCADOR_INVALIDO", "sem_preto_acima"
    if left and right:
        return "MARCADOR_AMBIGUO", "verde_ambiguo"
    if bottom and not (left or right):
        return "MARCADOR_INVALIDO", "verde_depois_ou_falso"
    if right and not left:
        return "MARCADOR_ESQUERDA", "marcador_esq"
    if left and not right:
        return "MARCADOR_DIREITA", "marcador_dir"
    return "MARCADOR_INVALIDO", "padrao_desconhecido"


def _intersecao_tipo(analise_intersecao):
    if not analise_intersecao:
        return "NENHUMA"
    return analise_intersecao.get("tipo_intersecao", "NENHUMA")


def _resumir_adj(contornos):
    confirmados = [item for item in contornos if item.get("confirmado")]
    if not confirmados:
        return "-"
    if len(confirmados) == 1:
        return confirmados[0].get("adj", "-")
    partes = []
    for item in confirmados[:5]:
        prefixo = "E" if item.get("lado") == "ESQUERDA" else "D" if item.get("lado") == "DIREITA" else "C"
        partes.append(f"{prefixo}:{item.get('adj', '-')}")
    return " ".join(partes)


def decidir_verde_obr_por_adjacencia(
    resultado_verde, mascara_linha_global, analise_intersecao=None
):
    """Decide ESQ/DIR/RETORNO/RETO por adjacencia preto-verde, sem movimento."""
    altura_frame, largura_frame = mascara_linha_global.shape[:2]
    tipo_intersecao = _intersecao_tipo(analise_intersecao)
    confianca, motivo_confianca = avaliar_confianca_intersecao(tipo_intersecao)
    contornos_detectados = [dict(item) for item in resultado_verde.get("contornos", [])]
    contornos_confirmados = [
        dict(item) for item in resultado_verde.get("contornos_confirmados", [])
    ]
    contornos_rejeitados = [
        dict(item) for item in contornos_detectados if not item.get("confirmado", False)
    ]
    for contorno in contornos_confirmados:
        analisar_adjacencia_preta_verde(
            contorno, mascara_linha_global, largura_frame, altura_frame
        )
        classe, motivo = classificar_marcador_verde_por_adjacencia(contorno)
        contorno["marcador"] = classe
        contorno["motivo_marcador"] = motivo

    detectados = resultado_verde.get(
        "qtd_contornos_detectados", len(contornos_detectados)
    )
    confirmados = resultado_verde.get(
        "qtd_contornos_confirmados", len(contornos_confirmados)
    )
    base = {
        "decisao": "RETO",
        "verde": "NENHUM",
        "qtd_detectados": detectados,
        "qtd_confirmados": confirmados,
        "qtd_validos": 0,
        "intersecao": tipo_intersecao,
        "confianca": confianca,
        "motivo_confianca": motivo_confianca,
        "motivo": "sem_verde",
        "adj": _resumir_adj(contornos_confirmados),
        "contornos": contornos_confirmados,
        "contornos_detectados": contornos_detectados,
        "contornos_confirmados": contornos_confirmados,
        "contornos_rejeitados": contornos_rejeitados,
        "analise_intersecao": analise_intersecao,
    }

    if confirmados == 0:
        motivo = "verde_nao_confirmado" if detectados else "sem_verde"
        return {**base, "motivo": motivo}

    validos_esq = [item for item in contornos_confirmados if item.get("marcador") == "MARCADOR_ESQUERDA"]
    validos_dir = [item for item in contornos_confirmados if item.get("marcador") == "MARCADOR_DIREITA"]
    validos = validos_esq + validos_dir
    ambiguos = [item for item in contornos_confirmados if item.get("marcador") == "MARCADOR_AMBIGUO"]
    falsos = [
        item
        for item in contornos_confirmados
        if item.get("motivo_marcador") == "verde_depois_ou_falso"
    ]
    base["qtd_validos"] = len(validos)

    if not validos:
        if falsos:
            motivo = "verde_depois_ou_falso"
        elif ambiguos:
            motivo = "verde_ambiguo"
        else:
            motivo = contornos_confirmados[0].get("motivo_marcador", "padrao_desconhecido")
        verde = "AMBIGUO" if motivo == "verde_ambiguo" else resultado_verde.get("tipo_confirmado", "NENHUM")
        return {**base, "verde": verde, "motivo": motivo}

    if ambiguos:
        return {**base, "verde": "AMBIGUO", "motivo": "verde_ambiguo"}
    if len(validos) > 2 or len(validos_esq) > 1 or len(validos_dir) > 1:
        return {**base, "verde": "AMBIGUO", "motivo": "verde_ambiguo"}
    if validos_esq and validos_dir:
        motivo = "duplo_valido" if confianca == "OK" else "duplo_sem_intersecao"
        return {
            **base,
            "decisao": "RETORNO",
            "verde": "DUPLO",
            "qtd_validos": len(validos),
            "motivo": motivo,
        }
    if validos_esq:
        motivo = "marcador_esq" if confianca == "OK" else "marcador_esq_sem_intersecao"
        return {
            **base,
            "decisao": "ESQ",
            "verde": "ESQUERDA",
            "qtd_validos": len(validos),
            "motivo": motivo,
        }
    motivo = "marcador_dir" if confianca == "OK" else "marcador_dir_sem_intersecao"
    return {
        **base,
        "decisao": "DIR",
        "verde": "DIREITA",
        "qtd_validos": len(validos),
        "motivo": motivo,
    }


def formatar_log_compacto(decisao):
    return (
        f"[GOBR] dec={decisao['decisao']} | "
        f"v={decisao['verde']} {decisao['qtd_confirmados']}/{decisao['qtd_detectados']} | "
        f"adj={decisao['adj']} | int={decisao['intersecao']} | "
        f"conf={decisao['confianca']} | "
        f"motivo={decisao['motivo']}"
    )


def formatar_log_detalhe(decisao):
    linhas = []
    confirmados = decisao.get("contornos_confirmados", decisao.get("contornos", []))
    rejeitados = decisao.get("contornos_rejeitados", [])
    if confirmados:
        linhas.append("confirmados:")
    for indice, contorno in enumerate(confirmados[:5], start=1):
        linhas.append(
            f"verde#{indice} lado={contorno.get('lado')} bbox={contorno.get('bbox')} "
            f"area={contorno.get('area', 0):.0f} S={contorno.get('mean_s', 0):.0f} "
            f"GR={contorno.get('g_minus_r', 0):.0f} GB={contorno.get('g_minus_b', 0):.0f} "
            f"conf={contorno.get('confirmado', False)} motivo={contorno.get('motivo_marcador')} "
            f"adj=T{int(contorno.get('black_top', False))}/B{int(contorno.get('black_bottom', False))}/"
            f"L{int(contorno.get('black_left', False))}/R{int(contorno.get('black_right', False))} "
            f"ratios={contorno.get('black_top_ratio', 0):.2f}/"
            f"{contorno.get('black_bottom_ratio', 0):.2f}/"
            f"{contorno.get('black_left_ratio', 0):.2f}/"
            f"{contorno.get('black_right_ratio', 0):.2f}"
        )
    if rejeitados:
        linhas.append("rejeitados:")
    for indice, contorno in enumerate(rejeitados[:5], start=1):
        linhas.append(
            f"verde#{indice} lado={contorno.get('lado')} bbox={contorno.get('bbox')} "
            f"area={contorno.get('area', 0):.0f} S={contorno.get('mean_s', 0):.0f} "
            f"GR={contorno.get('g_minus_r', 0):.0f} GB={contorno.get('g_minus_b', 0):.0f} "
            f"confirmado={contorno.get('confirmado', False)} "
            f"motivo_confirmacao={contorno.get('motivo_confirmacao')} "
            f"black={contorno.get('black_near_pixels', 0)} "
            f"areaZona={contorno.get('area_in_confirm_zone_ratio', 0):.2f}"
        )
    return "\n".join(linhas)


class LogCompacto:
    def __init__(self, intervalo=0.25):
        self.intervalo = intervalo
        self.ultima_linha = None
        self.ultimo_tempo = 0.0

    def deve_logar(self, linha, agora):
        mudou = linha != self.ultima_linha
        passou_tempo = agora - self.ultimo_tempo >= self.intervalo
        if mudou or passou_tempo:
            self.ultima_linha = linha
            self.ultimo_tempo = agora
            return True
        return False


def processar_frame(frame):
    resultado_linha = detectar_linha(frame)
    mascara_linha = criar_mascara_linha_global(resultado_linha)
    x_referencia = resultado_linha["centro_imagem_x"]
    verde = detectar_verde(frame, x_referencia, mascara_linha)
    intersecao = analisar_intersecao_preta(mascara_linha, x_referencia)
    decisao = decidir_verde_obr_por_adjacencia(verde, mascara_linha, intersecao)
    decisao["mascara_linha_global"] = mascara_linha
    return resultado_linha, verde, intersecao, decisao


def _desenhar_roi(debug, roi, ativo):
    x1, y1, x2, y2 = roi
    cor = (0, 0, 255) if ativo else (128, 128, 128)
    cv2.rectangle(debug, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), cor, 1)


def criar_debug_obr(frame, decisao):
    debug = frame.copy()
    mascara = decisao.get("mascara_linha_global")
    if mascara is not None:
        overlay = debug.copy()
        overlay[mascara > 0] = (255, 255, 0)
        debug = cv2.addWeighted(debug, 0.78, overlay, 0.22, 0)

    for contorno in decisao.get("contornos_rejeitados", []):
        x, y, w, h = contorno["bbox"]
        cor = (0, 140, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(
            debug,
            f"REJ {contorno.get('motivo_confirmacao', '')}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cor,
            1,
        )

    for contorno in decisao.get("contornos_confirmados", decisao.get("contornos", [])):
        x, y, w, h = contorno["bbox"]
        valido = contorno.get("marcador") in {"MARCADOR_ESQUERDA", "MARCADOR_DIREITA"}
        cor = (0, 255, 0) if valido else (0, 255, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), cor, 2)
        for nome in ("top", "bottom", "left", "right"):
            _desenhar_roi(debug, contorno["adj_rois"][nome], contorno.get(f"black_{nome}", False))
        cv2.putText(
            debug,
            f"{contorno.get('lado')} {contorno.get('adj')} {contorno.get('motivo_marcador')}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cor,
            1,
        )

    texto = (
        f"GOBR: {decisao['decisao']} | v={decisao['verde']} | "
        f"adj={decisao['adj']} | int={decisao['intersecao']}"
    )
    cv2.putText(debug, texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug, texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return debug


def salvar_debug(frame, decisao):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_green_obr_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    if not cv2.imwrite(str(caminho), criar_debug_obr(frame, decisao)):
        raise RuntimeError(f"Nao foi possivel salvar: {caminho}")
    print(f"Debug salvo: {caminho}")


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Green OBR analyzer visual-only.")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--camera", action="store_true")
    origem.add_argument("--imagem")
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--salvar-debug", action="store_true")
    parser.add_argument("--log-detalhe", action="store_true")
    parser.add_argument("--log-full", action="store_true")
    return parser.parse_args()


def _imprimir(decisao, logador, detalhe=False, full=False):
    linha = formatar_log_compacto(decisao)
    if logador.deve_logar(linha, time.monotonic()):
        print(linha)
        if detalhe:
            detalhes = formatar_log_detalhe(decisao)
            if detalhes:
                print(detalhes)
        if full:
            pprint.pprint(decisao)


def executar_imagem(caminho, mostrar, salvar, detalhe, full):
    frame = cv2.imread(str(caminho))
    if frame is None:
        raise RuntimeError("Nao foi possivel carregar a imagem.")
    _, _, _, decisao = processar_frame(frame)
    _imprimir(decisao, LogCompacto(0.0), detalhe, full)
    if salvar:
        salvar_debug(frame, decisao)
    if mostrar:
        cv2.imshow("Green OBR Analyzer", criar_debug_obr(frame, decisao))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def executar_camera(mostrar, salvar, detalhe, full):
    camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
    logador = LogCompacto()
    ultimo_estado, ultimo_debug = None, 0.0
    try:
        while True:
            frame = capturar_frame_bgr(camera)
            _, _, _, decisao = processar_frame(frame)
            _imprimir(decisao, logador, detalhe, full)
            estado = (decisao["decisao"], decisao["verde"], decisao["motivo"])
            agora = time.monotonic()
            if salvar and estado != ultimo_estado and agora - ultimo_debug >= GREEN_INTERVALO_DEBUG:
                salvar_debug(frame, decisao)
                ultimo_estado, ultimo_debug = estado, agora
            if mostrar:
                cv2.imshow("Green OBR Analyzer - q para sair", criar_debug_obr(frame, decisao))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.stop()
        if mostrar:
            cv2.destroyAllWindows()


def main():
    argumentos = ler_argumentos()
    if not argumentos.camera and not argumentos.imagem:
        print("Use --camera ou --imagem.")
        return 0
    try:
        if argumentos.camera:
            executar_camera(
                argumentos.mostrar,
                argumentos.salvar_debug,
                argumentos.log_detalhe,
                argumentos.log_full,
            )
        else:
            executar_imagem(
                argumentos.imagem,
                argumentos.mostrar,
                argumentos.salvar_debug,
                argumentos.log_detalhe,
                argumentos.log_full,
            )
        return 0
    except (RuntimeError, ValueError, cv2.error) as erro:
        print(f"Erro: {erro}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
