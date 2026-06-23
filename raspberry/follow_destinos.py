"""Segue-linha por destinos visuais hierarquicos amostrados por raios."""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    BAUD_RATE, CAMERA_HEIGHT, CAMERA_WIDTH, PASTA_CAPTURAS, SERIAL_PORT, TIMEOUT_SERIAL,
    DEST_ALPHA_SUAVIZACAO, DEST_ANGULO_MAX, DEST_ANGULO_MIN, DEST_BASE_CONFIRMAR,
    DEST_BASE_CURVA, DEST_BASE_FRENTE, DEST_BASE_RETORNO, DEST_CONFIRMAR_TROCA_TIPO_FRAMES,
    DEST_CONTINUIDADE_MIN_CURVA, DEST_CONTINUIDADE_MIN_FRENTE, DEST_CONTINUIDADE_MIN_RETORNO,
    DEST_CORRECAO_MAX_CONFIRMAR, DEST_CORRECAO_MAX_CURVA, DEST_CORRECAO_MAX_FRENTE,
    DEST_CORRECAO_MAX_RETORNO, DEST_CURVA_ANGULO_MAX, DEST_DELTA_MAX, DEST_DELTA_MAX_RETORNO,
    DEST_DIST_MIN_CURVA, DEST_DIST_MIN_FRENTE, DEST_DIST_MIN_RETORNO, DEST_FRAMES_DESTINO_CONFIAVEL,
    DEST_FRAMES_DESTINO_PERDIDO, DEST_FRENTE_ANGULO_MAX, DEST_INTERVALO, DEST_INTERVALO_DEBUG,
    DEST_K_ANGULO_CURVA, DEST_K_ANGULO_FRENTE, DEST_K_ANGULO_RETORNO, DEST_K_X_CURVA,
    DEST_K_X_FRENTE, DEST_K_X_RETORNO, DEST_MARGEM_TROCA_TIPO, DEST_PASSO_ANGULO,
    DEST_PESO_CONTINUIDADE, DEST_PESO_DISTANCIA, DEST_PESO_FRENTE, DEST_PESO_PIXELS,
    DEST_RAIO_DIST_MAX, DEST_RAIO_DIST_MIN, DEST_RAIO_JANELA, DEST_RAIO_MIN_PIXELS_HIT,
    DEST_RAIO_PASSO_DIST, DEST_VEL_RECUPERAR,
    DEST_RETORNO_ANGULO_MIN, DEST_ROBO_Y_REL, DEST_SALVAR_DEBUG_EVENTOS,
    DEST_SCORE_MIN_CURVA, DEST_SCORE_MIN_FRENTE, DEST_SCORE_MIN_RETORNO,
    DEST_SEGMENTO_GAP_MAX_MULT, DEST_SEGMENTO_MIN_HITS_CURVA, DEST_SEGMENTO_MIN_HITS_FRENTE,
    DEST_SEGMENTO_MIN_HITS_RETORNO, DEST_TEMPO_TROCA_VARREDURA, DEST_VEL_MAX,
    DEST_VEL_MIN_CURVA, DEST_VEL_MIN_FRENTE, DEST_VEL_MIN_RETORNO,
)
from line_test import criar_debug_linha, detectar_linha
from utils import abrir_serial, enviar_comando


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha por hierarquia de destinos visuais.")
    parser.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    parser.add_argument("--motores", action="store_true", help="Envia comandos ao Arduino.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou auto.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva imagens de debug.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra janela OpenCV.")
    return parser.parse_args()


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def enviar_seguro(conexao, comando, motores_ativos):
    if not motores_ativos:
        return None
    if conexao is None or not conexao.is_open:
        raise RuntimeError("Conexao serial indisponivel.")
    return enviar_comando(conexao, comando)


def enviar_parar_final(conexao, motores_ativos):
    try:
        enviar_seguro(conexao, "PARAR", motores_ativos)
    except Exception as erro:
        print(f"Aviso: nao foi possivel enviar PARAR final: {erro}")


def fechar_camera_segura(camera):
    if camera is not None:
        try:
            camera.stop()
        except Exception as erro:
            print(f"Aviso ao fechar camera: {erro}")


def fechar_serial_segura(conexao):
    if conexao is not None and conexao.is_open:
        try:
            conexao.close()
        except Exception as erro:
            print(f"Aviso ao fechar serial: {erro}")


def tipo_por_angulo(angulo):
    angulo_absoluto = abs(angulo)
    if angulo_absoluto <= DEST_FRENTE_ANGULO_MAX:
        return "FRENTE"
    if angulo_absoluto <= DEST_CURVA_ANGULO_MAX:
        return "CURVA"
    if angulo_absoluto >= DEST_RETORNO_ANGULO_MIN:
        return "RETORNO"
    return "CURVA"


def parametros_tipo(tipo):
    if tipo == "FRENTE":
        return DEST_SCORE_MIN_FRENTE, DEST_CONTINUIDADE_MIN_FRENTE, DEST_DIST_MIN_FRENTE, DEST_SEGMENTO_MIN_HITS_FRENTE
    if tipo == "CURVA":
        return DEST_SCORE_MIN_CURVA, DEST_CONTINUIDADE_MIN_CURVA, DEST_DIST_MIN_CURVA, DEST_SEGMENTO_MIN_HITS_CURVA
    return DEST_SCORE_MIN_RETORNO, DEST_CONTINUIDADE_MIN_RETORNO, DEST_DIST_MIN_RETORNO, DEST_SEGMENTO_MIN_HITS_RETORNO


def calcular_score_raio(tipo, pixels_total, continuidade, distancia_util, angulo):
    score = (
        DEST_PESO_PIXELS * pixels_total
        + DEST_PESO_CONTINUIDADE * continuidade
        + DEST_PESO_DISTANCIA * distancia_util
    )
    if tipo == "FRENTE":
        score += DEST_PESO_FRENTE * (1 - abs(angulo) / max(DEST_FRENTE_ANGULO_MAX, 1))
    if tipo == "RETORNO":
        score *= 0.85
    return score


def escolher_melhor_segmento(hits):
    """Separa hits por lacuna e retorna o segmento visual mais confiavel."""
    if not hits:
        return None
    gap_maximo = DEST_RAIO_PASSO_DIST * DEST_SEGMENTO_GAP_MAX_MULT
    segmentos, atual = [], [hits[0]]
    for hit in hits[1:]:
        if hit["distancia"] - atual[-1]["distancia"] <= gap_maximo:
            atual.append(hit)
        else:
            segmentos.append(atual)
            atual = [hit]
    segmentos.append(atual)

    def chave(segmento):
        suporte = sum(hit["pixels"] for hit in segmento)
        comprimento = max(segmento[-1]["distancia"] - segmento[0]["distancia"], DEST_RAIO_PASSO_DIST)
        continuidade = len(segmento) / max(round(comprimento / DEST_RAIO_PASSO_DIST) + 1, 1)
        return len(segmento), suporte, segmento[-1]["distancia"], continuidade

    return max(segmentos, key=chave)


def amostrar_raio(mascara, origem, angulo_graus):
    """Amostra um raio e retorna somente seu melhor segmento continuo."""
    altura, largura = mascara.shape[:2]
    origem_x, origem_y = origem
    angulo = math.radians(angulo_graus)
    hits, num_amostras, ultimo_ponto = [], 0, origem
    for distancia in range(DEST_RAIO_DIST_MIN, DEST_RAIO_DIST_MAX + 1, DEST_RAIO_PASSO_DIST):
        x = round(origem_x + math.sin(angulo) * distancia)
        y = round(origem_y - math.cos(angulo) * distancia)
        if x < 0 or x >= largura or y < 0 or y >= altura:
            continue
        ultimo_ponto = (x, y)
        num_amostras += 1
        x1, x2 = max(0, x - DEST_RAIO_JANELA), min(largura, x + DEST_RAIO_JANELA + 1)
        y1, y2 = max(0, y - DEST_RAIO_JANELA), min(altura, y + DEST_RAIO_JANELA + 1)
        pixels = int(cv2.countNonZero(mascara[y1:y2, x1:x2]))
        if pixels >= DEST_RAIO_MIN_PIXELS_HIT:
            hits.append({"distancia": distancia, "x": x, "y": y, "pixels": pixels})

    segmento = escolher_melhor_segmento(hits)
    if segmento is None:
        return {"ok": False, "angulo": angulo_graus, "origem_local": origem, "ponto_raio": ultimo_ponto, "hits": []}

    quantidade_finais = max(1, math.ceil(len(segmento) * 0.25))
    hits_finais = segmento[-quantidade_finais:]
    destino_local = (
        round(sum(hit["x"] for hit in hits_finais) / len(hits_finais)),
        round(sum(hit["y"] for hit in hits_finais) / len(hits_finais)),
    )
    comprimento = max(segmento[-1]["distancia"] - segmento[0]["distancia"], DEST_RAIO_PASSO_DIST)
    continuidade = len(segmento) / max(round(comprimento / DEST_RAIO_PASSO_DIST) + 1, 1)
    return {
        "ok": True,
        "angulo": angulo_graus,
        "origem_local": origem,
        "ponto_raio": ultimo_ponto,
        "pixels": sum(hit["pixels"] for hit in segmento),
        "continuidade": continuidade,
        "distancia": segmento[-1]["distancia"],
        "segmento_hits": len(segmento),
        "destino_local": destino_local,
        "hits": segmento,
    }


def candidato_confiavel(candidato):
    score_min, continuidade_min, distancia_min, hits_min = parametros_tipo(candidato["tipo"])
    return (
        candidato["score"] >= score_min
        and candidato["continuidade"] >= continuidade_min
        and candidato["distancia"] >= distancia_min
        and candidato["segmento_hits"] >= hits_min
    )


def escolher_destino(resultado):
    """Escolhe FRENTE, CURVA ou RETORNO apenas entre candidatos confiaveis."""
    mascara = resultado["mascara_limpa"]
    altura_roi = mascara.shape[0]
    robo_x = resultado["centro_imagem_x"] - resultado["x_inicio_roi"]
    robo_y = int(altura_roi * DEST_ROBO_Y_REL)
    origem = (robo_x, robo_y)
    raios, validos = [], {"FRENTE": [], "CURVA": [], "RETORNO": []}

    for angulo in range(DEST_ANGULO_MIN, DEST_ANGULO_MAX + 1, DEST_PASSO_ANGULO):
        raio = amostrar_raio(mascara, origem, angulo)
        raios.append(raio)
        if not raio["ok"]:
            continue
        tipo = tipo_por_angulo(angulo)
        x_local, y_local = raio["destino_local"]
        candidato = {
            "ok": True,
            "tipo": tipo,
            "angulo": angulo,
            "pixels": raio["pixels"],
            "continuidade": raio["continuidade"],
            "distancia": raio["distancia"],
            "segmento_hits": raio["segmento_hits"],
            "destino_local": raio["destino_local"],
            "destino_global": (resultado["x_inicio_roi"] + x_local, resultado["y_inicio_roi"] + y_local),
            "lado": "ESQUERDA" if angulo < -15 else "DIREITA" if angulo > 15 else "CENTRO",
            "origem_local": origem,
            "raio": raio,
        }
        candidato["score"] = calcular_score_raio(tipo, candidato["pixels"], candidato["continuidade"], candidato["distancia"], angulo)
        if candidato_confiavel(candidato):
            validos[tipo].append(candidato)

    for tipo, motivo in (("FRENTE", "FRENTE_PRIORITARIA"), ("CURVA", "CURVA_SEM_FRENTE"), ("RETORNO", "RETORNO_SEM_FRENTE_CURVA")):
        if validos[tipo]:
            escolhido = max(validos[tipo], key=lambda item: item["score"])
            escolhido["motivo"] = motivo
            escolhido["raios"] = raios
            escolhido["validos_por_tipo"] = validos
            vetor_x = escolhido["destino_local"][0] - robo_x
            vetor_y = robo_y - escolhido["destino_local"][1]
            escolhido["erro_x"] = vetor_x
            escolhido["angulo_destino"] = math.degrees(math.atan2(vetor_x, max(vetor_y, 1)))
            return escolhido

    return {
        "ok": False, "tipo": "PERDIDO", "motivo": "RECUPERAR", "raios": raios,
        "validos_por_tipo": validos, "origem_local": origem,
    }


def controlar_destino(destino, memoria, confirmar=False):
    """Transforma o vetor do destino em LADO, com confirmacao sempre positiva."""
    tipo = destino["tipo"]
    if confirmar:
        base, vel_min, correcao_max = DEST_BASE_CONFIRMAR, 0, DEST_CORRECAO_MAX_CONFIRMAR
        k_ang, k_x = DEST_K_ANGULO_CURVA, DEST_K_X_CURVA
    elif tipo == "FRENTE":
        base, vel_min, correcao_max = DEST_BASE_FRENTE, DEST_VEL_MIN_FRENTE, DEST_CORRECAO_MAX_FRENTE
        k_ang, k_x = DEST_K_ANGULO_FRENTE, DEST_K_X_FRENTE
    elif tipo == "CURVA":
        base, vel_min, correcao_max = DEST_BASE_CURVA, DEST_VEL_MIN_CURVA, DEST_CORRECAO_MAX_CURVA
        k_ang, k_x = DEST_K_ANGULO_CURVA, DEST_K_X_CURVA
    elif tipo == "RETORNO":
        base, vel_min, correcao_max = DEST_BASE_RETORNO, DEST_VEL_MIN_RETORNO, DEST_CORRECAO_MAX_RETORNO
        k_ang, k_x = DEST_K_ANGULO_RETORNO, DEST_K_X_RETORNO
    else:
        raise ValueError(f"Tipo de destino invalido: {tipo}")

    correcao = limitar(k_ang * destino["angulo_destino"] + k_x * destino["erro_x"], -correcao_max, correcao_max)
    correcao_suave = DEST_ALPHA_SUAVIZACAO * memoria["correcao_anterior"] + (1 - DEST_ALPHA_SUAVIZACAO) * correcao
    correcao_suave = limitar(correcao_suave, -correcao_max, correcao_max)
    vel_esq = limitar(base + correcao_suave, vel_min, DEST_VEL_MAX)
    vel_dir = limitar(base - correcao_suave, vel_min, DEST_VEL_MAX)
    delta_max = DEST_DELTA_MAX_RETORNO if tipo == "RETORNO" and not confirmar else DEST_DELTA_MAX
    vel_esq_ant, vel_dir_ant = memoria["vel_anterior"]
    vel_esq = limitar(vel_esq, vel_esq_ant - delta_max, vel_esq_ant + delta_max)
    vel_dir = limitar(vel_dir, vel_dir_ant - delta_max, vel_dir_ant + delta_max)
    vel_esq = limitar(vel_esq, vel_min, DEST_VEL_MAX)
    vel_dir = limitar(vel_dir, vel_min, DEST_VEL_MAX)
    if confirmar or tipo in ("FRENTE", "CURVA"):
        vel_esq, vel_dir = max(0, vel_esq), max(0, vel_dir)
    memoria["correcao_anterior"] = correcao_suave
    memoria["vel_anterior"] = (vel_esq, vel_dir)
    return f"LADO {round(vel_esq)} {round(vel_dir)}", vel_esq, vel_dir, correcao_suave


def lado_destino(destino):
    """Retorna o lado do vetor final usado pelo controle do destino."""
    angulo = destino.get("angulo_destino", destino.get("angulo", 0))
    if angulo < -15:
        return "ESQUERDA"
    if angulo > 15:
        return "DIREITA"
    return "CENTRO"


def resetar_confirmacao_lado(memoria):
    memoria["lado_pendente"] = "CENTRO"
    memoria["frames_confirmacao_lado"] = 0


def avaliar_retorno_lado_oposto(destino, memoria):
    """Confirma RETORNO oposto antes de substituir o lado de recuperacao."""
    if not destino.get("ok", False) or destino.get("tipo") != "RETORNO":
        resetar_confirmacao_lado(memoria)
        return "NAO_APLICA"

    lado_atual = lado_destino(destino)
    lado_anterior = memoria.get("ultimo_lado", "CENTRO")
    if lado_atual == "CENTRO":
        resetar_confirmacao_lado(memoria)
        return "NAO_APLICA"
    if lado_anterior == "CENTRO" or lado_atual == lado_anterior:
        resetar_confirmacao_lado(memoria)
        return "ACEITAR_RETORNO"

    if memoria.get("lado_pendente") == lado_atual:
        memoria["frames_confirmacao_lado"] += 1
    else:
        memoria["lado_pendente"] = lado_atual
        memoria["frames_confirmacao_lado"] = 1
    if memoria["frames_confirmacao_lado"] >= 2:
        return "ACEITAR_RETORNO"
    return "CONFIRMAR_LADO"


def atualizar_tipo_confiavel(destino, memoria):
    if destino["tipo"] == memoria["ultimo_tipo_confiavel"]:
        memoria["frames_mesmo_tipo"] += 1
    else:
        memoria["ultimo_tipo_confiavel"] = destino["tipo"]
        memoria["frames_mesmo_tipo"] = 1
    memoria["ultimo_score_confiavel"] = destino["score"]
    lado = lado_destino(destino)
    if lado in ("ESQUERDA", "DIREITA"):
        memoria["ultimo_lado"] = lado
    memoria["tipo_pendente"] = "PERDIDO"
    memoria["frames_confirmacao_tipo"] = 0
    resetar_confirmacao_lado(memoria)


def decidir_tipo(destino, memoria):
    """Retorna se o destino sera seguido ou confirmado antes de trocar o tipo."""
    anterior = memoria["ultimo_tipo_confiavel"]
    if anterior == "PERDIDO" or destino["tipo"] == anterior:
        atualizar_tipo_confiavel(destino, memoria)
        return False

    anterior_ainda_confiavel = bool(destino["validos_por_tipo"].get(anterior))
    troca_forte = destino["score"] >= memoria["ultimo_score_confiavel"] * DEST_MARGEM_TROCA_TIPO
    if not anterior_ainda_confiavel or troca_forte:
        atualizar_tipo_confiavel(destino, memoria)
        return False

    if memoria["tipo_pendente"] == destino["tipo"]:
        memoria["frames_confirmacao_tipo"] += 1
    else:
        memoria["tipo_pendente"] = destino["tipo"]
        memoria["frames_confirmacao_tipo"] = 1
    if memoria["frames_confirmacao_tipo"] > DEST_CONFIRMAR_TROCA_TIPO_FRAMES:
        atualizar_tipo_confiavel(destino, memoria)
        return False
    return True


def decidir_confirmacao_destino(destino, memoria):
    """Centraliza a confirmacao lateral de RETORNO e a histerese normal de tipo."""
    if not destino.get("ok", False):
        resetar_confirmacao_lado(memoria)
        return True

    estado_retorno = avaliar_retorno_lado_oposto(destino, memoria)
    if estado_retorno == "CONFIRMAR_LADO":
        return True
    if estado_retorno == "ACEITAR_RETORNO":
        atualizar_tipo_confiavel(destino, memoria)
        return False
    return decidir_tipo(destino, memoria)


def comando_recuperacao(ultimo_lado, controle_varredura):
    if ultimo_lado == "ESQUERDA":
        return f"GIRAR_ESQ {DEST_VEL_RECUPERAR}"
    if ultimo_lado == "DIREITA":
        return f"GIRAR_DIR {DEST_VEL_RECUPERAR}"
    agora = time.monotonic()
    if agora - controle_varredura["ultima_troca"] >= DEST_TEMPO_TROCA_VARREDURA:
        controle_varredura["etapa"] = (controle_varredura["etapa"] + 1) % 4
        controle_varredura["ultima_troca"] = agora
    sequencia = ["ESQUERDA", "DIREITA", "DIREITA", "ESQUERDA"]
    giro = "GIRAR_ESQ" if sequencia[controle_varredura["etapa"]] == "ESQUERDA" else "GIRAR_DIR"
    return f"{giro} {DEST_VEL_RECUPERAR}"


def comando_confirmar_sem_destino(memoria):
    vel_esq, vel_dir = memoria["vel_anterior"]
    vel_esq, vel_dir = max(0, vel_esq), max(0, vel_dir)
    memoria["vel_anterior"] = (vel_esq, vel_dir)
    memoria["correcao_anterior"] = 0
    return f"LADO {round(vel_esq)} {round(vel_dir)}"


def criar_debug_destinos(resultado, destino, estado, comando, memoria):
    debug = criar_debug_linha(resultado)
    x_inicio, y_inicio = resultado["x_inicio_roi"], resultado["y_inicio_roi"]
    origem = destino["origem_local"]
    origem_global = (x_inicio + origem[0], y_inicio + origem[1])
    cv2.circle(debug, origem_global, 7, (255, 0, 255), -1)
    for raio in destino["raios"]:
        ponto = raio.get("ponto_raio", origem)
        cv2.line(debug, origem_global, (x_inicio + ponto[0], y_inicio + ponto[1]), (90, 90, 90), 1)
    for candidatos in destino["validos_por_tipo"].values():
        for candidato in candidatos:
            ponto = candidato["destino_global"]
            cv2.line(debug, origem_global, ponto, (255, 0, 0), 1)
    if destino["ok"]:
        cv2.line(debug, origem_global, destino["destino_global"], (0, 255, 255), 3)
        cv2.circle(debug, destino["destino_global"], 8, (0, 0, 255), -1)

    quantidades = {tipo: len(destino["validos_por_tipo"][tipo]) for tipo in ("FRENTE", "CURVA", "RETORNO")}
    linhas = [
        f"Estado: {estado}", f"Tipo: {destino['tipo']}", f"Cmd: {comando}",
        f"Motivo: {destino['motivo']}", f"Score: {destino.get('score', 0):.1f}",
        f"Continuidade: {destino.get('continuidade', 0):.2f}", f"Distancia: {destino.get('distancia', 0):.0f}",
        f"Segmento hits: {destino.get('segmento_hits', 0)}", f"Validos F/C/R: {quantidades['FRENTE']}/{quantidades['CURVA']}/{quantidades['RETORNO']}",
        f"Ultimo lado: {memoria['ultimo_lado']}", f"Perdido: {memoria['frames_destino_perdido']}",
        f"Confiavel: {memoria['frames_destino_confiavel']}",
    ]
    for indice, texto in enumerate(linhas):
        y = 35 + indice * 22
        cv2.putText(debug, texto, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(debug, texto, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return debug


def salvar_debug(debug, estado):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_follow_destinos_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{estado}.jpg"
    if not cv2.imwrite(str(caminho), debug):
        raise RuntimeError(f"Nao foi possivel salvar debug: {caminho}")
    return caminho


def imprimir_log(estado, destino, comando, memoria):
    if destino["ok"]:
        print(
            f"Estado: {estado} | Tipo: {destino['tipo']} | Cmd: {comando} | "
            f"ang: {destino['angulo_destino']:.1f} | score: {destino['score']:.1f} | "
            f"cont: {destino['continuidade']:.2f} | dist: {destino['distancia']:.0f} | "
            f"segmento: {destino['segmento_hits']} | motivo: {destino['motivo']} | "
            f"lado_dest: {lado_destino(destino)} | ultimo: {memoria['ultimo_lado']} | "
            f"ultimo_tipo: {memoria['ultimo_tipo_confiavel']} | pend_tipo: {memoria['tipo_pendente']} | "
            f"pend_lado: {memoria.get('lado_pendente', 'CENTRO')} | "
            f"frames_lado: {memoria.get('frames_confirmacao_lado', 0)}"
        )
    else:
        print(f"Estado: {estado} | Tipo: PERDIDO | Cmd: {comando} | ultimo: {memoria['ultimo_lado']}")


def main():
    args = ler_argumentos()
    camera = None
    conexao = None
    try:
        if not args.camera:
            print("Use --camera.")
            return 1
        if args.motores:
            porta = SERIAL_PORT if args.porta == "auto" else args.porta
            conexao = abrir_serial(porta, BAUD_RATE, TIMEOUT_SERIAL)
            time.sleep(1.8)
            if hasattr(conexao, "reset_input_buffer"):
                conexao.reset_input_buffer()
                conexao.reset_output_buffer()
            print("Motores ativados. Iniciando segue-linha por destinos.")
        else:
            print("Simulacao sem motores.")

        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
        memoria = {
            "correcao_anterior": 0, "vel_anterior": (0, 0), "ultimo_tipo": "PERDIDO",
            "ultimo_tipo_confiavel": "PERDIDO", "ultimo_score_confiavel": 0,
            "frames_mesmo_tipo": 0, "tipo_pendente": "PERDIDO", "frames_confirmacao_tipo": 0,
            "ultimo_lado": "CENTRO", "frames_destino_perdido": 0,
            "frames_destino_confiavel": 0, "em_recuperacao": False,
            "lado_pendente": "CENTRO", "frames_confirmacao_lado": 0,
        }
        controle_varredura = {"etapa": 0, "ultima_troca": time.monotonic()}
        estado_anterior, ultimo_debug = None, 0

        while True:
            resultado = detectar_linha(capturar_frame_bgr(camera))
            destino = escolher_destino(resultado)
            if not destino["ok"]:
                resetar_confirmacao_lado(memoria)
            if resultado["encontrou_linha"] and destino["ok"]:
                memoria["frames_destino_confiavel"] += 1
                memoria["frames_destino_perdido"] = 0
            else:
                memoria["frames_destino_perdido"] += 1
                memoria["frames_destino_confiavel"] = 0

            deve_recuperar = not resultado["encontrou_linha"] or memoria["frames_destino_perdido"] >= DEST_FRAMES_DESTINO_PERDIDO
            if memoria["em_recuperacao"] and destino["ok"]:
                if memoria["frames_destino_confiavel"] < DEST_FRAMES_DESTINO_CONFIAVEL:
                    estado = "FOLLOW_CONFIRMAR"
                    comando, _, _, _ = controlar_destino(destino, memoria, confirmar=True)
                else:
                    memoria["em_recuperacao"] = False
                    confirmar = decidir_confirmacao_destino(destino, memoria)
                    estado = "FOLLOW_CONFIRMAR" if confirmar else "FOLLOW_DESTINO"
                    comando, _, _, _ = controlar_destino(destino, memoria, confirmar=confirmar)
            elif memoria["em_recuperacao"] or deve_recuperar:
                memoria["em_recuperacao"] = True
                estado = "RECUPERAR_GIRO" if memoria["ultimo_lado"] in ("ESQUERDA", "DIREITA") else "RECUPERAR_VARREDURA"
                comando = comando_recuperacao(memoria["ultimo_lado"], controle_varredura)
                memoria["correcao_anterior"] = 0
                memoria["vel_anterior"] = (0, 0)
            elif destino["ok"]:
                confirmar = decidir_confirmacao_destino(destino, memoria)
                estado = "FOLLOW_CONFIRMAR" if confirmar else "FOLLOW_DESTINO"
                comando, _, _, _ = controlar_destino(destino, memoria, confirmar=confirmar)
            else:
                estado = "FOLLOW_CONFIRMAR"
                comando = comando_confirmar_sem_destino(memoria)

            enviar_seguro(conexao, comando, args.motores)
            imprimir_log(estado, destino, comando, memoria)
            if args.salvar_debug:
                agora = time.monotonic()
                if DEST_SALVAR_DEBUG_EVENTOS and (estado != estado_anterior or agora - ultimo_debug >= DEST_INTERVALO_DEBUG):
                    salvar_debug(criar_debug_destinos(resultado, destino, estado, comando, memoria), estado)
                    ultimo_debug = agora
            estado_anterior = estado
            if args.mostrar:
                cv2.imshow("follow_destinos", criar_debug_destinos(resultado, destino, estado, comando, memoria))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            time.sleep(DEST_INTERVALO)
    except KeyboardInterrupt:
        print("CTRL+C recebido. Parando robo.")
        return 130
    except Exception as erro:
        print(f"Erro grave: {erro}")
        return 1
    finally:
        enviar_parar_final(conexao, args.motores)
        fechar_camera_segura(camera)
        fechar_serial_segura(conexao)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Recursos liberados.")


if __name__ == "__main__":
    raise SystemExit(main())
