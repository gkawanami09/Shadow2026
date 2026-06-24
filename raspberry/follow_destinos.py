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
    DEST_FRENTE_ANGULO_MAX, DEST_INTERVALO, DEST_INTERVALO_DEBUG,
    DEST_K_ANGULO_CURVA, DEST_K_ANGULO_FRENTE, DEST_K_ANGULO_RETORNO, DEST_K_X_CURVA,
    DEST_K_X_FRENTE, DEST_K_X_RETORNO, DEST_MARGEM_TROCA_TIPO, DEST_PASSO_ANGULO,
    DEST_PESO_CONTINUIDADE, DEST_PESO_DISTANCIA, DEST_PESO_FRENTE, DEST_PESO_PIXELS,
    DEST_RAIO_DIST_MAX, DEST_RAIO_DIST_MIN, DEST_RAIO_JANELA, DEST_RAIO_MIN_PIXELS_HIT,
    DEST_RAIO_PASSO_DIST, DEST_VEL_RECUPERAR,
    DEST_LADO_RECUPERACAO_CONFIRMAR_FRAMES, DEST_LADO_RECUPERACAO_CURVA_ANGULO_MIN,
    DEST_RETORNO_ANGULO_MIN, DEST_ROBO_Y_REL, DEST_SALVAR_DEBUG_EVENTOS,
    DEST_SCORE_MIN_CURVA, DEST_SCORE_MIN_FRENTE, DEST_SCORE_MIN_RETORNO,
    DEST_SEGMENTO_GAP_MAX_MULT, DEST_SEGMENTO_MIN_HITS_CURVA, DEST_SEGMENTO_MIN_HITS_FRENTE,
    DEST_SEGMENTO_MIN_HITS_RETORNO, DEST_TEMPO_TROCA_VARREDURA, DEST_VEL_MAX,
    DEST_VEL_MIN_CURVA, DEST_VEL_MIN_FRENTE, DEST_VEL_MIN_RETORNO,
    GREEN_APPROACH_BEFORE_TURN_SEC, GREEN_APPROACH_SPEED_LEFT, GREEN_APPROACH_SPEED_RIGHT,
    GREEN_FOLLOW_CLEAR_FRAMES_TO_RELEASE, GREEN_FOLLOW_CONFIRM_FRAMES,
    GREEN_FOLLOW_CONFIRM_MAX_MISSES, GREEN_FOLLOW_COOLDOWN_SEC, GREEN_LOG_INTERVAL_SEC,
    GREEN_REACQUIRE_CONFIRM_FRAMES, GREEN_REACQUIRE_MAX_SEC, GREEN_REACQUIRE_TURN_SPEED,
    GREEN_RETURN_DIRECTION, GREEN_RETURN_MAX_SEC, GREEN_RETURN_MIN_SEC, GREEN_RETURN_SPEED,
    GREEN_TURN_MAX_SEC, GREEN_TURN_MIN_SEC, GREEN_TURN_SPEED,
)
from green_action import analisar_intersecao_preta, decidir_verde_acionavel
from green_detector import criar_mascara_linha_global, detectar_verde
from line_test import criar_debug_linha, detectar_linha
from utils import abrir_serial, enviar_comando


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha por hierarquia de destinos visuais.")
    parser.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    parser.add_argument("--motores", action="store_true", help="Envia comandos ao Arduino.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou auto.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva imagens de debug.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra janela OpenCV.")
    parser.add_argument("--green-log", action="store_true", help="Loga verde/intersecao sem alterar movimento.")
    parser.add_argument("--green-move", action="store_true", help="Permite movimento real por verde.")
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


def calcular_acao_verde(frame_bgr, resultado_linha):
    """Calcula percepcao verde no frame atual, sem interferir no follow."""
    try:
        mascara_linha = criar_mascara_linha_global(resultado_linha)
        resultado_verde = detectar_verde(
            frame_bgr,
            x_referencia=resultado_linha["centro_imagem_x"],
            mascara_linha_global=mascara_linha,
        )
        analise = analisar_intersecao_preta(mascara_linha, resultado_linha["centro_imagem_x"])
        return resultado_verde, analise, decidir_verde_acionavel(resultado_verde, analise)
    except Exception as erro:
        print(f"[GREEN] erro_visual={erro}")
        analise = {
            "tipo_intersecao": "NENHUMA", "black_center": 0, "black_left": 0,
            "black_right": 0, "black_ratio_center": 0.0, "black_ratio_left": 0.0,
            "black_ratio_right": 0.0,
        }
        verde = {"tipo_detectado": "NENHUM", "tipo_confirmado": "NENHUM", "contornos_confirmados": []}
        acao = {
            "verde_acionavel": "NENHUM", "acao_visual": "SEGUIR_RETO",
            "motivo_acao": "erro_visual", "tipo_intersecao": "NENHUMA",
            "intersecao_detectada": False, "analise_intersecao": analise,
        }
        return verde, analise, acao


def criar_estado_verde(agora):
    return {
        "estado": "GREEN_IDLE", "acao_pendente": None, "acao_confirmada": None,
        "frames_confirmados": 0, "misses": 0, "inicio_estado": agora,
        "inicio_cooldown": 0.0, "clear_frames": 0, "reacquire_frames": 0,
        "ultimo_log": 0.0, "ultimo_estado_logado": None, "evento": "",
        "turn_saw_lost_line": False,
    }


def iniciar_estado_verde(estado_verde, novo_estado, agora):
    estado_verde["estado"] = novo_estado
    estado_verde["inicio_estado"] = agora
    if novo_estado == "GREEN_TURNING":
        estado_verde["turn_saw_lost_line"] = False
    if novo_estado == "GREEN_COOLDOWN":
        estado_verde["inicio_cooldown"] = agora
        estado_verde["clear_frames"] = 0


def acao_verde_eh_manobra(acao_visual):
    return acao_visual in ("PREPARAR_ESQUERDA", "PREPARAR_DIREITA", "PREPARAR_RETORNO")


def _direcao_acao_verde(acao):
    if acao == "PREPARAR_ESQUERDA":
        return "ESQUERDA"
    if acao == "PREPARAR_DIREITA":
        return "DIREITA"
    return GREEN_RETURN_DIRECTION


def _resetar_suavizacao_verde(memoria):
    memoria["correcao_anterior"] = 0
    memoria["vel_anterior"] = (0, 0)


def atualizar_estado_verde(estado_verde, resultado_acao, destino, estado_follow, green_move, agora, memoria):
    """Atualiza a maquina verde e nunca envia comandos ao Arduino."""
    estado = estado_verde["estado"]
    acao_visual = resultado_acao["acao_visual"]
    elegivel = (
        acao_verde_eh_manobra(acao_visual)
        and destino.get("ok", False)
        and estado_follow not in ("RECUPERAR_GIRO", "RECUPERAR_VARREDURA")
    )

    if estado == "GREEN_IDLE":
        if elegivel:
            estado_verde["evento"] = ""
            estado_verde["acao_pendente"] = acao_visual
            estado_verde["frames_confirmados"] = 1
            estado_verde["misses"] = 0
            iniciar_estado_verde(estado_verde, "GREEN_CONFIRMING", agora)
    elif estado == "GREEN_CONFIRMING":
        if elegivel and acao_visual == estado_verde["acao_pendente"]:
            estado_verde["frames_confirmados"] += 1
        else:
            estado_verde["misses"] += 1
        if estado_verde["misses"] > GREEN_FOLLOW_CONFIRM_MAX_MISSES:
            estado_verde["acao_pendente"] = None
            iniciar_estado_verde(estado_verde, "GREEN_IDLE", agora)
        elif estado_verde["frames_confirmados"] >= GREEN_FOLLOW_CONFIRM_FRAMES:
            if green_move:
                estado_verde["acao_confirmada"] = estado_verde["acao_pendente"]
                proximo = "GREEN_APPROACH" if GREEN_APPROACH_BEFORE_TURN_SEC > 0 else "GREEN_TURNING"
                iniciar_estado_verde(estado_verde, proximo, agora)
            else:
                iniciar_estado_verde(estado_verde, "GREEN_COOLDOWN", agora)
    elif estado == "GREEN_APPROACH":
        if agora - estado_verde["inicio_estado"] >= GREEN_APPROACH_BEFORE_TURN_SEC:
            iniciar_estado_verde(estado_verde, "GREEN_TURNING", agora)
    elif estado == "GREEN_TURNING":
        retorno = estado_verde["acao_confirmada"] == "PREPARAR_RETORNO"
        minimo = GREEN_RETURN_MIN_SEC if retorno else GREEN_TURN_MIN_SEC
        maximo = GREEN_RETURN_MAX_SEC if retorno else GREEN_TURN_MAX_SEC
        decorrido = agora - estado_verde["inicio_estado"]
        if not destino.get("ok", False):
            estado_verde["turn_saw_lost_line"] = True
        if decorrido >= maximo:
            estado_verde["evento"] = "green_turn_timeout"
            iniciar_estado_verde(estado_verde, "GREEN_REACQUIRE", agora)
            estado_verde["reacquire_frames"] = 0
        elif (
            decorrido >= minimo
            and estado_verde["turn_saw_lost_line"]
            and destino.get("ok", False)
        ):
            iniciar_estado_verde(estado_verde, "GREEN_REACQUIRE", agora)
            estado_verde["reacquire_frames"] = 0
    elif estado == "GREEN_REACQUIRE":
        if destino.get("ok", False):
            estado_verde["reacquire_frames"] += 1
        else:
            estado_verde["reacquire_frames"] = 0
        if estado_verde["reacquire_frames"] >= GREEN_REACQUIRE_CONFIRM_FRAMES:
            _resetar_suavizacao_verde(memoria)
            iniciar_estado_verde(estado_verde, "GREEN_COOLDOWN", agora)
        elif agora - estado_verde["inicio_estado"] >= GREEN_REACQUIRE_MAX_SEC:
            estado_verde["evento"] = "green_reacquire_timeout"
            _resetar_suavizacao_verde(memoria)
            iniciar_estado_verde(estado_verde, "GREEN_COOLDOWN", agora)
    elif estado == "GREEN_COOLDOWN":
        if resultado_acao["verde_acionavel"] == "NENHUM":
            estado_verde["clear_frames"] += 1
        else:
            estado_verde["clear_frames"] = 0
        if (
            agora - estado_verde["inicio_cooldown"] >= GREEN_FOLLOW_COOLDOWN_SEC
            and estado_verde["clear_frames"] >= GREEN_FOLLOW_CLEAR_FRAMES_TO_RELEASE
        ):
            estado_verde["acao_pendente"] = None
            estado_verde["acao_confirmada"] = None
            iniciar_estado_verde(estado_verde, "GREEN_IDLE", agora)
    return estado_verde


def comando_override_verde(estado_verde, destino, green_move):
    """Retorna o comando verde ou None para preservar o comando normal."""
    if not green_move:
        return None
    estado = estado_verde["estado"]
    acao = estado_verde["acao_confirmada"]
    direcao = _direcao_acao_verde(acao)
    if estado == "GREEN_APPROACH":
        return f"LADO {GREEN_APPROACH_SPEED_LEFT} {GREEN_APPROACH_SPEED_RIGHT}"
    if estado == "GREEN_TURNING":
        velocidade = GREEN_RETURN_SPEED if acao == "PREPARAR_RETORNO" else GREEN_TURN_SPEED
        giro = "GIRAR_ESQ" if direcao == "ESQUERDA" else "GIRAR_DIR"
        return f"{giro} {velocidade}"
    if estado == "GREEN_REACQUIRE" and not destino.get("ok", False):
        giro = "GIRAR_ESQ" if direcao == "ESQUERDA" else "GIRAR_DIR"
        return f"{giro} {GREEN_REACQUIRE_TURN_SPEED}"
    return None


def log_verde(resultado_verde, analise, resultado_acao, estado_verde, green_move, comando_final, agora):
    mudou = estado_verde["ultimo_estado_logado"] != estado_verde["estado"]
    if not mudou and agora - estado_verde["ultimo_log"] < GREEN_LOG_INTERVAL_SEC:
        return
    depois = any(item.get("possivel_verde_depois_intersecao", False) for item in resultado_verde.get("contornos_confirmados", []))
    print(
        f"[GREEN] det={resultado_verde['tipo_detectado']} conf={resultado_verde['tipo_confirmado']} "
        f"acionavel={resultado_acao['verde_acionavel']} acao={resultado_acao['acao_visual']} "
        f"motivo={resultado_acao['motivo_acao']} inter={analise['tipo_intersecao']} "
        f"C/L/R={analise['black_center']}/{analise['black_left']}/{analise['black_right']} "
        f"ratio={analise['black_ratio_center']:.3f}/{analise['black_ratio_left']:.3f}/{analise['black_ratio_right']:.3f} "
        f"depois={depois} move={'ON' if green_move else 'OFF'} estado={estado_verde['estado']} cmd={comando_final}"
        f" acao_confirmada={estado_verde['acao_confirmada']} turn_lost={estado_verde['turn_saw_lost_line']}"
        f" evento={estado_verde['evento']}"
    )
    estado_verde["ultimo_log"] = agora
    estado_verde["ultimo_estado_logado"] = estado_verde["estado"]


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
    validos_por_lado = {"ESQUERDA": [], "CENTRO": [], "DIREITA": []}

    for angulo in range(DEST_ANGULO_MIN, DEST_ANGULO_MAX + 1, DEST_PASSO_ANGULO):
        raio = amostrar_raio(mascara, origem, angulo)
        raios.append(raio)
        if not raio["ok"]:
            continue
        x_local, y_local = raio["destino_local"]
        vetor_x = x_local - robo_x
        vetor_y = robo_y - y_local
        angulo_destino = math.degrees(math.atan2(vetor_x, max(vetor_y, 1)))
        tipo = tipo_por_angulo(angulo_destino)
        lado = "ESQUERDA" if angulo_destino < -15 else "DIREITA" if angulo_destino > 15 else "CENTRO"
        candidato = {
            "ok": True,
            "tipo": tipo,
            "angulo_raio": angulo,
            "angulo_destino": angulo_destino,
            "pixels": raio["pixels"],
            "continuidade": raio["continuidade"],
            "distancia": raio["distancia"],
            "segmento_hits": raio["segmento_hits"],
            "destino_local": raio["destino_local"],
            "destino_global": (resultado["x_inicio_roi"] + x_local, resultado["y_inicio_roi"] + y_local),
            "lado": lado,
            "origem_local": origem,
            "raio": raio,
        }
        candidato["score"] = calcular_score_raio(
            tipo, candidato["pixels"], candidato["continuidade"], candidato["distancia"], angulo_destino
        )
        if candidato_confiavel(candidato):
            validos[tipo].append(candidato)
            validos_por_lado[lado].append(candidato)

    for tipo, motivo in (("FRENTE", "FRENTE_PRIORITARIA"), ("CURVA", "CURVA_SEM_FRENTE"), ("RETORNO", "RETORNO_SEM_FRENTE_CURVA")):
        if validos[tipo]:
            escolhido = max(validos[tipo], key=lambda item: item["score"])
            escolhido["motivo"] = motivo
            escolhido["raios"] = raios
            escolhido["validos_por_tipo"] = validos
            escolhido["validos_por_lado"] = validos_por_lado
            escolhido["erro_x"] = escolhido["destino_local"][0] - robo_x
            return escolhido

    return {
        "ok": False, "tipo": "PERDIDO", "motivo": "RECUPERAR", "raios": raios,
        "validos_por_tipo": validos, "validos_por_lado": validos_por_lado, "origem_local": origem,
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
    angulo = destino.get("angulo_destino", destino.get("angulo_raio", 0))
    if angulo < -15:
        return "ESQUERDA"
    if angulo > 15:
        return "DIREITA"
    return "CENTRO"


def resetar_confirmacao_lado_recuperacao(memoria):
    memoria["lado_recuperacao_pendente"] = "CENTRO"
    memoria["frames_confirmacao_lado_recuperacao"] = 0


def atualizar_memoria_recuperacao(destino, memoria):
    """Registra somente curvas fortes estaveis ou retornos ja aceitos."""
    if not destino.get("ok", False):
        resetar_confirmacao_lado_recuperacao(memoria)
        return

    lado = lado_destino(destino)
    if destino["tipo"] == "RETORNO":
        if lado in ("ESQUERDA", "DIREITA"):
            memoria["ultimo_lado_recuperacao"] = lado
        resetar_confirmacao_lado_recuperacao(memoria)
        return

    if destino["tipo"] != "CURVA" or lado == "CENTRO" or abs(destino["angulo_destino"]) < DEST_LADO_RECUPERACAO_CURVA_ANGULO_MIN:
        resetar_confirmacao_lado_recuperacao(memoria)
        return

    if memoria["lado_recuperacao_pendente"] == lado:
        memoria["frames_confirmacao_lado_recuperacao"] += 1
    else:
        memoria["lado_recuperacao_pendente"] = lado
        memoria["frames_confirmacao_lado_recuperacao"] = 1
    if memoria["frames_confirmacao_lado_recuperacao"] >= DEST_LADO_RECUPERACAO_CONFIRMAR_FRAMES:
        memoria["ultimo_lado_recuperacao"] = lado
        resetar_confirmacao_lado_recuperacao(memoria)


def avaliar_retorno_lado_oposto(destino, memoria):
    """Confirma RETORNO oposto antes de substituir o lado de recuperacao."""
    if not destino.get("ok", False) or destino.get("tipo") != "RETORNO":
        resetar_confirmacao_lado_recuperacao(memoria)
        return "NAO_APLICA"

    lado_atual = lado_destino(destino)
    lado_anterior = memoria.get("ultimo_lado_recuperacao", "CENTRO")
    if lado_atual == "CENTRO":
        resetar_confirmacao_lado_recuperacao(memoria)
        return "NAO_APLICA"
    if lado_anterior == "CENTRO" or lado_atual == lado_anterior:
        resetar_confirmacao_lado_recuperacao(memoria)
        return "ACEITAR_RETORNO"

    if memoria["tipo_pendente"] != "RETORNO":
        resetar_confirmacao_lado_recuperacao(memoria)
        memoria["tipo_pendente"] = "RETORNO"
    if memoria.get("lado_recuperacao_pendente") == lado_atual:
        memoria["frames_confirmacao_lado_recuperacao"] += 1
    else:
        memoria["lado_recuperacao_pendente"] = lado_atual
        memoria["frames_confirmacao_lado_recuperacao"] = 1
    if memoria["frames_confirmacao_lado_recuperacao"] >= DEST_LADO_RECUPERACAO_CONFIRMAR_FRAMES:
        return "ACEITAR_RETORNO"
    return "CONFIRMAR_LADO"


def atualizar_tipo_confiavel(destino, memoria):
    if destino["tipo"] == memoria["ultimo_tipo_confiavel"]:
        memoria["frames_mesmo_tipo"] += 1
    else:
        memoria["ultimo_tipo_confiavel"] = destino["tipo"]
        memoria["frames_mesmo_tipo"] = 1
    memoria["ultimo_score_confiavel"] = destino["score"]
    atualizar_memoria_recuperacao(destino, memoria)
    memoria["tipo_pendente"] = "PERDIDO"
    memoria["frames_confirmacao_tipo"] = 0


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
        resetar_confirmacao_lado_recuperacao(memoria)
        return True

    estado_retorno = avaliar_retorno_lado_oposto(destino, memoria)
    if estado_retorno == "CONFIRMAR_LADO":
        return True
    if estado_retorno == "ACEITAR_RETORNO":
        atualizar_tipo_confiavel(destino, memoria)
        return False
    return decidir_tipo(destino, memoria)


def avaliar_recuperacao(resultado, destino, memoria):
    """Prioriza destino visual valido sobre a flag global de linha."""
    if destino.get("ok", False):
        if not resultado.get("encontrou_linha", False):
            return False, "LINHA_FALSE_DESTINO_OK"
        if memoria.get("em_recuperacao", False):
            return False, "EM_RECUPERACAO_COM_DESTINO_OK"
        return False, "NAO_APLICA"

    if memoria.get("em_recuperacao", False):
        return True, "EM_RECUPERACAO_SEM_DESTINO"
    return True, "DESTINO_PERDIDO"


def deve_confirmar_destino(resultado, destino):
    """Mantem o robo cauteloso quando so a flag global de linha falha."""
    return destino["ok"] and not resultado["encontrou_linha"]


def comando_recuperacao(ultimo_lado_recuperacao, controle_varredura):
    if ultimo_lado_recuperacao == "ESQUERDA":
        return f"GIRAR_ESQ {DEST_VEL_RECUPERAR}"
    if ultimo_lado_recuperacao == "DIREITA":
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


def criar_debug_destinos(resultado, destino, estado, comando, memoria, motivo_recuperacao):
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
    quantidades_lado = {lado: len(destino["validos_por_lado"][lado]) for lado in ("ESQUERDA", "CENTRO", "DIREITA")}
    linhas = [
        f"Estado: {estado}", f"Tipo: {destino['tipo']}", f"Cmd: {comando}",
        f"Motivo: {destino['motivo']}", f"Score: {destino.get('score', 0):.1f}",
        f"Continuidade: {destino.get('continuidade', 0):.2f}", f"Distancia: {destino.get('distancia', 0):.0f}",
        f"Segmento hits: {destino.get('segmento_hits', 0)}", f"Validos F/C/R: {quantidades['FRENTE']}/{quantidades['CURVA']}/{quantidades['RETORNO']}",
        f"Validos E/C/D: {quantidades_lado['ESQUERDA']}/{quantidades_lado['CENTRO']}/{quantidades_lado['DIREITA']}",
        f"Linha encontrada: {resultado['encontrou_linha']}", f"Destino ok: {destino['ok']}",
        f"Motivo recuperacao: {motivo_recuperacao}", f"Ang raio/dest: {destino.get('angulo_raio', 0):.1f}/{destino.get('angulo_destino', 0):.1f}",
        f"Lado recuperacao: {memoria['ultimo_lado_recuperacao']}",
        f"Pend. recuperacao: {memoria['lado_recuperacao_pendente']} ({memoria['frames_confirmacao_lado_recuperacao']})",
        f"Perdido: {memoria['frames_destino_perdido']}",
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


def imprimir_log(resultado, estado, destino, comando, memoria, motivo_recuperacao):
    quantidades = {tipo: len(destino["validos_por_tipo"][tipo]) for tipo in ("FRENTE", "CURVA", "RETORNO")}
    quantidades_lado = {lado: len(destino["validos_por_lado"][lado]) for lado in ("ESQUERDA", "CENTRO", "DIREITA")}
    if destino["ok"]:
        print(
            f"Estado: {estado} | Tipo: {destino['tipo']} | Cmd: {comando} | "
            f"ang_raio: {destino['angulo_raio']:.1f} | ang_dest: {destino['angulo_destino']:.1f} | score: {destino['score']:.1f} | "
            f"cont: {destino['continuidade']:.2f} | dist: {destino['distancia']:.0f} | "
            f"segmento: {destino['segmento_hits']} | motivo: {destino['motivo']} | "
            f"linha_encontrada: {resultado['encontrou_linha']} | destino_ok: {destino['ok']} | motivo_recuperacao: {motivo_recuperacao} | "
            f"lado_dest: {lado_destino(destino)} | ultimo_lado_recuperacao: {memoria['ultimo_lado_recuperacao']} | "
            f"ultimo_tipo: {memoria['ultimo_tipo_confiavel']} | pend_tipo: {memoria['tipo_pendente']} | "
            f"lado_recuperacao_pendente: {memoria['lado_recuperacao_pendente']} | "
            f"frames_lado_recuperacao: {memoria['frames_confirmacao_lado_recuperacao']} | "
            f"validos_fcr: {quantidades['FRENTE']}/{quantidades['CURVA']}/{quantidades['RETORNO']} | "
            f"validos_ecd: {quantidades_lado['ESQUERDA']}/{quantidades_lado['CENTRO']}/{quantidades_lado['DIREITA']}"
        )
    else:
        print(
            f"Estado: {estado} | Tipo: PERDIDO | Cmd: {comando} | "
            f"linha_encontrada: {resultado['encontrou_linha']} | destino_ok: False | motivo_recuperacao: {motivo_recuperacao} | "
            f"ultimo_lado_recuperacao: {memoria['ultimo_lado_recuperacao']}"
        )


def main():
    args = ler_argumentos()
    green_log = args.green_log or args.green_move
    green_move = args.green_move and args.motores
    if args.green_move and not args.motores:
        print("[GREEN] --green-move solicitado sem --motores; usando modo log-only.")
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
            "ultimo_lado_recuperacao": "CENTRO", "frames_destino_perdido": 0,
            "frames_destino_confiavel": 0, "em_recuperacao": False,
            "lado_recuperacao_pendente": "CENTRO", "frames_confirmacao_lado_recuperacao": 0,
        }
        controle_varredura = {"etapa": 0, "ultima_troca": time.monotonic()}
        estado_anterior, ultimo_debug = None, 0
        estado_verde = criar_estado_verde(time.monotonic())

        while True:
            frame = capturar_frame_bgr(camera)
            resultado = detectar_linha(frame)
            destino = escolher_destino(resultado)
            if not destino["ok"]:
                resetar_confirmacao_lado_recuperacao(memoria)
            if destino["ok"]:
                memoria["frames_destino_confiavel"] += 1
                memoria["frames_destino_perdido"] = 0
            else:
                memoria["frames_destino_perdido"] += 1
                memoria["frames_destino_confiavel"] = 0

            deve_recuperar, motivo_recuperacao = avaliar_recuperacao(resultado, destino, memoria)
            if memoria["em_recuperacao"] and destino["ok"]:
                if memoria["frames_destino_confiavel"] < DEST_FRAMES_DESTINO_CONFIAVEL:
                    estado = "FOLLOW_CONFIRMAR"
                    comando, _, _, _ = controlar_destino(destino, memoria, confirmar=True)
                else:
                    memoria["em_recuperacao"] = False
                    confirmar = decidir_confirmacao_destino(destino, memoria)
                    estado = "FOLLOW_CONFIRMAR" if confirmar else "FOLLOW_DESTINO"
                    comando, _, _, _ = controlar_destino(destino, memoria, confirmar=confirmar)
            elif deve_recuperar:
                memoria["em_recuperacao"] = True
                estado = "RECUPERAR_GIRO" if memoria["ultimo_lado_recuperacao"] in ("ESQUERDA", "DIREITA") else "RECUPERAR_VARREDURA"
                comando = comando_recuperacao(memoria["ultimo_lado_recuperacao"], controle_varredura)
                memoria["correcao_anterior"] = 0
                memoria["vel_anterior"] = (0, 0)
            elif destino["ok"]:
                confirmar = deve_confirmar_destino(resultado, destino) or decidir_confirmacao_destino(destino, memoria)
                estado = "FOLLOW_CONFIRMAR" if confirmar else "FOLLOW_DESTINO"
                comando, _, _, _ = controlar_destino(destino, memoria, confirmar=confirmar)
            else:
                estado = "FOLLOW_CONFIRMAR"
                comando = comando_confirmar_sem_destino(memoria)

            comando_final = comando
            if green_log:
                resultado_verde, analise_intersecao, resultado_acao_verde = calcular_acao_verde(frame, resultado)
                agora_verde = time.monotonic()
                atualizar_estado_verde(
                    estado_verde, resultado_acao_verde, destino, estado, green_move, agora_verde, memoria
                )
                comando_verde = comando_override_verde(estado_verde, destino, green_move)
                if comando_verde is not None:
                    comando_final = comando_verde
                log_verde(
                    resultado_verde, analise_intersecao, resultado_acao_verde,
                    estado_verde, green_move, comando_final, agora_verde,
                )

            enviar_seguro(conexao, comando_final, args.motores)
            imprimir_log(resultado, estado, destino, comando_final, memoria, motivo_recuperacao)
            if args.salvar_debug:
                agora = time.monotonic()
                if DEST_SALVAR_DEBUG_EVENTOS and (estado != estado_anterior or agora - ultimo_debug >= DEST_INTERVALO_DEBUG):
                    salvar_debug(criar_debug_destinos(resultado, destino, estado, comando_final, memoria, motivo_recuperacao), estado)
                    ultimo_debug = agora
            estado_anterior = estado
            if args.mostrar:
                cv2.imshow("follow_destinos", criar_debug_destinos(resultado, destino, estado, comando_final, memoria, motivo_recuperacao))
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
