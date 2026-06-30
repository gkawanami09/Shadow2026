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
)
from line_test import criar_debug_linha, detectar_linha

try:
    from utils import abrir_serial, enviar_comando
except ImportError:
    abrir_serial = None
    enviar_comando = None

try:
    from verdes import analisar_verdes
except ImportError:
    analisar_verdes = None


TEMPO_SEGURAR_DECISAO_LOG = 0.80
INTERVALO_LOG_SIMPLES = 0.20
DECISOES_VERDE_LOG = ("RETO", "ESQUERDA", "DIREITA", "RETORNO", "INSEGURO")
ACOES_VERDE_VALIDAS = ("ESQUERDA", "DIREITA", "RETORNO")

TEMPO_CONFIRMAR_VERDE = 0.75
MIN_FRAMES_CONFIRMAR_VERDE = 3
MIN_VOTOS_LADO_VERDE = 2
MIN_VOTOS_RETORNO_VERDE = 2
MAX_FRAMES_NENHUM_CONFIRMAR_VERDE = 4
TEMPO_ARMAR_VERDE = 0.90

TEMPO_AVANCAR_APOS_VERDE = 0.45

TEMPO_EXECUTAR_LADO_VERDE = 1.20
TEMPO_EXECUTAR_RETORNO_VERDE = 2.10
TEMPO_MINIMO_CEGO_RETORNO_VERDE = 1.80

TEMPO_RECUPERAR_LINHA_RETORNO_VERDE = 0.80
TEMPO_MAX_BUSCA_LINHA_POS_VERDE = 1.20
TEMPO_COOLDOWN_VERDE = 1.30

MIN_FRAMES_LINHA_POS_VERDE = 2
MIN_FRAMES_SEM_VERDE_PARA_REARMAR = 3

# Curva fechada: parametros concentrados aqui para calibracao em pista.
TANQUE_90_ATIVO = True
TANQUE_90_ANGULO_MIN = 45
TANQUE_90_ERRO_X_REL_MIN = 0.22
TANQUE_90_SCORE_LATERAL_MIN = 600
TANQUE_90_CONTINUIDADE_LATERAL_MIN = 0.65
TANQUE_90_DISTANCIA_LATERAL_MIN = 90
TANQUE_90_SCORE_FRENTE_MAX = 900
TANQUE_90_DISTANCIA_FRENTE_MAX = 150
TANQUE_90_CONTINUIDADE_FRENTE_MAX = 0.55
TANQUE_90_TEMPO_MIN = 0.18
TANQUE_90_TEMPO_MAX = 0.75
TANQUE_90_VEL = DEST_VEL_RECUPERAR

MODOS_VERDE_BLOQUEIAM_TANQUE_90 = {
    "AVANCANDO_APOS_VERDE",
    "EXECUTANDO_VERDE",
    "RECUPERANDO_LINHA",
}


def interpretar_acoes_verde(valor):
    acoes = tuple(dict.fromkeys(item.strip().upper() for item in valor.split(",") if item.strip()))
    invalidas = [acao for acao in acoes if acao not in ACOES_VERDE_VALIDAS]
    if not acoes or invalidas:
        permitidas = ",".join(ACOES_VERDE_VALIDAS)
        detalhe = f" Invalidas: {','.join(invalidas)}." if invalidas else ""
        raise argparse.ArgumentTypeError(f"Use uma ou mais acoes entre {permitidas}.{detalhe}")
    return frozenset(acoes)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha por hierarquia de destinos visuais.")
    parser.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    parser.add_argument("--motores", action="store_true", help="Envia comandos ao Arduino.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou auto.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva imagens de debug.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra janela OpenCV.")
    parser.add_argument(
        "--stream-debug",
        action="store_true",
        help="Mostra stream visual em tempo real com linha, destino, verde, estado e comando.",
    )
    parser.add_argument("--verde-sombra", action="store_true", help="Analisa verdes sem interferir no movimento.")
    parser.add_argument("--verde-ativo", action="store_true", help="Ativa manobras confirmadas pelo detector de verdes.")
    parser.add_argument(
        "--verde-acoes",
        type=interpretar_acoes_verde,
        default=frozenset(ACOES_VERDE_VALIDAS),
        metavar="ACOES",
        help="Acoes verdes habilitadas, separadas por virgula.",
    )
    parser.add_argument("--log", action="store_true", help="Imprime log limpo de decisao visual.")
    return parser.parse_args()


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def criar_estado_log():
    return {
        "ultima_linha": None,
        "ultimo_tempo": 0.0,
        "decisao_segura": "NENHUM",
        "tempo_ultima_decisao": 0.0,
    }


def criar_estado_verde_ativo():
    return {
        "modo": "NORMAL",
        "decisao_confirmada": "NENHUM",
        "decisao_em_confirmacao": "NENHUM",
        "decisao_intencao": "NENHUM",
        "tempo_inicio_intencao": 0.0,
        "tempo_inicio": 0.0,
        "cooldown_ate": 0.0,
        "frames_confirmacao": 0,
        "frames_nenhum_confirmacao": 0,
        "frames_sem_verde": 0,
        "lado_busca_pos_verde": "CENTRO",
        "lado_giro_retorno": "CENTRO",
        "busca_pos_verde_ate": 0.0,
        "frames_linha_pos_verde": 0,
        "votos": {
            "ESQUERDA": 0,
            "DIREITA": 0,
            "RETORNO": 0,
            "OUTRO": 0,
        },
    }


def limpar_confirmacao_verde(estado_verde):
    estado_verde["decisao_confirmada"] = "NENHUM"
    estado_verde["decisao_em_confirmacao"] = "NENHUM"
    estado_verde["decisao_intencao"] = "NENHUM"
    estado_verde["tempo_inicio_intencao"] = 0.0
    estado_verde["frames_confirmacao"] = 0
    estado_verde["frames_nenhum_confirmacao"] = 0
    estado_verde["lado_busca_pos_verde"] = "CENTRO"
    estado_verde["lado_giro_retorno"] = "CENTRO"
    estado_verde["busca_pos_verde_ate"] = 0.0
    estado_verde["frames_linha_pos_verde"] = 0
    for chave in estado_verde["votos"]:
        estado_verde["votos"][chave] = 0


def entrar_cooldown_verde(estado_verde, agora, duracao=TEMPO_COOLDOWN_VERDE):
    estado_verde["modo"] = "COOLDOWN_VERDE"
    estado_verde["tempo_inicio"] = agora
    estado_verde["cooldown_ate"] = agora + duracao
    estado_verde["frames_sem_verde"] = 0


def registrar_voto_verde(estado_verde, decisao, acoes_habilitadas):
    chave = decisao if decisao in acoes_habilitadas else "OUTRO"
    estado_verde["votos"][chave] += 1
    estado_verde["frames_confirmacao"] += 1


def escolher_decisao_confirmada_verde(estado_verde, acoes_habilitadas):
    votos = estado_verde["votos"]
    if "RETORNO" in acoes_habilitadas and votos["RETORNO"] >= MIN_VOTOS_RETORNO_VERDE:
        return "RETORNO"
    if (
        "ESQUERDA" in acoes_habilitadas
        and votos["ESQUERDA"] >= MIN_VOTOS_LADO_VERDE
        and votos["DIREITA"] == 0
    ):
        return "ESQUERDA"
    if (
        "DIREITA" in acoes_habilitadas
        and votos["DIREITA"] >= MIN_VOTOS_LADO_VERDE
        and votos["ESQUERDA"] == 0
    ):
        return "DIREITA"
    return "NENHUM"


def confirmar_decisao_verde(estado_verde, decisao, agora):
    estado_verde["decisao_confirmada"] = decisao
    estado_verde["lado_busca_pos_verde"] = decisao
    estado_verde["frames_linha_pos_verde"] = 0
    estado_verde["modo"] = "AVANCANDO_APOS_VERDE"
    estado_verde["tempo_inicio"] = agora


def atualizar_estado_verde_ativo(
    estado_verde,
    decisao_crua,
    acoes_habilitadas,
    agora,
    destino=None,
):
    modo = estado_verde["modo"]

    if modo == "NORMAL":
        if decisao_crua not in acoes_habilitadas:
            return estado_verde
        limpar_confirmacao_verde(estado_verde)
        estado_verde["modo"] = "CONFIRMANDO_VERDE"
        estado_verde["decisao_em_confirmacao"] = decisao_crua
        estado_verde["decisao_intencao"] = decisao_crua
        estado_verde["tempo_inicio"] = agora
        estado_verde["tempo_inicio_intencao"] = agora
        registrar_voto_verde(estado_verde, decisao_crua, acoes_habilitadas)
        return estado_verde

    if modo == "CONFIRMANDO_VERDE":
        intencao = estado_verde["decisao_intencao"]
        if decisao_crua == intencao:
            estado_verde["frames_nenhum_confirmacao"] = 0
            registrar_voto_verde(estado_verde, decisao_crua, acoes_habilitadas)
        elif decisao_crua == "NENHUM":
            estado_verde["frames_nenhum_confirmacao"] += 1
            estado_verde["frames_confirmacao"] += 1
            if (
                estado_verde["frames_nenhum_confirmacao"]
                > MAX_FRAMES_NENHUM_CONFIRMAR_VERDE
            ):
                limpar_confirmacao_verde(estado_verde)
                entrar_cooldown_verde(estado_verde, agora, TEMPO_CONFIRMAR_VERDE)
                return estado_verde
        elif decisao_crua in acoes_habilitadas:
            estado_verde["frames_nenhum_confirmacao"] = 0
            if intencao != "RETORNO" and decisao_crua == "RETORNO":
                registrar_voto_verde(estado_verde, decisao_crua, acoes_habilitadas)
                if estado_verde["votos"]["RETORNO"] >= MIN_VOTOS_RETORNO_VERDE:
                    estado_verde["decisao_intencao"] = "RETORNO"
                    estado_verde["decisao_em_confirmacao"] = "RETORNO"
            else:
                limpar_confirmacao_verde(estado_verde)
                entrar_cooldown_verde(estado_verde, agora, TEMPO_CONFIRMAR_VERDE)
                return estado_verde
        else:
            limpar_confirmacao_verde(estado_verde)
            entrar_cooldown_verde(estado_verde, agora, TEMPO_CONFIRMAR_VERDE)
            return estado_verde

        tem_frames = estado_verde["frames_confirmacao"] >= MIN_FRAMES_CONFIRMAR_VERDE
        decisao = escolher_decisao_confirmada_verde(estado_verde, acoes_habilitadas)
        if tem_frames and decisao != "NENHUM":
            confirmar_decisao_verde(estado_verde, decisao, agora)
            return estado_verde

        tempo_intencao = agora - estado_verde["tempo_inicio_intencao"]
        if tempo_intencao >= TEMPO_ARMAR_VERDE:
            if decisao != "NENHUM" and tem_frames:
                confirmar_decisao_verde(estado_verde, decisao, agora)
            else:
                limpar_confirmacao_verde(estado_verde)
                entrar_cooldown_verde(estado_verde, agora, TEMPO_CONFIRMAR_VERDE)
        return estado_verde

    if modo == "AVANCANDO_APOS_VERDE":
        if agora - estado_verde["tempo_inicio"] >= TEMPO_AVANCAR_APOS_VERDE:
            estado_verde["modo"] = "EXECUTANDO_VERDE"
            estado_verde["tempo_inicio"] = agora
        return estado_verde

    if modo == "EXECUTANDO_VERDE":
        duracao = (
            max(TEMPO_EXECUTAR_RETORNO_VERDE, TEMPO_MINIMO_CEGO_RETORNO_VERDE)
            if estado_verde["decisao_confirmada"] == "RETORNO"
            else TEMPO_EXECUTAR_LADO_VERDE
        )
        if agora - estado_verde["tempo_inicio"] >= duracao:
            estado_verde["modo"] = "RECUPERANDO_LINHA"
            estado_verde["tempo_inicio"] = agora
            estado_verde["frames_linha_pos_verde"] = 0
            tempo_busca = (
                TEMPO_RECUPERAR_LINHA_RETORNO_VERDE
                if estado_verde["decisao_confirmada"] == "RETORNO"
                else TEMPO_MAX_BUSCA_LINHA_POS_VERDE
            )
            estado_verde["busca_pos_verde_ate"] = agora + tempo_busca
        return estado_verde

    if modo == "RECUPERANDO_LINHA":
        destino_busca = escolher_destino_busca_pos_verde(destino, estado_verde)
        if destino_busca is not None and destino_busca.get("ok", False):
            estado_verde["frames_linha_pos_verde"] += 1
        else:
            estado_verde["frames_linha_pos_verde"] = 0
        linha_estavel = estado_verde["frames_linha_pos_verde"] >= MIN_FRAMES_LINHA_POS_VERDE
        busca_expirada = agora >= estado_verde["busca_pos_verde_ate"]
        if linha_estavel or busca_expirada:
            entrar_cooldown_verde(estado_verde, agora)
        return estado_verde

    if modo == "COOLDOWN_VERDE":
        if decisao_crua == "NENHUM":
            estado_verde["frames_sem_verde"] += 1
        else:
            estado_verde["frames_sem_verde"] = 0
        if (
            agora >= estado_verde["cooldown_ate"]
            and estado_verde["frames_sem_verde"] >= MIN_FRAMES_SEM_VERDE_PARA_REARMAR
        ):
            limpar_confirmacao_verde(estado_verde)
            estado_verde["modo"] = "NORMAL"
            estado_verde["tempo_inicio"] = agora
        return estado_verde

    raise ValueError(f"Estado de verde ativo invalido: {modo}")


def log_estado_verde_ativo(estado_verde):
    modo = estado_verde["modo"]
    if modo == "NORMAL":
        return "SEGUE_LINHA"
    if modo == "AVANCANDO_APOS_VERDE":
        return "AVANCANDO_VERDE"
    if modo == "EXECUTANDO_VERDE":
        if estado_verde["decisao_confirmada"] == "RETORNO":
            return "VERDE_RETORNO_CEGO"
        return f"VERDE_{estado_verde['decisao_confirmada']}"
    if modo == "RECUPERANDO_LINHA":
        return f"BUSCANDO_LINHA_{estado_verde['lado_busca_pos_verde']}"
    return modo


def filtrar_decisao_verde_para_acao(decisao, acao_permitida, origem_decisao):
    if acao_permitida and origem_decisao in ("VERDE_ANTES_INTERSECAO", "VERDE_ANTES_TOLERADO"):
        return decisao
    return "NENHUM"


def formatar_log_simples(decisao):
    if decisao == "NENHUM" or decisao == "SEGUE_LINHA":
        return "[LOG] SEGUE_LINHA"
    return f"[LOG] {decisao}"


def atualizar_decisao_log_verde(resultado_verde, estado_log, agora):
    decisao_atual = "NENHUM"
    if resultado_verde is not None:
        decisao_atual = resultado_verde.get("decisao", "NENHUM")

    if decisao_atual in DECISOES_VERDE_LOG:
        estado_log["decisao_segura"] = decisao_atual
        estado_log["tempo_ultima_decisao"] = agora
        return decisao_atual

    if agora - estado_log["tempo_ultima_decisao"] <= TEMPO_SEGURAR_DECISAO_LOG:
        return estado_log["decisao_segura"]

    estado_log["decisao_segura"] = "NENHUM"
    return "SEGUE_LINHA"


def imprimir_log_simples(decisao, estado_log, agora):
    linha = formatar_log_simples(decisao)
    mudou = linha != estado_log.get("ultima_linha")
    passou_intervalo = agora - estado_log.get("ultimo_tempo", 0.0) >= INTERVALO_LOG_SIMPLES
    if mudou or passou_intervalo:
        print(linha)
        estado_log["ultima_linha"] = linha
        estado_log["ultimo_tempo"] = agora


def enviar_seguro(conexao, comando, motores_ativos):
    if not motores_ativos:
        return None
    if enviar_comando is None:
        raise RuntimeError("Suporte serial indisponivel.")
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


def _candidato_lateral_forte_tanque_90(candidato, largura_frame=None):
    angulo = candidato.get("angulo_destino", candidato.get("angulo_raio", 0))
    forte = (
        candidato.get("score", 0) >= TANQUE_90_SCORE_LATERAL_MIN
        and candidato.get("continuidade", 0) >= TANQUE_90_CONTINUIDADE_LATERAL_MIN
        and candidato.get("distancia", 0) >= TANQUE_90_DISTANCIA_LATERAL_MIN
        and abs(angulo) >= TANQUE_90_ANGULO_MIN
    )
    if not forte or not largura_frame:
        return forte
    origem_x = (candidato.get("origem_local") or (0, 0))[0]
    destino_x = (candidato.get("destino_local") or (origem_x, 0))[0]
    return abs(destino_x - origem_x) / float(largura_frame) >= TANQUE_90_ERRO_X_REL_MIN


def escolher_lado_tanque_90(validos_por_lado):
    """Escolhe uma unica lateral forte; duas laterais indicam intersecao/T."""
    melhores = {}
    for lado in ("ESQUERDA", "DIREITA"):
        fortes = [
            candidato
            for candidato in validos_por_lado.get(lado, [])
            if _candidato_lateral_forte_tanque_90(candidato)
        ]
        if fortes:
            melhores[lado] = max(fortes, key=lambda item: item.get("score", 0))
    if len(melhores) != 1:
        return None
    return next(iter(melhores))


def frente_parece_falsa_em_curva_90(destino, validos_por_lado, largura_frame):
    if not TANQUE_90_ATIVO:
        return False
    lado = escolher_lado_tanque_90(validos_por_lado)
    if lado is None:
        return False
    if not any(
        _candidato_lateral_forte_tanque_90(candidato, largura_frame)
        for candidato in validos_por_lado.get(lado, [])
    ):
        return False

    candidatos_frente = (destino.get("validos_por_tipo") or {}).get("FRENTE", [])
    if not candidatos_frente:
        return True
    frente = max(candidatos_frente, key=lambda item: item.get("score", 0))
    return (
        frente.get("score", 0) <= TANQUE_90_SCORE_FRENTE_MAX
        or frente.get("distancia", 0) <= TANQUE_90_DISTANCIA_FRENTE_MAX
        or frente.get("continuidade", 0) <= TANQUE_90_CONTINUIDADE_FRENTE_MAX
    )


def iniciar_tanque_90(memoria, lado, agora):
    memoria["tanque_90_ativo"] = True
    memoria["tanque_90_lado"] = lado
    memoria["tanque_90_inicio"] = agora
    memoria["motivo_tanque_90"] = "LATERAL_FORTE_FRENTE_FRACA"


def atualizar_tanque_90(memoria, destino, agora):
    if not memoria.get("tanque_90_ativo", False):
        return False
    decorrido = agora - memoria.get("tanque_90_inicio", agora)
    if decorrido >= TANQUE_90_TEMPO_MAX:
        memoria["tanque_90_ativo"] = False
        memoria["tanque_90_lado"] = "CENTRO"
        memoria["motivo_tanque_90"] = "TEMPO_MAX"
        memoria["tanque_90_aguarda_rearme"] = True
        return False
    frente_central_confiavel = (
        destino.get("ok", False)
        and destino.get("tipo") == "FRENTE"
        and lado_destino(destino) == "CENTRO"
        and candidato_confiavel(destino)
        and destino.get("score", 0) > TANQUE_90_SCORE_FRENTE_MAX
        and destino.get("distancia", 0) > TANQUE_90_DISTANCIA_FRENTE_MAX
        and destino.get("continuidade", 0) > TANQUE_90_CONTINUIDADE_FRENTE_MAX
    )
    if decorrido >= TANQUE_90_TEMPO_MIN and frente_central_confiavel:
        memoria["tanque_90_ativo"] = False
        memoria["tanque_90_lado"] = "CENTRO"
        memoria["motivo_tanque_90"] = "FRENTE_RECUPERADA"
        return False
    return True


def comando_tanque_90(memoria):
    if memoria.get("tanque_90_lado") == "ESQUERDA":
        return f"GIRAR_ESQ {TANQUE_90_VEL}"
    if memoria.get("tanque_90_lado") == "DIREITA":
        return f"GIRAR_DIR {TANQUE_90_VEL}"
    return None


def preparar_destino_preferido(candidato, destino_normal, motivo):
    escolhido = dict(candidato)
    escolhido["motivo"] = motivo
    escolhido["raios"] = destino_normal["raios"]
    escolhido["validos_por_tipo"] = destino_normal["validos_por_tipo"]
    escolhido["validos_por_lado"] = destino_normal["validos_por_lado"]
    escolhido["erro_x"] = escolhido["destino_local"][0] - escolhido["origem_local"][0]
    return escolhido


def escolher_destino_preferindo_frente(destino_normal):
    candidatos = destino_normal["validos_por_tipo"].get("FRENTE", [])
    if not candidatos:
        return destino_normal
    escolhido = max(candidatos, key=lambda item: item["score"])
    return preparar_destino_preferido(escolhido, destino_normal, "VERDE_AVANCAR_FRENTE")


def escolher_destino_preferindo_lado(destino_normal, lado):
    candidatos = destino_normal["validos_por_lado"].get(lado, [])
    if not candidatos:
        return destino_normal
    escolhido = max(candidatos, key=lambda item: item["score"])
    return preparar_destino_preferido(escolhido, destino_normal, f"VERDE_{lado}")


def escolher_destino_busca_pos_verde(destino_normal, estado_verde):
    if destino_normal is None:
        return None

    lado_busca = estado_verde["lado_busca_pos_verde"]
    if lado_busca == "RETORNO":
        lado_busca = estado_verde.get("lado_giro_retorno", "CENTRO")

    lados_permitidos = []
    if lado_busca in ("ESQUERDA", "DIREITA"):
        lados_permitidos.append(lado_busca)
    lados_permitidos.append("CENTRO")

    for lado in lados_permitidos:
        candidatos = destino_normal["validos_por_lado"].get(lado, [])
        if candidatos:
            escolhido = max(candidatos, key=lambda item: item["score"])
            return preparar_destino_preferido(
                escolhido,
                destino_normal,
                f"BUSCA_POS_VERDE_{lado}",
            )
    return None


def comando_giro_retorno_verde(estado_verde, memoria):
    lado = estado_verde.get("lado_giro_retorno", "CENTRO")
    if lado not in ("ESQUERDA", "DIREITA"):
        lado_memoria = memoria.get("ultimo_lado_recuperacao", "CENTRO")
        lado = lado_memoria if lado_memoria in ("ESQUERDA", "DIREITA") else "ESQUERDA"
        estado_verde["lado_giro_retorno"] = lado
    giro = "GIRAR_ESQ" if lado == "ESQUERDA" else "GIRAR_DIR"
    return f"{giro} {DEST_VEL_RECUPERAR}"


def comando_busca_linha_pos_verde(estado_verde, destino, memoria):
    destino_busca = escolher_destino_busca_pos_verde(destino, estado_verde)
    if destino_busca is not None:
        comando, _, _, _ = controlar_destino(destino_busca, memoria, confirmar=True)
        return comando

    lado = estado_verde["lado_busca_pos_verde"]
    if lado == "RETORNO":
        lado = estado_verde.get("lado_giro_retorno", "ESQUERDA")
    giro = "GIRAR_DIR" if lado == "DIREITA" else "GIRAR_ESQ"
    return f"{giro} {DEST_VEL_RECUPERAR}"


def aplicar_verde_ativo(destino_normal, estado_verde):
    modo = estado_verde["modo"]
    decisao = estado_verde["decisao_confirmada"]
    if modo == "AVANCANDO_APOS_VERDE":
        return escolher_destino_preferindo_frente(destino_normal)
    if modo == "EXECUTANDO_VERDE" and decisao in ("ESQUERDA", "DIREITA"):
        return escolher_destino_preferindo_lado(destino_normal, decisao)
    return destino_normal


def aplicar_comando_verde_ativo(comando_normal, destino, estado_verde, memoria):
    if estado_verde["modo"] != "EXECUTANDO_VERDE":
        return comando_normal

    decisao = estado_verde["decisao_confirmada"]
    if decisao == "RETORNO":
        return comando_giro_retorno_verde(estado_verde, memoria)

    if decisao in ("ESQUERDA", "DIREITA"):
        if destino.get("ok", False) and lado_destino(destino) == decisao:
            return comando_normal
        giro = "GIRAR_ESQ" if decisao == "ESQUERDA" else "GIRAR_DIR"
        return f"{giro} {DEST_VEL_RECUPERAR}"
    return comando_normal


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


def aplicar_comando_confirmando_verde(destino, memoria):
    if destino.get("ok", False):
        comando, _, _, _ = controlar_destino(destino, memoria, confirmar=True)
        return comando
    return comando_confirmar_sem_destino(memoria)


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


def _texto_stream(debug, texto, posicao, cor=(255, 255, 255), escala=0.43):
    """Desenha texto legivel sem depender do conteudo recebido."""
    cv2.putText(debug, str(texto), posicao, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(debug, str(texto), posicao, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)


def _numero_stream(valor, casas=2):
    try:
        return f"{float(valor):.{casas}f}"
    except (TypeError, ValueError):
        return "NA"


def criar_stream_debug(
    frame_bgr,
    resultado_linha,
    destino,
    estado,
    comando,
    memoria,
    motivo_recuperacao,
    resultado_verde=None,
    estado_verde_ativo=None,
):
    """Monta o stream visual usando apenas resultados ja calculados."""
    debug = frame_bgr.copy()
    resultado_linha = resultado_linha or {}
    destino = destino or {}
    memoria = memoria or {}

    # Linha e ROI, sem executar novamente o detector.
    x_inicio = int(resultado_linha.get("x_inicio_roi", 0) or 0)
    y_inicio = int(resultado_linha.get("y_inicio_roi", 0) or 0)
    x_fim = int(resultado_linha.get("x_fim_roi", debug.shape[1] - 1) or 0)
    y_fim = int(resultado_linha.get("y_fim_roi", debug.shape[0] - 1) or 0)
    cv2.rectangle(debug, (x_inicio, y_inicio), (x_fim, y_fim), (255, 255, 0), 1)
    contorno = resultado_linha.get("contorno_principal")
    if contorno is not None:
        cv2.drawContours(debug, [contorno], -1, (0, 180, 0), 2, offset=(x_inicio, y_inicio))
    centro_linha_x = resultado_linha.get("centro_linha_x")
    centro_linha_y = resultado_linha.get("centro_linha_y")
    if centro_linha_x is not None and centro_linha_y is not None:
        cv2.circle(debug, (int(centro_linha_x), int(centro_linha_y)), 6, (0, 0, 255), -1)

    # Destino escolhido e o melhor candidato de cada lado.
    origem_local = destino.get("origem_local")
    if origem_local is not None and len(origem_local) >= 2:
        origem_global = (x_inicio + int(origem_local[0]), y_inicio + int(origem_local[1]))
        cv2.circle(debug, origem_global, 6, (255, 0, 255), -1)
        for lado, candidatos in (destino.get("validos_por_lado") or {}).items():
            if not candidatos:
                continue
            melhor = max(candidatos, key=lambda item: item.get("score", 0))
            ponto = melhor.get("destino_global")
            if ponto is None and melhor.get("destino_local") is not None:
                ponto = (x_inicio + int(melhor["destino_local"][0]), y_inicio + int(melhor["destino_local"][1]))
            if ponto is not None:
                ponto = (int(ponto[0]), int(ponto[1]))
                cv2.circle(debug, ponto, 4, (255, 120, 0), -1)
                _texto_stream(debug, lado, (ponto[0] + 5, ponto[1] - 5), (255, 180, 80), 0.35)
        if destino.get("ok", False):
            ponto = destino.get("destino_global")
            if ponto is None and destino.get("destino_local") is not None:
                ponto = (x_inicio + int(destino["destino_local"][0]), y_inicio + int(destino["destino_local"][1]))
            if ponto is not None:
                ponto = (int(ponto[0]), int(ponto[1]))
                cv2.arrowedLine(debug, origem_global, ponto, (255, 0, 0), 3, tipLength=0.15)
                _texto_stream(
                    debug,
                    f"{destino.get('tipo', 'NA')} {destino.get('motivo', 'NA')} "
                    f"ang={_numero_stream(destino.get('angulo_destino'), 1)} "
                    f"erro={destino.get('erro_x', 'NA')}",
                    (max(5, ponto[0] - 80), max(18, ponto[1] - 12)),
                    (255, 160, 40),
                    0.38,
                )

    # Verdes e cruzamento ja analisados no loop.
    if resultado_verde is not None:
        cruzamento = resultado_verde.get("cruzamento") or {}
        y_cruzamento = cruzamento.get("y_cruzamento")
        if y_cruzamento is not None:
            y_cruzamento = int(y_cruzamento)
            cv2.line(debug, (0, y_cruzamento), (debug.shape[1] - 1, y_cruzamento), (255, 255, 255), 2)
        for verde in resultado_verde.get("verdes") or []:
            bbox = verde.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x, y, w, h = (int(valor) for valor in bbox[:4])
            falso_depois = bool(
                verde.get("falso_depois_cruzamento", False)
                or verde.get("verde_depois_intersecao", False)
            )
            if verde.get("valido", False) and not falso_depois:
                cor = (0, 220, 0)
                rotulo = f"OK_{verde.get('lado', 'NA')}"
            elif falso_depois:
                cor = (0, 0, 255)
                rotulo = "FALSO_DEPOIS"
            else:
                cor = (0, 220, 255)
                motivo = str(verde.get("motivo", "INVALIDO"))
                rotulo = "INSEGURO" if "insegur" in motivo.lower() else "INVALIDO"
            cv2.rectangle(debug, (x, y), (x + w, y + h), cor, 2)
            _texto_stream(
                debug,
                f"{rotulo} {verde.get('lado', 'NA')} c={_numero_stream(verde.get('confianca'))} "
                f"pos={verde.get('posicao_cruzamento', 'NA')} {verde.get('motivo', 'NA')} "
                f"tol={verde.get('tolerado_desalinhado', False)} "
                f"rec={verde.get('recuperado_desalinhado', False)} "
                f"parcial={verde.get('verde_parcial_desalinhado', False)}",
                (x, max(15, y - 6)),
                cor,
                0.36,
            )

    verdes = (resultado_verde or {}).get("verdes") or []
    verde_valido_antes = any(
        item.get("valido", False)
        and not item.get("verde_depois_intersecao", False)
        and not item.get("falso_depois_cruzamento", False)
        for item in verdes
    )
    verde_falso_depois = bool((resultado_verde or {}).get("tem_verde_falso_depois", False))
    linhas = [
        f"estado={estado if estado is not None else 'NA'}",
        f"comando={comando if comando is not None else 'NA'}",
        f"tipo_destino={destino.get('tipo', 'NA')}",
        f"motivo={motivo_recuperacao if motivo_recuperacao is not None else destino.get('motivo', 'NA')}",
        f"erro_x={destino.get('erro_x', resultado_linha.get('erro', 'NA'))}",
        f"linha_encontrada={resultado_linha.get('encontrou_linha', False)}",
        f"destino_ok={destino.get('ok', False)}",
        f"ultimo_lado_recuperacao={memoria.get('ultimo_lado_recuperacao', 'NA')}",
        f"tanque_90_ativo={memoria.get('tanque_90_ativo', False)}",
        f"tanque_90_lado={memoria.get('tanque_90_lado', 'CENTRO')}",
        f"tanque_90_tempo={max(0.0, time.monotonic() - memoria.get('tanque_90_inicio', time.monotonic())):.2f}" if memoria.get("tanque_90_ativo", False) else "tanque_90_tempo=0.00",
        f"motivo_tanque_90={memoria.get('motivo_tanque_90', 'NA')}",
    ]
    if not destino.get("ok", False):
        linhas.append("DESTINO PERDIDO")
    if resultado_verde is not None:
        linhas.extend([
            f"verde_decisao={resultado_verde.get('decisao', 'NA')}",
            f"verde_motivo={resultado_verde.get('motivo', 'NA')}",
            f"verde_conf={_numero_stream(resultado_verde.get('confianca'))}",
            f"verde_acao={resultado_verde.get('acao_permitida', False)}",
            f"verde_origem={resultado_verde.get('origem_decisao', 'NA')}",
            f"verde_valido_antes={verde_valido_antes}",
            f"verde_falso_depois={verde_falso_depois}",
            f"verde_tolerado={resultado_verde.get('tem_verde_tolerado_desalinhado', False)}",
            f"verde_parcial={resultado_verde.get('tem_verde_parcial_desalinhado', False)}",
        ])
        cruzamento = resultado_verde.get("cruzamento") or {}
        linhas.extend([
            f"cruzamento_detectado={cruzamento.get('detectado', False)}",
            f"ramo_esquerda={cruzamento.get('ramo_esquerda', False)}",
            f"ramo_direita={cruzamento.get('ramo_direita', False)}",
            f"ramo_frente={cruzamento.get('ramo_frente', False)}",
            f"linha_baixo={cruzamento.get('linha_baixo', False)}",
            f"y_cruzamento={cruzamento.get('y_cruzamento', 'NA')}",
        ])
    if estado_verde_ativo is not None:
        modo_verde = estado_verde_ativo.get("modo", "NA")
        linhas.extend([
            f"modo_verde={modo_verde}",
            f"verde_bloqueia_tanque_90={modo_verde in MODOS_VERDE_BLOQUEIAM_TANQUE_90}",
            f"decisao_confirmada={estado_verde_ativo.get('decisao_confirmada', 'NA')}",
            f"decisao_intencao={estado_verde_ativo.get('decisao_intencao', 'NA')}",
            f"frames_confirmacao={estado_verde_ativo.get('frames_confirmacao', 0)}",
            f"frames_nenhum_confirmacao={estado_verde_ativo.get('frames_nenhum_confirmacao', 0)}",
        ])

    altura_linha = 18
    max_por_coluna = max(1, (debug.shape[0] - 12) // altura_linha)
    largura_coluna = max(300, debug.shape[1] // 2)
    for indice, texto_linha in enumerate(linhas):
        coluna = indice // max_por_coluna
        linha = indice % max_por_coluna
        _texto_stream(debug, texto_linha, (8 + coluna * largura_coluna, 18 + linha * altura_linha))
    return debug


def salvar_stream_debug(debug):
    pasta = Path("debug") / "stream"
    pasta.mkdir(parents=True, exist_ok=True)
    agora = datetime.now()
    caminho = pasta / f"stream_{agora.strftime('%Y%m%d_%H%M%S')}_{agora.microsecond // 1000:03d}.jpg"
    if not cv2.imwrite(str(caminho), debug):
        raise RuntimeError(f"Nao foi possivel salvar stream debug: {caminho}")
    return caminho


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
    camera = None
    conexao = None
    try:
        if not args.camera:
            print("Use --camera.")
            return 1
        if (args.verde_sombra or args.verde_ativo) and analisar_verdes is None:
            print("Erro: nao foi possivel importar analisar_verdes de verdes.py")
            return 1
        if args.motores:
            if abrir_serial is None:
                print("Erro: suporte serial indisponivel. Verifique a instalacao do pyserial.")
                return 1
            porta = SERIAL_PORT if args.porta == "auto" else args.porta
            conexao = abrir_serial(porta, BAUD_RATE, TIMEOUT_SERIAL)
            time.sleep(1.8)
            if hasattr(conexao, "reset_input_buffer"):
                conexao.reset_input_buffer()
                conexao.reset_output_buffer()
            if not args.log:
                print("Motores ativados. Iniciando segue-linha por destinos.")
        else:
            if not args.log:
                print("Simulacao sem motores.")

        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
        memoria = {
            "correcao_anterior": 0, "vel_anterior": (0, 0), "ultimo_tipo": "PERDIDO",
            "ultimo_tipo_confiavel": "PERDIDO", "ultimo_score_confiavel": 0,
            "frames_mesmo_tipo": 0, "tipo_pendente": "PERDIDO", "frames_confirmacao_tipo": 0,
            "ultimo_lado_recuperacao": "CENTRO", "frames_destino_perdido": 0,
            "frames_destino_confiavel": 0, "em_recuperacao": False,
            "lado_recuperacao_pendente": "CENTRO", "frames_confirmacao_lado_recuperacao": 0,
            "tanque_90_ativo": False, "tanque_90_lado": "CENTRO", "tanque_90_inicio": 0.0,
            "motivo_tanque_90": "INATIVO", "tanque_90_aguarda_rearme": False,
        }
        controle_varredura = {"etapa": 0, "ultima_troca": time.monotonic()}
        estado_log = criar_estado_log()
        estado_verde_ativo = criar_estado_verde_ativo()
        estado_anterior, ultimo_debug = None, 0

        while True:
            frame = capturar_frame_bgr(camera)
            analisar_verde = args.verde_sombra or args.verde_ativo
            resultado_verde = analisar_verdes(frame) if analisar_verde else None
            decisao_verde_crua = (
                resultado_verde.get("decisao", "NENHUM")
                if resultado_verde is not None
                else "NENHUM"
            )
            acao_permitida = (
                resultado_verde.get("acao_permitida", False)
                if resultado_verde is not None
                else False
            )
            origem_decisao = (
                resultado_verde.get("origem_decisao", "DESCONHECIDA")
                if resultado_verde is not None
                else "DESCONHECIDA"
            )
            decisao_verde_para_acao = filtrar_decisao_verde_para_acao(
                decisao_verde_crua,
                acao_permitida,
                origem_decisao,
            )
            agora = time.monotonic()
            resultado = detectar_linha(frame)
            destino_normal = escolher_destino(resultado)
            if args.verde_ativo:
                atualizar_estado_verde_ativo(
                    estado_verde_ativo,
                    decisao_verde_para_acao,
                    args.verde_acoes,
                    agora,
                    destino_normal,
                )
            destino = (
                aplicar_verde_ativo(destino_normal, estado_verde_ativo)
                if args.verde_ativo
                else destino_normal
            )
            verde_bloqueia_tanque_90 = (
                args.verde_ativo
                and estado_verde_ativo["modo"] in MODOS_VERDE_BLOQUEIAM_TANQUE_90
            )
            curva_90_detectada = frente_parece_falsa_em_curva_90(
                destino_normal,
                destino_normal["validos_por_lado"],
                frame.shape[1],
            )
            if not curva_90_detectada:
                memoria["tanque_90_aguarda_rearme"] = False
            if verde_bloqueia_tanque_90 and memoria["tanque_90_ativo"]:
                memoria["tanque_90_ativo"] = False
                memoria["tanque_90_lado"] = "CENTRO"
                memoria["motivo_tanque_90"] = "BLOQUEADO_POR_VERDE"
            elif memoria["tanque_90_ativo"]:
                atualizar_tanque_90(memoria, destino_normal, agora)
            elif (
                not verde_bloqueia_tanque_90
                and not memoria["tanque_90_aguarda_rearme"]
                and curva_90_detectada
            ):
                lado_tanque_90 = escolher_lado_tanque_90(destino_normal["validos_por_lado"])
                if lado_tanque_90 is not None:
                    iniciar_tanque_90(memoria, lado_tanque_90, agora)
            executando_retorno_verde = (
                args.verde_ativo
                and estado_verde_ativo["modo"] == "EXECUTANDO_VERDE"
                and estado_verde_ativo["decisao_confirmada"] == "RETORNO"
            )
            recuperando_linha_verde = (
                args.verde_ativo
                and estado_verde_ativo["modo"] == "RECUPERANDO_LINHA"
            )
            confirmando_verde = (
                args.verde_ativo
                and estado_verde_ativo["modo"] == "CONFIRMANDO_VERDE"
            )
            if executando_retorno_verde:
                deve_recuperar = False
                motivo_recuperacao = "VERDE_RETORNO_CEGO"
            else:
                if not destino["ok"]:
                    resetar_confirmacao_lado_recuperacao(memoria)
                if destino["ok"]:
                    memoria["frames_destino_confiavel"] += 1
                    memoria["frames_destino_perdido"] = 0
                else:
                    memoria["frames_destino_perdido"] += 1
                    memoria["frames_destino_confiavel"] = 0
                deve_recuperar, motivo_recuperacao = avaliar_recuperacao(
                    resultado,
                    destino,
                    memoria,
                )
            executando_lado_verde = (
                args.verde_ativo
                and estado_verde_ativo["modo"] == "EXECUTANDO_VERDE"
                and estado_verde_ativo["decisao_confirmada"] in ("ESQUERDA", "DIREITA")
                and destino.get("ok", False)
                and lado_destino(destino) == estado_verde_ativo["decisao_confirmada"]
            )
            if executando_retorno_verde:
                estado = "VERDE_RETORNO_CEGO"
                comando = comando_giro_retorno_verde(estado_verde_ativo, memoria)
            elif recuperando_linha_verde:
                estado = f"BUSCANDO_LINHA_{estado_verde_ativo['lado_busca_pos_verde']}"
                comando = comando_busca_linha_pos_verde(
                    estado_verde_ativo,
                    destino_normal,
                    memoria,
                )
            elif executando_lado_verde:
                estado = "FOLLOW_DESTINO"
                comando, _, _, _ = controlar_destino(destino, memoria, confirmar=False)
            elif memoria["tanque_90_ativo"]:
                estado = f"TANQUE_90_{memoria['tanque_90_lado']}"
                comando = comando_tanque_90(memoria)
                memoria["correcao_anterior"] = 0
                memoria["vel_anterior"] = (0, 0)
            elif confirmando_verde:
                estado = "CONFIRMANDO_VERDE"
                comando = aplicar_comando_confirmando_verde(destino_normal, memoria)
            elif memoria["em_recuperacao"] and destino["ok"]:
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

            if args.verde_ativo:
                comando = aplicar_comando_verde_ativo(
                    comando,
                    destino,
                    estado_verde_ativo,
                    memoria,
                )
            enviar_seguro(conexao, comando, args.motores)
            if args.log:
                if memoria["tanque_90_ativo"]:
                    decisao_log = f"TANQUE_90_{memoria['tanque_90_lado']}"
                elif args.verde_ativo:
                    decisao_log = log_estado_verde_ativo(estado_verde_ativo)
                else:
                    decisao_log = atualizar_decisao_log_verde(resultado_verde, estado_log, agora)
                imprimir_log_simples(decisao_log, estado_log, agora)
            else:
                imprimir_log(resultado, estado, destino, comando, memoria, motivo_recuperacao)
            if args.salvar_debug:
                agora = time.monotonic()
                if DEST_SALVAR_DEBUG_EVENTOS and (estado != estado_anterior or agora - ultimo_debug >= DEST_INTERVALO_DEBUG):
                    salvar_debug(criar_debug_destinos(resultado, destino, estado, comando, memoria, motivo_recuperacao), estado)
                    ultimo_debug = agora
            estado_anterior = estado
            stream_debug = None
            if args.mostrar:
                cv2.imshow("follow_destinos", criar_debug_destinos(resultado, destino, estado, comando, memoria, motivo_recuperacao))
            if args.stream_debug:
                stream_debug = criar_stream_debug(
                    frame,
                    resultado,
                    destino,
                    estado,
                    comando,
                    memoria,
                    motivo_recuperacao,
                    resultado_verde,
                    estado_verde_ativo if args.verde_ativo else None,
                )
                cv2.imshow("Shadow Stream Debug", stream_debug)
            if args.mostrar or args.stream_debug:
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    raise KeyboardInterrupt
                if tecla == ord("s") and stream_debug is not None:
                    caminho_stream = salvar_stream_debug(stream_debug)
                    if not args.log:
                        print(f"Stream debug salvo: {caminho_stream}")
            time.sleep(DEST_INTERVALO)
    except KeyboardInterrupt:
        if not args.log:
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
        if not args.log:
            print("Recursos liberados.")


if __name__ == "__main__":
    raise SystemExit(main())
