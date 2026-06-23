"""Segue-linha continuo por centroline conectada e lookahead."""

import argparse
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_test import iniciar_camera, capturar_frame_bgr
from line_test import detectar_linha, criar_debug_linha
from utils import abrir_serial, enviar_comando
from config import (
    BAUD_RATE,
    TIMEOUT_SERIAL,
    SERIAL_PORT,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    PASTA_CAPTURAS,

    RUN_CLEAN_INTERVALO,
    RUN_CLEAN_NUM_FAIXAS,
    RUN_CLEAN_MIN_PIXELS_FAIXA,
    RUN_CLEAN_ATUAL_Y_MIN,
    RUN_CLEAN_ATUAL_Y_MAX,
    RUN_CLEAN_ALVO_Y_MIN,
    RUN_CLEAN_ALVO_Y_MAX,
    RUN_CLEAN_MAX_SALTO_X,

    RUN_CLEAN_BASE_RETA,
    RUN_CLEAN_BASE_CURVA,
    RUN_CLEAN_BASE_EXTREMA,
    RUN_CLEAN_VEL_MIN_NORMAL,
    RUN_CLEAN_VEL_MIN_CURVA,
    RUN_CLEAN_VEL_MIN_EXTREMA,
    RUN_CLEAN_VEL_MAX,

    RUN_CLEAN_K_ATUAL,
    RUN_CLEAN_K_ALVO,
    RUN_CLEAN_K_DX,
    RUN_CLEAN_K_ATUAL_EXTREMA,
    RUN_CLEAN_K_ALVO_EXTREMA,
    RUN_CLEAN_K_DX_EXTREMA,

    RUN_CLEAN_CORRECAO_MAX_NORMAL,
    RUN_CLEAN_CORRECAO_MAX_CURVA,
    RUN_CLEAN_CORRECAO_MAX_EXTREMA,

    RUN_CLEAN_DX_CURVA,
    RUN_CLEAN_DX_EXTREMA,
    RUN_CLEAN_ANGULO_CURVA,
    RUN_CLEAN_ANGULO_EXTREMA,
    RUN_CLEAN_COTOVELO_DELTA_ANGULO,
    RUN_CLEAN_COTOVELO_DX_MIN,
    RUN_CLEAN_COTOVELO_MIN_PONTOS,
    RUN_CLEAN_EXTREMA_DX_COM_ANGULO,
    RUN_CLEAN_EXTREMA_ANGULO_COM_DX,
    RUN_CLEAN_EXTREMA_INTERNA_INICIAL,
    RUN_CLEAN_EXTREMA_INTERNA_MAX_NEG,
    RUN_CLEAN_EXTREMA_EXTERNA_INICIAL,
    RUN_CLEAN_EXTREMA_EXTERNA_MAX,
    RUN_CLEAN_EXTREMA_DX_FULL,
    RUN_CLEAN_EXTREMA_ANGULO_FULL,

    RUN_CLEAN_ALPHA_SUAVIZACAO,
    RUN_CLEAN_DELTA_MAX_VELOCIDADE,
    RUN_CLEAN_DELTA_MAX_EXTREMA,

    RUN_CLEAN_VEL_RECUPERAR,
    RUN_CLEAN_VEL_VARREDURA,
    RUN_CLEAN_TEMPO_TROCA_VARREDURA,

    RUN_CLEAN_HISTORICO_LADO,
    RUN_CLEAN_DX_LADO_MIN,
    RUN_CLEAN_ERRO_LADO_MIN,

    RUN_CLEAN_SALVAR_DEBUG_EVENTOS,
    RUN_CLEAN_INTERVALO_DEBUG,
)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha limpo com lookahead e LADO assinado.")
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


def extrair_centro_faixa(mascara, y1, y2, x_inicio_roi, y_inicio_roi, centro_imagem_x, x_referencia=None):
    """Retorna o centro global do melhor contorno da faixa horizontal."""
    faixa = mascara[y1:y2, :]
    contornos, _ = cv2.findContours(faixa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < RUN_CLEAN_MIN_PIXELS_FAIXA:
            continue
        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        centro_x_local = momentos["m10"] / momentos["m00"]
        centro_y_local = momentos["m01"] / momentos["m00"]
        x_global = x_inicio_roi + centro_x_local
        y_global = y_inicio_roi + y1 + centro_y_local
        candidatos.append({
            "encontrou": True,
            "x": x_global,
            "y": y_global,
            "erro": x_global - centro_imagem_x,
            "pixels": area,
        })

    if not candidatos:
        return {"encontrou": False}

    referencia = centro_imagem_x if x_referencia is None else x_referencia
    return min(candidatos, key=lambda ponto: abs(ponto["x"] - referencia))


def normalizar_delta_angulo(delta):
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360
    return delta


def detectar_cotovelo(guia):
    """Detecta uma quebra acentuada na direcao da centroline conectada."""
    if not guia["ok"]:
        return False
    pontos = guia["pontos"]
    if len(pontos) < RUN_CLEAN_COTOVELO_MIN_PONTOS:
        return False

    ponto_baixo = guia["ponto_atual"]
    ponto_alvo = guia["ponto_alvo"]
    ponto_medio = pontos[len(pontos) // 2]
    dx1 = ponto_medio["x"] - ponto_baixo["x"]
    dy1 = ponto_baixo["y"] - ponto_medio["y"]
    dx2 = ponto_alvo["x"] - ponto_medio["x"]
    dy2 = ponto_medio["y"] - ponto_alvo["y"]
    ang1 = math.degrees(math.atan2(dx1, max(dy1, 1)))
    ang2 = math.degrees(math.atan2(dx2, max(dy2, 1)))
    delta = abs(normalizar_delta_angulo(ang2 - ang1))
    dx_total = abs(ponto_alvo["x"] - ponto_baixo["x"])
    return delta >= RUN_CLEAN_COTOVELO_DELTA_ANGULO and dx_total >= RUN_CLEAN_COTOVELO_DX_MIN


def extrair_guia_linha(resultado):
    """Monta uma centroline conectada e retorna seus pontos de controle."""
    mascara = resultado["mascara_limpa"]
    x_inicio_roi = resultado["x_inicio_roi"]
    y_inicio_roi = resultado["y_inicio_roi"]
    centro_imagem_x = resultado["centro_imagem_x"]
    altura_roi = mascara.shape[0]
    limites = np.linspace(0, altura_roi, RUN_CLEAN_NUM_FAIXAS + 1, dtype=int)

    pontos = []
    x_referencia = None
    for indice in range(RUN_CLEAN_NUM_FAIXAS - 1, -1, -1):
        y1, y2 = limites[indice], limites[indice + 1]
        if y2 <= y1:
            continue
        ponto = extrair_centro_faixa(
            mascara, y1, y2, x_inicio_roi, y_inicio_roi, centro_imagem_x, x_referencia,
        )
        if not ponto["encontrou"]:
            continue
        if x_referencia is not None and abs(ponto["x"] - x_referencia) > RUN_CLEAN_MAX_SALTO_X:
            continue
        pontos.append(ponto)
        x_referencia = ponto["x"]

    if not pontos:
        return {"ok": False, "pontos": pontos, "baixa": "PERDIDA", "lado": "CENTRO"}

    def dentro_regiao(ponto, minimo, maximo):
        y_relativo = (ponto["y"] - y_inicio_roi) / max(altura_roi, 1)
        return minimo <= y_relativo <= maximo

    pontos_atuais = [ponto for ponto in pontos if dentro_regiao(ponto, RUN_CLEAN_ATUAL_Y_MIN, RUN_CLEAN_ATUAL_Y_MAX)]
    if pontos_atuais:
        ponto_atual = max(pontos_atuais, key=lambda ponto: ponto["y"])
        baixa = "SEGURA"
    else:
        ponto_atual = max(pontos, key=lambda ponto: ponto["y"])
        baixa = "FRACA"

    pontos_alvo = [ponto for ponto in pontos if dentro_regiao(ponto, RUN_CLEAN_ALVO_Y_MIN, RUN_CLEAN_ALVO_Y_MAX)]
    if pontos_alvo:
        meio_alvo = (RUN_CLEAN_ALVO_Y_MIN + RUN_CLEAN_ALVO_Y_MAX) / 2
        ponto_alvo = min(
            pontos_alvo,
            key=lambda ponto: abs(((ponto["y"] - y_inicio_roi) / max(altura_roi, 1)) - meio_alvo),
        )
    elif len(pontos) >= 2:
        ponto_alvo = min(pontos, key=lambda ponto: ponto["y"])
    else:
        return {"ok": False, "pontos": pontos, "baixa": "PERDIDA", "lado": "CENTRO"}

    if ponto_alvo is ponto_atual:
        return {"ok": False, "pontos": pontos, "baixa": "PERDIDA", "lado": "CENTRO"}

    dx = ponto_alvo["x"] - ponto_atual["x"]
    dy = ponto_atual["y"] - ponto_alvo["y"]
    angulo = math.degrees(math.atan2(dx, max(dy, 1)))
    if dx < -RUN_CLEAN_DX_LADO_MIN:
        lado = "ESQUERDA"
    elif dx > RUN_CLEAN_DX_LADO_MIN:
        lado = "DIREITA"
    else:
        lado = "CENTRO"
    guia = {
        "ok": True,
        "pontos": pontos,
        "ponto_atual": ponto_atual,
        "ponto_alvo": ponto_alvo,
        "erro_atual": ponto_atual["erro"],
        "erro_alvo": ponto_alvo["erro"],
        "dx": dx,
        "dy": dy,
        "angulo": angulo,
        "lado": lado,
        "baixa": baixa,
    }
    guia["cotovelo"] = detectar_cotovelo(guia)
    return guia


def classificar_modo(guia):
    if not guia["ok"]:
        return "RECUPERAR"
    if guia.get("baixa") != "SEGURA":
        return "RECUPERAR"

    dx_abs = abs(guia["dx"])
    ang_abs = abs(guia["angulo"])
    if guia.get("cotovelo", False):
        return "FOLLOW_EXTREMA"
    if dx_abs >= RUN_CLEAN_EXTREMA_DX_COM_ANGULO and ang_abs >= RUN_CLEAN_EXTREMA_ANGULO_COM_DX:
        return "FOLLOW_EXTREMA"
    if dx_abs >= RUN_CLEAN_DX_EXTREMA and ang_abs >= RUN_CLEAN_ANGULO_EXTREMA:
        return "FOLLOW_EXTREMA"
    if dx_abs >= RUN_CLEAN_DX_CURVA or ang_abs >= RUN_CLEAN_ANGULO_CURVA:
        return "FOLLOW_CURVA"
    return "FOLLOW_NORMAL"


def calcular_intensidade_extrema(guia):
    """Retorna de 0 a 1 conforme a intensidade da curva extrema."""
    intensidade_dx = abs(guia["dx"]) / max(RUN_CLEAN_EXTREMA_DX_FULL, 1)
    intensidade_ang = abs(guia["angulo"]) / max(RUN_CLEAN_EXTREMA_ANGULO_FULL, 1)
    return limitar(max(intensidade_dx, intensidade_ang), 0.0, 1.0)


def calcular_comando_extrema(guia, memoria):
    """Calcula LADO assinado explicito para uma curva extrema visivel."""
    intensidade = calcular_intensidade_extrema(guia)
    interna = RUN_CLEAN_EXTREMA_INTERNA_INICIAL + intensidade * (
        RUN_CLEAN_EXTREMA_INTERNA_MAX_NEG - RUN_CLEAN_EXTREMA_INTERNA_INICIAL
    )
    externa = RUN_CLEAN_EXTREMA_EXTERNA_INICIAL + intensidade * (
        RUN_CLEAN_EXTREMA_EXTERNA_MAX - RUN_CLEAN_EXTREMA_EXTERNA_INICIAL
    )
    interna = limitar(interna, RUN_CLEAN_EXTREMA_INTERNA_MAX_NEG, 0)
    externa = limitar(externa, 0, RUN_CLEAN_EXTREMA_EXTERNA_MAX)
    if guia["dx"] < 0:
        vel_esq, vel_dir = interna, externa
    elif guia["dx"] > 0:
        vel_esq, vel_dir = externa, interna
    else:
        vel_esq = RUN_CLEAN_BASE_EXTREMA
        vel_dir = RUN_CLEAN_BASE_EXTREMA

    vel_esq_ant, vel_dir_ant = memoria["vel_anterior"]
    vel_esq = limitar(vel_esq, vel_esq_ant - RUN_CLEAN_DELTA_MAX_EXTREMA, vel_esq_ant + RUN_CLEAN_DELTA_MAX_EXTREMA)
    vel_dir = limitar(vel_dir, vel_dir_ant - RUN_CLEAN_DELTA_MAX_EXTREMA, vel_dir_ant + RUN_CLEAN_DELTA_MAX_EXTREMA)
    memoria["correcao_anterior"] = 0
    memoria["vel_anterior"] = (vel_esq, vel_dir)
    return f"LADO {round(vel_esq)} {round(vel_dir)}", vel_esq, vel_dir, intensidade


def calcular_comando_follow(guia, modo, memoria):
    """Retorna comando LADO, velocidades e correcao suavizada."""
    if modo == "FOLLOW_EXTREMA":
        return calcular_comando_extrema(guia, memoria)

    erro_atual = guia["erro_atual"]
    erro_alvo = guia["erro_alvo"]
    dx = guia["dx"]
    if modo == "FOLLOW_NORMAL":
        base, vel_min, correcao_max = RUN_CLEAN_BASE_RETA, RUN_CLEAN_VEL_MIN_NORMAL, RUN_CLEAN_CORRECAO_MAX_NORMAL
        correcao = RUN_CLEAN_K_ATUAL * erro_atual + RUN_CLEAN_K_ALVO * erro_alvo + RUN_CLEAN_K_DX * dx
    elif modo == "FOLLOW_CURVA":
        base, vel_min, correcao_max = RUN_CLEAN_BASE_CURVA, RUN_CLEAN_VEL_MIN_CURVA, RUN_CLEAN_CORRECAO_MAX_CURVA
        correcao = RUN_CLEAN_K_ATUAL * erro_atual + RUN_CLEAN_K_ALVO * erro_alvo + RUN_CLEAN_K_DX * dx
    else:
        raise ValueError(f"Modo de follow invalido: {modo}")

    alpha = RUN_CLEAN_ALPHA_SUAVIZACAO
    correcao_suave = alpha * memoria["correcao_anterior"] + (1 - alpha) * correcao
    correcao_suave = limitar(correcao_suave, -correcao_max, correcao_max)
    vel_esq = limitar(base + correcao_suave, vel_min, RUN_CLEAN_VEL_MAX)
    vel_dir = limitar(base - correcao_suave, vel_min, RUN_CLEAN_VEL_MAX)
    delta_max = RUN_CLEAN_DELTA_MAX_VELOCIDADE
    vel_esq_ant, vel_dir_ant = memoria["vel_anterior"]
    vel_esq = limitar(vel_esq, vel_esq_ant - delta_max, vel_esq_ant + delta_max)
    vel_dir = limitar(vel_dir, vel_dir_ant - delta_max, vel_dir_ant + delta_max)
    memoria["correcao_anterior"] = correcao_suave
    memoria["vel_anterior"] = (vel_esq, vel_dir)
    return f"LADO {round(vel_esq)} {round(vel_dir)}", vel_esq, vel_dir, correcao_suave


def atualizar_lado_valido(historico_lados, guia):
    if not guia["ok"]:
        return
    if guia["lado"] in ("ESQUERDA", "DIREITA"):
        historico_lados.append(guia["lado"])
    elif guia["erro_atual"] < -RUN_CLEAN_ERRO_LADO_MIN:
        historico_lados.append("ESQUERDA")
    elif guia["erro_atual"] > RUN_CLEAN_ERRO_LADO_MIN:
        historico_lados.append("DIREITA")


def comando_recuperacao(ultimo_lado, estado_recuperacao, controle_varredura):
    if ultimo_lado == "ESQUERDA":
        return f"GIRAR_ESQ {RUN_CLEAN_VEL_RECUPERAR}"
    if ultimo_lado == "DIREITA":
        return f"GIRAR_DIR {RUN_CLEAN_VEL_RECUPERAR}"

    agora = time.monotonic()
    if agora - controle_varredura["ultima_troca"] >= RUN_CLEAN_TEMPO_TROCA_VARREDURA:
        controle_varredura["etapa"] = (controle_varredura["etapa"] + 1) % 4
        controle_varredura["ultima_troca"] = agora
    sequencia = ["ESQUERDA", "DIREITA", "DIREITA", "ESQUERDA"]
    lado = sequencia[controle_varredura["etapa"]]
    comando = "GIRAR_ESQ" if lado == "ESQUERDA" else "GIRAR_DIR"
    return f"{comando} {RUN_CLEAN_VEL_VARREDURA}"


def criar_debug_run(resultado, guia, estado, comando, ultimo_lado):
    debug = criar_debug_linha(resultado)
    if guia["ok"]:
        for ponto in guia["pontos"]:
            cv2.circle(debug, (round(ponto["x"]), round(ponto["y"])), 3, (255, 0, 0), -1)
        atual = guia["ponto_atual"]
        alvo = guia["ponto_alvo"]
        ponto_atual = (round(atual["x"]), round(atual["y"]))
        ponto_alvo = (round(alvo["x"]), round(alvo["y"]))
        cv2.line(debug, ponto_atual, ponto_alvo, (0, 255, 255), 2)
        cv2.circle(debug, ponto_atual, 7, (0, 0, 255), -1)
        cv2.circle(debug, ponto_alvo, 7, (255, 0, 255), -1)
        linhas = [
            f"Estado: {estado}", f"Cmd: {comando}", f"Baixa: {guia['baixa']}",
            f"Atual: {atual['erro']:.1f}", f"Alvo: {alvo['erro']:.1f}",
            f"dx: {guia['dx']:.1f}", f"angulo: {guia['angulo']:.1f}",
            f"Lado: {guia['lado']}", f"Cotovelo: {guia.get('cotovelo', False)}", f"Ultimo lado: {ultimo_lado}",
        ]
    else:
        linhas = [f"Estado: {estado}", f"Cmd: {comando}", "Baixa: PERDIDA", f"Ultimo lado: {ultimo_lado}"]
    for indice, texto in enumerate(linhas):
        cv2.putText(debug, texto, (20, 65 + indice * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(debug, texto, (20, 65 + indice * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return debug


def salvar_debug_evento(debug, estado):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = pasta / f"debug_run_clean_{timestamp}_{estado}.jpg"
    if not cv2.imwrite(str(caminho), debug):
        raise RuntimeError(f"Nao foi possivel salvar debug: {caminho}")
    return caminho


def imprimir_log(guia, estado, comando, ultimo_lado, intensidade=None):
    if guia["ok"]:
        linha = (
            f"Estado: {estado} | Cmd: {comando} | Atual: {guia['erro_atual']:.0f} | "
            f"Alvo: {guia['erro_alvo']:.0f} | dx: {guia['dx']:.0f} | ang: {guia['angulo']:.1f} | "
            f"lado: {guia['lado']} | cotovelo: {guia.get('cotovelo', False)}"
        )
        if estado == "FOLLOW_EXTREMA" and intensidade is not None:
            linha += f" | intensidade: {intensidade:.2f}"
        print(f"{linha} | baixa: {guia['baixa']} | ultimo: {ultimo_lado}")
    else:
        print(f"Estado: {estado} | Cmd: {comando} | linha: PERDIDA | ultimo: {ultimo_lado}")


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
            print("Motores ativados. Iniciando segue-linha agora.")
        else:
            print("Simulacao sem motores.")

        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
        historico_lados = deque(maxlen=RUN_CLEAN_HISTORICO_LADO)
        memoria = {"correcao_anterior": 0, "vel_anterior": (0, 0)}
        estado_anterior = None
        ultimo_debug = 0
        controle_varredura = {"etapa": 0, "ultima_troca": time.monotonic()}

        while True:
            frame = capturar_frame_bgr(camera)
            resultado = detectar_linha(frame)
            guia = extrair_guia_linha(resultado)
            modo = classificar_modo(guia)
            if resultado["encontrou_linha"] and guia["ok"] and modo != "RECUPERAR":
                atualizar_lado_valido(historico_lados, guia)
                ultimo_lado = historico_lados[-1] if historico_lados else "CENTRO"
                comando, _, _, intensidade = calcular_comando_follow(guia, modo, memoria)
                estado = modo
            else:
                ultimo_lado = historico_lados[-1] if historico_lados else "CENTRO"
                estado = "RECUPERAR_GIRO" if ultimo_lado in ("ESQUERDA", "DIREITA") else "RECUPERAR_VARREDURA"
                comando = comando_recuperacao(ultimo_lado, estado, controle_varredura)
                intensidade = None
                memoria["correcao_anterior"] = 0
                memoria["vel_anterior"] = (0, 0)

            enviar_seguro(conexao, comando, args.motores)
            imprimir_log(guia, estado, comando, ultimo_lado, intensidade)
            if args.salvar_debug:
                agora = time.monotonic()
                if estado != estado_anterior or agora - ultimo_debug >= RUN_CLEAN_INTERVALO_DEBUG:
                    salvar_debug_evento(criar_debug_run(resultado, guia, estado, comando, ultimo_lado), estado)
                    ultimo_debug = agora
            estado_anterior = estado
            if args.mostrar:
                cv2.imshow("follow_clean", criar_debug_run(resultado, guia, estado, comando, ultimo_lado))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            time.sleep(RUN_CLEAN_INTERVALO)
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
