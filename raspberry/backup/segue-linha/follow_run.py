"""Segue-linha continuo com Tank Assist para recuperar curvas e perda da linha."""

import argparse
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    BAUD_RATE, CAMERA_HEIGHT, CAMERA_WIDTH, PASTA_CAPTURAS, RUN_CORRECAO_MAXIMA,
    RUN_CORRECAO_MAXIMA_CURVA_FORTE, RUN_CORRECAO_MAXIMA_REALINHAR,
    RUN_FRAMES_CORRECAO_SATURADA, RUN_FRAMES_REENCONTRO_BAIXA,
    RUN_FRAMES_RISCO_TANK, RUN_INTERVALO_LOOP, RUN_KP_CURVA_FORTE_DIRECAO,
    RUN_KP_CURVA_FORTE_LATERAL, RUN_KP_DIRECAO, RUN_KP_LATERAL,
    RUN_KP_REALINHAR, RUN_LIMIAR_CURVA_FORTE, RUN_LIMIAR_ERRO_BAIXA_RISCO,
    RUN_LIMIAR_BAIXA_LADO_CONFIAVEL, RUN_LIMIAR_CORRECAO_LADO_CONFIAVEL,
    RUN_LIMIAR_DIRECAO_LADO_CONFIAVEL, RUN_LIMIAR_DIRECAO_SAIDA_TANK,
    RUN_LIMIAR_ERRO_SAIDA_TANK,
    RUN_MIN_PIXELS_BAIXA_FRACA, RUN_MIN_PIXELS_BAIXA_SEGURA,
    RUN_SALVAR_DEBUG_EVENTOS, RUN_TAMANHO_HISTORICO_LADO,
    RUN_TEMPO_MAX_TANK, RUN_TEMPO_PULSO_BUSCA, RUN_TEMPO_PULSO_TANK,
    RUN_TEMPO_PULSO_VARREDURA, RUN_TEMPO_REALINHAR, RUN_TANK_ASSIST_IMEDIATO,
    RUN_VELOCIDADE_BASE,
    RUN_VELOCIDADE_BUSCA, RUN_VELOCIDADE_CURVA_FORTE, RUN_VELOCIDADE_MAXIMA,
    RUN_VELOCIDADE_MINIMA, RUN_VELOCIDADE_REALINHAR, RUN_VELOCIDADE_TANK,
    RUN_VELOCIDADE_VARREDURA, RUN_Y_BAIXA_FIM, RUN_Y_BAIXA_INICIO,
    RUN_ANGULO_SAIDA_TANK, RUN_ANGULO_TANK_ASSIST, RUN_DISTANCIA_MAX_CONEXAO_X,
    RUN_DX_CURVA_FORTE, RUN_DX_SAIDA_TANK, RUN_DX_TANK_ASSIST,
    RUN_KP_ALVO, RUN_KP_ALVO_CURVA_FORTE, RUN_KP_ATUAL,
    RUN_KP_ATUAL_CURVA_FORTE, RUN_KP_DX, RUN_KP_DX_CURVA_FORTE,
    RUN_MIN_PIXELS_PONTO_GUIA, RUN_NUM_FAIXAS_GUIA, RUN_PONTO_ALVO_Y_MAX,
    RUN_PONTO_ALVO_Y_MIN, RUN_PONTO_ATUAL_Y_MAX, RUN_PONTO_ATUAL_Y_MIN,
    SERIAL_PORT, TIMEOUT_SERIAL,
)
from line_test import criar_debug_linha, detectar_linha
from utils import abrir_serial, enviar_comando


SALVAR_DEBUG_ATIVO = False


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha continuo com Tank Assist.")
    parser.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    parser.add_argument("--motores", action="store_true", help="Envia comandos ao Arduino.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou 'auto'.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva debug nas mudancas de estado.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra o debug da linha.")
    return parser.parse_args()


def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


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


def medir_baixa(mascara_limpa, x_inicio_roi, centro_imagem_x):
    altura = mascara_limpa.shape[0]
    y1 = int(altura * RUN_Y_BAIXA_INICIO)
    y2 = int(altura * RUN_Y_BAIXA_FIM)
    faixa = mascara_limpa[y1:y2, :]
    pixels = cv2.countNonZero(faixa)
    if not pixels:
        return {"encontrou": False, "pixels": 0, "erro": None, "segura": False,
                "fraca": False, "risco": True}
    momentos = cv2.moments(faixa, binaryImage=True)
    centro_x = x_inicio_roi + int(momentos["m10"] / momentos["m00"])
    erro = centro_x - centro_imagem_x
    segura = pixels >= RUN_MIN_PIXELS_BAIXA_SEGURA
    fraca = RUN_MIN_PIXELS_BAIXA_FRACA <= pixels < RUN_MIN_PIXELS_BAIXA_SEGURA
    return {"encontrou": True, "pixels": pixels, "erro": erro, "segura": segura,
            "fraca": fraca, "risco": not segura or abs(erro) >= RUN_LIMIAR_ERRO_BAIXA_RISCO}


def extrair_pontos_guia(mascara_limpa, x_inicio_roi, y_inicio_roi, centro_imagem_x):
    """Segue a mesma linha da base para cima e retorna ponto atual e lookahead."""
    altura, largura = mascara_limpa.shape[:2]
    conectados, x_anterior = [], None
    for indice in range(RUN_NUM_FAIXAS_GUIA):
        y2 = altura - indice * altura // RUN_NUM_FAIXAS_GUIA
        y1 = altura - (indice + 1) * altura // RUN_NUM_FAIXAS_GUIA
        faixa = mascara_limpa[y1:y2, :]
        contornos, _ = cv2.findContours(faixa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        for contorno in contornos:
            if cv2.contourArea(contorno) < RUN_MIN_PIXELS_PONTO_GUIA:
                continue
            momentos = cv2.moments(contorno)
            if momentos["m00"]:
                candidatos.append((int(momentos["m10"] / momentos["m00"]), cv2.contourArea(contorno)))
        if not candidatos:
            continue
        escolhido = max(candidatos, key=lambda candidato: candidato[1]) if x_anterior is None else min(candidatos, key=lambda candidato: abs(candidato[0] - x_anterior))
        if x_anterior is not None and abs(escolhido[0] - x_anterior) > RUN_DISTANCIA_MAX_CONEXAO_X:
            continue
        x_anterior = escolhido[0]
        conectados.append({"x": x_inicio_roi + escolhido[0], "y": y_inicio_roi + (y1 + y2) // 2,
                           "erro": x_inicio_roi + escolhido[0] - centro_imagem_x, "pixels": int(escolhido[1]),
                           "fracao_y": (y1 + y2) / (2 * altura)})

    atuais = [ponto for ponto in conectados if RUN_PONTO_ATUAL_Y_MIN <= ponto["fracao_y"] <= RUN_PONTO_ATUAL_Y_MAX]
    alvos = [ponto for ponto in conectados if RUN_PONTO_ALVO_Y_MIN <= ponto["fracao_y"] <= RUN_PONTO_ALVO_Y_MAX]
    if not atuais or not alvos:
        return {"ok": False, "ponto_atual": None, "ponto_alvo": None, "dx": None, "dy": None,
                "angulo": None, "lado_curva": "CENTRO"}
    atual = min(atuais, key=lambda ponto: abs(ponto["fracao_y"] - (RUN_PONTO_ATUAL_Y_MIN + RUN_PONTO_ATUAL_Y_MAX) / 2))
    alvo = min(alvos, key=lambda ponto: abs(ponto["fracao_y"] - (RUN_PONTO_ALVO_Y_MIN + RUN_PONTO_ALVO_Y_MAX) / 2))
    dx, dy = alvo["x"] - atual["x"], alvo["y"] - atual["y"]
    angulo = math.degrees(math.atan2(abs(dx), max(1, abs(dy))))
    lado = "ESQUERDA" if dx < -RUN_DX_CURVA_FORTE else "DIREITA" if dx > RUN_DX_CURVA_FORTE else "CENTRO"
    return {"ok": True, "ponto_atual": atual, "ponto_alvo": alvo, "dx": dx, "dy": dy,
            "angulo": angulo, "lado_curva": lado}


def escolher_lado_busca(historico_lados, ultimo_lado_confiavel):
    lados = [lado for lado in historico_lados if lado != "CENTRO"]
    if lados:
        return Counter(lados).most_common(1)[0][0]
    return ultimo_lado_confiavel if ultimo_lado_confiavel in ("ESQUERDA", "DIREITA") else "CENTRO"


def comando_tank(lado, velocidade):
    if lado == "ESQUERDA":
        return f"GIRAR_ESQ {velocidade}"
    if lado == "DIREITA":
        return f"GIRAR_DIR {velocidade}"
    return "PARAR"


def comando_por_guia(guia, modo):
    """Calcula LADO a partir dos erros das duas bolinhas e do deslocamento dx."""
    parametros = {
        "NORMAL": (RUN_VELOCIDADE_BASE, RUN_KP_ATUAL, RUN_KP_ALVO, RUN_KP_DX, RUN_CORRECAO_MAXIMA),
        "CURVA_FORTE": (RUN_VELOCIDADE_CURVA_FORTE, RUN_KP_ATUAL_CURVA_FORTE, RUN_KP_ALVO_CURVA_FORTE, RUN_KP_DX_CURVA_FORTE, RUN_CORRECAO_MAXIMA_CURVA_FORTE),
        "REALINHAR": (RUN_VELOCIDADE_REALINHAR, RUN_KP_REALINHAR, RUN_KP_REALINHAR, 0.20, RUN_CORRECAO_MAXIMA_REALINHAR),
    }
    base, kp_atual, kp_alvo, kp_dx, maximo = parametros[modo]
    correcao = limitar(kp_atual * guia["ponto_atual"]["erro"] + kp_alvo * guia["ponto_alvo"]["erro"] + kp_dx * guia["dx"], -maximo, maximo)
    esquerda = round(limitar(base + correcao, RUN_VELOCIDADE_MINIMA, RUN_VELOCIDADE_MAXIMA))
    direita = round(limitar(base - correcao, RUN_VELOCIDADE_MINIMA, RUN_VELOCIDADE_MAXIMA))
    return f"LADO {esquerda} {direita}", correcao


def desenhar_guia(debug, guia):
    if guia and guia["ok"]:
        atual, alvo = guia["ponto_atual"], guia["ponto_alvo"]
        cv2.line(debug, (atual["x"], atual["y"]), (alvo["x"], alvo["y"]), (0, 255, 255), 2)
        cv2.circle(debug, (atual["x"], atual["y"]), 7, (0, 0, 255), -1)
        cv2.circle(debug, (alvo["x"], alvo["y"]), 7, (255, 0, 255), -1)
    return debug


def salvar_debug_evento(resultado, estado, comando, info_extra):
    if not (SALVAR_DEBUG_ATIVO and RUN_SALVAR_DEBUG_EVENTOS):
        return
    guia = info_extra.get("Guia")
    debug = desenhar_guia(criar_debug_linha(resultado), guia)
    linhas = [f"Estado: {estado}", f"Cmd: {comando}"]
    linhas.extend(f"{chave}: {valor}" for chave, valor in info_extra.items() if chave != "Guia")
    for indice, texto in enumerate(linhas):
        cv2.putText(debug, texto, (15, 65 + indice * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_run_stable_{datetime.now():%Y%m%d_%H%M%S_%f}_{estado}.jpg"
    if not cv2.imwrite(str(caminho), debug):
        raise RuntimeError(f"Nao foi possivel salvar debug: {caminho}")
    print(f"Debug salvo: {caminho}")


def main():
    global SALVAR_DEBUG_ATIVO
    args = ler_argumentos()
    SALVAR_DEBUG_ATIVO = args.salvar_debug
    camera = conexao = None
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

        estado, estado_anterior = "NORMAL", None
        historico_lados = deque(maxlen=RUN_TAMANHO_HISTORICO_LADO)
        ultimo_lado_confiavel = "CENTRO"
        lado_tank_ativo, origem_lado_tank = "CENTRO", "INDEFINIDO"
        frames_risco = frames_correcao_saturada = frames_reencontro = 0
        tempo_estado = time.monotonic()
        etapa_varredura = 0

        while True:
            resultado = detectar_linha(capturar_frame_bgr(camera))
            baixa = medir_baixa(resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["centro_imagem_x"])
            guia = extrair_pontos_guia(resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["y_inicio_roi"], resultado["centro_imagem_x"])
            if guia["ok"] and guia["lado_curva"] != "CENTRO":
                ultimo_lado_confiavel = guia["lado_curva"]
                historico_lados.append(ultimo_lado_confiavel)

            motivo, comando, correcao = "", "PARAR", 0
            if estado in ("TANK_ASSIST", "BUSCA_TANQUE", "VARREDURA"):
                baixa_reencontrada = (
                    baixa["segura"]
                    and guia["ok"]
                    and abs(guia["dx"]) <= RUN_DX_SAIDA_TANK
                    and abs(guia["angulo"]) <= RUN_ANGULO_SAIDA_TANK
                )
                frames_reencontro = frames_reencontro + 1 if baixa_reencontrada else 0
                if frames_reencontro >= RUN_FRAMES_REENCONTRO_BAIXA:
                    estado, tempo_estado, frames_reencontro = "REALINHAR", time.monotonic(), 0
                    frames_risco = frames_correcao_saturada = 0
                elif estado == "TANK_ASSIST" and time.monotonic() - tempo_estado >= RUN_TEMPO_MAX_TANK:
                    estado, tempo_estado = "BUSCA_TANQUE", time.monotonic()
                elif estado == "TANK_ASSIST":
                    comando, motivo = comando_tank(lado_tank_ativo, RUN_VELOCIDADE_TANK), f"risco_{origem_lado_tank}"
                    if comando == "PARAR":
                        estado, tempo_estado = "VARREDURA", time.monotonic()
                    else:
                        enviar_seguro(conexao, comando, args.motores); time.sleep(RUN_TEMPO_PULSO_TANK)
                        enviar_seguro(conexao, "PARAR", args.motores)
                if estado == "BUSCA_TANQUE":
                    comando = comando_tank(lado_tank_ativo, RUN_VELOCIDADE_BUSCA)
                    if comando == "PARAR":
                        estado, tempo_estado = "VARREDURA", time.monotonic()
                    else:
                        enviar_seguro(conexao, comando, args.motores); time.sleep(RUN_TEMPO_PULSO_BUSCA)
                        enviar_seguro(conexao, "PARAR", args.motores)
                if estado == "VARREDURA":
                    sequencia = ("ESQUERDA", "DIREITA", "DIREITA", "ESQUERDA")
                    comando = comando_tank(sequencia[etapa_varredura], RUN_VELOCIDADE_VARREDURA)
                    enviar_seguro(conexao, comando, args.motores); time.sleep(RUN_TEMPO_PULSO_VARREDURA)
                    enviar_seguro(conexao, "PARAR", args.motores)
                    etapa_varredura = (etapa_varredura + 1) % len(sequencia)
            elif not resultado["encontrou_linha"] or not guia["ok"]:
                lado_tank_ativo = escolher_lado_busca(historico_lados, ultimo_lado_confiavel)
                origem_lado_tank = "HISTORICO"
                tempo_estado = time.monotonic()
                if lado_tank_ativo == "CENTRO":
                    estado, comando, motivo = "VARREDURA", "PARAR", "linha_perdida_sem_lado"
                else:
                    estado = "BUSCA_TANQUE"
                    comando = comando_tank(lado_tank_ativo, RUN_VELOCIDADE_BUSCA)
                    motivo = f"linha_perdida_lado_{lado_tank_ativo}"
                    if args.motores:
                        enviar_seguro(conexao, comando, args.motores)
                        time.sleep(RUN_TEMPO_PULSO_BUSCA)
                        enviar_seguro(conexao, "PARAR", args.motores)
            else:
                comando_normal, correcao_normal = comando_por_guia(guia, "NORMAL")
                frames_risco = frames_risco + 1 if baixa["risco"] else 0
                frames_correcao_saturada = frames_correcao_saturada + 1 if abs(correcao_normal) >= RUN_CORRECAO_SATURADA_TANK else 0
                if estado == "REALINHAR" and time.monotonic() - tempo_estado < RUN_TEMPO_REALINHAR:
                    comando, correcao = comando_por_guia(guia, "REALINHAR")
                elif (abs(guia["dx"]) >= RUN_DX_TANK_ASSIST
                      or guia["angulo"] >= RUN_ANGULO_TANK_ASSIST
                      or (baixa["fraca"] and abs(guia["dx"]) >= RUN_DX_CURVA_FORTE)
                      or frames_risco >= RUN_FRAMES_RISCO_TANK
                      or frames_correcao_saturada >= RUN_FRAMES_CORRECAO_SATURADA):
                    lado_tank_ativo = guia["lado_curva"]
                    origem_lado_tank = "GUIA_DX"
                    tempo_estado = time.monotonic()
                    if lado_tank_ativo == "CENTRO":
                        estado, comando, motivo = "VARREDURA", "PARAR", "risco_sem_lado"
                    else:
                        estado = "TANK_ASSIST"
                        comando = comando_tank(lado_tank_ativo, RUN_VELOCIDADE_TANK)
                        motivo = f"risco_lado_{lado_tank_ativo}_origem_{origem_lado_tank}"
                        if args.motores and RUN_TANK_ASSIST_IMEDIATO:
                            enviar_seguro(conexao, comando, args.motores)
                            time.sleep(RUN_TEMPO_PULSO_TANK)
                            enviar_seguro(conexao, "PARAR", args.motores)
                elif abs(guia["dx"]) >= RUN_DX_CURVA_FORTE:
                    estado, comando, correcao = "CURVA_FORTE", *comando_por_guia(guia, "CURVA_FORTE")
                else:
                    estado, comando, correcao = "NORMAL", comando_normal, correcao_normal

            if estado != estado_anterior:
                salvar_debug_evento(resultado, estado, comando, {"Baixa": "SEGURA" if baixa["segura"] else "FRACA" if baixa["fraca"] else "PERDIDA", "Atual": guia["ponto_atual"]["erro"] if guia["ok"] else None, "Alvo": guia["ponto_alvo"]["erro"] if guia["ok"] else None, "dx": guia["dx"], "angulo": guia["angulo"], "Lado": guia["lado_curva"], "Guia": guia})
                estado_anterior = estado
            if args.mostrar:
                cv2.imshow("Segue-linha - q para parar", desenhar_guia(criar_debug_linha(resultado), guia))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            print(f"Estado: {estado} | Atual: {guia['ponto_atual']['erro'] if guia['ok'] else None} | Alvo: {guia['ponto_alvo']['erro'] if guia['ok'] else None} | dx: {guia['dx']} | Lado: {guia['lado_curva']} | Cmd: {comando}{' | Motivo: ' + motivo if motivo else ''}")
            if estado in ("NORMAL", "CURVA_FORTE", "REALINHAR"):
                enviar_seguro(conexao, comando, args.motores)
            time.sleep(RUN_INTERVALO_LOOP)
    except KeyboardInterrupt:
        print("CTRL+C recebido. Parando robo.")
    except Exception as erro:
        print(f"Erro grave: {erro}")
    finally:
        enviar_parar_final(conexao, args.motores)
        fechar_camera_segura(camera)
        fechar_serial_segura(conexao)
        cv2.destroyAllWindows()
        print("Recursos liberados.")


if __name__ == "__main__":
    raise SystemExit(main())
