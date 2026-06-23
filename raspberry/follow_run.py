"""Segue-linha continuo com Tank Assist para recuperar curvas e perda da linha."""

import argparse
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
    RUN_LIMIAR_ERRO_SAIDA_TANK, RUN_LIMIAR_LADO_CONFIAVEL,
    RUN_MIN_PIXELS_BAIXA_FRACA, RUN_MIN_PIXELS_BAIXA_SEGURA,
    RUN_SALVAR_DEBUG_EVENTOS, RUN_TAMANHO_HISTORICO_LADO,
    RUN_TEMPO_MAX_TANK, RUN_TEMPO_PULSO_BUSCA, RUN_TEMPO_PULSO_TANK,
    RUN_TEMPO_PULSO_VARREDURA, RUN_TEMPO_REALINHAR, RUN_VELOCIDADE_BASE,
    RUN_VELOCIDADE_BUSCA, RUN_VELOCIDADE_CURVA_FORTE, RUN_VELOCIDADE_MAXIMA,
    RUN_VELOCIDADE_MINIMA, RUN_VELOCIDADE_REALINHAR, RUN_VELOCIDADE_TANK,
    RUN_VELOCIDADE_VARREDURA, RUN_Y_BAIXA_FIM, RUN_Y_BAIXA_INICIO,
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


def extrair_pontos_vetor(mascara_limpa, x_inicio_roi, y_inicio_roi, centro_imagem_x):
    """Extrai o centro da linha em seis faixas, da parte baixa para a alta."""
    altura, largura = mascara_limpa.shape[:2]
    pontos = []
    for indice in range(6):
        y2 = altura - indice * altura // 6
        y1 = altura - (indice + 1) * altura // 6
        faixa = mascara_limpa[y1:y2, :]
        contornos, _ = cv2.findContours(faixa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno = max(contornos, key=cv2.contourArea, default=None)
        ponto = {"encontrou": False, "x": None, "y": y_inicio_roi + (y1 + y2) // 2,
                 "erro": None}
        if contorno is not None and cv2.contourArea(contorno) > 0:
            momentos = cv2.moments(contorno)
            if momentos["m00"]:
                x = x_inicio_roi + int(momentos["m10"] / momentos["m00"])
                ponto.update({"encontrou": True, "x": x, "erro": x - centro_imagem_x})
        pontos.append(ponto)
    return pontos


def calcular_controle_vetor(pontos):
    validos = [ponto for ponto in pontos if ponto["encontrou"]]
    if not validos:
        return None
    perto = validos[0]
    lookahead = validos[min(len(validos) - 1, max(1, len(validos) // 2))]
    return {"erro_lateral": perto["erro"], "erro_direcao": lookahead["erro"] - perto["erro"],
            "ponto_perto": perto, "ponto_lookahead": lookahead}


def atualizar_lado_confiavel(historico_lados, erro):
    if erro is None:
        return None
    lado = "ESQUERDA" if erro < -RUN_LIMIAR_LADO_CONFIAVEL else "DIREITA" if erro > RUN_LIMIAR_LADO_CONFIAVEL else "CENTRO"
    historico_lados.append(lado)
    return lado


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


def comando_lado_por_controle(erro_lateral, erro_direcao, modo):
    parametros = {
        "NORMAL": (RUN_VELOCIDADE_BASE, RUN_KP_LATERAL, RUN_KP_DIRECAO, RUN_CORRECAO_MAXIMA),
        "CURVA_FORTE": (RUN_VELOCIDADE_CURVA_FORTE, RUN_KP_CURVA_FORTE_LATERAL, RUN_KP_CURVA_FORTE_DIRECAO, RUN_CORRECAO_MAXIMA_CURVA_FORTE),
        "REALINHAR": (RUN_VELOCIDADE_REALINHAR, RUN_KP_REALINHAR, RUN_KP_REALINHAR, RUN_CORRECAO_MAXIMA_REALINHAR),
    }
    base, kp_lateral, kp_direcao, maximo = parametros[modo]
    correcao = limitar(kp_lateral * erro_lateral + kp_direcao * erro_direcao, -maximo, maximo)
    esquerda = round(limitar(base + correcao, RUN_VELOCIDADE_MINIMA, RUN_VELOCIDADE_MAXIMA))
    direita = round(limitar(base - correcao, RUN_VELOCIDADE_MINIMA, RUN_VELOCIDADE_MAXIMA))
    return f"LADO {esquerda} {direita}", correcao


def salvar_debug_evento(resultado, estado, comando, info_extra):
    if not (SALVAR_DEBUG_ATIVO and RUN_SALVAR_DEBUG_EVENTOS):
        return
    debug = criar_debug_linha(resultado)
    linhas = [f"Estado: {estado}", f"Cmd: {comando}"]
    linhas.extend(f"{chave}: {valor}" for chave, valor in info_extra.items())
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
        ultimo_lado_confiavel, ultimo_erro = "CENTRO", 0
        frames_risco = frames_correcao_saturada = frames_reencontro = 0
        tempo_estado = time.monotonic()
        etapa_varredura = 0

        while True:
            resultado = detectar_linha(capturar_frame_bgr(camera))
            baixa = medir_baixa(resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["centro_imagem_x"])
            pontos = extrair_pontos_vetor(resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["y_inicio_roi"], resultado["centro_imagem_x"])
            controle = calcular_controle_vetor(pontos)
            if baixa["encontrou"]:
                ultimo_erro = baixa["erro"]
                atualizar_lado_confiavel(historico_lados, ultimo_erro)
                lado = escolher_lado_busca(historico_lados, ultimo_lado_confiavel)
                if lado != "CENTRO":
                    ultimo_lado_confiavel = lado

            motivo, comando, correcao = "", "PARAR", 0
            if estado in ("TANK_ASSIST", "BUSCA_TANQUE", "VARREDURA"):
                baixa_reencontrada = baixa["segura"] and abs(baixa["erro"]) <= RUN_LIMIAR_ERRO_SAIDA_TANK
                frames_reencontro = frames_reencontro + 1 if baixa_reencontrada else 0
                if frames_reencontro >= RUN_FRAMES_REENCONTRO_BAIXA:
                    estado, tempo_estado, frames_reencontro = "REALINHAR", time.monotonic(), 0
                    frames_risco = frames_correcao_saturada = 0
                elif estado == "TANK_ASSIST" and time.monotonic() - tempo_estado >= RUN_TEMPO_MAX_TANK:
                    estado, tempo_estado = "BUSCA_TANQUE", time.monotonic()
                elif estado == "TANK_ASSIST":
                    comando, motivo = comando_tank(escolher_lado_busca(historico_lados, ultimo_lado_confiavel), RUN_VELOCIDADE_TANK), "risco"
                    if comando == "PARAR":
                        estado, tempo_estado = "VARREDURA", time.monotonic()
                    else:
                        enviar_seguro(conexao, comando, args.motores); time.sleep(RUN_TEMPO_PULSO_TANK)
                        enviar_seguro(conexao, "PARAR", args.motores)
                if estado == "BUSCA_TANQUE":
                    comando = comando_tank(escolher_lado_busca(historico_lados, ultimo_lado_confiavel), RUN_VELOCIDADE_BUSCA)
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
            elif not resultado["encontrou_linha"] or controle is None:
                estado, tempo_estado = "BUSCA_TANQUE", time.monotonic()
                comando, motivo = "PARAR", "linha perdida"
                enviar_seguro(conexao, comando, args.motores)
            else:
                erro_lateral, erro_direcao = controle["erro_lateral"], controle["erro_direcao"]
                comando_normal, correcao_normal = comando_lado_por_controle(erro_lateral, erro_direcao, "NORMAL")
                frames_risco = frames_risco + 1 if baixa["risco"] else 0
                frames_correcao_saturada = frames_correcao_saturada + 1 if abs(correcao_normal) >= RUN_CORRECAO_SATURADA_TANK else 0
                if estado == "REALINHAR" and time.monotonic() - tempo_estado < RUN_TEMPO_REALINHAR:
                    comando, correcao = comando_lado_por_controle(erro_lateral, erro_direcao, "REALINHAR")
                elif frames_risco >= RUN_FRAMES_RISCO_TANK or frames_correcao_saturada >= RUN_FRAMES_CORRECAO_SATURADA:
                    estado, tempo_estado, motivo = "TANK_ASSIST", time.monotonic(), "risco"
                    comando = "PARAR"
                    enviar_seguro(conexao, comando, args.motores)
                elif abs(erro_direcao) >= RUN_LIMIAR_CURVA_FORTE or abs(correcao_normal) >= RUN_CORRECAO_SATURADA_TANK:
                    estado, comando, correcao = "CURVA_FORTE", *comando_lado_por_controle(erro_lateral, erro_direcao, "CURVA_FORTE")
                else:
                    estado, comando, correcao = "NORMAL", comando_normal, correcao_normal

            if estado != estado_anterior:
                salvar_debug_evento(resultado, estado, comando, {"Baixa": "SEGURA" if baixa["segura"] else "FRACA" if baixa["fraca"] else "PERDIDA", "Lat": controle["erro_lateral"] if controle else None, "Dir": controle["erro_direcao"] if controle else None, "Lado conf": ultimo_lado_confiavel})
                estado_anterior = estado
            if args.mostrar:
                cv2.imshow("Segue-linha - q para parar", criar_debug_linha(resultado))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            print(f"Estado: {estado} | Baixa: {'SEGURA' if baixa['segura'] else 'FRACA' if baixa['fraca'] else 'PERDIDA'} | Lat: {controle['erro_lateral'] if controle else None} | Dir: {controle['erro_direcao'] if controle else None} | Cmd: {comando} | Lado conf: {ultimo_lado_confiavel}{' | Motivo: ' + motivo if motivo else ''}")
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
