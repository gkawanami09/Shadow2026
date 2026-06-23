"""Segue-linha real com curva fechada e recuperacao limitada da linha."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    BAUD_RATE, CAMERA_HEIGHT, CAMERA_WIDTH, CORRECAO_MAXIMA_CURVA_FECHADA,
    CORRECAO_MAXIMA_NORMAL, DURACAO_TESTE_SEGUE_LINHA,
    FAIXA_ALTA_FIM, FAIXA_ALTA_INICIO, FAIXA_BAIXA_FIM,
    FAIXA_BAIXA_INICIO, FAIXA_MEDIA_FIM, FAIXA_MEDIA_INICIO,
    FRAMES_LINHA_REENCONTRADA, INTERVALO_COMANDO_SEGUNDOS,
    KP_CURVA_FECHADA, KP_NORMAL, LIMIAR_ERRO_BAIXO_CURVA_FECHADA,
    LIMIAR_ERRO_CURVA_FECHADA, LIMIAR_ERRO_CURVA_90,
    LIMIAR_ERRO_MEDIA_ALTA_CURVA_90, LIMIAR_ERRO_SAIDA_CURVA_90,
    MAX_FRAMES_BUSCA_SEM_LINHA,
    MAX_FRAMES_SEM_LINHA_ANTES_BUSCA, PASTA_CAPTURAS, SERIAL_PORT,
    TEMPO_MAX_BUSCA_LINHA, TEMPO_MAX_CURVA_90, TIMEOUT_SERIAL, VELOCIDADE_BASE_CURVA_FECHADA,
    VELOCIDADE_BASE_NORMAL, VELOCIDADE_GIRO_BUSCA, VELOCIDADE_GIRO_CURVA_90,
    VELOCIDADE_MAXIMA_CURVA_FECHADA, VELOCIDADE_MAXIMA_SEGUE_LINHA,
    VELOCIDADE_MINIMA_CURVA_FECHADA, VELOCIDADE_MINIMA_SEGUE_LINHA,
    FRAMES_CONFIRMAR_CURVA_90, FRAMES_REENCONTRO_CURVA_90,
    PAUSA_ANTES_GIRO_90,
    TEMPO_AVANCO_ANTES_GIRO_90, VELOCIDADE_AVANCO_ANTES_GIRO_90,
    TEMPO_MAX_GIRAR_90, TEMPO_MAX_TOTAL_CURVA_90,
    FRAMES_LINHA_BAIXA_REENCONTRADA_90, LIMIAR_ERRO_SAIDA_90,
    TEMPO_REALINHAR_POS_90, VELOCIDADE_BASE_REALINHAR_POS_90,
    KP_REALINHAR_POS_90, TAMANHO_HISTORICO_ERRO_ZIGZAG,
    LIMIAR_ERRO_ZIGZAG, MIN_TROCAS_SINAL_ZIGZAG, TEMPO_MIN_ZIGZAG_SUAVE,
    TEMPO_BLOQUEIO_CURVA_90_APOS_ZIGZAG, VELOCIDADE_BASE_ZIGZAG,
    KP_ZIGZAG, CORRECAO_MAXIMA_ZIGZAG, PESO_ZIGZAG_BAIXA,
    PESO_ZIGZAG_MEDIA, PESO_ZIGZAG_ALTA,
)
from follow_test import calcular_comando, calcular_erro_final, calcular_erro_faixa, criar_debug_follow, extrair_caminho_linha, calcular_controle_vetor, medir_linha_baixa
from config import (FRAMES_CONFIRMAR_RISCO_PERDA, VELOCIDADE_BUSCA_TANQUE,
    TEMPO_BUSCA_TANQUE_LADO, TEMPO_MAX_BUSCA_TOTAL, TAMANHO_HISTORICO_LADO,
    LIMIAR_ERRO_LADO_CONFIAVEL)
from line_test import criar_debug_linha, detectar_linha


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha com recuperacao de curva fechada.")
    parser.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    parser.add_argument("--motores", action="store_true", help="Permite comandos reais ao Arduino.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou 'auto'.")
    parser.add_argument("--duracao", type=float, default=DURACAO_TESTE_SEGUE_LINHA, help="Duracao maxima em segundos.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva eventos importantes do teste.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra preview do debug, se houver interface grafica.")
    return parser.parse_args()


def calcular_faixas(resultado):
    mascara, x_inicio, centro = resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["centro_imagem_x"]
    return {
        "baixa": calcular_erro_faixa(mascara, x_inicio, centro, FAIXA_BAIXA_INICIO, FAIXA_BAIXA_FIM),
        "media": calcular_erro_faixa(mascara, x_inicio, centro, FAIXA_MEDIA_INICIO, FAIXA_MEDIA_FIM),
        "alta": calcular_erro_faixa(mascara, x_inicio, centro, FAIXA_ALTA_INICIO, FAIXA_ALTA_FIM),
    }


def detectar_curva_fechada(erro_final, faixas):
    """Identifica curva forte usando o erro geral e as faixas proximas."""
    baixo, medio = faixas["baixa"]["erro"], faixas["media"]["erro"]
    if abs(erro_final) >= LIMIAR_ERRO_CURVA_FECHADA:
        return True
    if baixo is not None and abs(baixo) >= LIMIAR_ERRO_BAIXO_CURVA_FECHADA:
        return True
    if not faixas["baixa"]["encontrou"] and (faixas["media"]["encontrou"] or faixas["alta"]["encontrou"]):
        return True
    return baixo is not None and medio is not None and baixo * medio > 0 and abs(baixo) >= 0.6 * LIMIAR_ERRO_BAIXO_CURVA_FECHADA and abs(medio) >= 0.6 * LIMIAR_ERRO_CURVA_FECHADA


def calcular_comando_estado(erro_final, curva_fechada):
    if curva_fechada:
        return calcular_comando(erro_final, KP_CURVA_FECHADA, VELOCIDADE_BASE_CURVA_FECHADA, CORRECAO_MAXIMA_CURVA_FECHADA, VELOCIDADE_MINIMA_CURVA_FECHADA, VELOCIDADE_MAXIMA_CURVA_FECHADA)
    return calcular_comando(erro_final, KP_NORMAL, VELOCIDADE_BASE_NORMAL, CORRECAO_MAXIMA_NORMAL, VELOCIDADE_MINIMA_SEGUE_LINHA, VELOCIDADE_MAXIMA_SEGUE_LINHA)


def comando_busca(ultimo_lado_linha, ultimo_erro_valido):
    if ultimo_lado_linha == "ESQUERDA" or (ultimo_lado_linha == "CENTRO" and ultimo_erro_valido < 0):
        return f"GIRAR_ESQ {VELOCIDADE_GIRO_BUSCA}"
    if ultimo_lado_linha == "DIREITA" or (ultimo_lado_linha == "CENTRO" and ultimo_erro_valido > 0):
        return f"GIRAR_DIR {VELOCIDADE_GIRO_BUSCA}"
    return "PARAR"


def detectar_curva_90(erro_final, faixas, ultimo_erro_valido):
    """Retorna indicio de curva 90, lado e motivo para o log."""
    if erro_final is not None and abs(erro_final) >= LIMIAR_ERRO_CURVA_90:
        return {"detectou": True, "lado": "ESQUERDA" if erro_final < 0 else "DIREITA", "motivo": "erro_final"}
    candidatos = [faixas["media"]["erro"], faixas["alta"]["erro"]]
    if not faixas["baixa"]["encontrou"]:
        for erro in candidatos:
            if erro is not None and abs(erro) >= LIMIAR_ERRO_MEDIA_ALTA_CURVA_90:
                return {"detectou": True, "lado": "ESQUERDA" if erro < 0 else "DIREITA", "motivo": "faixa_media_alta"}
    if erro_final is None and abs(ultimo_erro_valido) >= LIMIAR_ERRO_CURVA_90:
        return {"detectou": True, "lado": "ESQUERDA" if ultimo_erro_valido < 0 else "DIREITA", "motivo": "ultimo_erro"}
    return {"detectou": False, "lado": "DESCONHECIDO", "motivo": ""}


def comando_curva_90(lado):
    if lado == "ESQUERDA":
        return f"GIRAR_ESQ {VELOCIDADE_GIRO_CURVA_90}"
    if lado == "DIREITA":
        return f"GIRAR_DIR {VELOCIDADE_GIRO_CURVA_90}"
    return "PARAR"


def detectar_zigzag(historico_erros):
    sinais = []
    for erro in historico_erros:
        if abs(erro) >= LIMIAR_ERRO_ZIGZAG:
            sinais.append(-1 if erro < 0 else 1)
    trocas = sum(anterior != atual for anterior, atual in zip(sinais, sinais[1:]))
    return trocas >= MIN_TROCAS_SINAL_ZIGZAG


def calcular_erro_zigzag(faixas):
    pesos = {"baixa": PESO_ZIGZAG_BAIXA, "media": PESO_ZIGZAG_MEDIA, "alta": PESO_ZIGZAG_ALTA}
    encontradas = [nome for nome in pesos if faixas[nome]["erro"] is not None]
    if not encontradas:
        return None
    soma = sum(pesos[nome] for nome in encontradas)
    return sum(faixas[nome]["erro"] * pesos[nome] for nome in encontradas) / soma


def enviar_parar_seguro(conexao):
    if conexao is None or not conexao.is_open:
        return False
    try:
        from utils import enviar_comando
        resposta = enviar_comando(conexao, "PARAR")
        return not resposta or resposta.startswith("OK")
    except Exception as erro:
        print(f"Aviso: nao foi possivel enviar PARAR: {erro}")
        return False


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


def salvar_debug(imagem, estado):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_run_SPEC06_{datetime.now():%Y%m%d_%H%M%S_%f}_{estado.lower()}.jpg"
    if not cv2.imwrite(str(caminho), imagem):
        raise RuntimeError(f"Nao foi possivel salvar debug: {caminho}")
    print(f"Debug salvo em: {caminho}")


def adicionar_info_debug(imagem, estado, comando, ultimo_lado):
    debug = imagem.copy()
    textos = (f"Estado: {estado}", f"Cmd: {comando}", f"Ultimo lado: {ultimo_lado}")
    for indice, texto in enumerate(textos):
        cv2.putText(debug, texto, (15, debug.shape[0] - 65 + indice * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return debug


def mostrar_preview(imagem):
    cv2.imshow("Segue-linha SPEC 06 - q para parar", imagem)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main():
    argumentos = ler_argumentos()
    if not argumentos.camera:
        print("Use --camera para iniciar o teste.")
        return 1
    if argumentos.duracao <= 0:
        print("Erro: a duracao deve ser maior que zero.")
        return 1

    camera = None
    conexao = None
    debug_salvos = set()
    try:
        from utils import abrir_serial, enviar_comando
        porta = SERIAL_PORT if argumentos.porta == "auto" else argumentos.porta
        conexao = abrir_serial(porta, BAUD_RATE, TIMEOUT_SERIAL)
        time.sleep(2.0)
        conexao.reset_input_buffer()
        if not enviar_parar_seguro(conexao):
            raise RuntimeError("Nao foi possivel confirmar o PARAR inicial.")
        print("PARAR enviado antes do inicio.")
        print(f"Duracao: {argumentos.duracao:.1f}s | Modo: {'MOTORES' if argumentos.motores else 'SIMULACAO'}")
        if argumentos.motores:
            print("ATENCAO: MODO COM MOTORES ATIVADOS.")
            input("Pressione ENTER para iniciar ou CTRL+C para cancelar.")
        else:
            print("MODO SIMULACAO: motores nao serao acionados.")

        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
        inicio, estado = time.monotonic(), "NORMAL"
        ultimo_lado_linha, ultimo_erro_valido = "CENTRO", 0
        frames_sem_linha = frames_busca_sem_linha = frames_linha_reencontrada = 0
        tempo_inicio_busca = None
        ultimo_debug = None
        lado_curva_90 = "DESCONHECIDO"
        tempo_inicio_curva_90 = None
        tempo_inicio_estado_90 = None
        tempo_inicio_zigzag = None
        tempo_bloqueio_curva_90_ate = 0
        historico_erros = []
        historico_lados = []
        frames_risco_perda = 0
        tempo_inicio_busca_tanque = None
        lado_busca = "CENTRO"
        frames_reencontro_curva_90 = 0
        frames_suspeita_curva_90 = 0

        while time.monotonic() - inicio < argumentos.duracao and estado != "PARADO":
            resultado = detectar_linha(capturar_frame_bgr(camera))
            tempo = time.monotonic() - inicio
            pontos_vetor = extrair_caminho_linha(resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["y_inicio_roi"], resultado["centro_imagem_x"])
            controle_vetor = calcular_controle_vetor(pontos_vetor)
            baixa = medir_linha_baixa(resultado["mascara_limpa"], resultado["x_inicio_roi"], resultado["centro_imagem_x"])
            if baixa["segura"] and baixa["erro"] is not None:
                lado = "ESQUERDA" if baixa["erro"] < -LIMIAR_ERRO_LADO_CONFIAVEL else "DIREITA" if baixa["erro"] > LIMIAR_ERRO_LADO_CONFIAVEL else "CENTRO"
                historico_lados.append(lado)
                if len(historico_lados) > TAMANHO_HISTORICO_LADO: historico_lados.pop(0)
                frames_risco_perda = 0
            else:
                frames_risco_perda += 1
            if frames_risco_perda >= FRAMES_CONFIRMAR_RISCO_PERDA:
                lados = [lado for lado in historico_lados if lado != "CENTRO"]
                lado_busca = max(set(lados), key=lados.count) if lados else "CENTRO"
                if tempo_inicio_busca_tanque is None:
                    tempo_inicio_busca_tanque = time.monotonic()
                    print(f"Tempo: {tempo:.1f}s | Estado: TRAVA_PERDA | Cmd: PARAR | Ultimo lado: {lado_busca}")
                    if argumentos.motores: enviar_comando(conexao, "PARAR")
                if time.monotonic() - tempo_inicio_busca_tanque >= TEMPO_MAX_BUSCA_TOTAL:
                    print("Busca excedeu tempo maximo. PARAR.")
                    estado = "PARADO"; break
                comando = f"GIRAR_ESQ {VELOCIDADE_BUSCA_TANQUE}" if lado_busca == "ESQUERDA" else f"GIRAR_DIR {VELOCIDADE_BUSCA_TANQUE}" if lado_busca == "DIREITA" else "PARAR"
                print(f"Tempo: {tempo:.1f}s | Estado: BUSCA_TANQUE | Cmd: {comando} | Baixa: PERDIDA")
                if argumentos.motores and comando != "PARAR": enviar_comando(conexao, comando)
                time.sleep(INTERVALO_COMANDO_SEGUNDOS); continue
            tempo_inicio_busca_tanque = None
            if controle_vetor and controle_vetor["linha_baixa"] and estado not in ("AVANCAR_ANTES_GIRO_90", "GIRAR_90", "REALINHAR_POS_90"):
                estado = "LINHA_FORTE" if sum(p["encontrou"] for p in pontos_vetor) >= 4 else "LINHA_FRACA"
                comando = controle_vetor["comando"]
                print(f"Tempo: {tempo:.1f}s | Estado: {estado} | Lat: {controle_vetor['erro_lateral']:.0f} | Dir: {controle_vetor['erro_direcao']:.0f} | Corr: {controle_vetor['correcao']:.0f} | Cmd: {comando} | Baixa: SIM")
                if argumentos.motores:
                    enviar_comando(conexao, comando)
                time.sleep(INTERVALO_COMANDO_SEGUNDOS)
                continue
            if estado == "AVANCAR_ANTES_GIRO_90":
                comando = f"LADO {VELOCIDADE_AVANCO_ANTES_GIRO_90} {VELOCIDADE_AVANCO_ANTES_GIRO_90}"
                print(f"Tempo: {tempo:.1f}s | Estado: AVANCAR_ANTES_GIRO_90 | Lado: {lado_curva_90} | Cmd: {comando}")
                if argumentos.motores:
                    enviar_comando(conexao, comando)
                if time.monotonic() - tempo_inicio_estado_90 >= TEMPO_AVANCO_ANTES_GIRO_90:
                    estado, tempo_inicio_estado_90 = "GIRAR_90", time.monotonic()
                    print("Avanco antes do giro concluido. Indo para GIRAR_90.")
                time.sleep(INTERVALO_COMANDO_SEGUNDOS)
                continue
            if estado == "REALINHAR_POS_90":
                if time.monotonic() - tempo_inicio_estado_90 >= TEMPO_REALINHAR_POS_90:
                    estado = "NORMAL"
                    print("Realinhamento pos 90 concluido. Voltando para NORMAL.")
                    continue
                if resultado["encontrou_linha"]:
                    faixas_real = calcular_faixas(resultado)
                    erro_real = calcular_erro_final(faixas_real)
                    if erro_real is not None:
                        _, _, _, comando = calcular_comando(erro_real, KP_REALINHAR_POS_90, VELOCIDADE_BASE_REALINHAR_POS_90)
                        if argumentos.motores:
                            enviar_comando(conexao, comando)
                time.sleep(INTERVALO_COMANDO_SEGUNDOS)
                continue
            if estado == "GIRAR_90":
                erro_curva_90 = None
                if resultado["encontrou_linha"]:
                    faixas_90 = calcular_faixas(resultado)
                    erro_curva_90 = calcular_erro_final(faixas_90)
                    if faixas_90["baixa"]["encontrou"] and erro_curva_90 is not None and abs(erro_curva_90) <= LIMIAR_ERRO_SAIDA_90:
                        frames_reencontro_curva_90 += 1
                    else:
                        frames_reencontro_curva_90 = 0
                else:
                    frames_reencontro_curva_90 = 0
                if frames_reencontro_curva_90 >= FRAMES_LINHA_BAIXA_REENCONTRADA_90:
                    print("Linha reencontrada na faixa baixa. Indo para REALINHAR_POS_90.")
                    estado, tempo_inicio_estado_90 = "REALINHAR_POS_90", time.monotonic()
                    frames_suspeita_curva_90 = 0
                    continue
                comando = comando_curva_90(lado_curva_90)
                linha_texto = "OK" if resultado["encontrou_linha"] else "PERDIDA"
                if argumentos.salvar_debug and "curva_90_durante" not in debug_salvos:
                    salvar_debug(adicionar_info_debug(criar_debug_linha(resultado), "CURVA_90", comando, lado_curva_90), "curva_90_durante")
                    debug_salvos.add("curva_90_durante")
                print(f"Tempo: {tempo:.1f}s | Estado: CURVA_90 | Lado: {lado_curva_90} | Cmd: {comando} | Linha: {linha_texto} | Erro: {erro_curva_90}")
                if tempo_inicio_curva_90 is None or time.monotonic() - tempo_inicio_curva_90 >= TEMPO_MAX_TOTAL_CURVA_90 or time.monotonic() - tempo_inicio_estado_90 >= TEMPO_MAX_GIRAR_90 or comando == "PARAR":
                    enviar_parar_seguro(conexao)
                    estado = "PARADO"
                    print("Tempo maximo de CURVA_90 excedido. PARAR enviado.")
                    break
                if argumentos.motores:
                    enviar_comando(conexao, comando)
                time.sleep(INTERVALO_COMANDO_SEGUNDOS)
                continue
            if resultado["encontrou_linha"]:
                faixas = calcular_faixas(resultado)
                erro_final = calcular_erro_final(faixas)
                if erro_final is not None:
                    ultimo_erro_valido = erro_final
                    historico_erros.append(erro_final)
                    if len(historico_erros) > TAMANHO_HISTORICO_ERRO_ZIGZAG:
                        historico_erros.pop(0)
                    if detectar_zigzag(historico_erros):
                        estado = "ZIGZAG_SUAVE"
                        tempo_inicio_zigzag = time.monotonic()
                        tempo_bloqueio_curva_90_ate = time.monotonic() + TEMPO_BLOQUEIO_CURVA_90_APOS_ZIGZAG
                        print("ZIGZAG detectado. Bloqueando CURVA_90 temporariamente.")
                    ultimo_lado_linha = "ESQUERDA" if erro_final < -10 else "DIREITA" if erro_final > 10 else "CENTRO"
                    frames_sem_linha = 0
                    suspeita_90 = detectar_curva_90(erro_final, faixas, ultimo_erro_valido)
                    frames_suspeita_curva_90 = frames_suspeita_curva_90 + 1 if suspeita_90["detectou"] else 0
                    if frames_suspeita_curva_90 >= FRAMES_CONFIRMAR_CURVA_90 and time.monotonic() >= tempo_bloqueio_curva_90_ate:
                        estado = "AVANCAR_ANTES_GIRO_90"
                        lado_curva_90 = suspeita_90["lado"]
                        tempo_inicio_curva_90 = time.monotonic()
                        tempo_inicio_estado_90 = tempo_inicio_curva_90
                        frames_reencontro_curva_90 = 0
                        if PAUSA_ANTES_GIRO_90 > 0:
                            time.sleep(PAUSA_ANTES_GIRO_90)
                        if argumentos.salvar_debug and "curva_90_entrada" not in debug_salvos:
                            salvar_debug(adicionar_info_debug(criar_debug_linha(resultado), "CURVA_90", comando_curva_90(lado_curva_90), lado_curva_90), "curva_90_entrada")
                            debug_salvos.add("curva_90_entrada")
                        print(f"Entrando em CURVA_90 para {lado_curva_90} ({suspeita_90['motivo']}).")
                        continue
                    if estado == "BUSCA_LINHA":
                        frames_linha_reencontrada += 1
                        if frames_linha_reencontrada < FRAMES_LINHA_REENCONTRADA:
                            comando = "PARAR"
                            ultimo_debug = adicionar_info_debug(criar_debug_follow(resultado, faixas, erro_final, comando, VELOCIDADE_BASE_NORMAL, KP_NORMAL), estado, comando, ultimo_lado_linha)
                            print(f"Tempo: {tempo:.1f}s | Estado: BUSCA_LINHA | Linha reencontrada aguardando confirmacao")
                            if argumentos.motores:
                                enviar_comando(conexao, comando)
                            time.sleep(INTERVALO_COMANDO_SEGUNDOS)
                            continue
                        frames_linha_reencontrada, tempo_inicio_busca = 0, None
                        if argumentos.salvar_debug and "reencontrada" not in debug_salvos:
                            salvar_debug(criar_debug_linha(resultado), "reencontrada")
                            debug_salvos.add("reencontrada")

                    if estado == "ZIGZAG_SUAVE":
                        erro_zigzag = calcular_erro_zigzag(faixas)
                        erro_final = erro_zigzag if erro_zigzag is not None else erro_final
                        if time.monotonic() - tempo_inicio_zigzag >= TEMPO_MIN_ZIGZAG_SUAVE and not detectar_zigzag(historico_erros):
                            estado = "NORMAL"
                            print("Saindo de ZIGZAG_SUAVE. Voltando para NORMAL.")
                        _, _, _, comando = calcular_comando(erro_final, KP_ZIGZAG, VELOCIDADE_BASE_ZIGZAG, CORRECAO_MAXIMA_ZIGZAG)
                        base, kp, curva_fechada = VELOCIDADE_BASE_ZIGZAG, KP_ZIGZAG, False
                    else:
                        curva_fechada = detectar_curva_fechada(erro_final, faixas)
                        estado = "CURVA_FECHADA" if curva_fechada else "NORMAL"
                        _, _, _, comando = calcular_comando_estado(erro_final, curva_fechada)
                        base = VELOCIDADE_BASE_CURVA_FECHADA if curva_fechada else VELOCIDADE_BASE_NORMAL
                        kp = KP_CURVA_FECHADA if curva_fechada else KP_NORMAL
                    ultimo_debug = adicionar_info_debug(criar_debug_follow(resultado, faixas, erro_final, comando, base, kp), estado, comando, ultimo_lado_linha)
                    print(f"Tempo: {tempo:.1f}s | Estado: {estado} | Erro: {erro_final:.1f} | Cmd: {comando} | Linha: OK")
                    evento = "primeiro" if "primeiro" not in debug_salvos else "curva_fechada" if curva_fechada else None
                    if argumentos.salvar_debug and evento and evento not in debug_salvos:
                        salvar_debug(ultimo_debug, evento)
                        debug_salvos.add(evento)
                    if argumentos.motores:
                        resposta = enviar_comando(conexao, comando)
                        if resposta and not resposta.startswith("OK"):
                            raise RuntimeError(f"Arduino respondeu: {resposta}")
                    if argumentos.mostrar and mostrar_preview(ultimo_debug):
                        print("Preview encerrado pelo usuario.")
                        break
                    time.sleep(INTERVALO_COMANDO_SEGUNDOS)
                    continue

            # Linha nao encontrada, ou faixas sem pixels suficientes.
            faixas_vazias = {nome: {"encontrou": False, "erro": None} for nome in ("baixa", "media", "alta")}
            suspeita_90 = detectar_curva_90(None, faixas_vazias, ultimo_erro_valido)
            frames_suspeita_curva_90 = frames_suspeita_curva_90 + 1 if suspeita_90["detectou"] else 0
            if frames_suspeita_curva_90 >= FRAMES_CONFIRMAR_CURVA_90:
                estado = "CURVA_90"
                lado_curva_90 = suspeita_90["lado"]
                tempo_inicio_curva_90 = time.monotonic()
                frames_reencontro_curva_90 = 0
                print(f"Entrando em CURVA_90 para {lado_curva_90} ({suspeita_90['motivo']}).")
                continue
            if estado != "BUSCA_LINHA":
                frames_sem_linha += 1
                if frames_sem_linha >= MAX_FRAMES_SEM_LINHA_ANTES_BUSCA:
                    estado, tempo_inicio_busca = "BUSCA_LINHA", time.monotonic()
                    frames_busca_sem_linha = frames_linha_reencontrada = 0
            if estado == "BUSCA_LINHA":
                frames_busca_sem_linha += 1
                comando = comando_busca(ultimo_lado_linha, ultimo_erro_valido)
                ultimo_debug = adicionar_info_debug(criar_debug_linha(resultado), estado, comando, ultimo_lado_linha)
                print(f"Tempo: {tempo:.1f}s | Estado: BUSCA_LINHA | Ultimo lado: {ultimo_lado_linha} | Cmd: {comando} | Linha: PERDIDA")
                if argumentos.salvar_debug and "perdida" not in debug_salvos:
                    salvar_debug(ultimo_debug, "perdida")
                    debug_salvos.add("perdida")
                if comando == "PARAR" or frames_busca_sem_linha >= MAX_FRAMES_BUSCA_SEM_LINHA or time.monotonic() - tempo_inicio_busca >= TEMPO_MAX_BUSCA_LINHA:
                    enviar_parar_seguro(conexao)
                    estado = "PARADO"
                    print("Linha nao reencontrada dentro do tempo maximo. PARAR enviado por seguranca.")
                    break
                if argumentos.motores:
                    enviar_comando(conexao, comando)
            time.sleep(INTERVALO_COMANDO_SEGUNDOS)

        if argumentos.salvar_debug and ultimo_debug is not None:
            salvar_debug(ultimo_debug, "ultimo")
        print("Teste finalizado.")
        return 0
    except KeyboardInterrupt:
        print("Interrompido pelo usuario.")
        return 130
    except Exception as erro:
        print(f"Erro durante o teste: {erro}")
        return 1
    finally:
        if enviar_parar_seguro(conexao):
            print("PARAR enviado por seguranca.")
        fechar_camera_segura(camera)
        fechar_serial_segura(conexao)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Recursos liberados.")


if __name__ == "__main__":
    raise SystemExit(main())
