"""Primeiro segue-linha real, limitado e seguro."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    CAMERA_HEIGHT, CAMERA_WIDTH, DURACAO_TESTE_SEGUE_LINHA,
    INTERVALO_COMANDO_SEGUNDOS, KP_SEGUE_LINHA, MAX_FRAMES_SEM_LINHA,
    PARAR_SE_PERDER_LINHA, PASTA_CAPTURAS, SERIAL_PORT,
    TIMEOUT_SERIAL, BAUD_RATE, VELOCIDADE_BASE_SEGUE_LINHA,
)
from follow_test import (
    calcular_comando, calcular_erro_final, calcular_erro_faixa,
    criar_debug_follow,
)
from line_test import criar_debug_linha, detectar_linha
from config import (
    FAIXA_ALTA_FIM, FAIXA_ALTA_INICIO, FAIXA_BAIXA_FIM,
    FAIXA_BAIXA_INICIO, FAIXA_MEDIA_FIM, FAIXA_MEDIA_INICIO,
)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Segue-linha real de baixa velocidade.")
    parser.add_argument("--camera", action="store_true", help="Usa a camera CSI real.")
    parser.add_argument("--motores", action="store_true", help="Permite enviar comandos reais ao Arduino.")
    parser.add_argument("--porta", default="auto", help="Porta serial ou 'auto'.")
    parser.add_argument("--duracao", type=float, default=DURACAO_TESTE_SEGUE_LINHA, help="Duracao maxima em segundos.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva primeiro, ultimo e perda de linha.")
    parser.add_argument("--mostrar", action="store_true", help="Mostra preview do debug, se houver interface grafica.")
    return parser.parse_args()


def enviar_parar_seguro(conexao):
    """Tenta parar o Arduino sem deixar uma falha de limpeza interromper o fim."""
    if conexao is None or not conexao.is_open:
        return False
    try:
        from utils import enviar_comando
        resposta = enviar_comando(conexao, "PARAR")
        if resposta and not resposta.startswith("OK"):
            print(f"Aviso: resposta inesperada ao PARAR: {resposta}")
            return False
        return True
    except Exception as erro:
        print(f"Aviso: nao foi possivel enviar PARAR: {erro}")
        return False


def salvar_debug(imagem, nome):
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"debug_follow_{nome}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    if not cv2.imwrite(str(caminho), imagem):
        raise RuntimeError(f"Nao foi possivel salvar debug: {caminho}")
    print(f"Debug salvo em: {caminho}")


def calcular_faixas(resultado):
    mascara = resultado["mascara_limpa"]
    x_inicio = resultado["x_inicio_roi"]
    centro = resultado["centro_imagem_x"]
    return {
        "baixa": calcular_erro_faixa(mascara, x_inicio, centro, FAIXA_BAIXA_INICIO, FAIXA_BAIXA_FIM),
        "media": calcular_erro_faixa(mascara, x_inicio, centro, FAIXA_MEDIA_INICIO, FAIXA_MEDIA_FIM),
        "alta": calcular_erro_faixa(mascara, x_inicio, centro, FAIXA_ALTA_INICIO, FAIXA_ALTA_FIM),
    }


def mostrar_preview(imagem):
    cv2.imshow("Segue-linha - pressione q para parar", imagem)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main():
    argumentos = ler_argumentos()
    if not argumentos.camera:
        print("Use --camera para iniciar o teste.")
        return 1
    if argumentos.duracao <= 0:
        print("Erro: a duracao deve ser maior que zero.")
        return 1

    conexao = None
    camera = None
    try:
        from utils import abrir_serial, enviar_comando

        porta = SERIAL_PORT if argumentos.porta == "auto" else argumentos.porta
        conexao = abrir_serial(porta, BAUD_RATE, TIMEOUT_SERIAL)
        # A abertura da porta USB pode reiniciar o Arduino.
        time.sleep(2.0)
        conexao.reset_input_buffer()
        print(f"Duracao: {argumentos.duracao:.1f}s")
        print(f"Velocidade base: {VELOCIDADE_BASE_SEGUE_LINHA}")
        print(f"KP: {KP_SEGUE_LINHA:.2f}")
        print(f"Modo motores: {'ATIVADO' if argumentos.motores else 'SIMULACAO'}")

        if not enviar_parar_seguro(conexao):
            raise RuntimeError("Nao foi possivel confirmar o PARAR inicial.")
        print("PARAR enviado antes do inicio.")

        if argumentos.motores:
            print("ATENCAO: MODO COM MOTORES ATIVADOS.")
            print("Deixe o robo no chao, com espaco livre e mao perto da chave.")
            input("Pressione ENTER para iniciar ou CTRL+C para cancelar.")
        else:
            print("MODO SIMULACAO: motores nao serao acionados.")

        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)
        inicio = time.monotonic()
        frames_sem_linha = 0
        primeiro_debug_salvo = False

        while time.monotonic() - inicio < argumentos.duracao:
            frame = capturar_frame_bgr(camera)
            resultado = detectar_linha(frame)
            tempo = time.monotonic() - inicio

            if not resultado["encontrou_linha"]:
                frames_sem_linha += 1
                print(f"Tempo: {tempo:.1f}s | Linha perdida ({frames_sem_linha}/{MAX_FRAMES_SEM_LINHA})")
                if argumentos.salvar_debug:
                    salvar_debug(criar_debug_linha(resultado), "linha_perdida")
                if PARAR_SE_PERDER_LINHA and frames_sem_linha >= MAX_FRAMES_SEM_LINHA:
                    enviar_parar_seguro(conexao)
                    print("Linha perdida | PARAR")
                    break
            else:
                frames_sem_linha = 0
                faixas = calcular_faixas(resultado)
                erro_final = calcular_erro_final(faixas)
                if erro_final is None:
                    enviar_parar_seguro(conexao)
                    print(f"Tempo: {tempo:.1f}s | Faixas sem linha | PARAR")
                    break
                _, esquerda, direita, comando = calcular_comando(erro_final, KP_SEGUE_LINHA, VELOCIDADE_BASE_SEGUE_LINHA)
                debug = criar_debug_follow(resultado, faixas, erro_final, comando, VELOCIDADE_BASE_SEGUE_LINHA, KP_SEGUE_LINHA)
                print(f"Tempo: {tempo:.1f}s | Erro: {erro_final:.1f} | Comando: {comando} | Linha: OK")
                if argumentos.salvar_debug and not primeiro_debug_salvo:
                    salvar_debug(debug, "primeiro")
                    primeiro_debug_salvo = True
                if argumentos.motores:
                    resposta = enviar_comando(conexao, comando)
                    if resposta and not resposta.startswith("OK"):
                        raise RuntimeError(f"Arduino respondeu: {resposta}")
                if argumentos.mostrar and mostrar_preview(debug):
                    print("Preview encerrado pelo usuario.")
                    break

            time.sleep(INTERVALO_COMANDO_SEGUNDOS)

        if argumentos.salvar_debug and 'debug' in locals():
            salvar_debug(debug, "ultimo")
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
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if conexao is not None and conexao.is_open:
            conexao.close()


if __name__ == "__main__":
    raise SystemExit(main())
