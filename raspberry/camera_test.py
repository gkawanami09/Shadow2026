"""Teste simples da camera CSI e captura de imagens para debug.

Este modulo tambem e usado pelo segue-linha. Por isso a camera fica em modo
video continuo e a captura nao pode bloquear o loop principal do robo.
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH, PASTA_CAPTURAS

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_CALIBRACAO = RAIZ_PROJETO / "calibration" / "camera.json"

# Limites defensivos para impedir que a camera congele o controle do robo.
MAX_IDADE_FRAME_S = 0.35
TIMEOUT_PRIMEIRO_FRAME_S = 3.0


class CameraCSI:
    """Camera CSI em captura continua com buffer do frame mais recente.

    A chamada direta a ``Picamera2.capture_array`` pode bloquear se a pipeline
    da camera travar. Mantendo a captura numa thread separada, o segue-linha
    consegue detectar frame velho e parar com erro em vez de ficar preso para
    sempre dentro da leitura da camera.
    """

    def __init__(self, largura, altura, fps=CAMERA_FPS):
        if Picamera2 is None:
            raise RuntimeError(
                "biblioteca picamera2 nao encontrada. "
                "Instale/verifique a camera no Raspberry Pi OS antes de continuar."
            )
        if largura <= 0 or altura <= 0:
            raise ValueError("largura e altura da camera devem ser maiores que zero.")

        self.largura = int(largura)
        self.altura = int(altura)
        self.fps = int(fps) if fps else CAMERA_FPS
        self.picam2 = None
        self._lock = threading.RLock()
        self._condicao = threading.Condition(self._lock)
        self._parar = threading.Event()
        self._thread = None
        self._frame_rgb = None
        self._frame_id = 0
        self._ultimo_frame_t = 0.0
        self._ultimo_erro = None

        self._abrir_pipeline()
        self._thread = threading.Thread(target=self._loop_captura, name="CameraCSI", daemon=True)
        self._thread.start()
        self.aguardar_primeiro_frame()

    def _criar_configuracao_video(self, camera):
        frame_us = int(1_000_000 / max(self.fps, 1))
        principal = {"size": (self.largura, self.altura), "format": "RGB888"}
        try:
            return camera.create_video_configuration(
                main=principal,
                controls={"FrameDurationLimits": (frame_us, frame_us)},
                buffer_count=4,
            )
        except TypeError:
            # Compatibilidade com versoes antigas do Picamera2.
            return camera.create_video_configuration(main=principal)

    def _abrir_pipeline(self):
        with self._lock:
            camera = Picamera2()
            configuracao = self._criar_configuracao_video(camera)
            camera.configure(configuracao)
            camera.start()
            self.picam2 = camera
            self._ultimo_erro = None
            self._ultimo_frame_t = 0.0
            self._frame_rgb = None
            self._frame_id = 0

    def _fechar_pipeline(self):
        camera = None
        with self._lock:
            camera = self.picam2
            self.picam2 = None
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.close()
            except Exception:
                pass

    def _normalizar_rgb(self, frame):
        if frame is None:
            raise RuntimeError("camera retornou frame vazio")
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[2] == 3:
            return np.ascontiguousarray(frame)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        raise RuntimeError(f"formato inesperado da camera: shape={frame.shape}")

    def _loop_captura(self):
        while not self._parar.is_set():
            with self._lock:
                camera = self.picam2
            if camera is None:
                time.sleep(0.05)
                continue
            try:
                frame = self._normalizar_rgb(camera.capture_array("main"))
                agora = time.monotonic()
                with self._condicao:
                    self._frame_rgb = frame
                    self._frame_id += 1
                    self._ultimo_frame_t = agora
                    self._ultimo_erro = None
                    self._condicao.notify_all()
            except Exception as erro:
                with self._condicao:
                    self._ultimo_erro = f"{type(erro).__name__}: {erro}"
                    self._condicao.notify_all()
                time.sleep(0.05)

    def aguardar_primeiro_frame(self, timeout=TIMEOUT_PRIMEIRO_FRAME_S):
        prazo = time.monotonic() + timeout
        with self._condicao:
            while self._frame_rgb is None:
                restante = prazo - time.monotonic()
                if restante <= 0:
                    detalhe = f" Ultimo erro: {self._ultimo_erro}" if self._ultimo_erro else ""
                    raise RuntimeError(f"camera nao entregou o primeiro frame em {timeout:.1f}s.{detalhe}")
                self._condicao.wait(restante)

    def obter_frame_rgb(self, max_idade=MAX_IDADE_FRAME_S):
        with self._condicao:
            frame = self._frame_rgb
            ultimo_t = self._ultimo_frame_t
            erro = self._ultimo_erro

        agora = time.monotonic()
        idade = agora - ultimo_t if ultimo_t else float("inf")
        if frame is None or idade > max_idade:
            detalhe = f" Ultimo erro: {erro}" if erro else ""
            raise RuntimeError(
                f"camera sem frame recente: idade={idade:.3f}s, limite={max_idade:.3f}s.{detalhe}"
            )
        return frame.copy()

    def stop(self):
        self._parar.set()
        self._fechar_pipeline()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)



def ler_argumentos():
    parser = argparse.ArgumentParser(description="Teste da camera CSI do robo OBR.")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--salvar", action="store_true", help="Salva um unico frame.")
    modo.add_argument("--sequencia", type=int, metavar="QUANTIDADE", help="Salva varios frames.")
    modo.add_argument("--preview", action="store_true", help="Mostra frames em uma janela OpenCV.")
    parser.add_argument("--largura", type=int, default=CAMERA_WIDTH, help="Largura da imagem.")
    parser.add_argument("--altura", type=int, default=CAMERA_HEIGHT, help="Altura da imagem.")
    parser.add_argument("--intervalo", type=float, default=0.2, help="Intervalo entre frames da sequencia.")
    return parser.parse_args()



def criar_pasta_capturas():
    pasta = Path(PASTA_CAPTURAS)
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta



def iniciar_camera(largura, altura):
    return CameraCSI(largura, altura)



def capturar_frame_bgr(camera):
    """Retorna o frame mais recente da CSI em BGR para o OpenCV.

    Se a camera ficar sem entregar frame novo, levanta RuntimeError em vez de
    bloquear indefinidamente o loop do robo.
    """
    if isinstance(camera, CameraCSI):
        frame_rgb = camera.obter_frame_rgb()
    else:
        # Fallback para algum teste antigo que ainda passe Picamera2 direto.
        frame_rgb = camera.capture_array("main")
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)



def salvar_frame(frame, caminho):
    if not cv2.imwrite(str(caminho), frame):
        raise RuntimeError(f"Nao foi possivel salvar a imagem em: {caminho}")



def atualizar_calibracao(caminho_imagem):
    """Registra somente que a captura de teste foi concluida com sucesso."""
    try:
        dados = json.loads(ARQUIVO_CALIBRACAO.read_text(encoding="utf-8"))
        dados["camera_testada"] = True
        dados["ultima_imagem_teste"] = str(caminho_imagem)
        ARQUIVO_CALIBRACAO.write_text(
            json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError) as erro:
        print(f"Aviso: nao foi possivel atualizar camera.json: {erro}")



def salvar_imagem_unica(camera, pasta):
    print("Capturando frame...")
    inicio = time.monotonic()
    frame = capturar_frame_bgr(camera)
    tempo_captura = time.monotonic() - inicio
    caminho = pasta / f"frame_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    salvar_frame(frame, caminho)
    atualizar_calibracao(caminho)
    print(f"Imagem salva em: {caminho}")
    print(f"Tempo de captura: {tempo_captura:.3f} s")



def salvar_sequencia(camera, pasta, quantidade, intervalo):
    if quantidade <= 0:
        raise ValueError("A quantidade da sequencia deve ser maior que zero.")
    if intervalo < 0:
        raise ValueError("O intervalo nao pode ser negativo.")

    print(f"Capturando sequencia de {quantidade} frames...")
    inicio = time.monotonic()
    ultimo_caminho = None
    for indice in range(1, quantidade + 1):
        frame = capturar_frame_bgr(camera)
        caminho = pasta / f"frame_{indice:03d}.jpg"
        salvar_frame(frame, caminho)
        ultimo_caminho = caminho
        print(f"Imagem salva em: {caminho}")
        if indice < quantidade:
            time.sleep(intervalo)

    duracao = time.monotonic() - inicio
    fps = quantidade / duracao if duracao > 0 else 0
    if ultimo_caminho:
        atualizar_calibracao(ultimo_caminho)
    print(f"FPS estimado da sequencia: {fps:.1f}")



def mostrar_preview(camera):
    print("Preview iniciado. Pressione q para fechar.")
    print("Se estiver usando SSH sem interface grafica, o preview pode nao abrir.")
    try:
        while True:
            frame = capturar_frame_bgr(camera)
            cv2.imshow("Camera OBR - pressione q para sair", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except cv2.error as erro:
        raise RuntimeError(
            "Nao foi possivel abrir o preview. Em SSH sem interface grafica, use --salvar."
        ) from erro
    finally:
        cv2.destroyAllWindows()



def main():
    argumentos = ler_argumentos()
    if argumentos.largura <= 0 or argumentos.altura <= 0:
        print("Erro: largura e altura devem ser maiores que zero.")
        return 1

    print("Iniciando teste da camera...")
    print(f"Resolucao: {argumentos.largura}x{argumentos.altura}")
    print(f"FPS configurado de referencia: {CAMERA_FPS}")
    camera = None
    try:
        pasta = criar_pasta_capturas()
        camera = iniciar_camera(argumentos.largura, argumentos.altura)
        if argumentos.preview:
            mostrar_preview(camera)
        elif argumentos.sequencia is not None:
            salvar_sequencia(camera, pasta, argumentos.sequencia, argumentos.intervalo)
        else:
            salvar_imagem_unica(camera, pasta)
        print("Teste finalizado com sucesso.")
        return 0
    except (RuntimeError, ValueError) as erro:
        print("Erro ao abrir camera ou capturar imagem.")
        print(f"Detalhe: {erro}")
        print("Verifique a conexao da camera, o Raspberry Pi OS e se outro programa esta usando a camera.")
        return 1
    finally:
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
