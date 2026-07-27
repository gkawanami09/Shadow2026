"""Detector de vítimas por modelo treinado, com rastreamento temporal.

Divisão de responsabilidades
----------------------------
* **o modelo** julga APARÊNCIA — é prata ou preta, é vítima ou não;
* **``plausibilidade``** julga GEOMETRIA — cabe fisicamente naquela posição;
* **este módulo** julga TEMPO — só vira alvo o que aparece de forma
  consistente em frames distintos, e o alvo travado não é roubado.

Essa separação é deliberada. O detector anterior misturava as três coisas em
dez portões encadeados de aparência, e por isso colapsava quando a arena
mudava. Aqui, trocar de arena afeta só o modelo — e é justamente o modelo que
pode ser retreinado com imagens novas.

Sobre o modelo ausente
----------------------
O arquivo do modelo NÃO acompanha o repositório: ele depende de imagens da
câmera deste robô. Enquanto não existir, ``carregar()`` levanta
``ModeloAusenteError`` com a instrução do que fazer. Nunca devolve detecção
falsa, nunca finge estar pronto e nunca deixa o robô andar achando que vê.
"""

import math
from pathlib import Path

import numpy as np

import config_resgate as cfg
from visao.deteccao import VictimCandidate, VictimDetection
from visao.plausibilidade import PlausibilityGuard


SHADOW_ROOT = Path(__file__).resolve().parents[1]


class ModeloAusenteError(RuntimeError):
    """O modelo treinado não existe ainda."""


def caminho_do_modelo():
    caminho = Path(cfg.VICTIM_MODEL_PATH)
    if not caminho.is_absolute():
        caminho = SHADOW_ROOT / caminho
    return caminho


def modelo_disponivel():
    return caminho_do_modelo().is_file()


def _explicar_ausencia(caminho):
    return (
        f"modelo de vítimas não encontrado em {caminho}.\n"
        "A visão de vítimas NÃO funciona sem ele — e isso é proposital: "
        "melhor parar do que andar achando que vê.\n"
        "Para gerá-lo:\n"
        "  1. colete imagens com a câmera deste robô:\n"
        "     python3 shadow/tools/coletar_dataset.py --sessao <nome>\n"
        "  2. rotule no Roboflow (classes: black, silver)\n"
        "  3. treine um YOLOv8n e exporte para ONNX\n"
        "  4. salve o arquivo exportado neste caminho\n"
        "Enquanto isso, use --sem-vitimas para exercitar só os marcadores."
    )


class VictimModel:
    """Envolve a inferência. Isola o resto do código do runtime escolhido."""

    def __init__(self, caminho=None, tamanho_entrada=None):
        self.caminho = (
            caminho_do_modelo() if caminho is None else Path(caminho))
        self.tamanho = int(
            cfg.VICTIM_MODEL_INPUT
            if tamanho_entrada is None else tamanho_entrada)
        self._sessao = None
        self._nome_entrada = None

    def carregar(self):
        if not self.caminho.is_file():
            raise ModeloAusenteError(_explicar_ausencia(self.caminho))
        try:
            import onnxruntime
        except ImportError as err:
            raise ModeloAusenteError(
                "onnxruntime não está instalado no Pi.\n"
                "  pip install onnxruntime"
            ) from err
        opcoes = onnxruntime.SessionOptions()
        opcoes.intra_op_num_threads = 4
        self._sessao = onnxruntime.InferenceSession(
            str(self.caminho),
            sess_options=opcoes,
            providers=["CPUExecutionProvider"],
        )
        self._nome_entrada = self._sessao.get_inputs()[0].name
        return self

    @property
    def carregado(self):
        return self._sessao is not None

    def inferir(self, frame_bgr):
        """Devolve candidatos crus em coordenadas do frame original."""
        if self._sessao is None:
            raise ModeloAusenteError("modelo não foi carregado")
        import cv2

        altura, largura = frame_bgr.shape[:2]
        lado = self.tamanho
        # Letterbox preserva a proporção: sem isso a esfera vira elipse e o
        # modelo vê um objeto que nunca existiu no treino.
        escala = min(lado / largura, lado / altura)
        nova_largura = int(round(largura * escala))
        nova_altura = int(round(altura * escala))
        redimensionado = cv2.resize(
            frame_bgr, (nova_largura, nova_altura))
        tela = np.full((lado, lado, 3), 114, dtype=np.uint8)
        deslocamento_x = (lado - nova_largura) // 2
        deslocamento_y = (lado - nova_altura) // 2
        tela[
            deslocamento_y:deslocamento_y + nova_altura,
            deslocamento_x:deslocamento_x + nova_largura,
        ] = redimensionado

        entrada = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB)
        entrada = entrada.astype(np.float32) / 255.0
        entrada = np.transpose(entrada, (2, 0, 1))[None, ...]
        saida = self._sessao.run(None, {self._nome_entrada: entrada})[0]
        return self._decodificar(
            saida, escala, deslocamento_x, deslocamento_y)

    def _decodificar(self, saida, escala, deslocamento_x, deslocamento_y):
        """Converte a saída do YOLOv8 em candidatos, já com NMS."""
        predicoes = np.squeeze(saida)
        if predicoes.ndim != 2:
            return []
        # YOLOv8 exporta (4+classes, N); alguns exportadores dão (N, 4+c).
        if predicoes.shape[0] < predicoes.shape[1]:
            predicoes = predicoes.T

        numero_classes = len(cfg.VICTIM_MODEL_CLASSES)
        if predicoes.shape[1] < 4 + numero_classes:
            return []
        caixas = predicoes[:, :4]
        escores = predicoes[:, 4:4 + numero_classes]
        melhores = np.argmax(escores, axis=1)
        confiancas = escores[np.arange(len(escores)), melhores]

        manter = confiancas >= cfg.VICTIM_MODEL_MIN_CONFIDENCE
        caixas = caixas[manter]
        melhores = melhores[manter]
        confiancas = confiancas[manter]
        if len(caixas) == 0:
            return []

        # cx,cy,w,h -> x0,y0,x1,y1, desfazendo letterbox e escala
        cx, cy, largura, altura = (
            caixas[:, 0], caixas[:, 1], caixas[:, 2], caixas[:, 3])
        x0 = (cx - largura / 2 - deslocamento_x) / escala
        y0 = (cy - altura / 2 - deslocamento_y) / escala
        x1 = (cx + largura / 2 - deslocamento_x) / escala
        y1 = (cy + altura / 2 - deslocamento_y) / escala

        indices = self._nms(
            np.stack([x0, y0, x1, y1], axis=1), confiancas)
        candidatos = []
        for indice in indices:
            classe = cfg.VICTIM_MODEL_CLASSES[int(melhores[indice])]
            candidatos.append(VictimCandidate.from_xyxy(
                classe, x0[indice], y0[indice], x1[indice], y1[indice],
                float(confiancas[indice])))
        return candidatos

    @staticmethod
    def _nms(caixas, escores):
        ordem = np.argsort(-escores)
        mantidos = []
        while len(ordem) > 0:
            atual = ordem[0]
            mantidos.append(atual)
            if len(ordem) == 1:
                break
            resto = ordem[1:]
            xx0 = np.maximum(caixas[atual, 0], caixas[resto, 0])
            yy0 = np.maximum(caixas[atual, 1], caixas[resto, 1])
            xx1 = np.minimum(caixas[atual, 2], caixas[resto, 2])
            yy1 = np.minimum(caixas[atual, 3], caixas[resto, 3])
            inter = (
                np.maximum(xx1 - xx0, 0) * np.maximum(yy1 - yy0, 0))
            area_atual = (
                (caixas[atual, 2] - caixas[atual, 0])
                * (caixas[atual, 3] - caixas[atual, 1]))
            area_resto = (
                (caixas[resto, 2] - caixas[resto, 0])
                * (caixas[resto, 3] - caixas[resto, 1]))
            iou = inter / np.maximum(
                area_atual + area_resto - inter, 1e-6)
            ordem = resto[iou <= cfg.VICTIM_MODEL_NMS_IOU]
        return mantidos


class VictimDetector:
    """Modelo + plausibilidade física + rastreamento de um alvo único."""

    def __init__(self, model=None, target_kind="any", guard=None):
        if target_kind not in ("any", "silver", "black"):
            raise ValueError(
                "target_kind deve ser any, silver ou black")
        self.target_kind = target_kind
        self.model = VictimModel() if model is None else model
        self.guard = PlausibilityGuard() if guard is None else guard
        self.last_candidates = ()
        self.last_rejections = {}
        self.last_diagnostic = "inicio"
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._track_locked = False
        self._last_timestamp = None
        self._frame_width = 640

    def reset(self):
        self._tracked = None
        self._hits = 0
        self._misses = 0
        self._track_locked = False
        self._last_timestamp = None
        self.last_candidates = ()
        self.last_rejections = {}
        self.last_diagnostic = "reset"

    def detect(self, frame, timestamp=None):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame BGR invalido")
        timestamp = 0.0 if timestamp is None else float(timestamp)
        altura, largura = frame.shape[:2]
        self._frame_width = largura
        self.last_rejections = {}

        crus = self.model.inferir(frame)
        aprovados = []
        for candidato in crus:
            if (
                self.target_kind != "any"
                and candidato.kind != self.target_kind
            ):
                self._rejeitar("tipo")
                continue
            veredito = self.guard.check(candidato, frame.shape)
            if not veredito.accepted:
                self._rejeitar(veredito.reason)
                continue
            aprovados.append((candidato, veredito.truncated))

        self.last_candidates = tuple(item[0] for item in aprovados)
        self.last_diagnostic = (
            "ok" if aprovados
            else (max(self.last_rejections, key=self.last_rejections.get)
                  if self.last_rejections else "sem_candidato")
        )
        escolhido = self._selecionar(aprovados)
        return self._atualizar_track(escolhido, timestamp)

    # -- interno ---------------------------------------------------------
    def _rejeitar(self, motivo):
        self.last_rejections[motivo] = (
            self.last_rejections.get(motivo, 0) + 1)

    def _selecionar(self, aprovados):
        """Escolhe UM alvo. Depois do lock, o alvo travado tem prioridade."""
        if not aprovados:
            return None
        if self._tracked is not None:
            compativeis = [
                item for item in aprovados
                if self._combina(item[0])
            ]
            if compativeis:
                return min(
                    compativeis,
                    key=lambda item: self._distancia(item[0]))
            if self._track_locked:
                # Alvo travado não é roubado por outro candidato. Some por
                # ausência (misses), nunca por concorrência.
                return None
        # Sem track: prefere a mais próxima do robô (mais baixa no quadro),
        # com a confiança desempatando.
        return max(
            aprovados,
            key=lambda item: (item[0].bottom_y, item[0].confidence))

    def _distancia(self, candidato):
        return math.hypot(
            candidato.center_x - self._tracked.center_x,
            candidato.center_y - self._tracked.center_y,
        )

    def _combina(self, candidato):
        if self._tracked is None:
            return False
        portao = max(
            cfg.VICTIM_ASSOCIATION_MIN_PX,
            cfg.VICTIM_ASSOCIATION_RADIUS_FACTOR
            * max(candidato.radius, self._tracked.radius),
        )
        if self._distancia(candidato) > portao:
            return False
        razao = candidato.radius / max(self._tracked.radius, 1e-6)
        return (
            cfg.VICTIM_RADIUS_RATIO_MIN
            <= razao
            <= cfg.VICTIM_RADIUS_RATIO_MAX
        )

    def _atualizar_track(self, escolhido, timestamp):
        if escolhido is None:
            self._misses += 1
            if self._misses > cfg.VICTIM_MAX_TRACK_MISSES:
                self._tracked = None
                self._hits = 0
                self._track_locked = False
                self._last_timestamp = None
            return None

        candidato, truncada = escolhido
        combina = self._tracked is not None and self._combina(candidato)
        novo_instante = (
            self._last_timestamp is None
            or timestamp > self._last_timestamp + 1e-9
        )
        if combina:
            # Só frames DISTINTOS contam como confirmação. Reentregar o
            # mesmo frame não pode fabricar um lock.
            if novo_instante:
                self._hits += 1
        else:
            self._hits = 1
            self._track_locked = False
        self._tracked = candidato
        self._misses = 0
        if novo_instante:
            self._last_timestamp = timestamp
        if self._hits >= cfg.VICTIM_ACQUIRE_HITS:
            self._track_locked = True

        return VictimDetection(
            kind=candidato.kind,
            center_x=candidato.center_x,
            center_y=candidato.center_y,
            radius=candidato.radius,
            confidence=candidato.confidence,
            confirmed=self._hits >= cfg.VICTIM_ACQUIRE_HITS,
            hits=self._hits,
            timestamp=timestamp,
            track_locked=self._track_locked,
            truncated=truncada,
        )
