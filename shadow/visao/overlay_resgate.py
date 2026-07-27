"""Desenho do preview do resgate.

Separado da detecção de propósito: overlay é diagnóstico, não decisão. O
detector não deve nem saber que existe uma janela.

As cores seguem uma convenção que a equipe precisa conseguir ler de longe:

    prata      ciano
    preta      magenta
    verde      verde   (BGR 0,255,0)
    vermelho   vermelho (BGR 0,0,255)
    saída      azul

Verde e vermelho vêm de ``config_resgate.FINAL_TRIANGLE_OVERLAY_BGR`` e têm
teste dedicado contra inversão — um verde desenhado em vermelho faz a equipe
recalibrar a cor errada em campo.
"""

import cv2

import config_resgate as cfg


COR_POR_VITIMA = {
    "silver": (255, 255, 0),    # ciano
    "black": (255, 0, 255),     # magenta
}
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def _texto(frame, texto, posicao, cor, escala=0.52, espessura=2):
    cv2.putText(
        frame, texto, posicao, FONTE, escala, cor, espessura, cv2.LINE_AA)


def desenhar_vitima(frame, detection):
    """Círculo, cor, confiança e confirmações da vítima acompanhada."""
    if detection is None:
        return frame
    cor = COR_POR_VITIMA.get(detection.kind, (200, 200, 200))
    centro = (int(round(detection.center_x)), int(round(detection.center_y)))
    raio = max(int(round(detection.radius)), 1)
    cv2.circle(frame, centro, raio, cor, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, centro, cor, cv2.MARKER_CROSS, 12, 2)

    rotulo = (
        f"{detection.kind} {detection.confidence:.2f} "
        f"hits={detection.hits}"
        f"{' LOCK' if detection.track_locked else ''}"
        f"{' CORTADA' if detection.truncated else ''}"
    )
    _texto(frame, rotulo, (max(centro[0] - raio, 4),
                           max(centro[1] - raio - 8, 18)), cor)
    if detection.truncated:
        _texto(
            frame,
            "cortada: alinhar antes de coletar",
            (max(centro[0] - raio, 4),
             min(centro[1] + raio + 18, frame.shape[0] - 6)),
            cor, escala=0.45, espessura=1)
    return frame


def desenhar_marcadores(frame, deteccoes):
    """Verde e vermelho juntos, cada um na sua cor correta."""
    for tipo in ("green", "red"):
        deteccao = (deteccoes or {}).get(tipo)
        if deteccao is None:
            continue
        cor = cfg.FINAL_TRIANGLE_OVERLAY_BGR[tipo]
        x, y, largura, altura = deteccao.bbox
        canto0 = (int(round(x)), int(round(y)))
        canto1 = (int(round(x + largura)), int(round(y + altura)))
        cv2.rectangle(frame, canto0, canto1, cor, 2)
        _texto(
            frame,
            (
                f"{tipo} {deteccao.confidence:.2f} "
                f"hits={deteccao.hits}"
                f"{' LOCK' if deteccao.track_locked else ''}"
            ),
            (max(canto0[0], 4), max(canto0[1] - 8, 18)),
            cor,
        )
    return frame


def desenhar_estado(frame, estado, detalhe, motores_ativos, desempenho=""):
    """Faixa superior com estado do controle e telemetria da visão."""
    cor = (0, 165, 255) if motores_ativos else (0, 255, 255)
    _texto(frame, f"{estado}", (8, 24), cor, escala=0.62)
    if detalhe:
        _texto(frame, detalhe[:78], (8, 46), cor, escala=0.45, espessura=1)
    if desempenho:
        _texto(
            frame, desempenho, (8, frame.shape[0] - 10),
            (0, 255, 255), escala=0.44, espessura=1)
    if not motores_ativos:
        _texto(
            frame, "MOTORES DESATIVADOS",
            (frame.shape[1] - 250, 24), (0, 255, 0), escala=0.52)
    return frame


def desenhar_plausibilidade(frame, guard):
    """Mostra o horizonte útil: acima dele nenhuma vítima é aceita."""
    if guard is None or not getattr(guard, "enabled", False):
        return frame
    altura, largura = frame.shape[:2]
    linha = int(round(altura * cfg.PLAUSIBLE_MIN_CENTER_Y_RATIO))
    cv2.line(frame, (0, linha), (largura, linha), (90, 90, 90), 1)
    _texto(
        frame, "horizonte util", (8, max(linha - 6, 12)),
        (90, 90, 90), escala=0.40, espessura=1)
    if guard.last_reason:
        _texto(
            frame, f"veto fisico: {guard.last_reason}",
            (8, min(linha + 18, altura - 6)),
            (0, 140, 255), escala=0.42, espessura=1)
    return frame


def anotar(
    frame,
    detection=None,
    marcadores=None,
    estado="",
    detalhe="",
    motores_ativos=False,
    desempenho="",
    guard=None,
    copiar=True,
):
    """Monta o preview completo. Nunca altera o frame original por padrão."""
    tela = frame.copy() if copiar else frame
    desenhar_plausibilidade(tela, guard)
    desenhar_vitima(tela, detection)
    desenhar_marcadores(tela, marcadores)
    desenhar_estado(tela, estado, detalhe, motores_ativos, desempenho)
    return tela
