"""Segue-linha OBR v2 — estado duplo (linha + verde), drain serial, verde temporal.

Corrige o congelamento do loop com motores drenando as respostas nao lidas do
Arduino a cada frame e rate-limitando os envios. Substitui a validacao de verde
de verdes.py por estabilidade temporal com margens relaxadas, corrigindo a
deteccao diagonal apos curvas.
"""

import argparse
import time

from camera_test import capturar_frame_bgr, iniciar_camera
from config import (
    BAUD_RATE, CAMERA_HEIGHT, CAMERA_WIDTH, SERIAL_PORT, TIMEOUT_SERIAL,
    VERDE_G_MENOS_B_MIN, VERDE_G_MENOS_R_MIN,
    VERDE_PROPORCAO_MIN, VERDE_AREA_MIN_LONGE, VERDE_AREA_MAX_REL_QUADRO,
    DEST_VEL_RECUPERAR,
    OBR_V2_KEEPALIVE_INTERVAL,
    OBR_V2_GREEN_STABILITY_N, OBR_V2_GREEN_MARGIN_MULT,
    OBR_V2_GREEN_ASPECT_MIN, OBR_V2_GREEN_ASPECT_MAX, OBR_V2_GREEN_FILL_MIN,
    OBR_V2_MAX_LOST_TIME, OBR_V2_MAX_GAP_TIME, OBR_V2_GAP_MAX_LOSS_TIME,
    OBR_V2_MAX_GREEN_CAND_TIME, OBR_V2_MAX_GREEN_CONF_TIME,
    OBR_V2_MAX_GREEN_ACT_TIME, OBR_V2_MAX_GREEN_RETORNO_ACT_TIME,
    OBR_V2_GREEN_ACT_MIN_TIME, OBR_V2_MAX_GREEN_COOL_TIME, OBR_V2_GREEN_COOL_FRAMES,
    OBR_V2_LOOP_TARGET_S,
)
from follow_destinos import (
    escolher_destino,
    controlar_destino,
    frente_parece_falsa_em_curva_90,
    escolher_lado_tanque_90,
    comando_recuperacao,
    lado_destino,
    TANQUE_90_VEL,
    TANQUE_90_TEMPO_MIN,
    TANQUE_90_TEMPO_MAX,
)
from verdes import analisar_verdes

try:
    from utils import abrir_serial, detectar_porta_serial, enviar_comando
except ImportError:
    abrir_serial         = None
    detectar_porta_serial = None
    enviar_comando       = None


# ── Nomes de estado ────────────────────────────────────────────────────────────

LINE_FOLLOW = "LINE_FOLLOW"
SHARP_TURN  = "SHARP_TURN"
LOST_LINE   = "LOST_LINE"
GAP_CROSS   = "GAP_CROSS"
SAFE_STOP   = "SAFE_STOP"

NO_GREEN        = "NO_GREEN"
GREEN_CANDIDATE = "GREEN_CANDIDATE"
GREEN_CONFIRMED = "GREEN_CONFIRMED"
GREEN_ACTION    = "GREEN_ACTION"
GREEN_COOLDOWN  = "GREEN_COOLDOWN"


# ── Validacao de candidato verde ───────────────────────────────────────────────

def _validar_candidato_bruto(cand, frame_w, frame_h):
    """Filtros de cor, area e forma relaxados; determina lado pelo semiplano.

    Nao usa margem de centro do cruzamento de verdes.py — a estabilidade
    temporal lida com o ruido de abordagem diagonal.
    Retorna (lado, motivo) ou (None, motivo).
    """
    proporcao = cand.get("proporcao_verde", 0)
    g_menos_r = cand.get("g_menos_r", 0)
    g_menos_b = cand.get("g_menos_b", 0)
    area      = cand.get("area", 0)
    fill      = cand.get("fill_ratio", 0)
    bbox      = cand.get("bbox")
    centro    = cand.get("centro", (frame_w // 2, frame_h // 2))

    if proporcao < VERDE_PROPORCAO_MIN:
        return None, f"prop={proporcao:.2f}<{VERDE_PROPORCAO_MIN}"
    if g_menos_r < VERDE_G_MENOS_R_MIN:
        return None, f"g-r={g_menos_r}<{VERDE_G_MENOS_R_MIN}"
    if g_menos_b < VERDE_G_MENOS_B_MIN:
        return None, f"g-b={g_menos_b}<{VERDE_G_MENOS_B_MIN}"

    area_rel = area / float(frame_w * frame_h) if frame_w * frame_h > 0 else 0.0
    if area < VERDE_AREA_MIN_LONGE:
        return None, f"area={area:.0f}<{VERDE_AREA_MIN_LONGE}"
    if area_rel > VERDE_AREA_MAX_REL_QUADRO:
        return None, f"area_rel={area_rel:.2f}>{VERDE_AREA_MAX_REL_QUADRO}"

    if bbox is not None and len(bbox) == 4:
        _, _, w, h = bbox
        if h > 0:
            asp = w / float(h)
            if not (OBR_V2_GREEN_ASPECT_MIN <= asp <= OBR_V2_GREEN_ASPECT_MAX):
                return None, f"asp={asp:.2f}"
    if fill < OBR_V2_GREEN_FILL_MIN:
        return None, f"fill={fill:.2f}<{OBR_V2_GREEN_FILL_MIN}"

    # Semiplano: meia-largura da zona morta central = OBR_V2_GREEN_MARGIN_MULT * frame_w.
    # Verde precisa estar nitidamente num dos lados para evitar ambiguidade.
    cx       = centro[0]
    zona_esq = frame_w * (0.5 - OBR_V2_GREEN_MARGIN_MULT)
    zona_dir = frame_w * (0.5 + OBR_V2_GREEN_MARGIN_MULT)
    if cx < zona_esq:
        lado = "ESQUERDA"
    elif cx > zona_dir:
        lado = "DIREITA"
    else:
        return None, f"cx={cx:.0f} zona_morta=[{zona_esq:.0f},{zona_dir:.0f}]"

    return lado, f"ok(area={area_rel:.2f},fill={fill:.2f},prop={proporcao:.2f})"


def avaliar_candidato_verde(resultado_verde):
    """Extrai o melhor candidato lateral do frame, priorizando RETORNO de verdes.py.

    Retorna (lado, motivo) onde lado e "ESQUERDA", "DIREITA", "RETORNO" ou None.
    """
    if resultado_verde is None:
        return None, "no_result"

    # RETORNO requer dois marcadores; reutiliza a deteccao testada de verdes.py.
    if (resultado_verde.get("decisao") == "RETORNO"
            and resultado_verde.get("acao_permitida", False)):
        return "RETORNO", "retorno_verdes"

    candidatos = resultado_verde.get("verdes", [])
    if not candidatos:
        return None, "no_candidates"

    res_linha = resultado_verde.get("resultado_linha", {})
    fw = res_linha.get("largura", CAMERA_WIDTH)
    fh = res_linha.get("altura", CAMERA_HEIGHT)

    for cand in candidatos:
        lado, motivo = _validar_candidato_bruto(cand, fw, fh)
        if lado is not None:
            return lado, motivo

    return None, "all_rejected"


# ── Maquina de estado — verde ──────────────────────────────────────────────────

def criar_estado_verde():
    return {
        "estado":           NO_GREEN,
        "tempo_inicio":     0.0,
        "decisao":          "NENHUM",
        "candidato":        "NENHUM",
        "votos":            {"ESQUERDA": 0, "DIREITA": 0, "RETORNO": 0},
        "frames_candidato": 0,
        "frames_sem_verde": 0,
    }


def _resetar_candidato_verde(ev):
    ev["candidato"]        = "NENHUM"
    ev["votos"]            = {"ESQUERDA": 0, "DIREITA": 0, "RETORNO": 0}
    ev["frames_candidato"] = 0


def atualizar_estado_verde(ev, lado, motivo, resultado_linha, agora):
    """Avanca o estado verde e devolve (ev, info_str) para logging."""
    N      = OBR_V2_GREEN_STABILITY_N
    estado = ev["estado"]

    # ── NO_GREEN ──────────────────────────────────────────────────────────────
    if estado == NO_GREEN:
        if lado is not None:
            _resetar_candidato_verde(ev)
            ev["estado"]          = GREEN_CANDIDATE
            ev["tempo_inicio"]    = agora
            ev["candidato"]       = lado
            ev["votos"][lado]     = 1
            ev["frames_candidato"] = 1
            return ev, f"cand({lado},1/{N},{motivo})"
        return ev, f"none({motivo})"

    # ── GREEN_CANDIDATE ───────────────────────────────────────────────────────
    if estado == GREEN_CANDIDATE:
        if agora - ev["tempo_inicio"] > OBR_V2_MAX_GREEN_CAND_TIME:
            _resetar_candidato_verde(ev)
            ev["estado"] = NO_GREEN
            return ev, "timeout->no_green"

        if lado == ev["candidato"]:
            ev["votos"][lado] += 1
            ev["frames_candidato"] += 1
            votos = ev["votos"][lado]
            if votos >= N:
                ev["estado"]      = GREEN_CONFIRMED
                ev["tempo_inicio"] = agora
                ev["decisao"]     = lado
                return ev, f"confirmed({lado},{votos}/{N})"
            return ev, f"cand({lado},{votos}/{N},{motivo})"

        if lado is None:
            # Falta momentanea: tolera mas nao avanca o contador de votos.
            ev["frames_candidato"] += 1
            votos = ev["votos"].get(ev["candidato"], 0)
            return ev, f"cand({ev['candidato']},{votos}/{N},miss)"

        # Lado diferente do acumulado: reinicia.
        era = ev["candidato"]
        _resetar_candidato_verde(ev)
        ev["estado"] = NO_GREEN
        return ev, f"reset(era={era},got={lado})"

    # ── GREEN_CONFIRMED ───────────────────────────────────────────────────────
    if estado == GREEN_CONFIRMED:
        if agora - ev["tempo_inicio"] > OBR_V2_MAX_GREEN_CONF_TIME:
            ev["estado"]      = GREEN_ACTION
            ev["tempo_inicio"] = agora
            return ev, f"->action({ev['decisao']})"
        return ev, f"confirmed({ev['decisao']},t={agora - ev['tempo_inicio']:.1f}s)"

    # ── GREEN_ACTION ──────────────────────────────────────────────────────────
    if estado == GREEN_ACTION:
        decorrido   = agora - ev["tempo_inicio"]
        timeout_max = (
            OBR_V2_MAX_GREEN_RETORNO_ACT_TIME
            if ev["decisao"] == "RETORNO"
            else OBR_V2_MAX_GREEN_ACT_TIME
        )
        linha_ok      = resultado_linha.get("encontrou_linha", False)
        sai_por_linha = (
            ev["decisao"] != "RETORNO"
            and linha_ok
            and decorrido >= OBR_V2_GREEN_ACT_MIN_TIME
        )
        if sai_por_linha or decorrido >= timeout_max:
            motivo_saida = "linha" if sai_por_linha else "timeout"
            ev["estado"]          = GREEN_COOLDOWN
            ev["tempo_inicio"]    = agora
            ev["frames_sem_verde"] = 0
            return ev, f"->cooldown({motivo_saida})"
        return ev, f"action({ev['decisao']},t={decorrido:.1f}s)"

    # ── GREEN_COOLDOWN ────────────────────────────────────────────────────────
    if estado == GREEN_COOLDOWN:
        if lado is None:
            ev["frames_sem_verde"] += 1
        else:
            ev["frames_sem_verde"] = 0
        tempo_ok     = agora - ev["tempo_inicio"] >= OBR_V2_MAX_GREEN_COOL_TIME
        sem_verde_ok = ev["frames_sem_verde"] >= OBR_V2_GREEN_COOL_FRAMES
        if tempo_ok and sem_verde_ok:
            ev["decisao"]  = "NENHUM"
            ev["candidato"] = "NENHUM"
            ev["estado"]   = NO_GREEN
            return ev, "->no_green"
        return ev, (
            f"cooldown(t={agora - ev['tempo_inicio']:.1f}s,"
            f"sv={ev['frames_sem_verde']})"
        )

    raise ValueError(f"Estado verde invalido: {estado}")


# ── Maquina de estado — linha ──────────────────────────────────────────────────

def criar_estado_linha():
    agora = time.monotonic()
    return {
        "estado":                   LINE_FOLLOW,
        "tempo_inicio":             agora,
        "tempo_ultimo_linha":       agora,
        "ultimo_lado_recuperacao":  "CENTRO",
        "ultimo_tipo_destino":      "PERDIDO",
        "tanque_lado":              "CENTRO",
        "varredura":                {"etapa": 0, "ultima_troca": agora},
    }


def atualizar_estado_linha(el, resultado, destino, agora, verde_estado):
    """Avanca o estado de linha. Congela durante GREEN_ACTION."""
    # Atualiza memoria de lado e tipo, util para recuperacao posterior.
    if destino.get("ok", False):
        el["ultimo_tipo_destino"] = destino.get("tipo", "PERDIDO")
        ld = lado_destino(destino)
        if ld in ("ESQUERDA", "DIREITA") and destino.get("tipo") in ("CURVA", "RETORNO"):
            el["ultimo_lado_recuperacao"] = ld

    linha_ok = resultado.get("encontrou_linha", False) or destino.get("ok", False)
    if linha_ok:
        el["tempo_ultimo_linha"] = agora

    # Durante GREEN_ACTION o robo esta girando; congela o timer de LOST_LINE
    # para evitar SAFE_STOP durante manobras longas (ex.: RETORNO 4.5 s).
    if verde_estado == GREEN_ACTION:
        if el["estado"] in (LOST_LINE, GAP_CROSS):
            el["tempo_inicio"] = agora
        return el

    estado = el["estado"]

    if estado == LINE_FOLLOW:
        if not linha_ok:
            sem_linha = agora - el["tempo_ultimo_linha"]
            if sem_linha >= OBR_V2_GAP_MAX_LOSS_TIME:
                el["estado"]      = LOST_LINE
                el["tempo_inicio"] = agora
            elif sem_linha > 0.05 and el["ultimo_tipo_destino"] == "FRENTE":
                el["estado"]      = GAP_CROSS
                el["tempo_inicio"] = agora
        elif (destino.get("ok", False)
              and verde_estado not in (GREEN_CONFIRMED, GREEN_COOLDOWN)):
            vpl = destino.get("validos_por_lado", {})
            if frente_parece_falsa_em_curva_90(destino, vpl, CAMERA_WIDTH):
                lado = escolher_lado_tanque_90(vpl)
                if lado:
                    el["estado"]      = SHARP_TURN
                    el["tempo_inicio"] = agora
                    el["tanque_lado"] = lado

    elif estado == GAP_CROSS:
        if linha_ok:
            el["estado"]      = LINE_FOLLOW
            el["tempo_inicio"] = agora
        elif agora - el["tempo_inicio"] > OBR_V2_MAX_GAP_TIME:
            el["estado"]      = LOST_LINE
            el["tempo_inicio"] = agora

    elif estado == SHARP_TURN:
        decorrido = agora - el["tempo_inicio"]
        frente_ok = (
            destino.get("ok", False)
            and destino.get("tipo") == "FRENTE"
            and decorrido >= TANQUE_90_TEMPO_MIN
        )
        if frente_ok or decorrido >= TANQUE_90_TEMPO_MAX:
            el["estado"]      = LINE_FOLLOW
            el["tempo_inicio"] = agora

    elif estado == LOST_LINE:
        if linha_ok:
            el["estado"]      = LINE_FOLLOW
            el["tempo_inicio"] = agora
        elif agora - el["tempo_inicio"] > OBR_V2_MAX_LOST_TIME:
            el["estado"]      = SAFE_STOP
            el["tempo_inicio"] = agora

    # SAFE_STOP e terminal: sem transicoes automaticas.
    return el


# ── Calculo de comando ─────────────────────────────────────────────────────────

def calcular_comando(el, ev, destino, resultado_linha, memoria):
    """Retorna string de comando para o Arduino com base nos dois estados."""
    estado_l = el["estado"]
    estado_v = ev["estado"]

    # GREEN_ACTION tem prioridade sobre tudo exceto SAFE_STOP.
    if estado_v == GREEN_ACTION:
        decisao = ev["decisao"]
        if decisao == "DIREITA":
            return f"GIRAR_DIR {DEST_VEL_RECUPERAR}"
        return f"GIRAR_ESQ {DEST_VEL_RECUPERAR}"  # ESQUERDA e RETORNO

    # GREEN_CONFIRMED: avanco cauteloso ate o cruzamento.
    if estado_v == GREEN_CONFIRMED:
        if destino.get("ok", False):
            cmd, _, _, _ = controlar_destino(destino, memoria, confirmar=True)
            return cmd
        vel = max(DEST_VEL_RECUPERAR, 40)
        return f"LADO {vel} {vel}"

    if estado_l == SAFE_STOP:
        return "PARAR"

    if estado_l == SHARP_TURN:
        lado = el.get("tanque_lado", "ESQUERDA")
        if lado == "DIREITA":
            return f"GIRAR_DIR {TANQUE_90_VEL}"
        return f"GIRAR_ESQ {TANQUE_90_VEL}"

    if estado_l == LOST_LINE:
        return comando_recuperacao(el["ultimo_lado_recuperacao"], el["varredura"])

    if estado_l == GAP_CROSS:
        return f"LADO {DEST_VEL_RECUPERAR} {DEST_VEL_RECUPERAR}"

    # LINE_FOLLOW (e NO_GREEN / GREEN_CANDIDATE / GREEN_COOLDOWN).
    if destino.get("ok", False):
        cmd, _, _, _ = controlar_destino(destino, memoria)
        return cmd

    return comando_recuperacao(el["ultimo_lado_recuperacao"], el["varredura"])


# ── Helpers seriais ────────────────────────────────────────────────────────────

def _drenar_serial(conexao):
    """Descarta bytes nao lidos do Arduino para liberar o buffer TX dele.

    Raiz do congelamento: Arduino responde a cada LADO com 'OK LADO ...' mas o
    Pi nunca le. Quando o buffer HW UART (64 bytes) enche, Serial.println()
    bloqueia o loop do Arduino, que para de processar comandos e de alimentar
    o watchdog. Drena aqui a cada frame para manter o buffer vazio.
    """
    if conexao is not None and conexao.is_open:
        try:
            pendentes = conexao.in_waiting
            if pendentes > 0:
                conexao.read(pendentes)
        except Exception:
            pass


def _enviar_se_necessario(conexao, cmd, ctx, agora, motores_ativo):
    """Envia cmd somente se mudou ou o intervalo de keepalive expirou.

    ctx: dict mutavel {"last_cmd": str, "last_send_t": float}.
    Retorna (sent: bool, reason: str).
    Keepalive em OBR_V2_KEEPALIVE_INTERVAL (0.20 s) garante que o watchdog
    do Arduino (1000 ms) seja alimentado mesmo quando o robo fica parado.
    """
    if not motores_ativo:
        return False, "motors_off"
    if enviar_comando is None:
        return False, "no_serial"
    if conexao is None or not conexao.is_open:
        return False, "no_conn"

    mudou   = cmd != ctx["last_cmd"]
    expirou = (agora - ctx["last_send_t"]) >= OBR_V2_KEEPALIVE_INTERVAL

    if mudou or expirou:
        reason = "changed" if mudou else "keepalive"
        try:
            enviar_comando(conexao, cmd, esperar_resposta=False)
        except RuntimeError as e:
            print(f"[AVISO] serial: {e}", flush=True)
            return False, "error"
        ctx["last_cmd"]    = cmd
        ctx["last_send_t"] = agora
        return True, reason

    return False, "skip"


# ── Logging ────────────────────────────────────────────────────────────────────

def _log_iter(n, el, ev, cmd, sent, reason, res_linha, verde_info, destino,
              t_cap, t_vis, t_ser, loop_ms):
    ls       = el["estado"]
    gs       = ev["estado"]
    cmd_log  = cmd.replace(" ", "_")
    linha_ok = "ok" if res_linha.get("encontrou_linha") else "no"
    erro_x   = res_linha.get("erro", 0) or 0
    if destino.get("ok") and "destino_global" in destino:
        gx, gy   = destino["destino_global"]
        dest_str = f"{destino.get('tipo','?')}/{gx},{gy}"
    else:
        dest_str = "PERDIDO"
    print(
        f"t={time.monotonic():.2f} frame={n} lstate={ls} gstate={gs} "
        f"cmd={cmd_log} sent={int(sent)} reason={reason} "
        f"line={linha_ok}(err={erro_x:+d}) green={verde_info} "
        f"dest={dest_str} t_cap={t_cap:.0f}ms t_vis={t_vis:.0f}ms "
        f"t_ser={t_ser:.0f}ms loop={loop_ms:.0f}ms",
        flush=True,
    )


# ── Setup / teardown ───────────────────────────────────────────────────────────

def _fechar_camera(camera):
    if camera is not None:
        try:
            camera.stop()
        except Exception as e:
            print(f"Aviso ao fechar camera: {e}")


def _fechar_serial(conexao):
    if conexao is not None and conexao.is_open:
        try:
            conexao.close()
        except Exception as e:
            print(f"Aviso ao fechar serial: {e}")


def _enviar_parar(conexao, motores_ativo):
    if not motores_ativo or enviar_comando is None:
        return
    if conexao is None or not conexao.is_open:
        return
    try:
        enviar_comando(conexao, "PARAR", esperar_resposta=False)
        time.sleep(0.10)
    except Exception:
        pass


def _ler_argumentos():
    p = argparse.ArgumentParser(description="Segue-linha OBR v2.")
    p.add_argument("--camera",  action="store_true", help="Usa a camera CSI real.")
    p.add_argument("--motores", action="store_true", help="Envia comandos ao Arduino.")
    p.add_argument("--porta",   default="auto",      help="Porta serial ou 'auto'.")
    p.add_argument("--log",     action="store_true", help="Imprime log compacto por frame.")
    return p.parse_args()


# ── Loop principal ─────────────────────────────────────────────────────────────

def main():
    args    = _ler_argumentos()
    camera  = None
    conexao = None

    try:
        if not args.camera:
            print("Use --camera.")
            return 1
        if args.motores and abrir_serial is None:
            print("Erro: pyserial nao disponivel.")
            return 1

        camera = iniciar_camera(CAMERA_WIDTH, CAMERA_HEIGHT)

        if args.motores:
            if detectar_porta_serial is None:
                print("Erro: utils.detectar_porta_serial nao disponivel.")
                return 1
            porta = (
                detectar_porta_serial() if args.porta == "auto" else args.porta
            )
            if porta is None:
                print("Erro: nenhuma porta serial detectada.")
                return 1
            conexao = abrir_serial(porta, BAUD_RATE, TIMEOUT_SERIAL)
            time.sleep(1.8)
            conexao.reset_input_buffer()
            conexao.reset_output_buffer()
            print(f"[INFO] Motores ativos em {conexao.port}", flush=True)
        else:
            print("[INFO] Simulacao sem motores.", flush=True)

        estado_linha = criar_estado_linha()
        estado_verde = criar_estado_verde()
        memoria      = {"correcao_anterior": 0.0, "vel_anterior": (0, 0)}
        ctx_serial   = {"last_cmd": "PARAR", "last_send_t": 0.0}
        frame_n      = 0

        while True:
            t0    = time.perf_counter()
            agora = time.monotonic()

            # 1. Drena respostas nao lidas (CORRECAO DO CONGELAMENTO)
            _drenar_serial(conexao)

            # 2. Captura frame
            t_cap_ini = time.perf_counter()
            frame     = capturar_frame_bgr(camera)
            t_cap     = (time.perf_counter() - t_cap_ini) * 1000

            # 3. Visao: verde chama detectar_linha internamente (sem dupla chamada)
            t_vis_ini      = time.perf_counter()
            resultado_verde = analisar_verdes(frame)
            resultado_linha = resultado_verde["resultado_linha"]
            t_vis           = (time.perf_counter() - t_vis_ini) * 1000

            # 4. Candidato verde do frame
            lado_verde, verde_motivo = avaliar_candidato_verde(resultado_verde)

            # 5. Estado verde
            estado_verde, verde_info = atualizar_estado_verde(
                estado_verde, lado_verde, verde_motivo, resultado_linha, agora
            )

            # 6. Destino visual (ray-sampling de follow_destinos)
            destino = escolher_destino(resultado_linha)

            # 7. Estado de linha
            estado_linha = atualizar_estado_linha(
                estado_linha, resultado_linha, destino, agora, estado_verde["estado"]
            )

            # 8. Comando
            cmd = calcular_comando(
                estado_linha, estado_verde, destino, resultado_linha, memoria
            )

            # 9. Envio com rate-limit
            t_ser_ini = time.perf_counter()
            sent, send_reason = _enviar_se_necessario(
                conexao, cmd, ctx_serial, agora, args.motores
            )
            t_ser = (time.perf_counter() - t_ser_ini) * 1000

            # 10. Log
            loop_ms = (time.perf_counter() - t0) * 1000
            if args.log:
                _log_iter(
                    frame_n, estado_linha, estado_verde, cmd, sent, send_reason,
                    resultado_linha, verde_info, destino, t_cap, t_vis, t_ser, loop_ms,
                )

            frame_n += 1

            # 11. Rate-limit de loop
            decorrido = time.perf_counter() - t0
            restante  = OBR_V2_LOOP_TARGET_S - decorrido
            if restante > 0:
                time.sleep(restante)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C recebido. Parando robo.", flush=True)
    except Exception as e:
        print(f"[ERRO] {e}", flush=True)
    finally:
        _enviar_parar(conexao, args.motores)
        _fechar_camera(camera)
        _fechar_serial(conexao)
        print("[INFO] Encerrado.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
