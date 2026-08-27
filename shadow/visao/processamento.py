"""Processa cada imagem da câmera do segue-linha."""

import time
from multiprocessing import shared_memory

import cv2
import numpy as np

import config
from config import (BLACK_AVG_SIDE_MASK, DEBUG_SHM_NAME, VISION_MAX_FRAMES,
                    camera_x, camera_y)
from shared.dados_compartilhados import (add_time_value, black_average,
                                         config_manager, empty_time_arr,
                                         entry_armed,
                                         green_candidate,
                                         green_turn_target,
                                         get_time_average, last_bottom_point,
                                         last_bottom_point_y,
                                         line_ahead, line_angle, line_angle_y,
                                         line_detected, line_size,
                                         line_status, min_line_size,
                                         publicar_resultado_visao_rapida,
                                         preferencia_linha_esquerda,
                                         red_candidate, red_detected, status,
                                         steering_correction,
                                         steering_heading,
                                         steering_lateral_error,
                                         steering_left_pwm,
                                         steering_right_pwm, steering_state,
                                         terminate, timer, turn_dir,
                                         vision_ready)
from visao import linha as line_module
from visao import verde as green_module
from visao.captura import LineCamera
from visao.entrada_missao import build_entry_gate, update_entry_silver
from visao.gap import apply_gap_avoid_mask, publish_gap_geometry, reset_gap_values
from visao.linha import calculate_angle, determine_correct_line
from visao.trajetoria import extrair_ponto_futuro
from visao.verde import check_green, latch_turn_direction
from visao.vermelho import ConfirmadorVermelho, check_contour_size

# Cores carregadas do config.ini (fallback: valores do config.py)
black_min = np.array(config.BLACK_MIN_DEFAULT)
black_max_normal_top = np.array(config.BLACK_MAX_NORMAL_TOP_DEFAULT)
black_max_normal_bottom = np.array(config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT)
black_max_ramp_down_top = np.array(config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT)
green_min = np.array(config.GREEN_MIN_DEFAULT)
green_max = np.array(config.GREEN_MAX_DEFAULT)
red_min_1 = np.array(config.RED_MIN_1_DEFAULT)
red_max_1 = np.array(config.RED_MAX_1_DEFAULT)
red_min_2 = np.array(config.RED_MIN_2_DEFAULT)
red_max_2 = np.array(config.RED_MAX_2_DEFAULT)


def update_color_values():
    global black_max_normal_top, black_max_normal_bottom, \
        black_max_ramp_down_top, green_min, green_max, \
        red_min_1, red_max_1, red_min_2, red_max_2

    def read(name, fallback):
        value = config_manager.read_variable('color_values_line', name)
        return np.array(value) if value is not None else np.array(fallback)

    black_max_normal_top = read('black_max_normal_top', config.BLACK_MAX_NORMAL_TOP_DEFAULT)
    black_max_normal_bottom = read('black_max_normal_bottom', config.BLACK_MAX_NORMAL_BOTTOM_DEFAULT)
    black_max_ramp_down_top = read(
        'black_max_ramp_down_top',
        config.BLACK_MAX_RAMP_DOWN_TOP_DEFAULT,
    )

    green_min = read('green_min', config.GREEN_MIN_DEFAULT)
    green_max = read('green_max', config.GREEN_MAX_DEFAULT)

    red_min_1 = read('red_min_1', config.RED_MIN_1_DEFAULT)
    red_max_1 = read('red_max_1', config.RED_MAX_1_DEFAULT)
    red_min_2 = read('red_min_2', config.RED_MIN_2_DEFAULT)
    red_max_2 = read('red_max_2', config.RED_MAX_2_DEFAULT)


def _has_black_ahead(mask):
    """Há preto persistente no corredor central à frente do robô?

    A mesma geometria avalia os dois perfis de preto: o normal, que orienta a
    linha, e o específico da rampa. Uma barra transversal isolada não basta;
    o preto precisa ocupar muitas linhas do corredor na direção de marcha.
    """
    if mask is None or mask.ndim != 2:
        raise ValueError("_has_black_ahead exige uma mascara 2D")
    height, width = mask.shape
    ahead = mask[
        0:int(height * config.GAP_AHEAD_Y_MAX),
        int(width * config.GAP_AHEAD_X_MIN):int(
            width * config.GAP_AHEAD_X_MAX),
    ]
    if not ahead.size:
        return False
    row_fill = np.count_nonzero(ahead, axis=1) / ahead.shape[1]
    return bool(
        np.mean(row_fill >= config.GAP_AHEAD_ROW_FILL)
        >= config.GAP_AHEAD_ROW_PERSISTENCE
    )


def vision_loop(debug=False):
    # As operações trabalham em apenas 448×252. Paralelizá-las faz a Pi 5
    # acordar vários núcleos por poucos milissegundos sem mudar o resultado;
    # uma thread deixa o consumo mais uniforme e ainda cabe folgado em 25 ms.
    cv2.setNumThreads(max(1, int(config.VISION_OPENCV_THREADS)))
    line_module.init_tracker()
    inicio_aquecimento = time.perf_counter()
    print("[visão] preparando cálculos rápidos...")
    line_module.aquecer_numba()
    green_module.aquecer_numba()
    print("[visão] cálculos prontos em "
          f"{time.perf_counter() - inicio_aquecimento:.2f}s")

    bottom_y = camera_y

    time_line_angle = empty_time_arr()
    time_turn_direction = empty_time_arr()
    time_last_bottom_point_x = empty_time_arr()
    time_last_average_line_point = empty_time_arr()

    camera = LineCamera()

    shm = None
    shm_array = None
    if debug:
        shm = shared_memory.SharedMemory(name=DEBUG_SHM_NAME)
        shm_array = np.ndarray((camera_y, camera_x, 3), dtype=np.uint8, buffer=shm.buf)

    update_color_values()

    # Faixa prata de entrada. O detector vive aqui porque este é o único
    # processo que possui a câmera de linha; publicar o resultado por memória
    # compartilhada evita abrir a mesma câmera duas vezes. Sem `mission_mode`
    # ele nunca é construído e o custo é zero.
    entry_gate = build_entry_gate()
    ultimo_alinhamento_entrada = 0.

    # Matriz usada para reduzir ruídos das máscaras.
    kernal = np.ones((3, 3), np.uint8)
    confirmador_vermelho = ConfirmadorVermelho()

    # Contador e limitador de imagens por segundo.
    fps_time = time.perf_counter()
    counter = 0
    fps = 0
    fps_limit_time = time.perf_counter()

    timer.set_timer("multiple_bottom", .05)
    timer.set_timer("multiple_side_l", .05)
    timer.set_timer("multiple_side_r", .05)
    timer.set_timer("right_marker", .05)
    timer.set_timer("left_marker", .05)

    try:
        while not terminate.value:
            cv2_img = camera.get_frame()
            frame_captured_at = time.perf_counter()

            if time.perf_counter() - fps_limit_time <= 1 / VISION_MAX_FRAMES:
                continue
            fps_limit_time = time.perf_counter()
            inicio_processamento = time.perf_counter()

            # Valores locais do MESMO frame. Só são publicados juntos depois
            # que toda a análise terminar.
            linha_detectada_frame = False
            linha_a_frente_frame = False
            angulo_frame = 0.
            ponto_inferior_x_frame = camera_x / 2
            ponto_inferior_y_frame = 0.
            ponto_alvo_x_frame = camera_x / 2
            ponto_alvo_y_frame = 0.
            ponto_futuro_x_frame = camera_x / 2
            ponto_futuro_y_frame = camera_y
            ponto_futuro_valido_frame = False
            area_linha_frame = 0.
            candidato_verde_frame = False
            candidato_vermelho_frame = False
            hsv_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
            green_image = cv2.inRange(hsv_image, green_min, green_max)
            red_image = cv2.bitwise_or(
                cv2.inRange(hsv_image, red_min_1, red_max_1),
                cv2.inRange(hsv_image, red_min_2, red_max_2),
            )

            black_image = cv2.inRange(cv2_img, black_min, black_max_normal_bottom)
            limite_topo = int(camera_y * .4)
            black_image[:limite_topo] = cv2.inRange(
                cv2_img[:limite_topo],
                black_min,
                black_max_normal_top,
            )
            black_image = cv2.bitwise_and(
                black_image,
                cv2.bitwise_not(green_image),
            )

            media_preto = float(np.mean(black_image))
            black_average.value = media_preto

            # A rampa tem seu proprio teto de preto, calibrado no grupo 3.
            # Esta mascara nunca substitui a mascara normal da linha: ela so
            # acrescenta um veto independente a entrada de resgate.
            black_ramp_image = np.zeros_like(black_image)
            black_ramp_image[:limite_topo] = cv2.inRange(
                cv2_img[:limite_topo],
                black_min,
                black_max_ramp_down_top,
            )
            black_ramp_image[:limite_topo] = cv2.bitwise_and(
                black_ramp_image[:limite_topo],
                cv2.bitwise_not(green_image[:limite_topo]),
            )
            # A mascara normal muda no segue-linha abaixo. Preserva a medida
            # bruta para compara-la com a caixa prata deste mesmo frame.
            entry_black_mask = black_image.copy()
            entry_ramp_black_mask = black_ramp_image

            # Esta leitura e exclusiva do detector de gap. O veto da prata
            # usa as duas mascaras brutas, mas apenas acima da caixa que o
            # YOLO encontrou no mesmo frame.
            linha_a_frente_frame = _has_black_ahead(black_image)
            line_ahead.value = linha_a_frente_frame

            # Recorta partes que não devem participar da decisão.
            if line_status.value == "gap_avoid":
                apply_gap_avoid_mask(black_image)

            if (
                bottom_y < camera_y * .95
                and media_preto < BLACK_AVG_SIDE_MASK
                and line_status.value == "line_detected"
            ):
                cv2.rectangle(black_image, (0, 0), (int(camera_x * .25), camera_y), 0, -1)
                cv2.rectangle(black_image, (int(camera_x * .75), 0), (camera_x, camera_y), 0, -1)

            # Redução de ruído.
            if line_status.value == "gap_avoid":
                black_image = cv2.erode(black_image, kernal, iterations=5)
                black_image = cv2.dilate(black_image, kernal, iterations=8)
            else:
                black_image = cv2.erode(black_image, kernal, iterations=5)
                black_image = cv2.dilate(black_image, kernal, iterations=17)
                black_image = cv2.erode(black_image, kernal, iterations=9)

            green_image = cv2.erode(green_image, kernal, iterations=1)
            green_image = cv2.dilate(green_image, kernal, iterations=11)
            green_image = cv2.erode(green_image, kernal, iterations=9)

            red_image = cv2.erode(red_image, kernal, iterations=1)
            red_image = cv2.dilate(red_image, kernal, iterations=11)
            red_image = cv2.erode(red_image, kernal, iterations=9)

            # Encontra os contornos.
            contours_grn, _ = cv2.findContours(green_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(red_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_blk, _ = cv2.findContours(black_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            area_minima_linha = min_line_size.value
            contours_blk = [
                contorno
                for contorno in contours_blk
                if cv2.contourArea(contorno) > area_minima_linha
            ]

            # Procura a faixa vermelha.
            candidato_vermelho_frame = check_contour_size(
                contours_red,
                "red",
                debug_img=cv2_img if debug else None,
                frame_shape=red_image.shape,
            )
            red_candidate.value = candidato_vermelho_frame
            red_detected.value = confirmador_vermelho.atualizar(
                candidato_vermelho_frame)

            # Procura os marcadores verdes normalmente. A validacao da prata
            # nao altera a logica de verde.
            candidato_verde_frame = any(
                cv2.contourArea(contorno) > config.GREEN_MIN_AREA
                for contorno in contours_grn
            )
            green_candidate.value = candidato_verde_frame
            if len(contours_grn) > 0:
                turn_direction = check_green(
                    contours_grn, black_image,
                    debug_img=cv2_img if debug else None)
            else:
                turn_direction = "straight"

            time_turn_direction = latch_turn_direction(
                turn_direction, time_turn_direction)

            # Escolhe o contorno correto da linha.
            if len(contours_blk) > 0:
                linha_detectada_frame = True
                line_detected.value = True
                preferir_esquerda = bool(preferencia_linha_esquerda.value)
                alvo_verde = green_turn_target.value
                direcao_marcada = (
                    "left" if alvo_verde < 0 else
                    "right" if alvo_verde > 0 else turn_dir.value
                )
                direcao_geometria = (
                    "straight" if preferir_esquerda else direcao_marcada
                )
                preferir_esquerda_geometria = preferir_esquerda
                blackline, black_line_crop = determine_correct_line(
                    contours_blk,
                    preferir_esquerda=preferir_esquerda_geometria,
                    turn_direction=direcao_geometria,
                )
                area_linha_frame = float(cv2.contourArea(blackline))
                line_size.value = area_linha_frame

                # Calcula a geometria do gap.
                if line_status.value == "gap_detected":
                    publish_gap_geometry(blackline, cv2_img if debug else None)
                else:
                    reset_gap_values()

                # Calcula o ângulo de correção.
                last_bottom_point_x = float(get_time_average(time_last_bottom_point_x, .15))
                last_average_line_point = float(get_time_average(time_last_average_line_point, .15))

                line_angle.value, poi, bottom_point = calculate_angle(
                    blackline, black_line_crop,
                    float(get_time_average(time_line_angle, .3)),
                    direcao_geometria,
                    last_bottom_point_x,
                    last_average_line_point,
                    preferir_esquerda=preferir_esquerda_geometria,
                )
                angulo_frame = float(line_angle.value)
                line_angle_y.value = int(poi[1])
                ponto_alvo_x_frame = float(poi[0])
                ponto_alvo_y_frame = float(poi[1])

                ponto_futuro = extrair_ponto_futuro(
                    blackline,
                    mascara_linha=black_image,
                    origem_x=bottom_point[0],
                )
                ponto_futuro_x_frame = ponto_futuro.x
                ponto_futuro_y_frame = ponto_futuro.y
                ponto_futuro_valido_frame = ponto_futuro.valido



                time_line_angle = add_time_value(time_line_angle, line_angle.value)
                time_last_bottom_point_x = add_time_value(time_last_bottom_point_x, bottom_point[0])

                # Projeta o vetor da parte inferior até o topo da imagem.
                if bottom_point[0] != poi[0] and bottom_point[1] != poi[1]:
                    slope = (bottom_point[1] - poi[1]) / (bottom_point[0] - poi[0])
                    x = min(max(poi[0] + (0 - poi[1]) / slope, 0), camera_x)
                else:
                    x = poi[0]

                time_last_average_line_point = add_time_value(time_last_average_line_point, x)

                bottom_y = bottom_point[1]
                ponto_inferior_x_frame = float(bottom_point[0])
                ponto_inferior_y_frame = float(bottom_point[1])

                # Publica os valores usados pelo controle.
                last_bottom_point.value = bottom_point[0]
                last_bottom_point_y.value = bottom_point[1]
                if debug:
                    cv2.drawContours(cv2_img, [blackline], -1, (255, 0, 0), 2)
                    cv2.circle(cv2_img, (int(last_average_line_point), 0), 5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (int(poi[0]), int(poi[1])), 5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(cv2_img, (int(bottom_point[0]), int(bottom_point[1])), 5, (255, 255, 0), 1, cv2.LINE_AA)
                    if ponto_futuro.valido:
                        alvo_futuro = (
                            int(round(ponto_futuro.x)),
                            int(round(ponto_futuro.y)),
                        )
                        cv2.line(
                            cv2_img,
                            (camera_x // 2, camera_y - 1),
                            alvo_futuro,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            cv2_img, alvo_futuro, 6,
                            (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(cv2_img, (camera_x // 2, camera_y - 4),
                               5, (255, 0, 0), -1, cv2.LINE_AA)

            else:
                linha_detectada_frame = False
                line_detected.value = False
                line_angle.value = 0
                line_size.value = 0
                last_bottom_point_y.value = 0
                line_angle_y.value = -1
                reset_gap_values()

            # O modelo da entrada só roda na missão. O primeiro candidato usa
            # o mesmo frame em que a linha preta foi encontrada e centralizada;
            # depois de parar, a prata pode naturalmente cobrir esse fim.
            linha_alinhada = (
                linha_detectada_frame
                and abs(angulo_frame) <= config.ENTRY_LINE_MAX_ANGLE
                and abs(ponto_inferior_x_frame - camera_x / 2)
                <= config.ENTRY_LINE_MAX_BOTTOM_ERROR_PX
            )
            if linha_alinhada:
                ultimo_alinhamento_entrada = frame_captured_at
            # A faixa prata naturalmente tapa/termina a linha preta. Assim,
            # o modelo recebe até 0,5 s após o último alinhamento real, mas
            # nunca quando o robô já começou uma correção de linha perdida.
            entrada_alinhada = (
                frame_captured_at - ultimo_alinhamento_entrada
                <= config.ENTRY_ALIGNMENT_HOLD_S
            )
            update_entry_silver(
                entry_gate, cv2_img, frame_captured_at,
                line_aligned=entrada_alinhada,
                black_mask=entry_black_mask,
                ramp_black_mask=entry_ramp_black_mask)

            processamento_ms = (
                time.perf_counter() - inicio_processamento
            ) * 1000.
            publicar_resultado_visao_rapida(
                publicado_em=time.monotonic(),
                processamento_ms=processamento_ms,
                linha_detectada=linha_detectada_frame,
                linha_a_frente=linha_a_frente_frame,
                angulo=angulo_frame,
                ponto_inferior_x=ponto_inferior_x_frame,
                ponto_inferior_y=ponto_inferior_y_frame,
                ponto_alvo_x=ponto_alvo_x_frame,
                ponto_alvo_y=ponto_alvo_y_frame,
                area_linha=area_linha_frame,
                candidato_verde=candidato_verde_frame,
                candidato_vermelho=candidato_vermelho_frame,
                ponto_futuro_x=ponto_futuro_x_frame,
                ponto_futuro_y=ponto_futuro_y_frame,
                ponto_futuro_valido=ponto_futuro_valido_frame,
            )

            if not vision_ready.value:
                vision_ready.value = True
                print("[visão] primeiro frame processado — pipeline ativo")

            # FPS
            counter += 1
            if time.perf_counter() - fps_time > 1:
                fps = int(counter / (time.perf_counter() - fps_time))
                fps_time = time.perf_counter()
                counter = 0

            if debug:
                # A caixa vermelha desenhada por `check_green` é um candidato
                # VERDE, não a faixa prata. Mostre a entrada com outra cor e
                # com seus votos para o debug não induzir ao diagnóstico
                # errado.
                if entry_gate is not None and entry_armed.value:
                    entrada = entry_gate.last_detection
                    if entrada is not None:
                        x, y, w, h = entrada.bbox
                        cv2.rectangle(
                            cv2_img,
                            (int(x), int(y)),
                            (int(x + w - 1), int(y + h - 1)),
                            (255, 255, 0),
                            2,
                        )
                    motivo_entrada = (
                        entry_gate.last_reason or "candidata")
                    confianca_entrada = (
                        entrada.confidence if entrada is not None else 0.)
                    confianca_bruta = getattr(
                        getattr(entry_gate, "model", None),
                        "last_confidence",
                        None,
                    )
                    if entrada is None and confianca_bruta is not None:
                        confianca_entrada = confianca_bruta
                    cv2.putText(
                        cv2_img,
                        f"ONNX PRATA {entry_gate.votes}/"
                        f"{config.ENTRY_SILVER_VOTE_WINDOW} "
                        f"conf={confianca_entrada:.2f}/"
                        f"{config.ENTRY_MODEL_MIN_CONFIDENCE:.2f} "
                        f"{motivo_entrada}",
                        (5, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .35,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                nomes_controle = ("TRACK", "CORNER", "LOST", "SPECIAL")
                indice_controle = int(steering_state.value)
                nome_controle = (
                    nomes_controle[indice_controle]
                    if 0 <= indice_controle < len(nomes_controle)
                    else "?"
                )
                cv2.putText(cv2_img, f"{fps} fps  ang={line_angle.value}  {line_status.value}",
                            (5, camera_y - 8), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.putText(
                    cv2_img,
                    f"ctl={nome_controle} c={steering_correction.value:+.2f} "
                    f"L={steering_left_pwm.value} R={steering_right_pwm.value}",
                    (5, camera_y - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    cv2_img,
                    f"lateral={steering_lateral_error.value:+.2f} "
                    f"rumo={steering_heading.value:+.0f}deg",
                    (5, camera_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(cv2_img, str(status.value), (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.putText(
                    cv2_img,
                    f"proc={processamento_ms:.1f}ms  "
                    f"captura={camera.capture_fps:.0f}fps",
                    (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .35,
                    (0, 255, 255),
                    1,
                )
                shm_array[:] = cv2_img

    finally:
        if entry_gate is not None:
            entry_gate.close()
        camera.close()
        if shm is not None:
            shm.close()
