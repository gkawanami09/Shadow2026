"""Confirma um obstáculo próximo sem bloquear o segue-linha."""

from collections import deque
import statistics
import time

from config import (
    MAX_PWM,
    OBSTACLE_CONFIRM_READINGS,
    OBSTACLE_CONFIRM_WINDOW_S,
    OBSTACLE_FAST_SPEED_BLOCK_MM,
    OBSTACLE_FORWARD_PWM,
    OBSTACLE_FORWARD_TIME_S,
    OBSTACLE_HISTORY_SIZE,
    OBSTACLE_LATERAL_PWM,
    OBSTACLE_LATERAL_TIME_S,
    OBSTACLE_LINE_CONFIRM_TIME_S,
    OBSTACLE_LINE_SEARCH_PWM,
    OBSTACLE_LINE_SEARCH_TIMEOUT_S,
    OBSTACLE_MAX_VALID_MM,
    OBSTACLE_MIN_VALID_MM,
    OBSTACLE_READ_TIMEOUT_S,
    OBSTACLE_SAMPLE_INTERVAL_S,
    OBSTACLE_STOP_DISTANCE_MM,
)


class MonitorObstaculo:
    """Le o ultrassonico e trava apos a quantidade configurada de leituras."""

    def __init__(
        self,
        distancia_parada_mm=OBSTACLE_STOP_DISTANCE_MM,
        intervalo_s=OBSTACLE_SAMPLE_INTERVAL_S,
        timeout_s=OBSTACLE_READ_TIMEOUT_S,
        confirmacoes=OBSTACLE_CONFIRM_READINGS,
        tamanho_historico=OBSTACLE_HISTORY_SIZE,
        janela_s=OBSTACLE_CONFIRM_WINDOW_S,
        distancia_minima_mm=OBSTACLE_MIN_VALID_MM,
        distancia_maxima_mm=OBSTACLE_MAX_VALID_MM,
        distancia_bloqueio_rapido_mm=OBSTACLE_FAST_SPEED_BLOCK_MM,
    ):
        if not 1 <= confirmacoes <= tamanho_historico:
            raise ValueError(
                "confirmacoes deve ficar entre 1 e tamanho_historico")

        self.distancia_parada_mm = int(distancia_parada_mm)
        self.intervalo_s = float(intervalo_s)
        self.timeout_s = float(timeout_s)
        self.confirmacoes = int(confirmacoes)
        self.janela_s = float(janela_s)
        self.distancia_minima_mm = int(distancia_minima_mm)
        self.distancia_maxima_mm = int(distancia_maxima_mm)
        self.distancia_bloqueio_rapido_mm = int(
            distancia_bloqueio_rapido_mm)

        self._leituras = deque(maxlen=int(tamanho_historico))
        self._proxima_solicitacao = 0.0
        self.parada_confirmada = False
        self.distancia_confirmada_mm = None
        self.leituras_concluidas = 0
        self.leituras_invalidas_consecutivas = 0
        self.ultima_distancia_valida_mm = None
        self.ultima_leitura_valida_em = None

    @property
    def bloqueia_velocidade_rapida(self):
        """Uma leitura próxima já desacelera, mas não confirma o obstáculo."""
        return any(
            distancia <= self.distancia_bloqueio_rapido_mm
            for _, distancia, _ in self._leituras
        )

    @property
    def distancias_validas(self):
        """Cópia das medidas válidas ainda presentes na janela temporal."""
        return tuple(distancia for _, distancia, _ in self._leituras)

    def atualizar(self, arduino, agora=None):
        """Atualiza uma vez e retorna True quando a parada estiver travada."""
        agora = time.monotonic() if agora is None else float(agora)

        if self.parada_confirmada:
            return True

        concluido, distancia_mm = arduino.poll_ultrassom()
        if concluido:
            self.leituras_concluidas += 1
            if self._registrar_leitura(agora, distancia_mm):
                self.leituras_invalidas_consecutivas = 0
            else:
                self.leituras_invalidas_consecutivas += 1
            # Ao confirmar, nao abra uma terceira medicao. Isso deixa a
            # serial livre para o comando de parada e para o deposito.
            if self.parada_confirmada:
                return True

        self._descartar_antigas(agora)

        if agora >= self._proxima_solicitacao:
            if arduino.iniciar_ultrassom(timeout=self.timeout_s):
                self._proxima_solicitacao = agora + self.intervalo_s

        return self.parada_confirmada

    def _registrar_leitura(self, agora, distancia_mm):
        # None significa ausência de eco. Não confirma obstáculo nem é usado
        # como uma falsa leitura de distância livre.
        if distancia_mm is None:
            return False

        distancia_mm = int(distancia_mm)
        if not (
            self.distancia_minima_mm
            <= distancia_mm
            <= self.distancia_maxima_mm
        ):
            return False

        self.ultima_distancia_valida_mm = distancia_mm
        self.ultima_leitura_valida_em = agora

        proxima = distancia_mm <= self.distancia_parada_mm
        self._leituras.append((agora, distancia_mm, proxima))
        self._descartar_antigas(agora)

        leituras_proximas = [
            distancia
            for _, distancia, esta_proxima in self._leituras
            if esta_proxima
        ]
        if len(leituras_proximas) >= self.confirmacoes:
            self.parada_confirmada = True
            self.distancia_confirmada_mm = int(round(
                statistics.median(leituras_proximas)))
        return True

    def _descartar_antigas(self, agora):
        limite = agora - self.janela_s
        while self._leituras and self._leituras[0][0] < limite:
            self._leituras.popleft()

    def reiniciar(self):
        """Libera o monitor para detectar outro obstáculo futuramente."""
        self._leituras.clear()
        self._proxima_solicitacao = 0.0
        self.parada_confirmada = False
        self.distancia_confirmada_mm = None
        self.leituras_concluidas = 0
        self.leituras_invalidas_consecutivas = 0
        self.ultima_distancia_valida_mm = None
        self.ultima_leitura_valida_em = None

    def cancelar(self, arduino):
        """Cancela uma medição pendente sem expor a serial ao coordenador."""
        arduino.cancelar_ultrassom()
        self._proxima_solicitacao = 0.0


def desviar_obstaculo(
    arduino,
    pwm_lateral=OBSTACLE_LATERAL_PWM,
    duracao_lateral_s=OBSTACLE_LATERAL_TIME_S,
    pwm_avanco=OBSTACLE_FORWARD_PWM,
    duracao_avanco_s=OBSTACLE_FORWARD_TIME_S,
    deve_encerrar=None,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Desliza à esquerda, avança, volta à direita e para.

    As duas etapas laterais usam as rodas omnidirecionais em X e têm a mesma
    duração, preservando a orientação do robô durante todo o desvio.
    """
    pwm_lateral = int(round(pwm_lateral))
    pwm_avanco = int(round(pwm_avanco))
    duracao_lateral_s = float(duracao_lateral_s)
    duracao_avanco_s = float(duracao_avanco_s)
    deve_encerrar = deve_encerrar or (lambda: False)

    if not 1 <= pwm_lateral <= MAX_PWM:
        raise ValueError(f"PWM lateral deve ficar entre 1 e {MAX_PWM}")
    if not 1 <= pwm_avanco <= MAX_PWM:
        raise ValueError(f"PWM de avanço deve ficar entre 1 e {MAX_PWM}")
    if duracao_lateral_s <= 0:
        raise ValueError("duracao lateral deve ser positiva")
    if duracao_avanco_s <= 0:
        raise ValueError("duracao de avanço deve ser positiva")

    # Não deixa o último comando do segue-linha se misturar com o desvio.
    if arduino.parar() is False:
        raise RuntimeError("não foi possível parar antes do desvio")

    epoca_serial = arduino.connection_epoch

    def movimentar(enviar_comando, duracao_s, etapa):
        if enviar_comando() is False:
            raise RuntimeError(f"não foi possível iniciar {etapa}")

        fim = relogio() + duracao_s
        while not deve_encerrar():
            restante = fim - relogio()
            if restante <= 0:
                break

            arduino.refresh(fail_closed=True)
            if (
                not arduino.connected
                or arduino.connection_epoch != epoca_serial
            ):
                raise RuntimeError(
                    f"conexão serial mudou durante {etapa}")
            dormir(min(.05, restante))

    try:
        movimentar(
            lambda: arduino.rodas(
                -pwm_lateral,
                pwm_lateral,
                pwm_lateral,
                -pwm_lateral,
            ),
            duracao_lateral_s,
            "o desvio lateral",
        )
        if not deve_encerrar():
            movimentar(
                lambda: arduino.rodas(
                    pwm_avanco,
                    pwm_avanco,
                    pwm_avanco,
                    pwm_avanco,
                ),
                duracao_avanco_s,
                "o avanço",
            )
        if not deve_encerrar():
            movimentar(
                lambda: arduino.rodas(
                    pwm_lateral,
                    -pwm_lateral,
                    -pwm_lateral,
                    pwm_lateral,
                ),
                duracao_lateral_s,
                "o retorno lateral à direita",
            )
    finally:
        # Garante PARAR tanto no fim normal quanto em Ctrl+C ou falha serial.
        arduino.parar()


def _movimentar_ate_confirmar(
    arduino,
    enviar_comando,
    condicao,
    timeout_s,
    confirmacao_s,
    etapa,
    deve_encerrar=None,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Mantém um movimento até uma condição visual permanecer confirmada."""
    timeout_s = float(timeout_s)
    confirmacao_s = float(confirmacao_s)
    deve_encerrar = deve_encerrar or (lambda: False)

    if timeout_s <= 0:
        raise ValueError("timeout do movimento deve ser positivo")
    if confirmacao_s < 0:
        raise ValueError("tempo de confirmação não pode ser negativo")
    if arduino.parar() is False:
        raise RuntimeError(f"não foi possível parar antes de {etapa}")

    epoca_serial = arduino.connection_epoch
    inicio = relogio()
    confirmada_desde = None
    try:
        if enviar_comando() is False:
            raise RuntimeError(f"não foi possível iniciar {etapa}")

        while not deve_encerrar():
            agora = relogio()
            if condicao():
                if confirmada_desde is None:
                    confirmada_desde = agora
                if agora - confirmada_desde >= confirmacao_s:
                    return True
            else:
                confirmada_desde = None

            restante = timeout_s - (agora - inicio)
            if restante <= 1e-9:
                return False

            arduino.refresh(fail_closed=True)
            if (
                not arduino.connected
                or arduino.connection_epoch != epoca_serial
            ):
                raise RuntimeError(
                    f"conexão serial mudou durante {etapa}")
            dormir(min(.05, restante))
        return False
    finally:
        arduino.parar()


def avancar_ate_linha(
    arduino,
    linha_proxima,
    pwm=OBSTACLE_LINE_SEARCH_PWM,
    timeout_s=OBSTACLE_LINE_SEARCH_TIMEOUT_S,
    confirmacao_s=OBSTACLE_LINE_CONFIRM_TIME_S,
    deve_encerrar=None,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Avança até a linha estar próxima e centralizada pela função recebida."""
    pwm = int(round(pwm))
    if not 1 <= pwm <= MAX_PWM:
        raise ValueError(f"PWM de busca deve ficar entre 1 e {MAX_PWM}")

    return _movimentar_ate_confirmar(
        arduino,
        lambda: arduino.lado(pwm, pwm),
        linha_proxima,
        timeout_s,
        confirmacao_s,
        "a busca da linha",
        deve_encerrar,
        relogio,
        dormir,
    )


def orientacao_continuacao_saida(resultado, agora=None):
    """Devolve o lado da ramificacao ou ``centro`` quando ja alinhada."""
    import config

    agora = time.monotonic() if agora is None else float(agora)
    if resultado is None or int(getattr(resultado, "sequencia", 0)) <= 0:
        return None
    idade = agora - float(getattr(resultado, "publicado_em", 0.0))
    if not -0.05 <= idade <= config.EXIT_LINE_CONTINUATION_MAX_AGE_S:
        return None
    if not bool(getattr(resultado, "continuacao_saida_detectada", False)):
        return None

    alvo_x = float(getattr(resultado, "continuacao_saida_x", -1.0))
    alvo_y = float(getattr(resultado, "continuacao_saida_y", -1.0))
    distancia = float(getattr(
        resultado, "continuacao_saida_distancia", 0.0))
    if not (
        distancia >= config.EXIT_CONTINUATION_MIN_TARGET_DISTANCE_RATIO
        and 0.0 <= alvo_x <= config.camera_x
        and 0.0 <= alvo_y <= config.camera_y
    ):
        return None

    erro_x = alvo_x - config.camera_x / 2
    tolerancia = (
        config.camera_x * config.EXIT_CONTINUATION_ALIGN_X_TOLERANCE_RATIO)
    if abs(erro_x) <= tolerancia:
        return "centro"
    return "esquerda" if erro_x < 0 else "direita"

def procurar_continuacao_saida_pulsada(
    arduino,
    orientacao_ramificacao,
    pwm=None,
    duracao_pulso_s=None,
    pausa_assentamento_s=None,
    observacao_s=None,
    confirmacao_s=None,
    pulsos_esquerda=None,
    pulsos_direita=None,
    re_inicial_s=None,
    avanco_tentativa_s=None,
    re_final_s=None,
    deve_encerrar=None,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Encontra a continuacao apos a saida por pulsos de tanque parados.

    A sequencia e propositalmente igual ao principio da busca pulsada de
    bolinhas: mover por pouco tempo, parar, deixar o chassi assentar e so
    entao olhar. A primeira varredura faz dois pulsos para a esquerda e
    quatro para a direita. Cada ponta distante centralizada e mapeada pela
    posicao do pulso; ao fim, o robo volta ao melhor ponto mapeado e o
    confirma parado. Assim a barra preta transversal nunca vence apenas por
    estar no quadro.

    Se a primeira passagem falhar, retorna ao rumo de inicio, avanca um pouco
    e repete. Depois da segunda falha, executa a re maior e devolve ``None``:
    o chamador reaproxima em frente com a propria camera de segue-linha.
    """
    import config

    pwm = int(round(
        config.EXIT_LINE_PULSE_PWM if pwm is None else pwm))
    duracao_pulso_s = float(
        config.EXIT_LINE_PULSE_S
        if duracao_pulso_s is None else duracao_pulso_s)
    pausa_assentamento_s = float(
        config.EXIT_LINE_PULSE_SETTLE_S
        if pausa_assentamento_s is None else pausa_assentamento_s)
    observacao_s = float(
        config.EXIT_LINE_PULSE_OBSERVE_S
        if observacao_s is None else observacao_s)
    confirmacao_s = float(
        config.EXIT_LINE_PULSE_CONFIRM_S
        if confirmacao_s is None else confirmacao_s)
    pulsos_esquerda = int(
        config.EXIT_LINE_PULSES_LEFT
        if pulsos_esquerda is None else pulsos_esquerda)
    pulsos_direita = int(
        config.EXIT_LINE_PULSES_RIGHT
        if pulsos_direita is None else pulsos_direita)
    re_inicial_s = float(
        config.EXIT_LINE_INITIAL_REVERSE_S
        if re_inicial_s is None else re_inicial_s)
    avanco_tentativa_s = float(
        config.EXIT_LINE_RETRY_FORWARD_S
        if avanco_tentativa_s is None else avanco_tentativa_s)
    re_final_s = float(
        config.EXIT_LINE_FINAL_REVERSE_S
        if re_final_s is None else re_final_s)
    deve_encerrar = deve_encerrar or (lambda: False)

    if not 1 <= pwm <= MAX_PWM:
        raise ValueError(f"PWM dos pulsos deve ficar entre 1 e {MAX_PWM}")
    if min(duracao_pulso_s, pausa_assentamento_s, observacao_s) <= 0:
        raise ValueError("duracoes da busca pulsada devem ser positivas")
    if min(pulsos_esquerda, pulsos_direita) < 1:
        raise ValueError("a busca precisa de pelo menos um pulso por lado")
    if min(re_inicial_s, avanco_tentativa_s, re_final_s) <= 0:
        raise ValueError("movimentos de re/aproximacao devem ser positivos")
    if confirmacao_s < 0:
        raise ValueError("tempo de confirmacao nao pode ser negativo")
    if arduino.parar() is False:
        raise RuntimeError("nao foi possivel parar antes da busca da saida")

    epoca_serial = arduino.connection_epoch

    def atualizar_serial():
        arduino.refresh(fail_closed=True)
        if (
            not arduino.connected
            or arduino.connection_epoch != epoca_serial
        ):
            raise RuntimeError(
                "conexao serial mudou durante a busca pulsada da saida")

    def ler_orientacao():
        orientacao = orientacao_ramificacao()
        if orientacao in ("esquerda", "centro", "direita"):
            return orientacao
        return None

    def mover(enviar_comando, duracao_s, etapa):
        if enviar_comando() is False:
            raise RuntimeError(f"nao foi possivel iniciar {etapa}")
        inicio = relogio()
        try:
            while not deve_encerrar():
                restante = duracao_s - (relogio() - inicio)
                if restante <= 1e-9:
                    return True
                atualizar_serial()
                dormir(min(.025, restante))
            return False
        finally:
            arduino.parar()

    def observar_parado(mapa, posicao):
        """Le apenas apos o settle e exige centro por tempo continuo."""
        inicio = relogio()
        centro_desde = None
        while not deve_encerrar():
            agora = relogio()
            orientacao = ler_orientacao()
            if orientacao is not None:
                mapa.append((posicao, orientacao))
            if orientacao == "centro":
                if centro_desde is None:
                    centro_desde = agora
                if agora - centro_desde >= confirmacao_s - 1e-9:
                    return True
            else:
                centro_desde = None
            restante = observacao_s - (agora - inicio)
            if restante <= 1e-9:
                return False
            atualizar_serial()
            dormir(min(.025, restante))
        return False

    comandos = {
        "esquerda": lambda: arduino.rodas(-pwm, -pwm, pwm, pwm),
        "direita": lambda: arduino.rodas(pwm, pwm, -pwm, -pwm),
    }

    def pulso(lado, mapa, posicao):
        if not mover(comandos[lado], duracao_pulso_s, f"pulso {lado}"):
            return None
        # Nao usa imagens durante o giro nem logo apos frear. Esta e a mesma
        # separacao giro -> parar -> settle -> observar da busca da bolinha.
        if not mover(lambda: arduino.parar(), pausa_assentamento_s,
                     "assentamento do pulso"):
            return None
        return observar_parado(mapa, posicao)

    def varrer_e_voltar():
        """Varre esquerda/direita e volta ao pulso com ramo centralizado."""
        posicao = 0
        posicao_candidata = None
        mapa = []

        for _ in range(pulsos_esquerda):
            posicao -= 1
            if pulso("esquerda", mapa, posicao):
                posicao_candidata = posicao

        for _ in range(pulsos_direita):
            posicao += 1
            if pulso("direita", mapa, posicao):
                # A ultima confirmacao e a mais confiavel: foi vista depois
                # de a camera percorrer os dois lados da faixa.
                posicao_candidata = posicao

        destino = posicao_candidata if posicao_candidata is not None else 0
        while posicao != destino and not deve_encerrar():
            lado = "esquerda" if destino < posicao else "direita"
            if not mover(comandos[lado], duracao_pulso_s,
                          f"retorno ao ramo {lado}"):
                return None
            posicao += -1 if lado == "esquerda" else 1

        if deve_encerrar() or posicao_candidata is None:
            return False
        if not mover(lambda: arduino.parar(), pausa_assentamento_s,
                     "assentamento da confirmacao final"):
            return None
        confirmado = observar_parado(mapa, posicao)
        if mapa:
            resumo = ", ".join(
                f"p{indice}:{lado}" for indice, lado in mapa[-12:])
            print(f"[controle] mapa da saida: {resumo}")
        return confirmado

    try:
        if deve_encerrar():
            return None

        if not mover(
            lambda: arduino.rodas(
                -config.EXIT_LINE_INITIAL_REVERSE_PWM,
                -config.EXIT_LINE_INITIAL_REVERSE_PWM,
                -config.EXIT_LINE_INITIAL_REVERSE_PWM,
                -config.EXIT_LINE_INITIAL_REVERSE_PWM,
            ),
            re_inicial_s,
            "a re curta antes da varredura",
        ):
            return None
        if varrer_e_voltar():
            return "centro"
        if deve_encerrar():
            return None

        if not mover(
            lambda: arduino.rodas(
                config.EXIT_LINE_RETRY_FORWARD_PWM,
                config.EXIT_LINE_RETRY_FORWARD_PWM,
                config.EXIT_LINE_RETRY_FORWARD_PWM,
                config.EXIT_LINE_RETRY_FORWARD_PWM,
            ),
            avanco_tentativa_s,
            "o avanco para a segunda varredura",
        ):
            return None
        if varrer_e_voltar():
            return "centro"
        if deve_encerrar():
            return None

        mover(
            lambda: arduino.rodas(
                -config.EXIT_LINE_FINAL_REVERSE_PWM,
                -config.EXIT_LINE_FINAL_REVERSE_PWM,
                -config.EXIT_LINE_FINAL_REVERSE_PWM,
                -config.EXIT_LINE_FINAL_REVERSE_PWM,
            ),
            re_final_s,
            "a re maior antes da reaproximacao",
        )
        return None
    finally:
        arduino.parar()
