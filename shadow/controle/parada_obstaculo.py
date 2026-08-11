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

def procurar_continuacao_saida_pivo_dianteiro(
    arduino,
    orientacao_ramificacao,
    pwm=None,
    duracao_primeira_s=None,
    duracao_cruzada_s=None,
    confirmacao_s=None,
    deve_encerrar=None,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Procura a linha pos-resgate girando somente as rodas dianteiras.

    A propria faixa transversal tambem e preta, portanto
    ``orientacao_ramificacao`` deve devolver esquerda, centro, direita ou
    ``None`` usando a extremidade distante do trajeto, e nao a presenca
    generica de um contorno preto. Para nao confundir a faixa transversal com
    o percurso, a varredura esquerda-direita e sempre executada e somente a
    ponta distante centralizada confirma o ramo correto. Retorna ``"centro"``
    ou ``None`` se os dois lados falharem.

    O movimento inverte o pivo de 90 graus do segue-linha: TE e TD ficam em
    zero, enquanto FE e FD giram em sentidos opostos. Isso mantem a traseira
    como apoio e desloca a frente ate a ramificacao correta cruzar o centro
    da camera.
    """
    import config

    pwm = int(round(
        config.EXIT_LINE_FRONT_PIVOT_PWM if pwm is None else pwm))
    duracao_primeira_s = float(
        config.EXIT_LINE_FRONT_PIVOT_FIRST_S
        if duracao_primeira_s is None else duracao_primeira_s)
    duracao_cruzada_s = float(
        config.EXIT_LINE_FRONT_PIVOT_CROSS_S
        if duracao_cruzada_s is None else duracao_cruzada_s)
    confirmacao_s = float(
        config.EXIT_LINE_FRONT_PIVOT_CONFIRM_S
        if confirmacao_s is None else confirmacao_s)
    deve_encerrar = deve_encerrar or (lambda: False)

    if not 1 <= pwm <= MAX_PWM:
        raise ValueError(
            f"PWM do pivo dianteiro deve ficar entre 1 e {MAX_PWM}")
    if duracao_primeira_s <= 0 or duracao_cruzada_s <= 0:
        raise ValueError("duracoes do pivo dianteiro devem ser positivas")

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
                "conexao serial mudou durante o pivo dianteiro da saida")

    def ler_orientacao():
        orientacao = orientacao_ramificacao()
        if orientacao in ("esquerda", "centro", "direita"):
            return orientacao
        return None

    def confirmar_ramo_centralizado():
        if ler_orientacao() != "centro":
            return None
        inicio_confirmacao = relogio()
        while not deve_encerrar():
            orientacao_atual = ler_orientacao()
            if orientacao_atual != "centro":
                return None
            restante = confirmacao_s - (relogio() - inicio_confirmacao)
            if restante <= 1e-9:
                return "centro"
            atualizar_serial()
            dormir(min(.025, restante))
        return None

    def varrer(enviar_comando, duracao_s, etapa, procurar=True):
        movimento_acumulado = 0.0
        inicio_movimento = None
        try:
            if enviar_comando() is False:
                raise RuntimeError(f"nao foi possivel iniciar {etapa}")
            inicio_movimento = relogio()

            while not deve_encerrar():
                agora = relogio()
                movimento_atual = agora - inicio_movimento
                if procurar and ler_orientacao() == "centro":
                    movimento_acumulado += movimento_atual
                    inicio_movimento = None
                    if arduino.parar() is False:
                        raise RuntimeError(
                            f"nao foi possivel frear durante {etapa}")
                    orientacao_confirmada = confirmar_ramo_centralizado()
                    if orientacao_confirmada is not None:
                        return orientacao_confirmada
                    if movimento_acumulado >= duracao_s - 1e-9:
                        return None
                    if enviar_comando() is False:
                        raise RuntimeError(
                            f"nao foi possivel retomar {etapa}")
                    inicio_movimento = relogio()
                    continue

                restante = duracao_s - movimento_acumulado - movimento_atual
                if restante <= 1e-9:
                    return None
                atualizar_serial()
                dormir(min(.025, restante))
            return None
        finally:
            arduino.parar()

    try:
        if deve_encerrar():
            return None

        # A faixa preta ainda ocupa boa parte da imagem no primeiro frame.
        # Fazer as duas passagens, mesmo vendo algo lateral, obriga o ramo do
        # trajeto a atravessar o centro antes de ser escolhido.
        primeiro_lado = "esquerda"
        segundo_lado = "direita"
        comandos = {
            "esquerda": lambda: arduino.rodas(-pwm, 0, pwm, 0),
            "direita": lambda: arduino.rodas(pwm, 0, -pwm, 0),
        }

        orientacao_confirmada = varrer(
            comandos[primeiro_lado],
            duracao_primeira_s,
            f"o pivo dianteiro {primeiro_lado} da linha de saida",
            procurar=False,
        )
        if orientacao_confirmada is not None:
            return orientacao_confirmada
        if deve_encerrar():
            return None

        orientacao_confirmada = varrer(
            comandos[segundo_lado],
            duracao_cruzada_s,
            f"o pivo dianteiro {segundo_lado} da linha de saida",
        )
        if orientacao_confirmada is not None:
            return orientacao_confirmada
        return None
    finally:
        arduino.parar()
