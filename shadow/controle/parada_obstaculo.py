"""Confirma um obstáculo próximo sem bloquear o segue-linha."""

from collections import deque
import statistics
import time

from config import (
    MAX_PWM,
    OBSTACLE_CONFIRM_READINGS,
    OBSTACLE_CONFIRM_WINDOW_S,
    OBSTACLE_HISTORY_SIZE,
    OBSTACLE_LATERAL_PWM,
    OBSTACLE_LATERAL_TIME_S,
    OBSTACLE_MAX_VALID_MM,
    OBSTACLE_MIN_VALID_MM,
    OBSTACLE_READ_TIMEOUT_S,
    OBSTACLE_SAMPLE_INTERVAL_S,
    OBSTACLE_STOP_DISTANCE_MM,
)


class MonitorObstaculo:
    """Lê o ultrassônico e trava após confirmação 2-de-3."""

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

        self._leituras = deque(maxlen=int(tamanho_historico))
        self._proxima_solicitacao = 0.0
        self.parada_confirmada = False
        self.distancia_confirmada_mm = None

    def atualizar(self, arduino, agora=None):
        """Atualiza uma vez e retorna True quando a parada estiver travada."""
        agora = time.monotonic() if agora is None else float(agora)

        if self.parada_confirmada:
            return True

        concluido, distancia_mm = arduino.poll_ultrassom()
        if concluido:
            self._registrar_leitura(agora, distancia_mm)

        self._descartar_antigas(agora)

        if agora >= self._proxima_solicitacao:
            if arduino.iniciar_ultrassom(timeout=self.timeout_s):
                self._proxima_solicitacao = agora + self.intervalo_s

        return self.parada_confirmada

    def _registrar_leitura(self, agora, distancia_mm):
        # None significa ausência de eco. Não confirma obstáculo nem é usado
        # como uma falsa leitura de distância livre.
        if distancia_mm is None:
            return

        distancia_mm = int(distancia_mm)
        if not (
            self.distancia_minima_mm
            <= distancia_mm
            <= self.distancia_maxima_mm
        ):
            return

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

    def _descartar_antigas(self, agora):
        limite = agora - self.janela_s
        while self._leituras and self._leituras[0][0] < limite:
            self._leituras.popleft()


def deslizar_para_esquerda(
    arduino,
    pwm=OBSTACLE_LATERAL_PWM,
    duracao_s=OBSTACLE_LATERAL_TIME_S,
    deve_encerrar=None,
    relogio=time.monotonic,
    dormir=time.sleep,
):
    """Desliza para a esquerda com as quatro rodas e termina parado.

    Este é um movimento lateral de rodas omnidirecionais em X, não um pivô:
    FE e TD giram para trás enquanto TE e FD giram para frente.
    """
    pwm = int(round(pwm))
    duracao_s = float(duracao_s)
    deve_encerrar = deve_encerrar or (lambda: False)

    if not 1 <= pwm <= MAX_PWM:
        raise ValueError(f"PWM lateral deve ficar entre 1 e {MAX_PWM}")
    if duracao_s <= 0:
        raise ValueError("duracao lateral deve ser positiva")

    # Não deixa o último comando do segue-linha se misturar com o lateral.
    if arduino.parar() is False:
        raise RuntimeError("não foi possível parar antes do desvio lateral")

    epoca_serial = arduino.connection_epoch
    try:
        if arduino.rodas(-pwm, pwm, pwm, -pwm) is False:
            raise RuntimeError("não foi possível iniciar o desvio lateral")

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
                    "conexão serial mudou durante o desvio lateral")
            dormir(min(.05, restante))
    finally:
        # Garante PARAR tanto no fim normal quanto em Ctrl+C ou falha serial.
        arduino.parar()
