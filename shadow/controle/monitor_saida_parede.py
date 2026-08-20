"""Agenda MPU e um ultrassom por vez sem disputar a serial do resgate."""

import time

import config_resgate as cfg


class MonitorSensoresSaida:
    """Entrega leituras novas ao controlador e mantem uma consulta por vez.

    Conforme a etapa, consulta o frontal, o lateral ou ambos em rodizio. O
    MPU ocupa os intervalos entre leituras de distancia, sem duas consultas
    concorrentes na serial.
    """

    def __init__(self, arduino):
        self._arduino = arduino
        self._lado_ultrassom_pendente = None
        self._proximo_sensor = "MPU"
        self._proximo_lado_ultrassom = 0
        self._proxima_leitura_mpu = 0.0
        self._proxima_leitura_ultrassom = 0.0

    def atualizar_controlador(self, controlador, agora=None):
        """Consome respostas prontas e as registra com o mesmo timestamp."""
        instante = time.monotonic() if agora is None else float(agora)
        concluiu_mpu, leitura_mpu = self._arduino.poll_mpu()
        if concluiu_mpu and leitura_mpu is not None:
            controlador.observar_mpu(leitura_mpu.yaw_graus, instante)

        concluiu_ultrassom, distancia_mm = self._arduino.poll_ultrassom()
        if (
            concluiu_ultrassom
            and self._lado_ultrassom_pendente is not None
        ):
            controlador.observar_ultrassom(
                self._lado_ultrassom_pendente,
                distancia_mm,
                self._arduino.ultima_leitura_ultrassom_respondeu,
                instante,
            )
            self._lado_ultrassom_pendente = None

    def agendar_proxima(
        self,
        agora=None,
        priorizar_mpu=False,
        lado_ultrassom="LATERAL",
        lados_ultrassom=None,
    ):
        """Inicia no maximo uma consulta nao bloqueante, se estiver vencida.

        Durante um giro, a referencia de yaw tem precedencia. A parede nao
        muda enquanto o robô gira parado; alternar com o HC-SR04 nesse instante
        pode envelhecer o yaw e fazer a manobra abortar mesmo com o MPU bom.
        """
        instante = time.monotonic() if agora is None else float(agora)
        if lados_ultrassom is None:
            lados = (str(lado_ultrassom).upper(),)
        else:
            lados = tuple(str(lado).upper() for lado in lados_ultrassom)
        if not lados or any(lado not in ("FRENTE", "LATERAL") for lado in lados):
            raise ValueError(
                "lados_ultrassom deve conter somente FRENTE e/ou LATERAL")
        if self._arduino.consultas_sensores_pendentes:
            return False

        if priorizar_mpu:
            if instante < self._proxima_leitura_mpu:
                return False
            if self._arduino.iniciar_mpu(
                timeout=cfg.SAIDA_PAREDE_TIMEOUT_MPU_S,
            ):
                self._proxima_leitura_mpu = (
                    instante + cfg.SAIDA_PAREDE_INTERVALO_MPU_S)
                self._proximo_sensor = "ULTRASSOM"
                return True
            return False

        ordem = (
            ("MPU", "ULTRASSOM")
            if self._proximo_sensor == "MPU"
            else ("ULTRASSOM", "MPU")
        )
        for sensor in ordem:
            if sensor == "MPU" and instante >= self._proxima_leitura_mpu:
                if self._arduino.iniciar_mpu(
                    timeout=cfg.SAIDA_PAREDE_TIMEOUT_MPU_S,
                ):
                    self._proxima_leitura_mpu = (
                        instante + cfg.SAIDA_PAREDE_INTERVALO_MPU_S)
                    self._proximo_sensor = "ULTRASSOM"
                    return True
            if (
                sensor == "ULTRASSOM"
                and instante >= self._proxima_leitura_ultrassom
            ):
                lado = lados[self._proximo_lado_ultrassom % len(lados)]
                if self._arduino.iniciar_ultrassom(
                    timeout=cfg.SAIDA_PAREDE_TIMEOUT_ULTRASSOM_S,
                    lado=lado,
                ):
                    self._lado_ultrassom_pendente = lado
                    self._proxima_leitura_ultrassom = (
                        instante + cfg.SAIDA_PAREDE_INTERVALO_ULTRASSOM_S)
                    self._proximo_sensor = "MPU"
                    self._proximo_lado_ultrassom = (
                        self._proximo_lado_ultrassom + 1) % len(lados)
                    return True
        return False

    def cancelar(self):
        """Descarta pedidos da rota antes de trocar de camera ou zerar MPU."""
        self._arduino.cancelar_ultrassom()
        self._arduino.cancelar_mpu()
        self._lado_ultrassom_pendente = None
