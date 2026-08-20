"""Agenda MPU e um ultrassom por vez sem disputar a serial do resgate."""

import time

import config_resgate as cfg


class MonitorSensoresSaida:
    """Entrega leituras novas ao controlador e mantem uma consulta por vez.

    Primeiro a manobra usa o ultrassom lateral para alinhar. Depois usa o
    frontal para parar a 118 mm da parede. Em ambos os casos, o MPU alterna
    com o unico HC-SR04 necessario naquele estado.
    """

    def __init__(self, arduino):
        self._arduino = arduino
        self._lado_ultrassom_pendente = None
        self._proximo_sensor = "MPU"
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
    ):
        """Inicia no maximo uma consulta nao bloqueante, se estiver vencida.

        Durante um giro, a referencia de yaw tem precedencia. A parede nao
        muda enquanto o robô gira parado; alternar com o HC-SR04 nesse instante
        pode envelhecer o yaw e fazer a manobra abortar mesmo com o MPU bom.
        """
        instante = time.monotonic() if agora is None else float(agora)
        lado_ultrassom = str(lado_ultrassom).upper()
        if lado_ultrassom not in ("FRENTE", "LATERAL"):
            raise ValueError("lado_ultrassom deve ser FRENTE ou LATERAL")
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
                lado = lado_ultrassom
                if self._arduino.iniciar_ultrassom(
                    timeout=cfg.SAIDA_PAREDE_TIMEOUT_ULTRASSOM_S,
                    lado=lado,
                ):
                    self._lado_ultrassom_pendente = lado
                    self._proxima_leitura_ultrassom = (
                        instante + cfg.SAIDA_PAREDE_INTERVALO_ULTRASSOM_S)
                    self._proximo_sensor = "MPU"
                    return True
        return False

    def cancelar(self):
        """Descarta pedidos da rota antes de trocar de camera ou zerar MPU."""
        self._arduino.cancelar_ultrassom()
        self._arduino.cancelar_mpu()
        self._lado_ultrassom_pendente = None
