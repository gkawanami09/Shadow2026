"""Retoma a terceira linha ainda dentro da sessao do resgate.

O controle e sincrono de proposito: a mesma rotina possui a camera de linha e
a serial, portanto cada observacao acontece depois do comando correspondente,
sem snapshots compartilhados ou frames em voo entre processos.
"""

from dataclasses import dataclass
import math
import statistics
import time

import config
import config_resgate as cfg
from visao.continuacao_saida import (
    DIREITA_BAIXA,
    ESQUERDA_BAIXA,
    NIVEL,
    AnalisadorSaidaPreta,
)


class ErroRetomadaSaida(RuntimeError):
    pass


@dataclass(frozen=True)
class ResultadoRetomadaSaida:
    orientacao_soleira: str
    delta_y_ratio: float
    fase_encontro: str
    continuacao: object


class ControladorRetomadaSaida:
    """Executa avanco, tanque ou omni e para ao confirmar a terceira linha."""

    def __init__(
        self,
        camera,
        arduino,
        acao_direcao,
        analisador=None,
        relogio=time.monotonic,
        dormir=time.sleep,
        debug_callback=None,
    ):
        self.camera = camera
        self.arduino = arduino
        self.acao_direcao = acao_direcao
        self.analisador = (
            AnalisadorSaidaPreta() if analisador is None else analisador)
        self.relogio = relogio
        self.dormir = dormir
        self.debug_callback = debug_callback
        self._epoca_serial = arduino.connection_epoch
        self._prazo_total = None
        self._fase = "inicio"

    def executar(self):
        inicio = self.relogio()
        self._prazo_total = inicio + cfg.EXIT_POST_TOTAL_TIMEOUT_S
        try:
            self._parar("antes de medir a inclinacao")
            orientacao, delta = self._confirmar_pose()
            print(
                "[saida] pose da soleira: "
                f"{orientacao} (direita-esquerda={delta:+.1%})")

            # O trecho pedido e sempre completo: ele leva as rodas para alem
            # da faixa transversal antes de qualquer entrega ao segue-linha.
            self._fase = "avanco_0_3s"
            self._mover_sem_visao(
                lambda: self.acao_direcao(
                    0, cfg.EXIT_POST_FORWARD_SPEED),
                cfg.EXIT_POST_FORWARD_S,
                "avanco reto de 0,3 s",
            )
            self._assentar()
            continuacao = self._confirmar_continuacao()
            if continuacao is not None:
                return self._resultado(
                    orientacao, delta, "apos_avanco", continuacao)

            if orientacao == DIREITA_BAIXA:
                # A parte baixa indica o lado do qual o robo esta vindo; para
                # apontar para a terceira linha, o tanque gira ao lado oposto.
                continuacao = self._buscar_com_tanque(direita=False)
                fase = "tanque_esquerda"
            elif orientacao == ESQUERDA_BAIXA:
                continuacao = self._buscar_com_tanque(direita=True)
                fase = "tanque_direita"
            elif orientacao == NIVEL:
                continuacao, fase = self._buscar_com_omni()
            else:
                raise ErroRetomadaSaida(
                    f"orientacao de soleira desconhecida: {orientacao}")

            if continuacao is None:
                raise ErroRetomadaSaida(
                    "a terceira linha nao apareceu dentro da varredura "
                    "fisicamente mapeada")
            return self._resultado(orientacao, delta, fase, continuacao)
        finally:
            # O processo de segue-linha assumira somente depois que resgate.py
            # fechar camera e serial. Ate la o chassi fica inequivocamente
            # parado, inclusive em timeout, Ctrl+C ou falha do HC-SR04.
            self._parar_melhor_esforco()

    def _resultado(self, orientacao, delta, fase, continuacao):
        print(
            "[saida] terceira linha confirmada em "
            f"{fase}; alvo=({continuacao.alvo_x:.0f}, "
            f"{continuacao.alvo_y:.0f}); PARADO para o handoff")
        return ResultadoRetomadaSaida(
            orientacao_soleira=orientacao,
            delta_y_ratio=float(delta),
            fase_encontro=fase,
            continuacao=continuacao,
        )

    def _confirmar_pose(self):
        deteccoes = []
        for _ in range(cfg.EXIT_POST_POSE_WINDOW):
            analise = self._capturar_analise("medindo_pose")
            if analise.soleira is not None:
                deteccoes.append(analise.soleira)
        if len(deteccoes) < cfg.EXIT_POST_POSE_VOTES:
            raise ErroRetomadaSaida(
                "nao foi possivel medir os dois lados da soleira preta")
        delta = float(statistics.median(
            deteccao.delta_y_ratio for deteccao in deteccoes))
        if delta > cfg.EXIT_POST_LEVEL_DELTA_RATIO:
            orientacao = DIREITA_BAIXA
        elif delta < -cfg.EXIT_POST_LEVEL_DELTA_RATIO:
            orientacao = ESQUERDA_BAIXA
        else:
            orientacao = NIVEL
        return orientacao, delta

    def _confirmar_continuacao(self):
        candidatas = []
        diagonal = math.hypot(config.camera_x, config.camera_y)
        tolerancia = (
            diagonal * cfg.EXIT_POST_CONTINUATION_TARGET_TOLERANCE_RATIO)
        for _ in range(cfg.EXIT_POST_CONTINUATION_WINDOW):
            analise = self._capturar_analise("confirmando_terceira_linha")
            candidata = analise.continuacao
            if candidata is None:
                continue
            if candidatas:
                referencia = candidatas[-1]
                distancia = math.hypot(
                    candidata.alvo_x - referencia.alvo_x,
                    candidata.alvo_y - referencia.alvo_y,
                )
                if distancia > tolerancia:
                    candidatas = [candidata]
                else:
                    candidatas.append(candidata)
            else:
                candidatas.append(candidata)
            if len(candidatas) >= cfg.EXIT_POST_CONTINUATION_VOTES:
                return candidatas[-1]
        return None

    def _buscar_com_tanque(self, direita):
        sentido = "direita" if direita else "esquerda"
        angulo = (
            cfg.EXIT_POST_TANK_ANGLE
            if direita else -cfg.EXIT_POST_TANK_ANGLE)
        restante = float(cfg.EXIT_POST_TANK_TIMEOUT_S)
        while restante > 1e-9:
            duracao = min(cfg.EXIT_POST_TANK_PULSE_S, restante)
            self._fase = f"tanque_{sentido}"
            inicio = self.relogio()
            encontrada = self._mover_monitorando(
                lambda: self.acao_direcao(
                    angulo, cfg.EXIT_POST_TANK_SPEED),
                duracao,
                f"pulso tanque para a {sentido}",
            )
            restante -= min(max(self.relogio() - inicio, 0.0), duracao)
            if encontrada is not None:
                return encontrada
            self._assentar()
            encontrada = self._confirmar_continuacao()
            if encontrada is not None:
                return encontrada
        return None

    def _buscar_com_omni(self):
        pwm = int(cfg.EXIT_POST_OMNI_PWM)
        etapas = (
            (
                "omni_esquerda",
                cfg.EXIT_POST_OMNI_LEFT_S,
                lambda: self.arduino.rodas(-pwm, pwm, pwm, -pwm),
            ),
            (
                "omni_direita",
                cfg.EXIT_POST_OMNI_RIGHT_S,
                lambda: self.arduino.rodas(pwm, -pwm, -pwm, pwm),
            ),
        )
        for fase, duracao, comando in etapas:
            self._fase = fase
            encontrada = self._mover_monitorando(
                comando, duracao, fase.replace("_", " "))
            if encontrada is not None:
                return encontrada, fase
            self._assentar()
            encontrada = self._confirmar_continuacao()
            if encontrada is not None:
                return encontrada, fase
        return None, "omni_completo"

    def _mover_sem_visao(self, enviar, duracao, descricao):
        self._enviar(enviar, descricao)
        fim = self.relogio() + float(duracao)
        try:
            while self.relogio() < fim:
                self._vigiar()
                self.dormir(min(0.02, max(fim - self.relogio(), 0.0)))
        finally:
            self._parar_melhor_esforco()

    def _mover_monitorando(self, enviar, duracao, descricao):
        restante = float(duracao)
        self._enviar(enviar, descricao)
        ultimo_instante = self.relogio()
        try:
            while restante > 1e-9:
                self._vigiar()
                analise = self._capturar_analise(descricao)
                agora = self.relogio()
                restante -= max(agora - ultimo_instante, 0.0)
                ultimo_instante = agora
                if analise.continuacao is not None:
                    # Primeiro candidato manda parar imediatamente. Ele so
                    # termina a manobra se reaparecer em frames parados.
                    self._parar("candidato da terceira linha")
                    self._assentar()
                    confirmada = self._confirmar_continuacao()
                    if confirmada is not None:
                        return confirmada
                    if restante > 1e-9:
                        self._enviar(enviar, f"retomada de {descricao}")
                        ultimo_instante = self.relogio()
                self.dormir(min(0.01, max(restante, 0.0)))
            return None
        finally:
            self._parar_melhor_esforco()

    def _capturar_analise(self, fase):
        self._vigiar()
        frame = self.camera.get_frame()
        analise = self.analisador.analisar(frame)
        if self.debug_callback is not None:
            continuar = self.debug_callback(frame, analise, fase)
            if continuar is False:
                raise ErroRetomadaSaida("debug da saida cancelado")
        self._vigiar()
        return analise

    def _assentar(self):
        self._parar("assentamento")
        fim = self.relogio() + cfg.EXIT_POST_SETTLE_S
        while self.relogio() < fim:
            self._vigiar()
            self.dormir(min(0.01, max(fim - self.relogio(), 0.0)))

    def _vigiar(self):
        agora = self.relogio()
        if self._prazo_total is not None and agora >= self._prazo_total:
            raise ErroRetomadaSaida("timeout total da retomada da linha")
        self.arduino.refresh(fail_closed=True)
        if (
            not self.arduino.connected
            or self.arduino.connection_epoch != self._epoca_serial
        ):
            raise ErroRetomadaSaida(
                "serial mudou durante a retomada da linha")

    def _enviar(self, enviar, descricao):
        self._vigiar()
        if enviar() is False:
            raise ErroRetomadaSaida(
                f"nao foi possivel iniciar {descricao}")

    def _parar(self, descricao):
        if self.acao_direcao() is False:
            raise ErroRetomadaSaida(
                f"nao foi possivel parar durante {descricao}")

    def _parar_melhor_esforco(self):
        try:
            parou = self.acao_direcao()
            if parou is False:
                self.arduino.parar()
        except Exception:
            try:
                self.arduino.parar()
            except Exception:
                pass
