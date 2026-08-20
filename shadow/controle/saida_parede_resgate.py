"""Manobra curta apos o deposito vermelho.

Esta rotina deliberadamente nao procura a saida. Ela so afasta o robo do
marcador vermelho, gira 90 graus para a direita pelo MPU e usa o ultrassom
lateral direito para regular a distancia da parede. Ao terminar o
alinhamento, avanca reto ate encontrar a parede frontal a 118 mm. Entao
avanca em curva, mantendo a frente como referencia e movimentando mais a
traseira. Com o ultrassom lateral direito estavel por um segundo, translada a direita
por meio segundo. A camera frontal procura um triangulo verde; se confirmar,
gira 45 graus para a esquerda e repete a aproximacao e o alinhamento.
"""

from dataclasses import dataclass
import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


@dataclass(frozen=True)
class ResultadoSondaLinha:
    """Contrato mantido para o modulo de sonda, hoje fora desta rota."""

    resultado: str
    avanco_s: float


class ControladorSaidaParede:
    """Gira, alinha, procura o verde e repete uma unica rota de parede."""

    ZERAR_MPU = "EXIT_PAREDE_ZERAR_MPU"
    AFASTAR_VERMELHO = "EXIT_PAREDE_AFASTAR_VERMELHO"
    ASSENTAR_INICIAL = "EXIT_PAREDE_ASSENTAR_INICIAL"
    GIRO_INICIAL_DIREITA = "EXIT_PAREDE_GIRO_INICIAL_DIREITA"
    ALINHAR_DIREITA = "EXIT_PAREDE_ALINHAR_DIREITA"
    CORRIGIR_YAW_ALINHAMENTO = "EXIT_PAREDE_CORRIGIR_YAW_ALINHAMENTO"
    AVANCAR_ATE_PAREDE_FRENTE = "EXIT_PAREDE_AVANCAR_ATE_FRENTE"
    CORRIGIR_YAW_AVANCO_FRENTE = "EXIT_PAREDE_CORRIGIR_YAW_FRENTE"
    PIVO_TRASEIRO_ESTABILIZAR = "EXIT_PAREDE_PIVO_TRASEIRO_ESTABILIZAR"
    TRANSLADAR_DIREITA_FINAL = "EXIT_PAREDE_TRANSLADAR_DIREITA_FINAL"
    VERIFICAR_TRIANGULO_VERDE = "EXIT_PAREDE_VERIFICAR_TRIANGULO_VERDE"
    GIRO_TRIANGULO_VERDE = "EXIT_PAREDE_GIRO_TRIANGULO_VERDE"
    SAIDA_CONCLUIDA = "EXIT_PAREDE_SAIDA_CONCLUIDA"
    FALHA = "EXIT_PAREDE_FALHA"

    _ESTADOS_GIRO = {
        GIRO_INICIAL_DIREITA,
        CORRIGIR_YAW_ALINHAMENTO,
        CORRIGIR_YAW_AVANCO_FRENTE,
        GIRO_TRIANGULO_VERDE,
    }

    def __init__(self, start_time=None):
        agora = time.monotonic() if start_time is None else float(start_time)
        if cfg.SAIDA_PAREDE_LADO != "DIREITA":
            raise ValueError("a manobra exige o ultrassom lateral direito")
        self.state = self.ZERAR_MPU
        self._inicio_estado = agora
        self._comando_aceito = False
        self._detalhe_falha = ""

        self._yaw = None
        self._yaw_em = None
        self._sinal_yaw_por_giro_direita = None
        self._yaw_inicio_giro = None
        self._alvo_yaw = None
        self._heading_parede = None

        self._lateral_mm = None
        self._lateral_em = None
        self._lateral_em_antes_alinhamento = None
        self._frente_mm = None
        self._frente_em = None
        self._frente_em_antes_avanco = None
        self._lateral_em_antes_pivo = None
        self._lateral_em_processada_pivo = None
        self._lateral_referencia_pivo_mm = None
        self._estavel_desde_pivo = None
        self._translacao_por_timeout_pivo = False
        self._triangulo_verde_confirmado = False
        self._triangulo_verde_tratado = False
        self._tentativas_correcao = 0

    @property
    def terminal(self):
        return self.state in (self.SAIDA_CONCLUIDA, self.FALHA)

    @property
    def solicita_zerar_mpu(self):
        return self.state == self.ZERAR_MPU

    @property
    def prioriza_mpu(self):
        return self.state in self._ESTADOS_GIRO

    @property
    def usa_camera_triangulo_verde(self):
        return self.state == self.VERIFICAR_TRIANGULO_VERDE

    @property
    def lado_ultrassom_atual(self):
        if self.state in {
            self.AVANCAR_ATE_PAREDE_FRENTE,
            self.CORRIGIR_YAW_AVANCO_FRENTE,
        }:
            return "FRENTE"
        return "LATERAL"

    @property
    def heading_parede(self):
        return self._heading_parede

    def diagnostico_yaw(self, now=None):
        agora = time.monotonic() if now is None else float(now)
        yaw = "-" if self._yaw is None else f"{self._yaw:.1f}"
        alvo = "-" if self._alvo_yaw is None else f"{self._alvo_yaw:.1f}"
        idade = "-" if self._yaw_em is None else f"{agora - self._yaw_em:.2f}s"
        return f"yaw={yaw} alvo={alvo} idade={idade}"

    def observar_mpu(self, yaw_graus, timestamp=None):
        if yaw_graus is None:
            return
        self._yaw = self._normalizar(float(yaw_graus))
        self._yaw_em = time.monotonic() if timestamp is None else float(timestamp)

    def observar_ultrassom(self, lado, distancia_mm, respondeu, timestamp=None):
        """Registra a leitura do HC-SR04 pedida pelo estado atual."""
        if not respondeu:
            return
        agora = time.monotonic() if timestamp is None else float(timestamp)
        lado = str(lado).upper()
        if lado == "LATERAL":
            self._lateral_mm = None if distancia_mm is None else int(distancia_mm)
            self._lateral_em = agora
            return
        if lado == "FRENTE":
            self._frente_mm = None if distancia_mm is None else int(distancia_mm)
            self._frente_em = agora
            return
        raise ValueError("lado deve ser FRENTE ou LATERAL")

    def observar_triangulo_verde(self, confirmado, timestamp=None):
        """Recebe somente a confirmacao temporal do detector de marcadores."""
        if self.state != self.VERIFICAR_TRIANGULO_VERDE or not confirmado:
            return False
        self._triangulo_verde_confirmado = True
        return True

    def confirmar_mpu_zerado(self, sucesso, now=None):
        if self.state != self.ZERAR_MPU:
            return False
        if not sucesso:
            self._falhar("MPU ZERO nao foi confirmado", now)
            return False
        self._entrar(self.AFASTAR_VERMELHO, now)
        return True

    def notificar_comando_escrito(self, state, now=None):
        if state != self.state or self._comando_aceito:
            return False
        self._comando_aceito = True
        self._inicio_estado = time.monotonic() if now is None else float(now)
        return True

    def atualizar(self, now=None):
        agora = time.monotonic() if now is None else float(now)
        if self.state == self.SAIDA_CONCLUIDA:
            return self._parado(
                self.SAIDA_CONCLUIDA,
                "triangulo verde tratado, leitura lateral estabilizada e translacao final concluida; robo parado",
                terminal=True,
            )
        if self.state == self.FALHA:
            return self._parado(self.FALHA, self._detalhe_falha, terminal=True)

        if self.state in self._ESTADOS_GIRO:
            if not self._mpu_fresco(agora):
                return self._falhar("MPU sem leitura recente durante giro", agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_GIRO_S:
                return self._falhar("timeout no giro monitorado por yaw", agora)

        if self.state == self.ZERAR_MPU:
            return self._parado(
                self.ZERAR_MPU,
                "parado; zerando yaw relativo do MPU",
            )

        if self.state == self.AFASTAR_VERMELHO:
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_S:
                self._entrar(self.ASSENTAR_INICIAL, agora)
                return self.atualizar(agora)
            return self._frente(
                self.AFASTAR_VERMELHO,
                cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_PWM,
                "avanco curto para liberar o giro apos o deposito vermelho",
            )

        if self.state == self.ASSENTAR_INICIAL:
            if self._tempo_decorrido(agora) < cfg.SAIDA_PAREDE_ASSENTAMENTO_S:
                return self._parado(
                    self.ASSENTAR_INICIAL,
                    "parado antes do giro inicial",
                )
            if not self._mpu_fresco(agora):
                return self._falhar("MPU sem leitura antes do giro inicial", agora)
            self._yaw_inicio_giro = self._yaw
            self._alvo_yaw = None
            self._entrar(self.GIRO_INICIAL_DIREITA, agora)
            return self.atualizar(agora)

        if self.state == self.GIRO_INICIAL_DIREITA:
            if self._sinal_yaw_por_giro_direita is None:
                variacao = self._erro_yaw_assinado(
                    self._yaw,
                    self._yaw_inicio_giro,
                )
                if abs(variacao) < cfg.SAIDA_PAREDE_VARIACAO_MINIMA_YAW_GRAUS:
                    return self._girar_direita_para_calibrar()
                self._sinal_yaw_por_giro_direita = 1.0 if variacao > 0 else -1.0
                self._alvo_yaw = self._normalizar(
                    self._yaw_inicio_giro
                    + 90.0 * self._sinal_yaw_por_giro_direita)
            if self._giro_concluido():
                self._heading_parede = self._alvo_yaw
                self._entrar_alinhamento(agora)
                return self.atualizar(agora)
            return self._girar(
                self.GIRO_INICIAL_DIREITA,
                "girando 90 graus a direita pelo MPU",
            )

        if self.state == self.ALINHAR_DIREITA:
            # Nunca usa uma leitura feita antes de o chassi completar o giro:
            # depois de 90 graus, o ultrassom passou a olhar para outra parede.
            if not self._lateral_fresca(agora):
                if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                    return self._falhar(
                        "ultrassom lateral sem leitura nova apos o giro", agora)
                return self._parado(
                    self.ALINHAR_DIREITA,
                    "giro concluido; aguardando ultrassom lateral novo",
                )
            if self._lateral_mm is None:
                return self._falhar("parede direita nao encontrada", agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_ALINHAMENTO_S:
                return self._falhar("timeout ao alinhar a distancia da parede", agora)

            erro_distancia = (
                self._lateral_mm - cfg.SAIDA_PAREDE_DISTANCIA_ALVO_ALINHAMENTO_MM)
            if abs(erro_distancia) <= cfg.SAIDA_PAREDE_TOLERANCIA_ALINHAMENTO_MM:
                self._entrar_avanco_frente(agora)
                return self.atualizar(agora)
            if not self._mpu_fresco(agora):
                return self._falhar("MPU sem leitura durante alinhamento lateral", agora)
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_TRANSLACAO_YAW_GRAUS:
                self._tentativas_correcao += 1
                if self._tentativas_correcao > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO:
                    return self._falhar("yaw nao voltou ao rumo durante alinhamento", agora)
                self._preparar_giro_para(
                    self._heading_parede,
                    self.CORRIGIR_YAW_ALINHAMENTO,
                    agora,
                )
                return self.atualizar(agora)
            self._tentativas_correcao = 0
            if erro_distancia > 0:
                return self._lateral(
                    self.ALINHAR_DIREITA,
                    direita=True,
                    detalhe="parede distante; transladando para a direita",
                )
            return self._lateral(
                self.ALINHAR_DIREITA,
                direita=False,
                detalhe="parede proxima; transladando para a esquerda",
            )

        if self.state == self.CORRIGIR_YAW_ALINHAMENTO:
            if self._giro_concluido():
                self._entrar_alinhamento(agora)
                return self.atualizar(agora)
            return self._girar(
                self.CORRIGIR_YAW_ALINHAMENTO,
                "corrigindo yaw antes de continuar o alinhamento lateral",
            )

        if self.state == self.AVANCAR_ATE_PAREDE_FRENTE:
            if not self._frente_fresca(agora):
                if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                    return self._falhar(
                        "ultrassom frontal sem leitura nova apos alinhamento",
                        agora,
                    )
                return self._parado(
                    self.AVANCAR_ATE_PAREDE_FRENTE,
                    "alinhamento concluido; aguardando ultrassom frontal novo",
                )
            if self._frente_mm is None:
                # ``None`` aqui e uma resposta valida ``OK ULTRASSOM -1``:
                # o HC-SR04 frontal foi consultado, mas nao recebeu eco.
                # Nao e seguro interpreta-la como pista livre e continuar
                # reto; o robo fica parado ate o sensor ser corrigido.
                return self._falhar(
                    "ultrassom frontal respondeu sem eco; avanco bloqueado",
                    agora,
                )
            if (
                self._frente_mm <= cfg.SAIDA_PAREDE_DISTANCIA_FRENTE_FINAL_MM
            ):
                self._entrar_pivo_traseiro(agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_AVANCO_FRENTE_S:
                return self._falhar(
                    "timeout sem parede frontal a "
                    f"{cfg.SAIDA_PAREDE_DISTANCIA_FRENTE_FINAL_MM} mm "
                    f"(ultima leitura={self._frente_mm} mm)",
                    agora,
                )
            if not self._mpu_fresco(agora):
                return self._falhar("MPU sem leitura durante avanco reto", agora)
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_YAW_GRAUS:
                self._tentativas_correcao += 1
                if self._tentativas_correcao > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO:
                    return self._falhar(
                        "yaw nao voltou ao rumo durante avanco reto", agora)
                self._preparar_giro_para(
                    self._heading_parede,
                    self.CORRIGIR_YAW_AVANCO_FRENTE,
                    agora,
                )
                return self.atualizar(agora)
            self._tentativas_correcao = 0
            return self._frente(
                self.AVANCAR_ATE_PAREDE_FRENTE,
                cfg.SAIDA_PAREDE_AVANCO_ATE_FRENTE_PWM,
                "avancando reto ate o ultrassom frontal marcar 118 mm",
            )

        if self.state == self.CORRIGIR_YAW_AVANCO_FRENTE:
            if self._giro_concluido():
                self._entrar_avanco_frente(agora)
                return self.atualizar(agora)
            return self._girar(
                self.CORRIGIR_YAW_AVANCO_FRENTE,
                "corrigindo yaw antes de retomar o avanco reto",
            )

        if self.state == self.VERIFICAR_TRIANGULO_VERDE:
            if self._triangulo_verde_confirmado:
                if not self._mpu_fresco(agora):
                    if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                        return self._falhar(
                            "MPU sem leitura nova apos confirmar triangulo verde",
                            agora,
                        )
                    return self._parado(
                        self.VERIFICAR_TRIANGULO_VERDE,
                        "triangulo verde confirmado; aguardando yaw para giro de 45 graus",
                    )
                if self._sinal_yaw_por_giro_direita is None:
                    return self._falhar(
                        "sentido do yaw ausente para girar no triangulo verde",
                        agora,
                    )
                # A rota ja esta com a parede no lado direito: o desvio pelo
                # triangulo verde e fisicamente para a esquerda, independente
                # do sinal eletrico que o MPU use para a direita.
                self._alvo_yaw = self._normalizar(
                    self._yaw
                    - cfg.SAIDA_PAREDE_GIRO_TRIANGULO_VERDE_GRAUS
                    * self._sinal_yaw_por_giro_direita
                )
                self._triangulo_verde_tratado = True
                self._entrar(self.GIRO_TRIANGULO_VERDE, agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_TRIANGULO_VERDE_S:
                return self._falhar(
                    "triangulo verde nao foi confirmado pela camera frontal",
                    agora,
                )
            return self._parado(
                self.VERIFICAR_TRIANGULO_VERDE,
                "parado; camera frontal procurando triangulo verde com LED apagado",
            )

        if self.state == self.GIRO_TRIANGULO_VERDE:
            if self._giro_concluido():
                self._heading_parede = self._alvo_yaw
                self._entrar_avanco_frente(agora)
                return self.atualizar(agora)
            return self._girar(
                self.GIRO_TRIANGULO_VERDE,
                "triangulo verde confirmado; girando 45 graus para a esquerda",
            )

        if self.state == self.PIVO_TRASEIRO_ESTABILIZAR:
            if not self._lateral_fresca_desde_pivo(agora):
                if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                    return self._falhar(
                        "ultrassom lateral sem leitura nova durante pivo traseiro",
                        agora,
                    )
                return self._parado(
                    self.PIVO_TRASEIRO_ESTABILIZAR,
                    "parede frontal atingida; aguardando lateral novo para o pivo",
                )
            if self._lateral_mm is None:
                return self._falhar(
                    "ultrassom lateral respondeu sem eco durante pivo traseiro",
                    agora,
                )
            self._atualizar_estabilidade_lateral(agora)
            if self._estavel_desde_pivo is not None and (
                agora - self._estavel_desde_pivo
                >= cfg.SAIDA_PAREDE_TEMPO_ESTABILIDADE_LATERAL_S
            ):
                self._translacao_por_timeout_pivo = False
                self._entrar(self.TRANSLADAR_DIREITA_FINAL, agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_PIVO_TRASEIRO_S:
                self._translacao_por_timeout_pivo = True
                self._entrar(self.TRANSLADAR_DIREITA_FINAL, agora)
                return self.atualizar(agora)
            toque_frente_direita_pwm = self._toque_frente_direita_pwm(agora)
            detalhe = (
                "avancando em curva para a esquerda com frente priorizada; "
                f"lateral={self._lateral_mm} mm"
            )
            if toque_frente_direita_pwm:
                detalhe += "; toque na dianteira direita"
            return MotionCommand(
                self.PIVO_TRASEIRO_ESTABILIZAR,
                angle=cfg.SAIDA_PAREDE_ANGULO_CURVA_TRASEIRA,
                speed=cfg.SAIDA_PAREDE_PWM_PIVO_TRASEIRO / 120.0,
                detail=detalhe,
                pivo_traseiro=True,
                toque_frente_direita_pwm=toque_frente_direita_pwm,
            )

        if self.state == self.TRANSLADAR_DIREITA_FINAL:
            if self._tempo_decorrido(agora) >= (
                cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S
            ):
                if self._triangulo_verde_tratado:
                    self._entrar(self.SAIDA_CONCLUIDA, agora)
                else:
                    self._entrar_verificacao_triangulo_verde(agora)
                return self.atualizar(agora)
            detalhe = (
                "lateral oscilou por tempo demais; transladando para a direita "
                "mesmo sem estabilizar"
                if self._translacao_por_timeout_pivo else
                "leitura lateral estavel; transladando para a direita por 0,5 s"
            )
            return self._lateral(
                self.TRANSLADAR_DIREITA_FINAL,
                direita=True,
                pwm=cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                detalhe=detalhe,
            )

        return self._falhar(f"estado de alinhamento desconhecido: {self.state}", agora)

    def _entrar_alinhamento(self, agora):
        self._lateral_em_antes_alinhamento = self._lateral_em
        self._entrar(self.ALINHAR_DIREITA, agora)

    def _entrar_avanco_frente(self, agora):
        self._frente_em_antes_avanco = self._frente_em
        self._entrar(self.AVANCAR_ATE_PAREDE_FRENTE, agora)

    def _entrar_pivo_traseiro(self, agora):
        # A leitura que encontrou a parede so inicia a fase. A estabilidade
        # precisa ser provada por leituras novas, feitas ja com o pivo ativo.
        self._lateral_em_antes_pivo = self._lateral_em
        self._lateral_em_processada_pivo = None
        self._lateral_referencia_pivo_mm = None
        self._estavel_desde_pivo = None
        self._translacao_por_timeout_pivo = False
        self._entrar(self.PIVO_TRASEIRO_ESTABILIZAR, agora)

    def _entrar_verificacao_triangulo_verde(self, agora):
        self._triangulo_verde_confirmado = False
        self._entrar(self.VERIFICAR_TRIANGULO_VERDE, agora)

    def _atualizar_estabilidade_lateral(self, agora):
        if self._lateral_em == self._lateral_em_processada_pivo:
            return
        medida_atual = self._lateral_mm
        if self._lateral_referencia_pivo_mm is None:
            self._lateral_referencia_pivo_mm = medida_atual
            self._estavel_desde_pivo = agora
        elif abs(medida_atual - self._lateral_referencia_pivo_mm) > (
            cfg.SAIDA_PAREDE_TOLERANCIA_ESTABILIDADE_LATERAL_MM
        ):
            self._lateral_referencia_pivo_mm = medida_atual
            self._estavel_desde_pivo = agora
        self._lateral_em_processada_pivo = self._lateral_em

    def _toque_frente_direita_pwm(self, agora):
        if not self._comando_aceito:
            return 0
        periodo = max(
            float(cfg.SAIDA_PAREDE_PERIODO_TOQUE_FRENTE_DIREITA_S),
            0.01,
        )
        duracao = min(
            max(float(cfg.SAIDA_PAREDE_DURACAO_TOQUE_FRENTE_DIREITA_S), 0.0),
            periodo,
        )
        if duracao <= 0.0 or self._tempo_decorrido(agora) % periodo >= duracao:
            return 0
        return int(cfg.SAIDA_PAREDE_PWM_TOQUE_FRENTE_DIREITA)

    def _preparar_giro_para(self, alvo, estado, agora):
        if alvo is None or self._sinal_yaw_por_giro_direita is None:
            self._falhar("heading ou sentido de giro ausente", agora)
            return
        self._alvo_yaw = self._normalizar(alvo)
        self._entrar(estado, agora)

    def _giro_concluido(self):
        return (
            self._alvo_yaw is not None
            and self._yaw is not None
            and self._erro_yaw(self._alvo_yaw, self._yaw)
            <= cfg.SAIDA_PAREDE_TOLERANCIA_YAW_GRAUS
        )

    def _erro_heading(self):
        if self._heading_parede is None or self._yaw is None:
            return float("inf")
        return self._erro_yaw(self._heading_parede, self._yaw)

    def _girar(self, estado, detalhe):
        if self._alvo_yaw is None or self._yaw is None:
            return self._falhar("yaw ou alvo ausente durante giro")
        erro_assinado = self._erro_yaw_assinado(self._alvo_yaw, self._yaw)
        angulo = 180 if erro_assinado * self._sinal_yaw_por_giro_direita >= 0 else -180
        return MotionCommand(
            estado,
            angle=angulo,
            speed=cfg.SAIDA_PAREDE_PWM_GIRO / 120.0,
            detail=detalhe,
        )

    def _girar_direita_para_calibrar(self):
        return MotionCommand(
            self.GIRO_INICIAL_DIREITA,
            angle=180,
            speed=cfg.SAIDA_PAREDE_PWM_GIRO / 120.0,
            detail="tanque inicial a direita; calibrando o sinal do yaw",
        )

    @staticmethod
    def _frente(estado, pwm, detalhe):
        return MotionCommand(
            estado,
            angle=0,
            speed=float(pwm) / 120.0,
            detail=detalhe,
        )

    @staticmethod
    def _parado(estado, detalhe, terminal=False):
        return MotionCommand(estado, detail=detalhe, terminal=terminal)

    @staticmethod
    def _lateral(estado, direita, detalhe, pwm=None):
        if pwm is None:
            pwm = cfg.SAIDA_PAREDE_PWM_TRANSLACAO_ALINHAMENTO
        pwm = int(pwm)
        rodas = (pwm, -pwm, -pwm, pwm) if direita else (-pwm, pwm, pwm, -pwm)
        return MotionCommand(estado, detail=detalhe, wheel_speeds=rodas)

    def _mpu_fresco(self, agora):
        return (
            self._yaw is not None
            and self._yaw_em is not None
            and agora - self._yaw_em <= cfg.SAIDA_PAREDE_TIMEOUT_MPU_S
        )

    def _lateral_fresca(self, agora):
        return (
            self._lateral_em is not None
            and self._lateral_em_antes_alinhamento is not None
            and self._lateral_em > self._lateral_em_antes_alinhamento
            and agora - self._lateral_em <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S
        )

    def _frente_fresca(self, agora):
        return (
            self._frente_em is not None
            and (
                self._frente_em_antes_avanco is None
                or self._frente_em > self._frente_em_antes_avanco
            )
            and agora - self._frente_em <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S
        )

    def _lateral_fresca_desde_pivo(self, agora):
        return (
            self._lateral_em is not None
            and self._lateral_em_antes_pivo is not None
            and self._lateral_em > self._lateral_em_antes_pivo
            and agora - self._lateral_em <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S
        )

    def _tempo_decorrido(self, agora):
        if not self._comando_aceito:
            return 0.0
        return max(float(agora) - self._inicio_estado, 0.0)

    def _entrar(self, estado, agora=None):
        self.state = estado
        self._inicio_estado = time.monotonic() if agora is None else float(agora)
        self._comando_aceito = False

    def _falhar(self, detalhe, agora=None):
        self._detalhe_falha = str(detalhe)
        self._entrar(self.FALHA, agora)
        return self._parado(self.FALHA, self._detalhe_falha, terminal=True)

    @staticmethod
    def _erro_yaw(alvo, atual):
        return abs(((float(alvo) - float(atual) + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _erro_yaw_assinado(alvo, atual):
        return ((float(alvo) - float(atual) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _normalizar(angulo):
        return float(angulo) % 360.0


def executar_alinhamento_parede(
    arduino,
    *,
    intervalo_s=0.005,
    camera_index=None,
    debug=False,
    camera_factory=None,
    detector_verde_factory=None,
    fonte_assincrona_factory=None,
):
    """Executa a manobra pos-vermelho com uma verificacao frontal do verde.

    O chamador deve ter adquirido a trava dos motores e inicializado
    ``controle.direcao`` com este mesmo Arduino. A funcao nao fecha a serial:
    isso continua responsabilidade do programa que a chamou.
    """
    from controle.direcao import steer
    from controle.monitor_saida_parede import MonitorSensoresSaida

    if camera_factory is None:
        from visao.captura_resgate import RescueCamera
        camera_factory = RescueCamera
    if detector_verde_factory is None:
        from visao.marcador_resgate import MarkerDetector
        detector_verde_factory = MarkerDetector
    if fonte_assincrona_factory is None:
        from visao.resgate_assincrono import LatestFrameSource
        fonte_assincrona_factory = LatestFrameSource

    controlador = ControladorSaidaParede()
    monitor_sensores = MonitorSensoresSaida(arduino)
    epoca_serial = arduino.connection_epoch
    ultimo_estado = None
    ultimo_log_yaw = -float("inf")
    fonte_verde = None
    detector_verde = None
    ultima_sequencia_verde = 0

    def fechar_camera_verde():
        nonlocal fonte_verde, detector_verde, ultima_sequencia_verde
        if fonte_verde is not None:
            fonte_verde.close()
        fonte_verde = None
        detector_verde = None
        ultima_sequencia_verde = 0

    try:
        if arduino.led("APAGADO") is False:
            raise RuntimeError("nao foi possivel apagar LED da saida")
        print(
            "[saida] LED APAGADO; camera frontal sera aberta somente "
            "para procurar o triangulo verde")
        while True:
            agora = time.monotonic()
            if (
                not arduino.connected
                or arduino.connection_epoch != epoca_serial
            ):
                raise RuntimeError("serial mudou durante o alinhamento de saida")

            monitor_sensores.atualizar_controlador(controlador, agora)
            if controlador.usa_camera_triangulo_verde:
                if fonte_verde is None:
                    if arduino.led("APAGADO") is False:
                        raise RuntimeError(
                            "nao foi possivel manter o LED apagado na camera frontal")
                    fonte_verde = fonte_assincrona_factory(
                        camera_factory(camera_index))
                    detector_verde = detector_verde_factory("green")
                    print(
                        "[saida] camera frontal ativa com LED APAGADO; "
                        "procurando triangulo verde")
                quadro = fonte_verde.poll(ultima_sequencia_verde)
                if quadro is not None:
                    ultima_sequencia_verde = quadro.sequence
                    deteccao_verde = detector_verde.detect(
                        quadro.frame,
                        timestamp=quadro.captured_at,
                    )
                    if deteccao_verde is not None and (
                        deteccao_verde.confirmed
                    ) and controlador.observar_triangulo_verde(
                        True,
                        quadro.captured_at,
                    ):
                        print(
                            "[saida] triangulo verde confirmado pela camera "
                            f"em {deteccao_verde.hits} frame(s)")
                    elif debug and deteccao_verde is not None:
                        print(
                            "[saida] verde em aquisicao: "
                            f"hits={deteccao_verde.hits} "
                            f"confirmado={deteccao_verde.confirmed}")
            comando = controlador.atualizar(agora)

            # A camera frontal e fechada antes do giro/avanco seguinte. Isso
            # deixa so um pipeline de imagem ativo e impede processamento
            # concorrente com a camera de segue-linha em qualquer handoff.
            if not controlador.usa_camera_triangulo_verde and fonte_verde is not None:
                fechar_camera_verde()

            if comando.state != ultimo_estado:
                print(
                    f"[saida] {comando.state}: {comando.detail} "
                    f"({controlador.diagnostico_yaw(agora)})")
                ultimo_estado = comando.state
            elif (
                controlador.prioriza_mpu
                and agora - ultimo_log_yaw >= 0.40
            ):
                print(
                    f"[saida] giro monitorado: "
                    f"{controlador.diagnostico_yaw(agora)}")
                ultimo_log_yaw = agora

            if controlador.solicita_zerar_mpu:
                steer()
                monitor_sensores.cancelar()
                if not controlador.confirmar_mpu_zerado(arduino.zerar_mpu(), agora):
                    return None
                continue

            if comando.wheel_speeds is not None:
                enviado = arduino.rodas(*comando.wheel_speeds)
            else:
                enviado = steer(
                    comando.angle,
                    comando.speed,
                    rear_pivot_enabled=comando.pivo_traseiro,
                    toque_frente_direita_pwm=comando.toque_frente_direita_pwm,
                )
            if enviado is False:
                raise RuntimeError("comando de alinhamento nao foi enviado")
            controlador.notificar_comando_escrito(comando.state, time.monotonic())

            if comando.terminal:
                if comando.state == ControladorSaidaParede.SAIDA_CONCLUIDA:
                    return "saida_concluida"
                print(
                    f"[saida] falha: {comando.detail} "
                    f"({controlador.diagnostico_yaw(agora)})")
                return None

            monitor_sensores.agendar_proxima(
                time.monotonic(),
                priorizar_mpu=controlador.prioriza_mpu,
                lado_ultrassom=controlador.lado_ultrassom_atual,
            )
            arduino.refresh(fail_closed=True)
            time.sleep(intervalo_s)
    finally:
        try:
            fechar_camera_verde()
        except (OSError, RuntimeError):
            pass
        try:
            steer()
        except (OSError, RuntimeError):
            pass
