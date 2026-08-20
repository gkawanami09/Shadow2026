"""Saida de teste do resgate seguindo a parede direita, sem mapa previo.

A rota atual usa exclusivamente MPU e os dois ultrassons: apos o deposito
vermelho ela alinha o chassi, percorre cantos com parede frontal e lateral, e
para ao confirmar uma abertura direita. Nao desvia de triangulos nem entra na
abertura nesta versao. Toda falha de sensor, yaw ou serial deve parar o robo.
"""

from dataclasses import dataclass
import time

import config_resgate as cfg
from controle.aproximacao_resgate import MotionCommand


@dataclass(frozen=True)
class ResultadoSondaLinha:
    """Resultado da camera inferior e o tempo exato avancado na abertura."""

    resultado: str
    avanco_s: float


class ControladorSaidaParede:
    """Navega a parede direita e prepara uma tentativa reversivel de saida."""

    ZERAR_MPU = "EXIT_PAREDE_ZERAR_MPU"
    AFASTAR_VERMELHO = "EXIT_PAREDE_AFASTAR_VERMELHO"
    ASSENTAR_INICIAL = "EXIT_PAREDE_ASSENTAR_INICIAL"
    GIRO_INICIAL_DIREITA = "EXIT_PAREDE_GIRO_INICIAL_DIREITA"
    TRANSLADAR_DIREITA_INICIAL = "EXIT_PAREDE_ALINHAR_INICIAL_DIREITA"
    CORRIGIR_YAW_TRANSLACAO_INICIAL = "EXIT_PAREDE_CORRIGIR_YAW_INICIAL"
    SEGUIR_PAREDE = "EXIT_PAREDE_SEGUIR_DIREITA"
    CORRIGIR_YAW_SEGUINDO_PAREDE = "EXIT_PAREDE_CORRIGIR_YAW_SEGUINDO"
    PARAR_TRIANGULO = "EXIT_PAREDE_PARAR_TRIANGULO"
    DESVIAR_TRIANGULO = "EXIT_PAREDE_DESVIAR_TRIANGULO"
    PASSAR_TRIANGULO = "EXIT_PAREDE_PASSAR_TRIANGULO"
    RETORNAR_TRIANGULO = "EXIT_PAREDE_RETORNAR_TRIANGULO"
    PARAR_PAREDE = "EXIT_PAREDE_PARAR_FRENTE"
    GIRO_PAREDE_ESQUERDA = "EXIT_PAREDE_GIRO_PAREDE_ESQUERDA"
    CONFERIR_PAREDE_APOS_GIRO = "EXIT_PAREDE_CONFERIR_LATERAL"
    ALINHAR_DIREITA_APOS_GIRO = "EXIT_PAREDE_ALINHAR_DIREITA"
    CORRIGIR_YAW_ALINHAMENTO = "EXIT_PAREDE_CORRIGIR_YAW_ALINHAMENTO"
    PARAR_ABERTURA = "EXIT_PAREDE_PARAR_ABERTURA"
    AVANCAR_ENTRADA = "EXIT_PAREDE_AVANCAR_ENTRADA"
    TRANSLADAR_ESQUERDA = "EXIT_PAREDE_TRANSLADAR_ESQUERDA"
    CORRIGIR_YAW_TRANSLACAO = "EXIT_PAREDE_CORRIGIR_YAW_TRANSLACAO"
    GIRO_ENTRADA_DIREITA = "EXIT_PAREDE_GIRO_ENTRADA_DIREITA"
    PRONTO_SONDA_LINHA = "EXIT_PAREDE_PRONTO_SONDA_LINHA"
    AGUARDANDO_SONDA = "EXIT_PAREDE_AGUARDANDO_SONDA"
    RECUAR_SONDA = "EXIT_PAREDE_RECUAR_SONDA"
    GIRO_RETORNO_ESQUERDA = "EXIT_PAREDE_GIRO_RETORNO_ESQUERDA"
    TRANSLADAR_DIREITA = "EXIT_PAREDE_TRANSLADAR_DIREITA"
    ABRIR_CAMERA_FRONTAL = "EXIT_PAREDE_ABRIR_CAMERA_FRONTAL"
    IGNORAR_ABERTURA = "EXIT_PAREDE_IGNORAR_ABERTURA"
    ABERTURA_ENCONTRADA = "EXIT_PAREDE_ABERTURA_ENCONTRADA"
    SUCESSO = "EXIT_PAREDE_PRETO_CONFIRMADO"
    FALHA = "EXIT_PAREDE_FALHA"

    _ESTADOS_TEMPORIZADOS = {
        AFASTAR_VERMELHO,
        ASSENTAR_INICIAL,
        TRANSLADAR_DIREITA_INICIAL,
        PARAR_TRIANGULO,
        PASSAR_TRIANGULO,
        PARAR_PAREDE,
        ALINHAR_DIREITA_APOS_GIRO,
        PARAR_ABERTURA,
        AVANCAR_ENTRADA,
        TRANSLADAR_ESQUERDA,
        RECUAR_SONDA,
        TRANSLADAR_DIREITA,
        IGNORAR_ABERTURA,
    }
    _ESTADOS_GIRO_MONITORADO = {
        GIRO_INICIAL_DIREITA,
        CORRIGIR_YAW_TRANSLACAO_INICIAL,
        CORRIGIR_YAW_SEGUINDO_PAREDE,
        DESVIAR_TRIANGULO,
        RETORNAR_TRIANGULO,
        GIRO_PAREDE_ESQUERDA,
        CORRIGIR_YAW_ALINHAMENTO,
        CORRIGIR_YAW_TRANSLACAO,
        GIRO_ENTRADA_DIREITA,
        GIRO_RETORNO_ESQUERDA,
    }

    def __init__(self, start_time=None):
        agora = time.monotonic() if start_time is None else float(start_time)
        if cfg.SAIDA_PAREDE_LADO != "DIREITA":
            raise ValueError("a rota atual de saida exige ultrassom DIREITA")
        self.state = self.ZERAR_MPU
        self._criado_em = agora
        self._inicio_estado = agora
        self._comando_aceito = False
        self._detalhe_falha = ""
        self._yaw = None
        self._yaw_em = None
        self._sinal_yaw_por_giro_direita = None
        self._yaw_inicio_giro_inicial = None
        self._heading_parede = None
        self._alvo_yaw = None
        self._frente_mm = None
        self._frente_em = None
        self._lateral_mm = None
        self._lateral_em = None
        self._frente_proxima = 0
        self._lateral_aberta = 0
        self._lateral_parede = 0
        self._abertura_iniciada_em = None
        self._lateral_em_antes_do_giro = None
        # Um vao so existe depois de o sensor ter visto uma parede continua.
        # Isso impede que a zona aberta logo depois do deposito vermelho seja
        # tratada como uma saida antes de o robo alcancar a parede direita.
        self._parede_lateral_confirmada = False
        self._triangulo_visivel = False
        self._ultimo_triangulo_em = -float("inf")
        self._tentativas_translacao = 0
        self._tempo_translacao_inicial_restante = 0.0
        self._tentativas_correcao_seguimento = 0
        self._giros_parede = 0
        self._tempo_recuo_sonda = cfg.SAIDA_PAREDE_RECUO_MINIMO_S
        self._sonda_iniciada = False

    @property
    def terminal(self):
        return self.state in (self.ABERTURA_ENCONTRADA, self.SUCESSO, self.FALHA)

    @property
    def solicita_zerar_mpu(self):
        return self.state == self.ZERAR_MPU

    @property
    def solicita_sonda_linha(self):
        return self.state == self.PRONTO_SONDA_LINHA and not self._sonda_iniciada

    @property
    def solicita_camera_frontal(self):
        return self.state == self.ABRIR_CAMERA_FRONTAL

    @property
    def prioriza_mpu(self):
        """Durante um giro, yaw e mais urgente que medir as paredes."""
        return self.state in self._ESTADOS_GIRO_MONITORADO

    @property
    def heading_parede(self):
        return self._heading_parede

    def diagnostico_yaw(self, now=None):
        """Texto curto para tornar falhas de giro verificaveis no log."""
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
        """Registra uma leitura nova; ``None`` so e abertura se respondeu."""
        agora = time.monotonic() if timestamp is None else float(timestamp)
        lado = str(lado).upper()
        if lado == "FRENTE":
            if not respondeu:
                return
            self._frente_mm = None if distancia_mm is None else int(distancia_mm)
            self._frente_em = agora
            if (
                self._frente_mm is not None
                and self._frente_mm <= cfg.SAIDA_PAREDE_DISTANCIA_FRENTE_PARAR_MM
            ):
                self._frente_proxima += 1
            else:
                self._frente_proxima = 0
            return
        if lado != "LATERAL":
            raise ValueError("lado deve ser FRENTE ou LATERAL")
        if not respondeu:
            return
        self._lateral_mm = None if distancia_mm is None else int(distancia_mm)
        self._lateral_em = agora
        aberta = (
            self._lateral_mm is None
            or self._lateral_mm >= cfg.SAIDA_PAREDE_DISTANCIA_ABERTURA_MM
        )
        if aberta:
            if self._lateral_aberta == 0:
                self._abertura_iniciada_em = agora
            self._lateral_aberta += 1
            self._lateral_parede = 0
        else:
            self._lateral_parede += 1
            self._lateral_aberta = 0
            self._abertura_iniciada_em = None
            if self._lateral_parede >= cfg.SAIDA_PAREDE_CONFIRMACOES_PAREDE:
                self._parede_lateral_confirmada = True

    def observar_triangulo(self, visivel, timestamp=None):
        agora = time.monotonic() if timestamp is None else float(timestamp)
        self._triangulo_visivel = bool(visivel)
        if self._triangulo_visivel:
            self._ultimo_triangulo_em = agora

    def confirmar_mpu_zerado(self, sucesso, now=None):
        if self.state != self.ZERAR_MPU:
            return False
        if not sucesso:
            self._falhar("MPU ZERO nao foi confirmado")
            return False
        self._entrar(self.AFASTAR_VERMELHO, now)
        return True

    def iniciar_sonda_linha(self):
        if not self.solicita_sonda_linha:
            return False
        self._sonda_iniciada = True
        self._entrar(self.AGUARDANDO_SONDA)
        return True

    def registrar_resultado_sonda(self, resultado, avanco_s, now=None):
        if self.state != self.AGUARDANDO_SONDA:
            return False
        avanco_s = max(float(avanco_s), cfg.SAIDA_PAREDE_RECUO_MINIMO_S)
        if resultado == "preta":
            self._entrar(self.SUCESSO, now)
            return True
        self._tempo_recuo_sonda = avanco_s
        self._entrar(self.RECUAR_SONDA, now)
        return True

    def confirmar_camera_frontal_aberta(self, sucesso, now=None):
        if self.state != self.ABRIR_CAMERA_FRONTAL:
            return False
        if not sucesso:
            self._falhar("camera frontal nao reabriu apos rejeitar a abertura")
            return False
        self._entrar(self.IGNORAR_ABERTURA, now)
        return True

    def notificar_comando_escrito(self, state, now=None):
        if state != self.state or self._comando_aceito:
            return False
        self._comando_aceito = True
        self._inicio_estado = time.monotonic() if now is None else float(now)
        return True

    def atualizar(self, now=None):
        agora = time.monotonic() if now is None else float(now)
        if self.state == self.SUCESSO:
            return MotionCommand(self.SUCESSO, detail="faixa preta confirmada", terminal=True)
        if self.state == self.ABERTURA_ENCONTRADA:
            return MotionCommand(
                self.ABERTURA_ENCONTRADA,
                detail="abertura direita confirmada; robo parado para validacao",
                terminal=True,
            )
        if self.state == self.FALHA:
            return MotionCommand(self.FALHA, detail=self._detalhe_falha, terminal=True)

        if (
            self.state in self._ESTADOS_GIRO_MONITORADO
            and self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_GIRO_S
        ):
            return self._falhar("timeout no giro monitorado por yaw")

        if self._movimento_precisa_de_sensores() and not self._sensores_frescos(agora):
            return self._falhar("leitura de MPU ou ultrassom venceu durante movimento")

        if self.state == self.ZERAR_MPU:
            return MotionCommand(self.ZERAR_MPU, detail="parado; zerando yaw relativo do MPU")

        if self.state == self.AFASTAR_VERMELHO:
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_S:
                self._entrar(self.ASSENTAR_INICIAL, agora)
                return self.atualizar(agora)
            return self._frente(self.AFASTAR_VERMELHO, cfg.SAIDA_PAREDE_AVANCO_APOS_VERMELHO_PWM, "afastando do deposito vermelho")

        if self.state == self.ASSENTAR_INICIAL:
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_ASSENTAMENTO_S:
                if self._yaw is None:
                    return self._falhar("yaw ausente antes do giro inicial")
                # O primeiro comando e sempre um tanque fisico a direita.
                # Assim medimos o sinal real do yaw antes de usar qualquer
                # alvo angular, pois a montagem do MPU pode inverter Z.
                self._yaw_inicio_giro_inicial = self._yaw
                self._alvo_yaw = None
                self._entrar(self.GIRO_INICIAL_DIREITA, agora)
                return self.atualizar(agora)
            return self._parado(self.ASSENTAR_INICIAL, "assentando antes do giro inicial")

        if self.state == self.GIRO_INICIAL_DIREITA:
            if self._sinal_yaw_por_giro_direita is None:
                if self._yaw_inicio_giro_inicial is None or self._yaw is None:
                    return self._falhar("yaw ausente durante calibracao do giro inicial")
                variacao = self._erro_yaw_assinado(
                    self._yaw,
                    self._yaw_inicio_giro_inicial,
                )
                if abs(variacao) >= cfg.SAIDA_PAREDE_VARIACAO_MINIMA_YAW_GRAUS:
                    self._sinal_yaw_por_giro_direita = (
                        1.0 if variacao > 0 else -1.0)
                    self._alvo_yaw = self._normalizar(
                        self._yaw_inicio_giro_inicial
                        + 90.0 * self._sinal_yaw_por_giro_direita
                    )
                else:
                    return self._girar_direita_para_calibrar()
            if self._giro_concluido(agora):
                self._heading_parede = self._alvo_yaw
                self._tentativas_translacao = 0
                self._tempo_translacao_inicial_restante = (
                    cfg.SAIDA_PAREDE_TRANSLACAO_INICIAL_DIREITA_S)
                self._entrar(self.TRANSLADAR_DIREITA_INICIAL, agora)
                return self.atualizar(agora)
            return self._girar(self.GIRO_INICIAL_DIREITA, "girando 90 graus a direita para seguir a parede")

        if self.state == self.TRANSLADAR_DIREITA_INICIAL:
            if (
                self._tempo_translacao_inicial_restante <= 0.0
                or self._tempo_decorrido(agora)
                >= self._tempo_translacao_inicial_restante
            ):
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            if (
                self._erro_heading()
                > cfg.SAIDA_PAREDE_TOLERANCIA_TRANSLACAO_YAW_GRAUS
            ):
                self._tempo_translacao_inicial_restante = max(
                    self._tempo_translacao_inicial_restante
                    - self._tempo_decorrido(agora),
                    0.0,
                )
                self._tentativas_translacao += 1
                if (
                    self._tentativas_translacao
                    > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO
                ):
                    return self._falhar(
                        "yaw saiu da tolerancia na translacao inicial direita")
                self._preparar_giro_para(
                    self._heading_parede,
                    self.CORRIGIR_YAW_TRANSLACAO_INICIAL,
                    agora,
                )
                return self.atualizar(agora)
            return self._lateral(
                self.TRANSLADAR_DIREITA_INICIAL,
                esquerda=False,
                pwm=cfg.SAIDA_PAREDE_PWM_TRANSLACAO_INICIAL,
                detalhe="transladando 0,5 s a direita para alinhar inicialmente",
            )

        if self.state == self.CORRIGIR_YAW_TRANSLACAO_INICIAL:
            if self._giro_concluido(agora):
                self._entrar(self.TRANSLADAR_DIREITA_INICIAL, agora)
                return self.atualizar(agora)
            return self._girar(
                self.CORRIGIR_YAW_TRANSLACAO_INICIAL,
                "corrigindo yaw antes de retomar translacao inicial",
            )

        if self.state == self.SEGUIR_PAREDE:
            if self._frente_proxima >= cfg.SAIDA_PAREDE_CONFIRMACOES_PAREDE:
                self._entrar(self.PARAR_PAREDE, agora)
                return self.atualizar(agora)
            if self._abertura_confirmada(agora):
                self._entrar(self.ABERTURA_ENCONTRADA, agora)
                return self.atualizar(agora)
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_YAW_GRAUS:
                self._tentativas_correcao_seguimento += 1
                if (
                    self._tentativas_correcao_seguimento
                    > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO
                ):
                    return self._falhar(
                        "yaw nao voltou ao rumo enquanto seguia a parede")
                self._preparar_giro_para(
                    self._heading_parede,
                    self.CORRIGIR_YAW_SEGUINDO_PAREDE,
                    agora,
                )
                return self.atualizar(agora)
            self._tentativas_correcao_seguimento = 0
            return self._avancar_alinhando_parede()

        if self.state == self.CORRIGIR_YAW_SEGUINDO_PAREDE:
            if self._giro_concluido(agora):
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            return self._girar(
                self.CORRIGIR_YAW_SEGUINDO_PAREDE,
                "corrigindo yaw para manter a traseira paralela a parede",
            )

        if self.state == self.PARAR_TRIANGULO:
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_ASSENTAMENTO_S:
                self._preparar_giro(-cfg.SAIDA_PAREDE_GIRO_TRIANGULO_GRAUS, self.DESVIAR_TRIANGULO, agora)
                return self.atualizar(agora)
            return self._parado(self.PARAR_TRIANGULO, "triangulo confirmado; freando antes do desvio")

        if self.state == self.DESVIAR_TRIANGULO:
            if self._giro_concluido(agora):
                self._entrar(self.PASSAR_TRIANGULO, agora)
                return self.atualizar(agora)
            return self._girar(self.DESVIAR_TRIANGULO, "girando 45 graus para longe da parede")

        if self.state == self.PASSAR_TRIANGULO:
            decorrido = self._tempo_decorrido(agora)
            if (
                decorrido >= cfg.SAIDA_PAREDE_AVANCO_MIN_TRIANGULO_S
                and not self._triangulo_visivel
            ):
                self._preparar_giro_para(self._heading_parede, self.RETORNAR_TRIANGULO, agora)
                return self.atualizar(agora)
            if decorrido >= cfg.SAIDA_PAREDE_AVANCO_MAX_TRIANGULO_S:
                return self._falhar("triangulo nao saiu do campo durante o desvio")
            return self._frente(self.PASSAR_TRIANGULO, int(round(cfg.SAIDA_PAREDE_VELOCIDADE_APROXIMAR * 120)), "passando o triangulo")

        if self.state == self.RETORNAR_TRIANGULO:
            if self._giro_concluido(agora):
                self._ultimo_triangulo_em = agora
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            return self._girar(self.RETORNAR_TRIANGULO, "voltando ao rumo da parede")

        if self.state == self.PARAR_PAREDE:
            if (
                self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_ASSENTAMENTO_S
                and self._lateral_parede >= cfg.SAIDA_PAREDE_CONFIRMACOES_PAREDE
            ):
                self._preparar_giro(-90.0, self.GIRO_PAREDE_ESQUERDA, agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                return self._falhar(
                    "parede frontal sem parede lateral confirmada; nao girou")
            return self._parado(
                self.PARAR_PAREDE,
                "parede a frente; aguardando confirmacao de parede lateral",
            )

        if self.state == self.GIRO_PAREDE_ESQUERDA:
            if self._giro_concluido(agora):
                self._giros_parede += 1
                if self._giros_parede > cfg.SAIDA_PAREDE_MAX_GIROS_PAREDE:
                    return self._falhar("limite de giros na parede atingido sem saida")
                self._heading_parede = self._alvo_yaw
                self._frente_proxima = 0
                self._lateral_aberta = 0
                self._abertura_iniciada_em = None
                # A leitura lateral durante o giro pertence a parede que
                # acabou de bloquear a frente. Esperamos um eco novo antes de
                # decidir se vale aproximar a direita; se estiver aberto, pode
                # ser justamente a saida e nenhuma translacao deve ocorrer.
                self._lateral_em_antes_do_giro = self._lateral_em
                self._tentativas_translacao = 0
                self._entrar(self.CONFERIR_PAREDE_APOS_GIRO, agora)
                return self.atualizar(agora)
            return self._girar(self.GIRO_PAREDE_ESQUERDA, "girando 90 graus a esquerda e mantendo parede direita")

        if self.state == self.CONFERIR_PAREDE_APOS_GIRO:
            leitura_nova = (
                self._lateral_em is not None
                and (
                    self._lateral_em_antes_do_giro is None
                    or self._lateral_em > self._lateral_em_antes_do_giro
                )
            )
            if not leitura_nova:
                if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                    self._entrar(self.SEGUIR_PAREDE, agora)
                    return self.atualizar(agora)
                return self._parado(
                    self.CONFERIR_PAREDE_APOS_GIRO,
                    "giro concluido; aguardando ultrassom lateral novo",
                )
            if (
                self._lateral_mm is None
                or self._lateral_mm >= cfg.SAIDA_PAREDE_DISTANCIA_ABERTURA_MM
            ):
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            self._tentativas_translacao = 0
            self._entrar(self.ALINHAR_DIREITA_APOS_GIRO, agora)
            return self.atualizar(agora)

        if self.state == self.ALINHAR_DIREITA_APOS_GIRO:
            if (
                self._lateral_mm is None
                or self._lateral_mm >= cfg.SAIDA_PAREDE_DISTANCIA_ABERTURA_MM
            ):
                # A parede sumiu durante a aproximacao: nao atravessa a
                # abertura de lado, volta ao seguimento para testa-la.
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_TRANSLACAO_YAW_GRAUS:
                self._tentativas_translacao += 1
                if self._tentativas_translacao > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO:
                    return self._falhar("yaw saiu da tolerancia ao alinhar pela direita")
                self._preparar_giro_para(
                    self._heading_parede,
                    self.CORRIGIR_YAW_ALINHAMENTO,
                    agora,
                )
                return self.atualizar(agora)
            if (
                self._lateral_mm <= cfg.SAIDA_PAREDE_DISTANCIA_ALINHAMENTO_MM
                or self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_ALINHAMENTO_DIREITA_MAX_S
            ):
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            return self._lateral(
                self.ALINHAR_DIREITA_APOS_GIRO,
                esquerda=False,
                detalhe="parede lateral confirmada; transladando a direita para alinhar",
            )

        if self.state == self.CORRIGIR_YAW_ALINHAMENTO:
            if self._giro_concluido(agora):
                self._entrar(self.ALINHAR_DIREITA_APOS_GIRO, agora)
                return self.atualizar(agora)
            return self._girar(
                self.CORRIGIR_YAW_ALINHAMENTO,
                "corrigindo yaw antes de continuar alinhamento lateral",
            )

        if self.state == self.PARAR_ABERTURA:
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_ASSENTAMENTO_S:
                self._entrar(self.AVANCAR_ENTRADA, agora)
                return self.atualizar(agora)
            return self._parado(self.PARAR_ABERTURA, "abertura direita confirmada; preparando entrada")

        if self.state == self.AVANCAR_ENTRADA:
            if self._frente_proxima >= cfg.SAIDA_PAREDE_CONFIRMACOES_PAREDE:
                return self._falhar("parede frontal durante posicionamento da abertura")
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_AVANCO_ENTRADA_S:
                self._entrar(self.TRANSLADAR_ESQUERDA, agora)
                return self.atualizar(agora)
            return self._frente(self.AVANCAR_ENTRADA, int(round(cfg.SAIDA_PAREDE_VELOCIDADE_APROXIMAR * 120)), "avancando pouco para centralizar a abertura")

        if self.state == self.TRANSLADAR_ESQUERDA:
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_TRANSLACAO_YAW_GRAUS:
                self._tentativas_translacao += 1
                if self._tentativas_translacao > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO:
                    return self._falhar("yaw saiu da tolerancia na translacao esquerda")
                self._preparar_giro_para(self._heading_parede, self.CORRIGIR_YAW_TRANSLACAO, agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TRANSLACAO_ESQUERDA_S:
                self._preparar_giro(90.0, self.GIRO_ENTRADA_DIREITA, agora)
                return self.atualizar(agora)
            return self._lateral(self.TRANSLADAR_ESQUERDA, esquerda=True, detalhe="transladando a esquerda sem girar")

        if self.state == self.CORRIGIR_YAW_TRANSLACAO:
            if self._giro_concluido(agora):
                self._entrar(self.TRANSLADAR_ESQUERDA, agora)
                return self.atualizar(agora)
            return self._girar(self.CORRIGIR_YAW_TRANSLACAO, "corrigindo yaw antes de retomar translacao")

        if self.state == self.GIRO_ENTRADA_DIREITA:
            if self._giro_concluido(agora):
                self._entrar(self.PRONTO_SONDA_LINHA, agora)
                return self.atualizar(agora)
            return self._girar(self.GIRO_ENTRADA_DIREITA, "girando 90 graus a direita para testar a abertura")

        if self.state == self.PRONTO_SONDA_LINHA:
            return self._parado(self.PRONTO_SONDA_LINHA, "parado; trocar frontal por camera de linha com LED aceso")

        if self.state == self.AGUARDANDO_SONDA:
            return self._parado(self.AGUARDANDO_SONDA, "camera de linha testando preto na abertura")

        if self.state == self.RECUAR_SONDA:
            if self._tempo_decorrido(agora) >= self._tempo_recuo_sonda:
                self._preparar_giro_para(self._heading_parede, self.GIRO_RETORNO_ESQUERDA, agora)
                return self.atualizar(agora)
            return MotionCommand(self.RECUAR_SONDA, angle=200, speed=cfg.SAIDA_PAREDE_VELOCIDADE_APROXIMAR, detail="sem preto; recuando da abertura")

        if self.state == self.GIRO_RETORNO_ESQUERDA:
            if self._giro_concluido(agora):
                self._entrar(self.TRANSLADAR_DIREITA, agora)
                return self.atualizar(agora)
            return self._girar(self.GIRO_RETORNO_ESQUERDA, "girando 90 graus a esquerda para voltar a parede")

        if self.state == self.TRANSLADAR_DIREITA:
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_TRANSLACAO_YAW_GRAUS:
                return self._falhar("yaw saiu da tolerancia na translacao de retorno")
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TRANSLACAO_ESQUERDA_S:
                self._entrar(self.ABRIR_CAMERA_FRONTAL, agora)
                return self.atualizar(agora)
            return self._lateral(self.TRANSLADAR_DIREITA, esquerda=False, detalhe="restaurando distancia da parede pela direita")

        if self.state == self.ABRIR_CAMERA_FRONTAL:
            return self._parado(self.ABRIR_CAMERA_FRONTAL, "parado; reabrindo camera frontal com LED apagado")

        if self.state == self.IGNORAR_ABERTURA:
            if self._lateral_parede >= cfg.SAIDA_PAREDE_CONFIRMACOES_PAREDE:
                self._lateral_aberta = 0
                self._entrar(self.SEGUIR_PAREDE, agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_RETORNO_PAREDE_S:
                return self._falhar("parede lateral nao reapareceu apos abertura rejeitada")
            return self._frente(self.IGNORAR_ABERTURA, int(round(cfg.SAIDA_PAREDE_VELOCIDADE_SEGUIR * 120)), "ignorando abertura rejeitada ate reencontrar parede")

        return self._falhar(f"estado de saida por parede desconhecido: {self.state}")

    def _movimento_precisa_de_sensores(self):
        return self.state not in {
            self.ZERAR_MPU,
            self.ASSENTAR_INICIAL,
            self.PARAR_TRIANGULO,
            self.PARAR_PAREDE,
            self.CONFERIR_PAREDE_APOS_GIRO,
            self.PARAR_ABERTURA,
            self.PRONTO_SONDA_LINHA,
            self.AGUARDANDO_SONDA,
            self.ABRIR_CAMERA_FRONTAL,
        }

    def _sensores_frescos(self, agora):
        # Apos MPU ZERO e em toda troca de estado, o primeiro ciclo recebe as
        # consultas assincronas. A pequena janela evita declarar falha antes
        # de uma unica resposta poder chegar, sem tolerar sensor morto depois.
        if agora - self._inicio_estado <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
            return True
        if self._yaw is None or self._yaw_em is None:
            return False
        if agora - self._yaw_em > cfg.SAIDA_PAREDE_TIMEOUT_MPU_S:
            return False
        if self.state in {
            self.AFASTAR_VERMELHO,
            self.SEGUIR_PAREDE,
            self.PASSAR_TRIANGULO,
            self.AVANCAR_ENTRADA,
            self.IGNORAR_ABERTURA,
        }:
            frente_fresca = (
                self._frente_em is not None
                and agora - self._frente_em <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S
            )
            if self.state in (self.SEGUIR_PAREDE, self.IGNORAR_ABERTURA):
                lateral_fresca = (
                    self._lateral_em is not None
                    and agora - self._lateral_em <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S
                )
                return frente_fresca and lateral_fresca
            return frente_fresca
        if self.state == self.ALINHAR_DIREITA_APOS_GIRO:
            return (
                self._lateral_em is not None
                and agora - self._lateral_em <= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S
            )
        return True

    def _preparar_giro(self, delta, estado, agora):
        if self._yaw is None or self._sinal_yaw_por_giro_direita is None:
            self._falhar("yaw ou sentido do giro ausente ao preparar giro")
            return
        self._alvo_yaw = self._normalizar(
            self._yaw + float(delta) * self._sinal_yaw_por_giro_direita)
        self._entrar(estado, agora)

    def _preparar_giro_para(self, alvo, estado, agora):
        if alvo is None:
            self._falhar("heading da parede ausente")
            return
        self._alvo_yaw = self._normalizar(float(alvo))
        self._entrar(estado, agora)

    def _giro_concluido(self, agora):
        if self._alvo_yaw is None or self._yaw is None:
            return False
        return self._erro_yaw(self._alvo_yaw, self._yaw) <= cfg.SAIDA_PAREDE_TOLERANCIA_YAW_GRAUS

    def _erro_heading(self):
        if self._heading_parede is None or self._yaw is None:
            return float("inf")
        return self._erro_yaw(self._heading_parede, self._yaw)

    @staticmethod
    def _erro_yaw(alvo, atual):
        return abs(((float(alvo) - float(atual) + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _erro_yaw_assinado(alvo, atual):
        return ((float(alvo) - float(atual) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _normalizar(angulo):
        return float(angulo) % 360.0

    def _girar(self, estado, detalhe):
        if (
            self._alvo_yaw is None
            or self._yaw is None
            or self._sinal_yaw_por_giro_direita is None
        ):
            return self._falhar("yaw ou sentido ausente durante giro")
        erro_assinado = self._erro_yaw_assinado(self._alvo_yaw, self._yaw)
        # ``steer(180)`` e o tanque fisico a direita. O produto transforma
        # erro no eixo do MPU em direcao fisica, inclusive se Z estiver
        # invertido pela montagem do sensor.
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
            detail="tanque inicial a direita; identificando sinal do yaw",
        )

    def _frente(self, estado, pwm, detalhe):
        return MotionCommand(
            estado,
            angle=0,
            speed=float(pwm) / 120.0,
            detail=detalhe,
        )

    def _avancar_alinhando_parede(self):
        """Avanca e corrige lateralmente, preservando o heading pelo MPU."""
        pwm_frente = int(round(cfg.SAIDA_PAREDE_VELOCIDADE_SEGUIR * 120))
        pwm_lateral = int(cfg.SAIDA_PAREDE_PWM_CORRECAO_LATERAL)
        alvo = int(cfg.SAIDA_PAREDE_DISTANCIA_SEGUIR_ALVO_MM)
        tolerancia = int(cfg.SAIDA_PAREDE_TOLERANCIA_SEGUIR_LATERAL_MM)

        if not self._parede_lateral_confirmada:
            return self._frente(
                self.SEGUIR_PAREDE,
                pwm_frente,
                "seguindo reto ate confirmar a primeira parede direita",
            )
        if (
            self._lateral_mm is None
            or self._lateral_mm >= cfg.SAIDA_PAREDE_DISTANCIA_ABERTURA_MM
        ):
            return self._frente(
                self.SEGUIR_PAREDE,
                pwm_frente,
                "possivel abertura direita; seguindo reto ate confirmar",
            )
        if self._lateral_mm > alvo + tolerancia:
            return self._avanco_com_lateral(
                pwm_frente,
                pwm_lateral,
                "parede direita distante; avancando e aproximando a traseira",
            )
        if self._lateral_mm < alvo - tolerancia:
            return self._avanco_com_lateral(
                pwm_frente,
                -pwm_lateral,
                "parede direita proxima; avancando e afastando a traseira",
            )
        return self._frente(
            self.SEGUIR_PAREDE,
            pwm_frente,
            "seguindo reto alinhado com a parede direita",
        )

    def _avanco_com_lateral(self, pwm_frente, pwm_lateral, detalhe):
        """Soma vetor de frente ao de lateral das rodas omnidirecionais."""
        frente = int(pwm_frente)
        lateral = int(pwm_lateral)
        return MotionCommand(
            self.SEGUIR_PAREDE,
            detail=detalhe,
            wheel_speeds=(
                frente + lateral,
                frente - lateral,
                frente - lateral,
                frente + lateral,
            ),
        )

    def _abertura_confirmada(self, agora):
        """Aceita somente um vao lateral longo, com frente desimpedida."""
        if not self._parede_lateral_confirmada:
            return False
        if self._lateral_aberta < cfg.SAIDA_PAREDE_CONFIRMACOES_ABERTURA:
            return False
        if self._abertura_iniciada_em is None:
            return False
        if (
            agora - self._abertura_iniciada_em
            < cfg.SAIDA_PAREDE_TEMPO_MINIMO_ABERTURA_S
        ):
            return False
        return (
            self._frente_mm is None
            or self._frente_mm
            >= cfg.SAIDA_PAREDE_DISTANCIA_FRENTE_LIVRE_ABERTURA_MM
        )

    def _lateral(self, estado, esquerda, detalhe, pwm=None):
        pwm = int(cfg.SAIDA_PAREDE_PWM_TRANSLACAO if pwm is None else pwm)
        rodas = (
            (-pwm, pwm, pwm, -pwm)
            if esquerda else (pwm, -pwm, -pwm, pwm)
        )
        return MotionCommand(estado, detail=detalhe, wheel_speeds=rodas)

    def _parado(self, estado, detalhe):
        return MotionCommand(estado, detail=detalhe)

    def _tempo_decorrido(self, agora):
        if not self._comando_aceito:
            return 0.0
        return max(float(agora) - self._inicio_estado, 0.0)

    def _entrar(self, estado, now=None):
        self.state = estado
        self._inicio_estado = time.monotonic() if now is None else float(now)
        self._comando_aceito = False
        if estado != self.PRONTO_SONDA_LINHA:
            self._sonda_iniciada = False

    def _falhar(self, detalhe):
        self._detalhe_falha = str(detalhe)
        self._entrar(self.FALHA)
        return MotionCommand(self.FALHA, detail=self._detalhe_falha, terminal=True)
