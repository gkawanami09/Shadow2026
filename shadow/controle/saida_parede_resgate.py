"""Rota de parede apos o deposito vermelho.

Depois do avanco curto e do giro inicial de 90 graus, cada passagem segue a
mesma ordem: avanco ate 118 mm no ultrassom frontal, pivo traseiro ate o
lateral estabilizar (ou completar 2 s), translacao para a direita e
afastamento para a esquerda ate o lateral marcar ao menos 120 mm. A camera
frontal, com LED apagado, abre somente depois da primeira passagem. Se ela
confirmar o triangulo verde, o robo faz mais duas passagens e para apos o
ultimo afastamento a 120 mm.
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
    """Executa a rota pos-vermelho em tres passagens de parede."""

    ZERAR_MPU = "EXIT_PAREDE_ZERAR_MPU"
    AFASTAR_VERMELHO = "EXIT_PAREDE_AFASTAR_VERMELHO"
    ASSENTAR_INICIAL = "EXIT_PAREDE_ASSENTAR_INICIAL"
    GIRO_INICIAL_DIREITA = "EXIT_PAREDE_GIRO_INICIAL_DIREITA"
    AFASTAR_ESQUERDA_120 = "EXIT_PAREDE_AFASTAR_ESQUERDA_120"
    CORRIGIR_YAW_AFASTAMENTO_ESQUERDA = (
        "EXIT_PAREDE_CORRIGIR_YAW_AFASTAMENTO_ESQUERDA"
    )
    AVANCAR_ATE_PAREDE_FRENTE = "EXIT_PAREDE_AVANCAR_ATE_FRENTE"
    CORRIGIR_YAW_AVANCO_FRENTE = "EXIT_PAREDE_CORRIGIR_YAW_FRENTE"
    PIVO_TRASEIRO_ESTABILIZAR = "EXIT_PAREDE_PIVO_TRASEIRO_ESTABILIZAR"
    TRANSLADAR_DIREITA = "EXIT_PAREDE_TRANSLADAR_DIREITA"
    VERIFICAR_TRIANGULO_VERDE = "EXIT_PAREDE_VERIFICAR_TRIANGULO_VERDE"
    AGUARDAR_MPU_TRIANGULO_VERDE = "EXIT_PAREDE_AGUARDAR_MPU_TRIANGULO_VERDE"
    SAIDA_CONCLUIDA = "EXIT_PAREDE_SAIDA_CONCLUIDA"
    FALHA = "EXIT_PAREDE_FALHA"

    _DESTINO_CAMERA = "CAMERA"
    _DESTINO_AVANCO = "AVANCO"
    _DESTINO_PARAR = "PARAR"

    _ESTADOS_GIRO = {
        GIRO_INICIAL_DIREITA,
        CORRIGIR_YAW_AFASTAMENTO_ESQUERDA,
        CORRIGIR_YAW_AVANCO_FRENTE,
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
        self._lateral_em_antes_afastamento = None
        self._frente_mm = None
        self._frente_em = None
        self._frente_em_antes_avanco = None
        self._lateral_em_antes_pivo = None
        self._lateral_em_processada_pivo = None
        self._lateral_referencia_pivo_mm = None
        self._estavel_desde_pivo = None
        self._translacao_por_timeout_pivo = False
        self._triangulo_verde_confirmado = False
        self._yaw_em_antes_retomada_verde = None
        self._tentativas_correcao = 0
        self._passagens_direita_concluidas = 0
        self._destino_apos_afastamento = None

    @property
    def terminal(self):
        return self.state in (self.SAIDA_CONCLUIDA, self.FALHA)

    @property
    def solicita_zerar_mpu(self):
        return self.state == self.ZERAR_MPU

    @property
    def prioriza_mpu(self):
        return self.state in (
            self._ESTADOS_GIRO | {self.AGUARDAR_MPU_TRIANGULO_VERDE}
        )

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
                "triangulo verde tratado e tres passagens de parede concluidas; robo parado",
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
                # O proximo marco e exclusivamente a parede frontal a 118 mm;
                # nao ha alinhamento lateral logo apos os 90 graus.
                self._entrar_avanco_frente(agora)
                return self.atualizar(agora)
            return self._girar(
                self.GIRO_INICIAL_DIREITA,
                "girando 90 graus a direita pelo MPU",
            )

        if self.state == self.AFASTAR_ESQUERDA_120:
            if not self._lateral_fresca_desde_afastamento(agora):
                if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                    return self._falhar(
                        "ultrassom lateral sem leitura nova durante o "
                        "afastamento para a esquerda",
                        agora,
                    )
                return self._parado(
                    self.AFASTAR_ESQUERDA_120,
                    "translacao direita concluida; aguardando ultrassom "
                    "lateral novo para afastar a esquerda",
                )
            if self._lateral_mm is None:
                return self._falhar(
                    "ultrassom lateral respondeu sem eco durante o "
                    "afastamento para a esquerda",
                    agora,
                )
            if (
                self._lateral_mm
                >= cfg.SAIDA_PAREDE_DISTANCIA_MINIMA_ESQUERDA_MM
            ):
                self._executar_destino_apos_afastamento(agora)
                return self.atualizar(agora)
            if (
                self._tempo_decorrido(agora)
                >= cfg.SAIDA_PAREDE_TIMEOUT_AFASTAMENTO_ESQUERDA_S
            ):
                return self._falhar(
                    "timeout ao afastar a esquerda ate "
                    f"{cfg.SAIDA_PAREDE_DISTANCIA_MINIMA_ESQUERDA_MM} mm "
                    f"da parede direita (ultima leitura={self._lateral_mm} mm)",
                    agora,
                )
            if not self._mpu_fresco(agora):
                return self._falhar(
                    "MPU sem leitura durante afastamento para a esquerda",
                    agora,
                )
            if self._erro_heading() > cfg.SAIDA_PAREDE_TOLERANCIA_TRANSLACAO_YAW_GRAUS:
                self._tentativas_correcao += 1
                if self._tentativas_correcao > cfg.SAIDA_PAREDE_MAX_TENTATIVAS_TRANSLACAO:
                    return self._falhar(
                        "yaw nao voltou ao rumo durante afastamento para a esquerda",
                        agora,
                    )
                self._preparar_giro_para(
                    self._heading_parede,
                    self.CORRIGIR_YAW_AFASTAMENTO_ESQUERDA,
                    agora,
                )
                return self.atualizar(agora)
            self._tentativas_correcao = 0
            return self._lateral(
                self.AFASTAR_ESQUERDA_120,
                direita=False,
                detalhe=(
                    "lateral abaixo de "
                    f"{cfg.SAIDA_PAREDE_DISTANCIA_MINIMA_ESQUERDA_MM} mm; "
                    "transladando para a esquerda com yaw monitorado"
                ),
            )

        if self.state == self.CORRIGIR_YAW_AFASTAMENTO_ESQUERDA:
            if self._giro_concluido():
                self._entrar_afastamento_esquerda(agora)
                return self.atualizar(agora)
            return self._girar(
                self.CORRIGIR_YAW_AFASTAMENTO_ESQUERDA,
                "corrigindo yaw antes de continuar o afastamento a esquerda",
            )

        if self.state == self.AVANCAR_ATE_PAREDE_FRENTE:
            if not self._frente_fresca(agora):
                if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_SENSOR_S:
                    return self._falhar(
                        "ultrassom frontal sem leitura nova antes do avanco",
                        agora,
                    )
                return self._parado(
                    self.AVANCAR_ATE_PAREDE_FRENTE,
                    "aguardando ultrassom frontal novo antes do avanco",
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
                # O frame confirmado pode ser o ultimo antes de fechar a
                # camera. O robo fica parado ate haver uma leitura nova do
                # MPU, garantindo que a camera foi liberada antes do avanco.
                self._yaw_em_antes_retomada_verde = self._yaw_em
                self._entrar(self.AGUARDAR_MPU_TRIANGULO_VERDE, agora)
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

        if self.state == self.AGUARDAR_MPU_TRIANGULO_VERDE:
            if not self._mpu_fresco_depois_retomada_verde(agora):
                if self._tempo_decorrido(agora) >= (
                    cfg.SAIDA_PAREDE_TIMEOUT_MPU_APOS_CAMERA_S
                ):
                    return self._falhar(
                        "MPU nao respondeu apos fechar a camera do triangulo verde",
                        agora,
                    )
                return self._parado(
                    self.AGUARDAR_MPU_TRIANGULO_VERDE,
                    "triangulo verde confirmado; camera fechando e aguardando yaw novo",
                )
            self._entrar_avanco_frente(agora)
            return self.atualizar(agora)

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
                self._entrar_transladar_direita(agora)
                return self.atualizar(agora)
            if self._tempo_decorrido(agora) >= cfg.SAIDA_PAREDE_TIMEOUT_PIVO_TRASEIRO_S:
                self._translacao_por_timeout_pivo = True
                self._entrar_transladar_direita(agora)
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

        if self.state == self.TRANSLADAR_DIREITA:
            if self._tempo_decorrido(agora) >= (
                cfg.SAIDA_PAREDE_TRANSLACAO_FINAL_DIREITA_S
            ):
                self._passagens_direita_concluidas += 1
                destinos = {
                    1: self._DESTINO_CAMERA,
                    2: self._DESTINO_AVANCO,
                    3: self._DESTINO_PARAR,
                }
                destino = destinos.get(self._passagens_direita_concluidas)
                if destino is None:
                    return self._falhar(
                        "quantidade inesperada de passagens de parede",
                        agora,
                    )
                self._entrar_afastamento_esquerda(agora, destino)
                return self.atualizar(agora)
            detalhe = (
                "lateral oscilou por tempo demais; transladando para a direita "
                "mesmo sem estabilizar"
                if self._translacao_por_timeout_pivo else
                "leitura lateral estavel; transladando para a direita por 0,5 s"
            )
            return self._lateral(
                self.TRANSLADAR_DIREITA,
                direita=True,
                pwm=cfg.SAIDA_PAREDE_PWM_TRANSLACAO_FINAL_DIREITA,
                detalhe=detalhe,
            )

        return self._falhar(f"estado de rota desconhecido: {self.state}", agora)

    def _entrar_afastamento_esquerda(self, agora, destino=None):
        # A leitura que concluiu a translacao direita nao pode concluir esta
        # etapa: exige uma leitura lateral nova feita ja com o chassi parado.
        if destino is not None:
            self._destino_apos_afastamento = destino
        self._lateral_em_antes_afastamento = self._lateral_em
        self._tentativas_correcao = 0
        self._entrar(self.AFASTAR_ESQUERDA_120, agora)

    def _entrar_transladar_direita(self, agora):
        # O pivo traseiro muda propositalmente a orientacao do chassi. A
        # translacao que vem depois deve manter essa orientacao atual, e nao o
        # yaw guardado antes de encostar na parede frontal. Sem esta troca de
        # referencia, o afastamento esquerdo tentaria desfazer o pivo e viraria
        # no lugar em vez de deslizar lateralmente.
        if self._mpu_fresco(agora):
            self._heading_parede = self._yaw
        self._entrar(self.TRANSLADAR_DIREITA, agora)

    def _executar_destino_apos_afastamento(self, agora):
        if self._destino_apos_afastamento == self._DESTINO_CAMERA:
            self._entrar_verificacao_triangulo_verde(agora)
            return
        if self._destino_apos_afastamento == self._DESTINO_AVANCO:
            self._entrar_avanco_frente(agora)
            return
        if self._destino_apos_afastamento == self._DESTINO_PARAR:
            self._entrar(self.SAIDA_CONCLUIDA, agora)
            return
        self._falhar("destino ausente apos afastamento lateral", agora)

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
            pwm = cfg.SAIDA_PAREDE_PWM_TRANSLACAO_ESQUERDA
        pwm = int(pwm)
        rodas = (pwm, -pwm, -pwm, pwm) if direita else (-pwm, pwm, pwm, -pwm)
        return MotionCommand(estado, detail=detalhe, wheel_speeds=rodas)

    def _mpu_fresco(self, agora):
        return (
            self._yaw is not None
            and self._yaw_em is not None
            and agora - self._yaw_em <= cfg.SAIDA_PAREDE_TIMEOUT_MPU_S
        )

    def _mpu_fresco_depois_retomada_verde(self, agora):
        return (
            self._mpu_fresco(agora)
            and self._yaw_em_antes_retomada_verde is not None
            and self._yaw_em > self._yaw_em_antes_retomada_verde
        )

    def _lateral_fresca_desde_afastamento(self, agora):
        return (
            self._lateral_em is not None
            and self._lateral_em_antes_afastamento is not None
            and self._lateral_em > self._lateral_em_antes_afastamento
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
    detector_painel_verde_factory=None,
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
    if detector_painel_verde_factory is None:
        from visao.marcador_resgate import GreenRectangleDetector
        detector_painel_verde_factory = GreenRectangleDetector
    if fonte_assincrona_factory is None:
        from visao.resgate_assincrono import LatestFrameSource
        fonte_assincrona_factory = LatestFrameSource

    controlador = ControladorSaidaParede()
    monitor_sensores = MonitorSensoresSaida(arduino)
    epoca_serial = arduino.connection_epoch
    ultimo_estado = None
    ultimo_log_yaw = -float("inf")
    fonte_verde = None
    detector_triangulo_verde = None
    detector_painel_verde = None
    ultima_sequencia_verde = 0
    ultimo_frame_verde = None
    proximo_log_debug_verde = 0.0

    def fechar_camera_verde():
        nonlocal fonte_verde, detector_triangulo_verde, detector_painel_verde
        nonlocal ultima_sequencia_verde, ultimo_frame_verde
        if fonte_verde is not None:
            fonte_verde.close()
        fonte_verde = None
        detector_triangulo_verde = None
        detector_painel_verde = None
        ultima_sequencia_verde = 0
        ultimo_frame_verde = None

    def salvar_debug_verde(motivo):
        """Salva o quadro e as duas mascaras para calibrar na Raspberry."""
        if not debug or ultimo_frame_verde is None:
            return
        from pathlib import Path

        import cv2

        pasta = Path("/tmp/shadow-saida-verde")
        pasta.mkdir(parents=True, exist_ok=True)
        sufixo = str(int(time.time() * 1000))
        bruto = pasta / f"{sufixo}-bruto.png"
        anotado = ultimo_frame_verde.copy()
        altura, largura = anotado.shape[:2]
        topo_roi = int(round(altura * cfg.MARKER_ROI_TOP))
        cv2.line(anotado, (0, topo_roi), (largura - 1, topo_roi),
                 (0, 255, 255), 1)
        for candidato in detector_triangulo_verde.last_candidates:
            x, y, w, h = (int(valor) for valor in candidato.bbox)
            cv2.rectangle(anotado, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(anotado, motivo, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(bruto), ultimo_frame_verde)
        cv2.imwrite(str(pasta / f"{sufixo}-anotado.png"), anotado)
        if detector_triangulo_verde.last_mask is not None:
            cv2.imwrite(
                str(pasta / f"{sufixo}-mascara-triangulo.png"),
                detector_triangulo_verde.last_mask)
        if detector_painel_verde.last_mask is not None:
            cv2.imwrite(
                str(pasta / f"{sufixo}-mascara-painel.png"),
                detector_painel_verde.last_mask)
        print(
            "[saida] debug verde salvo em "
            f"{pasta} ({motivo})")

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
                raise RuntimeError("serial mudou durante a rota de saida")

            monitor_sensores.atualizar_controlador(controlador, agora)
            if controlador.usa_camera_triangulo_verde:
                if fonte_verde is None:
                    if arduino.led("APAGADO") is False:
                        raise RuntimeError(
                            "nao foi possivel manter o LED apagado na camera frontal")
                    fonte_verde = fonte_assincrona_factory(
                        camera_factory(camera_index))
                    detector_triangulo_verde = detector_verde_factory("green")
                    detector_painel_verde = detector_painel_verde_factory()
                    print(
                        "[saida] camera frontal ativa com LED APAGADO; "
                        "procurando triangulo verde")
                quadro = fonte_verde.poll(ultima_sequencia_verde)
                if quadro is not None:
                    ultima_sequencia_verde = quadro.sequence
                    ultimo_frame_verde = quadro.frame
                    deteccao_triangulo = detector_triangulo_verde.detect(
                        quadro.frame,
                        timestamp=quadro.captured_at,
                    )
                    deteccao_painel = detector_painel_verde.detect(
                        quadro.frame,
                        timestamp=quadro.captured_at,
                    )
                    deteccao_confirmada = next(
                        (
                            deteccao
                            for deteccao in (
                                deteccao_triangulo,
                                deteccao_painel,
                            )
                            if deteccao is not None and deteccao.confirmed
                        ),
                        None,
                    )
                    if deteccao_confirmada is not None and (
                        deteccao_confirmada.confirmed
                    ) and controlador.observar_triangulo_verde(
                        True,
                        quadro.captured_at,
                    ):
                        origem = (
                            "triangulo" if deteccao_confirmada is deteccao_triangulo
                            else "painel verde-ciano")
                        print(
                            "[saida] verde confirmado pela camera "
                            f"({origem}) em {deteccao_confirmada.hits} frame(s)")
                    elif debug and agora >= proximo_log_debug_verde:
                        proximo_log_debug_verde = agora + 0.40
                        mascara_triangulo = detector_triangulo_verde.last_mask
                        proporcao_triangulo = (
                            0.0 if mascara_triangulo is None else
                            int((mascara_triangulo > 0).sum())
                            / float(max(mascara_triangulo.size, 1))
                        )
                        print(
                            "[saida] debug verde: "
                            f"mascara-triangulo={proporcao_triangulo:.2%} "
                            f"candidatos={len(detector_triangulo_verde.last_candidates)} "
                            f"rejeicoes={detector_triangulo_verde.last_rejections}; "
                            f"mascara-painel={detector_painel_verde.last_mask_ratio:.2%} "
                            f"rejeicoes-painel={detector_painel_verde.last_rejections}")
            comando = controlador.atualizar(agora)

            if (
                comando.state == ControladorSaidaParede.FALHA
                and fonte_verde is not None
            ):
                salvar_debug_verde(comando.detail)

            # A camera frontal e fechada antes do avanco seguinte. Isso
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
                raise RuntimeError("comando da rota de saida nao foi enviado")
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
