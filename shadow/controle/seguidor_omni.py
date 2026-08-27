"""Controle visual do segue-linha para a base omni em X, sem depender de MPU."""

import time

import numpy as np

import config
from controle.direcao import drive_omni


class ControladorSeguidorOmni:
    def __init__(self):
        self.reset()

    def reset(self):
        self._ultimo_t = None
        self._filtrado = np.zeros(3, dtype=np.float64)
        self._rodas_anteriores = None
        self._recentrando = False
        self.modo = "fallback"

    @staticmethod
    def _zona_morta(valor):
        return 0. if abs(valor) < config.LINE_V2_ERROR_DEADBAND else valor

    def _filtrar(self, valores, agora):
        if self._ultimo_t is None:
            self._filtrado[:] = valores
        else:
            dt = max(agora - self._ultimo_t, 1e-3)
            alpha = dt / (config.LINE_V2_FILTER_TAU_S + dt)
            self._filtrado += alpha * (valores - self._filtrado)
        self._ultimo_t = agora
        return self._filtrado.copy()

    def _limitar_degrau(self, rodas):
        rodas = np.asarray(rodas, dtype=np.float64)
        if self._rodas_anteriores is not None:
            delta = np.clip(
                rodas - self._rodas_anteriores,
                -config.LINE_V2_MAX_PWM_STEP,
                config.LINE_V2_MAX_PWM_STEP,
            )
            rodas = self._rodas_anteriores + delta
        self._rodas_anteriores = rodas
        return rodas

    def atualizar(self, resultado, velocidade, agora=None):
        """Envia um comando e retorna True; False solicita fallback legado."""
        agora = time.monotonic() if agora is None else float(agora)
        fresco = (
            resultado.publicado_em > 0.
            and agora - resultado.publicado_em
            <= config.LINE_V2_MAX_RESULT_AGE_S
        )
        if (
            not resultado.valida
            or not fresco
            or resultado.confianca < config.LINE_V2_MIN_CONFIDENCE
        ):
            self.reset()
            return False

        valores = np.asarray((
            resultado.lateral,
            resultado.orientacao,
            resultado.curvatura,
        ), dtype=np.float64)
        lateral, orientacao, curvatura = self._filtrar(valores, agora)
        lateral = self._zona_morta(float(lateral))
        orientacao = self._zona_morta(float(orientacao))
        curvatura = self._zona_morta(float(curvatura))

        # O ponto inferior representa a faixa exatamente no eixo da camera.
        # Histerese impede que ruido perto da borda do corredor ligue/desligue
        # o modo a cada frame.
        if self._recentrando:
            if abs(lateral) <= config.LINE_V2_CENTER_EXIT_ERROR:
                self._recentrando = False
        elif abs(lateral) >= config.LINE_V2_CENTER_ENTER_ERROR:
            self._recentrando = True

        severidade = min(max(
            abs(orientacao),
            abs(curvatura) * .85,
            abs(lateral) * .55,
        ), 1.)
        pwm_base = min(
            float(velocidade) * config.MAX_PWM,
            float(config.LINE_V2_WHEEL_MAX_PWM),
        )
        proporcao_frente = max(
            config.LINE_V2_MIN_FORWARD_RATIO,
            1. - config.LINE_V2_CURVE_SPEED_REDUCTION * severidade,
        )
        if self._recentrando:
            proporcao_frente = min(
                proporcao_frente,
                config.LINE_V2_RECENTER_FORWARD_RATIO,
            )
        frente = pwm_base * proporcao_frente
        ganho_lateral = (
            config.LINE_V2_RECENTER_LATERAL_KP_PWM
            if self._recentrando else config.LINE_V2_LATERAL_KP_PWM
        )
        limite_lateral = (
            config.LINE_V2_RECENTER_LATERAL_MAX_PWM
            if self._recentrando else config.LINE_V2_LATERAL_MAX_PWM
        )
        ganho_yaw_lateral = (
            config.LINE_V2_RECENTER_LATERAL_YAW_KP_PWM
            if self._recentrando else config.LINE_V2_LATERAL_YAW_KP_PWM
        )
        escala_curvatura = (
            config.LINE_V2_RECENTER_CURVATURE_SCALE
            if self._recentrando else 1.
        )
        lateral_pwm = float(np.clip(
            ganho_lateral * lateral,
            -limite_lateral,
            limite_lateral,
        ))
        rotacao_pwm = float(np.clip(
            config.LINE_V2_HEADING_KP_PWM * orientacao
            + ganho_yaw_lateral * lateral
            + config.LINE_V2_CURVATURE_FF_PWM * curvatura * escala_curvatura,
            -config.LINE_V2_ROTATION_MAX_PWM,
            config.LINE_V2_ROTATION_MAX_PWM,
        ))

        rodas = np.asarray((
            frente + lateral_pwm + rotacao_pwm,
            frente - lateral_pwm + rotacao_pwm,
            frente - lateral_pwm - rotacao_pwm,
            frente + lateral_pwm - rotacao_pwm,
        ))
        pico = float(np.max(np.abs(rodas)))
        limite = min(pwm_base, float(config.LINE_V2_WHEEL_MAX_PWM))
        if pico > limite and pico > 0.:
            rodas *= limite / pico
        rodas = self._limitar_degrau(rodas)

        # Converte novamente em eixos para manter toda a saturacao e a ordem
        # fisica centralizadas em drive_omni().
        frente = float(np.mean(rodas))
        lateral_pwm = float((rodas[0] - rodas[1] - rodas[2] + rodas[3]) / 4.)
        rotacao_pwm = float((rodas[0] + rodas[1] - rodas[2] - rodas[3]) / 4.)
        self.modo = (
            "recentrando" if self._recentrando
            else "curva" if severidade >= .32
            else "normal"
        )
        drive_omni(frente, lateral_pwm, rotacao_pwm, limite)
        return True
