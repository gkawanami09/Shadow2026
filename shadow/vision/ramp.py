"""Filtro temporal para a deteccao de rampa usando somente a camera.

O sinal instantaneo vem da mudanca ampla de iluminacao/perspectiva ja medida
na parte superior da imagem. Este modulo nao interpreta um frame isolado como
rampa: a evidencia precisa permanecer continua durante o tempo de confirmacao.
"""

import time


class RampDetector:
    """Confirma candidatos persistentes e rejeita pulsos curtos."""

    def __init__(self, confirm_time, release_time, gap_time=0.):
        self.confirm_time = max(float(confirm_time), 0.)
        self.release_time = max(float(release_time), 0.)
        self.gap_time = max(float(gap_time), 0.)
        self._candidate_since = None
        self._last_candidate_at = None
        self._active_until = None

    def update(self, candidate, now=None):
        """Retorna True somente depois de uma evidencia temporal continua."""
        now = time.monotonic() if now is None else float(now)

        if candidate:
            if self._candidate_since is None:
                self._candidate_since = now
            self._last_candidate_at = now
            if now - self._candidate_since >= self.confirm_time:
                self._active_until = now + self.release_time
        elif (self._last_candidate_at is None
              or now - self._last_candidate_at > self.gap_time):
            self._candidate_since = None
            self._last_candidate_at = None

        return self._active_until is not None and now <= self._active_until

    @property
    def confirming(self):
        return self._candidate_since is not None
