"""Geometria e filtro temporal para detectar rampa somente pela camera.

O sinal instantaneo mede uma borda horizontal que separa dois trechos claros
do piso. A exigencia de claridade dos dois lados rejeita a barra preta de um
90 graus; a cobertura horizontal rejeita curvas e riscos pequenos.
"""

import time

import cv2
import numpy as np


def floor_edge_score(frame, bright_min, diff_min, offset, x_margin):
    """Retorna ``(cobertura, y)`` da maior borda horizontal entre pisos claros.

    ``cobertura`` e a fracao da largura util que apresenta, na mesma linha,
    uma mudanca vertical de luminosidade. Pixels acima e abaixo precisam ser
    claros, portanto uma faixa preta transversal nao conta como borda de
    rampa.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    offset = max(int(offset), 1)
    if height <= 2 * offset:
        return 0., -1

    # Media vertical curta reduz textura/ruido sem unir a linha preta ao piso.
    smooth = cv2.blur(gray, (1, 3))
    upper = smooth[:-2 * offset]
    lower = smooth[2 * offset:]

    edge = ((np.abs(lower - upper) >= float(diff_min))
            & (upper >= float(bright_min))
            & (lower >= float(bright_min)))

    margin = min(max(float(x_margin), 0.), .45)
    x0 = int(width * margin)
    x1 = int(width * (1 - margin))
    if x1 <= x0:
        return 0., -1

    row_coverage = np.mean(edge[:, x0:x1], axis=1)
    row = int(np.argmax(row_coverage))
    return float(row_coverage[row]), row + offset


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
