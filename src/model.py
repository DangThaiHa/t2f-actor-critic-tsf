import logging
import numpy as np
from upstream import t2f_base as core


def symmetric(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    assert a.shape == b.shape
    den = np.abs(a) + np.abs(b)
    return np.divide(2 * np.abs(a - b), den, out=np.zeros_like(den), where=den > 0)


core.smape = lambda a, b: 100 * symmetric(a, b).mean()
core.wsmae = lambda a, b, w: np.mean(np.asarray(w).ravel() * symmetric(a, b))


class RidgeVAR(core.SimpleVAR):
    def fit(self, x):
        if len(x) == 1:
            return super().fit(x)
        self.lag = min(len(x) - 1, max(1, len(x) // 4))
        design = np.column_stack(
            [x[j : len(x) - self.lag + j] for j in range(self.lag)]
        )
        gram = design.T @ design
        penalty = 0.05 * max(np.trace(gram) / len(gram), 1e-8)
        self.coefficients = np.linalg.solve(
            gram + penalty * np.eye(len(gram)), design.T @ x[self.lag :]
        )


class FailOnFallback(logging.Handler):
    def emit(self, record):
        if "failed for series" in record.getMessage():
            raise RuntimeError(record.getMessage())


logger = logging.getLogger("T2f")
logger.handlers = [FailOnFallback()]
