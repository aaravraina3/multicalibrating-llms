"""Platt scaling: q = sigmoid(a * f(p) + b), one input, two learned numbers."""

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


class Platt:
    """version:
    'code'  - Campos code: input raw p, sklearn default C=1. Produces their Table 1 numbers.
    'paper' - Campos paper Eq. 15: input log p, C=1e6 (almost no penalty).
    'logit' - common version: input logit p, C=1e6.
    """

    def __init__(self, version="code"):
        self.version = version

    def _input(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        if self.version == "code":
            return p.reshape(-1, 1)
        if self.version == "paper":
            return np.log(p).reshape(-1, 1)
        return logit(p).reshape(-1, 1)

    def fit(self, p, y, groups=None):
        C = 1.0 if self.version == "code" else 1e6
        self.model = LogisticRegression(C=C).fit(self._input(p), y)
        self.a, self.b = float(self.model.coef_[0, 0]), float(self.model.intercept_[0])
        return self

    def predict(self, p, groups=None):
        return self.model.predict_proba(self._input(p))[:, 1]
