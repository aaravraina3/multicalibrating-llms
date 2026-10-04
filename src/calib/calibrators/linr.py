"""LINR: one least squares fit that makes every group's average residual zero on train."""

import numpy as np
from sklearn.linear_model import LinearRegression


class LINR:
    """version:
    'code'  - Campos code: LinearRegression with intercept of y on [p, groups]. p gets a learned weight. No clip.
    'paper' - Campos Algorithm 1: least squares of (y - p) on the group indicators, q = clip(p + G @ lambda).
    """

    def __init__(self, version="code"):
        self.version = version

    def fit(self, p, y, groups):
        if self.version == "code":
            self.model = LinearRegression().fit(np.column_stack([p, groups]), y)
        else:
            self.lam, *_ = np.linalg.lstsq(groups.astype(float), y - p, rcond=None)
        return self

    def predict(self, p, groups):
        if self.version == "code":
            return self.model.predict(np.column_stack([p, groups]))
        return np.clip(p + groups.astype(float) @ self.lam, 0, 1)
