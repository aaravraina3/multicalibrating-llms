"""LOGR: logistic regression on [p, groups]."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class LOGR:
    """version:
    'code'  - Campos code: sklearn defaults, output from predict(), i.e. hard 0/1 labels.
    'paper' - same model, output from predict_proba(), i.e. probabilities.
    """

    def __init__(self, version="code"):
        self.version = version

    def fit(self, p, y, groups):
        self.model = LogisticRegression().fit(np.column_stack([p, groups]), y)
        return self

    def predict(self, p, groups):
        X = np.column_stack([p, groups])
        if self.version == "code":
            return self.model.predict(X).astype(float)
        return self.model.predict_proba(X)[:, 1]
