"""Feature rich base predictors for RQ2. Trained on base_train, hyperparameters picked on official validation."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calib.metrics import log_loss

VARIANTS = {
    "B0": ("logistic", ["avg"]),
    "B1": ("logistic", ["avg", "size", "logprob", "structure"]),
    "B2": ("logistic", ["avg", "size", "logprob", "structure", "self_consistency"]),
    "B3": ("boosting", ["avg", "size", "logprob", "structure", "self_consistency"]),
}
C_GRID = [0.01, 0.1, 1, 10]
BOOST_GRID = [(n, lr) for n in [50, 100, 200, 400] for lr in [0.03, 0.1]]


def logistic(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def boosting(max_iter, learning_rate):
    # early_stopping=False: its built in holdout splits by row and leaks across problems.
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=8,
                                          min_samples_leaf=50, early_stopping=False, random_state=0)


def fit_select(kind, X_tr, y_tr, X_va, y_va):
    """Fit every grid setting on train, keep the one with the lowest validation log loss."""
    grid = [(C,) for C in C_GRID] if kind == "logistic" else BOOST_GRID
    best = None
    for params in grid:
        model = logistic(*params) if kind == "logistic" else boosting(*params)
        model.fit(X_tr, y_tr)
        loss = log_loss(model.predict_proba(X_va)[:, 1], y_va)
        if best is None or loss < best[0]:
            best = (loss, params, model)
    return best[2], best[1], best[0]


def fit_fixed(kind, params, X, y):
    """Refit with already chosen hyperparameters (final run: no re-selection)."""
    model = logistic(*params) if kind == "logistic" else boosting(*params)
    return model.fit(X, y)
