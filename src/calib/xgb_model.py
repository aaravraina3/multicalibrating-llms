"""B4: XGBoost on the B2 features, hyperparameters from an Optuna TPE search scored by grouped cross validation."""

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from calib.metrics import log_loss

SEED = 0


def make_xgb(params, seed=SEED):
    return XGBClassifier(objective="binary:logistic", tree_method="hist", eval_metric="logloss",
                         random_state=seed, n_jobs=4, **params)


def space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def grouped_cv_loss(params, X, y, groups, n_splits=5, seed=SEED):
    """Mean log loss over GroupKFold by problem. No early stopping, so nothing leaks across folds."""
    losses = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_xgb(params, seed).fit(X[tr], y[tr])
        losses.append(log_loss(model.predict_proba(X[va])[:, 1], y[va]))
    return float(np.mean(losses))


def search(X, y, groups, n_trials=100, seed=SEED, params_space=space):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: grouped_cv_loss(params_space(t), X, y, groups), n_trials=n_trials)
    return study


def oof_models(params, X, y, groups, seed=SEED, n_splits=5):
    """The same grouped folds as the search: (validation indices, model fit on the rest) per fold."""
    return [(va, make_xgb(params, seed).fit(X[tr], y[tr])) for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups)]
