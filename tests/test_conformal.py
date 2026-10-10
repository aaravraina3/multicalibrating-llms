import numpy as np

from calib.conformal import choose_threshold, problem_losses, realized_risk


def synthetic(n_problems, rng):
    """10 answers per problem; pass probability shares a problem effect, scores are the true probabilities."""
    difficulty = rng.normal(0, 1.5, n_problems)
    task = np.repeat(np.arange(n_problems), 10)
    p = 1 / (1 + np.exp(-(difficulty[task] + rng.normal(0, 0.5, len(task)))))
    y = (rng.random(len(task)) < p).astype(int)
    return p, y, task


def test_losses_are_non_increasing_in_t():
    rng = np.random.default_rng(0)
    p, y, task = synthetic(50, rng)
    risks = [problem_losses(p, y, task, t).mean() for t in np.linspace(0, 1, 21)]
    assert all(a >= b for a, b in zip(risks, risks[1:]))


def test_conformal_risk_control_holds_on_average():
    """500 fresh calibration draws of 132 problems; average realized risk on fresh test draws <= target."""
    rng = np.random.default_rng(1)
    for target in [0.05, 0.10, 0.15]:
        risks = []
        for _ in range(500):
            p, y, task = synthetic(132, rng)
            t = choose_threshold(p, y, task, target)
            pt, yt, taskt = synthetic(500, rng)
            risks.append(realized_risk(pt, yt, taskt, t))
        assert np.mean(risks) <= target, (target, np.mean(risks))
