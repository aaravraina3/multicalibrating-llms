"""Boosted multicalibration with learned groups (after Globus-Harris et al. 2023).

Each round splits rows into 10 equal width confidence level sets by current prediction. Inside each level set
a depth 2 regression tree fit to the residuals y - p on the features picks the group to fix; the update is
p = clip(p + step * tree prediction). Early stops when held out Brier stops improving."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def level_index(p, n_levels):
    return np.minimum((np.clip(p, 0, 1) * n_levels).astype(int), n_levels - 1)


class BoostedMC:
    def __init__(self, n_levels=10, depth=2, min_leaf=400, step=0.5, max_rounds=50):
        self.n_levels = n_levels
        self.depth = depth
        self.min_leaf = min_leaf  # rows; about 40 problems' worth at 10 rows per problem
        self.step = step
        self.max_rounds = max_rounds

    def _round_update(self, p, X, trees):
        level = level_index(p, self.n_levels)
        delta = np.zeros(len(p))
        for lv, tree in trees.items():
            m = level == lv
            if m.any():
                delta[m] = tree.predict(X[m])
        return np.clip(p + self.step * delta, 0, 1)

    def fit(self, p, y, X, p_val, y_val, X_val):
        p, p_val = np.clip(p.astype(float), 0, 1), np.clip(p_val.astype(float), 0, 1)
        self.rounds, self.val_brier = [], [float(np.mean((p_val - y_val) ** 2))]
        self.stop_reason = "max_rounds"
        for _ in range(self.max_rounds):
            level, resid, trees = level_index(p, self.n_levels), y - p, {}
            for lv in range(self.n_levels):
                m = level == lv
                # A level set with fewer than 2 x min_leaf rows can't split, so its tree is one constant shift.
                if m.any():
                    trees[lv] = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=self.min_leaf,
                                                      random_state=0).fit(X[m], resid[m])
            if not trees:
                self.stop_reason = "no rows"
                break
            new_val = self._round_update(p_val, X_val, trees)
            brier = float(np.mean((new_val - y_val) ** 2))
            if brier >= self.val_brier[-1]:
                self.stop_reason = "validation"
                break
            self.rounds.append(trees)
            self.val_brier.append(brier)
            p, p_val = self._round_update(p, X, trees), new_val
        self.fitted_ = p
        return self

    def predict(self, p, X):
        """Replay the saved rounds in order, without labels."""
        p = np.clip(p.astype(float), 0, 1)
        for trees in self.rounds:
            p = self._round_update(p, X, trees)
        return p

    def leaf_rules(self, feature_names, first_rounds=5):
        """Leaves of the trees in the first rounds, as (round, level set, conditions, leaf value).
        Conditions are (feature, '<=' or '>', threshold) along the path from the root."""
        out = []
        for r, trees in enumerate(self.rounds[:first_rounds]):
            for lv, tree in trees.items():
                t = tree.tree_

                def walk(node, conds):
                    if t.children_left[node] == -1:
                        out.append((r, lv, tuple(conds), float(t.value[node][0][0]), int(t.n_node_samples[node])))
                        return
                    f, thr = feature_names[t.feature[node]], float(t.threshold[node])
                    walk(t.children_left[node], conds + [(f, "<=", thr)])
                    walk(t.children_right[node], conds + [(f, ">", thr)])

                walk(0, [])
        return out


def group_from_conditions(X, feature_names, conds):
    """Boolean membership for a discovered group: rows meeting every condition (level set ignored)."""
    idx = {f: i for i, f in enumerate(feature_names)}
    m = np.ones(len(X), bool)
    for f, op, thr in conds:
        m &= X[:, idx[f]] <= thr if op == "<=" else X[:, idx[f]] > thr
    return m
