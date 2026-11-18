"""Natural-scene decoding: image identity from population responses.

Classification is multinomial logistic regression on standardised responses, with the
scaler fit inside every CV fold (Pipeline), so no information leaks across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CHANCE = 1.0 / 118.0

DEFAULT_NEURON_COUNTS = (5, 10, 20, 50, 100, 200)


def ns_xy(rs):
    """Trials x cells responses and image labels, blanks (frame -1) dropped."""
    responses = np.asarray(rs["responses"], dtype=np.float64)
    frames = np.asarray(rs["trial_frame"], dtype=np.int64)
    keep = frames >= 0
    return responses[:, keep].T.copy(), frames[keep].astype(np.int64)


def build_pipeline(C):
    """Standardise inside the fold, then multinomial logistic regression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=C, max_iter=1000)),
        ]
    )


def decode_cv(X, y, C=0.01, n_splits=5, seed=0):
    """Mean cross-validated accuracy over stratified folds."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_index, test_index in splitter.split(X, y):
        model = build_pipeline(C)
        model.fit(X[train_index], y[train_index])
        scores.append(float(model.score(X[test_index], y[test_index])))
    return float(np.mean(scores))


def neuron_curve(X, y, ns=DEFAULT_NEURON_COUNTS, n_draws=10, seed=0):
    """Accuracy for random neuron subsets, always including all neurons."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_cells = X.shape[1]

    targets = sorted({int(n) for n in ns if 0 < int(n) <= n_cells} | {n_cells})
    rng = np.random.default_rng(seed)

    rows = []
    for n_neurons in targets:
        for draw in range(n_draws):
            if n_neurons == n_cells:
                columns = np.arange(n_cells)
            else:
                columns = rng.choice(n_cells, size=n_neurons, replace=False)
            rows.append(
                {
                    "n_neurons": n_neurons,
                    "draw": draw,
                    "accuracy": decode_cv(X[:, columns], y, seed=seed),
                }
            )
    return pd.DataFrame(rows, columns=["n_neurons", "draw", "accuracy"])


def permutation_pvalue(X, y, n_perm=100, seed=0, **decode_kw):
    """Accuracy against shuffled labels: (observed, null, p)."""
    y = np.asarray(y)
    observed = decode_cv(X, y, seed=seed, **decode_kw)

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=np.float64)
    for index in range(int(n_perm)):
        null[index] = decode_cv(X, rng.permutation(y), seed=seed, **decode_kw)

    p_value = (1.0 + float(np.sum(null >= observed))) / (1.0 + int(n_perm))
    return observed, null, p_value


def pca_geometry(X, y):
    """PCA of trial-averaged image vectors: variance ratios and participation ratio."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    labels = np.unique(y)
    means = np.stack([X[y == label].mean(axis=0) for label in labels])

    n_components = min(means.shape)
    pca = PCA(n_components=n_components).fit(means)
    variances = pca.explained_variance_
    total = float(variances.sum())
    participation = float(total**2 / float((variances**2).sum())) if total > 0 else float("nan")

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "participation_ratio": participation,
        "image_mean_responses": means,
        "image_ids": labels,
    }
