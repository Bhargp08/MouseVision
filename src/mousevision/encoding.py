"""Ridge encoding of image-evoked responses from CNN features.

Everything that touches the target images is fit inside the outer CV folds: the feature
scaler and the PCA are fit on training images only, and each neuron's ridge alpha is
chosen by an inner CV on those same training images.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mousevision.rsa import LAYER_ORDER, image_means, split_halves

DEFAULT_ALPHAS = np.logspace(-2, 6, 17)

# Assumed anatomical/functional ordering after Harris et al. 2019 and Siegle et al. 2021.
# This ordering is an assumption, not a measurement; the hierarchy test is only as good
# as it is.
AREA_HIERARCHY = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]


def ridge_fit_predict(Xtr, Ytr, Xte, alphas):
    """Ridge predictions for each alpha, via the SVD of the centred training features."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)

    x_mean = Xtr.mean(axis=0)
    y_mean = Ytr.mean(axis=0)
    U, singular, Vt = np.linalg.svd(Xtr - x_mean, full_matrices=False)
    projected = U.T @ (Ytr - y_mean)

    predictions = []
    for alpha in np.atleast_1d(np.asarray(alphas, dtype=np.float64)):
        shrink = singular / (singular**2 + alpha)
        weights = Vt.T @ (shrink[:, None] * projected)
        predictions.append((Xte - x_mean) @ weights + y_mean)
    return predictions


def _choose_alpha(Ztr, Ytr, alphas, seed, n_splits=5):
    """Per-neuron alpha index from inner CV on the training images."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sse = np.zeros((len(alphas), Ytr.shape[1]))
    for train_index, test_index in splitter.split(Ztr):
        fold_predictions = np.stack(
            ridge_fit_predict(Ztr[train_index], Ytr[train_index], Ztr[test_index], alphas),
            axis=0,
        )
        sse += ((fold_predictions - Ytr[test_index][None, :, :]) ** 2).sum(axis=1)
    return np.argmin(sse, axis=0)


def cv_r2(F, Y, alphas=DEFAULT_ALPHAS, n_splits=5, seed=0, n_pcs=50):
    """Out-of-fold R^2 per neuron, with scaler and PCA fit inside each fold."""
    F = np.asarray(F, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.zeros_like(Y)

    for train_index, test_index in splitter.split(F):
        scaler = StandardScaler().fit(F[train_index])
        train_features = scaler.transform(F[train_index])
        test_features = scaler.transform(F[test_index])

        n_components = max(1, min(n_pcs, train_features.shape[0] - 1, train_features.shape[1]))
        pca = PCA(n_components=n_components).fit(train_features)
        Ztr = pca.transform(train_features)
        Zte = pca.transform(test_features)

        alpha_index = _choose_alpha(Ztr, Y[train_index], alphas, seed)
        fold_predictions = np.stack(
            ridge_fit_predict(Ztr, Y[train_index], Zte, alphas), axis=0
        )
        chosen = np.take_along_axis(
            fold_predictions,
            np.broadcast_to(alpha_index[None, :], (fold_predictions.shape[1], Y.shape[1]))[
                None, :, :
            ],
            axis=0,
        )[0]
        predictions[test_index] = chosen

    sse = ((Y - predictions) ** 2).sum(axis=0)
    sst = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - sse / sst


def neuron_ceiling(X, y, n_splits=20, seed=0):
    """Spearman-Brown ceiling of each neuron's all-repeat image mean."""
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(int(n_splits)):
        first, second = split_halves(X, y, rng)
        correlations.append(
            [
                stats.pearsonr(first[:, index], second[:, index]).statistic
                for index in range(first.shape[1])
            ]
        )

    mean_r = np.asarray(correlations, dtype=np.float64).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * mean_r / (1.0 + mean_r)


def encoding_table(
    response_sets, features, grid=2, min_ceiling=0.3, alphas=DEFAULT_ALPHAS, **cv_kw
):
    """Area x layer summary; per-neuron rows are available as ``table.attrs["neurons"]``."""
    summary = []
    detail = []

    for area, response_set in response_sets.items():
        frames = np.asarray(response_set["trial_frame"], dtype=np.int64)
        keep = frames >= 0
        X = np.asarray(response_set["responses"], dtype=np.float64).T[keep]
        y = frames[keep]

        image_responses = image_means(X, y, n_images=np.unique(y).size)
        ceiling = neuron_ceiling(X, y)
        selected = np.isfinite(ceiling) & (ceiling >= min_ceiling)
        if not selected.any():
            continue

        for layer in LAYER_ORDER:
            key = f"{layer}_g{grid}"
            if key not in features:
                continue
            r2 = cv_r2(np.asarray(features[key]), image_responses[:, selected], alphas=alphas, **cv_kw)
            ratio = r2 / ceiling[selected]

            summary.append(
                {
                    "area": area,
                    "layer": layer,
                    "n_neurons": int(selected.sum()),
                    "median_r2": float(np.median(r2)),
                    "median_r2_over_ceiling": float(np.median(ratio)),
                    "ceiling": float(np.median(ceiling[selected])),
                }
            )
            for position, neuron_index in enumerate(np.flatnonzero(selected)):
                detail.append(
                    {
                        "area": area,
                        "layer": layer,
                        "neuron_index": int(neuron_index),
                        "ceiling": float(ceiling[neuron_index]),
                        "r2": float(r2[position]),
                        "r2_over_ceiling": float(ratio[position]),
                    }
                )

    table = pd.DataFrame(
        summary,
        columns=[
            "area",
            "layer",
            "n_neurons",
            "median_r2",
            "median_r2_over_ceiling",
            "ceiling",
        ],
    )
    table.attrs["neurons"] = pd.DataFrame(
        detail,
        columns=["area", "layer", "neuron_index", "ceiling", "r2", "r2_over_ceiling"],
    )
    return table


def layer_centroid(scores):
    """Layer-rank centroid: sum(i * max(s_i, 0)) / sum(max(s_i, 0)), layers 1..4."""
    if isinstance(scores, dict):
        values = [scores.get(layer, np.nan) for layer in LAYER_ORDER]
    else:
        values = list(scores)
    values = np.asarray(values, dtype=np.float64)
    weights = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    ranks = np.arange(1, weights.size + 1)
    return float((ranks * weights).sum() / total)


def hierarchy_test(centroids):
    """One-sided exact permutation test of centroids following AREA_HIERARCHY."""
    if isinstance(centroids, dict):
        values = np.asarray([centroids[area] for area in AREA_HIERARCHY], dtype=np.float64)
    else:
        values = np.asarray(centroids, dtype=np.float64)

    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    observed = float(stats.spearmanr(values, ranks).statistic)
    null = np.asarray(
        [stats.spearmanr(list(permutation), ranks).statistic for permutation in permutations(values)]
    )
    p_value = float(np.sum(null >= observed) / null.size)
    return observed, p_value
