"""Representational similarity analysis between brain and CNN layers.

Distances are correlation distances: ``RDM[i, j] = 1 - Pearson(r_i, r_j)``
(Kriegeskorte et al. 2008). Reliability is estimated by split-half correlation of
RDMs, corrected with the Spearman-Brown prophecy: ``ceiling = 2r / (1 + r)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LAYER_ORDER = ("layer1", "layer2", "layer3", "layer4")


def image_means(X, y, n_images=118):
    """Mean response per image, in sorted image-id order: (n_images, features)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    labels = np.unique(y)
    if labels.size != n_images:
        raise ValueError(f"expected {n_images} images, found {labels.size}")
    return np.stack([X[y == label].mean(axis=0) for label in labels])


def split_halves(X, y, rng):
    """Split each image's repeats into two random halves and average each half."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    labels = np.unique(y)
    first = np.empty((labels.size, X.shape[1]))
    second = np.empty_like(first)

    for index, label in enumerate(labels):
        rows = np.flatnonzero(y == label)
        if rows.size < 2:
            raise ValueError("every image needs at least two repeats")
        order = rng.permutation(rows)
        half = order.size // 2
        first[index] = X[order[:half]].mean(axis=0)
        second[index] = X[order[half:]].mean(axis=0)

    return first, second


def rdm(patterns):
    """Correlation-distance representational dissimilarity matrix."""
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim != 2:
        raise ValueError("patterns must be (items, features)")
    matrix = 1.0 - np.corrcoef(patterns)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compare_rdms(a, b):
    """Spearman rho between the upper triangles of two RDMs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("RDMs must have the same shape")
    upper = np.triu_indices(a.shape[0], k=1)
    return float(stats.spearmanr(a[upper], b[upper]).statistic)


def rdm_noise_ceiling(X, y, n_splits=20, seed=0):
    """Mean split-half RDM correlation and its Spearman-Brown ceiling."""
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(int(n_splits)):
        first, second = split_halves(X, y, rng)
        correlations.append(compare_rdms(rdm(first), rdm(second)))

    mean_r = float(np.mean(correlations))
    ceiling = 2.0 * mean_r / (1.0 + mean_r) if np.isfinite(mean_r) and mean_r > -1 else float("nan")
    return mean_r, float(ceiling)


def rsa_permutation_p(model_rdm, brain_rdm, n_perm=1000, seed=0):
    """Permutation test of an RSA rho, permuting model rows and columns together."""
    model_rdm = np.asarray(model_rdm, dtype=np.float64)
    brain_rdm = np.asarray(brain_rdm, dtype=np.float64)
    rho = compare_rdms(model_rdm, brain_rdm)

    rng = np.random.default_rng(seed)
    n_images = model_rdm.shape[0]
    null = np.empty(int(n_perm), dtype=np.float64)
    for index in range(int(n_perm)):
        permutation = rng.permutation(n_images)
        permuted = model_rdm[permutation][:, permutation]
        null[index] = compare_rdms(permuted, brain_rdm)

    p_value = (1.0 + float(np.sum(null >= rho))) / (1.0 + int(n_perm))
    return rho, null, p_value


def rsa_table(response_sets, features, grid=1, n_perm=1000, seed=0):
    """RSA per area and layer, with the brain noise ceiling."""
    rows = []
    for area, response_set in response_sets.items():
        responses = np.asarray(response_set["responses"], dtype=np.float64).T
        frames = np.asarray(response_set["trial_frame"], dtype=np.int64)
        keep = frames >= 0
        X, y = responses[keep], frames[keep]

        brain_rdm = rdm(image_means(X, y, n_images=np.unique(y).size))
        _, ceiling = rdm_noise_ceiling(X, y, seed=seed)

        for layer in LAYER_ORDER:
            key = f"{layer}_g{grid}"
            if key not in features:
                continue
            model_rdm = rdm(features[key])
            rho, _, p_value = rsa_permutation_p(
                model_rdm, brain_rdm, n_perm=n_perm, seed=seed
            )
            rows.append(
                {
                    "area": area,
                    "layer": layer,
                    "rho": rho,
                    "ceiling": ceiling,
                    "rho_over_ceiling": rho / ceiling if ceiling and np.isfinite(ceiling) else float("nan"),
                    "p": p_value,
                }
            )

    return pd.DataFrame(
        rows, columns=["area", "layer", "rho", "ceiling", "rho_over_ceiling", "p"]
    )
