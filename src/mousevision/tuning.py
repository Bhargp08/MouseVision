"""Direction and temporal-frequency tuning from drifting-grating response sets.

Allen conventions at the preferred temporal frequency:

* ``pref`` = argmax over (direction, TF)
* ``orth`` = mean of the two directions at pref +/- 90 degrees
* ``null`` = the direction opposite to pref
* ``OSI = (pref - orth) / (pref + orth)``, ``DSI = (pref - null) / (pref + null)``

dF/F responses can be negative, so OSI/DSI can fall outside [-1, 1]; values are
returned unclipped.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

DIRECTIONS = np.arange(8) * 45.0
TEMPORAL_FREQUENCIES = np.array([1.0, 2.0, 4.0, 8.0, 15.0])

N_DIRECTIONS = DIRECTIONS.size
N_TFS = TEMPORAL_FREQUENCIES.size


def _condition_masks(rs):
    """41 boolean trial masks: 40 direction x TF conditions, then blanks."""
    orientation = np.asarray(rs["trial_orientation"], dtype=np.float64)
    temporal_frequency = np.asarray(rs["trial_temporal_frequency"], dtype=np.float64)
    blank_sweep = np.asarray(rs["trial_blank_sweep"], dtype=bool)
    n_trials = orientation.size
    if blank_sweep.shape != (n_trials,):
        raise ValueError("trial_blank_sweep does not match the trial count")

    dir_idx = np.full(n_trials, -1, dtype=int)
    known = np.isfinite(orientation)
    dir_idx[known] = np.rint(orientation[known] / 45.0).astype(int) % N_DIRECTIONS

    tf_idx = np.full(n_trials, -1, dtype=int)
    for index, value in enumerate(TEMPORAL_FREQUENCIES):
        tf_idx[np.isclose(temporal_frequency, value)] = index

    masks = [
        (dir_idx == d) & (tf_idx == t) & ~blank_sweep
        for d in range(N_DIRECTIONS)
        for t in range(N_TFS)
    ]
    masks.append(blank_sweep)
    return masks


def condition_means(rs):
    """Return ``(M, directions, temporal_frequencies, blank_mean)``.

    ``M`` has shape (cells, 8 directions, 5 TFs); blanks are excluded from it and
    summarised separately in ``blank_mean`` (cells,).
    """
    responses = np.asarray(rs["responses"], dtype=np.float64)
    masks = _condition_masks(rs)
    n_cells = responses.shape[0]

    means = np.empty((n_cells, N_DIRECTIONS, N_TFS))
    for d in range(N_DIRECTIONS):
        for t in range(N_TFS):
            mask = masks[d * N_TFS + t]
            if not mask.any():
                raise ValueError(
                    f"no trials for direction index {d} and temporal frequency index {t}"
                )
            means[:, d, t] = responses[:, mask].mean(axis=1)

    blank_mask = masks[-1]
    if blank_mask.any():
        blank_mean = responses[:, blank_mask].mean(axis=1)
    else:
        blank_mean = np.full(n_cells, np.nan)

    return means, DIRECTIONS.copy(), TEMPORAL_FREQUENCIES.copy(), blank_mean


def osi_dsi(means):
    """Per-cell preferred condition, OSI and DSI (unclipped)."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 3 or means.shape[1] != N_DIRECTIONS:
        raise ValueError("means must have shape (cells, 8, n_tfs)")

    n_cells = means.shape[0]
    flat = means.reshape(n_cells, -1)
    pref_flat = np.argmax(flat, axis=1)
    pref_dir_idx = pref_flat // means.shape[2]
    pref_tf_idx = pref_flat % means.shape[2]
    rows = np.arange(n_cells)

    pref = means[rows, pref_dir_idx, pref_tf_idx]
    orth = 0.5 * (
        means[rows, (pref_dir_idx + 2) % N_DIRECTIONS, pref_tf_idx]
        + means[rows, (pref_dir_idx - 2) % N_DIRECTIONS, pref_tf_idx]
    )
    null = means[rows, (pref_dir_idx + 4) % N_DIRECTIONS, pref_tf_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        osi = (pref - orth) / (pref + orth)
        dsi = (pref - null) / (pref + null)

    return {
        "pref_dir_idx": pref_dir_idx.astype(np.int64),
        "pref_tf_idx": pref_tf_idx.astype(np.int64),
        "pref_dir_deg": DIRECTIONS[pref_dir_idx].astype(np.float64),
        "pref_tf_hz": TEMPORAL_FREQUENCIES[pref_tf_idx].astype(np.float64),
        "osi": np.asarray(osi, dtype=np.float64),
        "dsi": np.asarray(dsi, dtype=np.float64),
    }


def responsive(rs, alpha=0.01):
    """Boolean per cell: one-way ANOVA across the 40 conditions plus blank.

    The F statistic is computed directly rather than through ``scipy.stats.f_oneway``
    because the blank group has a different trial count and this scipy version's
    ``f_oneway`` requires equal-length samples. The result is identical for equal
    groups.
    """
    responses = np.asarray(rs["responses"], dtype=np.float64)
    masks = _condition_masks(rs)
    groups = [responses[:, mask] for mask in masks if mask.any()]
    if len(groups) < 2:
        return np.zeros(responses.shape[0], dtype=bool)

    n_groups = len(groups)
    n_trials = sum(group.shape[1] for group in groups)
    grand_mean = np.concatenate(groups, axis=1).mean(axis=1)

    ss_between = sum(
        group.shape[1] * (group.mean(axis=1) - grand_mean) ** 2 for group in groups
    )
    ss_within = sum(
        ((group - group.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        for group in groups
    )

    df_between = n_groups - 1
    df_within = n_trials - n_groups
    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = (ss_between / df_between) / (ss_within / df_within)
        p_value = stats.f.sf(f_statistic, df_between, df_within)
    return np.asarray(p_value < alpha, dtype=bool)
