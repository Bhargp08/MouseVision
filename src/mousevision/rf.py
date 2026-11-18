"""Receptive fields from locally sparse noise, with shuffle-based significance.

The stimulus is ternary: 255 = ON, 0 = OFF, 127 = gray. The map is a spike-triggered
average of z-scored trial responses over trials in which a pixel was ON (or OFF):

``STA[pix] = mean(z_response over trials with template[pix] == 255)``

Significance uses the shuffled max-pixel z, which controls the family-wise error rate
over all 16 x 28 pixels for each cell: the permutation p-value of the observed
max-pixel z against the shuffled max-pixel z distribution (the discrete form of
comparing against its (1 - alpha) quantile).
"""

from __future__ import annotations

import numpy as np

TEMPLATE_SHAPE = (16, 28)
ON = 255
OFF = 0


def _z_responses(rs):
    responses = np.asarray(rs["responses"], dtype=np.float64)
    mean = responses.mean(axis=1, keepdims=True)
    std = responses.std(axis=1, keepdims=True)
    safe = np.where(std == 0, 1.0, std)
    return (responses - mean) / safe


def _trial_pixels(rs):
    template = np.asarray(rs["lsn_template"])
    if template.ndim != 3:
        raise ValueError("lsn_template must have shape (images, rows, cols)")
    frames = np.asarray(rs["trial_frame"], dtype=np.int64)
    if frames.size != np.asarray(rs["responses"]).shape[1]:
        raise ValueError("trial_frame does not match the response trial count")
    flat = template.reshape(template.shape[0], -1)
    if frames.min() < 0:
        raise ValueError("locally sparse noise trials must all have a frame index")
    return flat[frames]


def _binary(pixels, value):
    return (pixels == value).astype(np.float64)


def _weighted_mean(z, binary):
    numerator = z @ binary
    counts = binary.sum(axis=0)
    return numerator / np.where(counts == 0, np.nan, counts)


def _maps(values, shape):
    return values.reshape(values.shape[0], *shape)


def sta_maps(rs):
    """ON and OFF spike-triggered average maps, each (cells, 16, 28)."""
    z = _z_responses(rs)
    pixels = _trial_pixels(rs)
    shape = np.asarray(rs["lsn_template"]).shape[1:]
    if tuple(shape) != TEMPLATE_SHAPE:
        raise ValueError(f"expected a {TEMPLATE_SHAPE} template, got {tuple(shape)}")
    return {
        "on": _maps(_weighted_mean(z, _binary(pixels, ON)), shape),
        "off": _maps(_weighted_mean(z, _binary(pixels, OFF)), shape),
    }


def rf_significance(rs, n_shuffles=200, alpha=0.05, rng=None):
    """z-scored maps and per-cell significance against shuffled trial labels."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_shuffles = int(n_shuffles)
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")

    z = _z_responses(rs)
    n_cells = z.shape[0]
    pixels = _trial_pixels(rs)
    shape = np.asarray(rs["lsn_template"]).shape[1:]
    on_binary = _binary(pixels, ON)
    off_binary = _binary(pixels, OFF)

    observed_on = _weighted_mean(z, on_binary)
    observed_off = _weighted_mean(z, off_binary)

    null_on = np.empty((n_shuffles, n_cells, on_binary.shape[1]))
    null_off = np.empty((n_shuffles, n_cells, off_binary.shape[1]))
    for index in range(n_shuffles):
        permutation = rng.permutation(z.shape[1])
        shuffled = z[:, permutation]
        null_on[index] = _weighted_mean(shuffled, on_binary)
        null_off[index] = _weighted_mean(shuffled, off_binary)

    def standardise(observed, null, counts):
        valid = np.isfinite(null)
        n_valid = valid.sum(axis=0)
        mean = np.where(valid, null, 0.0).sum(axis=0) / np.maximum(n_valid, 1)
        centered = np.where(valid, null - mean, 0.0)
        std = np.sqrt((centered**2).sum(axis=0) / np.maximum(n_valid, 1))
        # Pixels with few trials give an unstable shuffled std; floor it at the
        # standard error of a mean z-response (z has unit variance by construction).
        floor = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        safe = np.where(n_valid > 0, np.maximum(std, floor), np.nan)
        return (observed - mean) / safe, (null - mean) / safe

    z_on, null_on_z = standardise(observed_on, null_on, on_binary.sum(axis=0))
    z_off, null_off_z = standardise(observed_off, null_off, off_binary.sum(axis=0))

    # Max-pixel z of each shuffled map: (n_shuffles, cells), a null max distribution.
    null_max = np.nanmax(np.stack([null_on_z, null_off_z], axis=0), axis=3).max(axis=0)

    def significant(z_map):
        observed_max = np.nanmax(z_map, axis=1)
        exceeded = (null_max >= observed_max[None, :]).sum(axis=0)
        p_value = (1.0 + exceeded) / (1.0 + n_shuffles)
        return np.asarray(p_value < alpha, dtype=bool)

    return {
        "z_on": _maps(z_on, shape),
        "z_off": _maps(z_off, shape),
        "sig_on": significant(z_on),
        "sig_off": significant(z_off),
    }
