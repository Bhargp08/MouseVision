import numpy as np
import pytest

from mousevision.tuning import (
    TEMPORAL_FREQUENCIES,
    condition_means,
    osi_dsi,
    responsive,
)

DIRECTIONS = np.arange(8) * 45.0


def grid(n_reps=15):
    directions, temporal_frequencies = [], []
    for direction in DIRECTIONS:
        for tf in TEMPORAL_FREQUENCIES:
            directions.extend([direction] * n_reps)
            temporal_frequencies.extend([tf] * n_reps)
    return np.array(directions), np.array(temporal_frequencies)


def dg_response_set(cell_functions, n_reps=15, n_blanks=30):
    directions, temporal_frequencies = grid(n_reps)
    n_trials = directions.size
    responses = np.zeros((len(cell_functions), n_trials + n_blanks))
    for index, function in enumerate(cell_functions):
        responses[index, :n_trials] = [
            function(d, tf) for d, tf in zip(directions, temporal_frequencies)
        ]
        responses[index, n_trials:] = function(np.nan, np.nan)

    return {
        "responses": responses.astype(np.float32),
        "trial_orientation": np.concatenate([directions, np.full(n_blanks, np.nan)]),
        "trial_temporal_frequency": np.concatenate(
            [temporal_frequencies, np.full(n_blanks, np.nan)]
        ),
        "trial_blank_sweep": np.concatenate(
            [np.zeros(n_trials, dtype=bool), np.ones(n_blanks, dtype=bool)]
        ),
    }


def test_condition_means_shape_and_blanks_excluded():
    rs = dg_response_set([lambda d, tf: 1.0 if (d == 90.0 and tf == 2.0) else 0.0])
    rs["responses"][0, -30:] = 99.0

    means, directions, temporal_frequencies, blank_mean = condition_means(rs)

    assert means.shape == (1, 8, 5)
    assert directions.tolist() == DIRECTIONS.tolist()
    assert temporal_frequencies.tolist() == TEMPORAL_FREQUENCIES.tolist()
    assert means[0, 2, 1] == pytest.approx(1.0)
    assert np.count_nonzero(means) == 1
    assert blank_mean.shape == (1,)
    assert blank_mean[0] == pytest.approx(99.0)


def test_condition_means_missing_condition_raises():
    rs = dg_response_set([lambda d, tf: 1.0])
    keep = ~((rs["trial_orientation"] == 180.0) & (rs["trial_temporal_frequency"] == 8.0))
    rs["responses"] = rs["responses"][:, keep]
    rs["trial_orientation"] = rs["trial_orientation"][keep]
    rs["trial_temporal_frequency"] = rs["trial_temporal_frequency"][keep]
    rs["trial_blank_sweep"] = rs["trial_blank_sweep"][keep]

    with pytest.raises(ValueError):
        condition_means(rs)


def test_osi_dsi_textbook_cases():
    rs = dg_response_set(
        [
            lambda d, tf: 1.0 if (d == 90.0 and tf == 2.0) else 0.0,
            lambda d, tf: 1.0 if d in (90.0, 270.0) else 0.0,
            lambda d, tf: 0.5,
        ]
    )
    means, _, _, _ = condition_means(rs)

    out = osi_dsi(means)

    assert out["pref_dir_deg"][0] == pytest.approx(90.0)
    assert out["pref_tf_hz"][0] == pytest.approx(2.0)
    assert out["osi"][0] == pytest.approx(1.0)
    assert out["dsi"][0] == pytest.approx(1.0)

    assert out["osi"][1] == pytest.approx(1.0)
    assert out["dsi"][1] == pytest.approx(0.0)

    assert out["osi"][2] == pytest.approx(0.0)
    assert out["dsi"][2] == pytest.approx(0.0)

    for key in ("pref_dir_idx", "pref_tf_idx", "osi", "dsi"):
        assert out[key].shape == (3,)


def test_responsive_flags_tuned_cell_but_not_noise():
    rng = np.random.default_rng(0)
    directions, temporal_frequencies = grid(15)
    n_trials = directions.size

    responses = rng.normal(0.0, 1.0, size=(2, n_trials + 30))
    preferred = (directions == 90.0) & (temporal_frequencies == 2.0)
    responses[0, :n_trials][preferred] += 10.0

    rs = {
        "responses": responses,
        "trial_orientation": np.concatenate([directions, np.full(30, np.nan)]),
        "trial_temporal_frequency": np.concatenate(
            [temporal_frequencies, np.full(30, np.nan)]
        ),
        "trial_blank_sweep": np.concatenate(
            [np.zeros(n_trials, dtype=bool), np.ones(30, dtype=bool)]
        ),
    }

    flags = responsive(rs)

    assert flags.dtype == bool
    assert bool(flags[0]) is True
    assert bool(flags[1]) is False
