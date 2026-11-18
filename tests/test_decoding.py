import numpy as np
import pytest

from mousevision.decoding import (
    CHANCE,
    decode_cv,
    neuron_curve,
    ns_xy,
    pca_geometry,
    permutation_pvalue,
)


def separable_set(n_classes=10, n_cells=30, n_reps=20, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, size=(n_classes, n_cells))
    blocks = [means[label] + rng.normal(0.0, 0.1, size=(n_reps, n_cells)) for label in range(n_classes)]
    X = np.vstack(blocks)
    y = np.repeat(np.arange(n_classes), n_reps)
    return X, y


def noise_set(n_classes=10, n_trials=200, n_cells=30, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_trials, n_cells))
    y = rng.integers(0, n_classes, size=n_trials)
    return X, y


def test_ns_xy_drops_blanks_and_keeps_order():
    rs = {
        "responses": np.arange(6, dtype=np.float32).reshape(2, 3),
        "trial_frame": np.array([7, -1, 3]),
    }

    X, y = ns_xy(rs)

    assert X.shape == (2, 2)
    assert X.dtype == np.float64
    assert y.tolist() == [7, 3]
    assert X[:, 0].tolist() == [0.0, 2.0]
    assert X[:, 1].tolist() == [3.0, 5.0]


def test_decode_cv_separable_and_noise():
    X, y = separable_set()
    assert decode_cv(X, y) > 0.95

    noisy_X, noisy_y = noise_set()
    accuracy = decode_cv(noisy_X, noisy_y)
    assert abs(accuracy - 0.1) < 0.1


def test_permutation_pvalue_extremes():
    X, y = separable_set()

    observed, null, p_value = permutation_pvalue(X, y, n_perm=19)

    assert observed > 0.95
    assert p_value == pytest.approx(1.0 / 20.0)
    assert null.shape == (19,)
    assert np.all(null < observed)

    noisy_X, noisy_y = noise_set()
    _, _, noise_p = permutation_pvalue(noisy_X, noisy_y, n_perm=19)
    assert noise_p > 0.05


def test_neuron_curve_columns_and_monotone_means():
    X, y = separable_set()

    curve = neuron_curve(X, y, ns=(5, 10, 20, 50, 200), n_draws=3, seed=0)

    assert list(curve.columns) == ["n_neurons", "draw", "accuracy"]
    assert curve["n_neurons"].max() == X.shape[1]
    assert curve["n_neurons"].min() == 5
    assert (curve["n_neurons"] <= X.shape[1]).all()
    assert curve["draw"].nunique() == 3

    means = curve.groupby("n_neurons")["accuracy"].mean()
    assert means.loc[X.shape[1]] >= means.loc[5]


def test_neuron_curve_is_deterministic():
    X, y = separable_set()

    first = neuron_curve(X, y, ns=(5, 10), n_draws=2, seed=3)
    second = neuron_curve(X, y, ns=(5, 10), n_draws=2, seed=3)

    assert first.equals(second)


def test_pca_geometry_on_rank_two_means():
    rng = np.random.default_rng(0)
    class_means = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 3.0, 0.0]])
    X = np.vstack(
        [class_means[label] + rng.normal(0.0, 1e-6, size=(4, 3)) for label in (0, 1, 2)]
    )
    y = np.repeat([0, 1, 2], 4)

    geometry = pca_geometry(X, y)

    ratios = geometry["explained_variance_ratio"]
    assert ratios.sum() == pytest.approx(1.0)
    assert geometry["participation_ratio"] <= 2.0 + 1e-6


def test_chance_constant():
    assert CHANCE == pytest.approx(1 / 118)
