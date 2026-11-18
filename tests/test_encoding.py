import numpy as np
import pytest

from mousevision.encoding import (
    AREA_HIERARCHY,
    cv_r2,
    hierarchy_test,
    layer_centroid,
    neuron_ceiling,
    ridge_fit_predict,
)


def linear_problem(n_images=118, n_features=8, n_neurons=5, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0.0, 1.0, size=(n_images, n_features))
    weights = rng.normal(0.0, 1.0, size=(n_features, n_neurons))
    Y = F @ weights + noise * rng.normal(size=(n_images, n_neurons))
    return F, Y, weights


def test_ridge_matches_direct_solve():
    F, Y, _ = linear_problem(n_images=60)
    alphas = [0.3, 10.0]

    predictions = ridge_fit_predict(F[:40], Y[:40], F[40:], alphas)

    x_train = F[:40] - F[:40].mean(axis=0)
    y_train = Y[:40] - Y[:40].mean(axis=0)
    x_test = F[40:] - F[:40].mean(axis=0)
    for prediction, alpha in zip(predictions, alphas):
        weights = np.linalg.solve(
            x_train.T @ x_train + alpha * np.eye(x_train.shape[1]), x_train.T @ y_train
        )
        expected = x_test @ weights + Y[:40].mean(axis=0)
        assert np.allclose(prediction, expected)


def test_ridge_limits():
    F, Y, weights = linear_problem(n_images=80, n_features=5, n_neurons=3)

    small_alpha = ridge_fit_predict(F[:60], Y[:60], F[60:], [1e-10])[0]
    assert np.allclose(small_alpha, F[60:] @ weights + 0.0, atol=1e-4)

    large_alpha = ridge_fit_predict(F[:60], Y[:60], F[60:], [1e12])[0]
    assert np.allclose(large_alpha, np.tile(Y[:60].mean(axis=0), (20, 1)), atol=1e-5)


def test_cv_r2_noiseless_and_noise():
    F, Y, _ = linear_problem(noise=0.0)
    assert np.median(cv_r2(F, Y)) > 0.9

    rng = np.random.default_rng(1)
    noise_targets = rng.normal(size=(118, 4))
    assert np.median(cv_r2(F, noise_targets)) <= 0.05


def test_cv_r2_no_leakage_from_shuffled_targets():
    F, Y, _ = linear_problem(noise=0.05)
    rng = np.random.default_rng(2)
    shuffled = Y[rng.permutation(Y.shape[0])]

    assert np.median(cv_r2(F, shuffled)) <= 0.05


def test_neuron_ceiling_extremes():
    # TODO(MV-05): the split-half estimate is noisy for few images; this checks the
    # direction of the effect rather than a tight bound.
    rng = np.random.default_rng(0)
    n_images, n_cells, n_reps = 60, 12, 6
    means = rng.normal(size=(n_images, n_cells))
    clean = np.vstack(
        [means[index] + rng.normal(0.0, 1e-9, (n_reps, n_cells)) for index in range(n_images)]
    )
    clean_y = np.repeat(np.arange(n_images), n_reps)

    assert np.all(neuron_ceiling(clean, clean_y) > 0.99)

    # Per-neuron split-half r is noisy at this sample size; require the typical
    # magnitude to be small rather than bounding every cell.
    noisy = rng.normal(size=(n_images * n_reps, n_cells))
    assert np.median(np.abs(neuron_ceiling(noisy, clean_y))) < 0.25


def test_layer_centroid():
    assert layer_centroid([0.0, 0.0, 1.0, 0.0]) == pytest.approx(3.0)
    assert layer_centroid({"layer1": 0.0, "layer2": 0.0, "layer3": 0.0, "layer4": 2.0}) == pytest.approx(4.0)
    assert layer_centroid([1.0, 0.0, 0.0, 1.0]) == pytest.approx(2.5)
    assert np.isnan(layer_centroid([-1.0, 0.0, -2.0, -3.0]))


def test_hierarchy_test_extremes():
    assert AREA_HIERARCHY == ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]

    ordered = dict(zip(AREA_HIERARCHY, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strict=True))
    rho, p_value = hierarchy_test(ordered)
    assert rho == pytest.approx(1.0)
    assert p_value == pytest.approx(1.0 / 720.0)

    reversed_order = dict(zip(AREA_HIERARCHY, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], strict=True))
    rho_rev, p_rev = hierarchy_test(reversed_order)
    assert rho_rev == pytest.approx(-1.0)
    assert p_rev == pytest.approx(1.0)
