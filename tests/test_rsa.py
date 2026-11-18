import numpy as np
import pytest

from mousevision import rsa


def synthetic_repeats(n_images=20, n_cells=10, n_reps=6, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, size=(n_images, n_cells))
    X = np.vstack(
        [means[index] + rng.normal(0.0, noise, size=(n_reps, n_cells)) for index in range(n_images)]
    )
    y = np.repeat(np.arange(n_images), n_reps)
    return X, y


def test_rdm_is_symmetric_with_zero_diagonal():
    patterns = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    matrix = rsa.rdm(patterns)

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_compare_rdms_uses_upper_triangle_only():
    rng = np.random.default_rng(0)
    a = rsa.rdm(rng.normal(size=(8, 5)))
    b = rsa.rdm(rng.normal(size=(8, 5)))

    assert rsa.compare_rdms(a, a) == pytest.approx(1.0)

    changed = b.copy()
    changed[np.tril_indices(b.shape[0], k=-1)] = 0.0
    assert rsa.compare_rdms(changed, a) == pytest.approx(rsa.compare_rdms(b, a))


def test_noise_ceiling_extremes():
    noiseless_X, noiseless_y = synthetic_repeats(noise=1e-9)
    rho_half, ceiling = rsa.rdm_noise_ceiling(noiseless_X, noiseless_y, n_splits=5)
    assert rho_half > 0.99
    assert ceiling > 0.99

    rng = np.random.default_rng(1)
    noise_X = rng.normal(size=(20 * 6, 10))
    noise_y = np.repeat(np.arange(20), 6)
    _, noise_ceiling = rsa.rdm_noise_ceiling(noise_X, noise_y, n_splits=5, seed=0)
    assert abs(noise_ceiling) < 0.15


def test_spearman_brown_formula(monkeypatch):
    X, y = synthetic_repeats(n_images=6, n_cells=4, n_reps=4)
    monkeypatch.setattr(rsa, "compare_rdms", lambda a, b: 0.4)

    rho_half, ceiling = rsa.rdm_noise_ceiling(X, y, n_splits=3)

    assert rho_half == pytest.approx(0.4)
    assert ceiling == pytest.approx(2 * 0.4 / 1.4)


def test_rsa_permutation_extremes():
    rng = np.random.default_rng(0)
    brain = rsa.rdm(rng.normal(size=(12, 6)))

    rho, null, p_value = rsa.rsa_permutation_p(brain, brain, n_perm=19)

    assert rho == pytest.approx(1.0)
    assert null.shape == (19,)
    assert p_value == pytest.approx(1.0 / 20.0)

    unrelated = rsa.rdm(rng.normal(size=(12, 6)))
    _, _, unrelated_p = rsa.rsa_permutation_p(unrelated, brain, n_perm=99)
    assert unrelated_p > 0.05


def test_permuted_rdm_stays_valid():
    rng = np.random.default_rng(2)
    matrix = rsa.rdm(rng.normal(size=(10, 4)))
    permutation = rng.permutation(10)

    permuted = matrix[permutation][:, permutation]

    assert np.allclose(np.diag(permuted), 0.0)
    assert np.allclose(permuted, permuted.T)


def test_image_means_shape_and_validation():
    X, y = synthetic_repeats(n_images=8, n_cells=5)

    means = rsa.image_means(X, y, n_images=8)

    assert means.shape == (8, 5)

    with pytest.raises(ValueError):
        rsa.image_means(X, y, n_images=118)


def test_rsa_table_columns():
    X, y = synthetic_repeats(n_images=20, n_cells=8)
    response_set = {
        "responses": X.T,
        "trial_frame": y,
        "trial_orientation": np.full(y.size, np.nan),
        "trial_temporal_frequency": np.full(y.size, np.nan),
        "trial_blank_sweep": np.zeros(y.size, dtype=bool),
    }
    rng = np.random.default_rng(0)
    features = {
        f"{layer}_g1": rng.normal(size=(20, 12)) for layer in rsa.LAYER_ORDER
    }

    table = rsa.rsa_table({"VISp": response_set}, features, grid=1, n_perm=19)

    assert list(table.columns) == [
        "area",
        "layer",
        "rho",
        "ceiling",
        "rho_over_ceiling",
        "p",
    ]
    assert len(table) == 4
    assert table["area"].unique().tolist() == ["VISp"]
