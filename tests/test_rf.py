import numpy as np
import pytest

from mousevision.rf import rf_significance, sta_maps

N_PIXELS = 16 * 28
DRIVING_PIXEL = (5, 10)


def lsn_response_set(seed=0, n_frames=1500, n_trials=3000, n_cells=2):
    rng = np.random.default_rng(seed)

    order = rng.random((n_frames, N_PIXELS)).argsort(axis=1)
    flat = np.full((n_frames, N_PIXELS), 127, dtype=np.uint8)
    rows = np.arange(n_frames)[:, None]
    flat[rows, order[:, :6]] = 255
    flat[rows, order[:, 6:12]] = 0
    template = flat.reshape(n_frames, 16, 28)

    frames = rng.integers(0, n_frames, size=n_trials)
    responses = rng.normal(0.0, 1.0, size=(n_cells, n_trials))

    pixel = DRIVING_PIXEL[0] * 28 + DRIVING_PIXEL[1]
    on_mask = flat[frames, pixel] == 255
    responses[0] = np.where(on_mask, 4.0, 0.0) + rng.normal(0.0, 1.0, n_trials)

    return {
        "responses": responses.astype(np.float32),
        "lsn_template": template,
        "trial_frame": frames.astype(np.int64),
    }


def test_sta_maps_shape_and_peak():
    rs = lsn_response_set()

    maps = sta_maps(rs)

    assert maps["on"].shape == (2, 16, 28)
    assert maps["off"].shape == (2, 16, 28)
    peak = np.unravel_index(np.argmax(maps["on"][0]), (16, 28))
    assert peak == DRIVING_PIXEL


def test_significance_detects_the_driving_pixel():
    rs = lsn_response_set()
    rng = np.random.default_rng(0)

    out = rf_significance(rs, n_shuffles=100, alpha=0.05, rng=rng)

    assert out["z_on"].shape == (2, 16, 28)
    assert out["z_off"].shape == (2, 16, 28)
    assert out["sig_on"].dtype == bool
    assert out["sig_on"].shape == (2,)
    assert bool(out["sig_on"][0]) is True


def test_pure_noise_cell_is_rarely_significant():
    significant = []
    for seed in range(20):
        rs = lsn_response_set(seed=seed, n_frames=1200, n_trials=3000)
        out = rf_significance(rs, n_shuffles=100, alpha=0.05, rng=np.random.default_rng(seed))
        significant.append(bool(out["sig_on"][1]))

    assert sum(significant) <= 2


def test_rejects_bad_template_shape():
    rs = lsn_response_set()
    rs["lsn_template"] = rs["lsn_template"][:, :8, :]

    with pytest.raises(ValueError):
        sta_maps(rs)


def test_n_shuffles_must_be_positive():
    rs = lsn_response_set(n_frames=100, n_trials=200)

    with pytest.raises(ValueError):
        rf_significance(rs, n_shuffles=0)
