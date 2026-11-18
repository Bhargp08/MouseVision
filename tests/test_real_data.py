"""Slow tests against the local Allen cache; skipped when the NWB files are absent."""

import numpy as np
import pytest

from mousevision.allen import make_cache
from mousevision.fetch import is_valid_nwb
from mousevision.responses import build_response_set, load_response_set, response_path

pytestmark = pytest.mark.slow

NS_VISP = 510345479
DG_VISP = 510514474
LSN_VISP = 510174759


def cached_experiment_data(experiment_id):
    cache = make_cache()
    path = cache.get_cache_path(None, cache.EXPERIMENT_DATA_KEY, experiment_id)
    if not is_valid_nwb(path):
        pytest.skip(f"cached NWB for experiment {experiment_id} is not downloaded")
    return cache, path


def load_or_build(experiment_id, stimulus):
    path = response_path(experiment_id, stimulus)
    if path.is_file():
        return load_response_set(path)
    cache, _ = cached_experiment_data(experiment_id)
    dataset = cache.get_ophys_experiment_data(experiment_id)
    return build_response_set(dataset, stimulus, experiment_id)


def test_ns_visp_shapes():
    rs = load_or_build(NS_VISP, "natural_scenes")

    assert rs["responses"].shape == (331, 5950)
    frames = rs["trial_frame"]
    assert int(np.sum(frames == -1)) == 50
    images, counts = np.unique(frames[frames >= 0], return_counts=True)
    assert len(images) == 118
    assert set(counts.tolist()) == {50}
    assert 29.5 <= float(rs["frame_rate_hz"]) <= 30.5


def test_dg_visp_shapes():
    rs = load_or_build(DG_VISP, "drifting_gratings")

    assert rs["responses"].shape == (226, 628)
    assert int(np.sum(rs["trial_blank_sweep"])) == 30
    assert 29.5 <= float(rs["frame_rate_hz"]) <= 30.5


def test_lsn_visp_shapes():
    rs = load_or_build(LSN_VISP, "locally_sparse_noise")

    assert rs["responses"].shape == (202, 8880)
    template = rs["lsn_template"]
    assert template.shape == (9000, 16, 28)
    assert set(np.unique(template).tolist()) == {0, 127, 255}
