"""Slow tuning/RF checks against the local VISp response sets."""

import numpy as np
import pytest

from mousevision import rf, tuning
from mousevision.responses import load_response_set, response_path

pytestmark = pytest.mark.slow

DG_VISP = 510514474
LSN_VISP = 510174759


def load_or_skip(experiment_id, stimulus):
    path = response_path(experiment_id, stimulus)
    if not path.is_file():
        pytest.skip(f"{path} is not built; run scripts/build_responses.py")
    return load_response_set(path)


def test_most_visp_dg_cells_are_responsive():
    dg = load_or_skip(DG_VISP, "drifting_gratings")

    means, _, _, _ = tuning.condition_means(dg)
    flags = tuning.responsive(dg)

    assert means.shape[1:] == (8, 5)
    assert flags.mean() >= 0.5


def test_visp_lsn_rf_fraction_is_plausible():
    lsn = load_or_skip(LSN_VISP, "locally_sparse_noise")

    significance = rf.rf_significance(
        lsn, n_shuffles=100, rng=np.random.default_rng(0)
    )

    either = significance["sig_on"] | significance["sig_off"]
    assert 0.05 <= either.mean() <= 0.9
    assert np.isfinite(significance["z_on"]).any()
