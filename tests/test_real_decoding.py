"""Slow decoding check against the cached VISp natural-scenes responses."""

import numpy as np
import pytest

from mousevision.config import load_subset
from mousevision.decoding import CHANCE, decode_cv, ns_xy
from mousevision.responses import load_response_set, response_path

pytestmark = pytest.mark.slow

VISP_NS = 510345479


def test_visp_accuracy_beats_chance():
    path = response_path(VISP_NS, "natural_scenes")
    if not path.is_file():
        pytest.skip(f"{path} is not built; run scripts/build_responses.py")

    X, y = ns_xy(load_response_set(path))
    accuracy = decode_cv(X, y)

    assert X.shape[1] >= 300
    assert y.size >= 5000
    assert accuracy > 5 * CHANCE
    assert np.isfinite(accuracy)


def test_subset_has_six_natural_scenes_sessions():
    ids = load_subset()["core_experiments"]["natural_scenes"]

    assert len(ids) == 6
