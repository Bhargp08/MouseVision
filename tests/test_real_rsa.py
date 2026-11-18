"""Slow RSA checks: committed features plus the cached VISp natural scenes."""

import numpy as np
import pytest

from mousevision import rsa
from mousevision.features import FEATURES_PATH, load_features
from mousevision.responses import load_response_set, response_path

pytestmark = pytest.mark.slow

VISP_NS = 510345479


def test_committed_feature_cache_has_118_images_per_layer():
    if not FEATURES_PATH.is_file():
        pytest.skip(f"{FEATURES_PATH} is not committed; run scripts/extract_features.py")

    features = load_features()

    for layer in rsa.LAYER_ORDER:
        assert features[f"{layer}_g1"].shape[0] == 118
        assert features[f"{layer}_g1"].dtype == np.float64
    assert features["weights"] == "IMAGENET1K_V1"


def test_visp_rdm_ceiling_and_positive_rsa():
    path = response_path(VISP_NS, "natural_scenes")
    if not path.is_file() or not FEATURES_PATH.is_file():
        pytest.skip("needs both the response set and the feature cache")

    response_set = load_response_set(path)
    frames = np.asarray(response_set["trial_frame"], dtype=np.int64)
    keep = frames >= 0
    X = np.asarray(response_set["responses"], dtype=np.float64).T[keep]
    y = frames[keep]

    _, ceiling = rsa.rdm_noise_ceiling(X, y, n_splits=10)
    assert ceiling > 0.2

    brain_rdm = rsa.rdm(rsa.image_means(X, y))
    features = load_features()
    rhos = [
        rsa.compare_rdms(rsa.rdm(features[f"{layer}_g1"]), brain_rdm)
        for layer in rsa.LAYER_ORDER
    ]
    assert max(rhos) > 0
