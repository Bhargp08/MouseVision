import importlib.util
import sys

import numpy as np
import pytest

from mousevision import features


def test_module_imports_without_torch():
    assert "torch" not in sys.modules
    assert features.LAYERS == ("layer1", "layer2", "layer3", "layer4")


def test_features_path_location():
    assert features.FEATURES_PATH.name == "resnet18_ns.npz"
    assert features.FEATURES_PATH.parent.name == "features"
    assert features.FEATURES_PATH.parent.parent.name == "data"


def test_load_features_promotes_float16(tmp_path):
    path = tmp_path / "tiny.npz"
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        layer1_g1=rng.normal(size=(5, 4)).astype(np.float16),
        image_ids=np.arange(5, dtype=np.int64),
        weights="IMAGENET1K_V1",
        torch_version="2.2.2",
    )

    loaded = features.load_features(path)

    assert loaded["layer1_g1"].dtype == np.float64
    assert loaded["layer1_g1"].shape == (5, 4)
    assert loaded["image_ids"].tolist() == [0, 1, 2, 3, 4]
    assert loaded["weights"] == "IMAGENET1K_V1"
    assert loaded["torch_version"] == "2.2.2"


def test_extract_features_needs_torch():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; the script covers real extraction")

    with pytest.raises(ImportError):
        features.extract_features(np.zeros((2, 918, 1174), dtype=np.uint8))
