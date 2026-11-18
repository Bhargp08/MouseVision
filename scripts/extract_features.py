"""Cache ResNet-18 features for the natural-scene templates.

Needs the torch extra: ``uv sync --locked --group dev --extra cnn``. Bharg runs this
once on his Mac; the resulting npz (< 5 MB) is committed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mousevision.allen import make_cache
from mousevision.config import load_subset
from mousevision.features import FEATURES_PATH, extract_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=int, default=None)
    args = parser.parse_args(argv)

    experiment_id = args.experiment_id
    if experiment_id is None:
        experiment_id = load_subset()["core_experiments"]["natural_scenes"][0]

    cache = make_cache()
    dataset = cache.get_ophys_experiment_data(experiment_id)
    images = np.asarray(dataset.get_stimulus_template("natural_scenes"))
    print(f"template {images.shape} from experiment {experiment_id}")

    features = extract_features(images)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES_PATH, **features)

    size_mb = FEATURES_PATH.stat().st_size / 1e6
    print(f"wrote {FEATURES_PATH} ({size_mb:.2f} MB)")
    if size_mb >= 5.0:
        print("WARNING: the cache is 5 MB or larger; shrink the grids or drop a layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
