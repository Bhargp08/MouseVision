"""Smoke tests: the package imports and the pinned scientific stack is importable."""

import numpy
import pandas


def test_package_imports():
    import mousevision

    assert mousevision.__version__ == "0.0.1"


def test_pinned_numpy_and_pandas():
    assert numpy.__version__ == "1.23.5"
    assert pandas.__version__ == "1.5.3"


def test_allensdk_imports():
    # Catches the pkg_resources / statsmodels / hdmf breakages that bite allensdk.
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache

    assert BrainObservatoryCache is not None
