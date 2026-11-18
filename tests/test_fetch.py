from pathlib import Path

import h5py
import pytest

from mousevision.config import load_subset
from mousevision.fetch import core_experiment_ids, download_experiments, is_valid_nwb


def write_valid_nwb(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh.create_group("processing")


def test_core_experiment_ids():
    ids = core_experiment_ids(load_subset())

    assert len(ids) == 8
    assert len(set(ids)) == 8
    assert ids[0] == 510345479


def test_is_valid_nwb(tmp_path):
    good = tmp_path / "good.nwb"
    write_valid_nwb(good)
    assert is_valid_nwb(good) is True

    junk = tmp_path / "junk.nwb"
    junk.write_bytes(b"this is not hdf5")
    assert is_valid_nwb(junk) is False

    assert is_valid_nwb(tmp_path / "missing.nwb") is False


class FakeApi:
    def __init__(self):
        self.calls = []

    def save_ophys_experiment_data(self, experiment_id, path):
        self.calls.append((experiment_id, path))
        write_valid_nwb(path)


class FakeCache:
    EXPERIMENT_DATA_KEY = "ophys_experiment_data"

    def __init__(self, root):
        self.root = Path(root)
        self.api = FakeApi()

    def get_cache_path(self, file_name, key, experiment_id):
        return str(self.root / key / f"{experiment_id}.nwb")


def test_download_experiments_skips_downloads_and_repairs(tmp_path):
    cache = FakeCache(tmp_path)
    key = cache.EXPERIMENT_DATA_KEY
    write_valid_nwb(cache.get_cache_path(None, key, 1))

    junk = Path(cache.get_cache_path(None, key, 2))
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"truncated")

    result = download_experiments([1, 2, 3], cache)

    assert result["skipped"] == [1]
    assert result["redownloaded"] == [2]
    assert result["downloaded"] == [3]
    assert [call[0] for call in cache.api.calls] == [2, 3]
    assert is_valid_nwb(cache.get_cache_path(None, key, 2))


def test_second_call_downloads_nothing(tmp_path):
    cache = FakeCache(tmp_path)
    download_experiments([1, 2, 3], cache)
    cache.api.calls.clear()

    again = download_experiments([1, 2, 3], cache)

    assert again["downloaded"] == []
    assert again["redownloaded"] == []
    assert again["skipped"] == [1, 2, 3]
    assert cache.api.calls == []


def test_dry_run_makes_no_calls(tmp_path):
    cache = FakeCache(tmp_path)

    result = download_experiments([7, 8], cache, dry_run=True)

    assert result["downloaded"] == [7, 8]
    assert cache.api.calls == []
    assert not Path(cache.get_cache_path(None, "ophys_experiment_data", 7)).exists()


def test_duplicate_ids_are_rejected():
    subset = {
        "core_experiments": {
            "natural_scenes": [1, 1],
            "drifting_gratings_VISp": 2,
            "locally_sparse_noise_VISp": 3,
        }
    }

    with pytest.raises(ValueError):
        core_experiment_ids(subset)
