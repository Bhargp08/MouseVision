from pathlib import Path

import numpy as np
import pandas as pd

from mousevision import app_data

FIXTURES = Path(__file__).parent / "fixtures" / "results"

KEYS = [
    "tuning_VISp",
    "rf_VISp",
    "decoding_curve",
    "decoding_summary",
    "rsa",
    "encoding",
    "hierarchy",
]


def test_fixtures_load_all_keys():
    results = app_data.load_results(FIXTURES)

    for key in KEYS:
        assert results[key] is not None, key


def test_fixture_shapes_match_the_documented_schema():
    results = app_data.load_results(FIXTURES)

    tuning = results["tuning_VISp"]
    assert set(tuning) == {
        "cell_ids",
        "M",
        "osi",
        "dsi",
        "pref_dir",
        "pref_tf",
        "responsive",
    }
    assert tuning["M"].ndim == 3
    assert tuning["M"].shape[1:] == (8, 5)
    assert tuning["M"].shape[0] == tuning["cell_ids"].size

    rf = results["rf_VISp"]
    assert set(rf) == {"cell_ids", "z_on", "z_off", "sig_on", "sig_off"}
    assert rf["z_on"].shape[1:] == (16, 28)
    assert rf["z_on"].dtype == np.float16

    assert list(results["decoding_curve"].columns) == [
        "area",
        "n_cells",
        "acc_mean",
        "acc_lo",
        "acc_hi",
        "chance",
    ]
    assert list(results["decoding_summary"].columns) == [
        "area",
        "n_cells",
        "acc_all",
        "acc_at_common_n",
        "perm_p",
        "participation_ratio",
    ]
    assert list(results["rsa"].columns) == [
        "area",
        "layer",
        "rho",
        "ceiling",
        "rho_over_ceiling",
        "p",
    ]
    assert list(results["encoding"].columns) == [
        "area",
        "layer",
        "r2",
        "ceiling",
        "r2_over_ceiling",
    ]

    assert set(results["hierarchy"]) == {"encoding", "rsa"}
    assert set(results["hierarchy"]["encoding"]["centroids"]) <= set(app_data.AREAS)


def test_empty_dir_gives_none_for_every_key(tmp_path):
    results = app_data.load_results(tmp_path)

    for key in KEYS:
        assert results[key] is None, key


def test_missing_dir_does_not_raise(tmp_path):
    results = app_data.load_results(tmp_path / "does-not-exist")

    assert all(results[key] is None for key in KEYS)


def test_results_dir_defaults_to_env_then_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("MV_RESULTS_DIR", str(tmp_path))
    assert app_data.results_dir() == tmp_path

    monkeypatch.delenv("MV_RESULTS_DIR")
    assert app_data.results_dir() == app_data.REPO_ROOT / "results"


def test_first_npz_prefers_the_requested_area():
    results = {
        "tuning_VISl": {"cell_ids": np.array([1])},
        "tuning_VISp": {"cell_ids": np.array([2])},
    }

    area, payload = app_data.first_npz(results, "tuning_", area="VISl")
    assert area == "VISl"
    assert payload["cell_ids"].tolist() == [1]

    area, payload = app_data.first_npz(results, "tuning_")
    assert area == "VISp"  # AREAS order, not dict order

    assert app_data.first_npz(results, "rf_") == (None, None)


def test_csv_backed_files_are_dataframes():
    results = app_data.load_results(FIXTURES)

    assert isinstance(results["rsa"], pd.DataFrame)
    assert not results["rsa"].empty
