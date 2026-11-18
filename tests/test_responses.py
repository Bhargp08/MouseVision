import numpy as np
import pytest

from mousevision.responses import (
    LAG_FRAMES,
    build_response_set,
    load_response_set,
    save_response_set,
    trial_responses,
    trial_table,
)


def ramp_dff(n_cells=2, n_frames=200):
    return np.arange(n_frames, dtype=float)[None, :] + 100 * np.arange(n_cells)[:, None]


def test_lag_and_baseline_convention():
    dff = ramp_dff()
    table = {"start": np.array([40, 60]), "end": np.array([47, 67])}

    responses = trial_responses(dff, table, lag=4, baseline=30)

    assert responses.dtype == np.float32
    assert responses.shape == (2, 2)
    assert responses[0].tolist() == pytest.approx([22.5, 22.5])
    assert responses[1].tolist() == pytest.approx([22.5, 22.5])

    no_baseline = trial_responses(dff, table, lag=4, baseline=0)
    assert no_baseline[0].tolist() == pytest.approx([47.0, 67.0])


def test_windows_are_half_open():
    table = {"start": np.array([40]), "end": np.array([47])}

    included = np.zeros((1, 120))
    included[0, 44] = 1.0
    assert trial_responses(included, table, lag=4, baseline=0)[0, 0] == pytest.approx(1 / 7)

    excluded = np.zeros((1, 120))
    excluded[0, 51] = 1.0
    assert trial_responses(excluded, table, lag=4, baseline=0)[0, 0] == 0.0


def test_invalid_windows_raise():
    dff = ramp_dff(n_frames=100)

    with pytest.raises(ValueError):
        trial_responses(dff, {"start": np.array([10]), "end": np.array([20])}, lag=-1)
    with pytest.raises(ValueError):
        trial_responses(dff, {"start": np.array([90]), "end": np.array([99])}, lag=4)
    with pytest.raises(ValueError):
        trial_responses(dff, {"start": np.array([20]), "end": np.array([30])}, baseline=30)
    with pytest.raises(ValueError):
        trial_responses(dff, {"start": np.array([10]), "end": np.array([10])})


def test_trial_table_natural_scenes_keeps_blanks():
    table = {
        "start": np.array([1, 2, 3]),
        "end": np.array([4, 5, 6]),
        "frame": np.array([0.0, -1.0, 7.0]),
    }

    out = trial_table("natural_scenes", table)

    assert out["trial_frame"].tolist() == [0, -1, 7]
    assert out["trial_frame"].dtype == np.int64
    assert out["trial_start"].dtype == np.int64
    assert out["trial_end"].dtype == np.int64


def test_trial_table_drifting_gratings():
    table = {
        "start": np.array([1, 2, 3]),
        "end": np.array([4, 5, 6]),
        "orientation": np.array([90.0, np.nan, 45.0]),
        "temporal_frequency": np.array([2.0, np.nan, 4.0]),
        "blank_sweep": np.array([0.0, 1.0, 0.0]),
    }

    out = trial_table("drifting_gratings", table)

    assert out["trial_blank_sweep"].dtype == np.bool_
    assert out["trial_blank_sweep"].tolist() == [False, True, False]
    assert out["trial_orientation"].dtype == np.float64
    assert out["trial_orientation"][0] == pytest.approx(90.0)
    assert np.isnan(out["trial_orientation"][1])
    assert out["trial_temporal_frequency"][2] == pytest.approx(4.0)

    inconsistent = dict(table)
    inconsistent["blank_sweep"] = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        trial_table("drifting_gratings", inconsistent)


def test_trial_table_rejects_missing_columns_and_unknown_stimuli():
    with pytest.raises(ValueError):
        trial_table("natural_scenes", {"start": np.array([1]), "end": np.array([2])})
    with pytest.raises(ValueError):
        trial_table("spontaneous", {"start": np.array([1]), "end": np.array([2])})


def test_trial_table_accepts_locally_sparse_noise_variants():
    table = {"start": np.array([1]), "end": np.array([2]), "frame": np.array([3])}

    for name in ("locally_sparse_noise", "locally_sparse_noise_4deg", "locally_sparse_noise_8deg"):
        out = trial_table(name, table)
        assert out["trial_frame"].tolist() == [3]


class FakeDataset:
    """Duck-typed stand-in for allensdk's BrainObservatoryNwbDataSet."""

    def __init__(self, n_cells=5, n_frames=1000):
        self.timestamps = np.arange(n_frames) / 30.0
        self.dff = ramp_dff(n_cells=n_cells, n_frames=n_frames)
        self.cell_ids = np.arange(n_cells) + 1000
        self.tables = {
            "natural_scenes": {
                "start": np.array([100, 200, 300]),
                "end": np.array([110, 210, 310]),
                "frame": np.array([0.0, 1.0, -1.0]),
            },
            "locally_sparse_noise_4deg": {
                "start": np.array([400, 500]),
                "end": np.array([410, 510]),
                "frame": np.array([2.0, 3.0]),
            },
        }
        self.templates = {"locally_sparse_noise_4deg": np.zeros((10, 16, 28), dtype=np.uint8)}

    def get_dff_traces(self):
        return self.timestamps, self.dff

    def get_cell_specimen_ids(self):
        return self.cell_ids

    def get_stimulus_table(self, name):
        return self.tables[name]

    def list_stimuli(self):
        return list(self.tables)

    def get_stimulus_template(self, name):
        return self.templates[name]

    def get_metadata(self):
        return {
            "targeted_structure": "VISp",
            "session_type": "three_session_B",
            "cre_line": "Cux2-CreERT2",
            "imaging_depth": 175,
            "imaging_depth_units": "um",
        }


def test_build_response_set_keys_and_shapes():
    rs = build_response_set(FakeDataset(), "natural_scenes", 12345, lag=4, baseline=30)

    required = {
        "responses",
        "cell_specimen_ids",
        "trial_start",
        "trial_end",
        "trial_frame",
        "experiment_id",
        "stimulus",
        "stimulus_name",
        "targeted_structure",
        "session_type",
        "cre_line",
        "imaging_depth_um",
        "lag_frames",
        "baseline_frames",
        "frame_rate_hz",
        "convention",
        "mousevision_version",
    }
    assert required <= set(rs)
    assert rs["responses"].shape == (5, 3)
    assert rs["cell_specimen_ids"].shape == (5,)
    assert rs["trial_frame"].shape == (3,)
    assert "lsn_template" not in rs
    assert rs["stimulus"] == "natural_scenes"
    assert rs["mousevision_version"] == "0.0.1"
    assert 29.0 < float(rs["frame_rate_hz"]) < 31.0
    assert rs["imaging_depth_um"] == pytest.approx(175.0)


def test_build_response_set_lsn_template():
    rs = build_response_set(
        FakeDataset(), "locally_sparse_noise_4deg", 999, lag=4, baseline=30
    )

    assert rs["stimulus"] == "locally_sparse_noise"
    assert rs["stimulus_name"] == "locally_sparse_noise_4deg"
    assert rs["lsn_template"].shape == (10, 16, 28)
    assert rs["lsn_template"].dtype == np.uint8


def test_build_response_set_rejects_cell_count_mismatch():
    dataset = FakeDataset()
    dataset.cell_ids = dataset.cell_ids[:-1]

    with pytest.raises(ValueError):
        build_response_set(dataset, "natural_scenes", 1)


def test_save_and_load_roundtrip(tmp_path):
    rs = build_response_set(FakeDataset(), "natural_scenes", 12345)
    path = tmp_path / "nested" / "rs.npz"

    save_response_set(rs, path)
    assert path.is_file()

    back = load_response_set(path)

    np.testing.assert_array_equal(back["responses"], rs["responses"])
    np.testing.assert_array_equal(back["cell_specimen_ids"], rs["cell_specimen_ids"])
    assert back["experiment_id"] == 12345
    assert isinstance(back["experiment_id"], int)
    assert back["stimulus"] == "natural_scenes"
    assert back["lag_frames"] == LAG_FRAMES
