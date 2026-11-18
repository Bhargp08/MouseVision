"""Per-trial response matrices from Allen Brain Observatory stimulus presentations.

Convention
----------
``response[c, k] = mean(dff[c, start_k + LAG : end_k + LAG]) - mean(dff[c, start_k - BASELINE : start_k])``

* ``dff`` is the NWB's stored dF/F (``dataset.get_dff_traces()``).
* Windows are half-open; frame indices come from the stimulus table's ``start``/``end``
  columns (2-photon frames, ~30 Hz).
* ``LAG_FRAMES = 4`` (~133 ms at 30 Hz) accounts for the GCaMP6f rise/decay lag.
* ``BASELINE_FRAMES = 30`` is a 1 s pre-stimulus baseline; pass ``baseline=0`` to
  disable baseline subtraction.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mousevision import __version__
from mousevision.config import data_dir

LAG_FRAMES = 4
BASELINE_FRAMES = 30

CONVENTION = (
    "mean(dff[start+lag:end+lag]) - mean(dff[start-baseline:start]); "
    "half-open windows, dff = stored dF/F"
)

STIMULUS_ALIASES = {
    "natural_scenes": "natural_scenes",
    "drifting_gratings": "drifting_gratings",
    "locally_sparse_noise": "locally_sparse_noise",
    "locally_sparse_noise_4deg": "locally_sparse_noise",
    "locally_sparse_noise_8deg": "locally_sparse_noise",
}


def canonical_stimulus_name(stimulus: str) -> str:
    """Map a stimulus name (including the LSN degree variants) to its canonical form."""
    try:
        return STIMULUS_ALIASES[stimulus]
    except KeyError:
        raise ValueError(f"unknown stimulus {stimulus!r}") from None


def _column(table: Any, name: str) -> np.ndarray:
    if isinstance(table, dict):
        if name not in table:
            raise ValueError(f"stimulus table is missing the {name!r} column")
        return np.asarray(table[name])
    if hasattr(table, "columns"):
        if name not in table.columns:
            raise ValueError(f"stimulus table is missing the {name!r} column")
        return table[name].to_numpy()
    raise TypeError("stimulus table must be a mapping or a pandas DataFrame")


def trial_responses(
    dff: np.ndarray,
    table: Any,
    lag: int = LAG_FRAMES,
    baseline: int = BASELINE_FRAMES,
) -> np.ndarray:
    """Cells x trials mean responses, following the module-level convention."""
    dff = np.asarray(dff, dtype=np.float64)
    if dff.ndim != 2:
        raise ValueError("dff must have shape (cells, frames)")
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if baseline < 0:
        raise ValueError("baseline must be >= 0")

    starts = _column(table, "start").astype(np.int64)
    ends = _column(table, "end").astype(np.int64)
    if starts.shape != ends.shape:
        raise ValueError("start and end columns must have the same length")

    n_frames = dff.shape[1]
    if np.any(ends <= starts):
        raise ValueError("every trial needs a non-empty response window")
    if np.any(starts - baseline < 0):
        raise ValueError("the baseline window reaches before the first frame")
    if np.any(ends + lag > n_frames):
        raise ValueError("a response window reaches past the last frame")

    csum = np.concatenate(
        [np.zeros((dff.shape[0], 1)), np.cumsum(dff, axis=1)], axis=1
    )

    def window_mean(low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return (csum[:, high] - csum[:, low]) / (high - low)

    response = window_mean(starts + lag, ends + lag)
    if baseline > 0:
        response = response - window_mean(starts - baseline, starts)
    return response.astype(np.float32)


def trial_table(stimulus: str, table: Any) -> dict[str, np.ndarray]:
    """Trial windows plus the stimulus-specific condition columns."""
    canonical = canonical_stimulus_name(stimulus)
    trial_start = _column(table, "start").astype(np.int64)
    trial_end = _column(table, "end").astype(np.int64)
    out: dict[str, np.ndarray] = {
        "trial_start": trial_start,
        "trial_end": trial_end,
    }

    if canonical == "natural_scenes":
        out["trial_frame"] = _column(table, "frame").astype(np.int64)
    elif canonical == "drifting_gratings":
        orientation = _column(table, "orientation").astype(np.float64)
        temporal_frequency = _column(table, "temporal_frequency").astype(np.float64)
        blank_sweep = _column(table, "blank_sweep").astype(bool)
        nan_rows = np.isnan(orientation) | np.isnan(temporal_frequency)
        if not np.array_equal(blank_sweep, nan_rows):
            raise ValueError(
                "blank_sweep disagrees with the NaN orientation/temporal_frequency rows"
            )
        out["trial_orientation"] = orientation
        out["trial_temporal_frequency"] = temporal_frequency
        out["trial_blank_sweep"] = blank_sweep
    else:  # locally_sparse_noise
        out["trial_frame"] = _column(table, "frame").astype(np.int64)

    return out


def _imaging_depth_um(metadata: dict[str, Any]) -> float:
    depth = metadata.get("imaging_depth")
    if depth is None:
        return float("nan")
    units = str(metadata.get("imaging_depth_units", "um")).lower()
    depth = float(depth)
    if units in {"um", "micron", "microns", "µm"}:
        return depth
    if units == "mm":
        return depth * 1000.0
    return float("nan")


def build_response_set(
    dataset: Any,
    stimulus: str,
    experiment_id: int,
    lag: int = LAG_FRAMES,
    baseline: int = BASELINE_FRAMES,
) -> dict[str, Any]:
    """Assemble a response set from a duck-typed allensdk dataset object."""
    canonical = canonical_stimulus_name(stimulus)
    timestamps, dff = dataset.get_dff_traces()
    dff = np.asarray(dff)
    table = dataset.get_stimulus_table(stimulus)
    trials = trial_table(stimulus, table)
    responses = trial_responses(dff, table, lag=lag, baseline=baseline)

    cell_specimen_ids = np.asarray(dataset.get_cell_specimen_ids(), dtype=np.int64)
    if cell_specimen_ids.shape[0] != dff.shape[0]:
        raise ValueError(
            "cell id count does not match the number of dF/F rows "
            f"({cell_specimen_ids.shape[0]} != {dff.shape[0]})"
        )

    timestamps = np.asarray(timestamps, dtype=np.float64)
    frame_rate_hz = float("nan")
    if timestamps.size >= 2:
        dt = float(np.median(np.diff(timestamps)))
        if dt > 0:
            frame_rate_hz = 1.0 / dt

    metadata = dataset.get_metadata() or {}
    response_set: dict[str, Any] = {
        "responses": responses,
        "cell_specimen_ids": cell_specimen_ids,
        **trials,
        "experiment_id": np.int64(experiment_id),
        "stimulus": canonical,
        "stimulus_name": stimulus,
        "targeted_structure": str(metadata.get("targeted_structure", "")),
        "session_type": str(metadata.get("session_type", "")),
        "cre_line": str(metadata.get("cre_line", "")),
        "imaging_depth_um": np.float64(_imaging_depth_um(metadata)),
        "lag_frames": np.int64(lag),
        "baseline_frames": np.int64(baseline),
        "frame_rate_hz": np.float64(frame_rate_hz),
        "convention": CONVENTION,
        "mousevision_version": __version__,
    }

    if canonical == "locally_sparse_noise":
        template = np.asarray(dataset.get_stimulus_template(stimulus))
        if template.ndim != 3:
            raise ValueError("the locally sparse noise template must be 3-D")
        response_set["lsn_template"] = template.astype(np.uint8, copy=False)

    return response_set


def response_path(experiment_id: int, stimulus: str) -> Path:
    """Where a response set is cached; does not create anything."""
    canonical = canonical_stimulus_name(stimulus)
    return (
        data_dir()
        / "interim"
        / "responses"
        / f"{int(experiment_id)}_{canonical}.npz"
    )


def save_response_set(response_set: dict[str, Any], path: str | Path) -> Path:
    """Atomically write a response set as a compressed npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(
            temp_path, **{key: np.asarray(value) for key, value in response_set.items()}
        )
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def load_response_set(path: str | Path) -> dict[str, Any]:
    """Load a response set, unwrapping scalars stored as 0-d arrays."""
    with np.load(Path(path), allow_pickle=False) as data:
        out: dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            out[key] = value.item() if value.shape == () else value
        return out
