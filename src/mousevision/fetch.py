"""Download Allen experiments without re-downloading complete files.

AllenSDK streams straight to the final path, so an interrupted download leaves a
partial NWB file behind. Those are detected and re-downloaded instead of skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py


def core_experiment_ids(subset: dict[str, Any]) -> list[int]:
    """The frozen core set: 6 natural-scene sessions, then VISp DG and LSN."""
    core = subset["core_experiments"]
    ids = [int(eid) for eid in core["natural_scenes"]]
    ids.append(int(core["drifting_gratings_VISp"]))
    ids.append(int(core["locally_sparse_noise_VISp"]))
    if len(ids) != len(set(ids)):
        raise ValueError("the core experiment list contains duplicates")
    return ids


def is_valid_nwb(path: str | Path) -> bool:
    """True only for an HDF5 file that opens read-only and has a ``processing`` group."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return "processing" in handle
    except OSError:
        return False


def download_experiments(
    ids: Iterable[int], cache: Any, dry_run: bool = False
) -> dict[str, list[int]]:
    """Download missing or corrupt experiment NWBs; report what happened.

    Returns a dict with three lists: ``downloaded`` (absent before), ``redownloaded``
    (present but invalid) and ``skipped`` (already valid).
    """
    result: dict[str, list[int]] = {"downloaded": [], "skipped": [], "redownloaded": []}

    for experiment_id in ids:
        experiment_id = int(experiment_id)
        path = Path(
            cache.get_cache_path(None, cache.EXPERIMENT_DATA_KEY, experiment_id)
        )
        if is_valid_nwb(path):
            result["skipped"].append(experiment_id)
            continue

        existed = path.exists()
        result["redownloaded" if existed else "downloaded"].append(experiment_id)
        if dry_run:
            continue

        if existed:
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.api.save_ophys_experiment_data(experiment_id, str(path))

    return result
