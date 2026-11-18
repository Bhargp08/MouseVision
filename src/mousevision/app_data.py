"""Loading of the derived result files written by MV-02 .. MV-05.

``load_results`` is deliberately forgiving: a missing (or unreadable) file yields
``None`` rather than an exception, so the Streamlit app and the figure script can
show "not built yet" instead of crashing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mousevision.config import REPO_ROOT

# Ordered exactly as in configs/subset.toml; plots rely on this order.
AREAS = ["VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl"]
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4"]

RESULTS_DIR_ENV_VAR = "MV_RESULTS_DIR"

CSV_FILES = {
    "decoding_curve": "decoding_curve.csv",
    "decoding_summary": "decoding_summary.csv",
    "rsa": "rsa.csv",
    "encoding": "encoding.csv",
}

JSON_FILES = {"hierarchy": "hierarchy.json"}

# Per-area .npz files: tuning_<area>.npz and rf_<area>.npz, keyed by file stem.
NPZ_PREFIXES = ("tuning_", "rf_")


def results_dir(override: str | Path | None = None) -> Path:
    """Results directory: explicit value, else $MV_RESULTS_DIR, else <repo>/results."""
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(RESULTS_DIR_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return REPO_ROOT / "results"


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # TODO: narrow once the real schema from MV-02..05 lands
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # TODO: narrow once the real schema from MV-02..05 lands
        return None


def _load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception:  # TODO: narrow once the real schema from MV-02..05 lands
        return None


def load_results(results_dir_path: str | Path | None = None) -> dict[str, Any]:
    """Load every known result file.

    Returns a dict with one entry per CSV/JSON file (DataFrame / dict) plus one
    entry per ``tuning_<area>.npz`` / ``rf_<area>.npz`` file (dict of arrays),
    keyed by file stem. Missing files map to ``None``.
    """
    root = results_dir(results_dir_path)

    results: dict[str, Any] = {key: _load_csv(root / name) for key, name in CSV_FILES.items()}
    for key, name in JSON_FILES.items():
        results[key] = _load_json(root / name)

    for prefix in NPZ_PREFIXES:
        # Keep the documented VISp keys stable even when the file is absent.
        results[f"{prefix}{AREAS[0]}"] = None
        for path in sorted(root.glob(f"{prefix}*.npz")):
            results[path.stem] = _load_npz(path)

    return results


def first_npz(
    results: dict[str, Any], prefix: str, area: str | None = None
) -> tuple[str | None, dict[str, np.ndarray] | None]:
    """Pick a ``<prefix><area>.npz`` payload, preferring ``area`` then AREAS order.

    Returns ``(area, arrays)``; ``(None, None)`` when nothing was loaded.
    """
    candidates: list[str] = []
    if area is not None:
        candidates.append(f"{prefix}{area}")
    candidates.extend(f"{prefix}{a}" for a in AREAS)
    candidates.extend(key for key in results if key.startswith(prefix))

    for key in candidates:
        payload = results.get(key)
        if payload is not None:
            return key[len(prefix) :], payload
    return None, None
