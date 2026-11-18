"""Filesystem locations and frozen constants for MouseVision.

Nothing here creates directories on import: paths are computed, not materialised.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

# src/mousevision/config.py -> src/mousevision -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# The SDK default base URI is http://, which can fail; always use https.
ALLEN_BASE_URI = "https://api.brain-map.org"

SUBSET_PATH = REPO_ROOT / "configs" / "subset.toml"

DATA_DIR_ENV_VAR = "MV_DATA_DIR"


def data_dir() -> Path:
    """Root data directory: $MV_DATA_DIR when set, else <repo>/data."""
    override = os.environ.get(DATA_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return REPO_ROOT / "data"


def cache_dir() -> Path:
    """AllenSDK BrainObservatoryCache directory."""
    return data_dir() / "raw" / "allen_cache"


def manifest_path() -> Path:
    """Manifest file that indexes everything downloaded into the cache."""
    return cache_dir() / "manifest.json"


def load_subset(path: Path | None = None) -> dict[str, Any]:
    """Parse the frozen subset definition from configs/subset.toml."""
    subset_path = path if path is not None else SUBSET_PATH
    with subset_path.open("rb") as fh:
        return tomllib.load(fh)
