"""Allen Brain Observatory access.

allensdk is imported lazily so that ``import mousevision`` stays fast and cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mousevision.config import ALLEN_BASE_URI, manifest_path as default_manifest_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def make_cache(manifest: str | Path | None = None) -> "BrainObservatoryCache":
    """Build a BrainObservatoryCache pointed at the local manifest.

    Constructing the cache does not touch the network; it only creates the
    manifest file and its parent directory.
    """
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache

    path = Path(manifest) if manifest is not None else default_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return BrainObservatoryCache(manifest_file=str(path), base_uri=ALLEN_BASE_URI)
