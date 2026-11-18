"""Download the frozen Allen subset into the local BrainObservatoryCache.

The core set is 8 experiments (~3.2 GB). Re-running is a no-op: complete files are
skipped, truncated ones are re-downloaded.
"""

from __future__ import annotations

import argparse
import sys

from mousevision.allen import make_cache
from mousevision.config import load_subset
from mousevision.fetch import core_experiment_ids, download_experiments


def all_session_ids(cache, subset):
    ids = []
    for area in subset["areas"]:
        experiments = cache.get_ophys_experiments(
            experiment_container_ids=[subset["containers"][area]]
        )
        ids.extend(int(experiment["id"]) for experiment in experiments)
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="all sessions (A/B/C) of the 6 containers instead of the core set",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the plan; download nothing"
    )
    args = parser.parse_args(argv)

    subset = load_subset()
    cache = make_cache()

    ids = all_session_ids(cache, subset) if args.all else core_experiment_ids(subset)
    print(f"{'Would check' if args.dry_run else 'Checking'} {len(ids)} experiments")

    result = download_experiments(ids, cache, dry_run=args.dry_run)
    print(
        "downloaded={downloaded} redownloaded={redownloaded} skipped={skipped}".format(
            downloaded=len(result["downloaded"]),
            redownloaded=len(result["redownloaded"]),
            skipped=len(result["skipped"]),
        )
    )
    for label in ("downloaded", "redownloaded", "skipped"):
        for experiment_id in result[label]:
            print(f"  {label:>12}: {experiment_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
