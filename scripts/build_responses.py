"""Build per-trial response sets for every core experiment and target stimulus.

Writes data/interim/responses/<experiment>_<stimulus>.npz (gitignored): 6 natural
scenes files, VISp drifting gratings and VISp locally sparse noise.
"""

from __future__ import annotations

import argparse
import sys

from mousevision.allen import make_cache
from mousevision.config import load_subset
from mousevision.fetch import core_experiment_ids
from mousevision.responses import (
    build_response_set,
    response_path,
    save_response_set,
)

TARGET_STIMULI = {
    "natural_scenes",
    "drifting_gratings",
    "locally_sparse_noise",
    "locally_sparse_noise_4deg",
    "locally_sparse_noise_8deg",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; write nothing"
    )
    args = parser.parse_args(argv)

    cache = make_cache()
    written = skipped = 0

    for experiment_id in core_experiment_ids(load_subset()):
        dataset = cache.get_ophys_experiment_data(experiment_id)
        stimuli = [name for name in dataset.list_stimuli() if name in TARGET_STIMULI]
        if not stimuli:
            print(f"{experiment_id}: no target stimulus")
            continue
        for stimulus in stimuli:
            path = response_path(experiment_id, stimulus)
            if path.is_file() and not args.force:
                print(f"skip {path.name} (exists)")
                skipped += 1
                continue
            if args.dry_run:
                print(f"would write {path}")
                continue
            response_set = build_response_set(dataset, stimulus, experiment_id)
            save_response_set(response_set, path)
            written += 1
            print(
                f"wrote {path.name} responses={response_set['responses'].shape}"
            )

    print(f"written={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
