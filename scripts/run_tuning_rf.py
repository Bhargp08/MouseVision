"""VISp tuning curves and receptive fields: write results/ and docs/figures/."""

from __future__ import annotations

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mousevision import rf, tuning
from mousevision.config import REPO_ROOT
from mousevision.responses import load_response_set, response_path

matplotlib.use("Agg", force=True)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

DG_VISP = 510514474
LSN_VISP = 510174759


def load_or_exit(experiment_id, stimulus):
    path = response_path(experiment_id, stimulus)
    if not path.is_file():
        raise SystemExit(f"missing {path}; run scripts/build_responses.py first")
    return load_response_set(path)


def polar_examples(means, curves, flags, directions, path):
    ranked = [index for index in np.argsort(-np.nanmax(means, axis=(1, 2))) if flags[index]]
    cells = ranked[:6]
    if not cells:
        return

    figure, axes = plt.subplots(2, 3, subplot_kw={"projection": "polar"}, figsize=(9, 6))
    for axis, cell in zip(axes.ravel(), cells):
        tf_index = int(curves["pref_tf_idx"][cell])
        values = np.append(means[cell, :, tf_index], means[cell, 0, tf_index])
        angles = np.deg2rad(np.append(directions, directions[0]))
        axis.plot(angles, values, marker="o", markersize=3)
        axis.set_title(
            f"{curves['pref_dir_deg'][cell]:.0f} deg, {curves['pref_tf_hz'][cell]:.0f} Hz\n"
            f"OSI {curves['osi'][cell]:.2f}  DSI {curves['dsi'][cell]:.2f}",
            fontsize=8,
        )
    for axis in axes.ravel()[len(cells):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def osi_dsi_histogram(curves, path):
    figure, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(curves["osi"], bins=20, color="#3b6ea5")
    axes[0].set_xlabel("OSI")
    axes[0].set_ylabel("cells")
    axes[1].hist(curves["dsi"], bins=20, color="#a5543b")
    axes[1].set_xlabel("DSI")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def rf_examples(maps, path):
    cells = [index for index in range(maps["sig_on"].size) if maps["sig_on"][index]][:6]
    if not cells:
        return

    figure, axes = plt.subplots(2, min(3, len(cells)), figsize=(9, 5), squeeze=False)
    for column, cell in enumerate(cells[:3]):
        axes[0, column].imshow(maps["z_on"][cell], cmap="RdBu_r", vmin=-3, vmax=3)
        axes[0, column].set_title(f"cell {cell} ON z", fontsize=8)
        axes[1, column].imshow(maps["z_off"][cell], cmap="RdBu_r", vmin=-3, vmax=3)
        axes[1, column].set_title(f"cell {cell} OFF z", fontsize=8)
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-shuffles", type=int, default=200)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dg = load_or_exit(DG_VISP, "drifting_gratings")
    means, directions, temporal_frequencies, blank_mean = tuning.condition_means(dg)
    curves = tuning.osi_dsi(means)
    flags = tuning.responsive(dg)

    np.savez_compressed(
        RESULTS_DIR / "tuning_VISp.npz",
        M=means,
        directions=directions,
        temporal_frequencies=temporal_frequencies,
        blank_mean=blank_mean,
        responsive=flags,
        cell_specimen_ids=np.asarray(dg["cell_specimen_ids"], dtype=np.int64),
        **curves,
    )

    polar_examples(means, curves, flags, directions, FIGURES_DIR / "tuning_examples.png")
    osi_dsi_histogram(curves, FIGURES_DIR / "osi_dsi_hist.png")

    lsn = load_or_exit(LSN_VISP, "locally_sparse_noise")
    significance = rf.rf_significance(
        lsn, n_shuffles=args.n_shuffles, rng=np.random.default_rng(0)
    )
    np.savez_compressed(
        RESULTS_DIR / "rf_VISp.npz",
        z_on=significance["z_on"].astype(np.float16),
        z_off=significance["z_off"].astype(np.float16),
        sig_on=significance["sig_on"],
        sig_off=significance["sig_off"],
        cell_specimen_ids=np.asarray(lsn["cell_specimen_ids"], dtype=np.int64),
    )
    rf_examples(significance, FIGURES_DIR / "rf_examples.png")

    n_cells = flags.size
    print(f"responsive: {flags.sum()}/{n_cells} ({flags.mean():.0%}) of VISp DG cells")
    either = significance["sig_on"] | significance["sig_off"]
    print(
        f"significant RF: {either.sum()}/{either.size} ({either.mean():.0%}) of VISp LSN cells"
    )
    print(f"wrote {RESULTS_DIR/'tuning_VISp.npz'}, {RESULTS_DIR/'rf_VISp.npz'} and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
