"""Natural-scene decoding per area: accuracy vs neuron count, permutation tests, PCA."""

from __future__ import annotations

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mousevision.config import REPO_ROOT, load_subset
from mousevision.decoding import (
    CHANCE,
    neuron_curve,
    ns_xy,
    pca_geometry,
    permutation_pvalue,
)
from mousevision.responses import load_response_set, response_path

matplotlib.use("Agg", force=True)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

AREA_ORDER = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]


def load_natural_scenes():
    sets = {}
    for experiment_id in load_subset()["core_experiments"]["natural_scenes"]:
        path = response_path(experiment_id, "natural_scenes")
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/build_responses.py first")
        response_set = load_response_set(path)
        sets[response_set["targeted_structure"]] = response_set
    return sets


def plot_curves(curves, path):
    figure, axis = plt.subplots(figsize=(6, 4))
    for area in AREA_ORDER:
        if area not in curves:
            continue
        curve = curves[area]
        grouped = curve.groupby("n_neurons")["accuracy"]
        mean = grouped.mean()
        sem = grouped.std(ddof=1) / np.sqrt(grouped.count())
        axis.errorbar(
            mean.index,
            mean.values,
            yerr=1.96 * sem.values,
            marker="o",
            markersize=3,
            label=area,
        )
    axis.axhline(CHANCE, color="grey", linestyle="--", label="chance")
    axis.set_xscale("log")
    axis.set_xlabel("neurons")
    axis.set_ylabel("decoding accuracy")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_scree(geometries, path):
    figure, axis = plt.subplots(figsize=(6, 4))
    for area in AREA_ORDER:
        if area not in geometries:
            continue
        ratios = geometries[area]["explained_variance_ratio"][:30]
        axis.plot(np.arange(1, ratios.size + 1), np.cumsum(ratios), label=area)
    axis.set_xlabel("principal components")
    axis.set_ylabel("cumulative explained variance")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=10)
    parser.add_argument("--n-perm", type=int, default=100)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sets = load_natural_scenes()
    matrices = {area: ns_xy(rs) for area, rs in sets.items()}
    common_n = min(X.shape[1] for X, _ in matrices.values())

    curves = {}
    summary = []
    geometries = {}
    for area in AREA_ORDER:
        if area not in matrices:
            continue
        X, y = matrices[area]
        curve = neuron_curve(X, y, n_draws=args.n_draws)
        curves[area] = curve

        columns = np.arange(min(common_n, X.shape[1]))
        observed, _, p_value = permutation_pvalue(
            X[:, columns], y, n_perm=args.n_perm
        )
        geometry = pca_geometry(X, y)
        geometries[area] = geometry

        all_cells = curve[curve["n_neurons"] == X.shape[1]]["accuracy"].mean()
        summary.append(
            {
                "area": area,
                "n_cells": int(X.shape[1]),
                "acc_all": float(all_cells),
                "acc_at_common_n": float(observed),
                "common_n": int(common_n),
                "perm_p": float(p_value),
                "participation_ratio": float(geometry["participation_ratio"]),
            }
        )

    pd.concat(curves.values(), keys=curves.keys(), names=["area"]).reset_index(
        level=0
    ).to_csv(RESULTS_DIR / "decoding_curve.csv", index=False)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(RESULTS_DIR / "decoding_summary.csv", index=False)

    plot_curves(curves, FIGURES_DIR / "decoding_vs_n.png")
    plot_scree(geometries, FIGURES_DIR / "pca_scree.png")

    print(summary_frame.to_string(index=False))
    print(f"chance = {CHANCE:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
