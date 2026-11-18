"""RSA between natural-scene population RDMs and ResNet-18 layer RDMs."""

from __future__ import annotations

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mousevision.config import REPO_ROOT, load_subset
from mousevision.features import load_features
from mousevision.responses import load_response_set, response_path
from mousevision.rsa import LAYER_ORDER, image_means, rdm, rsa_table

matplotlib.use("Agg", force=True)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
AREA_ORDER = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]


def load_sets():
    sets = {}
    for experiment_id in load_subset()["core_experiments"]["natural_scenes"]:
        path = response_path(experiment_id, "natural_scenes")
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/build_responses.py first")
        response_set = load_response_set(path)
        sets[response_set["targeted_structure"]] = response_set
    return sets


def plot_heatmap(table, path):
    areas = [area for area in AREA_ORDER if area in set(table["area"])]
    columns = list(LAYER_ORDER)
    pivot = (
        table.pivot(index="area", columns="layer", values="rho_over_ceiling")
        .reindex(index=areas, columns=columns)
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(areas)), areas)
    for row_index, area in enumerate(areas):
        for column_index, layer in enumerate(columns):
            match = table[(table["area"] == area) & (table["layer"] == layer)]
            if match.empty:
                continue
            rho = match["rho"].iloc[0]
            ceiling = match["ceiling"].iloc[0]
            star = "*" if match["p"].iloc[0] < 0.05 else ""
            axis.text(
                column_index,
                row_index,
                f"{rho:.2f}/{ceiling:.2f}{star}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
    figure.colorbar(image, label="rho / ceiling")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_brain_rdms(sets, path):
    areas = [area for area in AREA_ORDER if area in sets]
    figure, axes = plt.subplots(2, 3, figsize=(9, 5), squeeze=False)
    for axis, area in zip(axes.ravel(), areas):
        response_set = sets[area]
        frames = np.asarray(response_set["trial_frame"], dtype=np.int64)
        keep = frames >= 0
        responses = np.asarray(response_set["responses"], dtype=np.float64).T[keep]
        means = image_means(responses, frames[keep])
        axis.imshow(rdm(means), cmap="magma")
        axis.set_title(area, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(areas):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sets = load_sets()
    features = load_features()
    table = rsa_table(sets, features, grid=args.grid, n_perm=args.n_perm)
    table.to_csv(RESULTS_DIR / "rsa.csv", index=False)

    plot_heatmap(table, FIGURES_DIR / "rsa_heatmap.png")
    plot_brain_rdms(sets, FIGURES_DIR / "brain_rdms.png")

    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
