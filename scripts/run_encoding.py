"""Ridge encoding per area x layer, R^2/ceiling, and the layer-area hierarchy test."""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mousevision.config import REPO_ROOT, load_subset
from mousevision.encoding import (
    AREA_HIERARCHY,
    encoding_table,
    hierarchy_test,
    layer_centroid,
)
from mousevision.features import load_features
from mousevision.responses import load_response_set, response_path
from mousevision.rsa import LAYER_ORDER

matplotlib.use("Agg", force=True)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def load_sets():
    sets = {}
    for experiment_id in load_subset()["core_experiments"]["natural_scenes"]:
        path = response_path(experiment_id, "natural_scenes")
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/build_responses.py first")
        response_set = load_response_set(path)
        sets[response_set["targeted_structure"]] = response_set
    return sets


def plot_layer_area(table, path):
    areas = [area for area in AREA_HIERARCHY if area in set(table["area"])]
    pivot = (
        table.pivot(index="area", columns="layer", values="r2_over_ceiling")
        .reindex(index=areas, columns=list(LAYER_ORDER))
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(pivot.to_numpy(dtype=float), cmap="cividis", aspect="auto")
    axis.set_xticks(range(len(LAYER_ORDER)), LAYER_ORDER)
    axis.set_yticks(range(len(areas)), areas)
    figure.colorbar(image, label="R2 / ceiling")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_hierarchy(centroids, rho, p_value, path):
    areas = [area for area in AREA_HIERARCHY if area in centroids]
    if not areas:
        return
    figure, axis = plt.subplots(figsize=(5, 3))
    axis.plot(np.arange(1, len(areas) + 1), [centroids[area] for area in areas], marker="o")
    axis.set_xticks(np.arange(1, len(areas) + 1), areas)
    axis.set_ylabel("layer centroid")
    axis.set_title(f"rho = {rho:.2f}, p = {p_value:.3f}", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def rsa_hierarchy():
    path = RESULTS_DIR / "rsa.csv"
    if not path.is_file():
        return {"rho": None, "p": None, "centroids": {}}
    table = pd.read_csv(path)
    centroids = {}
    for area, frame in table.groupby("area"):
        scores = dict(zip(frame["layer"], frame["rho_over_ceiling"], strict=False))
        centroids[area] = layer_centroid(scores)
    complete = {area: value for area, value in centroids.items() if area in AREA_HIERARCHY}
    if len(complete) < len(AREA_HIERARCHY):
        return {"rho": None, "p": None, "centroids": centroids}
    rho, p_value = hierarchy_test(complete)
    return {"rho": float(rho), "p": float(p_value), "centroids": centroids}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=2)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sets = load_sets()
    features = load_features()
    table = encoding_table(sets, features, grid=args.grid)

    output = table.rename(
        columns={"median_r2": "r2", "median_r2_over_ceiling": "r2_over_ceiling"}
    )[["area", "layer", "r2", "ceiling", "r2_over_ceiling"]]
    output.to_csv(RESULTS_DIR / "encoding.csv", index=False)
    table.attrs["neurons"].to_csv(RESULTS_DIR / "encoding_neurons.csv", index=False)

    centroids = {}
    for area, frame in output.groupby("area"):
        scores = dict(zip(frame["layer"], frame["r2_over_ceiling"], strict=False))
        centroids[area] = layer_centroid(scores)
    complete = {area: value for area, value in centroids.items() if area in AREA_HIERARCHY}
    if len(complete) == len(AREA_HIERARCHY):
        rho, p_value = hierarchy_test(complete)
    else:
        rho, p_value = float("nan"), float("nan")

    hierarchy = {
        "encoding": {"rho": float(rho), "p": float(p_value), "centroids": centroids},
        "rsa": rsa_hierarchy(),
    }
    (RESULTS_DIR / "hierarchy.json").write_text(json.dumps(hierarchy, indent=2) + "\n")

    plot_layer_area(output, FIGURES_DIR / "layer_area_map.png")
    plot_hierarchy(centroids, rho, p_value, FIGURES_DIR / "hierarchy.png")

    print(output.to_string(index=False))
    print(f"encoding hierarchy: rho={rho:.3f} p={p_value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
