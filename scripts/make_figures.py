"""Render one PNG per Streamlit section into docs/figures/ from the result files.

Usage::

    uv run python scripts/make_figures.py [--results-dir DIR] [--out-dir DIR]

Sections whose result file is missing are skipped with a warning, so the script
runs today against an empty results directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in tests or CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mousevision.app_data import (  # noqa: E402
    AREAS,
    LAYERS,
    first_npz,
    load_results,
)

# One figure per app section, in the order they are written.
FIGURES = [
    "tuning.png",
    "receptive_fields.png",
    "decoding.png",
    "rsa.png",
    "encoding.png",
    "hierarchy.png",
]


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir / name}")


def plot_tuning(results: dict[str, Any], out_dir: Path) -> bool:
    area, tuning = first_npz(results, "tuning_")
    if tuning is None:
        print("skip tuning.png: no tuning_<area>.npz")
        return False

    matrix = np.asarray(tuning["M"], dtype=float)  # cells x directions x TFs
    mean_dir = matrix.mean(axis=(0, 2))
    theta = np.linspace(0.0, 2.0 * np.pi, mean_dir.size, endpoint=False)

    fig = plt.figure(figsize=(11, 3.6))
    polar = fig.add_subplot(1, 3, 1, projection="polar")
    polar.plot(np.append(theta, theta[0]), np.append(mean_dir, mean_dir[0]))
    polar.set_title(f"{area}: population tuning")

    for column, metric, label in ((2, "osi", "OSI"), (3, "dsi", "DSI")):
        axes = fig.add_subplot(1, 3, column)
        if metric in tuning:
            axes.hist(np.asarray(tuning[metric], dtype=float), bins=15)
        axes.set_xlabel(label)
        axes.set_ylabel("cells")
        axes.set_title(f"{area}: {label}")

    fig.suptitle("Tuning")
    _save(fig, out_dir, "tuning.png")
    return True


def plot_receptive_fields(results: dict[str, Any], out_dir: Path) -> bool:
    area, rf = first_npz(results, "rf_")
    if rf is None:
        print("skip receptive_fields.png: no rf_<area>.npz")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    plotted = False
    for axes_item, key, title in ((axes[0], "z_on", "ON"), (axes[1], "z_off", "OFF")):
        if key not in rf:
            axes_item.set_axis_off()
            continue
        z = np.asarray(rf[key], dtype=float)
        image = axes_item.imshow(z.mean(axis=0), cmap="RdBu_r", aspect="auto")
        axes_item.set_title(f"{area}: mean {title} RF")
        axes_item.set_xticks([])
        axes_item.set_yticks([])
        fig.colorbar(image, ax=axes_item, fraction=0.046, label="z")
        plotted = True
    if not plotted:
        plt.close(fig)
        print("skip receptive_fields.png: neither z_on nor z_off present")
        return False

    fig.suptitle("Receptive fields")
    _save(fig, out_dir, "receptive_fields.png")
    return True


def plot_decoding(results: dict[str, Any], out_dir: Path) -> bool:
    curve = results.get("decoding_curve")
    if curve is None:
        print("skip decoding.png: no decoding_curve.csv")
        return False

    fig, axes = plt.subplots(figsize=(6.5, 4.0))
    for area in AREAS:
        subset = curve[curve["area"] == area].sort_values("n_cells")
        if subset.empty:
            continue
        line = axes.plot(subset["n_cells"], subset["acc_mean"], marker="o", label=area)[0]
        if {"acc_lo", "acc_hi"} <= set(subset.columns):
            axes.fill_between(
                subset["n_cells"],
                subset["acc_lo"],
                subset["acc_hi"],
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )

    if "chance" in curve.columns and not curve["chance"].dropna().empty:
        axes.axhline(
            float(curve["chance"].dropna().iloc[0]),
            linestyle="--",
            color="grey",
            label="chance",
        )
    axes.set_xlabel("cells")
    axes.set_ylabel("accuracy")
    axes.set_title("Decoding accuracy vs number of cells")
    axes.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "decoding.png")
    return True


def _heatmap(
    dataframe, value: str, title: str, cbar_label: str, out_dir: Path, name: str
) -> bool:
    if dataframe is None:
        print(f"skip {name}: column not loaded")
        return False
    if value not in dataframe.columns:
        print(f"skip {name}: no '{value}' column")
        return False

    pivot = dataframe.pivot(index="layer", columns="area", values=value).reindex(
        index=LAYERS, columns=AREAS
    )
    fig, axes = plt.subplots(figsize=(6.0, 3.8))
    image = axes.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    axes.set_xticks(range(len(pivot.columns)), list(pivot.columns))
    axes.set_yticks(range(len(pivot.index)), list(pivot.index))
    axes.set_title(title)
    fig.colorbar(image, ax=axes, fraction=0.046, label=cbar_label)
    _save(fig, out_dir, name)
    return True


def plot_rsa(results: dict[str, Any], out_dir: Path) -> bool:
    return _heatmap(results.get("rsa"), "rho", "RSA: Spearman rho", "rho", out_dir, "rsa.png")


def plot_encoding(results: dict[str, Any], out_dir: Path) -> bool:
    return _heatmap(
        results.get("encoding"), "r2", "Encoding: R2", "R2", out_dir, "encoding.png"
    )


def plot_hierarchy(results: dict[str, Any], out_dir: Path) -> bool:
    hierarchy = results.get("hierarchy")
    if hierarchy is None:
        print("skip hierarchy.png: no hierarchy.json")
        return False

    fig, axes = plt.subplots(figsize=(6.5, 3.8))
    width = 0.38
    labels = [key for key in ("encoding", "rsa") if (hierarchy.get(key) or {}).get("centroids")]
    for offset, key in zip((-width / 2, width / 2), labels):
        centroids = hierarchy[key]["centroids"]
        areas = [area for area in AREAS if area in centroids]
        positions = np.arange(len(areas)) + offset
        axes.bar(positions, [centroids[area] for area in areas], width=width, label=key)
        axes.set_xticks(np.arange(len(areas)), areas)
        rho, p = hierarchy[key].get("rho"), hierarchy[key].get("p")
        if rho is not None:
            axes.text(
                0.02,
                0.95 if offset < 0 else 0.86,
                f"{key}: rho={rho}, p={p}",
                transform=axes.transAxes,
                fontsize=8,
            )

    if not labels:
        plt.close(fig)
        print("skip hierarchy.png: no centroids in hierarchy.json")
        return False

    axes.set_ylabel("centroid")
    axes.set_title("Area centroids vs CNN hierarchy")
    axes.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "hierarchy.png")
    return True


PLOTTERS = [
    plot_tuning,
    plot_receptive_fields,
    plot_decoding,
    plot_rsa,
    plot_encoding,
    plot_hierarchy,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="directory with the result files (default: $MV_RESULTS_DIR or <repo>/results)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "docs" / "figures"),
        help="where to write the PNGs (default: docs/figures)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    for plotter in PLOTTERS:
        plotter(results, out_dir)
    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
