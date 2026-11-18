"""Streamlit explorer for MouseVision results (MV-06).

Run with::

    uv run streamlit run app/streamlit_app.py

Results are read from ``$MV_RESULTS_DIR`` (default ``<repo>/results``). Sections
whose file has not been produced yet show an ``st.info`` hint instead of failing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mousevision.app_data import (  # noqa: E402
    AREAS,
    LAYERS,
    first_npz,
    load_results,
    results_dir,
)
from mousevision.config import load_subset  # noqa: E402

PAGES = ["Overview", "Tuning", "Receptive fields", "Decoding", "Brains vs CNNs"]

MISSING_HINT = "No results for this section yet — run scripts/run_*.py to generate them."

QUESTION = """
**Question.** How is visual information encoded across the mouse visual cortex, and how
far do the layers of a convolutional network reproduce it?

We take the frozen subset (Cux2-CreERT2, 175 µm, six visual areas), characterise
orientation/direction tuning and receptive fields, decode stimulus identity from
population activity as a function of cell count, and compare population geometry
(RSA, encoding models) with CNN layer activations.
"""


@st.cache_data(show_spinner=False)
def cached_results(directory: str) -> dict[str, Any]:
    """Load all result files once per results directory."""
    return load_results(directory)


@st.cache_data(show_spinner=False)
def cached_subset() -> dict[str, Any]:
    """The frozen subset definition from configs/subset.toml."""
    try:
        return load_subset()
    except Exception:  # TODO: surface a real error once MV-01 lands
        return {}


def _missing(label: str) -> None:
    st.info(f"{MISSING_HINT} (missing: {label})")


def _rgba(color: str, alpha: float = 0.18) -> str:
    """Plotly colours are '#rrggbb'; add an alpha channel for error bands."""
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _heatmap(df: pd.DataFrame, value: str, title: str, cbar: str) -> go.Figure:
    pivot = df.pivot(index="layer", columns="area", values=value).reindex(
        index=LAYERS, columns=AREAS
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorbar=dict(title=cbar),
            hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def page_overview(results: dict[str, Any]) -> None:
    st.header("Overview")
    st.markdown(QUESTION)

    subset = cached_subset()
    if not subset:
        _missing("configs/subset.toml")
        return

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Frozen subset")
        st.dataframe(
            pd.DataFrame(
                [
                    {"field": "cre_line", "value": subset.get("cre_line")},
                    {"field": "imaging_depth", "value": subset.get("imaging_depth")},
                    {"field": "areas", "value": ", ".join(subset.get("areas", []))},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("Areas and containers")
        containers = subset.get("containers", {})
        st.dataframe(
            pd.DataFrame(
                [
                    {"area": area, "container_id": containers.get(area)}
                    for area in subset.get("areas", AREAS)
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Available results")
    st.dataframe(
        pd.DataFrame(
            [
                {"file": key, "loaded": payload is not None}
                for key, payload in results.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def page_tuning(results: dict[str, Any]) -> None:
    st.header("Tuning")
    area, tuning = first_npz(results, "tuning_")
    if tuning is None:
        _missing("tuning_<area>.npz")
        return

    cell_ids = np.asarray(tuning["cell_ids"])
    cell = st.selectbox(f"Cell ({area})", cell_ids.tolist())
    idx = int(np.where(cell_ids == cell)[0][0])

    matrix = np.asarray(tuning["M"][idx], dtype=float)  # directions x temporal freqs
    direction_deg = np.arange(matrix.shape[0]) * (360.0 / matrix.shape[0])

    polar = go.Figure(
        go.Scatterpolar(
            r=np.append(matrix.mean(axis=1), matrix.mean(axis=1)[0]),
            theta=np.append(direction_deg, direction_deg[0]),
            fill="toself",
            name="mean over TFs",
        )
    )
    polar.update_layout(
        title=f"Orientation tuning · cell {cell}",
        polar=dict(angularaxis=dict(rotation=90, direction="clockwise")),
        height=420,
        margin=dict(l=40, r=40, t=60, b=20),
    )

    col_polar, col_osi, col_dsi = st.columns([1.4, 1, 1])
    with col_polar:
        st.plotly_chart(polar, use_container_width=True)

    for column, metric, label, color in (
        (col_osi, "osi", "OSI", "#2a6fdb"),
        (col_dsi, "dsi", "DSI", "#e07b39"),
    ):
        if metric not in tuning:
            continue
        values = np.asarray(tuning[metric], dtype=float)
        histogram = go.Figure(
            go.Histogram(x=values, nbinsx=15, marker_color=color)
        )
        histogram.add_vline(
            x=float(values[idx]), line_dash="dash", annotation_text=f"cell {cell}"
        )
        histogram.update_layout(
            title=label,
            xaxis_range=[0, 1],
            height=420,
            margin=dict(l=10, r=10, t=60, b=20),
        )
        with column:
            st.plotly_chart(histogram, use_container_width=True)

    st.caption(
        f"{len(cell_ids)} cells; "
        f"responsive: {int(np.sum(np.asarray(tuning.get('responsive', []))))}"
        if "responsive" in tuning
        else f"{len(cell_ids)} cells"
    )


def page_receptive_fields(results: dict[str, Any]) -> None:
    st.header("Receptive fields")
    area, rf = first_npz(results, "rf_")
    if rf is None:
        _missing("rf_<area>.npz")
        return

    cell_ids = np.asarray(rf["cell_ids"])
    cell = st.selectbox(f"Cell ({area})", cell_ids.tolist())
    idx = int(np.where(cell_ids == cell)[0][0])

    col_on, col_off = st.columns(2)
    for column, key, title, colorscale in (
        (col_on, "z_on", "ON", "RdBu_r"),
        (col_off, "z_off", "OFF", "RdBu_r"),
    ):
        if key not in rf:
            column.info(f"no {key} in rf_{area}.npz")
            continue
        z = np.asarray(rf[key][idx], dtype=float)
        figure = go.Figure(
            go.Heatmap(z=z, colorscale=colorscale, zmid=0.0, colorbar=dict(title="z"))
        )
        # TODO: real mask outline once MV-02 ships the RF mask polygons.
        figure.update_layout(
            title=f"{title} receptive field · cell {cell}",
            yaxis=dict(autorange="reversed", scaleanchor="x"),
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        with column:
            st.plotly_chart(figure, use_container_width=True)

    stats = []
    if "sig_on" in rf:
        stats.append(f"significant ON: {int(np.sum(np.asarray(rf['sig_on'])))}")
    if "sig_off" in rf:
        stats.append(f"significant OFF: {int(np.sum(np.asarray(rf['sig_off'])))}")
    if stats:
        st.caption(f"{area} · {len(cell_ids)} cells · " + " · ".join(stats))


def page_decoding(results: dict[str, Any]) -> None:
    st.header("Decoding")
    curve = results.get("decoding_curve")
    if curve is None:
        _missing("decoding_curve.csv")
        return

    colors = px.colors.qualitative.Plotly
    figure = go.Figure()
    for i, area in enumerate(AREAS):
        subset = curve[curve["area"] == area].sort_values("n_cells")
        if subset.empty:
            continue
        color = colors[i % len(colors)]
        if {"acc_lo", "acc_hi"} <= set(subset.columns):
            figure.add_trace(
                go.Scatter(
                    x=subset["n_cells"],
                    y=subset["acc_hi"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["n_cells"],
                    y=subset["acc_lo"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_rgba(color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=subset["n_cells"],
                y=subset["acc_mean"],
                mode="lines+markers",
                name=area,
                line=dict(color=color),
            )
        )

    if "chance" in curve.columns and not curve["chance"].dropna().empty:
        figure.add_hline(
            y=float(curve["chance"].dropna().iloc[0]),
            line_dash="dash",
            line_color="grey",
            annotation_text="chance",
        )
    figure.update_layout(
        title="Stimulus decoding accuracy vs number of cells",
        xaxis_title="cells",
        yaxis_title="accuracy",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)

    summary = results.get("decoding_summary")
    if summary is None:
        st.info("decoding_summary.csv not found — run scripts/run_*.py for the per-area table.")
    else:
        st.subheader("Per-area summary")
        st.dataframe(summary, hide_index=True, use_container_width=True)


def page_brains_vs_cnns(results: dict[str, Any]) -> None:
    st.header("Brains vs CNNs")

    rsa = results.get("rsa")
    encoding = results.get("encoding")
    hierarchy = results.get("hierarchy")

    col_rsa, col_enc = st.columns(2)
    with col_rsa:
        if rsa is None:
            _missing("rsa.csv")
        else:
            st.plotly_chart(
                _heatmap(rsa, "rho", "RSA · Spearman ρ (area × layer)", "ρ"),
                use_container_width=True,
            )
    with col_enc:
        if encoding is None:
            _missing("encoding.csv")
        else:
            st.plotly_chart(
                _heatmap(encoding, "r2", "Encoding · R² (area × layer)", "R²"),
                use_container_width=True,
            )

    if hierarchy is None:
        _missing("hierarchy.json")
        return

    st.subheader("Hierarchy")
    figure = go.Figure()
    for key, label in (("encoding", "encoding"), ("rsa", "RSA")):
        block = hierarchy.get(key) or {}
        centroids = block.get("centroids") or {}
        if not centroids:
            continue
        areas = [area for area in AREAS if area in centroids]
        figure.add_trace(
            go.Bar(
                x=areas,
                y=[centroids[area] for area in areas],
                name=label,
                text=[f"ρ={block.get('rho'):.2f}, p={block.get('p')}" for _ in areas]
                if block.get("rho") is not None
                else None,
            )
        )
    figure.update_layout(
        title="Area centroids vs CNN hierarchy",
        yaxis_title="centroid value",
        barmode="group",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption("TODO: replace the centroid bars with the rank-correlation line once MV-05 lands.")


def main() -> None:
    st.set_page_config(page_title="MouseVision", layout="wide")
    st.title("MouseVision")

    directory = results_dir()
    st.sidebar.caption(f"results: {directory}")
    page = st.sidebar.radio("Page", PAGES)

    results = cached_results(str(directory))

    renderer = {
        "Overview": page_overview,
        "Tuning": page_tuning,
        "Receptive fields": page_receptive_fields,
        "Decoding": page_decoding,
        "Brains vs CNNs": page_brains_vs_cnns,
    }[page]
    renderer(results)


main()
