# MouseVision

Encoding and decoding mouse visual cortex activity from the Allen Brain Observatory, and
comparing those population codes with the layers of a convolutional neural network.

The question: when a mouse sees a natural image, how much of a visual area's response
can be predicted from a CNN layer, and does that match the anatomical hierarchy of
visual areas? MouseVision extracts per-trial dF/F responses to drifting gratings,
locally sparse noise and 118 natural scenes from six visual areas of one Cre line, then
measures tuning, receptive fields, decoding accuracy, representational similarity (RSA)
and ridge-encoding performance against ResNet-18 layers, always relative to a
split-half noise ceiling.

## Data

Allen Brain Observatory, Visual Coding 2-photon (public; no API key):

- Cre line `Cux2-CreERT2`, imaging depth 175 um.
- Areas: VISp, VISl, VISal, VISpm, VISam, VISrl.
- 8 core experiments (~3.2 GB): natural scenes in all six areas, drifting gratings and
  locally sparse noise in VISp. The exact ids live in `configs/subset.toml`.

Raw and interim data stay out of git: `data/raw/` and `data/interim/` are gitignored.
Point them elsewhere with `MV_DATA_DIR` (the cache then lives at
`$MV_DATA_DIR/raw/allen_cache`). Small derived files under `data/features/` and
`results/` are committed.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`.

```bash
uv sync --locked --group dev              # core: allensdk stack, no torch
uv sync --locked --group dev --extra cnn  # adds torch 2.2.2 + torchvision 0.17.2
```

The `cnn` extra uses macOS Intel (x86_64) wheels on darwin and CPU-only wheels on linux;
it is needed only for the ResNet-18 feature cache. CI never installs it.

```bash
uv run pytest -m "not slow"   # fast synthetic suite, no network, no NWB files
uv run ruff check .
```

Slow tests (`-m slow`) use real cached data and skip when a file is missing.

## Reproduce

```bash
# 1. download the frozen subset (idempotent; --dry-run to preview)
uv run python scripts/download.py

# 2. per-trial responses -> data/interim/responses/*.npz (8 files)
uv run python scripts/build_responses.py

# 3. tuning curves, OSI/DSI, receptive fields
uv run python scripts/run_tuning_rf.py

# 4. decoding accuracy vs neuron count, RSA-free PCA geometry
uv run python scripts/run_decoding.py

# 5. ResNet-18 features (once, on the Intel Mac, with the cnn extra)
uv run python scripts/extract_features.py

# 6. RSA per area x layer with noise ceilings
uv run python scripts/run_rsa.py

# 7. ridge encoding and the layer-area hierarchy test
uv run python scripts/run_encoding.py

# 8. regenerate the figure set from results/
uv run python scripts/make_figures.py

# 9. explore everything in the browser app
uv run streamlit run app/streamlit_app.py
```

## Results

Every script above writes small, committed outputs (schema fixed so the app can read
them):

| File | Contents |
| --- | --- |
| `results/tuning_VISp.npz` | `cell_ids, M, osi, dsi, pref_dir, pref_tf, responsive` |
| `results/rf_VISp.npz` | `cell_ids, z_on, z_off, sig_on, sig_off` |
| `results/decoding_curve.csv` | `area, n_cells, acc_mean, acc_lo, acc_hi, chance` |
| `results/decoding_summary.csv` | `area, n_cells, acc_all, acc_at_common_n, perm_p, participation_ratio` |
| `results/rsa.csv` | `area, layer, rho, ceiling, rho_over_ceiling, p` |
| `results/encoding.csv` | `area, layer, r2, ceiling, r2_over_ceiling` |
| `results/hierarchy.json` | `{"encoding": {"rho", "p", "centroids"}, "rsa": {...}}` |

Those files are regenerated from the local Allen cache; the numbers in them come from
the eight experiments above, so they are one Cre line at one depth, not a population
estimate. Key figures land in `docs/figures/`:

- `decoding_vs_n.png` — accuracy rises with neuron count in every area and more cells
  buy more accuracy, but areas differ at a common neuron count.
- `rsa_heatmap.png` — CNN layer vs area similarity, annotated with rho / ceiling and
  `p < 0.05` markers.
- `layer_area_map.png` — encoding R2 as a fraction of the noise ceiling per area and
  layer.
- `hierarchy.png` — per-area layer centroid against the assumed area order.
- `tuning_examples.png`, `osi_dsi_hist.png`, `rf_examples.png` — example cells and
  population tuning summaries.

Nothing here is a claim about mouse vision in general; treat the hierarchy test as
exploratory.

## Limitations

- dF/F from calcium imaging, not spikes: slow, non-linear, and correlated across
  neighbouring cells.
- About 130-330 neurons per area, from one Cre line (`Cux2-CreERT2`) at one imaging
  depth (175 um) in one animal per area.
- One model: ImageNet-trained ResNet-18. No architecture or training-set comparison.
- The area hierarchy `VISp, VISl, VISrl, VISal, VISpm, VISam` is an assumption from
  Harris et al. 2019 and Siegle et al. 2021, not something this repo measures; the
  permutation test only asks whether the data agree with that fixed order.
- Noise ceilings are estimated by split-half reliability, which is itself noisy at this
  sample size.
- Statistics are deliberately simple (5-fold CV, fixed alpha grids, 200-1000
  permutations) and can be tightened.

## Conventions and formulas

Response per cell `c` and trial `k` (frames are 2-photon frames, ~30 Hz, half-open):

```
response[c, k] = mean(dff[c, start_k + 4 : end_k + 4]) - mean(dff[c, start_k - 30 : start_k])
```

`LAG_FRAMES = 4` (~133 ms) accounts for the GCaMP6f lag; `BASELINE_FRAMES = 30` (1 s)
is the pre-stimulus baseline.

Tuning, at each cell's preferred direction/TF: `OSI = (pref - orth) / (pref + orth)` and
`DSI = (pref - null) / (pref + null)`; responses can be negative, so these are not
clipped to [-1, 1].

RSA uses correlation distances `RDM[i, j] = 1 - Pearson(r_i, r_j)` and Spearman rho over
the upper triangle. Reliability is the Spearman-Brown ceiling
`ceiling = 2 r_half / (1 + r_half)` from split-half RDMs. Encoding reports
`R2 = 1 - SSE/SST` and `r2 / ceiling` per neuron, and layer centroids are
`sum(i * max(s_i, 0)) / sum(max(s_i, 0))` over layers 1..4. Permutation p-values use
`p = (1 + #null >= observed) / (1 + n_perm)`.

## Versions

Python 3.11 is the newest interpreter that allensdk 2.16.2 supports and that still has
torch wheels for macOS x86_64. torch 2.2.2 / torchvision 0.17.2 are the last releases
shipping Intel-Mac wheels; allensdk pins its own old stack (numpy 1.23.5, pandas 1.5.3,
scipy 1.10.1) and `setuptools<81` keeps `pkg_resources` available. The resolver is
frozen with `exclude-newer = 2024-06-01`. Full table and the pin-change rule:
[`docs/VERSIONS.md`](docs/VERSIONS.md).

## Citation

Allen Institute for Brain Science. Allen Brain Observatory — Visual Coding 2-photon.
Available from: https://observatory.brain-map.org/visualcoding

## License

MIT — see [`LICENSE`](LICENSE).
