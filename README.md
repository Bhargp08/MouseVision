# MouseVision

Encoding and decoding mouse visual cortex activity from the Allen Brain Observatory
(Visual Coding 2-photon) and comparing it with layers of a convolutional neural
network. This repo is currently at MV-00 only: scaffold, pinned environment, CI and
data-cache configuration. No analysis code yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`.

Core environment (no torch):

```bash
uv sync --locked --group dev
```

Adds torch 2.2.2 + torchvision 0.17.2: macOS Intel (x86_64) wheels on darwin, CPU-only
wheels on linux; needed only for MV-04 feature extraction:

```bash
uv sync --locked --group dev --extra cnn
```

## Test

```bash
uv run pytest -m "not slow"
```

`slow` is reserved for tests that need network access or a populated Allen cache; CI
runs only the fast subset. Lint with `uv run ruff check .`.

## Data

Raw Allen data (NWB/HDF5 plus the SDK cache) goes to `data/raw/`, which is gitignored —
it never enters git. Point it somewhere else (an external drive, for example) with:

```bash
export MV_DATA_DIR=/Volumes/scratch/mousevision
```

The cache then lives at `$MV_DATA_DIR/raw/allen_cache`. Small derived `.npz` features
under `data/features/` are committed.

Fetch the frozen core set (8 experiments, ~3.2 GB; idempotent — a re-run skips
complete files and repairs truncated ones) and build the response matrices:

```bash
uv run python scripts/download.py          # --dry-run to preview, --all for every session
uv run python scripts/build_responses.py   # 8 response sets under data/interim/responses/
```

## Versions

Python 3.11 is the newest interpreter that allensdk 2.16.2 supports and that still has
torch wheels for macOS `x86_64`: torch 2.2.2 / torchvision 0.17.2 are the last
releases shipping Intel-Mac wheels (`cp311-none-macosx_10_9_x86_64`). allensdk pins its
own old stack (numpy<1.24, pandas==1.5.3, scipy<1.11), and `setuptools<81` keeps
`pkg_resources` available for it. Full table: [`docs/VERSIONS.md`](docs/VERSIONS.md).

## Brains vs CNNs: RSA

```bash
uv sync --locked --group dev --extra cnn   # once, on the Intel Mac
uv run python scripts/extract_features.py  # caches data/features/resnet18_ns.npz
uv run python scripts/run_rsa.py
```

ImageNet ResNet-18 activations (layer1..layer4, average-pooled 1x1 or 2x2) are compared
with each area's natural-scene response RDM. Distances are correlation distances,
`RDM[i, j] = 1 - Pearson(r_i, r_j)` (Kriegeskorte et al. 2008), and similarity is
Spearman rho over the upper triangle only. Reliability comes from split-half RDMs:
`r_half = mean rho(rdm(A), rdm(B))` and the Spearman-Brown ceiling
`ceiling = 2 r_half / (1 + r_half)`. Significance permutes model image labels, moving
rows and columns together, with `p = (1 + #null >= rho) / (1 + n_perm)`.

## Decoding

```bash
uv run python scripts/run_decoding.py
```

Image identity from population dF/F: multinomial logistic regression (`C=0.01`) on
standardised responses, 5-fold stratified CV with the scaler fit inside each fold, so
nothing leaks. Chance is `1/118`. Accuracy is reported against neuron count per area
(random subsets, 10 draws) plus a permutation test at a common neuron count and a PCA
participation ratio over the 118 trial-averaged image vectors.

## Tuning & RFs

```bash
uv run python scripts/run_tuning_rf.py
```

Writes `results/tuning_VISp.npz`, `results/rf_VISp.npz` and the tuning/RF figures.
OSI and DSI follow the Allen definitions at each cell's preferred direction and temporal
frequency: `OSI = (pref - orth) / (pref + orth)` with `orth` the mean of the two
orthogonal directions, and `DSI = (pref - null) / (pref + null)` with `null` the
opposite direction. Responses are mean dF/F over `[start+4, end+4)` frames minus the
1 s pre-stimulus baseline, so OSI/DSI are not clipped and can fall outside [-1, 1].
Receptive fields are spike-triggered averages of z-scored responses over locally sparse
noise trials; significance compares the max-pixel z against shuffled trial labels.

## Encoding & hierarchy

```bash
uv run python scripts/run_encoding.py
```

Ridge regression from 2x2-pooled ResNet-18 layer activations to each neuron's
image-mean response, with the feature scaler, the PCA (50 components) and the per-neuron
ridge alpha (inner 5-fold) all fit inside the outer folds. Neurons are kept when their
split-half ceiling is at least 0.3. R^2 is reported both raw and as a fraction of that
ceiling (`r2/ceiling`). Per-area layer centroids are
`sum(i * max(s_i, 0)) / sum(max(s_i, 0))` over layers 1..4, tested against the assumed
anatomical order `VISp, VISl, VISrl, VISal, VISpm, VISam` with an exact permutation test
over all 720 orderings (one-sided, identity included).

## Citation

Allen Institute for Brain Science. Allen Brain Observatory — Visual Coding 2-photon.
Available from: https://observatory.brain-map.org/visualcoding
