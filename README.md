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

## Citation

Allen Institute for Brain Science. Allen Brain Observatory — Visual Coding 2-photon.
Available from: https://observatory.brain-map.org/visualcoding
