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

Add the CNN extras (torch 2.2.2 + torchvision 0.17.2, macOS Intel) — needed for MV-04
feature extraction:

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

## Versions

Python 3.11 is the newest interpreter that allensdk 2.16.2 supports and that still has
torch wheels for macOS `x86_64`: torch 2.2.2 / torchvision 0.17.2 are the last
releases shipping Intel-Mac wheels (`cp311-none-macosx_10_9_x86_64`). allensdk pins its
own old stack (numpy<1.24, pandas==1.5.3, scipy<1.11), and `setuptools<81` keeps
`pkg_resources` available for it. Full table: [`docs/VERSIONS.md`](docs/VERSIONS.md).

## Citation

Allen Institute for Brain Science. Allen Brain Observatory — Visual Coding 2-photon.
Available from: https://observatory.brain-map.org/visualcoding
