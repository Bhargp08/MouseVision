# Pinned versions

`uv.lock` is committed; regenerate it only with `uv lock`. The resolver is frozen at
`exclude-newer = 2024-06-01T00:00:00Z` because allensdk 2.16.2's old pins
(numpy<1.24, pandas==1.5.3, scipy<1.11, xarray<2023.2.0) stop resolving against newer
statsmodels/hdmf/pynwb releases.

| Package | Pin | Locked | Notes |
| --- | --- | --- | --- |
| Python | 3.11 | 3.11 | newest interpreter allensdk 2.16.2 supports that also has torch wheels for macOS x86_64 |
| torch | ==2.2.2 | 2.2.2 (PyPI, macOS x86_64) / 2.2.2+cpu (linux) | last release with `cp311-none-macosx_10_9_x86_64` wheels |
| torchvision | ==0.17.2 | 0.17.2 / 0.17.2+cpu (linux) | matching pair for torch 2.2.2 |
| allensdk | ==2.16.2 | 2.16.2 | last release, 2023-11-30 |
| numpy | ==1.23.5 | 1.23.5 | allensdk requires numpy<1.24 |
| pandas | ==1.5.3 | 1.5.3 | allensdk requires pandas==1.5.3 |
| scipy | ==1.10.1 | 1.10.1 | allensdk requires scipy<1.11 |
| setuptools | <81 | 70.0.0 | allensdk imports pkg_resources, removed in setuptools>=81 |
| scikit-learn | unpinned | 1.5.0 | |
| streamlit | unpinned | 1.35.0 | |
| plotly | unpinned | 5.22.0 | |
| pytest | unpinned (dev group) | 8.2.1 | |
| ruff | unpinned (dev group) | 0.4.7 | |

## Why

torch 2.2.2 is the last release that ships macOS x86_64 wheels (newer releases are
arm64-only) and allensdk 2.16.2 pins its own old numpy/pandas/scipy, so Python 3.11 is
the newest interpreter satisfying both. On Linux torch comes from the `pytorch-cpu`
index; CI never installs the `cnn` extra.

## Rule

No pin changes without a note here.
