"""scripts/make_figures.py writes one PNG per section from the fixture results."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "results"
SCRIPT = ROOT / "scripts" / "make_figures.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writes_expected_pngs(tmp_path):
    make_figures = _load_script()

    assert make_figures.main(
        ["--results-dir", str(FIXTURES), "--out-dir", str(tmp_path)]
    ) == 0

    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == sorted(make_figures.FIGURES)
    for name in make_figures.FIGURES:
        assert (tmp_path / name).stat().st_size > 0, name

    assert {"decoding.png", "rsa.png", "encoding.png", "hierarchy.png"} <= set(written)


def test_empty_results_dir_does_not_crash(tmp_path):
    make_figures = _load_script()

    out_dir = tmp_path / "figures"
    assert make_figures.main(
        ["--results-dir", str(tmp_path / "nothing"), "--out-dir", str(out_dir)]
    ) == 0

    assert out_dir.is_dir()
    assert not list(out_dir.glob("*.png"))
