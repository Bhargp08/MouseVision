"""Every Streamlit page must render (with no exception) with and without results."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
FIXTURES = Path(__file__).parent / "fixtures" / "results"

PAGES = ["Overview", "Tuning", "Receptive fields", "Decoding", "Brains vs CNNs"]


def _assert_all_pages_render() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert app.sidebar.radio[0].options == PAGES

    for page in PAGES[1:]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)


def test_all_pages_render_with_fixtures(monkeypatch):
    monkeypatch.setenv("MV_RESULTS_DIR", str(FIXTURES))

    _assert_all_pages_render()


def test_all_pages_render_with_empty_results(monkeypatch, tmp_path):
    monkeypatch.setenv("MV_RESULTS_DIR", str(tmp_path))

    _assert_all_pages_render()


def test_missing_results_show_the_run_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("MV_RESULTS_DIR", str(tmp_path))

    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value("Tuning").run()

    assert not app.exception
    assert any("run scripts/run_*.py" in info.value for info in app.info)
