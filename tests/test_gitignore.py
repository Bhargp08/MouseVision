from mousevision.config import REPO_ROOT


def test_raw_data_is_ignored():
    text = (REPO_ROOT / ".gitignore").read_text()

    assert "data/raw/" in text
    assert "*.nwb" in text


def test_derived_features_are_not_ignored():
    text = (REPO_ROOT / ".gitignore").read_text()

    assert "data/features" not in text
