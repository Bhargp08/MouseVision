from mousevision import config


def test_cache_dir_honours_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("MV_DATA_DIR", str(tmp_path))

    assert config.cache_dir() == tmp_path / "raw" / "allen_cache"
    assert config.manifest_path() == config.cache_dir() / "manifest.json"
    assert str(config.manifest_path()).endswith("manifest.json")


def test_data_dir_defaults_to_repo_data(monkeypatch):
    monkeypatch.delenv("MV_DATA_DIR", raising=False)

    assert config.data_dir() == config.REPO_ROOT / "data"


def test_allen_base_uri_is_https():
    assert config.ALLEN_BASE_URI.startswith("https://")
