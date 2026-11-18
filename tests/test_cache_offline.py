import socket

from mousevision.allen import make_cache


def test_make_cache_works_offline(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise OSError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    manifest = tmp_path / "m.json"
    cache = make_cache(manifest)

    assert manifest.is_file()
    assert cache.api.api_url == "https://api.brain-map.org"
