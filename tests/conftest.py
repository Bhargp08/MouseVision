import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(request):
    """Fail loudly if a non-slow test tries to open a socket."""
    if request.node.get_closest_marker("slow") is not None:
        yield
        return

    original = socket.socket.connect

    def blocked(*args, **kwargs):
        raise OSError("network access is blocked in tests")

    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = original
