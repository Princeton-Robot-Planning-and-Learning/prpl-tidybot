"""Duplicate perception startup must fail before opening camera devices."""

import socket

import pytest

from prpl_tidybot.marker_detector import server


def test_busy_port_blocks_startup_before_children(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(
            server, "CAMERA_SERVER_PORTS", (listener.getsockname()[1], 0)
        )
        monkeypatch.setattr(server, "MARKER_DETECTOR_PORT", 0)
        monkeypatch.setattr(server, "CEILING_IMAGE_PORT", 0)
        monkeypatch.setattr(
            server,
            "Process",
            lambda **_: pytest.fail("Spawned a camera despite occupied port"),
        )
        with pytest.raises(RuntimeError, match="No new camera processes"):
            server.main()


def test_available_ports_leave_no_listeners(monkeypatch):
    monkeypatch.setattr(server, "CAMERA_SERVER_PORTS", (0, 0))
    monkeypatch.setattr(server, "MARKER_DETECTOR_PORT", 0)
    monkeypatch.setattr(server, "CEILING_IMAGE_PORT", 0)
    server.check_listener_ports("127.0.0.1")
