"""In-process client for a `CameraServer`.

Connects to the camera server's RPC, reads the shared-memory block name, then
hands out zero-copy numpy views of the latest frame. Used by the detector,
which runs in the same Python process as the camera servers.

Adapted from `yixuanhuang98/tidybot_server/server/camera_client.py`.
"""

from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import Client

import numpy as np

from prpl_tidybot.third_party.constants import CONN_AUTHKEY


# https://bugs.python.org/issue38119 — when a worker process opens an existing
# shm block, the resource_tracker will spuriously unlink it on shutdown and
# print warnings. We mask shared_memory from this process's tracker so only
# the CameraServer (the creator) ever unlinks.
def _remove_shm_from_resource_tracker() -> None:
    # pylint: disable=protected-access
    def fix_register(name, rtype):  # type: ignore[no-untyped-def]
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):  # type: ignore[no-untyped-def]
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister

    cleanup_funcs = getattr(resource_tracker, "_CLEANUP_FUNCS", {})
    if "shared_memory" in cleanup_funcs:
        del cleanup_funcs["shared_memory"]


class CameraClient:
    """Zero-copy reader for the shared-memory frame published by a `CameraServer`."""

    def __init__(self, port: int) -> None:
        self.conn = Client(("localhost", port), authkey=CONN_AUTHKEY)
        self.conn.send(None)
        data = self.conn.recv()
        _remove_shm_from_resource_tracker()
        self.shm = shared_memory.SharedMemory(name=data["name"])
        self.image: np.ndarray = np.ndarray(
            data["shape"], dtype=data["dtype"], buffer=self.shm.buf
        )
        self.conn.send(None)

    def get_image(self) -> np.ndarray:
        """Block until the server publishes a new frame, then return the view."""
        self.conn.recv()
        self.conn.send(None)
        return self.image

    def close(self) -> None:
        """Release the shared-memory view; the server still owns the block."""
        self.shm.close()
