"""A numpy-version-independent message format for the perception server.

The pick pipeline and the perception models run in different Python environments
(the models need a recent torch and numpy 2; the planner side is pinned to numpy 1),
so arrays cannot cross the boundary as pickled ``ndarray`` objects: a numpy 2 pickle
references ``numpy._core``, which numpy 1 cannot import. Messages are therefore
plain dicts of builtins, with each array replaced by its dtype, shape and raw bytes,
framed as a 8-byte big-endian length followed by a pickle of that dict.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np

_ARRAY_TAG = "__ndarray__"
_HEADER = struct.Struct(">Q")


def encode(value: Any) -> Any:
    """Replace every ndarray in a nested dict/list structure by a builtin record."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _ARRAY_TAG: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode(value: Any) -> Any:
    """Inverse of :func:`encode`."""
    if isinstance(value, dict):
        if value.get(_ARRAY_TAG):
            return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(
                value["shape"]
            )
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    """Write one framed message."""
    payload = pickle.dumps(encode(message), protocol=4)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    """Read one framed message; raises ConnectionError on a closed socket."""
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    return decode(pickle.loads(_recv_exact(sock, length)))  # type: ignore[no-any-return]


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("perception server connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
