"""Request/reply client for the segmentation service.

Unlike the marker-detector publisher (latest-payload push), the segmentation
service answers individual requests: the client sends a frame and a text
prompt, the server replies with the detected instances. The connection is
opened lazily on the first request and re-opened after any failure, so
constructing the client (e.g. from a Hydra config) never touches the network
and a service restart does not require a robot-side restart.
"""

import logging
from multiprocessing.connection import Client, Connection
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.segmentation.constants import SEGMENTATION_PORT
from prpl_tidybot.third_party.constants import CONN_AUTHKEY, SERVER_HOSTNAME

_logger = logging.getLogger(__name__)


class SegmentedInstance(TypedDict):
    """One detected instance, reduced to what the servo geometry needs."""

    left_x: float
    right_x: float
    top_y: float
    bottom_y: float
    score: float
    area: int


class SegmentationClient:
    """Lazy single-connection requester for the segmentation server."""

    def __init__(
        self,
        host: str = SERVER_HOSTNAME,
        port: int = SEGMENTATION_PORT,
        timeout_s: float = 5.0,
    ) -> None:
        self._address = (host, port)
        self._timeout_s = timeout_s
        self._conn: Connection | None = None

    def detect(
        self, image: NDArray[np.uint8], prompt: str
    ) -> list[SegmentedInstance] | None:
        """Segment ``prompt`` instances in ``image`` (RGB, HxWx3).

        Returns None when the service is unreachable or does not answer
        within the timeout; the connection is dropped so the next call
        reconnects.
        """
        try:
            if self._conn is None:
                self._conn = Client(self._address, authkey=CONN_AUTHKEY)
            self._conn.send({"image": np.ascontiguousarray(image), "prompt": prompt})
            if not self._conn.poll(timeout=self._timeout_s):
                raise TimeoutError(
                    f"segmentation service did not answer within {self._timeout_s}s"
                )
            reply: dict[str, Any] = self._conn.recv()
        except (OSError, EOFError, TimeoutError) as exc:
            _logger.warning(
                "Segmentation service at %s:%d unavailable (%s).",
                self._address[0],
                self._address[1],
                exc,
            )
            self.close()
            return None
        if "error" in reply:
            _logger.warning("Segmentation service error: %s", reply["error"])
            return None
        return list(reply["instances"])

    def close(self) -> None:
        """Drop the connection; the next request reconnects."""
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
