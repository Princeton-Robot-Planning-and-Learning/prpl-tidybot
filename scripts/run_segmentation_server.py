"""SAM 3 segmentation service for the perception laptop (issue #136).

Loads Meta's SAM 3 (text-promptable concept segmentation) onto the laptop's
GPU and answers requests over an authenticated multiprocessing connection:

    request:  {"image": HxWx3 uint8 RGB ndarray, "prompt": str}
    reply:    {"instances": [{"left_x", "right_x", "top_y", "bottom_y",
                              "score", "area"}, ...]}
              or {"error": str}

Run it on the laptop with the dedicated torch venv (NOT the repo venv):

    ~/sam3-venv/bin/python scripts/run_segmentation_server.py

The script is standalone on purpose — no prpl_tidybot imports — so the torch
venv needs only torch/transformers/numpy. The auth key mirrors
``prpl_tidybot.third_party.constants.CONN_AUTHKEY``; keep them in sync (or
pass ``--authkey``). The accept loop survives clients that fail the auth
handshake (see issue #89: a bare port probe must not kill the service).
"""

import argparse
import logging
import time
from multiprocessing.connection import Connection, Listener
from threading import Lock, Thread
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("segmentation_server")


class SamService:
    """SAM 3 wrapped for single-frame text-prompted instance queries."""

    def __init__(self, model_name: str, threshold: float) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self._threshold = threshold
        start = time.time()
        self._processor = Sam3Processor.from_pretrained(model_name)
        self._model = (
            Sam3Model.from_pretrained(model_name, dtype=torch.bfloat16)
            .to("cuda")
            .eval()
        )
        # Inference must be serialized across connection threads.
        self._lock = Lock()
        _logger.info("Loaded %s in %.1fs", model_name, time.time() - start)

    def segment(self, image: np.ndarray, prompt: str) -> list[dict[str, Any]]:
        """Per-instance column/row extents for ``prompt`` in ``image`` (RGB)."""
        height, width = image.shape[:2]
        with self._lock:
            start = time.time()
            inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
                "cuda"
            )
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            result = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._threshold,
                mask_threshold=0.5,
                target_sizes=[(height, width)],
            )[0]
        instances: list[dict[str, Any]] = []
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is not None:
            for i in range(len(masks)):
                mask = masks[i].cpu().numpy() > 0
                cols = np.where(mask.any(axis=0))[0]
                rows = np.where(mask.any(axis=1))[0]
                if len(cols) == 0:
                    continue
                instances.append(
                    {
                        "left_x": float(cols.min()),
                        "right_x": float(cols.max()),
                        "top_y": float(rows.min()),
                        "bottom_y": float(rows.max()),
                        "score": float(scores[i]) if scores is not None else 1.0,
                        "area": int(mask.sum()),
                    }
                )
        _logger.info(
            "prompt=%r -> %d instance(s) in %.0f ms",
            prompt,
            len(instances),
            1000 * (time.time() - start),
        )
        return instances


def _handle(conn: Connection, service: SamService) -> None:
    try:
        while True:
            request = conn.recv()
            try:
                instances = service.segment(
                    np.asarray(request["image"], dtype=np.uint8),
                    str(request["prompt"]),
                )
                conn.send({"instances": instances})
            except Exception as exc:  # pylint: disable=broad-except
                _logger.exception("segment failed")
                conn.send({"error": f"{type(exc).__name__}: {exc}"})
    except (ConnectionResetError, EOFError, BrokenPipeError):
        pass
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6004)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument(
        "--authkey",
        default="secret password",
        help="must match prpl_tidybot.third_party.constants.CONN_AUTHKEY",
    )
    args = parser.parse_args()

    service = SamService(args.model, args.threshold)
    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    _logger.info("Serving on %s:%d", args.host, args.port)
    while True:
        # A client that fails the auth handshake (or a bare port probe) raises
        # here; the service must keep accepting.
        try:
            conn = listener.accept()
        except Exception:  # pylint: disable=broad-except
            _logger.warning("Rejected a connection (bad handshake?)", exc_info=True)
            continue
        _logger.info("Client connected")
        Thread(target=_handle, args=(conn, service), daemon=True).start()


if __name__ == "__main__":
    main()
