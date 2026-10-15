"""GPU perception server for the cube pick: SAM 2 segmentation + FoundationPose.

Runs as a plain script in its own Python environment (recent torch, numpy 2) and talks
to the pick pipeline over a localhost TCP socket using the framing in the sibling
``wire.py``. One client connection at a time, many requests per connection. Models
are loaded lazily on first use and can be dropped with ``{"op": "release"}``.

Requests are dicts with an ``"op"`` key; every reply carries ``"ok"`` and, on
failure, ``"error"``. Ops: ``ping``, ``segment``, ``fp_register``, ``release``,
``shutdown``.

FoundationPose (NVlabs) is used under the NVIDIA Source Code License, which restricts
use to non-commercial research or evaluation. It is imported from a local checkout
(``--fp-dir``); none of its code lives in this repository.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import socket
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402  (sibling module, imported without installing the package)

_CUBEPOSE_ROOT = "/home/josh/Documents/repos/research/cubepose"
_DEFAULT_FP_DIR = os.environ.get("CUBEPOSE_FP_DIR", f"{_CUBEPOSE_ROOT}/FoundationPose")
_DEFAULT_SAM2_CKPT = os.environ.get(
    "CUBEPOSE_SAM2_CKPT", f"{_CUBEPOSE_ROOT}/checkpoints/sam2.1_hiera_small.pt"
)
_DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
_KNOWN_BACKENDS = ("sam2", "foundationpose")


def _log(msg: str) -> None:
    print(f"[perception_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Sam2Backend:
    """Point/box prompted SAM 2 image predictor."""

    def __init__(self, *, checkpoint: str, config: str) -> None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._torch = torch
        model = build_sam2(config, checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(model)

    def segment(self, req: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        rgb = np.ascontiguousarray(np.asarray(req["rgb"], dtype=np.uint8))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3 uint8, got shape {rgb.shape}")
        points = req.get("points")
        labels = req.get("point_labels")
        box = req.get("box")
        if points is not None:
            points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            labels = (
                np.ones(len(points), dtype=np.int32)
                if labels is None
                else np.asarray(labels, dtype=np.int32).reshape(-1)
            )
            if len(labels) != len(points):
                raise ValueError("point_labels must have one entry per point")
        else:
            labels = None
        if box is not None:
            box = np.asarray(box, dtype=np.float32).reshape(4)
        if points is None and box is None:
            raise ValueError("segment needs at least one of points or box")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=bool(req.get("multimask", True)),
            )
        order = np.argsort(-scores)
        return {
            "ok": True,
            "masks": np.asarray(masks[order] > 0.5, dtype=bool),
            "scores": np.asarray(scores[order], dtype=np.float32),
        }


class FoundationPoseBackend:
    """Model-based FoundationPose registration of a uniformly coloured cube."""

    def __init__(self, *, fp_dir: str) -> None:
        if fp_dir not in sys.path:
            sys.path.insert(0, fp_dir)
        import estimater  # FoundationPose's own module, from the local checkout
        import nvdiffrast.torch as dr
        import torch

        self._torch = torch
        self._fp = estimater
        self.scorer = estimater.ScorePredictor()
        self.refiner = estimater.PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.debug_dir = tempfile.mkdtemp(prefix="fp_debug_")
        self.estimators: dict[tuple[str, float], Any] = {}
        self.cube_symmetries = self._cube_rotation_group()

    @staticmethod
    def _cube_rotation_group() -> np.ndarray:
        """The 24 proper rotations mapping an axis-aligned cube to itself, as 4x4."""
        tfs = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                R = np.zeros((3, 3))
                for row, (col, sgn) in enumerate(zip(perm, signs)):
                    R[row, col] = sgn
                if np.linalg.det(R) > 0:
                    T = np.eye(4)
                    T[:3, :3] = R
                    tfs.append(T)
        return np.stack(tfs)

    @staticmethod
    def _cube_mesh(*, size: float, rgba: list[float]) -> Any:
        import trimesh

        box = trimesh.creation.box(extents=[size, size, size])
        box.unmerge_vertices()  # per-face vertices, so normals stay flat on each face
        verts, faces = trimesh.remesh.subdivide_to_size(
            box.vertices, box.faces, max_edge=size / 8
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        color = (np.clip(np.asarray(rgba, dtype=float).reshape(4), 0, 1) * 255).astype(
            np.uint8
        )
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, vertex_colors=np.tile(color, (len(mesh.vertices), 1))
        )
        return mesh

    def _estimator(self, *, key: str, size: float, rgba: list[float]) -> Any:
        cache_key = (key, round(float(size), 6))
        if cache_key not in self.estimators:
            mesh = self._cube_mesh(size=size, rgba=rgba)
            self.estimators[cache_key] = self._fp.FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                symmetry_tfs=self.cube_symmetries,
                mesh=mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=self.glctx,
                debug=0,
                debug_dir=self.debug_dir,
            )
        return self.estimators[cache_key]

    def register(self, req: dict[str, Any]) -> dict[str, Any]:
        rgb = np.array(req["rgb"], dtype=np.uint8)
        depth = np.array(req["depth"], dtype=np.float32)
        K = np.array(req["K"], dtype=np.float64).reshape(3, 3)
        mask = np.array(req["mask"], dtype=bool)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3 uint8, got shape {rgb.shape}")
        if depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"rgb {rgb.shape}, depth {depth.shape} and mask {mask.shape} "
                "must share one HxW"
            )
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        if int((mask & (depth > 0.001)).sum()) < 4:
            raise ValueError("mask contains fewer than 4 pixels with valid depth")
        size = float(req["cube_size"])
        if not size > 0:
            raise ValueError("cube_size must be positive")
        est = self._estimator(
            key=str(req.get("mesh_key", "cube")),
            size=size,
            rgba=req.get("rgba", [0.5, 0.5, 0.5, 1.0]),
        )
        with self._torch.inference_mode():
            pose = est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=int(req.get("iterations", 5)),
            )
        return {
            "ok": True,
            "pose": np.asarray(pose, dtype=np.float64).reshape(4, 4),
            "score": float(est.scores[0]),
        }


class PerceptionServer:
    def __init__(self, *, backends: list[str], args: argparse.Namespace) -> None:
        unknown = sorted(set(backends) - set(_KNOWN_BACKENDS))
        if unknown:
            raise SystemExit(f"unknown backends {unknown}; known: {_KNOWN_BACKENDS}")
        self.enabled = backends
        self.args = args
        self.sam2: Sam2Backend | None = None
        self.fp: FoundationPoseBackend | None = None
        self.running = True

    def _need(self, name: str) -> None:
        if name not in self.enabled:
            raise RuntimeError(f"backend {name!r} not enabled (--backends)")

    def _sam2(self) -> Sam2Backend:
        self._need("sam2")
        if self.sam2 is None:
            t0 = time.perf_counter()
            self.sam2 = Sam2Backend(
                checkpoint=self.args.sam2_checkpoint, config=self.args.sam2_config
            )
            _log(f"loaded sam2 in {time.perf_counter() - t0:.2f}s")
        return self.sam2

    def _fpose(self) -> FoundationPoseBackend:
        self._need("foundationpose")
        if self.fp is None:
            t0 = time.perf_counter()
            self.fp = FoundationPoseBackend(fp_dir=self.args.fp_dir)
            _log(f"loaded foundationpose in {time.perf_counter() - t0:.2f}s")
        return self.fp

    def _release(self) -> None:
        self.sam2 = None
        self.fp = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _gpu_mb(self) -> float:
        if "torch" not in sys.modules:
            return 0.0
        import torch

        return (
            torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        )

    def handle(self, req: Any) -> dict[str, Any]:
        if not isinstance(req, dict):
            raise ValueError("request must be a dict")
        op = req.get("op")
        if op == "ping":
            loaded = [
                n for n, b in (("sam2", self.sam2), ("foundationpose", self.fp)) if b
            ]
            return {"ok": True, "backends": list(self.enabled), "loaded": loaded}
        if op == "segment":
            return self._sam2().segment(req)
        if op == "fp_register":
            return self._fpose().register(req)
        if op == "release":
            self._release()
            return {"ok": True}
        if op == "shutdown":
            self.running = False
            return {"ok": True}
        raise ValueError(f"unknown op {op!r}")

    def serve_connection(self, conn: socket.socket) -> None:
        while self.running:
            try:
                req = wire.recv_message(conn)
            except ConnectionError:
                return
            except Exception as exc:  # malformed frame: the stream is unrecoverable
                _log(f"bad frame, dropping connection: {exc!r}")
                return
            op = req.get("op") if isinstance(req, dict) else None
            t0 = time.perf_counter()
            try:
                reply = self.handle(req)
            except Exception as exc:
                traceback.print_exc()
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            dt = (time.perf_counter() - t0) * 1000
            _log(
                f"op={op} ok={reply.get('ok')} {dt:.1f} ms gpu_alloc={self._gpu_mb():.0f} MiB"
                + ("" if reply.get("ok") else f" error={reply.get('error')}")
            )
            try:
                wire.send_message(conn, reply)
            except (ConnectionError, OSError):
                return

    def serve(self, *, host: str, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            _log(f"listening on {host}:{port}, backends={self.enabled}")
            while self.running:
                conn, addr = srv.accept()
                _log(f"client connected from {addr}")
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.serve_connection(conn)
                _log("client disconnected")
        self._release()
        _log("shut down")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50111)
    parser.add_argument("--backends", default="sam2,foundationpose")
    parser.add_argument("--fp-dir", default=_DEFAULT_FP_DIR)
    parser.add_argument("--sam2-checkpoint", default=_DEFAULT_SAM2_CKPT)
    parser.add_argument("--sam2-config", default=_DEFAULT_SAM2_CFG)
    args = parser.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    PerceptionServer(backends=backends, args=args).serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
