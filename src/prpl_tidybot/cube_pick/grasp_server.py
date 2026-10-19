"""GPU grasp-detection server for the cube pick: Contact-GraspNet and M2T2.

Runs as a plain script and talks to the pick pipeline over a localhost TCP socket
using the framing in the sibling ``wire.py``. One client connection at a time, many
requests per connection. Every reply carries ``"ok"`` and, on failure, ``"error"``.

Ops:
  ``ping``      -> ``{"ok", "models", "workers"}``
  ``grasps``    -> ``{"ok", "poses", "scores", "widths", "contacts", "convention"}``
  ``release``   stops every model worker, which frees all of its GPU memory
  ``shutdown``  stops the workers and exits

Each model runs in its own worker subprocess (this same file with ``--worker``), in
its own Python: Contact-GraspNet is the official TensorFlow implementation and M2T2
is PyTorch, and TensorFlow never returns GPU memory short of process exit, so
``release`` is implemented by ending the worker. Workers start lazily on the first
request that names them and talk to this process over a socketpair, with the same
``wire`` framing.

Poses are the gripper frame in the camera frame (OpenCV axes), in each model's own
native convention, unmodified: both were trained on the Franka Panda hand with the
approach along +z, the closing direction along +x, and the frame origin 0.1034 m
behind the finger-pad contact line.

Both models are NVIDIA research code under licences that restrict use to
non-commercial research or evaluation. Neither model's code lives in this
repository; each is imported from a local checkout given on the command line.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# A sibling module, imported without installing the package: the server runs as a
# standalone script in its own conda env.
import wire  # type: ignore[import-not-found]  # pylint: disable=wrong-import-position,import-error

# Local checkouts and interpreters for the two models live outside this repository;
# point at them with these environment variables or the matching flags.
_ROOT = os.environ.get("GRASPGEN_ROOT", os.path.expanduser("~/graspgen"))
_DEFAULTS = {
    "m2t2_python": os.environ.get("GRASPGEN_M2T2_PYTHON", "python"),
    "cgn_python": os.environ.get("GRASPGEN_CGN_PYTHON", "python"),
    "m2t2_dir": f"{_ROOT}/M2T2",
    "m2t2_ckpt": f"{_ROOT}/weights/m2t2/m2t2.pth",
    "cgn_dir": f"{_ROOT}/contact_graspnet",
    "cgn_ckpt_dir": (
        f"{_ROOT}/weights/cgn_tf/contact_graspnet_models/"
        "scene_test_2048_bs3_hor_sigma_001"
    ),
}
MODELS = ("contact_graspnet", "m2t2")

# Both models: Panda hand, z = approach, x = closing (from the predicted contact
# toward the opposite finger), origin 0.1034 m behind the finger-pad contact line
# (``gripper_depth`` in each repo's ``build_6d_grasp``); fingertips at 0.1122 m
# (``panda_gripper_coords.yml``); maximum opening 0.08 m.
_PANDA = {
    "gripper": "franka_panda",
    "approach_axis": "z",
    "closing_axis": "x",
    "origin_to_fingertip": 0.1034,
    "origin_to_tip_end": 0.112204,
    "max_width": 0.08,
}


def _log(msg: str) -> None:
    print(f"[grasp_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------
# Request validation (main process; numpy only)
# ---------------------------------------------------------------------------------


class RequestValidator:
    """Coerces a ``grasps`` request into well-typed arrays or raises ValueError."""

    @staticmethod
    def grasps(req: dict[str, Any]) -> dict[str, Any]:
        """Validate a ``grasps`` request and return its cleaned fields."""
        model = req.get("model")
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")
        if req.get("depth") is None or req.get("K") is None:
            raise ValueError("grasps needs 'depth' (HxW float32 metres) and 'K' (3x3)")
        depth = np.asarray(req["depth"], dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth must be HxW, got shape {depth.shape}")
        depth = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
        K = np.asarray(req["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {K.shape}")
        out: dict[str, Any] = {"op": "grasps", "model": model, "depth": depth, "K": K}
        rgb = req.get("rgb")
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.uint8)
            if rgb.shape != (*depth.shape, 3):
                raise ValueError(f"rgb must be {(*depth.shape, 3)}, got {rgb.shape}")
        out["rgb"] = rgb
        segmap = req.get("segmap")
        if segmap is not None:
            segmap = np.asarray(segmap, dtype=np.int32)
            if segmap.shape != depth.shape:
                raise ValueError(f"segmap must be {depth.shape}, got {segmap.shape}")
        out["segmap"] = segmap
        out["target_id"] = int(req.get("target_id", 1))
        if segmap is not None and not np.any(segmap == out["target_id"]):
            raise ValueError(f"segmap has no pixels with target_id {out['target_id']}")
        out["max_grasps"] = int(req.get("max_grasps", 100))
        z_range = [float(z) for z in req.get("z_range", [0.15, 1.8])]
        if len(z_range) != 2 or not z_range[0] < z_range[1]:
            raise ValueError(f"z_range must be [near, far] with near < far: {z_range}")
        out["z_range"] = z_range
        valid = (depth > z_range[0]) & (depth < z_range[1])
        if valid.sum() < 100:
            raise ValueError(f"only {int(valid.sum())} depth pixels inside z_range")
        out["forward_passes"] = max(1, int(req.get("forward_passes", 1)))
        # M2T2-only knobs. M2T2 was trained on world-frame clouds (z up, support
        # surface at z = 0), so it needs the camera pose in such a frame; the string
        # "fit_plane" instead fits the dominant plane, which picks a wall as readily
        # as a floor, so it is opt-in. mask_thresh and surface_range default to
        # M2T2's own eval config (config.yaml: eval.mask_thresh, eval.surface_range).
        T = req.get("T_world_cam")
        if isinstance(T, str):
            if T != "fit_plane":
                raise ValueError(
                    f"T_world_cam must be a 4x4 array or 'fit_plane': {T!r}"
                )
        elif T is not None:
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"T_world_cam must be 4x4, got {T.shape}")
        elif model == "m2t2":
            raise ValueError(
                "m2t2 needs 'T_world_cam' (4x4 world-from-camera, world z up with the "
                "support surface at z = 0), or 'fit_plane' to estimate it"
            )
        out["T_world_cam"] = T
        out["mask_thresh"] = float(req.get("mask_thresh", 0.4))
        out["surface_range"] = float(req.get("surface_range", 0.02))
        out["seed"] = int(req.get("seed", 0))
        return out


# ---------------------------------------------------------------------------------
# Worker management (main process)
# ---------------------------------------------------------------------------------


class Worker:
    """One model's subprocess, reached over a socketpair."""

    def __init__(self, *, model: str, args: argparse.Namespace) -> None:
        self.model = model
        parent, child = socket.socketpair()
        python = args.cgn_python if model == "contact_graspnet" else args.m2t2_python
        cmd = [
            python,
            os.path.abspath(__file__),
            "--worker",
            model,
            "--worker-fd",
            str(child.fileno()),
            "--m2t2-dir",
            args.m2t2_dir,
            "--m2t2-ckpt",
            args.m2t2_ckpt,
            "--cgn-dir",
            args.cgn_dir,
            "--cgn-ckpt-dir",
            args.cgn_ckpt_dir,
        ]
        env = dict(os.environ)
        if model == "contact_graspnet":
            # The checkout's tf.compat.v1 layers need Keras 2 under TF 2.20.
            env.setdefault("TF_USE_LEGACY_KERAS", "1")
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            # TF 2.20 wheels carry no sm_120 code for their MLIR-generated elementwise
            # kernels (cuLaunchKernel -> CUDA_ERROR_INVALID_HANDLE on the RTX 5090), so
            # every op XLA can take is compiled by XLA, which targets sm_120 itself.
            env.setdefault(
                "TF_XLA_FLAGS", "--tf_xla_auto_jit=2 --tf_xla_min_cluster_size=1"
            )
            # The PointNet++ custom ops launch on the legacy default stream, not TF's
            # stream; without serialised launches the second inference reads
            # unfinished buffers and every score comes back NaN.
            env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
            env.setdefault("PYOPENGL_PLATFORM", "egl")
        self.proc = subprocess.Popen(cmd, pass_fds=[child.fileno()], env=env)
        child.close()
        self.sock = parent
        self.sock.settimeout(args.worker_timeout)
        ready = wire.recv_message(self.sock)
        if not ready.get("ok"):
            self.stop()
            raise RuntimeError(f"{model} worker failed to start: {ready.get('error')}")
        _log(f"{model} worker ready (pid {self.proc.pid}, load {ready['load_s']:.1f}s)")

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        """Send one request to the worker and return its reply."""
        wire.send_message(self.sock, req)
        return wire.recv_message(self.sock)

    def alive(self) -> bool:
        """Whether the worker process is still running."""
        return self.proc.poll() is None

    def stop(self) -> None:
        """Shut the worker down, killing it if it does not exit."""
        try:
            if self.alive():
                wire.send_message(self.sock, {"op": "shutdown"})
                self.proc.wait(timeout=20)
        except Exception:  # pylint: disable=broad-exception-caught
            # The process is killed below regardless.
            pass
        if self.alive():
            self.proc.kill()
            self.proc.wait(timeout=20)
        self.sock.close()


def _gpu_mem_mib(pid: int) -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:  # pylint: disable=broad-exception-caught
        # Memory reporting is best effort.
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return None


class GraspServer:
    """Dispatches requests to lazily started per-model workers."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workers: dict[str, Worker] = {}

    def worker(self, model: str) -> Worker:
        """The running worker for a model, starting or restarting it."""
        w = self.workers.get(model)
        if w is not None and not w.alive():
            _log(f"{model} worker died (exit {w.proc.returncode}); restarting")
            w.stop()
            w = None
        if w is None:
            w = Worker(model=model, args=self.args)
            self.workers[model] = w
        return w

    def release(self) -> None:
        """Stop every worker, freeing its GPU memory."""
        for w in self.workers.values():
            w.stop()
        self.workers.clear()

    def handle(self, req: Any) -> dict[str, Any]:
        """Answer one request, raising on an invalid one."""
        if not isinstance(req, dict):
            raise ValueError("request must be a dict")
        op = req.get("op")
        if op == "ping":
            return {
                "ok": True,
                "models": list(MODELS),
                "workers": {
                    m: {"pid": w.proc.pid, "gpu_mem_mib": _gpu_mem_mib(w.proc.pid)}
                    for m, w in self.workers.items()
                    if w.alive()
                },
            }
        if op == "release":
            self.release()
            return {"ok": True}
        if op == "grasps":
            clean = RequestValidator.grasps(req)
            w = self.worker(clean["model"])
            try:
                reply = w.call(clean)
            except (socket.timeout, ConnectionError, OSError) as exc:
                w.stop()
                self.workers.pop(clean["model"], None)
                raise RuntimeError(f"{clean['model']} worker lost: {exc!r}") from exc
            if not reply.get("ok"):
                return reply
            return GraspServer._sorted(reply, clean["max_grasps"])
        raise ValueError(f"unknown op {op!r}")

    @staticmethod
    def _sorted(reply: dict[str, Any], max_grasps: int) -> dict[str, Any]:
        scores = np.asarray(reply["scores"], dtype=np.float64).reshape(-1)
        order = np.argsort(-scores, kind="stable")[: max(0, max_grasps)]
        poses = np.asarray(reply["poses"], dtype=np.float64).reshape(-1, 4, 4)[order]
        widths = np.asarray(reply["widths"], dtype=np.float64).reshape(-1)[order]
        contacts = np.asarray(reply["contacts"], dtype=np.float64).reshape(-1, 3)
        contacts = contacts[order] if len(contacts) == len(scores) else contacts[:0]
        out = dict(reply)
        out.update(
            poses=poses,
            scores=scores[order],
            widths=widths,
            contacts=contacts,
            num_raw=int(len(scores)),
        )
        return out

    def serve(self) -> None:
        """Accept clients one at a time until a shutdown request."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.args.host, self.args.port))
        srv.listen(1)
        _log(f"listening on {self.args.host}:{self.args.port}")
        try:
            while True:
                conn, addr = srv.accept()
                _log(f"client connected from {addr}")
                if self._serve_connection(conn):
                    break
        finally:
            self.release()
            srv.close()
            _log("shut down")

    def _serve_connection(self, conn: socket.socket) -> bool:
        """Serve one client; returns True on a shutdown request."""
        with conn:
            while True:
                try:
                    req = wire.recv_message(conn)
                except ConnectionError:
                    _log("client disconnected")
                    return False
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # An unreadable frame: resync is impossible.
                    _log(f"bad frame, dropping connection: {exc!r}")
                    try:
                        wire.send_message(
                            conn, {"ok": False, "error": f"bad frame: {exc!r}"}
                        )
                    except OSError:
                        pass
                    return False
                t0 = time.perf_counter()
                op = req.get("op") if isinstance(req, dict) else None
                try:
                    if op == "shutdown":
                        reply: dict[str, Any] = {"ok": True}
                    else:
                        reply = self.handle(req)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # Never crash on a bad request.
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                dt = (time.perf_counter() - t0) * 1000
                detail = ""
                if op == "grasps":
                    detail = f" model={req.get('model')}"
                    if reply.get("ok"):
                        detail += f" n={len(reply['scores'])}/{reply.get('num_raw')}"
                status = "ok" if reply.get("ok") else f"error={reply.get('error')}"
                _log(f"op={op}{detail} {status} {dt:.0f}ms")
                try:
                    wire.send_message(conn, reply)
                except OSError:
                    _log("client vanished before the reply")
                    return op == "shutdown"
                if op == "shutdown":
                    return True


# ---------------------------------------------------------------------------------
# Model backends (worker processes)
# ---------------------------------------------------------------------------------


class DepthCloud:
    """Back-projection shared by both backends (OpenCV camera axes)."""

    @staticmethod
    def points(
        *, depth: np.ndarray, K: np.ndarray, z_range: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (Nx3 camera-frame points, N flat pixel indices)."""
        v, u = np.nonzero((depth > z_range[0]) & (depth < z_range[1]))
        z = depth[v, u].astype(np.float64)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.stack([x, y, z], axis=1), v * depth.shape[1] + u


class ContactGraspNetBackend:
    """Official TF Contact-GraspNet (NVlabs), used through its own GraspEstimator."""

    def __init__(self, *, repo: str, ckpt_dir: str) -> None:
        # Worker-only dependencies, imported in the model's own environment.
        import yaml  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

        # The checkout calls yaml.load(f) with no Loader, which PyYAML >= 6 rejects.
        _orig_load = yaml.load
        yaml.load = lambda stream, Loader=yaml.FullLoader: _orig_load(stream, Loader)  # type: ignore[assignment]  # noqa: E501

        import tensorflow.compat.v1 as tf  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel,import-error

        tf.disable_eager_execution()
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        sys.path.insert(0, os.path.join(repo, "contact_graspnet"))
        import config_utils  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error
        from contact_grasp_estimator import (  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error
            GraspEstimator,
        )

        self.tf = tf
        self.cfg = config_utils.load_config(ckpt_dir, batch_size=1)
        self.estimator = GraspEstimator(self.cfg)
        self.estimator.build_network()
        saver = tf.train.Saver(save_relative_paths=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.estimator.load_weights(self.sess, saver, ckpt_dir, mode="test")
        self.ckpt = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path

    def grasps(self, req: dict[str, Any]) -> dict[str, Any]:
        """Contact-GraspNet grasps for one validated request."""
        segmap = req["segmap"]
        target = req["target_id"]
        use_seg = segmap is not None
        depth = np.array(req["depth"], dtype=np.float32)
        pc_full, pc_segments, _ = self.estimator.extract_point_clouds(
            depth,
            req["K"],
            segmap=segmap,
            segmap_id=target if use_seg else 0,
            z_range=req["z_range"],
        )
        if use_seg and len(pc_segments.get(target, [])) == 0:
            raise ValueError("target segment has no points inside z_range")
        poses, scores, contacts, widths = self.estimator.predict_scene_grasps(
            self.sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=use_seg,
            filter_grasps=use_seg,
            forward_passes=req["forward_passes"],
        )
        key = target if use_seg else -1
        return {
            "ok": True,
            "poses": np.asarray(poses.get(key, np.zeros((0, 4, 4)))).reshape(-1, 4, 4),
            "scores": np.asarray(scores.get(key, np.zeros(0))).reshape(-1),
            "widths": np.asarray(widths.get(key, np.zeros(0))).reshape(-1),
            "contacts": np.asarray(contacts.get(key, np.zeros((0, 3)))).reshape(-1, 3),
            "convention": dict(_PANDA),
            "info": {
                "implementation": "NVlabs/contact_graspnet (TensorFlow)",
                "checkpoint": self.ckpt,
                "local_regions": use_seg,
                "filter_grasps": use_seg,
            },
        }


class M2T2Backend:
    """M2T2 (NVlabs) with its released generic pick-and-place checkpoint."""

    def __init__(self, *, repo: str, ckpt: str) -> None:
        # Worker-only dependencies, imported in the model's own environment.
        import torch  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

        sys.path.insert(0, repo)
        # pylint: disable-next=import-outside-toplevel,import-error
        from m2t2.dataset import collate  # type: ignore[import-not-found]
        from m2t2.dataset_utils import (  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error
            normalize_rgb,
            sample_points,
        )

        # pylint: disable-next=import-outside-toplevel,import-error
        from m2t2.m2t2 import M2T2  # type: ignore[import-not-found]

        self.torch = torch
        self.collate, self.normalize_rgb, self.sample_points = (
            collate,
            normalize_rgb,
            sample_points,
        )
        self.cfg = OmegaConf.load(os.path.join(repo, "config.yaml"))
        model = M2T2.from_config(self.cfg.m2t2)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        self.model = model.cuda().eval()
        self.ckpt = ckpt

    @staticmethod
    def plane_frame(*, pts: np.ndarray, seed: int) -> np.ndarray:
        """World-from-camera for the dominant plane (z up toward the camera, z=0)."""
        rng = np.random.default_rng(seed)
        sub = pts[rng.choice(len(pts), size=min(len(pts), 20000), replace=False)]
        best, best_n, best_d = -1, None, 0.0
        for _ in range(300):
            a, b, c = sub[rng.choice(len(sub), size=3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            d = -n @ a
            count = int((np.abs(sub @ n + d) < 0.005).sum())
            if count > best:
                best, best_n, best_d = count, n, d
        if best_n is None:
            raise ValueError("could not fit a support plane")
        n, d = best_n, best_d
        if d < 0:  # orient the normal toward the camera (the camera is at the origin)
            n, d = -n, -d
        x = np.array([0.0, 0.0, 1.0]) - n * n[2]
        x = (
            x / np.linalg.norm(x)
            if np.linalg.norm(x) > 1e-6
            else np.cross(n, [1, 0, 0])
        )
        y = np.cross(n, x)
        R_cw = np.stack([x, y, n])  # rows: world axes in camera coordinates
        T = np.eye(4)
        T[:3, :3] = R_cw
        T[:3, 3] = np.array([0.0, 0.0, d])  # camera height above the plane
        return T

    def grasps(self, req: dict[str, Any]) -> dict[str, Any]:
        """M2T2 grasps for one validated request, in the camera frame."""
        torch = self.torch
        pts_cam, pix = DepthCloud.points(
            depth=req["depth"], K=req["K"], z_range=req["z_range"]
        )
        T_wc = req["T_world_cam"]
        frame = "given"
        if isinstance(T_wc, str):
            T_wc = M2T2Backend.plane_frame(pts=pts_cam, seed=req["seed"])
            frame = "ransac_plane"
        xyz = pts_cam @ T_wc[:3, :3].T + T_wc[:3, 3]
        if req["surface_range"] > 0:
            xyz[np.abs(xyz[:, 2]) < req["surface_range"], 2] = 0
        xyz_t = torch.from_numpy(xyz).float()
        if req["rgb"] is not None:
            rgb_img = self.normalize_rgb(np.ascontiguousarray(req["rgb"])).permute(
                1, 2, 0
            )
            rgb = rgb_img.reshape(-1, 3)[torch.from_numpy(pix)]
        else:
            rgb = torch.zeros(len(xyz), 3)
        seg = None if req["segmap"] is None else req["segmap"].reshape(-1)[pix]
        inputs = torch.cat([xyz_t - xyz_t.mean(dim=0), rgb], dim=1)
        eval_cfg = self.cfg.eval.copy()
        eval_cfg.mask_thresh = req["mask_thresh"]
        eval_cfg.world_coord = True
        torch.manual_seed(req["seed"])
        g_all, c_all, p_all, obj_all = [], [], [], []
        for _ in range(req["forward_passes"]):
            idx = self.sample_points(xyz_t, self.cfg.data.num_points)
            data = {
                "inputs": inputs[idx],
                "points": xyz_t[idx],
                "seg": torch.zeros(len(idx)),
                "object_inputs": torch.rand(self.cfg.data.num_object_points, 6),
                "ee_pose": torch.eye(4),
                "bottom_center": torch.zeros(3),
                "object_center": torch.zeros(3),
                "cam_pose": torch.eye(4),
                "task": "pick",
            }
            batch = self.collate([data])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            with torch.no_grad():
                out = self.model.infer(batch, eval_cfg)
            for i, (g, c, p) in enumerate(
                zip(
                    out["grasps"][0],
                    out["grasp_confidence"][0],
                    out["grasp_contacts"][0],
                )
            ):
                g_all.append(g.double().cpu().numpy())
                c_all.append(c.double().cpu().numpy())
                p_all.append(p.double().cpu().numpy())
                obj_all.append(np.full(len(c), i))
        poses_w = np.concatenate(g_all) if g_all else np.zeros((0, 4, 4))
        scores = np.concatenate(c_all) if c_all else np.zeros(0)
        contacts_w = np.concatenate(p_all) if p_all else np.zeros((0, 3))
        objects = np.concatenate(obj_all) if obj_all else np.zeros(0, int)
        if seg is not None and len(contacts_w):
            # Contacts are copies of input points; recover each one's pixel label.
            full = torch.from_numpy(xyz).float().cuda()
            labels = []
            for chunk in torch.from_numpy(contacts_w).float().cuda().split(4096):
                labels.append(torch.cdist(chunk, full).argmin(dim=1).cpu().numpy())
            keep = seg[np.concatenate(labels)] == req["target_id"]
            poses_w, scores = poses_w[keep], scores[keep]
            contacts_w, objects = contacts_w[keep], objects[keep]
        T_cw = np.linalg.inv(T_wc)
        poses = T_cw @ poses_w
        contacts = contacts_w @ T_cw[:3, :3].T + T_cw[:3, 3]
        # build_6d_grasp puts the origin at contact + x*width/2 - 0.1034*z, so the
        # predicted opening is recoverable exactly as 2 * x . (origin - contact).
        widths = 2 * np.einsum("ni,ni->n", poses[:, :3, 0], poses[:, :3, 3] - contacts)
        return {
            "ok": True,
            "poses": poses,
            "scores": scores,
            "widths": widths,
            "contacts": contacts,
            "convention": dict(_PANDA),
            "info": {
                "implementation": "NVlabs/M2T2 (PyTorch)",
                "checkpoint": self.ckpt,
                "frame": frame,
                "T_world_cam": T_wc,
                "object_query": objects,
                "mask_thresh": req["mask_thresh"],
                "uses_rgb": bool(self.cfg.m2t2.scene_encoder.use_rgb),  # noqa: E501
            },
        }


def run_worker(args: argparse.Namespace) -> None:
    """Load one model and answer requests over the parent's socket."""
    sock = socket.socket(fileno=args.worker_fd)
    t0 = time.perf_counter()
    try:
        if args.worker == "contact_graspnet":
            backend: Any = ContactGraspNetBackend(
                repo=args.cgn_dir, ckpt_dir=args.cgn_ckpt_dir
            )
        else:
            backend = M2T2Backend(repo=args.m2t2_dir, ckpt=args.m2t2_ckpt)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Reported to the parent, then exit.
        traceback.print_exc()
        wire.send_message(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    wire.send_message(sock, {"ok": True, "load_s": time.perf_counter() - t0})
    while True:
        try:
            req = wire.recv_message(sock)
        except ConnectionError:
            return
        if req.get("op") == "shutdown":
            return
        try:
            reply = backend.grasps(req)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Never crash on a bad request.
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        wire.send_message(sock, reply)


def main() -> None:
    """Parse the command line and run the server or a worker."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50112)
    p.add_argument("--m2t2-python", default=_DEFAULTS["m2t2_python"])
    p.add_argument("--cgn-python", default=_DEFAULTS["cgn_python"])
    p.add_argument("--m2t2-dir", default=_DEFAULTS["m2t2_dir"])
    p.add_argument("--m2t2-ckpt", default=_DEFAULTS["m2t2_ckpt"])
    p.add_argument("--cgn-dir", default=_DEFAULTS["cgn_dir"])
    p.add_argument("--cgn-ckpt-dir", default=_DEFAULTS["cgn_ckpt_dir"])
    p.add_argument(
        "--worker-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for a worker to load or answer",
    )
    p.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    p.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        run_worker(args)
        return
    GraspServer(args).serve()


if __name__ == "__main__":
    main()
