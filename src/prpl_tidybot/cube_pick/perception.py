"""Cube pose estimation from one wrist RGB-D frame, behind one interface.

Two backends, both starting from the same mask:

* :class:`DepthFitEstimator` ("SAM 2 + depth fit"): a segmenter masks the cube in the
  colour image; the masked, registered depth points are fitted with a cube of known
  size (:mod:`prpl_tidybot.cube_pick.cube_fit`).
* :class:`FoundationPoseEstimator`: the same mask and the RGB-D frame go to NVIDIA's
  FoundationPose, which registers a mesh of the known cube.

Segmenters: :class:`Sam2Segmenter` (Meta's SAM 2, prompted with the projection of the
pick's prior belief about where the cube is) and :class:`ColourSegmenter`, a
colour-threshold baseline. The learned models run in their own Python environment
behind :class:`PerceptionClient` (a TCP socket; `perception_server.py`), which is also
how the real robot would reach a GPU on the perception PC.

An estimator only ever sees an :class:`~prpl_tidybot.cube_pick.camera.RGBDFrame` and
a :class:`CubePrior`; it has no access to simulator state.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_pick.camera import RGBDFrame
from prpl_tidybot.cube_pick.cube_fit import (
    CubeFitError,
    SupportSurface,
    as_supports,
    cube_yaw_and_up,
    fit_cube,
    height_above_supports,
)
from prpl_tidybot.cube_pick.wire import recv_message, send_message

FloatArray = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class CubePrior:
    """What the pick believes before it looks: roughly where the cube is.

    On the robot this comes from the task-level state estimate (the ceiling marker
    detector, or the previous estimate); `sigma` is how far off it may be.
    """

    position: FloatArray
    size: float
    sigma: float = 0.05
    rgba: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 1.0)
    # Heights of horizontal support surfaces near the cube, from the map (the floor,
    # a container's floor): see `fit_cube`.
    support_heights: tuple[float | SupportSurface, ...] = (0.0,)


@dataclass(frozen=True)
class CubeEstimate:
    """A perceived cube pose."""

    world_from_cube: FloatArray
    yaw: float
    up_alignment: float
    size: float
    backend: str
    mask: Mask | None = None
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> FloatArray:
        """Cube centre in the world frame."""
        return self.world_from_cube[:3, 3]


@dataclass(frozen=True)
class PerceptionFailure:
    """Why no estimate was produced."""

    reason: str
    mask: Mask | None = None


class Segmenter(Protocol):
    """Masks the cube in a colour image."""

    name: str

    def segment(self, frame: RGBDFrame, prior: CubePrior) -> Mask | PerceptionFailure:
        """A boolean mask of the cube, or why there is none."""


class CubePoseEstimator(Protocol):
    """Estimates the cube's pose from one frame."""

    name: str

    def estimate(
        self, frame: RGBDFrame, prior: CubePrior
    ) -> CubeEstimate | PerceptionFailure:
        """The cube's pose, or why there is none."""


def prior_prompt(
    frame: RGBDFrame, prior: CubePrior
) -> tuple[FloatArray, FloatArray, float] | None:
    """The prior's projection: pixel centre, a box covering its uncertainty, and the
    expected pixel area of the cube.

    None if it is behind the camera or out of view.
    """
    uv, z = frame.project_world(prior.position[None])
    if z[0] <= 0.05:
        return None
    intr = frame.intrinsics
    centre = uv[0]
    radius_px = (prior.size * 0.9 + prior.sigma) * intr.fx / z[0]
    box = np.array(
        [
            centre[0] - radius_px,
            centre[1] - radius_px,
            centre[0] + radius_px,
            centre[1] + radius_px,
        ]
    )
    box = np.clip(
        box, 0, [intr.width - 1, intr.height - 1, intr.width - 1, intr.height - 1]
    )
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    area = (prior.size * intr.fx / z[0]) ** 2
    return centre, box, float(area)


def depth_prompt(
    frame: RGBDFrame, prior: CubePrior
) -> tuple[FloatArray, FloatArray] | None:
    """Where the cube is in the image, from depth alone: the compact blob, inside the
    prior's window, of points standing 0.3-1.3 cube edges above a known support.

    A noisy prior projects next to the cube, and a point prompt there segments the floor
    beside it; the cube's height above its support is a cue the robot has without any
    learned model. Returns a point prompt and a box around the blob, or None.
    """
    prompt = prior_prompt(frame, prior)
    if prompt is None:
        return None
    _, window, area = prompt
    x0, y0, x1, y1 = window.astype(int)
    sub = np.zeros(frame.depth.shape, bool)
    sub[y0 : y1 + 1, x0 : x1 + 1] = True
    points, pixels = frame.points_world(sub)
    if len(points) < 20:
        return None
    above = height_above_supports(points, as_supports(prior.support_heights))
    raised = (above > 0.3 * prior.size) & (above < 1.3 * prior.size)
    blob = np.zeros(frame.depth.shape, np.uint8)
    blob[pixels[raised, 0], pixels[raised, 1]] = 1
    blob = np.asarray(
        cv.morphologyEx(blob, cv.MORPH_OPEN, np.ones((3, 3), np.uint8)), dtype=np.uint8
    )
    count, labels, stats, centroids = cv.connectedComponentsWithStats(
        blob, connectivity=8
    )
    best = None
    best_score = -np.inf
    centre_px = prompt[0]
    for label in range(1, count):
        pixels_in = stats[label, cv.CC_STAT_AREA]
        if pixels_in < 0.15 * area:
            continue
        w, h = stats[label, cv.CC_STAT_WIDTH], stats[label, cv.CC_STAT_HEIGHT]
        fill = pixels_in / max(w * h, 1)
        # Compact, cube-sized, near the prior.
        score = (
            fill
            - abs(np.log(pixels_in / area))
            - np.linalg.norm(centroids[label] - centre_px) / max(np.sqrt(area), 1)
        )
        if score > best_score:
            best, best_score = label, score
    if best is None:
        return None
    x, y, w, h = (
        stats[best, k]
        for k in (cv.CC_STAT_LEFT, cv.CC_STAT_TOP, cv.CC_STAT_WIDTH, cv.CC_STAT_HEIGHT)
    )
    pad = 0.15 * max(w, h)
    box = np.array([x - pad, y - pad, x + w + pad, y + h + pad], dtype=float)
    intr = frame.intrinsics
    box = np.clip(
        box, 0, [intr.width - 1, intr.height - 1, intr.width - 1, intr.height - 1]
    )
    ys, xs = np.nonzero(labels == best)
    # The blob pixel nearest its centroid, so the point prompt lands on the cube.
    i = int(np.argmin((xs - centroids[best][0]) ** 2 + (ys - centroids[best][1]) ** 2))
    return np.array([float(xs[i]), float(ys[i])]), box


@dataclass
class ColourSegmenter:
    """Threshold the cube's known colour inside the prior's projected box.

    A deliberately simple baseline (and a fallback when no GPU is reachable): it needs
    the cube's colour, and it assumes nothing else of that colour sits near it.
    """

    hue_tolerance: float = 12.0
    min_saturation: int = 60
    min_value: int = 40
    name: str = "colour"

    def segment(self, frame: RGBDFrame, prior: CubePrior) -> Mask | PerceptionFailure:
        prompt = prior_prompt(frame, prior)
        if prompt is None:
            return PerceptionFailure("prior not in view")
        centre, box, _ = prompt
        hsv = np.asarray(cv.cvtColor(frame.rgb, cv.COLOR_RGB2HSV), dtype=np.uint8)
        target = np.array([[[int(255 * c) for c in prior.rgba[:3]]]], dtype=np.uint8)
        target_hue = float(cv.cvtColor(target, cv.COLOR_RGB2HSV)[0, 0, 0])
        hue = hsv[..., 0].astype(float)
        diff = np.minimum(np.abs(hue - target_hue), 180 - np.abs(hue - target_hue))
        mask = (
            (diff < self.hue_tolerance)
            & (hsv[..., 1] >= self.min_saturation)
            & (hsv[..., 2] >= self.min_value)
        )
        window = np.zeros_like(mask)
        x0, y0, x1, y1 = box.astype(int)
        window[y0 : y1 + 1, x0 : x1 + 1] = True
        mask &= window
        component = _largest_component(mask, centre)
        if component is None:
            return PerceptionFailure("no colour blob", mask)
        return component


class PerceptionClient:
    """Socket client for `perception_server.py` (SAM 2 and FoundationPose)."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 50111, timeout: float = 120.0
    ):
        self._address = (host, port)
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self.last_latency: float = 0.0

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send one request and wait for the reply."""
        if self._sock is None:
            self._sock = socket.create_connection(self._address, timeout=self._timeout)
        start = time.perf_counter()
        send_message(self._sock, message)
        reply = recv_message(self._sock)
        self.last_latency = time.perf_counter() - start
        return reply

    def close(self) -> None:
        """Close the connection."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None


@dataclass
class Sam2Segmenter:
    """SAM 2 prompted with the prior's projection (a point and a box).

    Of SAM 2's multimask outputs, the best-scoring mask whose area is within a factor of
    `area_tolerance` of the expected projected cube area wins.
    """

    client: PerceptionClient
    area_tolerance: float = 3.0
    name: str = "sam2"

    def segment(self, frame: RGBDFrame, prior: CubePrior) -> Mask | PerceptionFailure:
        prompt = prior_prompt(frame, prior)
        if prompt is None:
            return PerceptionFailure("prior not in view")
        centre, box, area = prompt
        located = depth_prompt(frame, prior)
        if located is not None:
            centre, box = located
        reply = self.client.request(
            {
                "op": "segment",
                "rgb": np.ascontiguousarray(frame.rgb),
                "points": [[float(centre[0]), float(centre[1])]],
                "point_labels": [1],
                "box": [float(v) for v in box],
                "multimask": True,
            }
        )
        if not reply.get("ok"):
            return PerceptionFailure(f"sam2 error: {reply.get('error')}")
        masks = np.asarray(reply["masks"]).astype(bool)
        scores = np.asarray(reply["scores"], dtype=float)
        plausible = []
        for mask, _score in sorted(zip(masks, scores), key=lambda ms: -ms[1]):
            pixels = mask.sum()
            if area / self.area_tolerance <= pixels <= area * self.area_tolerance:
                component = _largest_component(mask, centre)
                if component is not None:
                    plausible.append(component)
        if not plausible:
            return PerceptionFailure(
                "no sam2 mask of a plausible size", masks[0] if len(masks) else None
            )
        if len(plausible) == 1:
            return plausible[0]
        # SAM 2's best-scored mask is often one face of the cube, or the cube with a
        # patch of floor; keep the mask whose depth points look most like one cube.
        return max(
            plausible,
            key=lambda m: _cube_likeness(frame, m, prior.size, prior.support_heights),
        )


@dataclass
class DepthFitEstimator:
    """Segmenter mask + known-size cube fit to the registered depth."""

    segmenter: Segmenter
    erode_px: int = 2
    max_rms: float = 0.004
    min_inlier_fraction: float = 0.6
    name: str = "depthfit"

    def estimate(
        self, frame: RGBDFrame, prior: CubePrior
    ) -> CubeEstimate | PerceptionFailure:
        mask = self.segmenter.segment(frame, prior)
        if isinstance(mask, PerceptionFailure):
            return mask
        core = mask
        if self.erode_px > 0:
            kernel = np.ones((2 * self.erode_px + 1,) * 2, np.uint8)
            eroded = cv.erode(mask.astype(np.uint8), kernel).astype(bool)
            if eroded.sum() >= 30:
                core = eroded
        points, _ = frame.points_world(core)
        try:
            fit = fit_cube(
                points=points,
                size=prior.size,
                viewpoint=frame.world_from_camera[:3, 3],
                support_z=prior.support_heights,
            )
        except CubeFitError as e:
            return PerceptionFailure(f"depth fit: {e}", mask)
        quality = {
            "rms": fit.rms_residual,
            "inliers": fit.inlier_fraction,
            "points": float(fit.num_points),
        }
        if (
            fit.rms_residual > self.max_rms
            or fit.inlier_fraction < self.min_inlier_fraction
        ):
            return PerceptionFailure(f"depth fit rejected: {quality}", mask)
        return _gate(
            CubeEstimate(
                world_from_cube=fit.pose,
                yaw=fit.yaw,
                up_alignment=fit.up_alignment,
                size=prior.size,
                backend=f"{self.segmenter.name}+{self.name}",
                mask=mask,
                quality=quality,
            ),
            prior,
        )


@dataclass
class FoundationPoseEstimator:
    """Segmenter mask + FoundationPose registration of the known cube mesh."""

    segmenter: Segmenter
    client: PerceptionClient
    iterations: int = 5
    name: str = "foundationpose"

    def estimate(
        self, frame: RGBDFrame, prior: CubePrior
    ) -> CubeEstimate | PerceptionFailure:
        mask = self.segmenter.segment(frame, prior)
        if isinstance(mask, PerceptionFailure):
            return mask
        reply = self.client.request(
            {
                "op": "fp_register",
                "rgb": np.ascontiguousarray(frame.rgb),
                "depth": np.ascontiguousarray(frame.depth, dtype=np.float32),
                "K": frame.intrinsics.matrix,
                "mask": np.ascontiguousarray(mask),
                "mesh_key": f"cube{prior.size:.4f}",
                "cube_size": float(prior.size),
                "rgba": [float(c) for c in prior.rgba],
                "iterations": int(self.iterations),
            }
        )
        if not reply.get("ok"):
            return PerceptionFailure(
                f"foundationpose error: {reply.get('error')}", mask
            )
        world_from_cube = frame.world_from_camera @ np.asarray(
            reply["pose"], dtype=float
        )
        yaw, up = cube_yaw_and_up(world_from_cube[:3, :3])
        return _gate(
            CubeEstimate(
                world_from_cube=world_from_cube,
                yaw=yaw,
                up_alignment=up,
                size=prior.size,
                backend=f"{self.segmenter.name}+{self.name}",
                mask=mask,
                quality={"score": float(reply.get("score", np.nan))},
            ),
            prior,
        )


def _cube_likeness(
    frame: RGBDFrame, mask: Mask, size: float, supports: tuple = ()
) -> float:
    """Fraction of a mask's depth points within 4 mm of a fitted cube, penalised for
    masks much smaller than the cube's visible extent."""
    points, _ = frame.points_world(mask)
    if len(points) < 30:
        return -1.0
    try:
        fit = fit_cube(
            points=points,
            size=size,
            viewpoint=frame.world_from_camera[:3, 3],
            support_z=supports,
        )
    except CubeFitError:
        return -1.0
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    coverage = min(1.0, extent / size)
    return fit.inlier_fraction * coverage


def _gate(estimate: CubeEstimate, prior: CubePrior) -> CubeEstimate | PerceptionFailure:
    """Reject an estimate implausibly far from the prior."""
    offset = float(np.linalg.norm(estimate.position - prior.position))
    if offset > 3 * prior.sigma + prior.size:
        return PerceptionFailure(
            f"estimate {offset:.3f} m from the prior", estimate.mask
        )
    return estimate


def _largest_component(mask: Mask, centre: FloatArray) -> Mask | None:
    """The connected component containing (or nearest) `centre`."""
    count, labels_, stats_, centroids_ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    labels = np.asarray(labels_)
    stats = np.asarray(stats_, dtype=np.int64)
    centroids = np.asarray(centroids_, dtype=np.float64)
    if count <= 1:
        return None
    u, v = int(round(centre[0])), int(round(centre[1]))
    h, w = mask.shape
    if 0 <= v < h and 0 <= u < w and labels[v, u] > 0:
        label = labels[v, u]
    else:
        areas = stats[1:, cv.CC_STAT_AREA]
        distance = np.linalg.norm(centroids[1:] - centre, axis=1)
        label = 1 + int(np.argmin(distance - 1e-3 * areas))
    out = labels == label
    return out if out.sum() >= 10 else None
