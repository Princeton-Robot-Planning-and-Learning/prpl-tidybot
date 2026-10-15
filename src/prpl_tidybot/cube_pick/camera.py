"""The wrist RGB-D camera, as the pick pipeline sees it on the robot and in simulation.

The Kinova Gen3 vision module carries two sensors: a colour camera, which on our
robot streams 640x480 through a fisheye lens attachment, and an Intel RealSense depth
module (480x270, 72 deg diagonal, nothing usable closer than ~20 cm). The pipeline consumes one
:class:`RGBDFrame`: colour, depth already registered onto the colour pixel grid, the
colour intrinsics, and the colour camera's pose in the world, which the robot knows from
its own kinematics (forward kinematics plus the hand-eye calibration), not from looking
at anything.

Everything here is pure numpy so the real robot and the simulator run the same code;
the simulator only differs in where the raw colour and depth images come from (see
:mod:`prpl_tidybot.cube_pick.kinder_sim`), and it passes its renders through
:class:`DepthNoiseModel` so the depth it hands over looks like the sensor's.

Conventions: camera frames are OpenCV's (x right, y down, z forward); depth images are
z-depth in metres with 0 meaning "no return".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PinholeIntrinsics:
    """A pinhole camera with square pixels and pixel-centre coordinates."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_diagonal_fov(
        cls, *, width: int, height: int, diagonal_fov_deg: float
    ) -> PinholeIntrinsics:
        """Square-pixel intrinsics with the principal point at the image centre."""
        focal = (np.hypot(width, height) / 2) / np.tan(np.radians(diagonal_fov_deg) / 2)
        return cls(
            width=width,
            height=height,
            fx=float(focal),
            fy=float(focal),
            cx=(width - 1) / 2,
            cy=(height - 1) / 2,
        )

    @property
    def matrix(self) -> FloatArray:
        """The 3x3 camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def vertical_fov_deg(self) -> float:
        """Full vertical field of view."""
        return float(np.degrees(2 * np.arctan((self.height / 2) / self.fy)))

    @property
    def horizontal_fov_deg(self) -> float:
        """Full horizontal field of view."""
        return float(np.degrees(2 * np.arctan((self.width / 2) / self.fx)))

    def scaled(self, factor: float) -> PinholeIntrinsics:
        """The same camera at `factor` times the resolution."""
        width = int(round(self.width * factor))
        height = int(round(self.height * factor))
        sx, sy = width / self.width, height / self.height
        return PinholeIntrinsics(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Camera-frame points (N, 3) to pixel coordinates (N, 2) and depths (N,)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        z = points[:, 2]
        uv = np.stack(
            [self.fx * points[:, 0] / z + self.cx, self.fy * points[:, 1] / z + self.cy],
            axis=1,
        )
        return uv, z

    def backproject(self, uv: FloatArray, z: FloatArray) -> FloatArray:
        """Pixel coordinates (N, 2) with depths (N,) to camera-frame points (N, 3)."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        z = np.asarray(z, dtype=float).reshape(-1)
        x = (uv[:, 0] - self.cx) * z / self.fx
        y = (uv[:, 1] - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1)

    def depth_to_points(
        self, depth: NDArray[np.floating], mask: NDArray[np.bool_] | None = None
    ) -> tuple[FloatArray, NDArray[np.intp]]:
        """Back-project the valid (and masked) pixels of a depth image.

        Returns camera-frame points (N, 3) and their (row, col) indices (N, 2).
        """
        valid = depth > 0
        if mask is not None:
            valid &= mask
        rows, cols = np.nonzero(valid)
        uv = np.stack([cols, rows], axis=1).astype(float)
        points = self.backproject(uv, depth[rows, cols])
        return points, np.stack([rows, cols], axis=1)


@dataclass(frozen=True)
class FisheyeIntrinsics:
    """OpenCV's fisheye (equidistant) model: r = f * theta * (1 + k1 theta^2 + ...)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float]

    @property
    def matrix(self) -> FloatArray:
        """The 3x3 camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def coefficients(self) -> FloatArray:
        """The four fisheye distortion coefficients D."""
        return np.asarray(self.distortion, dtype=np.float64).reshape(4, 1)

    def project(self, points: FloatArray) -> FloatArray:
        """Camera-frame points (N, 3) to distorted pixel coordinates (N, 2)."""
        import cv2 as cv  # pylint: disable=import-outside-toplevel

        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
        uv, _ = cv.fisheye.projectPoints(
            points, np.zeros(3), np.zeros(3), self.matrix, self.coefficients
        )
        return uv.reshape(-1, 2)  # type: ignore[no-any-return]

    def undistort(
        self, image: NDArray, pinhole: PinholeIntrinsics, interpolation: int | None = None
    ) -> NDArray:
        """Resample a fisheye image into the virtual pinhole camera `pinhole`."""
        import cv2 as cv  # pylint: disable=import-outside-toplevel

        map1, map2 = cv.fisheye.initUndistortRectifyMap(
            self.matrix,
            self.coefficients,
            np.eye(3),
            pinhole.matrix,
            (pinhole.width, pinhole.height),
            cv.CV_32FC1,
        )
        return cv.remap(  # type: ignore[no-any-return]
            image, map1, map2, cv.INTER_LINEAR if interpolation is None else interpolation
        )

    def distort_from_pinhole(self, image: NDArray, source: PinholeIntrinsics) -> NDArray:
        """What this fisheye camera sees, resampled from a pinhole render `source` taken
        from the same centre (the simulator renders pinhole images only)."""
        import cv2 as cv  # pylint: disable=import-outside-toplevel

        v, u = np.mgrid[0 : self.height, 0 : self.width].astype(np.float64)
        pixels = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        rays = cv.fisheye.undistortPoints(pixels, self.matrix, self.coefficients).reshape(-1, 2)
        map_x = (rays[:, 0] * source.fx + source.cx).reshape(self.height, self.width)
        map_y = (rays[:, 1] * source.fy + source.cy).reshape(self.height, self.width)
        return cv.remap(  # type: ignore[no-any-return]
            image, map_x.astype(np.float32), map_y.astype(np.float32), cv.INTER_LINEAR
        )


# The Kinova Gen3 vision module, from its datasheet: colour streamed at 640x480 with a
# 65-69 deg diagonal field of view (67 is used), depth at 480x270. These stand in for
# the robot's calibrated values, which plug in here once measured. A fisheye lens
# attachment over the colour sensor has been reported but not verified, so it is not
# modelled; `FisheyeIntrinsics` is the hook if it turns out to be real.
KINOVA_WRIST_COLOR_FISHEYE: FisheyeIntrinsics | None = None
KINOVA_WRIST_COLOR = PinholeIntrinsics.from_diagonal_fov(
    width=640, height=480, diagonal_fov_deg=67.0
)
KINOVA_WRIST_DEPTH = PinholeIntrinsics.from_diagonal_fov(
    width=480, height=270, diagonal_fov_deg=72.0
)
# The datasheet's minimum depth is 18 cm; the robot's D410-class module returns
# nothing usable inside ~20 cm, so the simulator blacks out depth closer than that.
KINOVA_DEPTH_MIN_RANGE = 0.20
KINOVA_DEPTH_MAX_RANGE = 3.0


@dataclass
class RGBDFrame:
    """One wrist observation: colour, registered depth and where the camera was.

    `depth` is on the colour pixel grid (same height and width as `rgb`).
    `world_from_camera` is the colour camera's pose (OpenCV axes) in the world frame,
    from the robot's kinematics.
    """

    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    intrinsics: PinholeIntrinsics
    world_from_camera: FloatArray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.rgb.shape[:2] == self.depth.shape
        assert self.rgb.shape[:2] == (self.intrinsics.height, self.intrinsics.width)

    def points_world(
        self, mask: NDArray[np.bool_] | None = None
    ) -> tuple[FloatArray, NDArray[np.intp]]:
        """World-frame points of the valid (and masked) depth pixels, and their pixels."""
        points, pixels = self.intrinsics.depth_to_points(self.depth, mask)
        return transform_points(self.world_from_camera, points), pixels

    def project_world(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """World points to pixel coordinates and camera depths."""
        camera_from_world = np.linalg.inv(self.world_from_camera)
        return self.intrinsics.project(transform_points(camera_from_world, points))

    def resized(self, factor: float) -> RGBDFrame:
        """The frame at `factor` times the resolution (nearest-neighbour for depth)."""
        import cv2 as cv  # pylint: disable=import-outside-toplevel

        intr = self.intrinsics.scaled(factor)
        rgb = cv.resize(self.rgb, (intr.width, intr.height), interpolation=cv.INTER_AREA)
        depth = cv.resize(
            self.depth, (intr.width, intr.height), interpolation=cv.INTER_NEAREST
        )
        return RGBDFrame(rgb, depth, intr, self.world_from_camera, dict(self.metadata))


@dataclass(frozen=True)
class WristCalibration:
    """Everything needed to turn the two raw wrist streams into one RGBDFrame.

    On the robot these come from calibration (fisheye intrinsics from a checkerboard,
    depth intrinsics from the Kortex VisionConfig, the depth-to-colour extrinsics from
    a joint calibration); in simulation they are the values the renderer used.
    """

    color_fisheye: FisheyeIntrinsics | None
    depth: PinholeIntrinsics
    color_from_depth: FloatArray
    output: PinholeIntrinsics


def assemble_wrist_frame(
    *,
    color_fisheye_image: NDArray[np.uint8],
    depth_image: NDArray[np.floating],
    calibration: WristCalibration,
    world_from_color: FloatArray,
) -> RGBDFrame:
    """Undistort the colour (if a fisheye model is given) into the pinhole camera
    `calibration.output` and register the depth (metres, 0 = invalid) onto it. The
    same code path on the robot and in sim."""
    if calibration.color_fisheye is None:
        rgb = color_fisheye_image
    else:
        rgb = calibration.color_fisheye.undistort(color_fisheye_image, calibration.output)
    registered = register_depth_to_color(
        depth=depth_image,
        depth_intrinsics=calibration.depth,
        color_intrinsics=calibration.output,
        color_from_depth=calibration.color_from_depth,
    )
    frame = RGBDFrame(
        rgb=np.ascontiguousarray(rgb),
        depth=registered,
        intrinsics=calibration.output,
        world_from_camera=world_from_color,
    )
    frame.metadata["raw_depth"] = np.asarray(depth_image)
    frame.metadata["fisheye_rgb"] = color_fisheye_image
    return frame


def transform_points(transform: FloatArray, points: FloatArray) -> FloatArray:
    """Apply a 4x4 rigid transform to (N, 3) points."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def register_depth_to_color(
    *,
    depth: NDArray[np.floating],
    depth_intrinsics: PinholeIntrinsics,
    color_intrinsics: PinholeIntrinsics,
    color_from_depth: FloatArray,
    splat: int = 2,
) -> NDArray[np.float32]:
    """Reproject a depth image into the colour camera's pixel grid.

    The same operation the RealSense SDK's `align` performs: every valid depth pixel is
    back-projected, moved into the colour frame, projected, and written with a z-buffer
    (nearest wins). The depth sensor is lower resolution than the colour target, so each
    point is splatted over `splat` x `splat` target pixels to avoid a sieve of holes.
    """
    points, _ = depth_intrinsics.depth_to_points(np.asarray(depth))
    out = np.full((color_intrinsics.height, color_intrinsics.width), np.inf)
    if len(points):
        points = transform_points(color_from_depth, points)
        points = points[points[:, 2] > 1e-6]
        uv, z = color_intrinsics.project(points)
        base_u = np.round(uv[:, 0]).astype(int)
        base_v = np.round(uv[:, 1]).astype(int)
        offsets = range(-((splat - 1) // 2), splat // 2 + 1) if splat > 1 else (0,)
        for du in offsets:
            for dv in offsets:
                u, v = base_u + du, base_v + dv
                ok = (
                    (u >= 0)
                    & (u < color_intrinsics.width)
                    & (v >= 0)
                    & (v < color_intrinsics.height)
                )
                np.minimum.at(out, (v[ok], u[ok]), z[ok])
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


@dataclass
class DepthNoiseModel:
    """What a RealSense D4xx does to a perfect z-depth image.

    * nothing closer than `min_range` or farther than `max_range` (0 = no return);
    * axial Gaussian noise growing with the square of the range (stereo depth error is
      proportional to z^2 / (focal * baseline)), plus a floor;
    * random dropouts, and extra dropouts at depth discontinuities, where stereo
      matching fails ("flying pixels" are removed by the sensor's own filter);
    * millimetre quantization (the sensor's z16 unit).

    Deterministic given `seed`; call :meth:`reseed` between frames for independent
    noise.
    """

    min_range: float = KINOVA_DEPTH_MIN_RANGE
    max_range: float = KINOVA_DEPTH_MAX_RANGE
    sigma_floor: float = 0.001
    sigma_per_m2: float = 0.004
    dropout: float = 0.01
    edge_dropout: float = 0.5
    edge_threshold: float = 0.02
    quantization: float = 0.001
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reseed(self, seed: int) -> None:
        """Restart the noise stream."""
        self._rng = np.random.default_rng(seed)

    def apply(self, depth: NDArray[np.floating]) -> NDArray[np.float32]:
        """A noisy copy of `depth` (metres)."""
        depth = np.asarray(depth, dtype=np.float64)
        sigma = self.sigma_floor + self.sigma_per_m2 * depth**2
        noisy = depth + self._rng.normal(size=depth.shape) * sigma
        if self.quantization > 0:
            noisy = np.round(noisy / self.quantization) * self.quantization
        invalid = (depth < self.min_range) | (depth > self.max_range) | ~np.isfinite(depth)
        invalid |= self._rng.random(depth.shape) < self.dropout
        if self.edge_dropout > 0:
            grad = np.zeros_like(depth)
            grad[:, 1:] = np.maximum(grad[:, 1:], np.abs(np.diff(depth, axis=1)))
            grad[:, :-1] = np.maximum(grad[:, :-1], np.abs(np.diff(depth, axis=1)))
            grad[1:, :] = np.maximum(grad[1:, :], np.abs(np.diff(depth, axis=0)))
            grad[:-1, :] = np.maximum(grad[:-1, :], np.abs(np.diff(depth, axis=0)))
            edges = grad > self.edge_threshold
            invalid |= edges & (self._rng.random(depth.shape) < self.edge_dropout)
        noisy[invalid] = 0.0
        return noisy.astype(np.float32)
