"""Coordinate frame converter between two SE2 frames calibrated from a shared
observation expressed in both."""

import math

from spatialmath import SE2


class CoordFrameConverter:
    """Calibrate a fixed transform between frames A and B from a pose pair, then
    convert subsequent poses from frame A to frame B.

    The same physical pose observed in two frames defines a rigid transform
    between them; `update` solves for that transform (origin + basis angle) and
    `convert_pose` applies its inverse to map A-frame coordinates into B.

    The calibration drifts over time when the underlying frames aren't truly
    rigid w.r.t. each other (e.g. wheel odometry vs map frame from a marker
    detector); call `update` again with a fresh observation pair whenever a
    new measurement is available.
    """

    def __init__(
        self,
        pose_in_frame_a: SE2,
        pose_in_frame_b: SE2,
    ) -> None:
        self.origin: tuple[float, float] = (0.0, 0.0)
        self.basis: float = 0.0
        self.update(pose_in_frame_a, pose_in_frame_b)

    def update(
        self,
        pose_in_frame_a: SE2,
        pose_in_frame_b: SE2,
    ) -> None:
        """Refit the A→B transform from a new (pose_a, pose_b) observation."""
        self.basis = pose_in_frame_a.theta() - pose_in_frame_b.theta()
        dx = pose_in_frame_b.x * math.cos(self.basis) - pose_in_frame_b.y * math.sin(
            self.basis
        )
        dy = pose_in_frame_b.x * math.sin(self.basis) + pose_in_frame_b.y * math.cos(
            self.basis
        )
        self.origin = (pose_in_frame_a.x - dx, pose_in_frame_a.y - dy)

    def convert_pose(self, pose: SE2) -> SE2:
        """Convert a pose from frame A to frame B using the current calibration."""
        x, y, th = pose.x, pose.y, pose.theta()

        x = x - self.origin[0]
        y = y - self.origin[1]
        xp = x * math.cos(-self.basis) - y * math.sin(-self.basis)
        yp = x * math.sin(-self.basis) + y * math.cos(-self.basis)

        converted_th = th - self.basis

        return SE2(xp, yp, converted_th)
