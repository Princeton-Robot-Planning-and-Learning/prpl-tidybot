"""Learned 6-DoF grasp detectors (Contact-GraspNet, M2T2) behind the generator interface.

Both detectors run in their own environment behind `grasp_server.py` and see only
the wrist depth (registered to the colour camera), its intrinsics, the cube's mask,
and -- for M2T2, which was trained on world-frame clouds -- the camera's pose.

**Gripper convention.** Both were trained on the Franka Panda hand: z is the approach,
x the closing direction, and the frame's origin sits 0.1034 m behind the line
through the pad centres (`gripper_depth` in both codebases), opening at most 8 cm.
Our tool frame (the PyBullet Kinova model's `tool_frame`) uses the same axes (z
approach, x closing, y = z x x) with the pads' centre 5.2 mm ahead of it. So a
detector grasp G becomes our tool pose G @ Translate(0, 0, 0.1034 - 0.0052): same
orientation, the pads' centre at the same point. The Robotiq 2F-85 opens to 8.5 cm, so
any Panda width is reachable; the commanded opening is the detector's width plus a
margin, never less than the cube plus that margin. Finger *shape* differs (the
Robotiq's linkage and knuckles are bulkier than the Panda's straight fingers), which
is why every learned grasp is still checked against the Robotiq model in the planning
scene before it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_pick.grasps import GraspCandidate
from prpl_tidybot.cube_pick.perception import PerceptionClient

FloatArray = NDArray[np.float64]

PANDA_ORIGIN_TO_PADS = 0.1034
TOOL_TO_PADS = 0.0052


def panda_to_tool(panda_pose: FloatArray) -> FloatArray:
    """A Panda-convention grasp pose to our tool-frame pose (same pads' centre)."""
    offset = np.eye(4)
    offset[2, 3] = PANDA_ORIGIN_TO_PADS - TOOL_TO_PADS
    return panda_pose @ offset


@dataclass
class LearnedGraspGenerator:
    """Contact-GraspNet or M2T2 grasps on the cube, from the wrist depth."""

    client: PerceptionClient
    model: str
    cube_size: float
    opening_margin: float = 0.010
    max_grasps: int = 60
    max_opening: float = 0.085

    @property
    def name(self) -> str:
        """The detector's name."""
        return self.model

    def candidates(
        self, *, cube_frame: FloatArray, frame: Any, mask: Any
    ) -> list[GraspCandidate]:
        if frame is None:
            return []
        segmap = np.zeros(frame.depth.shape, np.int32)
        if mask is not None:
            segmap[np.asarray(mask, bool)] = 1
        request: dict[str, Any] = {
            "op": "grasps",
            "model": self.model,
            "depth": np.ascontiguousarray(frame.depth, dtype=np.float32),
            "K": frame.intrinsics.matrix,
            "rgb": np.ascontiguousarray(frame.rgb),
            "segmap": segmap if mask is not None else None,
            "target_id": 1,
            "max_grasps": self.max_grasps,
            "z_range": [0.15, 1.5],
            "T_world_cam": np.asarray(frame.world_from_camera, dtype=np.float64),
        }
        reply = self.client.request(request)
        if not reply.get("ok"):
            raise RuntimeError(f"{self.model} grasp server: {reply.get('error')}")
        poses = np.asarray(reply["poses"], dtype=float).reshape(-1, 4, 4)
        scores = np.asarray(reply["scores"], dtype=float).reshape(-1)
        widths = np.asarray(reply["widths"], dtype=float).reshape(-1)
        cube_from_world = np.linalg.inv(cube_frame)
        out = []
        for pose, score, width in zip(poses, scores, widths):
            world_tool = frame.world_from_camera @ panda_to_tool(pose)
            cube_from_tool = cube_from_world @ world_tool
            approach = world_tool[:3, 2]
            tilt = float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
            opening = float(
                np.clip(
                    max(width, self.cube_size) + self.opening_margin,
                    0.0,
                    self.max_opening,
                )
            )
            pads = cube_from_tool[:3, 3] + cube_from_tool[:3, 2] * TOOL_TO_PADS
            out.append(
                GraspCandidate(
                    cube_from_tool=cube_from_tool,
                    yaw_offset=float(
                        np.arctan2(cube_from_tool[1, 0], cube_from_tool[0, 0])
                    ),
                    tilt=tilt,
                    opening=opening,
                    depth=float(pads[2]),
                    source=self.model,
                    score=float(score),
                )
            )
        return out
