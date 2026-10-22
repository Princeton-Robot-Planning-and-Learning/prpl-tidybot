"""The real Robotiq 2F-85's collision shape, link by link, at a given opening.

Taken from KINDER's gen3_2f85 MJCF (the model the pick's 14.8 x 7.6 cm gripper box
was measured on): the gripper is settled at the finger command, and every collision
geom of each gripper link is sampled over its surface (so a horizontal cut at the
rim is exact to the sampling step) and expressed in the pinch site's frame, which is
the server's tool frame: x along the fingers' travel, y across them (the IK site's
'closing' axis), z the approach (towards the fingertips).

The opening is commanded as kinder-baselines #178 does: by the pad gap, through its
measured MuJoCo gap-to-command table (``MUJOCO_GRIPPER_GAPS`` / ``_COMMANDS`` in
``kinder_models.dynamic3d.tossing.parameterized_skills``).
"""

# mujoco is a compiled extension; pylint cannot see its members.
# pylint: disable=no-member

from __future__ import annotations

import functools
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

# kinder-baselines #178: the 2F-85's inner pad gap (m) against MuJoCo's finger
# command (0 open .. 1 closed), settled with gravity off.
MUJOCO_GRIPPER_GAPS = (0.0851, 0.0777, 0.0698, 0.0615, 0.0529, 0.0440)
MUJOCO_GRIPPER_COMMANDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
FULL_OPENING = 0.085

GRIPPER_HOUSING = "base"
# The links that move with the fingers: what B lets touch a wall lightly.
FINGER_LINKS = frozenset(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in ("driver", "coupler", "spring_link", "follower", "pad")
)
_SAMPLE_STEP = 0.003
_SETTLE_STEPS = 3000


def opening_command(opening: float) -> float:
    """The finger command (0 open ..

    1 closed) that opens the pads to ``opening`` metres (#178's
    ``gripper_command_for_opening``).
    """
    if opening >= FULL_OPENING:
        return 0.0
    return float(
        np.interp(opening, MUJOCO_GRIPPER_GAPS[::-1], MUJOCO_GRIPPER_COMMANDS[::-1])
    )


def _gripper_xml() -> Path:
    import kinder  # pylint: disable=import-outside-toplevel

    return (
        Path(kinder.__file__).parent
        / "envs"
        / "dynamic3d"
        / "models"
        / "kinova_gen3"
        / "gen3_2f85.xml"
    )


@functools.lru_cache(maxsize=8)
def gripper_link_points(command: float) -> dict[str, NDArray[np.float64]]:
    """Each gripper link's surface points (tool frame, m) with the fingers settled at
    ``command`` (0 open ..

    1 closed).
    """
    model = mujoco.MjModel.from_xml_path(str(_gripper_xml()))
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("retract").id)
    data.ctrl[:7] = data.qpos[:7]
    data.ctrl[7] = 255.0 * float(command)
    for _ in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)
    mujoco.mj_kinematics(model, data)
    site = model.site("pinch_site").id
    rot = data.site_xmat[site].reshape(3, 3)
    origin = data.site_xpos[site]
    links = FINGER_LINKS | {GRIPPER_HOUSING}
    out: dict[str, list[NDArray[np.float64]]] = {}
    for g in range(model.ngeom):
        body = str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        )
        if body not in links or not model.geom_contype[g]:
            continue
        local = _geom_surface(model, g)
        if local is None:
            continue
        # einsum, not @: see arm_bin_collision (OpenBLAS threads on large matmuls).
        xmat = data.geom_xmat[g].reshape(3, 3)
        world = data.geom_xpos[g] + np.einsum("ij,kj->ik", local, xmat)
        out.setdefault(body, []).append(np.einsum("ij,jk->ik", world - origin, rot))
    return {body: np.vstack(parts) for body, parts in out.items()}


def _geom_surface(model: mujoco.MjModel, g: int) -> NDArray[np.float64] | None:
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        return _sample_triangles(
            np.asarray(model.mesh_vert[va : va + vn], np.float64),
            np.asarray(model.mesh_face[fa : fa + fn]),
        )
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        size = np.asarray(model.geom_size[g], np.float64)
        axes = [
            np.linspace(-s, s, max(2, int(np.ceil(2 * s / _SAMPLE_STEP)) + 1))
            for s in size
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        on_face = np.any(np.isclose(np.abs(grid), size), axis=1)
        return np.asarray(grid[on_face], np.float64)
    return None


def _sample_triangles(
    verts: NDArray[np.float64], faces: NDArray[np.int32]
) -> NDArray[np.float64]:
    out = [verts]
    for tri in faces:
        p0, p1, p2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        longest = max(
            np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p0), np.linalg.norm(p2 - p1)
        )
        n = int(np.ceil(longest / _SAMPLE_STEP))
        if n <= 1:
            continue
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        out.append(p0 + a[:, None] * (p1 - p0) + b[:, None] * (p2 - p0))
    return np.vstack(out)
