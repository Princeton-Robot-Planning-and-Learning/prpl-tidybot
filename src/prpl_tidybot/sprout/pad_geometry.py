"""Derive pinch height from public gripper pad geometry and a floor-resting box."""

import itertools
from pathlib import Path

import mujoco
import numpy as np


def box_grasp_geometry(model_path: Path, top_height_m: float) -> dict:
    """Derive pinch height and jaw clearance from the public gripper model."""
    if not np.isfinite(top_height_m) or top_height_m <= 0:
        raise ValueError("Positive floor-to-top height required")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = model.site("pinch_site").id
    rotation = data.site_xmat[site].reshape(3, 3)
    pads = []
    for name in ("left_pad", "right_pad"):
        body = model.body(name).id
        points = []
        for geom in range(model.ngeom):
            if (
                model.geom_bodyid[geom] == body
                and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
            ):
                local = (
                    np.array(list(itertools.product((-1, 1), repeat=3)))
                    * model.geom_size[geom]
                )
                world = (
                    data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T
                )
                points.extend((world - data.site_xpos[site]) @ rotation)
        if not points:
            raise ValueError("Model lacks pad collision boxes")
        pads.append(np.asarray(points))
    pad_points = np.vstack(pads)
    centers = [p.mean(0) for p in pads]
    direction = centers[0] - centers[1]
    if abs(direction[0]) < 10 * max(abs(direction[1]), abs(direction[2]), 1e-9):
        raise ValueError("Expected gripper closing axis along pinch X")
    pad_center_z = float((pad_points[:, 2].max() + pad_points[:, 2].min()) / 2)
    pinch = top_height_m / 2 + pad_center_z
    return {
        "grasp_height_m": pinch,
        "pad_drop_m": float(pad_points[:, 2].max()),
        "predicted_pad_floor_clearance_m": pinch - float(pad_points[:, 2].max()),
        "maximum_open_width_m": float(pads[0][:, 0].min() - pads[1][:, 0].max()),
        "assumption": "forearm box rests on floor; pad centres meet half its height",
    }
