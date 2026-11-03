"""Replay the original PRPL executors offline, then render their Kinder shadow."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
from kinder.envs.kinematic3d.prpl3d import PrplLab3DEnvConfig

from prpl_tidybot.sim_env import KinderSimEnv
from prpl_tidybot.sprout.pipeline import (
    SproutPipeline,
    arm_pairs,
    base_pairs,
    fake_interface,
    recipe_phases,
)
from prpl_tidybot.video import subsample_indices, write_mp4


def _box(client: int, center, half, color) -> None:
    """Only perceived environmental geometry is a box, never the robot."""
    shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=client
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=shape,
        basePosition=center,
        physicsClientId=client,
    )


def render_pipeline_preview(
    report: dict,
    scene: dict,
    config: dict,
    trajectory: list[dict],
    output: Path,
    fps: int = 15,
) -> dict:
    """No hardware: execute the exact main stack with its original FakeInterface."""
    states, labels, commands = [], [], []
    phase = "initial measured state"

    def record(state, _sim_action, real_action, *_):
        states.append(state.copy())
        labels.append(phase)
        commands.append(
            {
                "phase": phase,
                "arm_goal": real_action.arm_goal,
                "base_goal_xyyaw": [
                    real_action.base_pose_target_map.x,
                    real_action.base_pose_target_map.y,
                    real_action.base_pose_target_map.theta(),
                ],
                "gripper_goal": real_action.gripper_goal,
            }
        )

    pipeline = SproutPipeline(
        fake_interface(report["observed"]),
        config,
        threading.Event(),
        control_period=0.0,
        on_step=record,
    )
    states.append(pipeline.state.copy())
    labels.append(phase)
    try:
        plan = report.get("base_plan")
        if plan and plan["required"]:
            phase = "base transit"
            pipeline.run(phase, base_pairs(pipeline.state, plan["path_xyyaw"]))
        for phase, frames in recipe_phases(trajectory):
            if phase not in ("base", "base_transit"):
                pipeline.run(phase, arm_pairs(pipeline.state, frames))
    finally:
        pipeline.close()

    shadow = KinderSimEnv(
        "kinder/PrplLab3D-o1-v0",
        num_cubes=0,
        allow_state_access=True,
        config=PrplLab3DEnvConfig(
            realistic_bg=False, render_image_width=960, render_image_height=720
        ),
    )
    frames_rgb = []
    try:
        shadow.reset(seed=0)
        # Same inner shadow world used by main's preview/recording layer.
        inner = shadow._inner_env  # pylint: disable=protected-access
        client = inner.physics_client_id
        body = scene.get("body_obstacle")
        if body:
            center, half = np.asarray(body["center"]), np.asarray(body["half_size"])
            # Wire box keeps the floor arm/tape visible through the body envelope.
            for axis in range(3):
                others = [i for i in range(3) if i != axis]
                for a in (-1, 1):
                    for b in (-1, 1):
                        pos = center.copy()
                        pos[others] += half[others] * [a, b]
                        size = np.full(3, 0.004)
                        size[axis] = half[axis]
                        _box(client, pos, size, [0.6, 0.2, 0.85, 1])
        tape = scene.get("white_tape_xyz", report["observed"]["grasp_xyz"])
        _box(client, tape, [0.018, 0.012, 0.004], [1, 1, 1, 1])
        _box(client, report["observed"]["grasp_xyz"], [0.008] * 3, [0.1, 1, 0.1, 1])
        indices = subsample_indices(len(states), 160)
        for index in indices:
            shadow.set_state(states[index])
            image = np.asarray(shadow.render()).copy()
            cv2.rectangle(image, (0, 0), (image.shape[1], 66), (25, 30, 40), -1)
            cv2.putText(
                image,
                "PRPL main Runner + executors | " + labels[index],
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                "Purple: Sprout box | White: tape | Green: grasp",
                (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            frames_rgb.append(image)
        write_mp4(frames_rgb, output / "preview.mp4", fps)
    finally:
        shadow.close()
    (output / "executor-trace.json").write_text(json.dumps(commands, indent=2) + "\n")
    return {
        "execution_backend": "prpl_main_runner",
        "shadow_sim": "KinderSimEnv",
        "executor_ticks": len(commands),
        "camera_age_checks": False,
    }
