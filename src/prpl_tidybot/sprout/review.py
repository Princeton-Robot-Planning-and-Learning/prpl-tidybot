"""Bind terminal approval to a saved preview and fresh scene before execution."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approve_and_execute(
    review_path: Path,
    observe: Callable[[], dict[str, Any]],
    execute: Callable[[dict[str, Any]], Any],
    *,
    prompt: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> Any:
    """No unattended Enter, stale approvals, changed artifacts or changed scene.

    observe is READ-ONLY and returns current calibrated camera/proprioception.
    execute must construct the commissioned compliant backend only after this
    gate. Approval covers the recipe and explicitly bounded visual corrections,
    not arbitrary new goals. Hardware guards still apply during execution.
    """
    review_hash = digest(review_path)
    review = json.loads(review_path.read_text())
    if review.get("execution_ready") is not True:
        raise RuntimeError(
            "Preview only: live scene adapter, limits and stop checks are not commissioned"
        )
    if review.get("completed") is not True:
        raise RuntimeError("Simulation did not complete; execution refused")
    if review.get("source") != "live_camera":
        raise RuntimeError("A synthetic or recorded scene cannot authorize hardware")
    if not (sys.stdin.isatty() if interactive is None else interactive):
        raise RuntimeError("Real execution requires an interactive terminal")
    artifacts = review["artifact_sha256"]
    if not {"preview.mp4", "trajectory.json", "scene.json"}.issubset(artifacts):
        raise ValueError("Missing reviewed video/trajectory/scene hashes")

    def check_artifacts() -> None:
        if digest(review_path) != review_hash:
            raise RuntimeError("Review changed; generate a new preview")
        for name, expected in artifacts.items():
            path = (review_path.parent / name).resolve()
            if (
                not path.is_relative_to(review_path.parent.resolve())
                or digest(path) != expected
            ):
                raise RuntimeError("Preview artifact changed; approval invalid")

    check_artifacts()
    print(
        f"Review {review_hash[:12]} | video: {review_path.parent / 'preview.mp4'}",
        flush=True,
    )
    try:
        answer = prompt(
            "After watching the preview: Enter or yes executes ONE real cycle; anything else cancels: "
        )
    except (EOFError, KeyboardInterrupt):
        raise RuntimeError("Approval cancelled") from None
    if answer.strip().lower() not in ("", "yes"):
        raise RuntimeError("Approval declined")
    check_artifacts()
    fresh = observe()
    maximum_age = float(review["maximum_observation_age_s"])
    if not math.isfinite(maximum_age) or maximum_age <= 0:
        raise ValueError("Invalid observation age bound")
    age = time.time() - float(fresh["captured_at_unix"])
    if fresh.get("ready") is not True or not 0 <= age <= maximum_age:
        raise RuntimeError("Hardware or fresh camera/proprioception is not ready")
    for key in ("grasp_xyz", "base_xyyaw", "arm_joints_rad"):
        old, new = review["observed"][key], fresh[key]
        expected_size = 7 if key == "arm_joints_rad" else 3
        tolerance = float(review["revalidation_tolerances"][key])
        if (
            len(old) != expected_size
            or len(new) != expected_size
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("Invalid snapshot or revalidation tolerance")
        for index, (a, b) in enumerate(zip(old, new)):
            delta = float(a) - float(b)
            if key == "arm_joints_rad" or (key == "base_xyyaw" and index == 2):
                delta = math.remainder(delta, 2 * math.pi)
            if not math.isfinite(delta) or abs(delta) > tolerance:
                raise RuntimeError(f"{key} changed; capture and preview again")
    check_artifacts()
    return execute(review)
