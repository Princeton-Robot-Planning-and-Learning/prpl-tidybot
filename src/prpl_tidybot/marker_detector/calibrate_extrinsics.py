"""Calibrate the ceiling cameras' extrinsics from markers on the floor.

Tape ArUco `DICT_4X4_50` markers flat on the floor (print them with
`scripts/create_calibration_markers.py`), measure each marker centre's
map-frame position with a tape measure, and list marker id -> position in
a YAML file (see `conf/calibration/markers_example.yaml`, which also
explains how the positions define the map frame). Then run:

    python scripts/calibrate_camera_extrinsics.py --lab prpl --markers <yaml>

For each camera this detects the markers, averages their pixel centres over
several frames, solves the camera pose with solvePnP, writes
`camera_params/<lab>/<serial>_extrinsics.json`, and prints a report: RMS
reprojection error, per-marker floor residuals, the derived camera height,
and — for markers seen by both cameras — the cross-camera disagreement.

Marker placement: spread markers across each camera's view and include at
least two near the floor midline so both cameras see them. Each camera
needs at least four detected markers (not all in a line); six or more is
recommended. Avoid the robot-sticker and target ids in
`DETECTED_MARKER_IDS`.

The cameras shift slightly after power-on and stabilise after ~10 minutes
of streaming. Frames are taken from running camera servers when available,
otherwise the cameras are opened directly. If the cameras have not already
been streaming for a while, pass `--warmup-minutes 10` to stream (and
discard) frames for that long before capturing.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml  # type: ignore[import-untyped]

from prpl_tidybot.marker_detector import utils
from prpl_tidybot.marker_detector.camera_client import CameraClient
from prpl_tidybot.marker_detector.constants import (
    CAMERA_SERIALS,
    CAMERA_SERVER_PORTS,
    DETECTED_MARKER_IDS,
    MARKER_DICT_ID,
)
from prpl_tidybot.marker_detector.extrinsics import (
    FloorProjector,
    extrinsics_path,
    save_extrinsics,
    solve_extrinsics,
)

DEFAULT_NUM_FRAMES = 30
# A marker must be detected in at least this fraction of the captured
# frames to be trusted (rules out spurious detections).
MIN_DETECTION_FRACTION = 0.5
MIN_MARKERS_PER_CAMERA = 4
RECOMMENDED_MARKERS_PER_CAMERA = 6


def load_marker_positions(path: Path) -> dict[int, tuple[float, float]]:
    """Load the marker id -> map-frame (x, y) table from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    positions = {
        int(marker_id): (float(x), float(y))
        for marker_id, (x, y) in config["marker_positions"].items()
    }
    overlap = set(positions) & set(DETECTED_MARKER_IDS)
    assert not overlap, (
        f"calibration marker ids {sorted(overlap)} collide with robot/target "
        f"ids {DETECTED_MARKER_IDS}; use different ids"
    )
    return positions


def average_marker_centers(
    per_frame_centers: list[dict[int, np.ndarray]],
    min_fraction: float = MIN_DETECTION_FRACTION,
) -> dict[int, np.ndarray]:
    """Average each marker's pixel centre across frames.

    Markers detected in fewer than `min_fraction` of the frames are
    discarded as unreliable.
    """
    counts: dict[int, list[np.ndarray]] = {}
    for centers in per_frame_centers:
        for marker_id, center in centers.items():
            counts.setdefault(marker_id, []).append(center)
    min_count = min_fraction * len(per_frame_centers)
    return {
        marker_id: np.mean(observations, axis=0)
        for marker_id, observations in counts.items()
        if len(observations) >= min_count
    }


def solve_from_detections(
    detected_centers: dict[int, np.ndarray],
    marker_positions: dict[int, tuple[float, float]],
    camera_matrix: np.ndarray,
) -> dict[str, Any]:
    """Solve one camera's pose from averaged marker detections.

    Returns a report dict with `rvec`, `tvec`, `rms_px`, the per-marker
    map-frame `floor_residuals` (metres, detected centre projected to the
    floor vs. its known position), and the `marker_ids` used.
    """
    marker_ids = sorted(set(detected_centers) & set(marker_positions))
    assert len(marker_ids) >= MIN_MARKERS_PER_CAMERA, (
        f"only {len(marker_ids)} known markers detected "
        f"({MIN_MARKERS_PER_CAMERA} required); detected ids: "
        f"{sorted(detected_centers)}"
    )
    pixels = np.array([detected_centers[i] for i in marker_ids])
    floor_xy = np.array([marker_positions[i] for i in marker_ids])
    rvec, tvec, rms_px = solve_extrinsics(floor_xy, pixels, camera_matrix)
    projector = FloorProjector(camera_matrix, rvec, tvec)
    on_floor = projector.project_to_heights(pixels, 0.0)
    floor_residuals = {
        marker_id: float(np.linalg.norm(on_floor[i] - floor_xy[i]))
        for i, marker_id in enumerate(marker_ids)
    }
    return {
        "rvec": rvec,
        "tvec": tvec,
        "rms_px": rms_px,
        "floor_residuals": floor_residuals,
        "marker_ids": marker_ids,
    }


def _detect_marker_centers(image: np.ndarray) -> dict[int, np.ndarray]:
    """Detect all DICT_4X4_50 markers in an undistorted frame."""
    aruco: Any = cv.aruco
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(MARKER_DICT_ID), params
    )
    # pylint: disable=unpacking-non-sequence
    corners, ids, _ = detector.detectMarkers(image)
    # pylint: enable=unpacking-non-sequence
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(marker_corners, dtype=np.float64)
        .reshape(4, 2)
        .mean(axis=0)
        for marker_id, marker_corners in zip(ids.reshape(-1), corners)
    }


class _FrameSource:
    """Undistorted-frame reader over a camera server or a direct capture.

    Holding one open per camera lets every camera stream simultaneously,
    so a single warm-up window covers them all.
    """

    def __init__(self, serial: str, port: int) -> None:
        self._client: CameraClient | None
        self._cap: cv.VideoCapture | None = None
        try:
            self._client = CameraClient(port)
        except ConnectionRefusedError:
            print(f"No camera server on port {port}; opening camera {serial} directly.")
            self._client = None
            (
                image_width,
                image_height,
                self._camera_matrix,
                self._dist_coeffs,
            ) = utils.get_camera_params(serial)
            self._cap = utils.get_video_cap(serial, image_width, image_height)

    def read(self) -> np.ndarray | None:
        """Return one undistorted frame, or None on a failed direct read."""
        if self._client is not None:
            return self._client.get_image().copy()
        assert self._cap is not None
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        return cv.undistort(image, self._camera_matrix, self._dist_coeffs)

    def close(self) -> None:
        """Release the server connection or the capture device."""
        if self._client is not None:
            self._client.close()
        elif self._cap is not None:
            self._cap.release()


def _warm_up_all(sources: list[_FrameSource], warmup_s: float) -> None:
    """Stream (and discard) frames from every camera at once for `warmup_s`,
    so cold-started sensors stabilise before the frames that matter."""
    if warmup_s <= 0:
        return
    deadline = time.monotonic() + warmup_s
    print(f"Warming up {len(sources)} camera(s) for {warmup_s / 60:.1f} min...")
    next_report = time.monotonic() + 60
    while time.monotonic() < deadline:
        for source in sources:
            source.read()
        if time.monotonic() >= next_report:
            remaining = (deadline - time.monotonic()) / 60
            print(f"  warm-up: {remaining:.1f} min remaining")
            next_report += 60


def _calibrate_camera(
    serial: str,
    source: _FrameSource,
    marker_positions: dict[int, tuple[float, float]],
    num_frames: int,
) -> dict[str, Any]:
    """Capture, detect, solve, save, and print the report for one camera."""
    _, _, camera_matrix, _ = utils.get_camera_params(serial)
    frames: list[np.ndarray] = []
    while len(frames) < num_frames:
        frame = source.read()
        if frame is not None:
            frames.append(frame)
    detected = average_marker_centers(
        [_detect_marker_centers(frame) for frame in frames]
    )
    report = solve_from_detections(detected, marker_positions, camera_matrix)

    path = extrinsics_path(serial)
    save_extrinsics(
        path,
        report["rvec"],
        report["tvec"],
        metadata={
            "reprojection_rms_px": report["rms_px"],
            "marker_ids": report["marker_ids"],
        },
    )
    projector = FloorProjector(camera_matrix, report["rvec"], report["tvec"])

    print(f"\nCamera {serial}: wrote {path}")
    print(f"  markers used: {report['marker_ids']}")
    if len(report["marker_ids"]) < RECOMMENDED_MARKERS_PER_CAMERA:
        print(
            f"  note: fewer than {RECOMMENDED_MARKERS_PER_CAMERA} markers; "
            "more markers improve accuracy"
        )
    print(f"  reprojection RMS: {report['rms_px']:.2f} px")
    for marker_id, residual in sorted(report["floor_residuals"].items()):
        print(f"  marker {marker_id} floor residual: {residual * 100:.1f} cm")
    x, y, z = projector.camera_position
    print(f"  derived camera position: x={x:+.3f} m, y={y:+.3f} m, height={z:.3f} m")

    report["detected_centers"] = detected
    report["projector"] = projector
    return report


def _print_cross_camera_report(reports: list[dict[str, Any]]) -> None:
    """Print the disagreement between cameras on markers they both saw."""
    shared = set.intersection(*(set(report["detected_centers"]) for report in reports))
    if not shared:
        print(
            "\nNo marker was seen by every camera; place markers near the "
            "floor midline to enable the cross-camera check."
        )
        return
    print("\nCross-camera disagreement (both cameras' floor projections):")
    for marker_id in sorted(shared):
        positions = [
            report["projector"].project_to_heights(
                np.array([report["detected_centers"][marker_id]]), 0.0
            )[0]
            for report in reports
        ]
        distance = float(np.linalg.norm(positions[0] - positions[1]))
        print(f"  marker {marker_id}: {distance * 100:.1f} cm")


def main(
    markers_path: Path,
    lab: str | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    warmup_minutes: float = 0.0,
) -> None:
    """Calibrate every ceiling camera and report cross-camera consistency."""
    marker_positions = load_marker_positions(markers_path)
    if lab is not None:
        conf_path = Path(__file__).parents[3] / "conf" / "lab" / f"{lab}.yaml"
        with open(conf_path, "r", encoding="utf-8") as f:
            camera_serials = yaml.safe_load(f)["camera_serials"]
    else:
        camera_serials = list(CAMERA_SERIALS)

    sources = [
        _FrameSource(serial, port)
        for serial, port in zip(camera_serials, CAMERA_SERVER_PORTS)
    ]
    try:
        _warm_up_all(sources, warmup_minutes * 60)
        reports = [
            _calibrate_camera(serial, source, marker_positions, num_frames)
            for serial, source in zip(camera_serials, sources)
        ]
    finally:
        for source in sources:
            source.close()
    if len(reports) > 1:
        _print_cross_camera_report(reports)
    print(
        "\nRestart the marker-detector server to pick up the new extrinsics, "
        "then check the published target_residuals."
    )


def cli() -> None:
    """Argparse entrypoint shared by `python -m ...` and `scripts/`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--markers",
        type=Path,
        required=True,
        help=(
            "YAML file mapping marker id -> map-frame [x, y] of each floor "
            "marker's centre (see conf/calibration/markers_example.yaml)."
        ),
    )
    parser.add_argument(
        "--lab",
        default=None,
        help=(
            "Lab name (e.g. 'prpl') to load camera serials from "
            "conf/lab/<lab>.yaml; omit to use whatever PRPL_LAB resolves "
            "at import time."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Frames to average marker detections over, per camera.",
    )
    parser.add_argument(
        "--warmup-minutes",
        type=float,
        default=0.0,
        help=(
            "Stream (and discard) frames from all cameras together for this "
            "long before capturing, so cold-started cameras stabilise. Omit "
            "if the camera servers have already been running for 10+ minutes."
        ),
    )
    args = parser.parse_args()
    main(
        markers_path=args.markers,
        lab=args.lab,
        num_frames=args.num_frames,
        warmup_minutes=args.warmup_minutes,
    )


if __name__ == "__main__":
    cli()
