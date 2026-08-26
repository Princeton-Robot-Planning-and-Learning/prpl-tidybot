"""Interactive ChArUco intrinsics calibration for a ceiling camera.

Print `conf/calibration/charuco_board.pdf` at 100% scale (verify the scale
bar with a ruler) and mount it as flat as possible, e.g. on a clipboard —
or use a purchased rigid target and describe its geometry with the board
flags (`--squares-x/-y`, `--square-length-mm`, `--marker-length-mm`,
`--dictionary`, `--legacy`); the values are printed on the target.
Run this on the perception PC with a display and the camera servers
stopped — the tool opens the camera directly, since calibration needs raw
(distorted) frames:

    python scripts/calibrate_camera_intrinsics.py bottom --lab prpl

A live view opens with detected board corners overlaid. Hold the board in
view and press `c` to capture the current view; Esc finishes and
calibrates. Capture at least 15 views: vary the tilt (up to ~45 degrees)
and distance, and cover the whole field of view — the corners and edges of
the image matter most for fitting distortion.

The result overwrites `camera_params/<lab>/<serial>.yml` (the old file
stays recoverable via git). Camera extrinsics are solved on top of the
intrinsics, so re-run `scripts/calibrate_camera_extrinsics.py` afterwards.

Adapted from `yixuanhuang98/tidybot_server/server/calibrate_charuco.py`,
rewritten for the OpenCV 4.7+ aruco API.
"""

import argparse
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml  # type: ignore[import-untyped]

from prpl_tidybot.marker_detector import utils
from prpl_tidybot.marker_detector.constants import (
    CAMERA_SERIALS,
    CHARUCO_BOARD_PARAMS,
    MARKER_DICT_ID,
)

# Camera index per placement, matching the (top, bottom) serial order in
# the lab config.
PLACEMENT_TO_CAMERA_INDEX = {"top": 0, "bottom": 1}

# Brighter-than-detection capture settings so the handheld board is visible.
CALIBRATION_EXPOSURE = 312
CALIBRATION_GAIN = 0

# A view must contain at least this many interpolated chessboard corners to
# be worth keeping; sparser views mostly add noise.
MIN_CORNERS_PER_VIEW = 8
RECOMMENDED_VIEWS = 15


def make_board(
    squares_x: int = int(CHARUCO_BOARD_PARAMS["squares_x"]),
    squares_y: int = int(CHARUCO_BOARD_PARAMS["squares_y"]),
    square_length: float = CHARUCO_BOARD_PARAMS["square_length"],
    marker_length: float = CHARUCO_BOARD_PARAMS["marker_length"],
    dict_id: int = MARKER_DICT_ID,
    legacy: bool = False,
) -> Any:
    """Construct a ChArUco board; defaults to `CHARUCO_BOARD_PARAMS`.

    `legacy` selects the pre-OpenCV-4.6 chessboard/marker layout, which
    boards manufactured before that release (e.g. older calib.io targets)
    use when their row count is even.
    """
    aruco: Any = cv.aruco
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco.getPredefinedDictionary(dict_id),
    )
    if legacy:
        board.setLegacyPattern(True)
    return board


def calibrate_from_views(
    captured_corners: list[np.ndarray],
    captured_ids: list[np.ndarray],
    image_size: tuple[int, int],
    board: Any = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calibrate intrinsics from per-view ChArUco corner detections.

    Returns `(rms_px, camera_matrix, dist_coeffs)`.
    """
    board = board if board is not None else make_board()
    object_points = []
    image_points = []
    for corners, ids in zip(captured_corners, captured_ids):
        obj, img = board.matchImagePoints(corners, ids)
        object_points.append(obj)
        image_points.append(img)
    # The cv2 stubs reject None for the initial camera matrix, but the C++
    # accepts it as "estimate from scratch".
    calibrate: Any = cv.calibrateCamera
    rms, camera_matrix, dist_coeffs, _, _ = calibrate(
        object_points, image_points, image_size, None, None
    )
    return float(rms), camera_matrix, dist_coeffs


def write_camera_params(
    path: Path,
    image_width: int,
    image_height: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rms_px: float,
) -> None:
    """Write an intrinsics .yml in the format `utils.get_camera_params` reads."""
    fs = cv.FileStorage(str(path), cv.FILE_STORAGE_WRITE)
    fs.write("image_width", image_width)
    fs.write("image_height", image_height)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("avg_reprojection_error", rms_px)
    fs.release()


def _capture_views(
    serial: str, image_width: int, image_height: int, board: Any
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Interactive live view: collect ChArUco detections until Esc."""
    aruco: Any = cv.aruco
    detector = aruco.CharucoDetector(board)
    cap = utils.get_video_cap(serial, image_width, image_height)
    window = f"calibrate intrinsics ({serial})"
    cv.namedWindow(window)
    captured_corners: list[np.ndarray] = []
    captured_ids: list[np.ndarray] = []
    try:
        while True:
            cap.set(cv.CAP_PROP_EXPOSURE, CALIBRATION_EXPOSURE)
            cap.set(cv.CAP_PROP_GAIN, CALIBRATION_GAIN)
            ok, image = cap.read()
            if not ok or image is None:
                continue
            # pylint: disable=unpacking-non-sequence
            charuco_corners, charuco_ids, marker_corners, _ = detector.detectBoard(
                image
            )
            # pylint: enable=unpacking-non-sequence
            num_corners = 0 if charuco_ids is None else len(charuco_ids)
            display = image.copy()
            if marker_corners:
                aruco.drawDetectedMarkers(display, marker_corners)
            if num_corners:
                aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            cv.putText(
                display,
                f"{len(captured_ids)} views captured; {num_corners} corners in "
                "view. 'c' captures, Esc calibrates.",
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            cv.imshow(window, display)
            key = cv.waitKey(10) & 0xFF
            if key == 27:
                break
            if key == ord("c"):
                if num_corners >= MIN_CORNERS_PER_VIEW:
                    captured_corners.append(charuco_corners)
                    captured_ids.append(charuco_ids)
                    print(f"Captured view {len(captured_ids)} ({num_corners} corners)")
                else:
                    print(
                        f"Not captured: only {num_corners} corners in view "
                        f"({MIN_CORNERS_PER_VIEW} required)."
                    )
    finally:
        cap.release()
        cv.destroyAllWindows()
    return captured_corners, captured_ids


def main(placement: str, lab: str | None = None, board: Any = None) -> None:
    """Run the interactive capture, calibrate, and overwrite the .yml."""
    board = board if board is not None else make_board()
    serial = _resolve_serial(placement, lab)
    yml_candidates = list(utils.CAMERA_PARAMS_DIR.glob(f"*/{serial}.yml"))
    assert (
        len(yml_candidates) == 1
    ), f"expected one intrinsics file for {serial}, found: {yml_candidates}"
    path = yml_candidates[0]
    image_width, image_height, _, _ = utils.get_camera_params(serial)

    print(f"Calibrating intrinsics for camera {serial} ({placement}).")
    print(
        f"Capture at least {RECOMMENDED_VIEWS} views: vary tilt and distance "
        "and cover the whole field, especially image corners and edges."
    )
    captured_corners, captured_ids = _capture_views(
        serial, image_width, image_height, board
    )
    assert (
        len(captured_ids) >= 4
    ), f"only {len(captured_ids)} views captured; at least 4 required"
    if len(captured_ids) < RECOMMENDED_VIEWS:
        print(
            f"note: only {len(captured_ids)} views; accuracy improves with "
            f"{RECOMMENDED_VIEWS}+"
        )
    rms_px, camera_matrix, dist_coeffs = calibrate_from_views(
        captured_corners, captured_ids, (image_width, image_height), board=board
    )
    write_camera_params(
        path, image_width, image_height, camera_matrix, dist_coeffs, rms_px
    )
    print(f"Wrote {path}")
    print(f"  reprojection RMS: {rms_px:.3f} px (good calibrations are < 1 px)")
    print(
        "The stored extrinsics were solved with the old intrinsics; re-run "
        "scripts/calibrate_camera_extrinsics.py now."
    )


def _resolve_serial(placement: str, lab: str | None) -> str:
    if lab is not None:
        conf_path = Path(__file__).parents[3] / "conf" / "lab" / f"{lab}.yaml"
        with open(conf_path, "r", encoding="utf-8") as f:
            camera_serials = yaml.safe_load(f)["camera_serials"]
    else:
        camera_serials = list(CAMERA_SERIALS)
    return camera_serials[PLACEMENT_TO_CAMERA_INDEX[placement]]


def cli() -> None:
    """Argparse entrypoint shared by `python -m ...` and `scripts/`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "placement",
        choices=sorted(PLACEMENT_TO_CAMERA_INDEX),
        help="Which camera to calibrate, by its floor placement.",
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
    board_group = parser.add_argument_group(
        "board geometry",
        "Defaults describe conf/calibration/charuco_board.pdf. For a "
        "purchased target, copy the values printed on it (e.g. a calib.io "
        "label 'CharuCo / 9x12 / 15 mm / DICT_5X5' means --squares-x 12 "
        "--squares-y 9 --square-length-mm 15 --marker-length-mm <printed "
        "marker size> --dictionary DICT_5X5_100).",
    )
    board_group.add_argument(
        "--squares-x",
        type=int,
        default=CHARUCO_BOARD_PARAMS["squares_x"],
        help="Number of squares along the board's long side.",
    )
    board_group.add_argument(
        "--squares-y",
        type=int,
        default=CHARUCO_BOARD_PARAMS["squares_y"],
        help="Number of squares along the board's short side.",
    )
    board_group.add_argument(
        "--square-length-mm",
        type=float,
        default=CHARUCO_BOARD_PARAMS["square_length"] * 1000,
        help="Side length of one chessboard square, in mm.",
    )
    board_group.add_argument(
        "--marker-length-mm",
        type=float,
        default=CHARUCO_BOARD_PARAMS["marker_length"] * 1000,
        help="Side length of one ArUco marker, in mm.",
    )
    board_group.add_argument(
        "--dictionary",
        default=None,
        help=(
            "ArUco dictionary name, e.g. DICT_5X5_100; default is the "
            "repo's marker dictionary (DICT_4X4_50)."
        ),
    )
    board_group.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Use the pre-OpenCV-4.6 board layout. Needed for boards "
            "manufactured before that release when the row count is even; "
            "try this if markers detect but no chessboard corners appear."
        ),
    )
    args = parser.parse_args()
    dict_id = (
        MARKER_DICT_ID
        if args.dictionary is None
        else getattr(cv.aruco, args.dictionary)
    )
    board = make_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length_mm / 1000,
        marker_length=args.marker_length_mm / 1000,
        dict_id=dict_id,
        legacy=args.legacy,
    )
    main(placement=args.placement, lab=args.lab, board=board)


if __name__ == "__main__":
    cli()
