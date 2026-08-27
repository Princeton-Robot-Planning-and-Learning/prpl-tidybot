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

With `--auto` no key presses are needed (useful when the camera is mounted
out of reach of the keyboard): good, novel views are captured hands-free
with a terminal bell on each, and calibration runs by itself after
`--num-views` views. Start the script, walk under the camera, and move the
board around until the bell stops.

The result overwrites `camera_params/<lab>/<serial>.yml` (the old file
stays recoverable via git). Camera extrinsics are solved on top of the
intrinsics, so re-run `scripts/calibrate_camera_extrinsics.py` afterwards.

Adapted from `yixuanhuang98/tidybot_server/server/calibrate_charuco.py`,
rewritten for the OpenCV 4.7+ aruco API.
"""

import argparse
import pickle
import tempfile
import time
from pathlib import Path
from threading import Thread
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

# Auto-capture pacing: never capture more often than the minimum interval;
# capture when the board's mean corner position has moved at least the
# novelty distance since every previous capture, or — so that tilt-in-place
# views are also collected — after the fallback interval regardless.
AUTO_MIN_INTERVAL_S = 1.5
AUTO_NOVELTY_PX = 40.0
AUTO_FALLBACK_INTERVAL_S = 6.0
AUTO_DEFAULT_VIEWS = 20

# V4L2 reads can block forever after a USB hiccup; a read that takes longer
# than this is treated as stalled and the camera is reopened.
READ_TIMEOUT_S = 2.0
# Re-latch exposure/gain this often (in frames). Once per frame is control-
# transfer spam; never re-latching risks silent driver resets.
SETTINGS_REFRESH_FRAMES = 30


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


def should_auto_capture(
    mean_corner: np.ndarray,
    captured_means: list[np.ndarray],
    seconds_since_last: float,
) -> bool:
    """Decide whether the auto-capture mode should keep the current view."""
    if seconds_since_last < AUTO_MIN_INTERVAL_S:
        return False
    if seconds_since_last >= AUTO_FALLBACK_INTERVAL_S:
        return True
    return all(
        float(np.linalg.norm(mean_corner - previous)) >= AUTO_NOVELTY_PX
        for previous in captured_means
    )


def autosave_path(serial: str) -> Path:
    """Where in-progress captured views are checkpointed for `--resume`."""
    return Path(tempfile.gettempdir()) / f"charuco_views_{serial}.pkl"


def _save_views(path: Path, corners: list[np.ndarray], ids: list[np.ndarray]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"corners": corners, "ids": ids}, f)


def _load_views(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["corners"], data["ids"]


def _apply_capture_settings(cap: Any) -> None:
    cap.set(cv.CAP_PROP_EXPOSURE, CALIBRATION_EXPOSURE)
    cap.set(cv.CAP_PROP_GAIN, CALIBRATION_GAIN)


def _read_frame(cap: Any, timeout_s: float = READ_TIMEOUT_S) -> np.ndarray | None:
    """`cap.read()` with a stall watchdog; None means timed out or failed.

    A stalled V4L2 read blocks forever, so it runs in a helper thread; on
    timeout the caller should release and reopen the camera, which also
    unblocks the abandoned read.
    """
    box: dict[str, Any] = {}

    def worker() -> None:
        box["result"] = cap.read()

    thread = Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return None
    ok, image = box.get("result", (False, None))
    return image if ok else None


def _capture_views(
    serial: str,
    image_width: int,
    image_height: int,
    board: Any,
    auto: bool = False,
    target_views: int = AUTO_DEFAULT_VIEWS,
    resume: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Interactive live view: collect ChArUco detections until Esc.

    With `auto`, views are captured hands-free (terminal bell on each) and
    the loop ends after `target_views`; Esc still finishes early. Every
    capture is checkpointed to `autosave_path(serial)`; `resume` reloads a
    previous session's checkpoint so a crash costs no progress.
    """
    aruco: Any = cv.aruco
    detector = aruco.CharucoDetector(board)
    captured_corners: list[np.ndarray] = []
    captured_ids: list[np.ndarray] = []
    captured_means: list[np.ndarray] = []
    checkpoint = autosave_path(serial)
    if resume and checkpoint.exists():
        captured_corners, captured_ids = _load_views(checkpoint)
        captured_means = [
            np.asarray(corners).reshape(-1, 2).mean(axis=0)
            for corners in captured_corners
        ]
        print(f"Resumed {len(captured_ids)} previously captured views.")
    cap = utils.get_video_cap(serial, image_width, image_height)
    _apply_capture_settings(cap)
    window = f"calibrate intrinsics ({serial})"
    cv.namedWindow(window)
    last_capture_time = -float("inf")
    frames_since_settings = 0
    try:
        while True:
            image = _read_frame(cap)
            if image is None:
                print("Camera read stalled; reopening the camera...")
                cap.release()
                cap = utils.get_video_cap(serial, image_width, image_height)
                _apply_capture_settings(cap)
                continue
            frames_since_settings += 1
            if frames_since_settings >= SETTINGS_REFRESH_FRAMES:
                _apply_capture_settings(cap)
                frames_since_settings = 0
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
            if auto:
                status = (
                    f"auto: {len(captured_ids)}/{target_views} views; "
                    f"{num_corners} corners in view. Esc finishes early."
                )
            else:
                status = (
                    f"{len(captured_ids)} views captured; {num_corners} corners "
                    "in view. 'c' captures, Esc calibrates."
                )
            cv.putText(
                display,
                status,
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

            good_view = num_corners >= MIN_CORNERS_PER_VIEW
            if auto:
                if good_view:
                    mean_corner = (
                        np.asarray(charuco_corners).reshape(-1, 2).mean(axis=0)
                    )
                    since_last = time.monotonic() - last_capture_time
                    if should_auto_capture(mean_corner, captured_means, since_last):
                        captured_corners.append(charuco_corners)
                        captured_ids.append(charuco_ids)
                        captured_means.append(mean_corner)
                        last_capture_time = time.monotonic()
                        _save_views(checkpoint, captured_corners, captured_ids)
                        # \a rings the terminal bell so progress is audible
                        # from under the camera, away from the screen.
                        print(
                            f"\aCaptured view {len(captured_ids)}/{target_views} "
                            f"({num_corners} corners)"
                        )
                if len(captured_ids) >= target_views:
                    print("Target number of views reached; calibrating.")
                    break
            elif key == ord("c"):
                if good_view:
                    captured_corners.append(charuco_corners)
                    captured_ids.append(charuco_ids)
                    _save_views(checkpoint, captured_corners, captured_ids)
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


def main(
    placement: str,
    lab: str | None = None,
    board: Any = None,
    auto: bool = False,
    target_views: int = AUTO_DEFAULT_VIEWS,
    resume: bool = False,
) -> None:
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
        serial,
        image_width,
        image_height,
        board,
        auto=auto,
        target_views=target_views,
        resume=resume,
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
    autosave_path(serial).unlink(missing_ok=True)
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
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Capture views hands-free: whenever a good detection is novel "
            "(the board moved since previous captures, with a time-based "
            "fallback for tilt-in-place views), ringing the terminal bell "
            "each time, and calibrate automatically after --num-views."
        ),
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=AUTO_DEFAULT_VIEWS,
        help="Views to collect before auto mode stops and calibrates.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reload the views checkpointed by a previous run that crashed "
            "or was killed (every capture is autosaved)."
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
    main(
        placement=args.placement,
        lab=args.lab,
        board=board,
        auto=args.auto,
        target_views=args.num_views,
        resume=args.resume,
    )


if __name__ == "__main__":
    cli()
