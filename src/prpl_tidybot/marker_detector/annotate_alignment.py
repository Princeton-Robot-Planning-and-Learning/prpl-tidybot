"""Interactive tool to (re-)annotate a ceiling camera's floor alignment.

The marker detector maps pixels to the floor with a homography fitted from
four floor corners annotated per camera in
`camera_params/<lab>/<serial>.json`. When those annotations go stale (a
camera was nudged or remounted), the two cameras disagree about the position
of a shared marker; the marker-detector server reports this as a large
`target_residuals` entry in its payload.

The tool shows a live, brightened, undistorted view from the chosen camera
with the current corner quad overlaid. Double-click to move the nearest
corner to the clicked pixel; every edit is saved immediately. Pass
`--center` to instead move the camera center (the floor point directly
under the lens). Press Esc or close the window to exit. Corner order
matters — the floor point each corner must land on is printed at startup.

The camera center is stored both in pixels (`camera_center`) and relative
to the corner quad (`camera_center_relative`). Moving a corner re-derives
the pixel center from the relative one, so the center stays pinned to the
same floor point and a rotation-only camera bump needs corner edits only.

Run on the perception PC with a display, with the camera servers stopped
(`scripts/stop_servers.sh`) — the tool opens the camera directly.

Adapted from `yixuanhuang98/tidybot_server/server/camera_alignment.py`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from prpl_tidybot.marker_detector import utils
from prpl_tidybot.marker_detector.constants import CAMERA_SERIALS

# Serial-list index of the camera covering each placement (matches the
# order the marker-detector server starts its camera servers in).
PLACEMENT_TO_CAMERA_INDEX = {"top": 0, "top_only": 0, "bottom": 1}

# Brighter-than-detection capture settings so floor-tile corners are visible.
ANNOTATION_EXPOSURE = 624
ANNOTATION_GAIN = 0

_UNIT_SQUARE = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)


def _unit_square_transform(corners: list[list[int]]) -> np.ndarray:
    """Homography sending the pixel corner quad to the unit square."""
    return cv.getPerspectiveTransform(
        np.array(corners, dtype=np.float32), _UNIT_SQUARE
    ).astype(np.float32)


def relative_from_center(
    corners: list[list[int]], center: tuple[int, int] | list[int]
) -> list[float]:
    """Express a pixel camera center relative to the corner quad."""
    point = np.array([[center]], dtype=np.float32)
    return (
        cv.perspectiveTransform(point, _unit_square_transform(corners))
        .reshape(2)
        .tolist()
    )


def center_from_relative(
    corners: list[list[int]], center_relative: list[float]
) -> list[int]:
    """Recover the pixel camera center from its quad-relative coordinates."""
    _, inverse = cv.invert(_unit_square_transform(corners))
    point = np.array([[center_relative]], dtype=np.float32)
    return cv.perspectiveTransform(point, inverse).reshape(2).astype(int).tolist()


def nearest_corner_index(
    corners: list[list[int]], point: tuple[int, int] | list[int]
) -> int:
    """Index of the corner closest to `point`."""
    diffs = np.array(corners, dtype=np.float32) - np.array(point, dtype=np.float32)
    return int(np.argmin((diffs**2).sum(axis=1)))


class Annotator:
    """Live-view editor for one camera's floor-alignment annotations."""

    def __init__(self, serial: str, save_path: Path, edit_center: bool = False) -> None:
        self.serial = serial
        self.save_path = save_path
        self.edit_center = edit_center
        self.window_name = f"annotate {serial}"
        (
            self.image_width,
            self.image_height,
            self.camera_matrix,
            self.dist_coeffs,
        ) = utils.get_camera_params(serial)

        self.labels: dict[str, Any] = {}
        if save_path.exists():
            with open(save_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

        default_padding = 50
        self.camera_corners: list[list[int]] = self.labels.get("camera_corners") or [
            [default_padding, default_padding],
            [self.image_width - default_padding, default_padding],
            [self.image_width - default_padding, self.image_height - default_padding],
            [default_padding, self.image_height - default_padding],
        ]
        self.camera_center_relative: list[float] = self.labels.get(
            "camera_center_relative"
        ) or [0.5, 0.5]
        self.camera_center: list[int] = self.labels.get(
            "camera_center"
        ) or center_from_relative(self.camera_corners, self.camera_center_relative)

    def save(self) -> None:
        """Write the current annotations back to the JSON file."""
        self.labels["camera_corners"] = self.camera_corners
        self.labels["camera_center"] = self.camera_center
        self.labels["camera_center_relative"] = self.camera_center_relative
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)
            f.write("\n")
        print(f"Saved {self.save_path}: {self.labels}")

    def on_mouse(self, event: int, x: int, y: int, *_args: Any) -> None:
        """Double-click handler: move the nearest corner, or the center."""
        if event != cv.EVENT_LBUTTONDBLCLK:
            return
        if self.edit_center:
            self.camera_center = [x, y]
            self.camera_center_relative = relative_from_center(
                self.camera_corners, (x, y)
            )
        else:
            idx = nearest_corner_index(self.camera_corners, (x, y))
            self.camera_corners[idx] = [x, y]
            # Keep the center pinned to the same floor point by re-deriving
            # its pixel position through the updated quad.
            self.camera_center = center_from_relative(
                self.camera_corners, self.camera_center_relative
            )
        self.save()

    def draw_overlay(self, image: np.ndarray) -> None:
        """Draw the corner quad and camera center on a display frame."""
        for i, corner in enumerate(self.camera_corners):
            cv.circle(image, tuple(corner), 5, (0, 0, 255))
            cv.putText(
                image,
                str(i + 1),
                (corner[0] + 8, corner[1] - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv.line(
                image,
                tuple(corner),
                tuple(self.camera_corners[(i + 1) % 4]),
                (0, 0, 255),
            )
        cv.circle(image, tuple(self.camera_center), 5, (255, 0, 0))

    def run(self) -> None:
        """Stream the camera and edit annotations until Esc / window close."""
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.on_mouse)
        cap = utils.get_video_cap(self.serial, self.image_width, self.image_height)
        try:
            while True:
                cap.set(cv.CAP_PROP_EXPOSURE, ANNOTATION_EXPOSURE)
                cap.set(cv.CAP_PROP_GAIN, ANNOTATION_GAIN)
                if cv.waitKey(1) == 27 or (
                    cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 0.5
                ):
                    break
                image = None
                while image is None:
                    _, image = cap.read()
                image = cv.undistort(image, self.camera_matrix, self.dist_coeffs)
                self.draw_overlay(image)
                cv.imshow(self.window_name, image)
        finally:
            cap.release()
            cv.destroyAllWindows()


def main(placement: str, lab: str | None = None, edit_center: bool = False) -> None:
    """Resolve the camera and save path, then run the annotator."""
    if lab is not None:
        camera_serials, _ = utils.load_lab_camera_config(lab)
    else:
        camera_serials = list(CAMERA_SERIALS)
    serial = camera_serials[PLACEMENT_TO_CAMERA_INDEX[placement]]

    candidates = list(utils.CAMERA_PARAMS_DIR.glob(f"*/{serial}.json"))
    if candidates:
        assert len(candidates) == 1, f"multiple alignment files for {serial}"
        save_path = candidates[0]
    elif lab is not None:
        save_path = utils.CAMERA_PARAMS_DIR / lab / f"{serial}.json"
    else:
        raise SystemExit(
            f"No alignment file exists for camera {serial}; pass --lab to "
            "choose which camera_params/<lab>/ directory to create it in."
        )

    print(f"Annotating camera {serial} ({placement}); edits save to {save_path}.")
    print("Each numbered corner must land on this floor point (map frame):")
    for i, (x, y) in enumerate(utils.get_floor_corner_coords(placement)):
        print(f"  corner {i + 1}: (x={x:+.2f} m, y={y:+.2f} m)")
    if edit_center:
        print("Center mode: double-click the floor point directly under the lens.")
    else:
        print("Double-click to move the nearest corner. Esc exits.")
    Annotator(serial, save_path, edit_center=edit_center).run()
    print(
        "Restart the marker-detector server to pick up the new alignment, "
        "then check that the published target_residuals have dropped."
    )


def cli() -> None:
    """Argparse entrypoint shared by `python -m ...` and `scripts/`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "placement",
        choices=sorted(PLACEMENT_TO_CAMERA_INDEX),
        help="Which camera to annotate, by its floor placement.",
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
        "--center",
        action="store_true",
        help="Edit the camera center (nadir) instead of the corners.",
    )
    args = parser.parse_args()
    main(placement=args.placement, lab=args.lab, edit_center=args.center)


if __name__ == "__main__":
    cli()
