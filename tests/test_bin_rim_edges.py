"""Tests for the rim-edge bin detector (``bin.method: rim_edges``) on real ceiling
frames from the 2026-10-20 pilot.

Each file in ``tests/data/bin_rim`` is one saved frame set reduced to what the
detector reads: the mean of its last three stitched frames (top camera over bottom,
1280 x 1440), stored as a quality-90 JPEG (~125 kB instead of ~20 MB of PNGs).

* ``close_18-10-36_08``: the bin at the close spot, the weakest true bin of the
  replay (edge score 0.034 against the 0.014 threshold).
* ``far_shelf_10-37-51_00``: the bin at the far spot with the wooden shelf at
  (1.5, 1.5) in view, which the saturation mask at 150 took for the bin.

The spots are the bin's labelled centres that day, as the replay harness used them.
"""

import json
import math
from pathlib import Path

import cv2 as cv
import pytest

from prpl_tidybot.bin_localizer import (
    BinModel,
    CeilingBinSource,
    localize_bin_rim_edges,
)
from prpl_tidybot.cube_localizer import load_ceiling_cameras

SERIALS = ("AD8A29AE", "5A5C29AE")
MODEL = BinModel(length=0.39, width=0.29, height=0.21, wall=0.005)
SEARCH = (-1.8, 2.0, -1.8, 1.8)
DATA = Path(__file__).resolve().parent / "data" / "bin_rim"
CLOSE_SPOT = (-0.299, -0.774)
FAR_SPOT = (0.226, 1.374)
SHELF = (1.5, 1.5)


def _frame(name):
    frame = cv.imread(str(DATA / f"{name}.jpg"))
    assert frame is not None and frame.shape == (1440, 1280, 3)
    return frame


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    return load_ceiling_cameras(SERIALS)


@pytest.fixture(scope="module", name="close_frame")
def _close_frame():
    return _frame("close_18-10-36_08")


@pytest.fixture(scope="module", name="far_frame")
def _far_frame():
    return _frame("far_shelf_10-37-51_00")


def _dist(fit, spot):
    return math.hypot(fit.x - spot[0], fit.y - spot[1])


def test_rim_edges_finds_the_close_spot_bin(cameras, close_frame):
    """The close spot's bin, seen by the bottom camera, is found within 3 cm."""
    diag: dict = {}
    fit = localize_bin_rim_edges(
        [close_frame], cameras, MODEL, SEARCH, diagnostics=diag
    )
    assert fit is not None, diag
    assert _dist(fit, CLOSE_SPOT) < 0.03
    assert fit.camera == 1
    assert (fit.length, fit.width, fit.height) == (0.39, 0.29, 0.21)
    assert diag["reason"] == "ok"
    assert diag["score"] >= 0.014 and diag["dark"] < -0.3 and diag["margin_ok"]


def test_rim_edges_finds_the_far_spot_bin(cameras, far_frame):
    """The far spot's bin, seen by the top camera, is found within 3 cm."""
    fit = localize_bin_rim_edges([far_frame], cameras, MODEL, SEARCH)
    assert fit is not None
    assert _dist(fit, FAR_SPOT) < 0.03
    assert fit.camera == 0


@pytest.mark.parametrize(
    "name, spot",
    [("close_18-10-36_08", CLOSE_SPOT), ("far_shelf_10-37-51_00", FAR_SPOT)],
)
def test_a_hidden_bin_is_a_miss_never_a_wrong_bin(cameras, name, spot):
    """With the bin's own footprint excluded from the search (the replay's "bin hidden"
    negatives), whatever is left is not a bin: a miss, with the reason."""
    diag: dict = {}
    fit = localize_bin_rim_edges(
        [_frame(name)],
        cameras,
        MODEL,
        SEARCH,
        exclude=[(spot[0], spot[1], 0.40)],
        diagnostics=diag,
    )
    assert fit is None
    assert diag["reason"] != "ok"


def test_the_shelf_is_not_returned_as_the_bin(cameras, far_frame):
    """The shelf at (1.5, 1.5) that the saturation mask took for the bin is not one:

    searched for on its own, nothing is returned.
    """
    diag: dict = {}
    fit = localize_bin_rim_edges(
        [far_frame],
        cameras,
        MODEL,
        (SHELF[0] - 0.35, SHELF[0] + 0.35, SHELF[1] - 0.35, SHELF[1] + 0.3),
        diagnostics=diag,
    )
    assert fit is None
    assert diag["reason"] != "ok"


class _FakeImages:
    """Serves one stitched frame."""

    def __init__(self, frame):
        self._frame = frame

    def get_image(self):
        """The frame."""
        return self._frame

    def close(self):
        """Nothing to release."""


class _FakeMarkers:
    """A marker detector that only reports the robot's pose."""

    def __init__(self, robot_xy):
        self._robot = robot_xy

    def get_latest(self):
        """A payload with the robot pose and no targets."""
        return {"poses": {0: (self._robot[0], self._robot[1], 0.0)}, "targets": {}}

    def close(self):
        """Nothing to release."""


def _source(frame, robot_xy, failure_dir=None, method="rim_edges"):
    return CeilingBinSource(
        SERIALS,
        image_client=_FakeImages(frame),
        marker_client=_FakeMarkers(robot_xy),
        length=MODEL.length,
        width=MODEL.width,
        height=MODEL.height,
        wall=MODEL.wall,
        search_bounds=SEARCH,
        num_frames=1,
        frame_period_s=0.0,
        failure_dir=failure_dir,
        method=method,
    )


def test_the_ceiling_bin_source_runs_rim_edges(far_frame):
    """CeilingBinSource(method="rim_edges") returns the rim-edge fit."""
    fit = _source(far_frame, (1.1, -0.78)).localize()
    assert _dist(fit, FAR_SPOT) < 0.03


def test_rim_edges_still_excludes_the_robot(far_frame, tmp_path):
    """A bin under the robot's reported pose is ignored, as before; the miss saves its
    frames and the detector's scores."""
    source = _source(far_frame, FAR_SPOT, failure_dir=tmp_path)
    with pytest.raises(RuntimeError, match="no bin"):
        source.localize()
    (saved,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(list(saved.glob("frame_*.png"))) == 1
    info = json.loads((saved / "localization.json").read_text())
    assert info["method"] == "rim_edges"
    assert info["exclusions"] == [[FAR_SPOT[0], FAR_SPOT[1], 0.45]]
    scores = info["rim_edges"]
    assert scores["reason"] != "ok"
    for key in ("score", "dark", "margin_ok", "min_score", "max_darkness"):
        assert key in scores
    if scores.get("x") is not None:
        assert math.hypot(scores["x"] - FAR_SPOT[0], scores["y"] - FAR_SPOT[1]) > 0.3


def test_an_unknown_method_is_refused(far_frame):
    """Only rim_edges and saturation exist."""
    with pytest.raises(ValueError, match="method"):
        _source(far_frame, (1.1, -0.78), method="colour")


def test_the_skill_server_config_selects_rim_edges():
    """conf/skill_server.yaml runs the rim-edge detector on the real bin; the saturation
    thresholds stay beside it as the fallback."""
    from hydra import (  # pylint: disable=import-outside-toplevel
        compose,
        initialize_config_dir,
    )

    conf = Path(__file__).resolve().parent.parent / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf)):
        cfg = compose(config_name="skill_server", overrides=["mode=fake"])
    real = cfg.bin.pipelines.real
    assert cfg.bin.method == "rim_edges"
    assert real.method == "rim_edges"
    assert real.rim_min_score == pytest.approx(0.014)
    assert real.rim_max_darkness == pytest.approx(-0.3)
    assert dict(real.min_saturation) == {"AD8A29AE": 160.0, "5A5C29AE": 130.0}
