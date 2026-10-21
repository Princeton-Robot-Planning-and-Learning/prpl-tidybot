"""The real2sim benchmark of the pick out of the bin
(:mod:`prpl_tidybot.in_bin_pick_benchmark`): every real in-bin cube rest so far
(tests/data/in_bin_rests.json), replayed through the skill server's own pick in
``mode=sim``, against what the real robot did."""

import math

import pytest

from prpl_tidybot.gripper_shape import FINGER_LINKS
from prpl_tidybot.in_bin_pick_benchmark import (
    CURRENT,
    VARIANTS,
    Controller,
    load_cases,
    run_case,
    summarize,
)

CASES = load_cases()
BY_ID = {c["id"]: c for c in CASES}
WITH_VERDICT = [c["id"] for c in CASES if c["real"]["verdict"] is not None]
PICKED = [c["id"] for c in CASES if c["real"]["verdict"] == "picked"]

# The current controller in sim disagrees with the real robot here, and why. Each is
# a sim-to-real finding, kept visible rather than tuned away.
KNOWN_DISAGREEMENTS: dict[str, str] = {
    "2026-10-20/11-56-48#c1-2": (
        "run 39: the real pick was planned and started, but the wrist camera's "
        "lateral servo found the cube ~3 cm from the ceiling estimate and its next "
        "step was refused by the wall check (the gripper box within 6 mm of the far "
        "wall; run 39 also ran the old gripper box and the 0.40 x 0.32 x 0.19 "
        "placeholder bin), so the gripper never closed. Sim mode has no wrist servo: "
        "it descends on the ceiling estimate and holds."
    ),
}

pytestmark = pytest.mark.xdist_group("in_bin_pick_benchmark")


@pytest.fixture(scope="module", name="results")
def _results():
    """The current controller on every case, once per module."""
    return {}


def _result(results, case_id):
    if case_id not in results:
        results[case_id] = run_case(BY_ID[case_id], CURRENT)
    return results[case_id]


def test_the_fixture_is_every_near_bin_rest_with_its_real_verdict():
    """14 rests: run 10's 5 refused (2.3-5.6 cm from a wall) and 2 picked (9.8 and 10.0
    cm), run 47's pick, runs 39-40's hand-placed looks and run 38's last landing (no
    pick followed).

    None lies past the barrier.
    """
    assert len(CASES) == 14
    run10 = [c for c in CASES if c["source"]["session"] == "2026-10-21/10-29-48"]
    refused = sorted(
        round(100 * c["wall_distance_m"], 1)
        for c in run10
        if c["real"]["verdict"] == "refused"
    )
    picked = sorted(
        round(100 * c["wall_distance_m"], 1)
        for c in run10
        if c["real"]["verdict"] == "picked"
    )
    assert refused == [2.3, 2.4, 3.6, 5.3, 5.6]
    assert picked == [9.8, 10.0]
    counts = {
        v: sum(c["real"]["verdict"] == v for c in CASES)
        for v in ("refused", "picked", "executed_missed", None)
    }
    assert counts == {"refused": 8, "picked": 4, "executed_missed": 1, None: 1}
    for c in CASES:
        # The robot's side of the tape (y = 0.349 + 0.0267 x).
        assert c["cube"]["y"] < 0.349 + 0.0267 * c["cube"]["x"]
        assert 0.0 < c["wall_distance_m"] < 0.15
        for key in ("x", "y", "yaw", "length", "width", "height", "wall"):
            assert math.isfinite(c["bin"][key])
        assert len(c["robot"]) == 3 and len(c["arm_seed"]) == 7


@pytest.mark.parametrize("case_id", WITH_VERDICT)
def test_the_current_controller_in_sim_reproduces_the_real_verdict(results, case_id):
    """Refused where the real robot refused, held where it held; a known disagreement
    must still disagree (or be removed from the list)."""
    result = _result(results, case_id)
    if case_id in KNOWN_DISAGREEMENTS:
        assert not result["agrees_with_real"], KNOWN_DISAGREEMENTS[case_id]
    else:
        assert result["agrees_with_real"], (result["verdict"], result["reason"])


@pytest.mark.parametrize("case_id", PICKED)
def test_every_real_pick_is_picked_and_held_in_sim(results, case_id):
    """The cubes the real robot lifted out of the bin are lifted above the rim in sim
    too, through the waypoint above it."""
    result = _result(results, case_id)
    assert result["verdict"] == "held"
    assert result["lifted_above_rim"]
    steps = [s["step"] for s in result["steps"]]
    assert steps.index("above") < steps.index("hover") < steps.index("grasp")


def test_a_refusal_carries_the_planners_reason_and_moves_nothing(results):
    """The planner's refusal is the verdict, with its text; the arm never moves."""
    case_id = next(i for i in WITH_VERDICT if BY_ID[i]["real"]["verdict"] == "refused")
    result = _result(results, case_id)
    assert result["verdict"] == "refused"
    assert result["reason"].startswith("cube too close to the bin wall to grasp")
    assert result["steps"] == []


def test_the_controller_under_test_is_selectable_by_config():
    """A candidate is config overrides on the current controller: a wall margin no grasp
    can keep turns a real pick into a refusal."""
    case = BY_ID[PICKED[0]]
    strict = Controller(
        name="strict", overrides=("pick.top_down.in_bin.wall_margin=0.2",)
    )
    assert run_case(case, strict)["verdict"] == "refused"


def test_wall_contacts_are_reported_by_link_and_depth(results):
    """Run 10's 2.3 cm rest, with the wall check switched off (a negative margin):

    the arm goes in anyway and the TidyBot model's gripper meets the wall, reported by
    link and by how far past the wall's face. A real pick touches nothing.
    """
    unchecked = Controller(
        name="unchecked", overrides=("pick.top_down.in_bin.wall_margin=-0.3",)
    )
    result = run_case(BY_ID["2026-10-21/10-29-48#87"], unchecked)
    assert result["verdict"] != "refused"
    assert result["wall_contacts"]
    assert result["min_wall_clearance_m"]["gripper"] < 0
    for contact in result["wall_contacts"]:
        assert set(contact) >= {"step", "link", "depth_m"}
        assert contact["depth_m"] > 0
    assert {c["step"] for c in result["wall_contacts"]} & {"hover", "grasp"}
    clean = _result(results, PICKED[0])
    assert clean["wall_contacts"] == []
    assert clean["min_wall_clearance_m"]["gripper"] > 0


def test_the_summary_counts_agreement_as_x_of_y(results):
    """Counts come as [x, y]."""
    picked = [_result(results, i) | {"id": i} for i in PICKED]
    summary = summarize(picked, CASES)
    assert summary["real_picked_held_in_sim"] == [len(PICKED), len(PICKED)]


# The candidate controllers on the real rests (sim mode; a "held" is the kinematic
# sim's geometric hold check, with walls that stop nothing).
A_RESCUES = {
    "2026-10-20/11-56-48#c1-1",  # run 39, 6.4 cm, tilt +15 at 70 mm
    "2026-10-20/12-34-53#c1-1",  # run 40, 4.4 cm, square to the wall, tilt +10
    "2026-10-20/12-34-53#c1-2",  # run 40, 4.2 cm, square to the wall, tilt -20
}
REAL_PICKS_AND_REFUSALS = [
    c["id"] for c in CASES if c["real"]["verdict"] in ("picked", "refused")
]


@pytest.mark.parametrize("case_id", REAL_PICKS_AND_REFUSALS)
def test_the_tilted_grasp_keeps_every_real_pick_and_rescues_none_of_run_10(case_id):
    """A (tilts 0/+-10/+-15/+-20 deg, 85 then 70 mm, the real gripper's links): the 4
    real picks are still held, untilted and fully open; of the 8 real refusals only runs
    39-40's 3 hand-placed cubes are rescued, none of run 10's 5 toss rests (2.3-5.6 cm
    from a wall, 24-60 deg to it); no pose touches a wall in sim."""
    result = run_case(BY_ID[case_id], VARIANTS["A"])
    real = BY_ID[case_id]["real"]["verdict"]
    if real == "picked":
        assert result["verdict"] == "held"
        assert (result["tilt_deg"], result["open_position"]) == (0.0, 0.0)
    else:
        assert (result["verdict"] == "held") == (case_id in A_RESCUES)
    assert result["wall_contacts"] == []


def test_light_finger_contact_rescues_a_run_10_refusal_by_touching_the_wall():
    """B (finger links up to 1 cm past the inner face): run 10's 5.6 cm rest is reached,
    and it is reached by the finger linkage going into the wall in sim:

    what B buys is contact, which the sim's walls do not resist.
    """
    result = run_case(BY_ID["2026-10-21/10-29-48#96"], VARIANTS["B1.0"])
    assert result["verdict"] == "held"
    links = {c["link"] for c in result["wall_contacts"]}
    assert links and links <= FINGER_LINKS
