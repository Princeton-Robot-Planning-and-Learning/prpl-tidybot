"""Fake-mode tests for same-side tossing practice (conf/same_side_practice.yaml):
every cycle localizes the bin and the cube, picks the cube out of the bin or off the
floor, and throws from the toss pose relative to the localized bin."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from spatialmath import SE2

from prpl_tidybot.aimed_toss import Workspace, footprint
from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.interfaces.arm_interface import FakeArmInterface
from prpl_tidybot.interfaces.base_interface import FakeBaseInterface
from prpl_tidybot.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_tidybot.robot_skills import arm_bin_collision
from prpl_tidybot.top_down_pick import forward_kinematics, geometry_from_config
from prpl_tidybot.toss_cycles import run_toss_cycles

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
BIN = (-0.29, -0.77)  # the fake bin: the same-side spot, long side along y
IN_BIN = "[-0.27,-0.75,0.025]"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """The fake arm needs no real-time ticks."""
    monkeypatch.setattr("prpl_tidybot.robot_skills.POLICY_CONTROL_PERIOD", 0.0)


@pytest.fixture(autouse=True)
def _start_at_the_toss_pose(monkeypatch):
    """The fake base starts at the same-side toss pose."""
    original = FakeBaseInterface.__init__

    def init(self):
        original(self)
        self.base_state = SE2(1.10, -0.78, math.pi)
        self.map_base_state = SE2(1.10, -0.78, math.pi)

    monkeypatch.setattr(FakeBaseInterface, "__init__", init)


@pytest.fixture(name="base_log")
def _base_log(monkeypatch):
    """Every map-frame pose the fake base is sent to."""
    log = []
    original = FakeBaseInterface.execute_action

    def execute_action(self, action):
        log.append((float(action.x), float(action.y), float(action.theta())))
        original(self, action)

    monkeypatch.setattr(FakeBaseInterface, "execute_action", execute_action)
    return log


@pytest.fixture(name="tosses")
def _tosses(monkeypatch):
    log = []
    original = FakeArmInterface.toss

    def toss(self, **kwargs):
        log.append(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(FakeArmInterface, "toss", toss)
    return log


class _Operator:
    """Answers every prompt 'y' and keeps them; `on` maps a prompt fragment to an
    action run before answering (e.g. the person moving the cube)."""

    def __init__(self, on=None):
        self.prompts = []
        self.on = on or {}

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for fragment, action in self.on.items():
            if fragment in prompt:
                action()
        return "y"


def _compose(**extra):
    overrides = ["mode=fake", "seed=0", "interactive=true", "settle_after_toss_s=0"]
    overrides += ["cycles=1", "env.cube_pose=[0.6,-0.5,0.025]"]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="same_side_practice", overrides=overrides)


def _record(tmp_path, k=0):
    return json.loads((tmp_path / "cycles.json").read_text())["cycles"][k]


def _bin_fit():
    """The fake bin, as the config builds it."""
    b = _compose().bin
    return BinFit(BIN[0], BIN[1], math.pi / 2, b.length, b.width, b.height, b.wall)


def test_the_layout_is_the_measured_tape_and_the_opposite_sides_geometry():
    """The barrier's robot-side face is the measured tape, y = 0.349 + 0.0267 x, with
    the robot on -y; the aimed toss stands 1.39 m from the bin centre approaching from
    +x at 120 deg/s, 675 ms, the opposite-side layout's geometry."""
    cfg = _compose()
    workspace = Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds)
    for x in (-1.5, 0.23, 1.7):
        assert workspace.barrier.depth((x, 0.349 + 0.0267 * x)) == pytest.approx(
            0.0, abs=0.002
        )
    assert workspace.barrier.depth((0.23, 0.0)) > 0.3
    assert cfg.toss.aimed.standoff == pytest.approx(1.39)
    assert cfg.toss.aimed.direction == pytest.approx(0.0)
    assert (cfg.toss.aimed.speed, cfg.toss.aimed.release_ms) == (120.0, 675.0)
    assert cfg.bin_stop_box is None


def test_a_floor_cube_is_picked_and_thrown_from_beside_the_localized_bin(
    tmp_path, base_log, tosses
):
    """Pick off the floor, then throw from 1.39 m +x of the localized bin, facing it;
    no base pose the robot is sent to touches the bin or the barrier."""
    result = run_toss_cycles(_compose(), log_dir=tmp_path, prompt_fn=_Operator())
    assert result.completed == 1, result.finish_reason
    record = _record(tmp_path)
    assert record["cube_before_pick"]["in_bin"] is False
    x, y, theta = record["aimed"]["pose"]
    assert (x, y) == pytest.approx((BIN[0] + 1.39, BIN[1]))
    assert math.cos(theta - math.pi) == pytest.approx(1.0)
    assert tosses == [{"heading": 0.0, "max_vel": 120.0, "gripper_release_ms": 675}]
    cfg = _compose()
    workspace = Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds)
    assert base_log
    for pose in base_log:
        assert workspace.pose_problem(pose, _bin_fit()) is None, pose


def test_a_cube_in_the_bin_is_picked_through_a_waypoint_above_the_rim(
    tmp_path, base_log
):
    """A cube inside the bin is picked, not handed to the person: the arm goes to a
    waypoint with the fingertips above the rim before the hover and after the lift,
    and the base never touches the bin."""
    cfg = _compose(**{"env.cube_pose": IN_BIN})
    operator = _Operator()
    result = run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 1, result.finish_reason
    assert not [p for p in operator.prompts if "put it on the floor" in p]
    record = _record(tmp_path)
    assert record["cube_before_pick"]["in_bin"] is True
    moves = record["pick_attempts"][-1]["moves"]
    names = [m["name"] for m in moves]
    assert names.index("above") < names.index("hover") < names.index("grasp")
    assert names.index("lift") < len(names) - 1 - names[::-1].index("above")
    geom = geometry_from_config(cfg.pick.top_down)
    above = next(m for m in moves if m["name"] == "above")
    rim = geom.floor_z + cfg.bin.height
    fingertips = above["planned_pinch_z"] - cfg.pick.top_down.in_bin.fingertip_drop
    assert fingertips >= rim + cfg.pick.top_down.in_bin.rim_clearance - 1e-3
    workspace = Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds)
    for pose in base_log:
        assert workspace.pose_problem(pose, _bin_fit()) is None, pose


def test_every_planned_arm_pose_of_the_bin_pick_is_clear_of_the_walls(tmp_path):
    """The pick out of the bin plans no arm pose whose links or gripper box reach a
    wall below the rim."""
    cfg = _compose(**{"env.cube_pose": IN_BIN})
    result = run_toss_cycles(cfg, log_dir=tmp_path, prompt_fn=_Operator())
    assert result.completed == 1, result.finish_reason
    record = _record(tmp_path)
    attempt = record["pick_attempts"][-1]
    geom = geometry_from_config(cfg.pick.top_down)
    base = tuple(attempt["base"])
    for move in attempt["moves"]:
        if move["name"] == "carry":
            continue
        joints = [math.radians(v) for v in move["planned_deg"]]
        problem = arm_bin_collision(
            joints, geom, base, _bin_fit(), cfg.pick.top_down.in_bin
        )
        assert problem is None, (move["name"], problem)


def test_the_collision_check_refuses_a_gripper_over_a_wall():
    """A grasp straddling the bin's wall is a collision; the same grasp at the bin's
    centre, and a pose with the gripper above the rim, are not."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    in_bin = cfg.pick.top_down.in_bin
    from prpl_tidybot.top_down_pick import (  # pylint: disable=import-outside-toplevel
        arm_frame_target,
        top_down_joint_targets,
    )

    base = (BIN[0] + 0.67, BIN[1], math.pi)
    for cube, hits in [((BIN[0], BIN[1]), False), ((BIN[0] + 0.15, BIN[1]), True)]:
        ax, ay = arm_frame_target(base, cube, geom)
        _, grasp = top_down_joint_targets(ax, ay, 0.0, geom)
        assert (
            arm_bin_collision(grasp, geom, base, _bin_fit(), in_bin) is not None
        ) == (hits), cube
    high = top_down_joint_targets(
        *arm_frame_target(base, (BIN[0] + 0.15, BIN[1]), geom),
        0.0,
        dataclasses.replace(geom, hover_height=0.30),
    )[0]
    assert forward_kinematics(high, geom)[0][2] > geom.floor_z + 0.3
    assert arm_bin_collision(high, geom, base, _bin_fit(), in_bin) is None


def test_a_cube_against_the_bin_wall_is_handed_to_the_person(tmp_path, monkeypatch):
    """A cube the gripper cannot reach without touching a wall is not grasped: the
    person is asked to nudge it, and the pick goes on once it has moved."""
    cube = [BIN[0] + 0.12, BIN[1], 0.025]  # 2 cm from the inner wall

    monkeypatch.setattr(ConstantTargetSource, "get_target", lambda self: tuple(cube))

    def nudge():
        cube[0] = BIN[0]

    operator = _Operator(on={"bin wall": nudge})
    result = run_toss_cycles(_compose(), log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 1, result.finish_reason
    assert [p for p in operator.prompts if "bin wall" in p]


def test_a_cube_past_the_barrier_is_handed_to_the_person(tmp_path, monkeypatch):
    """A cube on the far side of the tape is seen but not picked: the person is asked
    to bring it back, and the cycle goes on."""
    cube = [0.3, 0.9, 0.025]
    monkeypatch.setattr(ConstantTargetSource, "get_target", lambda self: tuple(cube))

    def bring_back():
        cube[:2] = [0.6, -0.5]

    operator = _Operator(on={"past the barrier": bring_back})
    result = run_toss_cycles(_compose(), log_dir=tmp_path, prompt_fn=operator)
    assert result.completed == 1, result.finish_reason
    assert [p for p in operator.prompts if "past the barrier" in p]


def test_each_throw_is_scored_by_where_the_next_cycle_finds_the_cube(tmp_path):
    """The next cycle's localization says where the previous throw ended, and whether
    it scored (inside the bin's scoring region); the last one by the final look."""
    result = run_toss_cycles(
        _compose(cycles=2, **{"env.cube_pose": IN_BIN}),
        log_dir=tmp_path,
        prompt_fn=_Operator(),
    )
    assert result.completed == 2, result.finish_reason
    for k in (0, 1):
        landing = _record(tmp_path, k)["landing"]
        assert landing["in_bin"] is True and landing["scored"] is True, landing


def test_the_floor_cube_footprint_check_does_not_block_the_bin_pick():
    """A cube inside the bin is on the robot's side: the barrier does not refuse it."""
    cfg = _compose()
    workspace = Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds)
    assert workspace.cube_problem(json.loads(IN_BIN)[:2], 0.0) is None
    assert min(workspace.barrier.depth(c) for c in footprint((1.10, -0.78, 0.0))) > 0


def test_the_gripper_box_is_long_along_the_fingers_travel():
    """The open 2F-85 spans 14.8 cm along its fingers' travel (the tool's x axis, not
    the IK site's 'closing' axis: the TidyBot model's fingers move along x) and 7.6
    cm across. 7.5 cm from a wall the gripper touches it with the fingers spread
    towards the wall, and not with them along it (there the wrist's bracelet, 4 cm
    below the 21 cm rim at the grasp, is what touches)."""
    cfg = _compose()
    geom = geometry_from_config(cfg.pick.top_down)
    in_bin = cfg.pick.top_down.in_bin
    from prpl_tidybot.top_down_pick import (  # pylint: disable=import-outside-toplevel
        arm_frame_target,
        top_down_joint_targets,
    )

    fit = _bin_fit()
    base = (fit.x + 0.67, fit.y, math.pi)
    cube = (fit.x, fit.y + fit.length / 2 - fit.wall - 0.075)  # along y: the length
    ax, ay = arm_frame_target(base, cube, geom)
    gripper_hits = {}
    for turn in (0.0, math.pi / 2):
        g = dataclasses.replace(geom, closing_turn=turn)
        _, grasp = top_down_joint_targets(ax, ay, 0.0, g)
        _, approach, closing = forward_kinematics(grasp, g)
        fingers = np.cross(closing, approach)[:2]  # the fingers' travel, arm frame
        towards_wall = abs(fingers[1]) > 0.9  # arm y is map -y (the base faces -x)
        problem = arm_bin_collision(grasp, g, base, fit, in_bin) or ""
        gripper_hits[towards_wall] = "the gripper" in problem
    assert gripper_hits == {True: True, False: False}


def test_the_grasp_on_the_other_faces_is_used_when_only_it_clears():
    """With a gripper wide enough that its finger direction decides, a cube near a
    wall gets the grasp turned a quarter (fingers along the wall)."""
    cfg = _compose(
        **{
            "pick.top_down.in_bin.finger_half_span": 0.10,
            "pick.top_down.in_bin.gripper_model": "box",
            "env.cube_pose": IN_BIN,
        }
    )
    geom = geometry_from_config(cfg.pick.top_down)
    fit = _bin_fit()
    base = (fit.x + 0.67, fit.y, math.pi)
    cube = {"x": fit.x, "y": fit.y + fit.length / 2 - fit.wall - 0.10, "yaw_deg": 0.0}
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        _bin_grasp_geometry,
    )
    from prpl_tidybot.third_party.constants import (  # pylint: disable=import-outside-toplevel
        RETRACT_ARM_CONF,
    )

    chosen, problem = _bin_grasp_geometry(
        geom, base, cube, fit, cfg.pick.top_down.in_bin, RETRACT_ARM_CONF
    )
    assert problem is None
    assert chosen is not None and chosen.closing_turn == pytest.approx(math.pi / 2)


# The 2026-10-21 10:39 scene: a floor cube 3 cm outside the bin's +x wall.
WALL_BIN = BinFit(-0.305, -0.395, 1.55, 0.39, 0.29, 0.21, 0.005)
WALL_CUBE = {"x": -0.105, "y": -0.454, "yaw_deg": 0.0, "in_bin": False}
TOSS_POSE = (1.10, -0.78, math.pi)


def _plan(cube, fit=WALL_BIN, **extra):
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        plan_pick,
    )
    from prpl_tidybot.third_party.constants import (  # pylint: disable=import-outside-toplevel
        RETRACT_ARM_CONF,
    )

    cfg = _compose(**extra)
    return plan_pick(
        TOSS_POSE,
        cube,
        geometry_from_config(cfg.pick.top_down),
        tuple(cfg.preview.floor_bounds),
        Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds),
        fit,
        cfg.pick.top_down.in_bin,
        list(RETRACT_ARM_CONF),
    )


def test_a_floor_cube_against_the_bin_wall_outside_is_refused_before_moving():
    """The 10:39 scene: the floor pick beside the bin is checked against its walls,
    and refused (both closing orientations touch the wall) before anything moves."""
    stand, why = _plan(WALL_CUBE)
    assert stand is None
    assert why is not None and why.startswith("cube too close to the bin wall to grasp")


def test_the_floor_pick_refuses_with_the_in_bin_pick_off_too():
    """The bin is an obstacle whether or not the pick out of it is on."""
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        plan_pick,
    )
    from prpl_tidybot.third_party.constants import (  # pylint: disable=import-outside-toplevel
        RETRACT_ARM_CONF,
    )

    cfg = _compose()
    stand, why = plan_pick(
        TOSS_POSE,
        WALL_CUBE,
        geometry_from_config(cfg.pick.top_down),
        tuple(cfg.preview.floor_bounds),
        Workspace.from_config(cfg.barrier, cfg.preview.floor_bounds),
        WALL_BIN,
        None,
        list(RETRACT_ARM_CONF),
    )
    assert stand is None
    assert why is not None and why.startswith("cube too close to the bin wall to grasp")


def test_the_fake_skills_can_pick_refuses_the_cube_against_the_wall():
    """``can_pick`` (the planner's question) refuses the same scene."""
    from prpl_tidybot.skill_server import (  # pylint: disable=import-outside-toplevel
        FakeTossingRobot,
    )

    skills = FakeTossingRobot(
        robot=TOSS_POSE,
        cube=(WALL_CUBE["x"], WALL_CUBE["y"], 0.0),
        bin_fit=WALL_BIN,
    )
    reply = skills.can_pick()
    assert reply["refused"] is True, reply
    assert reply["reason"].startswith("cube too close to the bin wall to grasp")


def test_a_floor_cube_30_cm_from_the_bin_is_still_pickable():
    """The bin wall check does not refuse a floor cube well clear of it."""
    fit = WALL_BIN
    cube = {"x": fit.x + fit.width / 2 + 0.30, "y": -0.454, "yaw_deg": 0.0}
    cube["in_bin"] = False
    stand, why = _plan(cube)
    assert why is None and stand is not None


def test_the_floor_pick_uses_the_other_faces_when_only_they_clear():
    """With a gripper wide enough that its finger direction decides, a floor cube
    beside the wall gets the grasp turned a quarter (fingers along the wall)."""
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        _floor_grasp_geometry,
    )
    from prpl_tidybot.third_party.constants import (  # pylint: disable=import-outside-toplevel
        RETRACT_ARM_CONF,
    )

    # The box model (the mesh, the default since bd85a79, has no finger_half_span).
    cfg = _compose(
        **{
            "pick.top_down.in_bin.finger_half_span": 0.10,
            "pick.top_down.in_bin.gripper_model": "box",
        }
    )
    geom = geometry_from_config(cfg.pick.top_down)
    in_bin = cfg.pick.top_down.in_bin
    fit = _bin_fit()
    base = (fit.x + 0.67, fit.y, math.pi)
    # Beside the +y end wall (the bin's length is along y), outside it.
    cube = {"x": fit.x, "y": fit.y + fit.length / 2 + 0.09, "yaw_deg": 0.0}
    results = {}
    for turn in (0.0, math.pi / 2):
        g = dataclasses.replace(geom, closing_turn=turn)
        results[turn], _ = _floor_grasp_geometry(
            g, base, cube, fit, in_bin, RETRACT_ARM_CONF, turns=(turn,)
        )
    assert results[0.0] is None and results[math.pi / 2] is not None, results
    chosen, problem = _floor_grasp_geometry(
        geom, base, cube, fit, in_bin, RETRACT_ARM_CONF
    )
    assert problem is None
    assert chosen is not None and chosen.closing_turn == pytest.approx(math.pi / 2)
