# Test doubles retain production signatures and pytest fixture names.
# pylint: disable=unused-argument,redefined-outer-name,protected-access
"""Offline failure-ordering tests; never import a robot interface."""

import threading
from dataclasses import replace

import pytest

from prpl_tidybot.sprout.offline_server import SYNTHETIC_GEOMETRY, FakeBackend
from prpl_tidybot.sprout.sequence import (
    ForearmSkill,
    GraspPose,
    Recipe,
    Stopped,
    make_plan,
)


def test_cycle_and_map_vertical_geometry():
    """Verify cycle and map vertical geometry."""
    backend = FakeBackend(0)
    result = ForearmSkill(backend, Recipe(0.12, 0.08), SYNTHETIC_GEOMETRY).cycle()
    assert result["completed"] == [
        "pregrasp",
        "align",
        "descend",
        "close",
        "lift",
        "lower",
        "open",
        "retreat",
    ]
    plan = result["plan"]
    assert plan["lifted"]["z"] - plan["grasp"]["z"] == pytest.approx(0.15)
    assert plan["put_down"] == plan["grasp"]
    assert plan["retreat"]["z"] == pytest.approx(0.18)
    assert backend.events.index(
        "synthetic_support_confirmation"
    ) < backend.events.index("open")


@pytest.mark.parametrize("confirmed", ["grasp_confirmed", "support_confirmed"])
def test_failed_confirmation_never_releases_or_retracts(confirmed):
    """Verify failed confirmation never releases or retracts."""
    backend = FakeBackend(0)
    setattr(backend, confirmed, lambda: False)
    with pytest.raises(RuntimeError):
        ForearmSkill(backend, Recipe(0.1, 0.1), SYNTHETIC_GEOMETRY).cycle()
    assert backend.closed
    assert "open" not in backend.events and "retreat" not in backend.events
    assert "stop" in backend.events
    if confirmed == "grasp_confirmed":
        assert "lift" not in backend.events


def test_stop_latched_before_cycle():
    """Verify stop latched before cycle."""
    backend = FakeBackend(0)
    skill = ForearmSkill(backend, Recipe(0.1, 0.1), SYNTHETIC_GEOMETRY)
    skill.stop()
    with pytest.raises(Stopped):
        skill.cycle()
    assert set(backend.events) == {"stop"}


def test_stop_interrupts_inflight_backend():
    """Verify stop interrupts inflight backend."""
    backend = FakeBackend(10)
    skill = ForearmSkill(backend, Recipe(0.1, 0.1), SYNTHETIC_GEOMETRY)
    failed = []

    def run():
        """Provide the run test fixture or callback."""
        try:
            skill.cycle()
        except Stopped:
            failed.append(True)

    worker = threading.Thread(target=run)
    worker.start()
    skill.stop()
    worker.join(timeout=1)
    assert not worker.is_alive() and failed == [True]
    assert "close" not in backend.events


def test_alignment_cannot_change_support_height():
    """Verify alignment cannot change support height."""
    backend = FakeBackend(0)
    backend.align = lambda pose: replace(backend.grasp, z=0.2)
    with pytest.raises(ValueError, match="support height"):
        ForearmSkill(backend, Recipe(0.1, 0.1), SYNTHETIC_GEOMETRY).cycle()
    assert "descend" not in backend.events


@pytest.mark.parametrize("height", [0, -0.1, float("nan"), float("inf"), 0.16])
def test_invalid_lift(height):
    """Verify invalid lift."""
    with pytest.raises(ValueError):
        make_plan(GraspPose(0.5, 0, 0.1, 0), Recipe(0.1, 0.1, height))


@pytest.mark.parametrize(
    "changes", [{"frame": "tool"}, {"target": "right_arm"}, {"z": 0}]
)
def test_invalid_target(changes):
    """Verify invalid target."""
    with pytest.raises(ValueError):
        make_plan(replace(GraspPose(0.5, 0, 0.1, 0), **changes), Recipe(0.1, 0.1))


def test_active_assistance_refused():
    """Verify active assistance refused."""
    with pytest.raises(ValueError, match="Active"):
        Recipe(0.1, 0.1, sprout_mode="active").validate()
