# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Headless tests for the stand-and-walk scaffold's pure core.

No Isaac, no GPU.  ``build_walk_plan`` is the seam the live Newton runner
stands on — if the plan's step count, joint set, units, or centering are
wrong, the dog falls before physics gets a vote — so those invariants are
proven here with an injected lib-shaped mock, exactly how the live runner
binds tritium-lib's ``joint_targets_at``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WALK_PATH = EXAMPLES / "newton_stand_and_walk.py"

# Module roots that must NOT be dragged in by merely importing the example.
BANNED_ROOTS = ("isaacsim", "pxr", "omni", "carb", "torch",
                "tritium", "tritium_lib")


def _load():
    spec = importlib.util.spec_from_file_location(
        "example_newton_stand_and_walk", WALK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_before_import = set(sys.modules)
walk = _load()
_new_modules = set(sys.modules) - _before_import


def _lib_shaped_mock(t: float, *, gait: str, speed: float | None):
    """The injection contract in person: joint_targets_at's exact signature,
    returning 12 joints in RADIANS via the driver's cartoon trot."""
    assert gait == "trot" and speed == 0.4  # the plan must pass these through
    return walk._drv.mock_targets_fn(t)


DT = 1.0 / 60.0


@pytest.fixture(scope="module")
def plan():
    # 2.0 s = exactly two mock-trot periods (1 Hz), so per-joint means sit
    # on the stand pose to floating-point precision.
    return walk.build_walk_plan(2.0, DT, "trot", 0.4, _lib_shaped_mock)


# ------------------------------------------------------------------- hygiene
def test_module_imports_with_no_isaac_no_pxr_no_tritium():
    """Loading the example (already done above) must not have pulled in any
    heavy runtime or the tritium package — order-independent: we compare
    sys.modules before/after OUR OWN fresh load, so an earlier test importing
    tritium_lib cannot mask or fake a violation."""
    offenders = {m for m in _new_modules
                 if m.split(".")[0] in BANNED_ROOTS}
    assert not offenders, f"module-scope import dragged in: {sorted(offenders)}"


def test_no_banned_imports_at_module_scope_by_ast():
    """Belt to the runtime check's braces: no top-level import statement names
    a banned root.  Function-scoped imports (the guarded live half) are
    exactly what this allows."""
    tree = ast.parse(WALK_PATH.read_text())
    top_level: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level.append(node.module.split(".")[0])
    bad = [m for m in top_level if m in BANNED_ROOTS]
    assert not bad, f"banned module-scope imports: {bad}"


# ---------------------------------------------------------------- stand pose
def test_stand_pose_is_the_newton_validated_numbers():
    assert set(walk.STAND_POSE_DEG) == set(walk.JOINT_NAMES)
    assert len(walk.STAND_POSE_DEG) == 12
    for leg in walk.LEG_NAMES:
        assert walk.STAND_POSE_DEG[f"{leg}_hip"] == 0.0
        assert walk.STAND_POSE_DEG[f"{leg}_thigh"] == 50.0
        assert walk.STAND_POSE_DEG[f"{leg}_calf"] == -100.0


def test_stand_pose_sits_inside_the_actuator_envelope():
    for joint, deg in walk.STAND_POSE_DEG.items():
        lo, hi = walk._drv.limit_for(joint, walk.DEFAULT_LIMITS_DEG)
        assert lo <= deg <= hi, f"{joint} stand {deg} outside [{lo}, {hi}]"


# ------------------------------------------------------------ build_walk_plan
def test_plan_has_the_right_step_count_and_clock(plan):
    assert len(plan) == walk._drv.steps_for_duration(2.0, DT) == 120
    times = [t for t, _ in plan]
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(119 * DT)
    deltas = [b - a for a, b in zip(times, times[1:])]
    assert all(d == pytest.approx(DT) for d in deltas)


def test_plan_steps_carry_12_valid_joints_in_degrees(plan):
    for t, deg in plan:
        assert set(deg) == set(walk.JOINT_NAMES) and len(deg) == 12
        for joint, val in deg.items():
            assert math.isfinite(val), f"NaN/inf for {joint} at t={t}"
            lo, hi = walk._drv.limit_for(joint, walk.DEFAULT_LIMITS_DEG)
            assert lo <= val <= hi, f"{joint}={val:.2f} outside [{lo}, {hi}]"
    # Degrees, not radians: the thigh must swing around +50 deg — a
    # radian-scale bug would put every thigh sample near 0.87.
    thighs = [deg["FL_thigh"] for _, deg in plan]
    assert 35.0 < min(thighs) and max(thighs) < 65.0


def test_plan_is_centered_on_the_stand_pose(plan):
    """Over whole gait periods every joint's mean is its stand angle — the
    trajectory oscillates AROUND the validated stand, it does not drift."""
    for joint in walk.JOINT_NAMES:
        mean = sum(deg[joint] for _, deg in plan) / len(plan)
        assert mean == pytest.approx(walk.STAND_POSE_DEG[joint], abs=0.5), (
            f"{joint} mean {mean:.2f} not centered on stand"
        )


def test_plan_validates_its_inputs():
    with pytest.raises(ValueError, match="duration_s"):
        walk.build_walk_plan(0.0, DT, "trot", 0.4, _lib_shaped_mock)
    with pytest.raises(ValueError):
        walk.build_walk_plan(1.0, 0.0, "trot", 0.4, _lib_shaped_mock)
    with pytest.raises(TypeError, match="callable"):
        walk.build_walk_plan(1.0, DT, "trot", 0.4, "not-callable")


def test_plan_with_the_real_lib_trajectory_if_installed():
    """The actual wiring the live runner uses: tritium-lib's joint_targets_at
    through the addon scheduler.  Skips cleanly on a box without the lib —
    the pure-core tests above stay green regardless."""
    gt = pytest.importorskip("tritium_lib.models.gait_trajectory")
    plan = walk.build_walk_plan(2.0, DT, "trot", 0.35, gt.joint_targets_at)
    assert len(plan) == 120
    for t, deg in plan:
        assert set(deg) == set(walk.JOINT_NAMES)
        for joint, val in deg.items():
            assert math.isfinite(val)
            lo, hi = walk._drv.limit_for(joint, walk.DEFAULT_LIMITS_DEG)
            assert lo <= val <= hi, f"{joint}={val:.2f} outside [{lo}, {hi}]"
    for joint in walk.JOINT_NAMES:
        mean = sum(deg[joint] for _, deg in plan) / len(plan)
        assert abs(mean - walk.STAND_POSE_DEG[joint]) < 10.0, (
            f"{joint} mean {mean:.2f} far from stand"
        )


# ------------------------------------------------------------- helpers + CLI
def test_plan_stats_summarizes_honestly(plan):
    stats = walk.plan_stats(plan)
    assert stats["steps"] == 120 and stats["joints_per_step"] == 12
    assert stats["t0"] == 0.0
    assert stats["part_means_deg"]["thigh"] == pytest.approx(50.0, abs=0.5)
    assert stats["part_means_deg"]["calf"] == pytest.approx(-100.0, abs=0.5)
    assert walk.plan_stats([]) == {"steps": 0}


def test_default_newton_experience_is_a_pure_probe():
    """Callable with no Isaac anywhere: returns None or an existing .kit."""
    exp = walk.default_newton_experience()
    assert exp is None or (exp.endswith(".kit") and Path(exp).is_file())


def test_selftest_passes_gpu_free(capsys):
    assert walk.selftest() == 0
    assert "SELFTEST OK" in capsys.readouterr().out


def test_emit_plan_cli_writes_valid_json_with_mock(tmp_path):
    out = tmp_path / "plan.json"
    rc = walk.main(["--emit-plan", str(out), "--mock",
                    "--seconds", "1.0", "--speed", "0.4"])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == "tritium.walk_plan/1"
    assert doc["stats"]["steps"] == 60          # 1.0 s at 60 Hz
    t0, deg0 = doc["plan"][0]
    assert t0 == 0.0 and len(deg0) == 12
    assert all(math.isfinite(v) for v in deg0.values())
