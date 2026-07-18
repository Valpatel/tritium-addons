# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""No-GPU tests for the Newton gait driver connector.

The driver is the seam between a gait trajectory (injected callable returning
RADIANS) and Isaac's USD angular drives (DEGREES).  Everything here runs with
no Isaac, no pxr, no GPU, and no tritium_lib installed — the injection point
IS the contract, so the tests bind mocks exactly the way the live runner
binds the lib's ``joint_targets_at``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_CONN = Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "connectors"


def _load():
    spec = importlib.util.spec_from_file_location(
        "conn_newton_gait_driver", _CONN / "newton_gait_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drv = _load()


# ------------------------------------------------------- rad -> deg conversion
def test_radians_to_degrees_exact():
    out = drv.radians_to_drive_targets({
        "FL_hip": 0.0,
        "FL_thigh": math.pi / 2,
        "FL_calf": -math.pi,
        "RR_thigh": 1.0,
    })
    assert out["FL_hip"] == 0.0
    assert out["FL_thigh"] == pytest.approx(90.0)
    assert out["FL_calf"] == pytest.approx(-180.0)
    assert out["RR_thigh"] == pytest.approx(57.29577951308232)


def test_conversion_covers_the_full_go2_joint_set():
    angles = {j: 0.1 for j in drv.JOINT_NAMES}
    out = drv.radians_to_drive_targets(angles)
    assert set(out) == set(drv.JOINT_NAMES) and len(out) == 12
    assert all(v == pytest.approx(math.degrees(0.1)) for v in out.values())


def test_nonfinite_angle_raises_never_reaches_the_solver():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            drv.radians_to_drive_targets({"FL_hip": bad})


# ------------------------------------------------------------------- clamping
def test_part_suffix_limit_clamps_all_four_legs():
    angles = {f"{leg}_thigh": math.pi for leg in drv.LEG_NAMES}  # 180 deg ask
    out = drv.radians_to_drive_targets(angles, limits={"thigh": (0.0, 103.0)})
    assert all(out[f"{leg}_thigh"] == 103.0 for leg in drv.LEG_NAMES)


def test_full_joint_name_limit_beats_part_limit():
    angles = {"FL_hip": 1.0, "FR_hip": 1.0}     # ~57.3 deg ask
    out = drv.radians_to_drive_targets(
        angles, limits={"hip": (-90.0, 90.0), "FL_hip": (-5.0, 5.0)})
    assert out["FL_hip"] == 5.0                  # the specific clamp
    assert out["FR_hip"] == pytest.approx(math.degrees(1.0))  # part allows it


def test_no_limits_means_passthrough_and_inverted_limit_raises():
    out = drv.radians_to_drive_targets({"FL_calf": -3.0})
    assert out["FL_calf"] == pytest.approx(math.degrees(-3.0))
    with pytest.raises(ValueError, match="inverted"):
        drv.radians_to_drive_targets({"FL_calf": 0.0},
                                     limits={"calf": (10.0, -10.0)})


def test_default_limits_mirror_the_newton_validated_envelope():
    lim = drv.DEFAULT_LIMITS_DEG
    assert lim["hip"] == (pytest.approx(-34.377, abs=0.01),
                          pytest.approx(34.377, abs=0.01))
    assert lim["thigh"][0] == 0.0
    assert lim["calf"][1] == pytest.approx(math.degrees(-1.2))


# ------------------------------------------------------------- step arithmetic
def test_period_and_step_count_helpers():
    assert drv.period_s(2.0) == pytest.approx(0.5)
    assert drv.steps_per_period(2.0, dt=0.01) == 50
    assert drv.steps_for_duration(1.0, dt=1.0 / 60.0) == 60
    assert drv.steps_for_duration(0.0, dt=0.01) == 0
    with pytest.raises(ValueError):
        drv.period_s(0.0)
    with pytest.raises(ValueError):
        drv.steps_for_duration(1.0, dt=0.0)


# ------------------------------------------------------------------ scheduler
def test_scheduler_uses_the_injected_callable_and_advances_dt():
    seen: list[float] = []

    def targets_fn(t: float) -> dict:
        seen.append(t)
        return {"FL_hip": 0.5, "FL_thigh": t}    # radians

    sched = drv.GaitScheduler(targets_fn, dt=0.25, t0=1.0)
    t1, deg1 = sched.step()
    t2, deg2 = sched.step()
    assert seen == [1.0, 1.25]                   # injected fn got the clock
    assert (t1, t2) == (1.0, 1.25)
    assert deg1["FL_hip"] == pytest.approx(math.degrees(0.5))
    assert deg2["FL_thigh"] == pytest.approx(math.degrees(1.25))
    assert sched.steps == 2


def test_scheduler_run_covers_exactly_one_period():
    stride_hz, dt = 2.0, 0.01                    # period 0.5 s -> 50 steps
    sched = drv.GaitScheduler(drv.mock_targets_fn, dt=dt)
    rows = list(sched.run(drv.period_s(stride_hz)))
    assert len(rows) == drv.steps_per_period(stride_hz, dt) == 50
    assert sched.steps == 50
    times = [t for t, _ in rows]
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(49 * dt)


def test_scheduler_applies_clamps_per_step():
    sched = drv.GaitScheduler(lambda t: {"FL_thigh": 10.0},   # absurd radians
                              dt=0.1, limits={"thigh": (0.0, 103.0)})
    _, deg = sched.step()
    assert deg["FL_thigh"] == 103.0


def test_scheduler_rejects_a_shape_shifting_trajectory():
    def unstable(t: float) -> dict:
        return {"FL_hip": 0.0} if t < 0.05 else {"FR_hip": 0.0}

    sched = drv.GaitScheduler(unstable, dt=0.1)
    sched.step()
    with pytest.raises(ValueError, match="joint set changed"):
        sched.step()


def test_scheduler_rejects_bad_construction():
    with pytest.raises(TypeError):
        drv.GaitScheduler("not-callable")
    with pytest.raises(ValueError):
        drv.GaitScheduler(drv.mock_targets_fn, dt=0.0)


# -------------------------------------------------- stage mapping (duck-typed)
class _FakePrim:
    def __init__(self, name: str, path: str):
        self._name, self._path = name, path

    def GetName(self):
        return self._name

    def GetPath(self):
        return self._path


class _FakeStage:
    def __init__(self, prims):
        self._prims = prims

    def Traverse(self):
        return iter(self._prims)


def test_find_joint_prim_paths_matches_go2_suffix_convention():
    stage = _FakeStage([
        _FakePrim("base", "/World/go2/base"),
        _FakePrim("FL_hip_joint", "/World/go2/base/FL_hip_joint"),
        _FakePrim("RR_calf_joint", "/World/go2/RR_thigh/RR_calf_joint"),
        _FakePrim("FR_thigh", "/World/go2/FR_thigh"),      # exact-name match
        _FakePrim("FL_hip_joint", "/World/dup/FL_hip_joint"),  # first wins
    ])
    found = drv.find_joint_prim_paths(stage, drv.JOINT_NAMES)
    assert found == {
        "FL_hip": "/World/go2/base/FL_hip_joint",
        "RR_calf": "/World/go2/RR_thigh/RR_calf_joint",
        "FR_thigh": "/World/go2/FR_thigh",
    }


# --------------------------------------------------------- hygiene + selftest
def test_module_imports_with_no_isaacsim_and_no_tritium():
    """The hygiene gate in person: loading the driver must drag in neither a
    heavy runtime nor the tritium package — the trajectory arrives injected."""
    for banned in ("isaacsim", "pxr"):
        assert banned not in sys.modules, f"{banned} loaded by import"
    src = (_CONN / "newton_gait_driver.py").read_text()
    needle_import = "import " + "tritium"
    needle_from = "from " + "tritium"
    assert needle_import not in src and needle_from not in src
    # The Isaac-touching surface still exists, just lazily guarded.
    assert hasattr(drv, "apply_to_stage")
    assert hasattr(drv, "find_joint_prim_paths")


def test_apply_to_stage_is_lazy_guarded():
    """Calling the Isaac-only applier outside Isaac's python raises
    ImportError (pxr missing) — it must never no-op silently."""
    if "pxr" in sys.modules:                     # pragma: no cover
        pytest.skip("pxr present — guard untestable here")
    with pytest.raises(ImportError):
        drv.apply_to_stage(_FakeStage([]), {"FL_hip": 0.0},
                           {"FL_hip": "/World/x"})


def test_selftest_passes_gpu_free():
    class _Args:
        hz = 60.0
        periods = 2

    assert drv.selftest(_Args()) == 0
