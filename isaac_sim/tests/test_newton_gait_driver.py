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


# ------------------------------------------- attitude stabilization (injected)
#
# The closed loop is what turns the measured 71% upright (open loop) into
# 34/34 (100%) — see examples/NEWTON-GAIT-FINDINGS.md tick 19.  The controller
# itself is tritium_lib's AttitudeStabilizer and arrives INJECTED, so these
# tests bind mocks exactly the way the live runner binds the lib — and pin
# that a scheduler WITHOUT the injection is byte-identical to the old one.

def _open_loop_reference(targets_fn, dt, t0, limits, n):
    """The pre-hook contract, composed from the public pieces: sample the
    injected trajectory on an ACCUMULATED clock (t += dt, the scheduler's
    float behavior since day one), convert, clamp.  Byte-identical is
    measured against THIS, not against another scheduler instance."""
    out, t = [], float(t0)
    for _ in range(n):
        out.append((t, drv.radians_to_drive_targets(targets_fn(float(t)),
                                                    limits)))
        t = t + float(dt)
    return out


def test_no_stabilize_fn_is_byte_identical_to_the_open_loop_contract():
    dt, t0, n = 1.0 / 60.0, 0.25, 24
    sched = drv.GaitScheduler(drv.mock_targets_fn, dt=dt, t0=t0,
                              limits=drv.DEFAULT_LIMITS_DEG)
    got = [sched.step() for _ in range(n)]
    assert got == _open_loop_reference(drv.mock_targets_fn, dt, t0,
                                       drv.DEFAULT_LIMITS_DEG, n)
    assert sched.stabilized == 0


def test_stabilize_fn_injected_but_never_fed_changes_nothing():
    """Injection alone must not alter output — only a measured attitude may."""
    dt, n = 0.02, 16
    plain = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG)
    wired = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG,
                              stabilize_fn=drv.mock_stabilize_fn)
    assert [wired.step() for _ in range(n)] == [plain.step() for _ in range(n)]
    assert wired.stabilized == 0


def test_attitude_without_stabilize_fn_raises_loudly():
    """A consumer who feeds attitude into an open-loop scheduler believes the
    loop is closed while running the 71% arm — the exact silent trap the hook
    exists to remove, so it must be loud."""
    sched = drv.GaitScheduler(drv.mock_targets_fn, dt=0.02)
    with pytest.raises(ValueError, match="no stabilize_fn"):
        sched.step(attitude=(0.1, 0.0))


def test_stabilize_fn_must_be_callable():
    with pytest.raises(TypeError, match="stabilize_fn"):
        drv.GaitScheduler(drv.mock_targets_fn, stabilize_fn="not-callable")


def test_first_measured_step_is_dry_and_dt_spans_measured_steps():
    """Mirrors the example driver exactly: the first measured step only
    records its time (no rate basis -> no trim); after a skipped measurement
    the stabilizer receives the TRUE interval since the last measured step."""
    calls: list[tuple[dict, object, float]] = []

    def spy(targets_rad, attitude, dt):
        calls.append((dict(targets_rad), attitude, dt))
        return targets_rad

    att = (0.05, -0.02)
    sched = drv.GaitScheduler(lambda t: {"FL_thigh": 0.5}, dt=0.25,
                              stabilize_fn=spy)
    _, deg1 = sched.step(attitude=att)          # t=0.00: dry — records time
    sched.step()                                # t=0.25: no measurement
    sched.step(attitude=att)                    # t=0.50: dt spans two steps
    sched.step(attitude=att)                    # t=0.75: dt is one step
    assert deg1["FL_thigh"] == pytest.approx(math.degrees(0.5))
    assert len(calls) == 2 and sched.stabilized == 2
    assert calls[0][2] == pytest.approx(0.50)   # since the DRY measured step
    assert calls[1][2] == pytest.approx(0.25)
    assert calls[0][1] is att                   # attitude is opaque passthrough
    assert calls[0][0] == {"FL_thigh": 0.5}     # raw RADIANS, pre-conversion


def test_trim_is_applied_before_conversion_and_clamping():
    """The injected trim lands on RADIAN targets; the actuator envelope still
    has the last word on the DEGREE output."""
    def add_trim(targets_rad, attitude, dt):
        targets_rad["FL_thigh"] += attitude     # attitude doubles as the trim
        return targets_rad

    sched = drv.GaitScheduler(lambda t: {"FL_thigh": 0.8}, dt=0.1,
                              limits={"thigh": (0.0, 103.0)},
                              stabilize_fn=add_trim)
    sched.step(attitude=0.0)                    # dry
    _, deg = sched.step(attitude=0.1)           # moderate trim: converts
    assert deg["FL_thigh"] == pytest.approx(math.degrees(0.9))
    _, deg = sched.step(attitude=10.0)          # absurd trim: clamp wins
    assert deg["FL_thigh"] == 103.0


def test_zero_attitude_error_is_a_no_op():
    dt, n = 0.02, 12
    plain = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG)
    level = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG,
                              stabilize_fn=drv.mock_stabilize_fn)
    got = [level.step(attitude=(0.0, 0.0)) for _ in range(n)]
    assert got == [plain.step() for _ in range(n)]
    assert level.stabilized == n - 1            # the loop RAN, trimming zero


def test_synthetic_roll_trims_the_two_sides_apart():
    """+roll is right-side-down: the right calves must extend (trim UP from
    the open-loop track) and the left calves retract — the lib's pinned sign
    convention, visible through the whole scheduler pipeline."""
    dt, n = 0.02, 12
    plain = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG)
    rolled = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                               limits=drv.DEFAULT_LIMITS_DEG,
                               stabilize_fn=drv.mock_stabilize_fn)
    open_run = [plain.step() for _ in range(n)]
    closed_run = [rolled.step(attitude=(0.1, 0.0)) for _ in range(n)]
    assert closed_run[0] == open_run[0]         # dry first step
    for (_, deg_o), (_, deg_c) in zip(open_run[1:], closed_run[1:]):
        for leg, sign in (("FR", 1), ("RR", 1), ("FL", -1), ("RL", -1)):
            delta = deg_c[f"{leg}_calf"] - deg_o[f"{leg}_calf"]
            assert delta * sign > 0.0, f"{leg}_calf trimmed the wrong way"
        for joint, val in deg_c.items():        # clamps still respected
            lim = drv.limit_for(joint, drv.DEFAULT_LIMITS_DEG)
            assert lim[0] <= val <= lim[1]


def test_run_stays_open_loop():
    """run() samples no attitude, so an injected stabilizer must never fire."""
    def explode(targets_rad, attitude, dt):     # pragma: no cover
        raise AssertionError("stabilize_fn fired during run()")

    sched = drv.GaitScheduler(drv.mock_targets_fn, dt=0.05,
                              stabilize_fn=explode)
    assert len(list(sched.run(0.5))) == 10
    assert sched.stabilized == 0


# --------------------------------------------------- step reflex (injected)
#
# The trim is an ankle strategy: it re-weights planted feet; a push beyond
# what it can absorb inverts the body anyway — the old "~5 N*s inverts it"
# figure was a capture artifact (mid-run --capture stalls the control
# callback: capture-on 0/8 vs capture-free 8/8 upright; capture-free, a
# 5 N*s push survives 6/10) — and the only recovery is to MOVE a foot
# (capture-point stepping, the lib's StepReflex — which gates on deviation
# from a REQUIRED ``nominal_vel_xy`` after live Newton disproved absolute
# gating: 6/6 upright fell to 0/6 undisturbed).  lib has since MEASURED the
# deviation gate unfit for a WALKING gait as well (over threshold on 100.0%
# of undisturbed walking ticks at the legal nominal; live A/B 6/6 -> 0/5,
# Fisher p = 0.0022 — the authoritative verdict lives in
# tritium_lib.control.step_reflex; standing bodies only).  These tests prove
# the injection SEAM, not the reflex: like the stabilizer the reflex arrives
# INJECTED, so they bind mocks exactly the way a live runner binds the lib,
# pin the mirrored semantics (dry first measured step,
# positive-interval-only, opaque measurement, raise-on-misuse, byte-identity
# when absent) and pin the composition ORDER: reflex placement first,
# stabilizer height trim second, conversion + clamp last.

def test_reflex_fn_injected_but_never_fed_changes_nothing():
    """Injection alone must not alter output — only a measured velocity may.
    Wired alongside the stabilizer, an unfed reflex leaves the whole
    scheduler byte-identical to a plain one."""
    dt, n = 0.02, 16
    plain = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG)
    wired = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG,
                              stabilize_fn=drv.mock_stabilize_fn,
                              reflex_fn=drv.mock_reflex_fn)
    assert [wired.step() for _ in range(n)] == [plain.step() for _ in range(n)]
    assert wired.reflexed == 0 and wired.stabilized == 0


def test_velocity_without_reflex_fn_raises_loudly():
    """A consumer who feeds velocity into a reflex-less scheduler believes
    push recovery is armed while it is absent — they would learn the truth
    at the first real shove, so the seam must be loud instead.  An injected
    stabilizer does NOT arm the reflex; the hooks are independent."""
    sched = drv.GaitScheduler(drv.mock_targets_fn, dt=0.02)
    with pytest.raises(ValueError, match="no reflex_fn"):
        sched.step(velocity=(0.5, 0.0))
    stab_only = drv.GaitScheduler(drv.mock_targets_fn, dt=0.02,
                                  stabilize_fn=drv.mock_stabilize_fn)
    with pytest.raises(ValueError, match="no reflex_fn"):
        stab_only.step(attitude=(0.0, 0.0), velocity=(0.5, 0.0))


def test_reflex_fn_must_be_callable():
    with pytest.raises(TypeError, match="reflex_fn"):
        drv.GaitScheduler(drv.mock_targets_fn, reflex_fn="not-callable")


def test_first_measured_reflex_step_is_dry_and_dt_spans_measured_steps():
    """Mirrors the stabilizer hook exactly: the first measured reflex step
    only records its time (no interval -> no reflex); after a skipped
    measurement the reflex receives the TRUE interval since the last
    reflex-measured step."""
    calls: list[tuple[dict, object, float]] = []

    def spy(targets_rad, velocity, dt):
        calls.append((dict(targets_rad), velocity, dt))
        return targets_rad

    vel = (0.5, -0.1)
    sched = drv.GaitScheduler(lambda t: {"FL_thigh": 0.5}, dt=0.25,
                              reflex_fn=spy)
    _, deg1 = sched.step(velocity=vel)          # t=0.00: dry — records time
    sched.step()                                # t=0.25: no measurement
    sched.step(velocity=vel)                    # t=0.50: dt spans two steps
    sched.step(velocity=vel)                    # t=0.75: dt is one step
    assert deg1["FL_thigh"] == pytest.approx(math.degrees(0.5))
    assert len(calls) == 2 and sched.reflexed == 2
    assert calls[0][2] == pytest.approx(0.50)   # since the DRY measured step
    assert calls[1][2] == pytest.approx(0.25)
    assert calls[0][1] is vel                   # velocity is opaque passthrough
    assert calls[0][0] == {"FL_thigh": 0.5}     # raw RADIANS, pre-conversion


def test_reflex_fires_above_the_gate_and_not_below():
    """The lib's layering contract, visible through the scheduler: below the
    MOCK's own 0.05 m absolute-capture gate (NOT the lib's — the lib gates
    on deviation from nominal) the reflex is a pure pass-through (byte-identical
    output even though it RAN), above it the stepping leg's placement leaves
    the open-loop track while the clamps keep the last word."""
    dt, n = 0.02, 12
    plain = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG)
    open_run = [plain.step() for _ in range(n)]

    below = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG,
                              reflex_fn=drv.mock_reflex_fn)
    assert [below.step(velocity=(0.2, 0.0)) for _ in range(n)] == open_run
    assert below.reflexed == n - 1              # it ran, deciding "no step"

    above = drv.GaitScheduler(drv.mock_targets_fn, dt=dt,
                              limits=drv.DEFAULT_LIMITS_DEG,
                              reflex_fn=drv.mock_reflex_fn)
    shoved_run = [above.step(velocity=(0.5, 0.1)) for _ in range(n)]
    assert shoved_run[0] == open_run[0]         # dry first step
    for (_, deg_o), (_, deg_c) in zip(open_run[1:], shoved_run[1:]):
        assert deg_c["FL_thigh"] != deg_o["FL_thigh"]   # placement moved
        for joint, val in deg_c.items():        # clamps still respected
            lim = drv.limit_for(joint, drv.DEFAULT_LIMITS_DEG)
            assert lim[0] <= val <= lim[1]
    assert above.reflexed == n - 1


def test_reflex_placement_applies_before_the_stabilizer_height_trim():
    """The PINNED composition order: targets -> reflex_fn -> stabilize_fn ->
    convert/clamp.  The step strategy (coarse placement, relocates the
    support polygon) runs first; the ankle strategy (fine height trim on
    whatever support exists) runs last, because its Jacobian is evaluated at
    the commanded pose — the stabilizer must SEE the post-reflex targets or
    a stepped leg's trim is computed stale and clobbered."""
    order: list[str] = []

    def reflex(targets_rad, velocity, dt):
        order.append("reflex")
        targets_rad["FL_thigh"] += 0.10         # coarse placement change
        return targets_rad

    def stabilize(targets_rad, attitude, dt):
        order.append("stabilize")
        assert targets_rad["FL_thigh"] == pytest.approx(0.60), \
            "stabilizer must receive the REFLEX-adjusted pose"
        targets_rad["FL_thigh"] += 0.01         # fine trim on top
        return targets_rad

    sched = drv.GaitScheduler(lambda t: {"FL_thigh": 0.5}, dt=0.25,
                              stabilize_fn=stabilize, reflex_fn=reflex)
    sched.step(attitude=(0.0, 0.0), velocity=(9.0, 0.0))    # both dry
    _, deg = sched.step(attitude=(0.0, 0.0), velocity=(9.0, 0.0))
    assert order == ["reflex", "stabilize"]
    assert deg["FL_thigh"] == pytest.approx(math.degrees(0.61))


def test_run_stays_reflex_free():
    """run() samples no velocity, so an injected reflex must never fire."""
    def explode(targets_rad, velocity, dt):     # pragma: no cover
        raise AssertionError("reflex_fn fired during run()")

    sched = drv.GaitScheduler(drv.mock_targets_fn, dt=0.05,
                              reflex_fn=explode)
    assert len(list(sched.run(0.5))) == 10
    assert sched.reflexed == 0


def test_mock_reflex_fn_is_calm_below_the_gate_and_steps_one_leg_above():
    """Below the gate the mock returns the targets untouched (no bias that
    could mask a broken pass-through path above); above it, ONLY the
    stepping leg's placement joints move — never the calves, never another
    leg."""
    targets = {j: 0.1 * i for i, j in enumerate(drv.JOINT_NAMES)}
    assert drv.mock_reflex_fn(dict(targets), (0.2, 0.0), 0.02) == targets
    out = drv.mock_reflex_fn(dict(targets), (0.5, 0.1), 0.02)
    changed = {j for j in drv.JOINT_NAMES if out[j] != targets[j]}
    assert changed == {"FL_hip", "FL_thigh"}    # the stepping leg only


# ------------------------------------ apply_foot_height_trim (the pure math)

def test_foot_trim_magnitude_pinned_at_the_stand_pose():
    """At stand (thigh +50 deg, calf -100 deg) the leg is symmetric about
    vertical, so j1 = 0 and a dz extension lands entirely on the calf as
    dz / j2 — the closed-form pin for the least-norm split."""
    q1, q2 = math.radians(50.0), math.radians(-100.0)
    j2 = -drv.GO2_CALF_LEN_M * math.sin(q1 + q2)          # +0.1632 m/rad
    out = drv.apply_foot_height_trim(
        {"FL_thigh": q1, "FL_calf": q2}, {"FL": 0.01})
    assert out["FL_calf"] - q2 == pytest.approx(0.01 / j2)  # ~ +0.0613 rad
    assert out["FL_thigh"] == pytest.approx(q1)             # j1 = 0 exactly
    # Retraction mirrors extension.
    neg = drv.apply_foot_height_trim(
        {"FL_thigh": q1, "FL_calf": q2}, {"FL": -0.01})
    assert neg["FL_calf"] - q2 == pytest.approx(-0.01 / j2)


def test_foot_trim_direction_extend_means_foot_further_down():
    """Positive dz must INCREASE the leg's downward reach at the trimmed
    pose — checked against the forward kinematics, not against a sign table."""
    def depth(q1, q2):
        return (drv.GO2_THIGH_LEN_M * math.cos(q1)
                + drv.GO2_CALF_LEN_M * math.cos(q1 + q2))

    q1, q2 = math.radians(35.0), math.radians(-80.0)      # asymmetric pose
    out = drv.apply_foot_height_trim(
        {"RR_thigh": q1, "RR_calf": q2}, {"RR": 0.02})
    assert depth(out["RR_thigh"], out["RR_calf"]) > depth(q1, q2)


def test_foot_trim_clamps_singular_and_partial_legs():
    q1, q2 = math.radians(50.0), math.radians(-100.0)
    # A huge offset saturates at MAX_TRIM_RAD per joint, never beyond.
    big = drv.apply_foot_height_trim(
        {"FL_thigh": q1, "FL_calf": q2}, {"FL": 5.0})
    assert big["FL_calf"] - q2 == pytest.approx(drv.MAX_TRIM_RAD)
    # A straight leg is singular (no joint motion moves the foot vertically):
    # trimming it would divide by ~0, so it is skipped untouched.
    straight = drv.apply_foot_height_trim(
        {"FL_thigh": 0.0, "FL_calf": 0.0}, {"FL": 0.01})
    assert straight == {"FL_thigh": 0.0, "FL_calf": 0.0}
    # A leg absent from the targets is skipped; wired legs still trim; the
    # input dict is never mutated.
    src = {"FL_thigh": q1, "FL_calf": q2}
    out = drv.apply_foot_height_trim(src, {"FL": 0.01, "RR": 0.01})
    assert out["FL_calf"] != q2 and "RR_calf" not in out
    assert src == {"FL_thigh": q1, "FL_calf": q2}


def test_mock_stabilize_fn_is_level_neutral():
    """Zero tilt in, targets out unchanged — the mock must not inject a bias
    that would mask a broken zero-error path in the scheduler tests above."""
    targets = {f"{leg}_{part}": 0.1 * i
               for i, (leg, part) in enumerate(
                   (l, p) for l in drv.LEG_NAMES for p in drv.JOINT_PARTS)}
    assert drv.mock_stabilize_fn(dict(targets), (0.0, 0.0), 0.02) == targets


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
