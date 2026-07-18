#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Newton gait driver — gait trajectory -> per-joint USD drive targets (Go2).

The applier half of the quadruped gait pipeline.  tritium-lib's
``joint_targets_at(t, gait=..., speed=...)`` produces 12 named joint angles in
RADIANS per control tick (keys ``FL/FR/RL/RR`` x ``hip/thigh/calf``, e.g.
``"FL_hip"``); Isaac's USD angular drives want DEGREES.  This module is the
reusable, testable seam between the two: schedule the trajectory over control
steps, convert rad -> deg, clamp into the actuator envelope, and (only inside
Isaac's python) write ``drive:angular:physics:targetPosition`` on the Go2's
RevoluteJoint prims so the Newton solver walks the dog.

Both North Star halves: FUN — the sim dog visibly trots across the tactical
scene instead of standing frozen.  PRODUCTION — the exact rad->deg/clamp/step
scheduling that will feed a physical quadruped's joint controllers is proven
here against Newton before any real actuator moves.

Dependency hygiene (the isaac-bridge rule): this module is tritium-free AND
Isaac-free at module scope — plain ``python3`` imports it with no GPU, no
pxr, no isaacsim, and no tritium_lib on the box.  The trajectory source is an
INJECTED callable ``targets_fn(t) -> dict[joint, radians]`` (the live runner
binds tritium-lib's ``joint_targets_at``; tests and ``--selftest`` bind a
mock), and ``apply_to_stage`` imports ``pxr`` lazily only when actually
applying to a live stage.  No dependency bleed in either direction.

Pieces
------
  * ``radians_to_drive_targets(angles_rad, limits=None)`` — pure rad -> deg
    with optional per-joint clamping (full joint name beats part suffix).
  * ``GaitScheduler(targets_fn, dt=...)`` — walks injected trajectory time in
    fixed control steps, yielding per-step drive-target dicts in DEGREES.
  * ``steps_for_duration`` / ``steps_per_period`` / ``period_s`` — the step
    arithmetic a runner needs to size a run.
  * ``apply_to_stage(stage, drive_targets_deg, joint_prim_paths)`` — the ONLY
    Isaac-touching function: sets USD angular-drive targetPosition (degrees)
    on RevoluteJoint prims.  ``find_joint_prim_paths`` maps joint names to
    prim paths by walking a live stage.

Run
---
    # No-GPU self-test: schedule a mock trajectory, prove the contract
    python3 isaac_sim_addon/connectors/newton_gait_driver.py --selftest

Live wiring (inside Isaac's python, RTX host — see examples/go2_newton_gait.py
for the full kit workflow).  The runner brings the trajectory itself — bind
``joint_targets_at`` from ``tritium_lib.models.gait_trajectory`` (the hygiene
gate keeps that binding OUT of this module)::

    jt = ...  # the runner's own binding of the lib's joint_targets_at
    sched = GaitScheduler(lambda t: jt(t, gait="trot", speed=1.0),
                          dt=1.0 / 60.0, limits=DEFAULT_LIMITS_DEG)
    paths = find_joint_prim_paths(stage, JOINT_NAMES)
    # per physics/app-update step:
    _, targets_deg = sched.step()
    apply_to_stage(stage, targets_deg, paths)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from typing import Callable, Iterator, Mapping

log = logging.getLogger("newton-gait-driver")

DEG_PER_RAD = 180.0 / math.pi

# The Go2 joint set the lib trajectory names — legs FL/FR/RL/RR, each with
# hip (ab/adduction), thigh (hip pitch), calf (knee).
LEG_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")
JOINT_PARTS: tuple[str, ...] = ("hip", "thigh", "calf")
JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{part}" for leg in LEG_NAMES for part in JOINT_PARTS
)

# Actuator envelope in DEGREES, keyed by joint part.  Mirrors the lib's
# Newton-validated JOINT_LIMITS_RAD (hip +/-0.6, thigh 0.0..1.8,
# calf -2.4..-1.2 rad) — duplicated as literals because this module stays
# importable with no tritium_lib on the box (the no-GPU hygiene gate).
DEFAULT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "hip": (-0.6 * DEG_PER_RAD, 0.6 * DEG_PER_RAD),        # +/-34.38 deg
    "thigh": (0.0, 1.8 * DEG_PER_RAD),                     # 0..103.13 deg
    "calf": (-2.4 * DEG_PER_RAD, -1.2 * DEG_PER_RAD),      # -137.51..-68.75
}


# --------------------------------------------------------------------------- #
# Pure conversion + step arithmetic (no Isaac, no GPU — fully unit-tested).
# --------------------------------------------------------------------------- #

def limit_for(joint: str,
              limits: Mapping[str, tuple[float, float]] | None,
              ) -> tuple[float, float] | None:
    """The (lo, hi) clamp for ``joint`` from ``limits``, or None.

    A full-joint key (``"FL_hip"``) wins over a part key (``"hip"``); the part
    is the suffix after the last underscore, so one ``"calf"`` entry covers
    all four calves.  ``None`` limits means no clamping anywhere."""
    if not limits:
        return None
    if joint in limits:
        return limits[joint]
    part = joint.rsplit("_", 1)[-1]
    return limits.get(part)


def radians_to_drive_targets(
    joint_angles_rad: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Convert named joint angles (RADIANS) to USD drive targets (DEGREES).

    ``limits`` (DEGREES, keyed by full joint name or part suffix — see
    :func:`limit_for`) clamps each output into its actuator envelope.  A
    non-finite input angle raises ``ValueError`` — a NaN must never reach a
    physics solver as a drive target."""
    out: dict[str, float] = {}
    for joint, rad in joint_angles_rad.items():
        rad = float(rad)
        if not math.isfinite(rad):
            raise ValueError(f"non-finite angle for {joint!r}: {rad!r}")
        deg = rad * DEG_PER_RAD
        lim = limit_for(joint, limits)
        if lim is not None:
            lo, hi = float(lim[0]), float(lim[1])
            if lo > hi:
                raise ValueError(f"inverted limit for {joint!r}: ({lo}, {hi})")
            deg = min(max(deg, lo), hi)
        out[joint] = deg
    return out


def period_s(stride_hz: float) -> float:
    """One gait period in seconds for a stride frequency in Hz."""
    if stride_hz <= 0.0:
        raise ValueError(f"stride_hz must be > 0, got {stride_hz}")
    return 1.0 / float(stride_hz)


def steps_for_duration(duration_s: float, dt: float) -> int:
    """Control steps needed to cover ``duration_s`` at step size ``dt``."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if duration_s <= 0.0:
        return 0
    return max(1, round(float(duration_s) / float(dt)))


def steps_per_period(stride_hz: float, dt: float) -> int:
    """Control steps in ONE gait period (stride) at step size ``dt``."""
    return steps_for_duration(period_s(stride_hz), dt)


class GaitScheduler:
    """Walk an injected joint-trajectory over fixed control steps.

    ``targets_fn(t) -> dict[joint_name, radians]`` is the ONLY trajectory
    source — the live Isaac runner binds tritium-lib's ``joint_targets_at``
    closure; tests and ``--selftest`` bind a mock.  This class never imports
    the trajectory's home module, so it stays provable with no GPU and no
    tritium_lib installed.

    Each :meth:`step` advances the clock by ``dt`` and returns
    ``(t, drive_targets_deg)`` — the converted, clamped dict ready for
    :func:`apply_to_stage`.  :meth:`run` yields a whole timed run."""

    def __init__(
        self,
        targets_fn: Callable[[float], Mapping[str, float]],
        dt: float = 1.0 / 60.0,
        t0: float = 0.0,
        limits: Mapping[str, tuple[float, float]] | None = None,
    ):
        if not callable(targets_fn):
            raise TypeError("targets_fn must be callable: (t) -> {joint: rad}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.targets_fn = targets_fn
        self.dt = float(dt)
        self.t = float(t0)
        self.limits = limits
        self.steps = 0
        self.joints: tuple[str, ...] = ()   # set from the first sample

    def targets_at(self, t: float) -> dict[str, float]:
        """Drive targets (DEGREES) at absolute trajectory time ``t`` — pure
        lookup, does not advance the scheduler clock."""
        deg = radians_to_drive_targets(self.targets_fn(float(t)), self.limits)
        if not self.joints:
            self.joints = tuple(sorted(deg))
        elif tuple(sorted(deg)) != self.joints:
            raise ValueError(
                f"trajectory joint set changed at t={t:.4f}: "
                f"{sorted(deg)} != {list(self.joints)}"
            )
        return deg

    def step(self) -> tuple[float, dict[str, float]]:
        """Sample at the current clock, then advance by ``dt``.

        Returns ``(t_sampled, drive_targets_deg)`` — call once per control
        step (e.g. per Isaac app-update tick) and hand the dict to
        :func:`apply_to_stage`."""
        t = self.t
        deg = self.targets_at(t)
        self.t = t + self.dt
        self.steps += 1
        return t, deg

    def run(self, duration_s: float) -> Iterator[tuple[float, dict[str, float]]]:
        """Yield ``(t, drive_targets_deg)`` for every control step covering
        ``duration_s`` (see :func:`steps_for_duration`)."""
        for _ in range(steps_for_duration(duration_s, self.dt)):
            yield self.step()


# --------------------------------------------------------------------------- #
# Isaac-side application (lazy pxr import — module imports fine without it).
# --------------------------------------------------------------------------- #

def apply_to_stage(stage, drive_targets_deg: Mapping[str, float],
                   joint_prim_paths: Mapping[str, str]) -> int:
    """Write drive targets (DEGREES) onto USD RevoluteJoint angular drives.

    For each joint in ``drive_targets_deg`` with a path in
    ``joint_prim_paths``, sets ``drive:angular:physics:targetPosition`` on the
    prim (creating the angular DriveAPI if absent) — the attribute the Newton
    solver reads as the joint's position setpoint.  USD angular drives are
    native DEGREES, hence this module's unit boundary.  Returns the number of
    joints applied; joints without a mapped path are skipped (counted out) so
    a partial rig degrades honestly instead of raising mid-run.

    Isaac-only: imports ``pxr`` lazily — calling this outside Isaac's python
    raises ImportError; merely importing this module never does."""
    from pxr import UsdPhysics  # type: ignore    # lazy: Isaac's python only

    applied = 0
    for joint, deg in drive_targets_deg.items():
        path = joint_prim_paths.get(joint)
        if not path:
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            log.warning("no prim at %s for joint %s", path, joint)
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            log.warning("%s is not a RevoluteJoint (joint %s)", path, joint)
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTargetPositionAttr().Set(float(deg))
        applied += 1
    return applied


def find_joint_prim_paths(stage, joint_names,
                          suffix: str = "_joint") -> dict[str, str]:
    """Map gait joint names to RevoluteJoint prim paths on a live stage.

    Walks the stage once and matches each prim whose name is
    ``<joint_name><suffix>`` (Go2 USD convention: ``FL_hip_joint``) or exactly
    ``<joint_name>``.  Returns only the joints actually found — the caller
    compares against its expected set to detect a partial rig.  Isaac-only
    (needs a pxr stage); imported lazily via the stage object itself, so this
    module still loads without pxr."""
    wanted = {f"{n}{suffix}": n for n in joint_names}
    wanted.update({n: n for n in joint_names})
    found: dict[str, str] = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        joint = wanted.get(name)
        if joint is not None and joint not in found:
            found[joint] = str(prim.GetPath())
    return found


# --------------------------------------------------------------------------- #
# Self-test (no GPU, no Isaac, no tritium_lib) + entry point.
# --------------------------------------------------------------------------- #

def mock_targets_fn(t: float) -> dict[str, float]:
    """A stand-in trajectory with the lib contract's exact shape: 12 joints,
    RADIANS, centered on the Newton-validated stand (hip 0 / thigh +50 deg /
    calf -100 deg), thighs swinging +/-0.25 rad at 1 Hz with diagonal pairs
    (FL+RR vs FR+RL) in antiphase — a cartoon trot, NOT the real gait."""
    stand = {"hip": 0.0, "thigh": math.radians(50.0), "calf": math.radians(-100.0)}
    diag = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}
    out: dict[str, float] = {}
    for leg in LEG_NAMES:
        phase = 2.0 * math.pi * t + diag[leg]
        out[f"{leg}_hip"] = stand["hip"] + 0.05 * math.sin(phase)
        out[f"{leg}_thigh"] = stand["thigh"] + 0.25 * math.sin(phase)
        out[f"{leg}_calf"] = stand["calf"] + 0.15 * math.sin(phase + math.pi / 2)
    return out


def selftest(args) -> int:
    """No-GPU: schedule the mock trajectory for a few periods and assert the
    drive-target contract — step count, full 12-joint set every step, finite
    DEGREE values inside the clamp envelope, and the clock actually advances."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dt = 1.0 / args.hz
    stride_hz = 1.0                      # the mock trot's stride frequency
    duration = args.periods * period_s(stride_hz)
    expect_steps = steps_for_duration(duration, dt)
    assert expect_steps == args.periods * steps_per_period(stride_hz, dt)

    sched = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG)
    lo = {j: math.inf for j in JOINT_NAMES}
    hi = {j: -math.inf for j in JOINT_NAMES}
    steps = 0
    last_t = -1.0
    for t, deg in sched.run(duration):
        assert t > last_t, "scheduler clock must advance"
        last_t = t
        assert set(deg) == set(JOINT_NAMES), f"joint set broke at t={t}"
        for joint, val in deg.items():
            assert math.isfinite(val), f"NaN/inf target for {joint} at t={t}"
            lim = limit_for(joint, DEFAULT_LIMITS_DEG)
            assert lim[0] <= val <= lim[1], \
                f"{joint}={val:.2f} deg outside {lim} at t={t}"
            lo[joint] = min(lo[joint], val)
            hi[joint] = max(hi[joint], val)
        steps += 1
    assert steps == expect_steps, f"ran {steps} steps, expected {expect_steps}"
    assert sched.steps == expect_steps
    # Degrees, not radians: the mock thigh centre is 50 deg — a radian-scale
    # bug would read ~0.87.
    thigh_mid = (lo["FL_thigh"] + hi["FL_thigh"]) / 2.0
    assert 40.0 < thigh_mid < 60.0, f"thigh centre {thigh_mid:.2f} not in degrees"

    span = {p: (min(lo[f"{l}_{p}"] for l in LEG_NAMES),
                max(hi[f"{l}_{p}"] for l in LEG_NAMES)) for p in JOINT_PARTS}
    print(f"SELFTEST OK steps={steps} periods={args.periods} dt={dt:.4f}s "
          f"joints={len(JOINT_NAMES)} no_nan clamped "
          + " ".join(f"{p}=[{a:.1f}..{b:.1f}]deg" for p, (a, b) in span.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Newton gait driver: trajectory -> USD drive targets (deg)")
    ap.add_argument("--selftest", action="store_true",
                    help="no-GPU contract check over a mock trajectory")
    ap.add_argument("--hz", type=float, default=60.0,
                    help="control rate (steps per second)")
    ap.add_argument("--periods", type=int, default=3,
                    help="selftest: gait periods to schedule")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args)
    ap.print_help()
    print("\nThis module is a library — the live runner (Isaac's python) "
          "binds a trajectory callable\ninto GaitScheduler and applies each "
          "step via apply_to_stage.  Use --selftest to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
