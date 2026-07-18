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

Closed-loop attitude stabilization — a consumer MUST inject it
--------------------------------------------------------------
The open-loop gait table survives **17/24 trials (71%)** upright; with the
closed attitude loop it survives **34/34 (100%)** across two independent kit
processes, with tilt distributions that do not overlap (5-10 deg closed vs
22-30 deg open — see ``examples/NEWTON-GAIT-FINDINGS.md``, tick 19).  The
controller is ``tritium_lib.control.AttitudeStabilizer`` (PD on body
roll/pitch, distributed to the legs as foot-height offsets); like the
trajectory, it arrives INJECTED so this module stays lib-free:

    ``GaitScheduler(targets_fn, ..., stabilize_fn=...)`` where
    ``stabilize_fn(targets_rad, attitude, dt) -> dict[joint, radians]``

Per control step the runner measures the body attitude and passes it to
``step(attitude=...)``; the scheduler hands the RAW radian targets, the
opaque attitude, and the seconds since the previous measured step to the
injected stabilizer, then converts/clamps its trimmed output.  A scheduler
wired without ``stabilize_fn`` is the OPEN-LOOP arm — any consumer that skips
the injection inherits the 71% number, not the 100% one.  See
:class:`GaitScheduler` for the exact live binding.

Gated step reflex — push recovery arrives INJECTED the same way
---------------------------------------------------------------
The closed attitude loop is an *ankle strategy*: it re-weights feet that are
already planted, and a push above ~5 N*s still inverts the body — the only
recovery is to MOVE a foot under the fall.  ``tritium_lib.control`` ships
``StepReflex`` (capture-point stepping, gated at 0.05 m of capture-point
excursion so an undisturbed walk never crosses it); a live campaign is
measuring it against real Newton physics.  Like the trajectory and the
stabilizer, it arrives INJECTED so this module stays lib-free:

    ``GaitScheduler(..., reflex_fn=...)`` where
    ``reflex_fn(targets_rad, velocity, dt) -> dict[joint, radians]``

Per control step the runner measures the body's horizontal velocity and
passes it to ``step(velocity=...)``; the scheduler hands the RAW radian
targets, the opaque velocity, and the seconds since the previous measured
reflex step to the injected reflex, then feeds its output to the stabilizer
— placement first, height trim second, conversion/clamp last; see
:class:`GaitScheduler` for why that order is pinned.  Below the lib's gate
the reflex is a pure pass-through, so an undisturbed run is byte-identical
with or without it.

Pieces
------
  * ``radians_to_drive_targets(angles_rad, limits=None)`` — pure rad -> deg
    with optional per-joint clamping (full joint name beats part suffix).
  * ``GaitScheduler(targets_fn, dt=..., stabilize_fn=None, reflex_fn=None)``
    — walks injected trajectory time in fixed control steps, yielding
    per-step drive-target dicts in DEGREES; optionally closes the attitude
    loop and/or the step reflex per step.
  * ``apply_foot_height_trim(targets_rad, leg_offsets_m)`` — pure leg-Jacobian
    mapping from per-leg vertical foot offsets (metres) to thigh/calf angle
    trims (radians) — the exact application math the 34/34 example driver
    uses, reusable so a runner's ``stabilize_fn`` is a 3-line closure.
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

# Go2 leg geometry for the attitude trim — real URDF values, duplicated as
# literals for the same reason as the limits above: this module must import
# with no tritium_lib on the box.  Thigh and calf links are the same length;
# placements are (x, y) of each FOOT in the body frame (REP-103: +X forward,
# +Y left), the lever arms the stabilizer's leg-height offsets assume.
GO2_THIGH_LEN_M = 0.213
GO2_CALF_LEN_M = 0.213
GO2_LEG_PLACEMENTS_M: dict[str, tuple[float, float]] = {
    "FL": (0.1881, 0.1300),
    "FR": (0.1881, -0.1300),
    "RL": (-0.1881, 0.1300),
    "RR": (-0.1881, -0.1300),
}
# Per-joint trim clamp (radians), so one bad pose read cannot fling a leg —
# the same 0.30 the measured 34/34 closed-loop runs used.
MAX_TRIM_RAD = 0.30


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


def apply_foot_height_trim(
    targets_rad: Mapping[str, float],
    leg_offsets_m: Mapping[str, float],
    thigh_len_m: float = GO2_THIGH_LEN_M,
    calf_len_m: float = GO2_CALF_LEN_M,
    max_trim_rad: float = MAX_TRIM_RAD,
) -> dict[str, float]:
    """Map per-leg vertical foot offsets (METRES) to thigh/calf trims (RADIANS).

    The application half of the closed attitude loop, verbatim from the
    example driver that measured 34/34 upright (``examples/go2_newton_gait.py``
    ``_apply_trim``), re-keyed by joint NAME because names are this module's
    currency.  For a 2-link planar leg the downward reach is
    ``depth = L1*cos(q1) + L2*cos(q1 + q2)``, so the row Jacobian is
    ``d(depth)/d(q1, q2)`` evaluated at the CURRENTLY commanded pose — the
    gait sweeps the legs through a wide arc, and a Jacobian frozen at stand
    would be badly wrong mid-stride.  The correction is distributed least-norm
    (``J^T * dz / (J . J)``) so both joints share it instead of one joint
    taking all of it and hitting its limit; each joint's trim is clamped to
    ``+/-max_trim_rad`` so one bad pose read cannot fling a leg.

    ``leg_offsets_m`` is keyed by leg name (``"FL"``...); positive extends the
    leg (foot reaches further down, that corner of the body rises) — the
    contract of ``AttitudeCorrection.leg_height_offsets`` in
    ``tritium_lib.control``.  A leg at a singular pose (straight — no small
    joint change moves the foot vertically) is skipped, exactly like the
    example: the pseudo-inverse would divide by ~0 and command an enormous
    trim.  A leg whose thigh/calf keys are absent from ``targets_rad`` is
    skipped too, mirroring the example's wiring-time drop — but note its
    runner then FAILS the trial unless all four legs are wired; a consumer
    trimming a partial rig should check the same or it will read as a weak
    controller instead of a wiring bug.  Returns a new dict; the input is not
    mutated."""
    out = dict(targets_rad)
    for leg, dz in leg_offsets_m.items():
        dz = float(dz)
        if dz == 0.0:
            continue
        key_thigh, key_calf = f"{leg}_thigh", f"{leg}_calf"
        if key_thigh not in out or key_calf not in out:
            continue
        q1 = float(out[key_thigh])
        q2 = float(out[key_calf])
        j1 = -thigh_len_m * math.sin(q1) - calf_len_m * math.sin(q1 + q2)
        j2 = -calf_len_m * math.sin(q1 + q2)
        denom = j1 * j1 + j2 * j2
        if denom < 1e-6:
            continue
        scale = dz / denom
        out[key_thigh] = q1 + max(-max_trim_rad, min(max_trim_rad, j1 * scale))
        out[key_calf] = q2 + max(-max_trim_rad, min(max_trim_rad, j2 * scale))
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
    :func:`apply_to_stage`.  :meth:`run` yields a whole timed run.

    Closed-loop attitude stabilization (INJECTED — required for the 100% arm)
    ------------------------------------------------------------------------
    ``stabilize_fn(targets_rad, attitude, dt) -> dict[joint_name, radians]``
    closes the attitude loop.  Without it this scheduler is the measured
    OPEN-LOOP arm: 17/24 trials (71%) upright vs the closed loop's 34/34
    (100%) — see ``examples/NEWTON-GAIT-FINDINGS.md`` tick 19.  Per step the
    caller measures the body attitude (roll/pitch — how far body-up is from
    world-up) and passes it to ``step(attitude=...)``; the scheduler hands
    the RAW radian targets, the attitude (OPAQUE — never interpreted here),
    and the seconds since the previous measured step to ``stabilize_fn``,
    then converts and clamps its trimmed output.  Semantics mirror the
    example driver that produced the 34/34 exactly:

      * the trim is applied to RADIAN targets BEFORE the rad->deg conversion
        and actuator clamping;
      * the first measured step only records its time — no previous sample
        means no rate estimate, so no trim (``stabilized`` stays 0);
      * ``stabilize_fn`` runs only with a positive ``dt`` since the previous
        measured step, and ``stabilized`` counts exactly those runs;
      * measure the attitude EVERY step — feedback at a logging cadence is an
        order-of-magnitude slower controller than the one the 34/34
        characterized;
      * passing ``attitude`` with no ``stabilize_fn`` injected raises — a
        consumer who believes the loop is closed while running open would
        silently inherit the 71%, the exact trap this hook exists to remove.

    The live binding (runner-side; the hygiene gate keeps it OUT of this
    module) — ``AttitudeStabilizer`` and ``LegPlacement`` come out of
    ``tritium_lib.control``, and the attitude is the body quaternion in WXYZ
    (Isaac's root transform hands back XYZW — the reorder is load-bearing)::

        stab = AttitudeStabilizer(kp=0.8, kd=0.3)
        legs = [LegPlacement(n, x, y)
                for n, (x, y) in GO2_LEG_PLACEMENTS_M.items()]

        def stabilize_fn(targets_rad, quat_wxyz, dt):
            corr = stab.update(quat_wxyz, dt)
            return apply_foot_height_trim(targets_rad,
                                          corr.leg_height_offsets(legs))

        sched = GaitScheduler(jt, dt=1.0 / 60.0, limits=DEFAULT_LIMITS_DEG,
                              stabilize_fn=stabilize_fn)
        # per control step, with a FRESH pose read:
        _, targets_deg = sched.step(attitude=(qw, qx, qy, qz))

    Gated step reflex (INJECTED — push recovery beyond the trim's ceiling)
    ----------------------------------------------------------------------
    ``reflex_fn(targets_rad, velocity, dt) -> dict[joint_name, radians]``
    adjusts foot PLACEMENT — where a swing foot lands — from a measured body
    velocity: the capture-point stepping layer that recovers pushes the
    attitude trim cannot (the trim-only stack inverts above ~5 N*s).  The
    hook mirrors ``stabilize_fn`` exactly:

      * the placement change lands on RADIAN targets BEFORE the rad->deg
        conversion and actuator clamping;
      * the first measured step only records its time — no previous sample
        means no interval, so no reflex (``reflexed`` stays 0);
      * ``reflex_fn`` runs only with a positive ``dt`` since the previous
        measured reflex step, and ``reflexed`` counts exactly those runs
        (its clock is independent of the stabilizer's — measure each at its
        own cadence, though every step is best for both);
      * passing ``velocity`` with no ``reflex_fn`` injected raises — a
        consumer who believes push recovery is armed while it is absent
        would discover the truth at the first real shove, the same silent
        trap the stabilizer hook removes.

    Composition order — PINNED: reflex first, stabilizer second
    -----------------------------------------------------------
    On a step where both hooks run, the scheduler applies
    ``targets -> reflex_fn -> stabilize_fn -> convert/clamp``.  The order
    matches the physics, not a preference:

      * the reflex is the STEP strategy — a coarse, discrete change to
        where a foot lands, relocating the support polygon.  The stabilizer
        is the ANKLE strategy — a fine, continuous re-balance on whatever
        support exists.  The fine layer must act on the final stance, so it
        runs last;
      * the height trim's least-norm split (:func:`apply_foot_height_trim`)
        evaluates the leg Jacobian at the CURRENTLY commanded pose.  Were
        the trim applied first and a reflex placement rewrote that leg's
        targets afterwards, the trim baked into the stepping leg would have
        been computed at a stale pose and then clobbered.  Reflex-first
        means the trim always sees — and corrects — the pose that will
        actually be commanded;
      * the lib's layering contract points the same way: its
        ``ReflexDecision`` passes trim offsets through untouched — the
        reflex never edits the trim.  Running the stabilizer last
        guarantees that structurally: its output goes straight to
        convert/clamp and nothing rewrites it.

    The live binding (runner-side, like the stabilizer's): ``StepReflex``,
    ``ReachLimits`` and ``LegPlacement`` come out of ``tritium_lib.control``;
    the runner owns the IK that turns a ``StepDecision`` landing point into
    swing-leg joint targets, applied at the next swing slot its gait
    allows::

        reflex = StepReflex(com_height_m=0.30)
        legs = [LegPlacement(n, x, y)
                for n, (x, y) in GO2_LEG_PLACEMENTS_M.items()]
        reach = ReachLimits(max_dx=0.10, max_dy=0.06)

        def reflex_fn(targets_rad, vel_xy, dt):
            decision = reflex.decide(vel_xy, legs, reach_limits=reach)
            if decision.step is None:
                return targets_rad        # below the gate: pass-through
            return swing_leg_ik(targets_rad, decision.step)  # runner's IK

        sched = GaitScheduler(jt, dt=1.0 / 60.0, limits=DEFAULT_LIMITS_DEG,
                              stabilize_fn=stabilize_fn, reflex_fn=reflex_fn)
        # per control step, with FRESH pose + velocity reads:
        _, targets_deg = sched.step(attitude=quat_wxyz, velocity=vel_xy)
    """

    def __init__(
        self,
        targets_fn: Callable[[float], Mapping[str, float]],
        dt: float = 1.0 / 60.0,
        t0: float = 0.0,
        limits: Mapping[str, tuple[float, float]] | None = None,
        stabilize_fn: Callable[[dict[str, float], object, float],
                               Mapping[str, float]] | None = None,
        reflex_fn: Callable[[dict[str, float], object, float],
                            Mapping[str, float]] | None = None,
    ):
        if not callable(targets_fn):
            raise TypeError("targets_fn must be callable: (t) -> {joint: rad}")
        if stabilize_fn is not None and not callable(stabilize_fn):
            raise TypeError("stabilize_fn must be callable: "
                            "(targets_rad, attitude, dt) -> {joint: rad}")
        if reflex_fn is not None and not callable(reflex_fn):
            raise TypeError("reflex_fn must be callable: "
                            "(targets_rad, velocity, dt) -> {joint: rad}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.targets_fn = targets_fn
        self.stabilize_fn = stabilize_fn
        self.reflex_fn = reflex_fn
        self.dt = float(dt)
        self.t = float(t0)
        self.limits = limits
        self.steps = 0
        self.stabilized = 0                 # steps on which the trim ran
        self.reflexed = 0                   # steps on which the reflex ran
        self.joints: tuple[str, ...] = ()   # set from the first sample
        self._att_t_prev: float | None = None   # clock of last measured step
        self._vel_t_prev: float | None = None   # ... of last reflex-measured

    def _finish(self, t: float, angles_rad: Mapping[str, float]) -> dict[str, float]:
        """rad -> deg + clamp + joint-set stability check (shared tail)."""
        deg = radians_to_drive_targets(angles_rad, self.limits)
        if not self.joints:
            self.joints = tuple(sorted(deg))
        elif tuple(sorted(deg)) != self.joints:
            raise ValueError(
                f"trajectory joint set changed at t={t:.4f}: "
                f"{sorted(deg)} != {list(self.joints)}"
            )
        return deg

    def targets_at(self, t: float) -> dict[str, float]:
        """Drive targets (DEGREES) at absolute trajectory time ``t`` — pure
        lookup, does not advance the scheduler clock and never stabilizes
        or reflexes."""
        return self._finish(t, self.targets_fn(float(t)))

    def step(self, attitude: object | None = None,
             velocity: object | None = None) -> tuple[float, dict[str, float]]:
        """Sample at the current clock, then advance by ``dt``.

        Returns ``(t_sampled, drive_targets_deg)`` — call once per control
        step (e.g. per Isaac app-update tick) and hand the dict to
        :func:`apply_to_stage`.  ``attitude`` is this step's fresh body
        attitude measurement, forwarded opaquely to the injected
        ``stabilize_fn``; ``velocity`` is this step's fresh horizontal
        body-velocity measurement, forwarded opaquely to the injected
        ``reflex_fn`` (see the class docstring for both, including the
        pinned composition order: reflex placement first, stabilizer height
        trim second).  Omit them and the step is open-loop and
        byte-identical to a scheduler with neither hook."""
        t = self.t
        rad: Mapping[str, float] = self.targets_fn(float(t))
        if velocity is not None:
            if self.reflex_fn is None:
                raise ValueError(
                    "step() got a velocity but no reflex_fn was injected — "
                    "the push recovery you think is armed is absent (above "
                    "the trim's ~5 N*s ceiling the body inverts with no "
                    "step to catch it); construct "
                    "GaitScheduler(..., reflex_fn=...)"
                )
            t_prev = self._vel_t_prev
            step_dt = (t - t_prev) if t_prev is not None else 0.0
            if step_dt > 0.0:
                rad = self.reflex_fn(dict(rad), velocity, step_dt)
                self.reflexed += 1
            self._vel_t_prev = t
        if attitude is not None:
            if self.stabilize_fn is None:
                raise ValueError(
                    "step() got an attitude but no stabilize_fn was injected "
                    "— the loop you think is closed is open (71% upright, "
                    "not 100%); construct GaitScheduler(..., stabilize_fn=...)"
                )
            t_prev = self._att_t_prev
            step_dt = (t - t_prev) if t_prev is not None else 0.0
            if step_dt > 0.0:
                rad = self.stabilize_fn(dict(rad), attitude, step_dt)
                self.stabilized += 1
            self._att_t_prev = t
        deg = self._finish(t, rad)
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


def mock_stabilize_fn(targets_rad: dict[str, float], attitude: object,
                      dt: float) -> dict[str, float]:
    """A stand-in for the injected closed loop — NOT the production controller.

    The live runner binds ``tritium_lib.control.AttitudeStabilizer`` (see the
    :class:`GaitScheduler` docstring); this mock exists so tests and
    ``--selftest`` can prove the scheduler-side contract lib-free.  It takes
    ``attitude`` as a plain ``(roll_rad, pitch_rad)`` pair — how far body-up
    leans from world-up about each axis — runs a cartoon P-only law with the
    lib's SIGN conventions (positive roll is left-side-up, positive pitch is
    nose-down; the restoring command opposes the tilt), spreads the command
    over the legs as mean-centred foot-height offsets
    (``dz = -pitch_cmd*x + roll_cmd*y``, so the trim rotates the body without
    changing ride height), and applies them through the real
    :func:`apply_foot_height_trim`."""
    roll_rad, pitch_rad = (float(v) for v in attitude)  # type: ignore[misc]
    kp = 0.8                            # the measured runs' default gain
    roll_cmd = -kp * roll_rad
    pitch_cmd = -kp * pitch_rad
    raw = {leg: -pitch_cmd * x + roll_cmd * y
           for leg, (x, y) in GO2_LEG_PLACEMENTS_M.items()}
    mean = sum(raw.values()) / len(raw)
    offsets = {leg: dz - mean for leg, dz in raw.items()}
    return apply_foot_height_trim(targets_rad, offsets)


# Mock reflex geometry — a Go2-class ride height and the lib's default gate,
# duplicated as literals for the hygiene gate (no tritium_lib on the box).
MOCK_REFLEX_COM_HEIGHT_M = 0.30
MOCK_REFLEX_GATE_M = 0.05
_GRAVITY_MPS2 = 9.80665


def mock_reflex_fn(targets_rad: dict[str, float], velocity: object,
                   dt: float) -> dict[str, float]:
    """A stand-in for the injected step reflex — NOT the production reflex.

    The live runner binds ``tritium_lib.control.StepReflex`` (see the
    :class:`GaitScheduler` docstring); this mock exists so tests and
    ``--selftest`` can prove the scheduler-side contract lib-free.  It takes
    ``velocity`` as a plain ``(vx, vy)`` pair (m/s, body frame), computes the
    lib's capture point ``v * sqrt(z0 / g)`` for a Go2-class ride height, and
    applies the lib's gate at the same 0.05 m: at or below it the targets
    pass through UNTOUCHED — the layering contract that keeps an undisturbed
    run byte-identical.  Above it, a cartoon step: the leg whose home
    placement sits closest to the capture point (first wins a tie, like the
    lib) swings toward it — thigh by the forward excursion, hip by the
    lateral, 1 rad per metre.  A cartoon selection AND a cartoon law: the
    real reflex clamps to reach and reports its residual; this only proves
    the seam."""
    vx, vy = (float(v) for v in velocity)  # type: ignore[misc]
    tc = math.sqrt(MOCK_REFLEX_COM_HEIGHT_M / _GRAVITY_MPS2)
    cx, cy = vx * tc, vy * tc
    if math.hypot(cx, cy) <= MOCK_REFLEX_GATE_M:
        return targets_rad
    leg = min(GO2_LEG_PLACEMENTS_M,
              key=lambda n: math.hypot(GO2_LEG_PLACEMENTS_M[n][0] - cx,
                                       GO2_LEG_PLACEMENTS_M[n][1] - cy))
    out = dict(targets_rad)
    if f"{leg}_thigh" in out:
        out[f"{leg}_thigh"] += cx
    if f"{leg}_hip" in out:
        out[f"{leg}_hip"] += cy
    return out


def selftest(args) -> int:
    """No-GPU: schedule the mock trajectory for a few periods and assert the
    drive-target contract — step count, full 12-joint set every step, finite
    DEGREE values inside the clamp envelope, and the clock actually advances.
    Then the stabilizer hook: no attitude / zero error = byte-identical to
    open-loop; a synthetic roll trims the two sides apart; clamps hold.
    Then the reflex hook: no velocity / below-gate velocity = byte-identical;
    an above-gate shove moves the stepping leg's placement; clamps hold."""
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

    # ---- the attitude-stabilization hook (still no GPU, no lib) ----------
    n_check = min(30, expect_steps)
    base = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG)
    open_run = [base.step() for _ in range(n_check)]

    # No attitude passed -> byte-identical to the open loop, trim never runs.
    idle = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG,
                         stabilize_fn=mock_stabilize_fn)
    assert [idle.step() for _ in range(n_check)] == open_run
    assert idle.stabilized == 0, "trim ran without a single attitude sample"

    # Zero attitude error -> zero offsets -> byte-identical, trim DID run.
    level = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG,
                          stabilize_fn=mock_stabilize_fn)
    assert [level.step(attitude=(0.0, 0.0)) for _ in range(n_check)] == open_run
    assert level.stabilized == n_check - 1, "first measured step must be dry"

    # A constant +roll (right-side-down): right calves extend ABOVE the
    # open-loop track, left calves retract below it, everything clamped.
    rolled = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG,
                           stabilize_fn=mock_stabilize_fn)
    closed_run = [rolled.step(attitude=(0.1, 0.0)) for _ in range(n_check)]
    assert closed_run[0] == open_run[0], "step 1 has no rate basis — no trim"
    for (t_o, deg_o), (t_c, deg_c) in zip(open_run[1:], closed_run[1:]):
        assert t_o == t_c
        for leg, sign in (("FR", 1.0), ("RR", 1.0), ("FL", -1.0), ("RL", -1.0)):
            delta = deg_c[f"{leg}_calf"] - deg_o[f"{leg}_calf"]
            assert delta * sign > 0.0, \
                f"{leg}_calf trimmed the wrong way at t={t_c:.3f}: {delta:+.3f}"
        for joint, val in deg_c.items():
            lim = limit_for(joint, DEFAULT_LIMITS_DEG)
            assert lim[0] <= val <= lim[1], f"trim broke clamp on {joint}"
    assert rolled.stabilized == n_check - 1

    # Trim magnitude, pinned at the stand pose: the vertical Jacobian there
    # has j1 = 0 (thigh 50 deg, calf -100 deg are symmetric about vertical),
    # so a +1 cm extension lands entirely on the calf as dz / j2.
    stand = {"thigh": math.radians(50.0), "calf": math.radians(-100.0)}
    trimmed = apply_foot_height_trim(
        {"FL_thigh": stand["thigh"], "FL_calf": stand["calf"]}, {"FL": 0.01})
    j2 = -GO2_CALF_LEN_M * math.sin(stand["thigh"] + stand["calf"])
    want = 0.01 / j2
    got = trimmed["FL_calf"] - stand["calf"]
    assert abs(got - want) < 1e-9, f"calf trim {got:.6f} != dz/j2 {want:.6f}"
    assert abs(trimmed["FL_thigh"] - stand["thigh"]) < 1e-9, "j1=0 pose"

    # ---- the step-reflex hook (still no GPU, no lib) ---------------------
    # No velocity passed -> byte-identical to the open loop, reflex never ran.
    calm = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG,
                         reflex_fn=mock_reflex_fn)
    assert [calm.step() for _ in range(n_check)] == open_run
    assert calm.reflexed == 0, "reflex ran without a single velocity sample"

    # Below the gate -> the reflex RAN every measured step and passed the
    # targets through untouched: byte-identical to the open loop.
    slow = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG,
                         reflex_fn=mock_reflex_fn)
    assert [slow.step(velocity=(0.2, 0.0)) for _ in range(n_check)] == open_run
    assert slow.reflexed == n_check - 1, "first measured reflex step is dry"

    # Above the gate -> the stepping leg (FL for this shove) leaves the
    # open-loop track, everything still clamped.
    shoved = GaitScheduler(mock_targets_fn, dt=dt, limits=DEFAULT_LIMITS_DEG,
                           reflex_fn=mock_reflex_fn)
    shoved_run = [shoved.step(velocity=(0.5, 0.1)) for _ in range(n_check)]
    assert shoved_run[0] == open_run[0], "step 1 has no interval — no reflex"
    for (t_o, deg_o), (t_c, deg_c) in zip(open_run[1:], shoved_run[1:]):
        assert t_o == t_c
        assert deg_c["FL_thigh"] != deg_o["FL_thigh"], \
            f"reflex did not move the stepping leg at t={t_c:.3f}"
        for joint, val in deg_c.items():
            lim = limit_for(joint, DEFAULT_LIMITS_DEG)
            assert lim[0] <= val <= lim[1], f"reflex broke clamp on {joint}"
    assert shoved.reflexed == n_check - 1

    span = {p: (min(lo[f"{l}_{p}"] for l in LEG_NAMES),
                max(hi[f"{l}_{p}"] for l in LEG_NAMES)) for p in JOINT_PARTS}
    print(f"SELFTEST OK steps={steps} periods={args.periods} dt={dt:.4f}s "
          f"joints={len(JOINT_NAMES)} no_nan clamped "
          f"stabilize_hook(idle={idle.stabilized} level={level.stabilized} "
          f"rolled={rolled.stabilized} calf_trim={math.degrees(got):+.2f}deg) "
          f"reflex_hook(calm={calm.reflexed} slow={slow.reflexed} "
          f"shoved={shoved.reflexed}) "
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
