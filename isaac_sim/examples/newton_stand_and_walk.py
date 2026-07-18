#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Stand the Go2 under Newton, then attempt a low-speed trot — repeatably.

The committed, testable scaffold of the Newton locomotion lane.  Two halves:

* **Pure core** — :func:`build_walk_plan` turns an injected lib-shaped
  trajectory callable (``targets_fn(t, gait=..., speed=...)`` returning 12
  named joint angles in RADIANS — the exact signature of
  ``tritium_lib.models.gait_trajectory.joint_targets_at``) into a fully
  materialized control plan ``[(t, {joint: degrees}), ...]`` via the addon's
  :class:`GaitScheduler`.  No Isaac, no GPU, no tritium — unit-tested headless
  in ``tests/test_newton_stand_and_walk.py``.

* **Live runner** — ``--live`` boots Isaac's own python against the **Newton**
  kit experience, spawns the Go2 under ``/World/Tritium/go2``, registers it in
  the Newton solver the way that is known to work live (spawn while stopped +
  ``World.reset()`` — see ``NEWTON-GAIT-FINDINGS.md`` lead 1), STANDS it via
  USD drive ``targetPosition`` set before play (hip 0 / thigh +50 deg /
  calf -100 deg, the Newton-validated stand), then runs a low-speed trot by
  applying ``GaitScheduler`` drive targets each control step, capturing a
  viewport PNG and writing an honest JSON record scored with the same
  non-gameable ``score_trace`` metrics as ``go2_newton_gait.py``.

The live CONTINUOUS-DRIVE path (whether per-step target writes actually
propel the dog under this Newton build) is being validated in a separate live
lane; this file is the committed, repeatable scaffold that lane lands on.
A stand that holds is a pass for this scaffold (``rc 0``); pass
``--require-moved`` to gate the exit code on the walk verdict too.

Both North Star halves: FUN — the tactical scene's dog stands up and tries to
trot instead of T-posing.  PRODUCTION — the exact lib-trajectory -> scheduler
-> drive-target pipeline that will feed a physical quadruped's joint
controllers is exercised end-to-end against a real physics solver.

Dependency hygiene: this module imports NOTHING heavy at module scope — no
``isaacsim``, no ``pxr``/``omni``, no ``tritium_lib``.  All of those are
imported inside the live/emit code paths only, so the module (and the pure
core) loads on a box with none of them installed.

Run
---
    # No-GPU self-test of the pure core (plain python3, anywhere):
    python3 examples/newton_stand_and_walk.py --selftest

    # Emit a walk plan JSON (uses tritium_lib if installed; --mock without):
    python3 examples/newton_stand_and_walk.py --emit-plan plan.json --mock

    # Live, on the RTX host, under Isaac's python (Newton experience).
    # NOTE: this boots its OWN kit — stop the bridged Newton kit first
    # (./newton_kit.sh stop 8212) or the two will fight for VRAM:
    ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
        examples/newton_stand_and_walk.py --live --headless \
        --seconds 8 --speed 0.35 \
        --capture /tmp/go2_walk.png --record /tmp/go2_walk.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

log = logging.getLogger("newton-stand-and-walk")

_HERE = Path(__file__).resolve().parent
_DRIVER_PATH = (_HERE.parent / "isaac_sim_addon" / "connectors"
                / "newton_gait_driver.py")

ROBOT_PATH = "/World/Tritium/go2"
BASE_PATH = f"{ROBOT_PATH}/base"
GO2_ASSET = "/Isaac/Robots/Unitree/Go2/go2.usd"
CONTROL_HZ = 60.0


def _load_by_path(name: str, path: Path):
    """Load a sibling module by file path — examples/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The addon gait driver (pure python at module scope — the hygiene gate in
# tests/test_no_gpu.py proves it drags in neither Isaac nor tritium).
_drv = _load_by_path("newton_gait_driver_for_walk", _DRIVER_PATH)

GaitScheduler = _drv.GaitScheduler
JOINT_NAMES: tuple[str, ...] = _drv.JOINT_NAMES
LEG_NAMES: tuple[str, ...] = _drv.LEG_NAMES
JOINT_PARTS: tuple[str, ...] = _drv.JOINT_PARTS
DEFAULT_LIMITS_DEG: dict[str, tuple[float, float]] = _drv.DEFAULT_LIMITS_DEG

# The Newton-validated stable STAND, in DEGREES (USD angular drives are native
# degrees): hip 0 / thigh +50 / calf -100.  Applied to the drives while the
# sim is STOPPED; the solver reads them on play and the dog stands.
_STAND_PART_DEG: dict[str, float] = {"hip": 0.0, "thigh": 50.0, "calf": -100.0}
STAND_POSE_DEG: dict[str, float] = {
    f"{leg}_{part}": _STAND_PART_DEG[part]
    for leg in LEG_NAMES for part in JOINT_PARTS
}


# --------------------------------------------------------------------------- #
# Pure core — NO Isaac, NO tritium.  Unit-tested headless.
# --------------------------------------------------------------------------- #

def build_walk_plan(
    duration_s: float,
    dt: float,
    gait: str,
    speed: float,
    targets_fn: Callable[..., Mapping[str, float]],
) -> list[tuple[float, dict[str, float]]]:
    """Materialize a walk as ``[(t, {joint: degrees}), ...]`` control steps.

    ``targets_fn`` is lib-shaped: called as ``targets_fn(t, gait=gait,
    speed=speed)`` and must return the 12 named joint angles in RADIANS —
    exactly how the live runner binds tritium-lib's ``joint_targets_at``.
    The plan is scheduled by the addon's :class:`GaitScheduler` at fixed
    ``dt`` steps covering ``duration_s``, converted to DEGREES and clamped
    into :data:`DEFAULT_LIMITS_DEG` — each entry is ready for
    ``apply_to_stage`` (or a real joint controller) verbatim.

    Pure: no Isaac, no GPU, no tritium import — the trajectory arrives
    injected, so the whole plan is provable on a bare CI box.
    """
    if not callable(targets_fn):
        raise TypeError(
            "targets_fn must be callable: (t, *, gait, speed) -> {joint: rad}"
        )
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    sched = GaitScheduler(
        lambda t: targets_fn(t, gait=gait, speed=speed),
        dt=dt,
        limits=DEFAULT_LIMITS_DEG,
    )
    return list(sched.run(duration_s))


def mock_targets_fn(
    t: float, *, gait: str = "trot", speed: float | None = None
) -> dict[str, float]:
    """Lib-shaped stand-in trajectory (``gait``/``speed`` accepted, ignored).

    Delegates to the gait driver's cartoon-trot mock — 12 joints, RADIANS,
    centered on the stand pose — so the pure core and CLI work with no
    tritium_lib on the box."""
    return _drv.mock_targets_fn(t)


def plan_stats(plan: list[tuple[float, dict[str, float]]]) -> dict:
    """Honest summary of a plan: step count, clock span, per-part envelopes
    and means (DEGREES).  Pure; used by --selftest and --emit-plan."""
    if not plan:
        return {"steps": 0}
    parts: dict[str, list[float]] = {p: [] for p in JOINT_PARTS}
    for _, deg in plan:
        for joint, val in deg.items():
            parts[joint.rsplit("_", 1)[-1]].append(val)
    return {
        "steps": len(plan),
        "t0": plan[0][0],
        "t_last": plan[-1][0],
        "joints_per_step": len(plan[0][1]),
        "part_envelopes_deg": {
            p: [round(min(v), 2), round(max(v), 2)] for p, v in parts.items()
        },
        "part_means_deg": {
            p: round(sum(v) / len(v), 2) for p, v in parts.items()
        },
    }


def default_newton_experience() -> str | None:
    """Path to the local Newton kit experience, or None if not present.

    Resolution mirrors ``newton_kit.sh``: ``$ISAAC_RELEASE`` (or the standard
    local build path) + ``apps/isaacsim.exp.full.newton.kit``.  Pure — no
    Isaac import; just a filesystem probe."""
    release = os.environ.get(
        "ISAAC_RELEASE",
        str(Path.home() / "Code/isaac-sim/IsaacSim/_build/linux-x86_64/release"),
    )
    kit = Path(release) / "apps" / "isaacsim.exp.full.newton.kit"
    return str(kit) if kit.is_file() else None


def selftest(hz: float = CONTROL_HZ, seconds: float = 2.0) -> int:
    """No-GPU contract check: build a mock plan and assert the invariants the
    live runner depends on — step count, 12 joints, DEGREES, finite, clamped,
    centered on the stand pose."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dt = 1.0 / hz
    plan = build_walk_plan(seconds, dt, "trot", 0.35, mock_targets_fn)
    expect = _drv.steps_for_duration(seconds, dt)
    assert len(plan) == expect, f"{len(plan)} steps, expected {expect}"
    sums = {j: 0.0 for j in JOINT_NAMES}
    for t, deg in plan:
        assert set(deg) == set(JOINT_NAMES), f"joint set broke at t={t}"
        for joint, val in deg.items():
            assert math.isfinite(val), f"NaN/inf for {joint} at t={t}"
            lo, hi = _drv.limit_for(joint, DEFAULT_LIMITS_DEG)
            assert lo <= val <= hi, f"{joint}={val:.2f} outside [{lo}, {hi}]"
            sums[joint] += val
    for joint, total in sums.items():
        mean = total / len(plan)
        want = STAND_POSE_DEG[joint]
        assert abs(mean - want) < 10.0, (
            f"{joint} mean {mean:.2f} deg not centered on stand {want:.1f}"
        )
    print(f"SELFTEST OK {json.dumps(plan_stats(plan))}")
    return 0


# --------------------------------------------------------------------------- #
# Live runner — Isaac's python on the render host ONLY.  Every heavy import
# lives inside these functions; merely importing this module touches none.
# --------------------------------------------------------------------------- #

def _to_numpy(t):
    """Newton getters can hand back torch tensors on cuda:0."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return t


def _make_pose_reader(stage):
    """Best pose source available: the Newton tensor view (the validated read
    path — see NEWTON-GAIT-FINDINGS.md), falling back to the USD xform."""
    try:
        from isaacsim.core.simulation_manager import SimulationManager as SM

        view = SM.get_physics_sim_view().create_articulation_view(BASE_PATH)

        def read_view() -> list[float]:
            root = _to_numpy(view.get_root_transforms())
            return [float(v) for v in root[0][:3]]

        read_view()  # probe once; fall back if the view is unusable
        return read_view, "newton_tensor_view"
    except Exception as exc:  # noqa: BLE001 — any failure means fall back
        log.warning("tensor-view pose reader unavailable (%s); using USD", exc)

    import omni.usd

    prim = stage.GetPrimAtPath(BASE_PATH)

    def read_usd() -> list[float]:
        m = omni.usd.get_world_transform_matrix(prim)
        t = m.ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

    return read_usd, "usd_xform"


def _capture_viewport(sim_app, path: Path) -> bool:
    """Write a viewport PNG; captures are async, so pump a few frames."""
    try:
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            get_active_viewport,
        )

        capture_viewport_to_file(get_active_viewport(), str(path))
        for _ in range(30):
            sim_app.update()
            if path.is_file() and path.stat().st_size > 0:
                return True
            time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001 — capture is best-effort evidence
        log.warning("viewport capture failed: %s", exc)
    return path.is_file() and path.stat().st_size > 0


def _spawn_go2(stage) -> dict:
    """Reference the Go2 under ROBOT_PATH, select its physics payload variant,
    and lift the base to spawn height.  Must run while STOPPED."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, UsdGeom

    robot = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot.IsValid():
        add_reference_to_stage(
            usd_path=get_assets_root_path() + GO2_ASSET, prim_path=ROBOT_PATH
        )
        robot = stage.GetPrimAtPath(ROBOT_PATH)

    # "physx" is the asset's VARIANT name (the rigid-body/joint payload), NOT
    # the engine — the Newton kit is what selects the solver.  The default
    # variant "None" ships no joints at all.
    variant = None
    vsets = robot.GetVariantSets()
    if "Physics" in vsets.GetNames():
        vs = vsets.GetVariantSet("Physics")
        if vs.GetVariantSelection() != "physx":
            vs.SetVariantSelection("physx")
        variant = vs.GetVariantSelection()

    xf = UsdGeom.Xformable(robot)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.45))
    return {"robot_valid": bool(robot.IsValid()), "variant": variant}


def _configure_stand_drives(stage, stiffness: float, damping: float,
                            paths: Mapping[str, str]) -> int:
    """PD gains + STAND targetPositions on the USD angular drives, while
    STOPPED — the known-working stand path: the solver reads these on play.
    Gains are DEGREE-based (USD angular drives are native degrees)."""
    from pxr import UsdPhysics

    configured = 0
    for path in paths.values():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(stiffness))
        drive.CreateDampingAttr().Set(float(damping))
        configured += 1
    # Stand pose goes through the same applier the walk uses.
    _drv.apply_to_stage(stage, STAND_POSE_DEG, paths)
    return configured


def run_live(args) -> int:
    """Boot the Newton kit, stand the Go2, attempt the trot, record honestly."""
    from isaacsim import SimulationApp  # must precede other isaacsim imports

    app_kwargs = {}
    experience = args.experience or default_newton_experience()
    if experience:
        app_kwargs["experience"] = experience
    else:
        log.warning("no Newton kit experience found — booting the default "
                    "experience, which may not select the Newton solver")
    sim_app = SimulationApp({"headless": bool(args.headless)}, **app_kwargs)
    try:
        return _run_session(sim_app, args, experience)
    finally:
        sim_app.close()


def _run_session(sim_app, args, experience: str | None) -> int:
    import omni.usd
    from isaacsim.core.api import World

    # The ONLY tritium import in this file — function-scoped, render host only.
    from tritium_lib.models.gait_trajectory import joint_targets_at

    scorer = _load_by_path("go2_newton_gait_scorer",
                           _HERE / "go2_newton_gait.py")

    record: dict = {
        "schema": "tritium.newton_stand_and_walk/1",
        "experience": experience,
        "gait": args.gait, "speed": args.speed,
        "control_hz": CONTROL_HZ,
        "stand_seconds": args.stand_seconds, "walk_seconds": args.seconds,
        "stiffness": args.stiffness, "damping": args.damping,
    }

    # ---- scene, while STOPPED: World registers the articulation with the
    # solver on reset() — the path that stands the dog under Newton.
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()   # real collider: dynamic contact
    stage = omni.usd.get_context().get_stage()
    record["spawn"] = _spawn_go2(stage)

    paths = _drv.find_joint_prim_paths(stage, JOINT_NAMES)
    missing = sorted(set(JOINT_NAMES) - set(paths))
    record["joints_found"] = len(paths)
    record["joints_missing"] = missing
    if missing:
        log.error("partial rig — missing joints: %s", missing)

    record["drives_configured"] = _configure_stand_drives(
        stage, args.stiffness, args.damping, paths)

    world.reset()                            # spawn + reset: solver admission
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[1.8, 1.8, 1.2], target=[0.0, 0.0, 0.35])
    except Exception as exc:  # noqa: BLE001 — framing is cosmetic
        log.warning("camera framing failed: %s", exc)

    read_pose, record["pose_reader"] = _make_pose_reader(stage)
    control_dt = 1.0 / CONTROL_HZ

    def sample(trace: list, t: float) -> None:
        try:
            trace.append([round(t, 3)] + read_pose() + [0.0, 0.0, 0.0])
        except Exception as exc:  # noqa: BLE001 — a lost sample, not a crash
            log.warning("pose sample failed at t=%.2f: %s", t, exc)

    # ---- phase 1: STAND.  Drives already hold the stand pose; settle.
    stand_trace: list = []
    stand_steps = _drv.steps_for_duration(args.stand_seconds, control_dt)
    for i in range(stand_steps):
        world.step(render=True)
        if i % args.sample_every == 0:
            sample(stand_trace, i * control_dt)
    sample(stand_trace, stand_steps * control_dt)
    stand_score = scorer.score_trace(stand_trace)
    record["stand"] = stand_score
    stood = bool(stand_trace) and not stand_score.get("collapsed", True)
    log.info("stand: %s", stand_score.get("verdict"))

    # ---- phase 2: WALK.  Lib trajectory -> addon scheduler -> drive targets,
    # applied every control step (the continuous-drive path — the live lane
    # validating it lands here).
    sched = GaitScheduler(
        lambda t: joint_targets_at(t, gait=args.gait, speed=args.speed),
        dt=control_dt,
        limits=DEFAULT_LIMITS_DEG,
    )
    walk_trace: list = []
    applied_total = 0
    walk_steps = _drv.steps_for_duration(args.seconds, control_dt)
    for i in range(walk_steps):
        t, deg = sched.step()
        applied_total += _drv.apply_to_stage(stage, deg, paths)
        world.step(render=True)
        if i % args.sample_every == 0:
            sample(walk_trace, t)
    sample(walk_trace, walk_steps * control_dt)
    walk_score = scorer.score_trace(walk_trace)
    record["walk"] = walk_score
    record["walk_control_steps"] = walk_steps
    record["walk_targets_applied"] = applied_total
    log.info("walk: %s (%d targets applied over %d steps)",
             walk_score.get("verdict"), applied_total, walk_steps)

    # ---- evidence
    if args.capture:
        cap = Path(args.capture)
        record["capture"] = str(cap) if _capture_viewport(sim_app, cap) else None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(record, indent=1))
        log.info("record -> %s", rec_path)
    print(json.dumps({k: record[k] for k in
                      ("stand", "walk", "joints_found", "pose_reader")},
                     indent=1))

    # Honest rc: the scaffold passes when the dog STOOD and the walk phase ran
    # to completion without a partial rig.  --require-moved additionally gates
    # on the walk verdict (the separately-validated continuous-drive path).
    ok = stood and not missing and applied_total == walk_steps * len(paths)
    if args.require_moved:
        ok = ok and walk_score.get("verdict") == "MOVED"
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Newton Go2: stand, then attempt a low-speed trot.")
    ap.add_argument("--selftest", action="store_true",
                    help="no-GPU contract check of the pure plan core")
    ap.add_argument("--emit-plan", metavar="PATH",
                    help="write the walk plan JSON and exit (no Isaac)")
    ap.add_argument("--mock", action="store_true",
                    help="emit-plan: use the mock trajectory (no tritium_lib)")
    ap.add_argument("--live", action="store_true",
                    help="run the live Isaac session (render host only)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--experience",
                    help="kit experience file (default: the local Newton kit)")
    ap.add_argument("--gait", default="trot", choices=("walk", "trot", "bound"))
    ap.add_argument("--speed", type=float, default=0.35,
                    help="commanded speed m/s (low-speed trot default)")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="walk-phase duration")
    ap.add_argument("--stand-seconds", type=float, default=2.0,
                    help="settle time in the stand pose before walking")
    ap.add_argument("--stiffness", type=float, default=60.0,
                    help="USD angular drive stiffness (degree-based)")
    ap.add_argument("--damping", type=float, default=4.0,
                    help="USD angular drive damping (degree-based)")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="record a pose sample every N control steps")
    ap.add_argument("--capture", help="write a viewport PNG here")
    ap.add_argument("--record", help="write the JSON run record here")
    ap.add_argument("--require-moved", action="store_true",
                    help="exit nonzero unless the walk verdict is MOVED")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.emit_plan:
        if args.mock:
            fn = mock_targets_fn
        else:
            # Guarded: only this branch needs the lib installed.
            from tritium_lib.models.gait_trajectory import joint_targets_at as fn
        plan = build_walk_plan(args.seconds, 1.0 / CONTROL_HZ,
                               args.gait, args.speed, fn)
        doc = {
            "schema": "tritium.walk_plan/1",
            "gait": args.gait, "speed": args.speed,
            "control_hz": CONTROL_HZ,
            "stand_pose_deg": STAND_POSE_DEG,
            "stats": plan_stats(plan),
            "plan": [[round(t, 6), {j: round(v, 4) for j, v in deg.items()}]
                     for t, deg in plan],
        }
        Path(args.emit_plan).write_text(json.dumps(doc, indent=1))
        print(f"wrote {args.emit_plan}: {json.dumps(doc['stats'])}")
        return 0

    if args.live:
        logging.basicConfig(level=logging.INFO,
                            format="[stand+walk] %(message)s")
        return run_live(args)

    ap.print_help()
    print("\nPick a mode: --selftest (no GPU), --emit-plan (no Isaac), "
          "or --live (Isaac python on the render host).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
