#!/usr/bin/env python3
"""Drive a Newton-stepped Unitree Go2 with a tritium-lib gait trajectory.

This is the GPU half of the Newton locomotion lane.  The gait itself is pure
kinematics living in ``tritium_lib.models.gait_trajectory`` -- it imports on a
bare Jetson with no Isaac and no GPU.  This script is the thin driver that
ships that trajectory into a live Isaac Sim 6.0 **Newton** session and applies
it to a real 12-DOF articulation, then measures whether the body actually
moved.

Why the Newton tensor API and not ``isaacsim.core.prims.SingleArticulation``:
on this build, constructing ``SingleArticulation`` against a Newton-stepped
stage raises a **sticky** ``CUDA error: an illegal memory access was
encountered`` from inside ``Articulation._on_physics_ready`` ->
``get_world_poses`` -> the *torch* backend rotation utils -- even though
``SimulationManager.get_backend()`` reports ``numpy``.  Once it fires, every
later CUDA call in that process dies too, so the kit must be restarted
(``newton_kit.sh restart``).  The Newton-native
``NewtonSimulationView.create_articulation_view()`` underneath that wrapper
works fine, so we use it directly.  See NEWTON-GAIT-FINDINGS.md.

Frame note: Newton/Isaac here is Z-up, meters, and the Go2 asset's forward
axis is +X.  Joint order comes from the view's own ``joint_names`` -- never
assume the lib's FL/FR/RL/RR ordering matches the solver's.

Usage:
    # 1. generate the trajectory (any box, no GPU, no Isaac):
    python go2_newton_gait.py --emit-gait trot --speed 0.6 -o gait_trot.json
    # 2. drive it into a live Newton kit (on the RTX host):
    python go2_newton_gait.py --gait-file gait_trot.json --seconds 6

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import time

DEFAULT_PORT = 8212  # the Newton kit's MCP bridge (8211 is the PhysX kit)
ROBOT_PATH = "/World/Tritium/go2"
ARTICULATION_PATH = f"{ROBOT_PATH}/base"
GROUND_PATH = "/World/GroundSlab"
GROUND_SIZE_M = 50.0
GROUND_THICKNESS_M = 1.0
# Where the Isaac process should find tritium-lib.  The driver runs inside the
# kit's own interpreter, which has its own site-packages and no knowledge of
# this checkout, so the path is injected rather than assumed importable.
LIB_SRC = "/home/scubasonar/Code/tritium/tritium-lib/src"


# --------------------------------------------------------------------------
# Minimal MCP-bridge client.  Inlined (~50 lines) so this example has no
# dependency on the omniverse-mcp checkout, which lives outside the repo.
# --------------------------------------------------------------------------
class Bridge:
    """Speaks the isaacsim.mcp.bridge HTTP/1.1-over-TCP protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout: float = 240.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout

    def request(self, path: str, body: dict | None = None) -> dict:
        payload = json.dumps(body or {}).encode()
        head = (
            f"POST {path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: keep-alive\r\n\r\n"
        ).encode()
        with socket.create_connection((self.host, self.port), 15) as s:
            s.settimeout(self.timeout)
            s.sendall(head + payload)
            # The extension holds the connection open, so reading to EOF would
            # block until the socket timeout.  Read the headers, then exactly
            # Content-Length bytes of body.
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    raise ConnectionError("bridge closed during headers")
                buf += chunk
            head_end = buf.index(b"\r\n\r\n") + 4
            headers = buf[:head_end].decode("utf-8", "replace")
            length = 0
            for line in headers.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
            body_bytes = buf[head_end:]
            while len(body_bytes) < length:
                chunk = s.recv(min(length - len(body_bytes), 65536))
                if not chunk:
                    raise ConnectionError("bridge closed mid-response")
                body_bytes += chunk
        return json.loads(body_bytes[:length].decode())

    def execute(self, code: str) -> dict:
        return self.request("/execute", {"code": code})

    def sim_control(self, action: str) -> dict:
        return self.request("/sim/control", {"action": action})

    def sim_state(self) -> dict:
        return self.request("/sim/state", {})

    def capture(self, width: int = 960, height: int = 540) -> dict:
        return self.request("/sim/capture", {"width": width, "height": height})

    def camera_look_at(self, prim_path: str, distance: float = 3.0,
                       azimuth: float = 55.0, elevation: float = 18.0) -> dict:
        return self.request("/camera/look_at", {
            "prim_path": prim_path, "distance": distance,
            "azimuth": azimuth, "elevation": elevation,
        })


def _ok(tag: str, resp: dict) -> dict:
    """Raise with the remote traceback rather than a bare KeyError."""
    if resp.get("status") != "success":
        raise RuntimeError(
            f"{tag} failed: {resp.get('error')}\n{resp.get('traceback', '')[:2000]}"
        )
    result = resp.get("result", {})
    return result.get("return_value", result) if isinstance(result, dict) else result


# --------------------------------------------------------------------------
# Step 1 -- gait generation (pure tritium-lib, no Isaac, no GPU)
# --------------------------------------------------------------------------
def emit_gait(gait: str, speed: float, steps: int = 48) -> dict:
    from tritium_lib.models.gait_trajectory import (
        NEUTRAL_STAND_RAD,
        QuadrupedGaitCycle,
    )

    cycle = QuadrupedGaitCycle(gait, speed=speed)
    return {
        "gait": gait,
        "speed": speed,
        "stride_hz": cycle.stride_hz,
        "duty_factor": cycle.duty_factor,
        "stand": dict(NEUTRAL_STAND_RAD),
        "table": [{"phase": ph, "angles": a} for ph, a in cycle.sample_cycle(steps=steps)],
    }


# --------------------------------------------------------------------------
# Step 2 -- scene build.  Must happen while the sim is STOPPED so the physics
# view is constructed with the robot already present.
# --------------------------------------------------------------------------
SCENE_TEMPLATE = """
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, UsdPhysics, Gf

stage = omni.usd.get_context().get_stage()

# The ground is a box SLAB, never `GroundPlane`.  A GroundPlane is a
# zero-thickness quad, its four corners are coplanar, and Newton's MuJoCo
# solver convex-hulls every collision shape -- qhull cannot seed a simplex from
# a rank-2 point set.  The resulting exception escapes the physics extension's
# initializer with its `_initializing` latch still set, so physics is dead for
# the life of the process while the timeline, the frame counter and sim time
# all keep advancing.  This scene ran against that for four ticks and every
# joint command went into a world that never integrated.  Geometry from
# `tritium_lib.geo.collider_shape.ground_slab` (rank 3, surface at __TOP_Z__).
if not stage.GetPrimAtPath("__GROUND_PATH__").IsValid():
    slab = UsdGeom.Cube.Define(stage, "__GROUND_PATH__")
    slab.CreateSizeAttr(2.0)  # unit cube; the scale op below sets real extents
    xf = UsdGeom.Xformable(slab.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*__SLAB_CENTER__))
    xf.AddScaleOp().Set(Gf.Vec3f(*__SLAB_HALF__))
    UsdPhysics.CollisionAPI.Apply(slab.GetPrim())
    # Deliberately NO RigidBodyAPI: static collider, so it does not fall itself.

robot = stage.GetPrimAtPath("__ROBOT_PATH__")
if not robot.IsValid():
    usd_path = get_assets_root_path() + "/Isaac/Robots/Unitree/Go2/go2.usd"
    add_reference_to_stage(usd_path=usd_path, prim_path="__ROBOT_PATH__")
    robot = stage.GetPrimAtPath("__ROBOT_PATH__")

# The asset ships a Physics variant set; "None" is the default and carries no
# rigid bodies or joints at all.  "physx" is the VARIANT name (the physics
# payload), NOT the engine -- the kit is what selects Newton.
vsets = robot.GetVariantSets()
variant = None
if "Physics" in vsets.GetNames():
    vs = vsets.GetVariantSet("Physics")
    if vs.GetVariantSelection() != "physx":
        vs.SetVariantSelection("physx")
    variant = vs.GetVariantSelection()

from pxr import UsdGeom, Gf, UsdPhysics
xf = UsdGeom.Xformable(robot)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.45))

# PD gains go on the USD angular drives while STOPPED -- the tensor API's
# set_dof_stiffnesses() is broken on this build (NewtonStage has no `solver`).
# The solver reads these on play.  USD angular drives are in degrees, so the
# stiffness/damping here are deg-based, not the rad-based numbers you would
# hand the tensor API.
drives = 0
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    if not prim.GetPath().pathString.startswith("__ROBOT_PATH__"):
        continue
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(__STIFFNESS__)
    drive.CreateDampingAttr().Set(__DAMPING__)
    drives += 1

result = {"variant": variant, "robot_valid": bool(robot.IsValid()),
          "drives_configured": drives}
"""


def build_scene_code(stiffness: float, damping: float) -> str:
    """Render the scene-build snippet.

    Plain .replace() rather than .format()/f-string: the body is Python source
    full of dict and set literals, and every one of those braces would have to
    be doubled.
    """
    from tritium_lib.geo.collider_shape import check_convex_hull_input, ground_slab

    slab = ground_slab(size_m=GROUND_SIZE_M, thickness_m=GROUND_THICKNESS_M, top_z=0.0)
    # Assert the thing that silently killed this lane, on the client, before the
    # geometry is ever authored into the stage.  Free, and it can only fail if
    # someone edits the numbers above into something degenerate.
    report = check_convex_hull_input(slab.vertices)
    if not report.is_hullable:
        raise ValueError(f"ground slab is not hullable: {report.reason}")

    return (SCENE_TEMPLATE
            .replace("__ROBOT_PATH__", ROBOT_PATH)
            .replace("__GROUND_PATH__", GROUND_PATH)
            .replace("__SLAB_CENTER__", repr(tuple(slab.center)))
            .replace("__SLAB_HALF__", repr(tuple(slab.half_extents)))
            .replace("__TOP_Z__", repr(float(slab.top_z)))
            .replace("__STIFFNESS__", repr(float(stiffness)))
            .replace("__DAMPING__", repr(float(damping))))


# --------------------------------------------------------------------------
# Step 3 -- install the gait driver INSIDE the Isaac process.  It runs off a
# physics-step callback (driving from the client would be gated by RPC
# latency, not physics time) and records a pose trace we can score later.
# --------------------------------------------------------------------------
DRIVER_CODE = """
import json, math
import numpy as np
from isaacsim.core.simulation_manager import SimulationManager as SM

gait = json.loads(GAIT_JSON)
table = gait["table"]
stride_hz = float(gait["stride_hz"])

view = SM.get_physics_sim_view().create_articulation_view("ARTICULATION_PATH")

# joint_names comes back per-articulation, i.e. a list of lists shaped
# (count, dof) -- not a flat list of strings.  Take the first articulation's
# ordering (the views here are homogeneous).
raw_names = list(view.joint_names)
if raw_names and isinstance(raw_names[0], (list, tuple)):
    raw_names = list(raw_names[0])
joint_names = [str(n).split("/")[-1].replace("_joint", "") for n in raw_names]

# Map the lib's joint-name keys onto the solver's own DOF ordering.  If the
# solver exposes a joint the gait doesn't name, hold it at the stand pose.
stand = gait["stand"]
stand_vec = np.array([float(stand.get(n, 0.0)) for n in joint_names], dtype=np.float32)

def angles_to_vec(angles):
    return np.array([float(angles.get(n, stand.get(n, 0.0))) for n in joint_names],
                    dtype=np.float32)

# Pre-bake the table into solver DOF order so the callback does no dict work.
phases = np.array([row["phase"] for row in table], dtype=np.float32)
frames = np.stack([angles_to_vec(row["angles"]) for row in table])

count = view.count
# The Newton tensor API requires a REAL index array on every setter -- unlike
# the PhysX API it does not accept None as "all articulations" (it dies with
# AttributeError: 'NoneType' object has no attribute 'shape').
idx = np.arange(count, dtype=np.int32)

def tile(vec):
    return np.tile(vec.reshape(1, -1), (count, 1)).astype(np.float32)

# NB: PD gains are NOT set here.  view.set_dof_stiffnesses()/set_dof_dampings()
# are broken in this Isaac build -- they call
# _notify_joint_dof_properties_changed(), which touches
# `self._newton_stage.solver`, an attribute NewtonStage does not have:
#     AttributeError: 'NewtonStage' object has no attribute 'solver'
# Gains are therefore applied as USD drive attributes while the sim is
# STOPPED (see SCENE_CODE), which the solver picks up on play.  Only the
# per-step position targets go through the tensor API.

# ---- closed-loop attitude feedback -------------------------------------
# The controller itself is NOT reimplemented here.  It is
# tritium_lib.control.AttitudeStabilizer, the same object the headless unit
# tests drive -- which is the whole point of the lib/addon split: the code
# that keeps a simulated Go2 upright is the code that would keep a real one
# upright, and it is tested without a GPU in the room.
import sys as _sys
if __LIB_SRC__ not in _sys.path:
    _sys.path.insert(0, __LIB_SRC__)
# Drop any cached tritium_lib modules before importing.  The kit is a
# long-lived process that has usually imported the lib on a previous run, so
# without this an edit to tritium-lib is invisible until the kit is
# restarted -- and worse, it fails as a confusing ImportError for a symbol
# that plainly exists on disk.
for _m in [m for m in _sys.modules if m == "tritium_lib" or m.startswith("tritium_lib.")]:
    del _sys.modules[_m]
# Purging sys.modules is NOT enough on its own: importlib's FileFinder caches
# each directory's listing, so a module file added to tritium-lib AFTER the
# kit last scanned that package is invisible and raises ModuleNotFoundError
# for a file that is plainly on disk.  This is what makes lib edits land in a
# long-running kit without a restart.
import importlib as _importlib
_importlib.invalidate_caches()
from tritium_lib.control import AttitudeStabilizer, DisturbanceSchedule, Impulse, LegPlacement

# ---- the disturbance ----------------------------------------------------
# A controller that regulates tilt can only be credited with rejecting a
# disturbance that actually happened.  Three sessions of this gait produced
# upright rates of 67%, 29% and 100% from an identical gait file, so the
# falls it was built to prevent arrive by luck and cannot be A/B'd.  This
# injects a KNOWN push at a KNOWN sim time, identically in both arms, so the
# manipulated variable is ours instead of the weather.
#
# Scheduling is tritium_lib.control.DisturbanceSchedule -- again the same
# object the headless tests drive, not a reimplementation.  Firing is decided
# by SIM TIME, so both arms are kicked at the same moment in the stride
# regardless of frame rate.
_disturb_spec = __DISTURB_JSON__
disturb = (DisturbanceSchedule([Impulse(at_time=d["at_time"],
                                        linear=tuple(d["linear"]),
                                        label=d.get("label", ""))
                                for d in _disturb_spec])
           if _disturb_spec else None)

# Go2 foot placements in the body frame (REP-103: +X forward, +Y left).
# Hip x-offset is the real URDF value; y is the FOOT's lateral stance, which
# is wider than the hip joint because the legs splay outward.
GO2_LEGS = (
    LegPlacement("FL", x=0.1881, y=0.1300),
    LegPlacement("FR", x=0.1881, y=-0.1300),
    LegPlacement("RL", x=-0.1881, y=0.1300),
    LegPlacement("RR", x=-0.1881, y=-0.1300),
)
GO2_THIGH_LEN = 0.213  # metres; Go2 thigh and calf links are the same length
GO2_CALF_LEN = 0.213
MAX_TRIM_RAD = 0.30    # per-joint clamp, so one bad pose read cannot fling a leg

stab = AttitudeStabilizer(kp=__KP__, kd=__KD__)

# Resolve each leg's thigh/calf DOF index once.  A leg whose joints are not
# both present is dropped rather than guessed at.
leg_dof = {}
for _leg in GO2_LEGS:
    try:
        leg_dof[_leg.name] = (joint_names.index(_leg.name + "_thigh"),
                              joint_names.index(_leg.name + "_calf"))
    except ValueError:
        pass


def _apply_trim(targets, offsets):
    # Map a desired vertical foot movement to thigh/calf angles through the
    # leg's linearised vertical Jacobian, evaluated at the CURRENTLY commanded
    # pose rather than a fixed stand pose -- the gait sweeps the legs through a
    # wide arc, and a Jacobian frozen at stand would be badly wrong mid-stride.
    #
    # For a 2-link planar leg the downward reach is
    #     depth = L1*cos(q1) + L2*cos(q1 + q2)
    # so the row Jacobian is d(depth)/d(q1, q2).  Distributing the correction
    # by least-norm (J^T * dz / (J . J)) splits it across both joints in the
    # smallest total joint movement, instead of asking one joint to do all of
    # it and hitting its limit.
    for name, (i_th, i_ca) in leg_dof.items():
        dz = offsets.get(name, 0.0)
        if dz == 0.0:
            continue
        q1 = float(targets[i_th])
        q2 = float(targets[i_ca])
        j1 = -GO2_THIGH_LEN * math.sin(q1) - GO2_CALF_LEN * math.sin(q1 + q2)
        j2 = -GO2_CALF_LEN * math.sin(q1 + q2)
        denom = j1 * j1 + j2 * j2
        if denom < 1e-6:
            # Singular: the leg is straight, so no small joint change moves the
            # foot vertically.  Skipping is correct -- the pseudo-inverse would
            # divide by ~0 and command an enormous trim.
            continue
        scale = dz / denom
        targets[i_th] += max(-MAX_TRIM_RAD, min(MAX_TRIM_RAD, j1 * scale))
        targets[i_ca] += max(-MAX_TRIM_RAD, min(MAX_TRIM_RAD, j2 * scale))
    return targets


state = {"trace": [], "steps": 0, "t0": None, "err": None,
         "t_prev": None, "stabilized": 0, "max_cmd_seen": 0.0,
         # Disturbance bookkeeping.  `kicks` records what actually landed and
         # when -- a run that reports an empty list did NOT run the experiment
         # it claims to, and scoring it would compare two undisturbed arms.
         "kicks": [], "d_prev": 0.0, "body_mass": None, "push": None}


PUSH_WINDOW_S = __PUSH_WINDOW__

def _root_vel():
    return [float(v) for v in _to_numpy(view.get_root_velocities())[0][:3]]


def _apply_impulse(imp, elapsed):
    # An impulse is delivered as a FORCE HELD OVER A WINDOW: J = F * T.
    #
    # The obvious route -- read the root velocity, add J/m, write it back --
    # was tried first and is silently a no-op: NewtonArticulationView accepts
    # set_root_velocities() on a floating-base articulation and the solver
    # discards it, so the velocity reads back completely unchanged.  Every
    # counter still incremented, and the run cheerfully reported "recovered
    # 1/1" from an experiment in which nothing was ever pushed.  Forces go
    # through the solver's own accumulation path and actually move the body.
    if state["body_mass"] is None:
        # Summed link masses, read from the solver rather than hardcoded --
        # a constant here would silently lie the moment the asset changes.
        state["body_mass"] = float(_to_numpy(view.get_masses())[0].sum())
    state["push"] = {
        "force": [imp.linear[i] / PUSH_WINDOW_S for i in range(3)],
        "until": elapsed + PUSH_WINDOW_S,
        "vel_at_start": _root_vel(),
        "record": {
            "at_time": imp.at_time, "fired_at": round(float(elapsed), 4),
            "impulse_ns": list(imp.linear),
            "window_s": PUSH_WINDOW_S,
            "expected_dv_mps": [round(imp.linear[i] / state["body_mass"], 4)
                                for i in range(3)],
            "label": imp.label,
        },
    }


def _pump_push(elapsed):
    # Hold the push force across its window, then measure what it did.
    # (Comment, not a docstring: this whole block is a triple-quoted Python
    # string, so a nested docstring would terminate it.)
    push = state.get("push")
    if push is None:
        return
    if elapsed <= push["until"]:
        # Force array is (count, links, 3); the push goes on link 0, the base.
        fd = np.zeros((count, 17, 3), dtype=np.float32)
        fd[:, 0, :] = np.array(push["force"], dtype=np.float32)
        view.apply_forces(fd, idx, True)
        return
    # Window closed: compare the velocity the body actually gained against
    # J/m.  This is the ONLY evidence that the disturbance was real, and it
    # is what caught the silent no-op above.
    end_vel = _root_vel()
    rec = push["record"]
    rec["vel_before"] = [round(v, 4) for v in push["vel_at_start"]]
    rec["vel_after"] = [round(v, 4) for v in end_vel]
    rec["measured_dv_mps"] = [round(end_vel[i] - push["vel_at_start"][i], 4)
                              for i in range(3)]
    rec["measured_dv_mag"] = round(
        sum((end_vel[i] - push["vel_at_start"][i]) ** 2 for i in range(3)) ** 0.5, 4)
    state["kicks"].append(rec)
    state["push"] = None

def _to_numpy(t):
    # Newton getters hand back torch tensors on cuda:0, and np.array() on one
    # raises "can not convert cuda:0 device type tensor to numpy".
    # (Comment, not a docstring: this whole block is itself a triple-quoted
    # Python string, so a nested docstring would terminate it.)
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)

def _sample_targets(t):
    ph = (t * stride_hz) % 1.0
    i = int(np.searchsorted(phases, ph, side="right")) - 1
    j = (i + 1) % len(phases)
    span = (phases[j] - phases[i]) % 1.0
    frac = 0.0 if span <= 0 else float(((ph - phases[i]) % 1.0) / span)
    return frames[i] * (1.0 - frac) + frames[j] * frac

def _on_step(dt):
    try:
        t = SM.get_simulation_time()
        if state["t0"] is None:
            state["t0"] = t
        elapsed = t - state["t0"]
        if elapsed > DURATION:
            return

        # Kick BEFORE this step's targets are commanded, so the controller
        # first sees the disturbance on the very next step rather than
        # half a stride later.
        if disturb is not None:
            for _imp in disturb.due(state["d_prev"], elapsed):
                _apply_impulse(_imp, elapsed)
            state["d_prev"] = elapsed
            _pump_push(elapsed)

        targets = _sample_targets(elapsed).copy()

        # The root transform is read EVERY step when stabilising, not on the
        # SAMPLE_EVERY cadence -- feedback at the trace's logging rate would be
        # a control loop running an order of magnitude slower than the
        # disturbance it is meant to reject.
        root = None
        if __STABILIZE__:
            root = _to_numpy(view.get_root_transforms())
            t_prev = state["t_prev"]
            step_dt = (elapsed - t_prev) if t_prev is not None else 0.0
            if step_dt > 0.0:
                # Isaac's root transform is position + an XYZW quaternion;
                # tritium_lib speaks WXYZ, so this reorder is load-bearing.
                qx, qy, qz, qw = (float(v) for v in root[0][3:7])
                corr = stab.update((qw, qx, qy, qz), step_dt)
                targets = _apply_trim(
                    targets, corr.leg_height_offsets(GO2_LEGS))
                state["stabilized"] += 1
                state["max_cmd_seen"] = max(
                    state["max_cmd_seen"],
                    abs(corr.roll_cmd), abs(corr.pitch_cmd))
            state["t_prev"] = elapsed

        view.set_dof_position_targets(tile(targets), idx)
        state["steps"] += 1
        if state["steps"] % SAMPLE_EVERY == 0:
            if root is None:
                root = _to_numpy(view.get_root_transforms())
            state["trace"].append([round(float(elapsed), 3)]
                                  + [round(float(v), 4) for v in root[0][:7]])
    except Exception as exc:  # keep a solver exception from killing the app
        state["err"] = repr(exc)

view.set_dof_position_targets(tile(stand_vec), idx)

# Drive off the APP UPDATE stream, not SimulationManager's
# POST_PHYSICS_STEP.  The latter silently never fires here (physics_steps
# stayed 0 across a full run) -- those default callbacks are pumped by a
# World instance, and this scene is driven straight through the MCP bridge
# with no World.  The app update stream fires every rendered frame (~60 Hz)
# regardless, which is ample for a ~1 Hz stride; physics still integrates at
# its own 500 Hz underneath.
import omni.kit.app
_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    lambda e: _on_step(0.0), name="tritium_gait_driver")
handle = _sub

import builtins
builtins._tritium_gait = {"state": state, "handle": handle, "view": view,
                          "joint_names": joint_names}
result = {"joint_names": joint_names, "count": int(count),
          "dof": int(len(joint_names)), "stabilize": bool(__STABILIZE__),
          "legs_wired": sorted(leg_dof)}
"""


def build_driver_code(gait: dict, duration: float, stiffness: float,
                      damping: float, sample_every: int,
                      stabilize: bool = True, kp: float = None,
                      kd: float = None, lib_src: str = LIB_SRC,
                      disturb: list[dict] | None = None,
                      push_window: float = 0.1) -> str:
    from tritium_lib.control.attitude_stabilizer import DEFAULT_KD, DEFAULT_KP

    code = DRIVER_CODE
    code = code.replace("GAIT_JSON", repr(json.dumps(gait)))
    code = code.replace("ARTICULATION_PATH", ARTICULATION_PATH)
    code = code.replace("STIFFNESS", repr(stiffness))
    code = code.replace("DAMPING", repr(damping))
    code = code.replace("DURATION", repr(duration))
    code = code.replace("SAMPLE_EVERY", str(sample_every))
    code = code.replace("__STABILIZE__", repr(bool(stabilize)))
    code = code.replace("__LIB_SRC__", repr(lib_src))
    code = code.replace("__DISTURB_JSON__", repr(list(disturb or [])))
    code = code.replace("__PUSH_WINDOW__", repr(float(push_window)))
    code = code.replace("__KP__", repr(float(DEFAULT_KP if kp is None else kp)))
    code = code.replace("__KD__", repr(float(DEFAULT_KD if kd is None else kd)))
    # IsaacEvents moved around between releases; resolve it remotely.
    code = code.replace(
        "CALLBACK_EVENT",
        "__import__('isaacsim.core.simulation_manager', fromlist=['IsaacEvents'])"
        ".IsaacEvents.POST_PHYSICS_STEP",
    )
    return code


COLLECT_CODE = """
import builtins
g = builtins._tritium_gait
st = g["state"]
result = {"steps": st["steps"], "err": st["err"], "trace": st["trace"],
          "joint_names": g["joint_names"], "kicks": st["kicks"],
          "body_mass": st["body_mass"]}
"""

CLEANUP_CODE = """
import builtins
from isaacsim.core.simulation_manager import SimulationManager as SM
g = getattr(builtins, "_tritium_gait", None)
removed = False
if g is not None:
    try:
        # an omni.kit update subscription is released by dropping the ref
        g["handle"] = None
        removed = True
    except Exception:
        pass
    del builtins._tritium_gait
result = {"removed": removed}
"""


def score_trace(trace: list[list[float]]) -> dict:
    """Honest, non-gameable motion metrics from the recorded root transforms.

    Each row is [t, x, y, z, qx, qy, qz, qw] -- Isaac's root transform is
    position followed by an **xyzw** quaternion, while ``tritium_lib.geo``
    speaks wxyz, so the reorder below is load-bearing.

    Two independent gates, because neither is sufficient alone:

    * **Displacement** in the ground plane -- did it cover ground?  On its own
      this certifies a body that is merely sliding.
    * **Attitude** -- did it stay the right way up?  On its own this certifies
      a body standing perfectly still.

    Height retention is kept for context but is explicitly NOT a fall
    detector.  It was the only attitude-ish check here for one tick and it
    passed a robot that was lying on its back: an inverted quadruped sits at
    almost exactly standing height, so height cannot see rotation.  That is
    what ``max_tilt_deg`` is for.
    """
    if len(trace) < 2:
        return {"verdict": "NO_TRACE", "samples": len(trace)}
    t0, x0, y0, z0 = trace[0][0], trace[0][1], trace[0][2], trace[0][3]
    t1, x1, y1, z1 = trace[-1][0], trace[-1][1], trace[-1][2], trace[-1][3]
    dx, dy = x1 - x0, y1 - y0
    dist = (dx * dx + dy * dy) ** 0.5
    zs = [row[3] for row in trace]
    min_z = min(zs)
    dt = max(t1 - t0, 1e-6)
    collapsed = min_z < 0.6 * z0

    from tritium_lib.geo.body_attitude import (
        DEFAULT_MAX_TILT_DEG,
        tilt_from_upright_deg,
    )

    tilts = []
    for row in trace:
        if len(row) < 8:
            continue  # a short row predates the quaternion; skip, don't guess
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]
        try:
            tilts.append(tilt_from_upright_deg((qw, qx, qy, qz)))
        except ValueError:
            continue
    max_tilt = max(tilts) if tilts else None
    end_tilt = tilts[-1] if tilts else None
    tumbled = max_tilt is not None and max_tilt > DEFAULT_MAX_TILT_DEG

    return {
        "samples": len(trace),
        "duration_s": round(dt, 3),
        "start_xyz": [round(x0, 4), round(y0, 4), round(z0, 4)],
        "end_xyz": [round(x1, 4), round(y1, 4), round(z1, 4)],
        "displacement_m": round(dist, 4),
        "forward_dx_m": round(dx, 4),
        "lateral_dy_m": round(dy, 4),
        "mean_speed_mps": round(dist / dt, 4),
        "min_z_m": round(min_z, 4),
        "height_retained": round(min_z / z0, 3) if z0 else None,
        "collapsed": collapsed,
        "max_tilt_deg": round(max_tilt, 2) if max_tilt is not None else None,
        "end_tilt_deg": round(end_tilt, 2) if end_tilt is not None else None,
        "tumbled": tumbled,
        # Order matters: a tumble outranks distance, because a tumbling body
        # covers ground and would otherwise be reported as a working gait.
        "verdict": (
            "TUMBLED" if tumbled
            else "COLLAPSED" if collapsed
            else "WALKED" if dist > 0.10
            else "STATIONARY"
        ),
    }


def disturb_spec(args) -> list[dict]:
    """Build the impulse schedule from the CLI, or an empty list for none."""
    if args.disturb_at is None:
        return []
    parts = [p.strip() for p in args.disturb_impulse.split(",")]
    if len(parts) != 3:
        raise SystemExit(
            f"--disturb-impulse wants X,Y,Z in N-s, got {args.disturb_impulse!r}")
    linear = [float(p) for p in parts]
    if not any(linear):
        raise SystemExit(
            "--disturb-impulse is all zeros: that run would measure nothing, "
            "because the two arms would be identical undisturbed gaits")
    return [{"at_time": float(args.disturb_at), "linear": linear,
             "label": "cli"}]


def score_disturbance(collected: dict, args) -> dict:
    """Score recovery from the kick, or say plainly that no kick landed.

    The trap this guards is specific: if the impulse never fires, both arms
    score identically well and the null result reads as successful rejection.
    So a run configured with a disturbance that did not land is reported as
    ``NOT_APPLIED``, never as a clean recovery.
    """
    kicks = collected.get("kicks") or []
    if args.disturb_at is None:
        return {"disturbance": None}
    if not kicks:
        return {"disturbance": {"verdict": "NOT_APPLIED"},
                "disturb_ok": False}

    from tritium_lib.control import score_recovery
    from tritium_lib.geo.body_attitude import tilt_from_upright_deg

    samples = []
    for row in collected.get("trace", []):
        if len(row) < 8:
            continue
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]
        try:
            samples.append((float(row[0]), tilt_from_upright_deg((qw, qx, qy, qz))))
        except ValueError:
            continue

    fired_at = kicks[0]["fired_at"]
    try:
        rec = score_recovery(samples, disturbed_at=fired_at,
                             settled_below_deg=args.settle_below,
                             hold_for=args.settle_hold)
    except ValueError as exc:
        # e.g. the kick landed after the last logged sample.  Reporting the
        # reason beats reporting a number nobody can trust.
        return {"disturbance": {"verdict": "UNSCORABLE", "reason": str(exc)},
                "disturb_ok": False}

    payload = rec.as_dict()
    payload["verdict"] = "RECOVERED" if rec.recovered else "NOT_RECOVERED"
    payload["kicks"] = kicks
    payload["body_mass_kg"] = collected.get("body_mass")
    return {"disturbance": payload, "disturb_ok": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit-gait", metavar="GAIT",
                    help="generate a gait table from tritium-lib and exit (no Isaac)")
    ap.add_argument("--speed", type=float, default=0.6, help="commanded speed m/s")
    ap.add_argument("--steps", type=int, default=48, help="samples per gait cycle")
    ap.add_argument("-o", "--out", default="-", help="gait table output path")
    ap.add_argument("--gait-file", help="gait table JSON to drive into Isaac")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=6.0, help="drive duration")
    ap.add_argument("--stiffness", type=float, default=60.0)
    ap.add_argument("--damping", type=float, default=4.0)
    ap.add_argument("--sample-every", type=int, default=10,
                    help="record a pose sample every N physics steps")
    ap.add_argument("--capture", help="write a mid-window viewport PNG here")
    ap.add_argument("--keep-running", action="store_true",
                    help="leave the step callback installed after scoring")
    ap.add_argument("--trials", type=int, default=1, metavar="N",
                    help="run N trials per arm and report the rate. A gait is "
                         "a distribution, not a run: this lane has quoted 67%% "
                         "and 29%% for the same gait file off single handfuls "
                         "of trials, which is why one run proves nothing.")
    ap.add_argument("--stabilize", choices=("on", "off", "both"), default="on",
                    help="closed-loop attitude feedback: on, off (open-loop "
                         "control), or both arms A/B in one session")
    ap.add_argument("--disturb-at", type=float, default=None, metavar="T",
                    help="inject a push at sim time T seconds. Without this "
                         "the run waits for a spontaneous fall, which across "
                         "three sessions arrived 67%%, 29%% and 0%% of the "
                         "time -- you cannot A/B rejection of a disturbance "
                         "that shows up by luck.")
    ap.add_argument("--disturb-impulse", default="0,15,0", metavar="X,Y,Z",
                    help="world-frame linear impulse in N-s (default a 15 N-s "
                         "LATERAL shove; the body walks +X, so +Y pushes it "
                         "sideways, which is the axis a trot is least able to "
                         "recover on its own)")
    ap.add_argument("--disturb-window", type=float, default=0.1, metavar="SEC",
                    help="how long the push force is held. The impulse is "
                         "delivered as force*time (J = F*T), because a "
                         "root-velocity write is silently discarded by the "
                         "Newton solver on a floating-base articulation.")
    ap.add_argument("--settle-below", type=float, default=10.0, metavar="DEG",
                    help="tilt considered level again for recovery scoring")
    ap.add_argument("--settle-hold", type=float, default=1.0, metavar="SEC",
                    help="how long tilt must STAY below --settle-below before "
                         "recovery is believed")
    ap.add_argument("--kp", type=float, default=None,
                    help="attitude proportional gain (default: tritium-lib's)")
    ap.add_argument("--kd", type=float, default=None,
                    help="attitude derivative gain (default: tritium-lib's)")
    args = ap.parse_args()

    if args.emit_gait:
        table = emit_gait(args.emit_gait, args.speed, args.steps)
        text = json.dumps(table, indent=1)
        if args.out == "-":
            print(text)
        else:
            with open(args.out, "w") as fh:
                fh.write(text)
            print(f"wrote {args.out}: {len(table['table'])} samples, "
                  f"stride_hz={table['stride_hz']:.3f}")
        return 0

    if not args.gait_file:
        ap.error("need --gait-file (or --emit-gait to generate one)")
    with open(args.gait_file) as fh:
        gait = json.load(fh)

    br = Bridge(args.host, args.port)
    print(f"[gait] bridge {args.host}:{args.port} -> {br.sim_state().get('result', {}).get('state')}")

    # Which arms to run.  Running the open-loop control in the SAME session,
    # against the same kit and the same slab, is what makes the closed-loop
    # number mean anything -- the two recorded rates for this gait (67% then
    # 29%) came from different sessions and cannot be compared to each other.
    arms = []
    if args.stabilize in ("on", "both"):
        arms.append(True)
    if args.stabilize in ("off", "both"):
        arms.append(False)

    results: dict[bool, list[dict]] = {}
    for stabilize in arms:
        label = "closed-loop" if stabilize else "open-loop (control)"
        runs = []
        for trial in range(1, args.trials + 1):
            capture = None
            if args.capture and trial == 1:
                suffix = "closed" if stabilize else "open"
                capture = args.capture.replace(".png", f"-{suffix}.png")
            score = run_trial(br, gait, args, stabilize, capture)
            runs.append(score)
            line = (f"[gait] {label} trial {trial}/{args.trials}: "
                    f"{score.get('verdict')} "
                    f"dist={score.get('displacement_m')} "
                    f"max_tilt={score.get('max_tilt_deg')}")
            dist_info = score.get("disturbance")
            if dist_info:
                kicks = dist_info.get("kicks") or []
                # Print the measured velocity delta, not just "fired" -- this
                # is the line that shows whether the solver honoured the push.
                dv = kicks[0]["measured_dv_mps"] if kicks else None
                line += (f" | kick={dist_info.get('verdict')}"
                         f" dv_measured={dv}"
                         f" peak={dist_info.get('peak_deg')}"
                         f" settle={dist_info.get('settle_time')}")
            print(line)
        results[stabilize] = runs

    print("\n" + "=" * 62)
    print("HONEST GAIT RATE — every trial counted, none discarded")
    print("=" * 62)
    summary = {}
    for stabilize, runs in results.items():
        label = "closed_loop" if stabilize else "open_loop_control"
        walked = sum(1 for r in runs if r.get("verdict") == "WALKED")
        upright = sum(1 for r in runs if not r.get("tumbled"))
        tilts = [r["max_tilt_deg"] for r in runs if r.get("max_tilt_deg") is not None]
        dists = [r["displacement_m"] for r in runs if r.get("displacement_m") is not None]
        summary[label] = {
            "trials": len(runs),
            "walked": walked,
            "walked_rate": round(walked / len(runs), 3) if runs else None,
            "upright": upright,
            "upright_rate": round(upright / len(runs), 3) if runs else None,
            "median_max_tilt_deg": round(sorted(tilts)[len(tilts) // 2], 2) if tilts else None,
            "worst_max_tilt_deg": round(max(tilts), 2) if tilts else None,
            "median_displacement_m": round(sorted(dists)[len(dists) // 2], 3) if dists else None,
            "verdicts": [r.get("verdict") for r in runs],
        }
        if args.disturb_at is not None:
            # Recovery is scored only over trials where the kick actually
            # landed.  Counting a NOT_APPLIED run as a recovery would inflate
            # exactly the number this experiment exists to measure.
            applied = [r for r in runs if r.get("disturb_ok")]
            recovered = sum(1 for r in applied
                            if r["disturbance"].get("verdict") == "RECOVERED")
            peaks = [r["disturbance"]["peak_deg"] for r in applied]
            settles = [r["disturbance"]["settle_time"] for r in applied
                       if r["disturbance"].get("settle_time") is not None]
            summary[label]["disturbance"] = {
                "kick_applied": len(applied),
                "kick_missing": len(runs) - len(applied),
                "recovered": recovered,
                "recovered_rate": (round(recovered / len(applied), 3)
                                   if applied else None),
                "median_peak_tilt_deg": (round(sorted(peaks)[len(peaks) // 2], 2)
                                         if peaks else None),
                "worst_peak_tilt_deg": round(max(peaks), 2) if peaks else None,
                "median_settle_s": (round(sorted(settles)[len(settles) // 2], 3)
                                    if settles else None),
            }
        print(f"{label:20s} walked {walked}/{len(runs)}  upright {upright}/{len(runs)}  "
              f"median_tilt={summary[label]['median_max_tilt_deg']}deg"
              + (f"  recovered {summary[label]['disturbance']['recovered']}/"
                 f"{summary[label]['disturbance']['kick_applied']}"
                 if args.disturb_at is not None else ""))
    print(json.dumps(summary, indent=1))

    if len(results) == 2:
        c = summary["closed_loop"]["upright_rate"]
        o = summary["open_loop_control"]["upright_rate"]
        print(f"\n[gait] upright rate: closed-loop {c} vs open-loop control {o}")
        if args.disturb_at is not None:
            cd = summary["closed_loop"]["disturbance"]
            od = summary["open_loop_control"]["disturbance"]
            print(f"[gait] push recovery: closed-loop {cd['recovered']}/"
                  f"{cd['kick_applied']} (peak {cd['median_peak_tilt_deg']}deg) "
                  f"vs open-loop {od['recovered']}/{od['kick_applied']} "
                  f"(peak {od['median_peak_tilt_deg']}deg)")
            missing = cd["kick_missing"] + od["kick_missing"]
            if missing:
                print(f"[gait] WARNING: {missing} trial(s) had NO kick applied — "
                      "those are undisturbed runs and prove nothing about "
                      "rejection.")
        if args.trials < 10:
            print(f"[gait] NOTE: {args.trials} trials per arm is too few to call "
                  "a difference significant; this is a direction, not a rate.")

    # With --trials the exit code reflects the ARM, not one lucky run: a single
    # WALKED out of ten is not a working gait, and an exit code that says so
    # is the whole reason this flag exists.
    primary = results.get(True, results.get(False, []))
    if not primary:
        return 1
    walked = sum(1 for r in primary if r.get("verdict") == "WALKED")
    return 0 if walked * 2 > len(primary) else 1


def run_trial(br: "Bridge", gait: dict, args, stabilize: bool,
              capture: str | None = None) -> dict:
    """One scored run from a freshly rebuilt scene.

    The scene is torn down and rebuilt per trial rather than reusing the
    stage.  A Go2 left holding its last joint targets topples within seconds,
    so a second trial starting from wherever the first one ended would be
    scoring the recovery of a fallen robot, not the gait.
    """
    br.sim_control("stop")
    time.sleep(1.0)
    _ok("scene", br.execute(build_scene_code(args.stiffness, args.damping)))
    br.sim_control("play")
    time.sleep(2.0)

    info = _ok("driver", br.execute(build_driver_code(
        gait, args.seconds, args.stiffness, args.damping, args.sample_every,
        stabilize=stabilize, kp=args.kp, kd=args.kd,
        disturb=disturb_spec(args), push_window=args.disturb_window)))
    if stabilize and len(info.get("legs_wired", [])) != 4:
        # Silently trimming two legs would look like a weak controller rather
        # than a wiring bug, so it is fatal instead.
        raise RuntimeError(
            f"attitude trim wired to {info.get('legs_wired')} — expected all 4 "
            f"legs; solver DOF names are {info.get('joint_names')}")

    # Capture from the MIDDLE of the scored window.  An end-of-run frame is
    # evidence about a moment nobody measured: the driver stops commanding at
    # DURATION and the robot topples immediately after.
    mid_deadline = time.time() + args.seconds * 0.5
    if capture:
        br.camera_look_at(ROBOT_PATH, distance=3.0)
    time.sleep(max(0.0, mid_deadline - time.time()))
    mid_shot = br.capture() if capture else None
    time.sleep(max(0.0, args.seconds * 0.5 + 2.0))

    collected = _ok("collect", br.execute(COLLECT_CODE))
    if collected.get("err"):
        print(f"[gait] callback error: {collected['err']}", file=sys.stderr)
    _ok("cleanup", br.execute(CLEANUP_CODE))

    score = score_trace(collected.get("trace", []))
    score.update(score_disturbance(collected, args))
    score["physics_steps"] = collected.get("steps")
    score["stabilized_steps"] = collected.get("stabilized")
    score["max_cmd_rad"] = collected.get("max_cmd_seen")
    score["stabilize"] = stabilize

    if capture and mid_shot:
        data = mid_shot.get("result", {})
        b64 = data.get("image_base64") or data.get("image") or data.get("data")
        if b64:
            with open(capture, "wb") as fh:
                fh.write(base64.b64decode(b64))
            print(f"[gait] wrote {capture} (mid-window)")
    return score


if __name__ == "__main__":
    raise SystemExit(main())
