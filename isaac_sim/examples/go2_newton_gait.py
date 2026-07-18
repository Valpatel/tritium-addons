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
import dataclasses
import json
import math
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
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

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

# Obstacles: static box colliders standing on the slab.  These exist so the
# body has something to route AROUND -- capability 3 is pathfinding, and a
# route through an empty world is indistinguishable from walking straight.
# Same geometry contract as tritium_lib.planning.SceneObstacle (center +
# half-extents in meters), so the boxes the planner avoids and the boxes the
# solver collides with are the same numbers, not two hand-kept lists.
#
# Every obstacle prim is REMOVED and re-authored, and any left over from a
# longer previous run is removed too.  Skipping a path that already exists --
# the obvious way to write this -- is what made the guarantee above a lie in
# practice: a stale /World/obstacle_0 survived across runs, so the planner
# routed around the box it was handed while the solver held a box at a
# different place and size (measured: asked for center (1.5,0.4) half
# (0.5,0.5,0.35), stage held AABB x 1.65..2.35, y -1.0..1.0).  Every
# clearance number that run produced graded geometry that was not there.
# The stage is a live process that outlives any one trial, so identity of
# path is not identity of geometry, and only re-authoring makes it so.
_obs_specs = __OBSTACLES_JSON__
_stale = 0
while True:
    _path = "/World/obstacle_%d" % (len(_obs_specs) + _stale)
    if not stage.GetPrimAtPath(_path).IsValid():
        break
    stage.RemovePrim(_path)
    _stale += 1
for _i, _obs in enumerate(_obs_specs):
    _path = "/World/obstacle_%d" % _i
    if stage.GetPrimAtPath(_path).IsValid():
        stage.RemovePrim(_path)
    _box = UsdGeom.Cube.Define(stage, _path)
    _box.CreateSizeAttr(2.0)  # unit cube; the scale op sets real half-extents
    _bxf = UsdGeom.Xformable(_box.GetPrim())
    _bxf.ClearXformOpOrder()
    _bxf.AddTranslateOp().Set(Gf.Vec3d(*_obs["center"]))
    _bxf.AddScaleOp().Set(Gf.Vec3f(*_obs["half"]))
    UsdPhysics.CollisionAPI.Apply(_box.GetPrim())
    # Static, like the slab: no RigidBodyAPI, so a nudge cannot shove a wall.
    _box.GetPrim().CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(0.9, 0.2, 0.3)])

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


def build_scene_code(stiffness: float, damping: float,
                     obstacles: "list | None" = None) -> str:
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
            .replace("__OBSTACLES_JSON__", repr(obstacle_specs(obstacles or [])))
            .replace("__STIFFNESS__", repr(float(stiffness)))
            .replace("__DAMPING__", repr(float(damping))))


def obstacle_specs(obstacles) -> list[dict]:
    """Convert ``SceneObstacle``s into the plain dicts the scene code authors.

    The conversion exists so the SAME obstacle objects that are handed to the
    planner are the ones authored into the solver's stage.  Keeping two lists
    -- one to route around, one to collide with -- is how a demo ends up
    routing around a wall that isn't there, or walking through one that is.
    """
    specs = []
    for obs in obstacles:
        if obs.yaw_deg:
            # The scene authors axis-aligned boxes only.  Silently dropping a
            # yaw would put the planner's obstacle and the solver's obstacle
            # in different places -- exactly the divergence this function
            # exists to prevent -- so refuse instead.
            raise ValueError(
                f"{obs.prim_path}: yaw_deg={obs.yaw_deg} is not authored by "
                "this scene; pass an axis-aligned box")
        specs.append({
            "center": tuple(float(v) for v in obs.center),
            "half": tuple(float(v) for v in obs.half_extents),
        })
    return specs


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
         # Stall tripwire.  This callback rides the APP-UPDATE stream, so
         # anything synchronous on the kit main thread -- a viewport render,
         # a bridge RPC -- freezes it while Newton keeps integrating
         # underneath.  The largest sim-time gap between consecutive
         # callbacks makes such a stall VISIBLE in the collected numbers: a
         # healthy run sits near the frame period (~17 ms), while the mid-run
         # capture that minted the retired "~5 N*s inverts it" ceiling shows
         # up as a multi-hundred-ms hole in the control loop.
         "t_last_cb": None, "max_cb_gap_s": 0.0,
         # Disturbance bookkeeping.  `kicks` records what actually landed and
         # when -- a run that reports an empty list did NOT run the experiment
         # it claims to, and scoring it would compare two undisturbed arms.
         "kicks": [], "d_prev": 0.0, "body_mass": None, "push": None,
         # Route following.  Present but empty when no route was given, so a
         # straight run and a followed run collect the same shape.
         "follow": [], "arrived_at": None}


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


# ---- open-loop steering -------------------------------------------------
# The mixing law is NOT reimplemented here either: tritium_lib.control's
# differential_stride is the same function the headless tests drive.  This
# block only knows which DOF belongs to which side of the body.
_TWIST = __TWIST_JSON__
_STEER_BIAS = None
if _TWIST:
    from tritium_lib.control import TwistCommand, differential_stride
    _STEER_BIAS = differential_stride(
        TwistCommand(linear_mps=float(_TWIST["linear"]),
                     angular_rps=float(_TWIST["angular"])),
        track_width_m=float(_TWIST["track"]),
        nominal_mps=float(_TWIST["nominal"]),
    )
    state["steer"] = {"left": round(_STEER_BIAS.left_scale, 4),
                      "right": round(_STEER_BIAS.right_scale, 4)}

# ---- LIVE command surface ----------------------------------------------
# Both blocks above decide the body's motion BEFORE it starts moving: the
# steer bias is baked from the command line, the route is fixed at build
# time.  Either way a run is a recording.  This block is the first path by
# which something outside the process can change what the body is doing
# WHILE it is doing it -- which is what a teleop link, an operator console
# or a re-planner all need, and none of them could have.
#
# UDP and non-blocking, both deliberately.  Kit is main-thread-only and a
# blocking read inside a physics step callback hangs the whole app silently
# (the failure mode tick 20 spent itself on), so the socket is drained with
# recvfrom until it would block and never waits.  UDP because a command
# stream wants the newest sample, not every sample: a TCP link that stalls
# delivers a queue of stale commands, and a body executing history is worse
# than a body that stopped.  Loss is handled by the watchdog, not by retry.
#
# The decode, the ordering gate and the staleness rules are NOT implemented
# here -- they are tritium_lib.control.CommandLink, the same object the
# headless tests drive.  This block only knows how to hold a socket.
_LIVE = __LIVE_JSON__
_LIVE_LINK = None
_LIVE_SOCK = None
if _LIVE:
    import socket as _socket
    from tritium_lib.control import CommandLimits, CommandLink, differential_stride
    _LIVE_LINK = CommandLink(
        limits=CommandLimits(max_linear_mps=float(_LIVE["max_linear"]),
                             max_angular_rps=float(_LIVE["max_angular"])),
        timeout_s=float(_LIVE["timeout"]),
    )
    _LIVE_SOCK = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _LIVE_SOCK.setblocking(False)
    _LIVE_SOCK.bind(("0.0.0.0", int(_LIVE["port"])))
    state["live"] = []
    state["live_sock_err"] = None


def _drain_live(elapsed):
    # Read every datagram waiting right now; the link keeps the last accepted.
    #
    # Bounded on purpose.  An unbounded drain hands a flooding sender control
    # of how long the physics step takes, which is a denial of service against
    # the simulation dressed up as a control input.
    for _ in range(__LIVE_DRAIN__):
        try:
            _pkt, _ = _LIVE_SOCK.recvfrom(2048)
        except BlockingIOError:
            break
        except OSError as exc:
            state["live_sock_err"] = repr(exc)
            break
        _LIVE_LINK.ingest(_pkt, elapsed)

# ---- closed-loop route following ---------------------------------------
# The difference from the block above is the whole point of this tick: that
# steer bias is computed ONCE from the command line and never changes, so the
# body drives a fixed arc and cannot be said to be following anything.  This
# one recomputes the bias every step from the body's OWN measured pose, which
# is what makes it closed-loop over position rather than open-loop over time.
#
# The tracker is not reimplemented here either -- it is
# tritium_lib.control.PurePursuitFollower, the same object the headless tests
# drive.  This block only knows how to read a pose out of Newton.
_ROUTE = __ROUTE_JSON__
_FOLLOWER = None
_YAW_LOOP = None
_ROUTE_PTS = []
if _ROUTE:
    from tritium_lib.control import PurePursuitFollower, TwistCommand, differential_stride
    # Imported unconditionally, NOT inside the yaw_closed_loop branch below:
    # the step callback measures the achieved yaw rate in BOTH arms (the
    # open-loop arm records it for diagnosis without acting on it), so a
    # conditional import here is a NameError that kills the control arm too.
    from tritium_lib.control import yaw_rate_from_headings
    from tritium_lib.geo.isaac_frame import quat_to_yaw_deg
    _ROUTE_PTS = [(float(p[0]), float(p[1])) for p in _ROUTE["waypoints"]]
    _FOLLOWER = PurePursuitFollower(
        lookahead_m=float(_ROUTE["lookahead"]),
        cruise_mps=float(_ROUTE["cruise"]),
        max_angular_rps=float(_ROUTE["max_angular"]),
        goal_tolerance_m=float(_ROUTE["goal_tolerance"]),
        slow_radius_m=float(_ROUTE["slow_radius"]),
    )
    # Standing still is stride scale ZERO on both sides, not None: None means
    # "never steered" and would leave the gait walking straight past the goal.
    _STOP_BIAS = differential_stride(
        TwistCommand.stop(), track_width_m=float(_ROUTE["track"]),
        nominal_mps=float(_ROUTE["nominal"]))
    state["follow"] = []
    state["arrived_at"] = None

    # Inner rate loop.  The follower says WHERE to point; this says HOW HARD,
    # by watching the yaw rate the body actually delivered and correcting the
    # demand by the shortfall.  Without it the Go2 delivers ~12% of what is
    # asked and the route ends short rather than wrong.
    _YAW_LOOP = None
    if _ROUTE.get("yaw_closed_loop"):
        from tritium_lib.control import YawRateLoop
        _YAW_LOOP = YawRateLoop(
            kp=float(_ROUTE["yaw_kp"]), ki=float(_ROUTE["yaw_ki"]),
            max_output_rps=float(_ROUTE["yaw_max"]))
    # Imported unconditionally for the same reason yaw_rate_from_headings is:
    # the trace records the filtered rate in BOTH arms so the two are
    # comparable, and a name bound only under a flag is a NameError at call
    # time that compile() cannot see.
    from tritium_lib.control import StrideFilter
    _YAW_FILT = None
    if _ROUTE.get("yaw_filter"):
        # stride_hz comes from the GAIT FILE, which is the frequency the body
        # actually emits -- and it is NOT a function of --speed.  The gait
        # table's own generation speed sets the stride (0.6 m/s -> 0.975 Hz);
        # --speed is only the pure-pursuit cruise.  Deriving the window from
        # the cruise instead would notch a frequency nothing is producing.
        _YAW_FILT = StrideFilter.from_stride_hz(stride_hz)

    state["yaw_prev"] = None
    state["yaw_t_prev"] = None

# Left legs are FL/RL, right are FR/RR -- read off LegPlacement.y > 0 rather
# than hardcoded, so a body with a different naming scheme stays correct.
_SIDE_DOF = {"left": [], "right": []}
for _leg in GO2_LEGS:
    if _leg.name in leg_dof:
        _SIDE_DOF["left" if _leg.y > 0 else "right"].extend(leg_dof[_leg.name])


def _apply_steer(targets, bias):
    # Scale each leg's stride about the STAND pose, not about zero.  The
    # stride is the deviation from stand; scaling the absolute angle would
    # also drag the body's neutral crouch up and down and change its ride
    # height, which reads as a limp rather than a turn.  A negative scale
    # therefore swings that side's legs backwards through stand, which is
    # exactly how a body spins in place.
    for _side, _scale in (("left", bias.left_scale), ("right", bias.right_scale)):
        for _i in _SIDE_DOF[_side]:
            targets[_i] = stand_vec[_i] + (targets[_i] - stand_vec[_i]) * _scale
    return targets


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

        # Record the largest sim-time hole between consecutive callbacks.
        # Only in-window gaps count: after DURATION the subscription idles
        # until cleanup, and the collect RPC itself would register as a
        # (harmless) stall.
        if state["t_last_cb"] is not None:
            _gap = elapsed - state["t_last_cb"]
            if _gap > state["max_cb_gap_s"]:
                state["max_cb_gap_s"] = _gap
        state["t_last_cb"] = elapsed

        # Kick BEFORE this step's targets are commanded, so the controller
        # first sees the disturbance on the very next step rather than
        # half a stride later.
        if disturb is not None:
            for _imp in disturb.due(state["d_prev"], elapsed):
                _apply_impulse(_imp, elapsed)
            state["d_prev"] = elapsed
            _pump_push(elapsed)

        targets = _sample_targets(elapsed).copy()

        # Route following runs BEFORE the steer is applied, because it is what
        # decides the steer.  Like the stabiliser, it reads the root EVERY
        # step rather than on the trace's SAMPLE_EVERY cadence: a position
        # loop closed at logging rate is a different, much slower controller
        # than the one the headless tests characterise.
        root = None
        _bias = _STEER_BIAS

        # The live link is polled EVERY step, not on the trace's sample
        # cadence: a command surface read at logging rate is a slower, laggier
        # controller than the one the headless tests characterise, and the
        # watchdog's timeout would be quantised to the log interval.
        if _LIVE_LINK is not None:
            _drain_live(elapsed)
            _live_twist = _LIVE_LINK.poll(elapsed)
            _bias = differential_stride(
                _live_twist, track_width_m=float(_LIVE["track"]),
                nominal_mps=float(_LIVE["nominal"]))
            if state["steps"] % SAMPLE_EVERY == 0:
                # Recorded so the run can be scored against what was actually
                # SENT.  Without this column a live run is indistinguishable
                # from a baked one -- the whole claim of this path is that the
                # body followed an external command, and that claim needs the
                # command in the trace to be checkable at all.
                state["live"].append(
                    [round(float(elapsed), 3),
                     round(float(_live_twist.linear_mps), 4),
                     round(float(_live_twist.angular_rps), 4),
                     int(_LIVE_LINK.accepted),
                     int(_LIVE_LINK.rejected)])

        if _FOLLOWER is not None:
            root = _to_numpy(view.get_root_transforms())
            # Isaac hands back position + an XYZW quaternion; quat_to_yaw_deg
            # speaks WXYZ.  This reorder is the same load-bearing one the
            # stabiliser below does -- get it wrong and the body chases a
            # heading rotated out from under it.
            qx, qy, qz, qw = (float(v) for v in root[0][3:7])
            _hdg = math.radians(quat_to_yaw_deg((qw, qx, qy, qz)))
            _fs = _FOLLOWER.update(
                (float(root[0][0]), float(root[0][1]), _hdg), _ROUTE_PTS)

            # Measure the yaw rate the body ACHIEVED over the last step,
            # before deciding what to ask for next.  Measured from the root
            # transform, never from the command -- a loop closed on its own
            # output is not a loop.
            _yaw_dt = 0.0
            _yaw_raw = 0.0
            if state["yaw_prev"] is not None:
                _yaw_dt = elapsed - state["yaw_t_prev"]
                _yaw_raw = yaw_rate_from_headings(
                    state["yaw_prev"], _hdg, _yaw_dt)
            state["yaw_prev"] = _hdg
            state["yaw_t_prev"] = elapsed

            # Null the gait out of the measurement.  _yaw_raw is the body's
            # turn PLUS its own stride rock, and the rock is the larger of the
            # two; a boxcar over one stride period zeroes the rock and its
            # harmonics exactly, leaving the net turn.  Filtered in both arms
            # so the trace's two columns mean the same thing in each.
            _yaw_meas = _yaw_raw
            _yaw_ready = True
            if _YAW_FILT is not None:
                _yaw_meas = _YAW_FILT.update(elapsed, _yaw_raw)
                # Until a full window has passed the average is over a partial
                # window, which does NOT null the gait -- feeding it to the
                # integrator would inject exactly the stride-synchronous error
                # the filter exists to remove.  Hold the loop off instead.
                _yaw_ready = _YAW_FILT.ready

            # The twist actually handed to the mixer, whatever path produced
            # it -- this is what the trace must compare the measurement to.
            _bias_twist = None
            if _fs.arrived:
                if state["arrived_at"] is None:
                    state["arrived_at"] = round(float(elapsed), 3)
                _bias_twist = TwistCommand.stop()
                _bias = _STOP_BIAS
                if _YAW_LOOP is not None:
                    # Standing still with a charged integrator would spin the
                    # body on the spot at the goal.
                    _YAW_LOOP.reset()
            else:
                _twist = _fs.twist
                _yaw_cmd = _twist.angular_rps
                if _YAW_LOOP is not None and _yaw_ready:
                    _corr = _YAW_LOOP.update(_yaw_cmd, _yaw_meas, _yaw_dt)
                    _twist = TwistCommand(linear_mps=_twist.linear_mps,
                                          angular_rps=_corr.compensated_rps)
                _bias_twist = _twist
                _bias = differential_stride(
                    _twist, track_width_m=float(_ROUTE["track"]),
                    nominal_mps=float(_ROUTE["nominal"]))
            if state["steps"] % SAMPLE_EVERY == 0:
                # Recorded for diagnosis only.  The VERDICT is scored offline
                # from the ground-truth pose trace against the route, never
                # from these numbers -- a controller grading its own tracking
                # error is marking its own homework.
                # Column 3 is the command the body was ACTUALLY GIVEN, i.e.
                # post-compensation.  It used to be _fs.twist.angular_rps --
                # the follower's raw pre-loop demand -- so whenever the rate
                # loop was on, the plant-gain ratio was computed against a
                # command that never reached the mixer.  That is what made
                # tick 18's "12% steering authority" and its -0.186 sign
                # uninterpretable.  Both the raw and the filtered measurement
                # are recorded so the gait's contribution stays visible
                # instead of being quietly averaged away.
                _cmd_given = (_bias_twist.angular_rps
                              if _bias_twist is not None else 0.0)
                state["follow"].append(
                    [round(float(elapsed), 3),
                     round(float(_fs.cross_track_m), 4),
                     round(float(_fs.distance_to_goal_m), 4),
                     round(float(_cmd_given), 4),
                     int(_fs.target_index),
                     round(float(_yaw_meas), 4),
                     round(float(_bias.left_scale), 4),
                     round(float(_bias.right_scale), 4),
                     round(float(_yaw_raw), 4),
                     round(float(_fs.twist.angular_rps), 4)])

        # Steering is applied BEFORE the attitude trim, so the stabiliser's
        # vertical corrections ride on top of the steered stride rather than
        # being scaled away by it.
        if _bias is not None:
            targets = _apply_steer(targets, _bias)

        # The root transform is read EVERY step when stabilising, not on the
        # SAMPLE_EVERY cadence -- feedback at the trace's logging rate would be
        # a control loop running an order of magnitude slower than the
        # disturbance it is meant to reject.
        if __STABILIZE__:
            if root is None:
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
                          "joint_names": joint_names,
                          "live_sock": _LIVE_SOCK}
result = {"joint_names": joint_names, "count": int(count),
          "dof": int(len(joint_names)), "stabilize": bool(__STABILIZE__),
          "legs_wired": sorted(leg_dof)}
"""


def build_driver_code(gait: dict, duration: float, stiffness: float,
                      damping: float, sample_every: int,
                      stabilize: bool = True, kp: float = None,
                      kd: float = None, lib_src: str = LIB_SRC,
                      disturb: list[dict] | None = None,
                      push_window: float = 0.1,
                      twist: dict | None = None,
                      route: dict | None = None,
                      live: dict | None = None,
                      live_drain: int = 64) -> str:
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
    code = code.replace("__TWIST_JSON__", repr(dict(twist) if twist else None))
    code = code.replace("__ROUTE_JSON__", repr(dict(route) if route else None))
    code = code.replace("__LIVE_JSON__", repr(dict(live) if live else None))
    code = code.replace("__LIVE_DRAIN__", repr(int(live_drain)))
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
          "body_mass": st["body_mass"],
          "stabilized": st.get("stabilized"),
          "max_cmd_seen": st.get("max_cmd_seen"),
          "max_cb_gap_s": st.get("max_cb_gap_s"),
          "follow": st.get("follow", []), "arrived_at": st.get("arrived_at"),
          "live": st.get("live", []), "live_sock_err": st.get("live_sock_err")}
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
    # Close the live socket explicitly.  Leaving it to GC leaks the bound
    # port for the life of the kit process, so the NEXT run of this script
    # dies on bind -- and since the kit is long-lived and reused across
    # runs, that turns one leak into every subsequent run failing.
    try:
        _sock = g.get("live_sock")
        if _sock is not None:
            _sock.close()
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

    # Heading change -- the only metric that can see a TURN.  Displacement and
    # tilt are both blind to yaw by construction (tilt_from_upright_deg
    # deliberately excludes it), so a steering run scored on those alone would
    # grade a perfect circle identically to a straight line of the same arc
    # length.  Unwrapped, because a turn that crosses +/-180 must not read as
    # a turn back the other way.
    def _yaw_deg(qx, qy, qz, qw):
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.degrees(math.atan2(siny, cosy))

    yaws = [_yaw_deg(r[4], r[5], r[6], r[7]) for r in trace if len(r) >= 8]
    yaw_total = 0.0
    for a, b in zip(yaws, yaws[1:]):
        step = (b - a + 180.0) % 360.0 - 180.0
        yaw_total += step

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
        "yaw_change_deg": round(yaw_total, 2) if yaws else None,
        "yaw_rate_dps": round(yaw_total / dt, 3) if yaws else None,
        # Order matters: a tumble outranks distance, because a tumbling body
        # covers ground and would otherwise be reported as a working gait.
        "verdict": (
            "TUMBLED" if tumbled
            else "COLLAPSED" if collapsed
            else "WALKED" if dist > 0.10
            else "STATIONARY"
        ),
    }


def split_scoreable(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition trials into (scoreable, capture-perturbed).

    A trial that captured mid-window did not measure the gait.  The viewport
    render runs synchronously on the kit main thread -- the same thread that
    pumps the app-update stream driving the control callback -- so for the
    duration of the render the controller is frozen while Newton keeps
    integrating underneath (measured: same command and push, 0/8 upright
    capture-on vs 8/8 capture-free).  Every rate this tool reports goes
    through this filter, so the exclusion is enforced by structure rather
    than by an operator having read a help string -- the failure mode that
    let the false "~5 N*s inverts the body" ceiling reach three documents.
    """
    scoreable = [r for r in runs if not r.get("capture_perturbed")]
    perturbed = [r for r in runs if r.get("capture_perturbed")]
    return scoreable, perturbed


def arm_summary(runs: list[dict], disturb_at: float | None = None) -> dict:
    """Aggregate one arm's trials into its reported rates.

    Rates are computed over the scoreable trials ONLY (``split_scoreable``);
    a capture-perturbed trial that reaches this function is excluded and
    reported under ``capture_excluded`` instead of silently disappearing --
    or worse, silently counting.  If nothing scoreable remains, the summary
    says so (``trials: 0``, rates ``None``) rather than inventing a rate
    from perturbed runs.
    """
    runs, perturbed = split_scoreable(runs)
    walked = sum(1 for r in runs if r.get("verdict") == "WALKED")
    upright = sum(1 for r in runs if not r.get("tumbled"))
    tilts = [r["max_tilt_deg"] for r in runs if r.get("max_tilt_deg") is not None]
    dists = [r["displacement_m"] for r in runs if r.get("displacement_m") is not None]
    summary = {
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
    if perturbed:
        summary["capture_excluded"] = len(perturbed)
        summary["capture_excluded_verdicts"] = [r.get("verdict") for r in perturbed]
    # Yaw is reported unconditionally, not just when --steer is set: a
    # straight run's heading drift is the baseline that makes a steered
    # run's number mean anything, and it is only credible if it is
    # collected the same way in both arms rather than switched on for the
    # arm being advertised.
    yaws = [r["yaw_change_deg"] for r in runs
            if r.get("yaw_change_deg") is not None]
    if yaws:
        summary["median_yaw_change_deg"] = round(
            sorted(yaws)[len(yaws) // 2], 2)
        summary["yaw_changes_deg"] = [round(y, 2) for y in yaws]
    if disturb_at is not None:
        # Recovery is scored only over trials where the kick actually
        # landed.  Counting a NOT_APPLIED run as a recovery would inflate
        # exactly the number this experiment exists to measure.
        applied = [r for r in runs if r.get("disturb_ok")]
        recovered = sum(1 for r in applied
                        if r["disturbance"].get("verdict") == "RECOVERED")
        peaks = [r["disturbance"]["peak_deg"] for r in applied]
        settles = [r["disturbance"]["settle_time"] for r in applied
                   if r["disturbance"].get("settle_time") is not None]
        summary["disturbance"] = {
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
    return summary


def twist_spec(args) -> dict | None:
    """Build the steering command from the CLI, or None for straight ahead.

    ``None`` rather than a zero twist on purpose: a zero twist still runs the
    mixer and rescales every stride by 1.0, and "multiply by one" is exactly
    the kind of no-op that quietly stops being a no-op after a refactor.  The
    unsteered arm of an A/B must run the *original* code path untouched, or it
    is not a control.
    """
    if not args.steer:
        return None
    return {
        "linear": float(args.speed),
        "angular": float(args.steer),
        "track": float(args.steer_track),
        "nominal": float(args.speed),
    }


def live_spec(args) -> dict | None:
    """Build the live-command-surface config, or None to stay batch-driven.

    ``None`` for the same reason ``twist_spec`` returns None: a run with no
    ``--live-port`` must not open a socket at all, so the existing batch path
    stays byte-for-byte the code every previous tick measured.
    """
    if not args.live_port:
        return None
    return {
        "port": int(args.live_port),
        "timeout": float(args.live_timeout),
        "max_linear": float(args.live_max_linear),
        "max_angular": float(args.live_max_angular),
        "track": float(args.steer_track),
        "nominal": float(args.speed),
    }


def parse_points(text: str) -> list[tuple[float, float]]:
    """Parse ``"x,y;x,y;..."`` into world-meter points."""
    points = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"waypoint wants X,Y in meters, got {chunk!r}")
        points.append((float(parts[0]), float(parts[1])))
    return points


def parse_obstacles(specs) -> list:
    """Parse ``--obstacle X,Y,HX,HY,HZ`` into ``SceneObstacle``s.

    Z is not asked for: an obstacle a ground body must route around stands ON
    the ground, so the box is seated on the slab and ``HZ`` is its height
    above it.  A floating box would be a different feature (and would not
    obstruct anything a quadruped does).
    """
    from tritium_lib.planning.scene_costmap import SceneObstacle

    obstacles = []
    for i, spec in enumerate(specs or []):
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 5:
            raise SystemExit(
                f"--obstacle wants X,Y,HX,HY,HZ in meters, got {spec!r}")
        x, y, hx, hy, hz = (float(p) for p in parts)
        if hx < 0 or hy < 0 or hz < 0:
            # SceneObstacle would raise ValueError here.  A typo in a CLI flag
            # should read as a CLI error, not a library traceback.
            raise SystemExit(
                f"--obstacle half-extents must be >= 0, got {spec!r}")
        obstacles.append(SceneObstacle(
            prim_path=f"/World/obstacle_{i}",
            center=(x, y, hz),  # seated on the slab: center z == half height
            half_extents=(hx, hy, hz),
        ))
    return obstacles


def route_spec(args, waypoints) -> dict | None:
    """Build the route-following command, or None to walk straight.

    ``None`` rather than an empty route for the same reason ``twist_spec``
    returns None: the unfollowed arm of an A/B must run the original code
    path untouched, or it is not a control.
    """
    if not waypoints:
        return None
    return {
        "waypoints": [list(p) for p in waypoints],
        "lookahead": float(args.lookahead),
        "cruise": float(args.speed),
        "max_angular": float(args.max_angular),
        "goal_tolerance": float(args.goal_tolerance),
        "slow_radius": float(args.slow_radius),
        "track": float(args.steer_track),
        "nominal": float(args.speed),
        # The inner rate loop is OFF by default and that is deliberate: the
        # open-loop arm has to stay reachable unchanged, or the A/B that
        # justifies this loop has no control to compare against.
        "yaw_closed_loop": bool(args.closed_loop_yaw),
        "yaw_kp": float(args.yaw_kp),
        "yaw_ki": float(args.yaw_ki),
        "yaw_max": float(args.yaw_max),
        # Boxcar the measured yaw rate over one stride period before any loop
        # sees it.  The window is derived from the gait's OWN stride_hz rather
        # than passed in, because the null has to sit on the frequency this
        # body actually emits -- a hardcoded window notches whatever the last
        # tuning session happened to walk at.  The measurement is filtered in
        # BOTH arms when this is on; only the closed-loop arm acts on it.
        "yaw_filter": bool(args.yaw_filter),
        # The scorer stands off by the BODY RADIUS, not by the planner's
        # clearance.  These are two different quantities and conflating them
        # breaks the referee in both directions:
        #
        #   plan_clearance is an INFLATION radius -- a preference for how much
        #   room to leave, satisfied on a discrete grid, so a route planned at
        #   0.15 m resolution legitimately lands ~0.18 m from a box it asked
        #   to clear by 0.30 m.  Grading against it fails the planner's own
        #   optimal path (measured: COLLIDED at clearance 0.177), so no run
        #   could ever pass.
        #   body_radius is the FOOTPRINT -- the half-width that physically
        #   collides.  Below it the hull is inside the box, which is the only
        #   thing "collided" can honestly mean.
        #
        # Nav2 draws exactly this line between inflation and footprint.
        "clearance": float(args.body_radius),
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

    # A kick RECORD is not a kick.  The solver can accept the force and the
    # body still gain almost nothing along the axis it was pushed on -- which
    # is what happens when the body is busy falling.  The first live 3 N-s
    # A/B logged dv=[-0.089, 0.0121, 0.1921] for a +Y push: the right
    # MAGNITUDE, almost none of it on Y.  That trial's 178-degree tumble was
    # a spontaneous fall being charged to the controller.
    #
    # So the trial is only admitted if the body moved along the COMMANDED
    # direction.  Note this cuts both ways and is not a thumb on the scale:
    # the same run had a closed-loop trial measure dv_y = -0.087 and score
    # RECOVERED, and that favourable trial is excluded too.
    from tritium_lib.control import kick_landed

    mass = collected.get("body_mass")
    verdicts = []
    for k in kicks:
        try:
            v = kick_landed(commanded=k["impulse_ns"],
                            measured_dv=k["measured_dv_mps"],
                            body_mass=float(mass))
        except (ValueError, KeyError, TypeError) as exc:
            return {"disturbance": dict(payload, verdict="UNSCORABLE",
                                        reason=f"kick check: {exc}"),
                    "disturb_ok": False}
        verdicts.append(v)
        k["landed"] = v.as_dict()

    if not all(v.landed for v in verdicts):
        worst = min(verdicts, key=lambda v: v.fraction)
        payload["verdict"] = "NOT_APPLIED"
        payload["reason"] = (
            f"push did not land on its commanded axis: projected "
            f"{worst.projected_dv:.4f} m/s vs expected {worst.expected_dv:.4f} "
            f"({worst.fraction:.0%} of J/m)")
        return {"disturbance": payload, "disturb_ok": False}

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
    ap.add_argument("--steer", type=float, default=0.0, metavar="RAD_S",
                    help="commanded yaw rate. Positive turns to PORT (left) "
                         "per REP-103. Mixed to a left/right stride ratio by "
                         "tritium_lib.control.differential_stride.")
    ap.add_argument("--steer-track", type=float, default=0.26, metavar="M",
                    help="body track width used by the mixer (Go2: 0.26)")
    ap.add_argument("--live-port", type=int, metavar="PORT",
                    help="listen on this UDP port for live twist commands "
                         "({\"cmd\":\"twist\",\"seq\":N,\"linear_mps\":..,"
                         "\"angular_rps\":..}), overriding --steer every "
                         "physics step. Without it the run is batch-driven "
                         "exactly as before.")
    ap.add_argument("--live-timeout", type=float, default=0.5, metavar="S",
                    help="stop the body if no valid command arrives for this "
                         "long (tritium_lib.control watchdog; default 0.5)")
    ap.add_argument("--live-max-linear", type=float, default=1.0, metavar="MPS",
                    help="clamp on commanded forward speed (default 1.0)")
    ap.add_argument("--live-max-angular", type=float, default=2.0, metavar="RPS",
                    help="clamp on commanded yaw rate (default 2.0)")
    ap.add_argument("--route", metavar="X,Y;X,Y;...",
                    help="walk this polyline with closed-loop pure pursuit "
                         "instead of a fixed steer")
    ap.add_argument("--plan-to", metavar="X,Y",
                    help="plan a route from the body to this goal around "
                         "--obstacle boxes, then walk it (implies --route)")
    ap.add_argument("--obstacle", action="append", metavar="X,Y,HX,HY,HZ",
                    help="axis-aligned box on the slab; repeatable")
    ap.add_argument("--lookahead", type=float, default=0.8, metavar="M",
                    help="pure-pursuit lookahead distance")
    ap.add_argument("--max-angular", type=float, default=0.8, metavar="RAD_S",
                    help="yaw-rate clamp for the follower")
    ap.add_argument("--dump-live", metavar="PATH.json",
                    help="write the live command column beside the "
                         "ground-truth pose trace, on one clock — the "
                         "evidence a live run needs to be graded at all")
    ap.add_argument("--dump-follow", metavar="PATH.json",
                    help="write the per-step follower trace for offline "
                         "diagnosis (commanded vs MEASURED yaw rate). Never "
                         "an input to any verdict")
    ap.add_argument("--yaw-filter", action="store_true",
                    help="boxcar the MEASURED yaw rate over one stride period "
                         "(tritium_lib.control.StrideFilter) before the loop "
                         "sees it, nulling the gait's own left-right rock. "
                         "Costs half a stride period of group delay")
    ap.add_argument("--closed-loop-yaw", action="store_true",
                    help="close an inner PI loop on MEASURED yaw rate "
                         "(tritium_lib.control.YawRateLoop). Off by default "
                         "so the open-loop arm stays available as a control")
    ap.add_argument("--yaw-kp", type=float, default=1.0, metavar="K")
    ap.add_argument("--yaw-ki", type=float, default=6.0, metavar="K",
                    help="integral gain — this is the term that actually "
                         "cancels a constant plant-gain shortfall")
    ap.add_argument("--yaw-max", type=float, default=4.0, metavar="RAD_S",
                    help="demand ceiling for the rate loop. Set ABOVE "
                         "--max-angular: the point is to ask for more than "
                         "the follower wanted so that what ARRIVES is what "
                         "the follower wanted")
    ap.add_argument("--goal-tolerance", type=float, default=0.35, metavar="M")
    ap.add_argument("--slow-radius", type=float, default=0.0, metavar="M",
                    help="ease off the throttle within this range of the goal")
    ap.add_argument("--plan-resolution", type=float, default=0.15, metavar="M")
    ap.add_argument("--plan-clearance", type=float, default=0.30, metavar="M",
                    help="planner INFLATION radius — how much room to prefer")
    ap.add_argument("--body-radius", type=float, default=0.16, metavar="M",
                    help="body FOOTPRINT half-width used to judge collision; "
                         "the Go2 is ~0.31 m wide. Not the same as "
                         "--plan-clearance, which is only a preference")
    ap.add_argument("--capture",
                    help="write a viewport PNG here, taken from a DEDICATED "
                         "UNSCORED evidence run per arm (-closed/-open "
                         "suffix), never from a scored trial. A mid-run "
                         "render stalls the app-update loop driving the "
                         "control callback (measured 0/8 upright capture-on "
                         "vs 8/8 capture-free, same command and push), so "
                         "the harness quarantines captures structurally: "
                         "every rate comes from capture-free trials only, "
                         "and a captured run's verdict reads UNSCORED[...].")
    ap.add_argument("--capture-distance", type=float, default=3.0, metavar="M",
                    help="camera standoff for --capture. The 3 m default "
                         "frames the BODY; framing a whole ROUTE needs enough "
                         "range to see the obstacle and both endpoints, or "
                         "the frame proves nothing about where it walked")
    ap.add_argument("--capture-elevation", type=float, default=18.0,
                    metavar="DEG", help="camera elevation for --capture; near "
                                        "90 is top-down, which is the view "
                                        "that shows a route being followed")
    ap.add_argument("--capture-at", type=float, default=None, metavar="SEC",
                    help="sim-time to capture at, default mid-window. Clamped "
                         "to the scored window: a frame taken after the driver "
                         "stops is evidence about a moment nobody measured.")
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

    # Obstacles are authored into the stage AND handed to the planner from
    # this one list, so the boxes the route avoids are the boxes the solver
    # can trip over.
    obstacles = parse_obstacles(args.obstacle)
    if args.route and args.plan_to:
        ap.error("--route and --plan-to are mutually exclusive: one hands the "
                 "polyline over, the other plans it")
    if args.plan_to:
        waypoints = plan_route(args, obstacles)
    else:
        waypoints = parse_points(args.route) if args.route else []
    route = route_spec(args, waypoints)
    if obstacles:
        print(f"[gait] {len(obstacles)} obstacle(s) on the slab")

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
        if args.capture:
            # The frame comes from a DEDICATED, UNSCORED evidence run --
            # never from a scored trial.  A mid-run capture stalls the
            # app-update loop that drives the control callback (measured:
            # same command and push, 0/8 upright capture-on vs 8/8
            # capture-free), so a captured run frames the scene but its
            # numbers measure the stall.  Splitting the run keeps the
            # evidence AND keeps every scored trial byte-for-byte the
            # capture-free code path: a scored trial's outcome cannot
            # depend on whether a capture happened, because no scored
            # trial ever captures.
            suffix = "closed" if stabilize else "open"
            evidence = run_trial(
                br, gait, args, stabilize,
                args.capture.replace(".png", f"-{suffix}.png"),
                route=route, obstacles=obstacles)
            print(f"[gait] {label} evidence run: {evidence.get('verdict')} "
                  f"dist={evidence.get('displacement_m')} "
                  f"max_tilt={evidence.get('max_tilt_deg')} "
                  f"cb_gap={evidence.get('max_cb_gap_s')} "
                  "— frame only, excluded from every rate", flush=True)
        runs = []
        for trial in range(1, args.trials + 1):
            score = run_trial(br, gait, args, stabilize, None,
                              route=route, obstacles=obstacles)
            if args.dump_follow and score.get("follow_trace"):
                path = args.dump_follow.replace(
                    ".json", f"-{'closed' if stabilize else 'open'}"
                    f"-t{trial}.json")
                with open(path, "w") as fh:
                    json.dump(score["follow_trace"], fh)
                print(f"[gait] wrote {path}")
            if args.dump_live:
                path = args.dump_live.replace(
                    ".json", f"-{'closed' if stabilize else 'open'}"
                    f"-t{trial}.json")
                with open(path, "w") as fh:
                    json.dump({"live": score.get("live_trace", []),
                               "pose": score.get("pose_trace", []),
                               "live_sock_err": score.get("live_sock_err")}, fh)
                print(f"[gait] wrote {path}")
            score.pop("live_trace", None)
            score.pop("pose_trace", None)
            score.pop("follow_trace", None)  # too big for the summary line
            runs.append(score)
            line = (f"[gait] {label} trial {trial}/{args.trials}: "
                    f"{score.get('verdict')} "
                    f"dist={score.get('displacement_m')} "
                    f"max_tilt={score.get('max_tilt_deg')} "
                    # The stall tripwire, printed on every scored line: a
                    # value far above the ~17 ms frame period means SOMETHING
                    # froze the control callback mid-window and this trial
                    # deserves suspicion whatever its verdict says.
                    f"cb_gap={score.get('max_cb_gap_s')}")
            if route:
                line += (f" | route={score.get('route_verdict')}"
                         f" gap={score.get('route_final_gap_m')}"
                         f" xtrack={score.get('route_max_cross_track_m')}"
                         f" clear={score.get('route_min_clearance_m')}"
                         f" progress={score.get('route_progress_ratio')}")
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
            # flush=True is load-bearing: stdout is block-buffered when
            # this runs over ssh or into a pipe, so a --trials run killed
            # by a timeout loses EVERY result it had already produced.
            print(line, flush=True)
        results[stabilize] = runs

    print("\n" + "=" * 62)
    print("HONEST GAIT RATE — every capture-free trial counted, none discarded")
    print("=" * 62)
    summary = {}
    for stabilize, runs in results.items():
        label = "closed_loop" if stabilize else "open_loop_control"
        summary[label] = s = arm_summary(runs, disturb_at=args.disturb_at)
        if s.get("capture_excluded"):
            # Belt over braces: main() never puts a captured run into `runs`,
            # so this fires only if someone re-wires the loop above -- and
            # then the excluded count is printed rather than silently folded
            # into the rate.
            print(f"[gait] WARNING: {label}: {s['capture_excluded']} "
                  "capture-perturbed trial(s) EXCLUDED from every rate")
        print(f"{label:20s} walked {s['walked']}/{s['trials']}  "
              f"upright {s['upright']}/{s['trials']}  "
              f"median_tilt={s['median_max_tilt_deg']}deg"
              f"  yaw={s.get('median_yaw_change_deg')}deg"
              + (f"  recovered {s['disturbance']['recovered']}/"
                 f"{s['disturbance']['kick_applied']}"
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
    # is the whole reason this flag exists.  Filtered through split_scoreable
    # like every other rate: if only perturbed runs exist, the tool FAILS
    # rather than grading the stall.
    primary, _ = split_scoreable(results.get(True, results.get(False, [])))
    if not primary:
        return 1
    walked = sum(1 for r in primary if r.get("verdict") == "WALKED")
    return 0 if walked * 2 > len(primary) else 1


def score_route(collected: dict, route: dict, obstacles) -> dict:
    """Grade the run against the route it was told to walk.

    Deliberately scored from the GROUND-TRUTH pose trace, not from the
    follower's own ``cross_track_m`` samples.  The follower computes its error
    from the same pose estimate it steers on, so a controller with a broken
    pose reads zero error while walking into a wall; recomputing from the
    solver's root transforms cannot flatter itself that way.  The follower's
    own numbers are still reported, prefixed, for diagnosis only.
    """
    from tritium_lib.control.route_trace import score_route_trace

    trace = collected.get("trace", [])
    positions = [(row[1], row[2]) for row in trace]
    waypoints = [(float(x), float(y)) for x, y in route["waypoints"]]
    scored = score_route_trace(
        positions, waypoints, obstacles,
        goal_tolerance_m=float(route["goal_tolerance"]),
        # Stand the referee off by the same half-width the planner used.  At
        # clearance 0 the scorer grades the body's CENTER point, so a run whose
        # hull clips a box by up to a half-width scores clean -- the referee
        # would be more permissive about a wall than the plan that avoided it.
        clearance_m=float(route.get("clearance", 0.0)),
        # Every box here was typed on the command line, so the list is already
        # curated and the library's terrain cap must not apply: it drops boxes
        # over 100 m as scenery, which would score a body jammed against a
        # long wall as a clean arrival.  Must match plan_route's costmap, or
        # the scorer forgives exactly the walls the planner avoided.
        max_footprint_m=None)
    out = {f"route_{k}": v for k, v in dataclasses.asdict(scored).items()}
    out["route_verdict"] = scored.verdict
    out["route_arrived_at_s"] = collected.get("arrived_at")
    follow = collected.get("follow", [])
    out["follower_samples"] = len(follow)
    if follow:
        # Self-reported, and labelled as such.
        out["follower_final_cross_track_m"] = follow[-1][1]
        # The whole trace, carried out for offline diagnosis via
        # --dump-follow.  It is NOT part of any verdict -- the referee in
        # tritium_lib.control.route_trace takes ground-truth poses only.
        # Columns: t, cross_track, dist_to_goal, cmd_yaw, target_idx,
        #          measured_yaw, left_scale, right_scale.
        out["follow_trace"] = follow
    return out


def plan_route(args, obstacles) -> list[tuple[float, float]]:
    """Plan around ``obstacles`` with the planner lib already ships.

    No planner is written here.  ``tritium_lib.planning`` has a tested
    8-connected A* and a costmap builder, and ``scene_costmap`` already
    projects 3-D boxes onto it through a body band -- the same boxes this
    script authors into the stage.  This function is only the glue.
    """
    from tritium_lib.planning.astar import plan_route as _plan
    from tritium_lib.planning.scene_costmap import costmap_from_scene

    start = (0.0, 0.0)  # the scene seats the body at the origin
    goals = parse_points(args.plan_to)
    if not goals:
        raise SystemExit(f"--plan-to wants X,Y in meters, got {args.plan_to!r}")
    goal = goals[0]
    grid = costmap_from_scene(
        obstacles, resolution=args.plan_resolution,
        bounds=_route_bounds(start, goal, margin_m=4.0),
        # The obstacle list is hand-typed, not scraped off a stage, so the
        # library's terrain cap has nothing to protect against here -- and
        # applied, it silently DROPS any box over 100 m across, handing back a
        # straight line through a wall the scene really authors.  None keeps
        # every box the operator asked for.  score_route uses the same value.
        max_footprint_m=None)
    # clearance_m belongs to the PLANNER, not the costmap: it is the body's
    # standoff from a wall, which is a property of the Go2 (~0.3 m half-width),
    # not of the wall.
    result = _plan(grid, start, goal, clearance_m=args.plan_clearance)
    if not result.success:
        raise SystemExit(f"no route from {start} to {goal}: {result.reason}")
    if result.clearance_relaxed:
        # Say so rather than walking a route that grazes a wall while the log
        # claims the standoff was honoured.
        print("[gait] WARNING: clearance could not be met; route relaxed to 0",
              file=sys.stderr)
    print(f"[gait] planned {len(result.path)} waypoints, cost {result.cost:.2f}, "
          f"{result.expansions} expansions, strategy {result.strategy}")
    return [(float(x), float(y)) for x, y in result.path]


def _route_bounds(start, goal, margin_m: float):
    """Crop the costmap to the start->goal corridor.

    Sizing the grid to the whole stage is what produced a 39.8M-cell map on
    the city scene; the corridor is the only part a route can use.
    """
    return (min(start[0], goal[0]) - margin_m, min(start[1], goal[1]) - margin_m,
            max(start[0], goal[0]) + margin_m, max(start[1], goal[1]) + margin_m)


def run_trial(br: "Bridge", gait: dict, args, stabilize: bool,
              capture: str | None = None,
              route: dict | None = None,
              obstacles: "list | None" = None) -> dict:
    """One run from a freshly rebuilt scene.

    The scene is torn down and rebuilt per trial rather than reusing the
    stage.  A Go2 left holding its last joint targets topples within seconds,
    so a second trial starting from wherever the first one ended would be
    scoring the recovery of a fallen robot, not the gait.

    ``capture`` makes this run EVIDENCE, not a data point: the mid-window
    viewport render stalls the control callback (see the comment at the
    capture block), so the result comes back stamped ``capture_perturbed``
    with every verdict wrapped ``UNSCORED[...]`` and can never enter a rate.
    Pass ``None`` for a scored trial.
    """
    br.sim_control("stop")
    time.sleep(1.0)
    _ok("scene", br.execute(build_scene_code(args.stiffness, args.damping,
                                             obstacles=obstacles)))
    if route:
        # Draw the ROUTE the body was told to walk, so a captured frame shows
        # the plan and the body in the same picture.  A frame of a robot on an
        # empty slab cannot distinguish following a route from wandering.
        # Reuses nav_bridge's drawing rather than a second polyline authority.
        from isaac_sim_addon.clients.nav_bridge import draw_route_src
        _ok("draw", br.execute(draw_route_src(
            [(p[0], p[1]) for p in route["waypoints"]])))
    br.sim_control("play")
    time.sleep(2.0)

    info = _ok("driver", br.execute(build_driver_code(
        gait, args.seconds, args.stiffness, args.damping, args.sample_every,
        stabilize=stabilize, kp=args.kp, kd=args.kd,
        disturb=disturb_spec(args), push_window=args.disturb_window,
        twist=twist_spec(args), route=route, live=live_spec(args))))
    if stabilize and len(info.get("legs_wired", [])) != 4:
        # Silently trimming two legs would look like a weak controller rather
        # than a wiring bug, so it is fatal instead.
        raise RuntimeError(
            f"attitude trim wired to {info.get('legs_wired')} — expected all 4 "
            f"legs; solver DOF names are {info.get('joint_names')}")

    # Capture from INSIDE the scored window.  An end-of-run frame is evidence
    # about a moment nobody measured: the driver stops commanding at DURATION
    # and the robot topples immediately after, so a frame taken then shows a
    # collapse the score never saw.
    #
    # The default is mid-window, but it must be movable, and the 3 N-s A/B is
    # why: the open-loop arm reaches 179.97 degrees, yet its mid-window frame
    # at t=4s shows the body still upright, because that arm's tumble
    # develops later.  A fixed 50% capture therefore produced two frames that
    # looked alike for two arms whose traces could not be more different.
    # Anywhere in [0, seconds] is still a measured moment; past it is not.
    #
    # BUT a capture is not a neutral observer.  /sim/capture (and
    # /camera/look_at) render synchronously on the kit MAIN thread -- the
    # thread that pumps the app-update stream _on_step rides -- so for the
    # duration of the render the gait targets freeze mid-stride, the
    # stabiliser is blind, any open push window straddles the gap, and on
    # resume the sim-time jump snaps the gait phase discontinuously.
    # Measured: same command and push, 0/8 upright capture-on vs 8/8
    # capture-free (the retired "~5 N*s inverts it" ceiling was this
    # artifact).  So a capture run is EVIDENCE, never a data point: the
    # result is stamped capture_perturbed, its verdicts are wrapped
    # UNSCORED[...], and split_scoreable() bars it from every rate.  main()
    # only ever passes `capture` on a dedicated evidence run; scored trials
    # take the `else` branch below, whose bridge traffic is identical
    # whether or not --capture was given.
    mid_shot = None
    if capture:
        at = args.seconds * 0.5 if args.capture_at is None else args.capture_at
        at = max(0.0, min(float(at), args.seconds))
        mid_deadline = time.time() + at
        br.camera_look_at(ROBOT_PATH, distance=args.capture_distance,
                          elevation=args.capture_elevation)
        time.sleep(max(0.0, mid_deadline - time.time()))
        mid_shot = br.capture()
        # Sleep out the REMAINDER of the window, measured from the actual
        # capture time.  The old `seconds * 0.5 + 2.0` assumed a mid-window
        # capture, so an early --capture-at made the total wait shorter than
        # the window itself and CLEANUP tore the driver down mid-run.
        time.sleep(max(0.0, (args.seconds - at) + 2.0))
    else:
        # No mid-window bridge traffic of any kind: one uninterrupted wait.
        time.sleep(args.seconds + 2.0)

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
    # The stall tripwire, carried out of the kit for every run: the largest
    # sim-time hole between consecutive control callbacks inside the scored
    # window.  A healthy run sits near the ~17 ms frame period; a mid-window
    # render shows up here as a hole orders of magnitude wider.
    score["max_cb_gap_s"] = collected.get("max_cb_gap_s")
    # Carried out of the run so a live session can be graded against what was
    # actually COMMANDED, not just how far the body got.  Distance alone
    # cannot tell a body that obeyed a live operator from one running a baked
    # twist -- both walk.  The command column is the only thing that can.
    score["live_trace"] = collected.get("live", [])
    score["live_sock_err"] = collected.get("live_sock_err")
    # Ground-truth pose on the SAME clock as the command column, so the two
    # can be sliced at the moment the link went quiet.
    score["pose_trace"] = collected.get("trace", [])
    if route:
        score.update(score_route(collected, route, obstacles or []))

    # Quarantine -- LAST, so every verdict any scorer added is already in the
    # dict.  A run that captured had its control callback stalled by the
    # render, so it is stamped and its verdicts wrapped: even a consumer that
    # ignores split_scoreable() and string-matches "WALKED"/"RECOVERED" can
    # never count this trial.  The raw measurements stay -- they are
    # diagnostics (they are how the stall was caught) -- but no verdict from
    # this run reads as a clean one.
    score["capture_perturbed"] = capture is not None
    if capture is not None:
        score["verdict"] = f"UNSCORED[{score['verdict']}]"
        if score.get("route_verdict"):
            score["route_verdict"] = f"UNSCORED[{score['route_verdict']}]"
        if score.get("disturbance"):
            score["disturbance"]["verdict"] = (
                f"UNSCORED[{score['disturbance']['verdict']}]")
        if "disturb_ok" in score:
            # A perturbed trial can never serve as disturbance evidence.
            score["disturb_ok"] = False

    if capture and mid_shot:
        data = mid_shot.get("result", {})
        b64 = data.get("image_base64") or data.get("image") or data.get("data")
        if b64:
            with open(capture, "wb") as fh:
                fh.write(base64.b64decode(b64))
            print(f"[gait] wrote {capture} (t={at:.2f}s of {args.seconds:.2f}s)")
    return score


if __name__ == "__main__":
    raise SystemExit(main())
