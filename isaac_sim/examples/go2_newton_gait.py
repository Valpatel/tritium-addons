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
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath("/World/GroundPlane").IsValid():
    GroundPlane(prim_path="/World/GroundPlane", z_position=0.0)

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
    return (SCENE_TEMPLATE
            .replace("__ROBOT_PATH__", ROBOT_PATH)
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

state = {"trace": [], "steps": 0, "t0": None, "err": None}

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
        view.set_dof_position_targets(tile(_sample_targets(elapsed)), idx)
        state["steps"] += 1
        if state["steps"] % SAMPLE_EVERY == 0:
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
          "dof": int(len(joint_names))}
"""


def build_driver_code(gait: dict, duration: float, stiffness: float,
                      damping: float, sample_every: int) -> str:
    code = DRIVER_CODE
    code = code.replace("GAIT_JSON", repr(json.dumps(gait)))
    code = code.replace("ARTICULATION_PATH", ARTICULATION_PATH)
    code = code.replace("STIFFNESS", repr(stiffness))
    code = code.replace("DAMPING", repr(damping))
    code = code.replace("DURATION", repr(duration))
    code = code.replace("SAMPLE_EVERY", str(sample_every))
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
          "joint_names": g["joint_names"]}
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

    Each row is [t, x, y, z, qx, qy, qz].  Displacement is measured in the
    ground plane; height is the mean z; "upright" means the body never
    dropped below 60% of its starting height (a collapsed dog reads as a
    large z drop, and a dog that merely vibrates reads as ~0 displacement).
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
        "verdict": "COLLAPSED" if collapsed else ("MOVED" if dist > 0.10 else "STATIONARY"),
    }


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
    ap.add_argument("--capture", help="write a viewport PNG here when done")
    ap.add_argument("--keep-running", action="store_true",
                    help="leave the step callback installed after scoring")
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

    # Build the scene stopped, then play so physics initializes with the
    # robot present.
    br.sim_control("stop")
    time.sleep(1.0)
    print(f"[gait] scene: {_ok('scene', br.execute(build_scene_code(args.stiffness, args.damping)))}")
    br.sim_control("play")
    time.sleep(2.0)

    info = _ok("driver", br.execute(build_driver_code(
        gait, args.seconds, args.stiffness, args.damping, args.sample_every)))
    print(f"[gait] driving {info['dof']} DOF: {info['joint_names']}")

    time.sleep(args.seconds + 2.0)

    collected = _ok("collect", br.execute(COLLECT_CODE))
    if collected.get("err"):
        print(f"[gait] callback error: {collected['err']}", file=sys.stderr)
    if not args.keep_running:
        _ok("cleanup", br.execute(CLEANUP_CODE))

    score = score_trace(collected.get("trace", []))
    score["physics_steps"] = collected.get("steps")
    print(json.dumps(score, indent=1))

    if args.capture:
        br.camera_look_at(ROBOT_PATH, distance=3.0)
        time.sleep(0.5)
        shot = br.capture()
        data = shot.get("result", {})
        b64 = data.get("image_base64") or data.get("image") or data.get("data")
        if b64:
            with open(args.capture, "wb") as fh:
                fh.write(base64.b64decode(b64))
            print(f"[gait] wrote {args.capture}")
        else:
            print(f"[gait] no image in capture response: {list(data)[:8]}",
                  file=sys.stderr)

    return 0 if score.get("verdict") == "MOVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
