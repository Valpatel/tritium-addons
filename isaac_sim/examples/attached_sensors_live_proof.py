#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Live proof: attached-mode sensors carry a NEWTON-STEPPED body's motion.

Runs ON THE RENDER HOST with plain ``python3`` (stdlib only — the render
host's system python has neither numpy nor PIL, and a proof harness that
needs an env nobody has is a proof nobody runs).  It drives the dedicated
Newton kit through the MCP bridge and leaves every byte on disk for the
off-box grader (``attached_sensors_grade.py``), which decodes depth16 with
the CANONICAL ``tritium_lib.perception.depth_codec``.

Scene (all under ``/World/Tritium``; box-slab ground, never a zero-thickness
plane — qhull cannot hull one and the latched ``_initializing`` flag then
silently kills integration for the whole process):

  * a blue WALL at an exactly known camera distance (the depth yardstick);
  * a RED dynamic cube authored IN THE AIR — Newton must land it at
    half-height, the tick-9 integration proof, and its fall is sampled
    against the sim's own pose;
  * a YELLOW bouncy ball — sustained motion, so "the frame follows the sim"
    is provable long after camera warm-up;
  * a CYAN cube authored KINEMATIC — released mid-run to answer whether a
    pre-registered body can be dropped on command on this build.

Hard-won constraints this harness encodes (verified live, tick attached-01):

  * EVERYTHING dynamic is authored BEFORE the timeline plays — Newton does
    not register prims added mid-play (probed: fabric never takes them);
  * a dynamic body's dimensions live in geometry attrs (``Cube.size``,
    ``Sphere.radius``), NEVER in an xformOp scale — physics takes the
    transform over and the authored scale is dropped (probed: a 0.6 m cube
    authored via scale simulated and rendered as its unscaled 2 m self);
  * poses are read through ``RigidPrim.get_world_poses()`` (Fabric), because
    the USD/usdrt transforms stay frozen at their authored values while
    physics runs — and on this build the view hands back CUDA torch tensors,
    so conversion goes through ``float()``, never ``.numpy()``;
  * annotators produce nothing until the timeline plays, so warm-up happens
    with physics already running.

Safety rails: refuses port 8211 unconditionally (a FOREIGN PhysX instance),
never stops the timeline, touches nothing outside ``/World/Tritium``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import sys
import time
import urllib.request

# ---- the one place the scene geometry lives (grader reads geometry.json) -- #
GEOM = {
    "cam_eye": [-4.0, 0.0, 1.0],
    "cam_target": [6.0, 0.0, 1.0],
    "hfov_deg": 70.0,
    "width": 640,
    "height": 480,
    "baseline_m": 0.30,
    # Wall: half-extents (0.1, 3.0, 1.5) at (6.0, 0, 1.5) -> front face x=5.9,
    # camera planar depth 9.9 m; spans y +-3, z 0..3.
    "wall_center": [6.0, 0.0, 1.5],
    "wall_half": [0.1, 3.0, 1.5],
    "wall_front_x": 5.9,
    "wall_planar_depth_m": 9.9,
    # RED body: 0.6 m cube (Cube.size, NOT scale) dropped from the air;
    # Newton must land it at z = 0.3 — half-height on the slab.
    "body_half": 0.3,
    "body_start": [2.0, 0.0, 6.0],
    "body_rest_z": 0.3,
    # YELLOW ball: bouncy sphere for sustained motion.
    "ball_radius": 0.35,
    "ball_start": [0.5, -1.5, 14.0],
    # CYAN kinematic cube: pre-registered aloft, released mid-run.
    "kin_half": 0.25,
    "kin_start": [3.5, 1.2, 2.5],
    # LiDAR: off the camera axis so its prim can never shadow the depth
    # samples; 2D sweep plane z=0.5 crosses body (z 0..0.6) and wall (z 0..3).
    "lidar_pos": [0.0, 2.0, 0.5],
    "lidar_range_max": 30.0,
    # Stereo converges on cam_target (10 m out): disparity(Z) = fx*B*(1/Z-1/10).
    "converge_dist_m": 10.0,
}

BODY = "/World/Tritium/body"
BALL = "/World/Tritium/ball"
KIN = "/World/Tritium/kin_cube"


class Bridge:
    """Minimal stdlib client for the MCP bridge (HTTP/1.1 + JSON)."""

    def __init__(self, host: str, port: int, timeout: float):
        if port == 8211:
            raise SystemExit(
                "REFUSED: 127.0.0.1:8211 is a FOREIGN PhysX instance "
                "belonging to another project — never touch it.")
        self.host, self.port, self.timeout = host, port, timeout

    def request(self, path: str, body: dict | None = None,
                method: str = "POST") -> dict:
        conn = http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout)
        try:
            conn.request(method, path, body=json.dumps(body or {}),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            return json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def execute(self, code: str) -> dict:
        return self.request("/execute", {"code": code})

    def exec_json(self, code: str) -> dict:
        """Run code that prints ONE json line; return it parsed.  Anything
        else — traceback, empty stdout, non-json — raises with the full
        response, because a proof that swallows its errors proves nothing."""
        resp = self.execute(code)
        result = resp.get("result", resp) if isinstance(resp, dict) else {}
        stdout = (result or {}).get("stdout", "") or ""
        for line in reversed([l for l in stdout.strip().splitlines() if l.strip()]):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"exec_json: no json in response: {json.dumps(resp)[:2000]}")


# --------------------------------------------------------------------------- #
# In-kit code snippets.
# --------------------------------------------------------------------------- #

SCENE_CODE = """
import json
import builtins
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf
builtins._tritium_pose_views = {{}}   # new scene invalidates cached views
st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(st, "/World/Tritium")

def static_box(path, half, center, color):
    # Statics may use xformOp scale (physics never takes their transform).
    c = UsdGeom.Cube.Define(st, path)
    c.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(c.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*half))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c.GetPrim()

def dynamic_cube(path, size, center, color, mass, kinematic=False):
    # Dynamics: dimensions in Cube.size, NEVER scale — physics drops the
    # authored scale the moment it owns the transform (probed live).
    c = UsdGeom.Cube.Define(st, path)
    c.GetSizeAttr().Set(float(size))
    UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(float(mass))
    return c.GetPrim()

ground = static_box("/World/Tritium/ground_slab", (25.0, 25.0, 0.5),
                    (0.0, 0.0, -0.5), (0.35, 0.35, 0.38))
mp = "/World/Tritium/ground_mat"
UsdShade.Material.Define(st, mp)
mprim = st.GetPrimAtPath(mp)
UsdPhysics.MaterialAPI.Apply(mprim)
mat = UsdPhysics.MaterialAPI(mprim)
mat.CreateStaticFrictionAttr(1.0)
mat.CreateDynamicFrictionAttr(1.0)
mat.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(ground).Bind(
    UsdShade.Material(mprim), UsdShade.Tokens.weakerThanDescendants, "physics")

static_box("/World/Tritium/wall", {wall_half}, {wall_center}, (0.15, 0.25, 0.85))
dynamic_cube("{body}", {body_size}, {body_start}, (0.9, 0.08, 0.08), 5.0)
dynamic_cube("{kin}", {kin_size}, {kin_start}, (0.1, 0.85, 0.9), 3.0,
             kinematic=True)

ball = UsdGeom.Sphere.Define(st, "{ball}")
ball.GetRadiusAttr().Set({ball_radius})
UsdGeom.Xformable(ball.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*{ball_start}))
ball.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.85, 0.05)])
UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
UsdPhysics.MassAPI.Apply(ball.GetPrim()).CreateMassAttr(1.0)
bmp = "/World/Tritium/ball_mat"
UsdShade.Material.Define(st, bmp)
bmprim = st.GetPrimAtPath(bmp)
UsdPhysics.MaterialAPI.Apply(bmprim)
bmat = UsdPhysics.MaterialAPI(bmprim)
bmat.CreateStaticFrictionAttr(0.6)
bmat.CreateDynamicFrictionAttr(0.6)
bmat.CreateRestitutionAttr(0.85)
UsdShade.MaterialBindingAPI.Apply(ball.GetPrim()).Bind(
    UsdShade.Material(bmprim), UsdShade.Tokens.weakerThanDescendants, "physics")

if not st.GetPrimAtPath("/World/Tritium/dome").IsValid():
    d = UsdLux.DomeLight.Define(st, "/World/Tritium/dome")
    d.GetIntensityAttr().Set(1000.0)
print(json.dumps({{"scene": True}}))
"""

INSTALL_CODE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location(
    "tritium_attached_sensor_server", {module_path!r})
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
out = m.install(cam_port={cam_port}, lidar_port={lidar_port},
                width={width}, height={height}, hfov_deg={hfov},
                cam_pos={cam_eye}, cam_target={cam_target},
                stereo_baseline={baseline}, lidar_pos={lidar_pos},
                lidar_range_max={lidar_range_max}, play_timeline=False)
print(json.dumps(out))
"""

MODULE_STATUS_CODE = """
import json, sys
m = sys.modules.get("tritium_attached_sensor_server")
print(json.dumps(m.status() if m else {"installed": False, "missing": True}))
"""

# Fabric-aware pose read: RigidPrim views (cached in builtins — /execute gets
# a fresh namespace every call), floats via float() (CUDA tensors, no numpy).
POSES_CODE = """
import json, time
import builtins
views = getattr(builtins, "_tritium_pose_views", None)
if views is None:
    views = {}
    builtins._tritium_pose_views = views
out = {"t": time.time()}
for path in %s:
    try:
        if path not in views:
            from isaacsim.core.prims import RigidPrim
            views[path] = RigidPrim(path)
        pos, _rot = views[path].get_world_poses()
        p = pos[0]
        out[path.rsplit("/", 1)[-1]] = [round(float(p[i]), 4) for i in range(3)]
    except Exception as e:
        out[path.rsplit("/", 1)[-1]] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
""" % json.dumps([BODY, BALL, KIN])

NEWTON_PROBE_CODE = """
import json
out = {}
try:
    from isaacsim.physics.newton import acquire_stage
    ns = acquire_stage()
    out["newton_stage"] = ns is not None
    out["steps"] = int(getattr(ns, "simulation_step_count", -1))
    out["initialized"] = bool(getattr(ns, "initialized", False))
except Exception as e:
    out["newton_error"] = f"{type(e).__name__}: {e}"
import omni.timeline
out["timeline_playing"] = bool(
    omni.timeline.get_timeline_interface().is_playing())
print(json.dumps(out))
"""

PLAY_CODE = """
import json
import omni.timeline
omni.timeline.get_timeline_interface().play()
print(json.dumps({"playing": True}))
"""

RELEASE_KIN_CODE = """
import json
import omni.usd
from pxr import UsdPhysics
st = omni.usd.get_context().get_stage()
rb = UsdPhysics.RigidBodyAPI(st.GetPrimAtPath("%s"))
rb.GetKinematicEnabledAttr().Set(False)
print(json.dumps({"released": True}))
""" % KIN

VISIBILITY_CODE = """
import json
import omni.usd
from pxr import UsdGeom
st = omni.usd.get_context().get_stage()
img = UsdGeom.Imageable(st.GetPrimAtPath("%s"))
img.%s()
print(json.dumps({"visibility": "%s"}))
"""


# --------------------------------------------------------------------------- #
# HTTP capture helpers (the sensors' own wire, loopback on the render host).
# --------------------------------------------------------------------------- #

def fetch(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch_json(url: str, timeout: float = 10.0) -> dict:
    return json.loads(fetch(url, timeout).decode("utf-8"))


def capture_phase(out_dir: str, phase: str, cam_base: str, lidar_base: str,
                  results: dict) -> None:
    """Save every channel for one phase.  A missing channel is RECORDED as
    missing, not skipped silently."""
    got: dict = {}
    for name, url in (
        ("main", f"{cam_base}/snapshot?channel=main"),
        ("right", f"{cam_base}/snapshot?channel=right"),
        ("depth_view", f"{cam_base}/snapshot?channel=depth"),
    ):
        try:
            blob = fetch(url)
            with open(os.path.join(out_dir, f"{phase}_{name}.jpg"), "wb") as f:
                f.write(blob)
            got[name] = len(blob)
        except Exception as exc:  # noqa: BLE001
            got[name] = f"FAILED: {exc}"
    try:
        blob = fetch(f"{cam_base}/snapshot?channel=depth16")
        with open(os.path.join(out_dir, f"{phase}_depth16.png"), "wb") as f:
            f.write(blob)
        got["depth16"] = len(blob)
        got["depth16_png_magic"] = blob[:8].hex()
    except Exception as exc:  # noqa: BLE001
        got["depth16"] = f"FAILED: {exc}"
    try:
        scan = fetch_json(f"{lidar_base}/scan")
        with open(os.path.join(out_dir, f"{phase}_scan.json"), "w") as f:
            json.dump(scan, f)
        got["scan"] = scan_summary(scan)
    except Exception as exc:  # noqa: BLE001
        got["scan"] = f"FAILED: {exc}"
    results.setdefault("phases", {})[phase] = got


def scan_summary(scan: dict) -> dict:
    ranges = scan.get("ranges", [])
    if not ranges:
        return {"empty": True}
    rmin = min(ranges)
    idx = ranges.index(rmin)
    az = math.degrees(scan["angle_min"] + idx * scan["angle_increment"])
    return {
        "beams": len(ranges), "min_range_m": round(rmin, 3),
        "min_bearing_deg": round(az, 1),
        "never_returned": scan.get("never_returned"),
        "returns_below_max": sum(1 for r in ranges
                                 if r < scan.get("range_max", 1e9) - 1e-6),
    }


def poll(fn, ok, timeout_s: float, interval: float = 1.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            last = fn()
            if ok(last):
                return last, True
        except Exception as exc:  # noqa: BLE001
            last = {"error": str(exc)}
        time.sleep(interval)
    return last, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8212,
                    help="MCP bridge port of the DEDICATED kit (8211 refused)")
    ap.add_argument("--cam-port", type=int, default=8130)
    ap.add_argument("--lidar-port", type=int, default=8131)
    ap.add_argument("--out", required=True, help="evidence directory")
    ap.add_argument("--connector", default="", help="attached_sensor_server.py "
                    "path (default: sibling of this script's addon tree)")
    ap.add_argument("--skip-scene", action="store_true",
                    help="reuse the kit's current scene (no new_scene)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    connector = args.connector or os.path.join(
        here, "..", "isaac_sim_addon", "connectors", "attached_sensor_server.py")
    connector = os.path.abspath(connector)
    if not os.path.isfile(connector):
        raise SystemExit(f"connector not found: {connector}")

    os.makedirs(args.out, exist_ok=True)
    results: dict = {"t0": time.time(), "geometry": GEOM,
                     "connector": connector}
    with open(os.path.join(args.out, "geometry.json"), "w") as f:
        json.dump(GEOM, f, indent=2)

    br = Bridge("127.0.0.1", args.port, timeout=300.0)
    cam_base = f"http://127.0.0.1:{args.cam_port}"
    lidar_base = f"http://127.0.0.1:{args.lidar_port}"

    def save(name: str, obj) -> None:
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=2)

    # ---- 0. preflight ---------------------------------------------------- #
    results["health"] = br.request("/health", {}, method="GET")
    probe0 = br.exec_json(NEWTON_PROBE_CODE)
    results["newton_before"] = probe0
    print(f"[0] kit up, newton probe: {probe0}")

    # ---- 1. scene (EVERYTHING dynamic pre-authored — Newton registers
    #         bodies at play, never after) ------------------------------- #
    if not args.skip_scene:
        br.request("/scene/new", {})
        time.sleep(1.0)
        scene = br.exec_json(SCENE_CODE.format(
            wall_half=tuple(GEOM["wall_half"]),
            wall_center=tuple(GEOM["wall_center"]),
            body=BODY, body_size=GEOM["body_half"] * 2,
            body_start=tuple(GEOM["body_start"]),
            kin=KIN, kin_size=GEOM["kin_half"] * 2,
            kin_start=tuple(GEOM["kin_start"]),
            ball=BALL, ball_radius=GEOM["ball_radius"],
            ball_start=tuple(GEOM["ball_start"])))
        results["scene"] = scene
        print(f"[1] scene: {scene}")

    # ---- 2. install the attached sensors, assert provenance -------------- #
    install = br.exec_json(INSTALL_CODE.format(
        module_path=connector, cam_port=args.cam_port,
        lidar_port=args.lidar_port, width=GEOM["width"],
        height=GEOM["height"], hfov=GEOM["hfov_deg"],
        cam_eye=tuple(GEOM["cam_eye"]), cam_target=tuple(GEOM["cam_target"]),
        baseline=GEOM["baseline_m"], lidar_pos=tuple(GEOM["lidar_pos"]),
        lidar_range_max=GEOM["lidar_range_max"]))
    results["install"] = install
    save("install.json", install)
    expect_dir = os.path.dirname(connector)
    for k, v in install.get("provenance", {}).items():
        if os.path.dirname(v) != expect_dir:
            raise SystemExit(
                f"STALE MODULE: {k} loaded from {v}, expected {expect_dir} — "
                "the kit cached an old module; restart the kit.")
    if install.get("lidar_error"):
        print(f"[2] WARNING lidar failed to stand up: {install['lidar_error']}")
    print(f"[2] installed: cam:{args.cam_port} lidar:{args.lidar_port} "
          f"channels={install.get('channels')}")

    # ---- 3. play, then sample the fall while cameras warm ----------------- #
    br.exec_json(PLAY_CODE)
    t_play = time.time()
    fall: list = []
    for i in range(40):
        try:
            p = br.exec_json(POSES_CODE)
            p["t_rel"] = round(p.pop("t") - t_play, 3)
            fall.append(p)
        except Exception as exc:  # noqa: BLE001
            fall.append({"error": str(exc)})
        if i % 2 == 1:
            try:
                blob = fetch(f"{cam_base}/snapshot?channel=main", timeout=2)
                with open(os.path.join(args.out, f"fall_{i:02d}.jpg"), "wb") as f:
                    f.write(blob)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.05)
    save("fall_log.json", fall)
    body_z = [p["body"][2] for p in fall if isinstance(p.get("body"), list)]
    ball_z = [p["ball"][2] for p in fall if isinstance(p.get("ball"), list)]
    results["fall_body_z_first_last"] = [body_z[0], body_z[-1]] if body_z else []
    results["fall_body_distinct_z"] = len({round(z, 3) for z in body_z})
    results["fall_ball_distinct_z"] = len({round(z, 3) for z in ball_z})
    fall_jpgs = [n for n in os.listdir(args.out) if n.startswith("fall_")]
    print(f"[3] fall: {len(body_z)} poses, body distinct z="
          f"{results['fall_body_distinct_z']} "
          f"(z {body_z[0] if body_z else '?'} -> {body_z[-1] if body_z else '?'}), "
          f"ball distinct z={results['fall_ball_distinct_z']}, "
          f"{len(fall_jpgs)} fall frames on camera")

    # ---- 4. REST: Newton must have landed the RED body -------------------- #
    def read_poses() -> dict:
        return br.exec_json(POSES_CODE)

    pose_rest, settled = poll(
        read_poses,
        lambda p: isinstance(p.get("body"), list)
        and abs(p["body"][2] - GEOM["body_rest_z"]) < 0.05,
        timeout_s=25)
    results["pose_rest"] = pose_rest
    results["settled_at_half_height"] = settled
    probe1 = br.exec_json(NEWTON_PROBE_CODE)
    results["newton_after"] = probe1
    results["newton_stepped"] = (probe1.get("steps", -1)
                                 > probe0.get("steps", -1) >= -1)
    if not settled:
        print(f"[4] PHYSICS PROBLEM: red body did not settle at z~"
              f"{GEOM['body_rest_z']}: {pose_rest}. If z is unchanged from "
              "launch, integration is dead (latched-_initializing failure "
              "mode); if z<0 it fell through the slab.")
    else:
        print(f"[4] REST body={pose_rest['body']} ball={pose_rest.get('ball')} "
              f"(newton steps {probe0.get('steps')} -> {probe1.get('steps')})")

    # camera + lidar warm-up (needs the playing timeline).
    status, cam_warm = poll(
        lambda: fetch_json(f"{cam_base}/status"),
        lambda s: s.get("frames", 0) >= 3, timeout_s=90)
    results["cam_warm"] = cam_warm
    results["cam_status"] = status
    lstat, lidar_warm = poll(
        lambda: fetch_json(f"{lidar_base}/status"),
        lambda s: s.get("scans", 0) >= 3, timeout_s=120)
    results["lidar_warm"] = lidar_warm
    results["lidar_status"] = lstat
    print(f"[4] cam warm={cam_warm} frames={status.get('frames')} | "
          f"lidar warm={lidar_warm} scans={lstat.get('scans')}")
    if not cam_warm:
        save("results_live.json", results)
        print("RESULT: LIVE-SIDE FAILED (cameras never produced a frame)")
        return 1
    with open(os.path.join(args.out, "intrinsics.json"), "wb") as f:
        f.write(fetch(f"{cam_base}/intrinsics"))
    capture_phase(args.out, "rest", cam_base, lidar_base, results)

    # ---- 5. MOVING pair: the bouncing ball, two instants ------------------ #
    results["pose_moving_1"] = read_poses()
    capture_phase(args.out, "moving_1", cam_base, lidar_base, results)
    time.sleep(1.5)
    results["pose_moving_2"] = read_poses()
    capture_phase(args.out, "moving_2", cam_base, lidar_base, results)
    print(f"[5] moving pair: ball {results['pose_moving_1'].get('ball')} -> "
          f"{results['pose_moving_2'].get('ball')}")

    # ---- 6. RELEASED: flip the pre-registered kinematic cube dynamic ------ #
    br.exec_json(RELEASE_KIN_CODE)
    time.sleep(2.0)
    pose_rel = read_poses()
    results["pose_released"] = pose_rel
    kin = pose_rel.get("kin_cube")
    kin_fell = (isinstance(kin, list)
                and abs(kin[2] - GEOM["kin_half"]) < 0.06)
    results["kinematic_release_worked"] = kin_fell
    if kin_fell:
        capture_phase(args.out, "released", cam_base, lidar_base, results)
        print(f"[6] kinematic release WORKED: cyan cube fell to {kin}")
    else:
        print(f"[6] kinematic release DID NOT drop the cube (pose {kin}) — "
              "recorded honestly; on this build a registered body cannot be "
              "released mid-run this way.")

    # ---- 7. HIDDEN / SHOWN: the stale-server killer ----------------------- #
    br.exec_json(VISIBILITY_CODE % (BODY, "MakeInvisible", "invisible"))
    time.sleep(1.0)
    results["pose_hidden"] = read_poses()
    capture_phase(args.out, "hidden", cam_base, lidar_base, results)
    br.exec_json(VISIBILITY_CODE % (BODY, "MakeVisible", "visible"))
    time.sleep(1.0)
    results["pose_shown"] = read_poses()
    capture_phase(args.out, "shown", cam_base, lidar_base, results)
    print("[7] hidden/shown captured")

    # ---- 8. liveness counters + in-kit ledger ----------------------------- #
    s1 = fetch_json(f"{cam_base}/status")
    time.sleep(2.0)
    s2 = fetch_json(f"{cam_base}/status")
    results["frames_advancing"] = s2.get("frames", 0) > s1.get("frames", 0)
    results["cam_status_final"] = s2
    results["module_status_final"] = br.exec_json(MODULE_STATUS_CODE)
    save("results_live.json", results)

    # ---- honest verdict --------------------------------------------------- #
    print("\n=== LIVE-SIDE VERDICT (pixel grading happens off-box) ===")
    print(f"newton stepped:        {results.get('newton_stepped')}")
    print(f"body settled (0.3 m):  {settled}")
    print(f"body fall distinct z:  {results.get('fall_body_distinct_z')}")
    print(f"ball fall distinct z:  {results.get('fall_ball_distinct_z')}")
    print(f"kinematic release:     {results.get('kinematic_release_worked')}")
    print(f"frames advancing:      {results.get('frames_advancing')}")
    print(f"lidar warm:            {results.get('lidar_warm')}")
    for phase, got in results.get("phases", {}).items():
        print(f"  {phase}: " + ", ".join(f"{k}={v}" for k, v in got.items()))
    ok = (bool(results.get("newton_stepped")) and settled
          and results.get("frames_advancing")
          and results.get("fall_body_distinct_z", 0) > 3)
    print("RESULT:", "LIVE-SIDE OK" if ok else "LIVE-SIDE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
