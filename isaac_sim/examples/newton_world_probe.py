#!/usr/bin/env python3
"""Newton lane probe: is the Go2 articulation actually STEPPED by the solver?

Background (see NEWTON-GAIT-FINDINGS.md): the gait driver wrote joint targets
into a Newton articulation view 471 times over 7.8 s and the dog never moved.
The earlier diagnostic concluded the articulation was "present in the tensor
view but not in the solver's model".  This probe tests that, and in doing so
found the real cause one layer up.

**The finding: a headless kit does not pump its own update loop.**  With the
timeline playing, `omni.timeline`'s clock advanced only 0.033 s across 6 s of
wall clock -- roughly one frame per bridge call.  Physics time advanced only
when something poked the app.  So the old driver's "471 writes at ~60 Hz" were
real writes into a world that was barely stepping between them, and the sim
time it observed advancing was an artifact of its own polling.

The fix is to drive the loop explicitly: `omni.kit.app.get_app().update()` in
a loop steps physics deterministically (240 updates advanced sim time by 4.08 s
on this build).  That is the stepping primitive the gait lane was missing.

Second trap this probe encodes: **read poses through the physics tensor view,
never off USD.**  Physics writes to Fabric, not back to the USD attributes, so
`UsdGeom.Xformable.GetLocalTransformation()` returns the authored value forever
and makes a perfectly healthy falling body look frozen.

The test itself has a falsifiable answer.  With every joint drive zeroed and
the base held clear of the ground:

    does GRAVITY move the body?

A stepped articulation must fall.  An unstepped one holds its authored pose.
No gait, no targets, no trajectory -- so a NOT_STEPPED verdict cannot be blamed
on the controller.

Usage:
    python3 newton_world_probe.py [--port 8212] [--updates 240] [--drop 0.8]

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_PORT = 8212
ROBOT_PATH = "/World/Tritium/probe_go2"


class Bridge:
    """Minimal client for the Omniverse MCP bridge's /execute endpoint."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout: float = 600.0) -> None:
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def execute(self, code: str) -> dict:
        """Run code in the kit and return the snippet's own ``result`` value.

        The bridge wraps it twice: {"status", "result": {"stdout", "stderr",
        "return_value"}}.  Unwrap both so callers see just their report.
        """
        data = json.dumps({"code": code}).encode()
        req = urllib.request.Request(
            self.base + "/execute", data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())

        inner = payload.get("result") or {}
        if not isinstance(inner, dict):
            return {"error": f"unexpected bridge response: {payload!r}"[:400]}
        value = inner.get("return_value")
        if value is None:
            return {"error": "no return_value",
                    "stderr": str(inner.get("stderr", ""))[:800]}
        return value


PROBE_CODE = r"""
import numpy as np

report = {"stage": "start"}

try:
    import omni.usd, omni.kit.app, omni.timeline
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.simulation_manager import SimulationManager as SM
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from pxr import UsdGeom, Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    tl = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()

    # Author the scene while STOPPED -- the solver reads USD on play, and a
    # variant switch mid-play does not rebuild the Newton model.
    if tl.is_playing():
        tl.stop()
        for _ in range(4):
            app.update()

    if not stage.GetPrimAtPath("/World/GroundPlane").IsValid():
        GroundPlane(prim_path="/World/GroundPlane", z_position=0.0)

    robot = stage.GetPrimAtPath("__ROBOT_PATH__")
    if not robot.IsValid():
        usd_path = get_assets_root_path() + "/Isaac/Robots/Unitree/Go2/go2.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path="__ROBOT_PATH__")
        robot = stage.GetPrimAtPath("__ROBOT_PATH__")

    # "physx" is the ASSET's variant-set name (it selects the rigid-body/joint
    # payload).  It is NOT the engine -- the kit is what chooses Newton.
    vsets = robot.GetVariantSets()
    if "Physics" in vsets.GetNames():
        vs = vsets.GetVariantSet("Physics")
        if vs.GetVariantSelection() != "physx":
            vs.SetVariantSelection("physx")
        report["variant"] = vs.GetVariantSelection()

    xf = UsdGeom.Xformable(robot)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, __DROP__))

    # Zero every drive so nothing holds the legs up.  Gravity alone decides.
    drives = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if not prim.GetPath().pathString.startswith("__ROBOT_PATH__"):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drives += 1
    report["drives_zeroed"] = drives
    report["stage"] = "scene"

    # Play, then pump a few frames so the solver builds its model before we
    # ask for a view of it.
    tl.play()
    for _ in range(8):
        app.update()
    report["physics_scenes"] = len(SM.get_physics_scenes())

    view = SM.get_physics_sim_view().create_articulation_view("__ROBOT_PATH__/base")
    report["view_type"] = type(view).__name__

    def np_(t):
        # Getters hand back cuda:0 torch tensors on this build.
        try:
            return t.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(t)

    def sample():
        return (np_(view.get_dof_positions())[0].astype(float),
                np_(view.get_root_transforms())[0].astype(float))

    dof0, root0 = sample()
    t0 = float(SM.get_simulation_time())

    # THE STEP.  A headless kit will not do this for us.
    for _ in range(__UPDATES__):
        app.update()

    dof1, root1 = sample()
    t1 = float(SM.get_simulation_time())

    dof_delta = float(np.max(np.abs(dof1 - dof0)))
    root_drop = float(root0[2] - root1[2])
    moved = float(np.linalg.norm(root1[:3] - root0[:3]))

    report.update({
        "stage": "done",
        "dof_count": int(dof0.shape[0]),
        "sim_time": [round(t0, 4), round(t1, 4)],
        "sim_time_advanced": round(t1 - t0, 4),
        "dof_first6_t0": [round(v, 5) for v in dof0[:6]],
        "dof_first6_t1": [round(v, 5) for v in dof1[:6]],
        "root_t0": [round(v, 5) for v in root0[:3]],
        "root_t1": [round(v, 5) for v in root1[:3]],
        "max_dof_delta_rad": round(dof_delta, 5),
        "root_drop_m": round(root_drop, 5),
        "root_moved_m": round(moved, 5),
        # Loose thresholds on purpose: this separates "moved at all" from
        # "bit-identical", it does not measure precision.
        "verdict": "STEPPED" if (root_drop > 0.05 or dof_delta > 0.05)
                   else "NOT_STEPPED",
    })

except Exception as exc:  # noqa: BLE001 -- the failure mode IS the datum here
    import traceback
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()[-1500:]

result = report
"""


def build_probe_code(updates: int, drop: float) -> str:
    """Render the probe body.

    Plain .replace() rather than .format(): the body is Python source full of
    dict literals, and every brace would have to be doubled.
    """
    return (PROBE_CODE
            .replace("__ROBOT_PATH__", ROBOT_PATH)
            .replace("__UPDATES__", str(int(updates)))
            .replace("__DROP__", repr(float(drop))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--updates", type=int, default=240,
                    help="app.update() calls (240 ~ 4 s of sim time)")
    ap.add_argument("--drop", type=float, default=0.8,
                    help="base height in metres; must clear the ground")
    ap.add_argument("--print-code", action="store_true",
                    help="print the probe body and exit (no kit needed)")
    args = ap.parse_args()

    code = build_probe_code(args.updates, args.drop)
    if args.print_code:
        print(code)
        return 0

    report = Bridge(args.host, args.port).execute(code)
    print(json.dumps(report, indent=2))
    return 0 if report.get("verdict") == "STEPPED" else 1


if __name__ == "__main__":
    sys.exit(main())
