#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Verify a body-mounted camera against a LIVE Isaac Sim.

Every Tritium camera until now has been bolted to a wall: a constant lat/lon
and a constant heading.  A camera on a robot is a different object -- the
mount is rigid in the BODY's frame, so the lens physically swings through an
arc whenever the body turns, and "where is this camera looking" becomes a
function re-evaluated on every telemetry tick.
:class:`tritium_lib.geo.camera_mount.CameraMount` is that function.  This
script checks it against the only authority that can settle it.

**The check.**  A camera prim is parented UNDER the robot in the running
stage, so USD itself composes the rigid mount.  For each of several
ground-truth body poses we then ask USD where that camera actually ended up
(``ComputeLocalToWorldTransform``) and compare against what ``CameraMount``
predicted independently in plain Python.

Two different engines answer the same question, which is what makes this hard
to fool.  The classic implementation bug -- adding the mount offset in WORLD
axes instead of body axes -- agrees at heading 0 and diverges at every other
heading, so a sweep across headings is the thing that catches it.  A single
north-facing test would pass either way.

**Transport.**  The MCP bridge listens only on the GPU host's loopback and
resets forwarded connections, so requests are posted by a ``curl`` running on
that host over ssh.  The JSON goes in on stdin, never argv, to keep the
remote shell out of the quoting path.

**Capture note.**  Use the bridge's ``/sim/capture``, not
``capture_viewport_to_file``.  The latter is asynchronous and only the first
call per session lands; later ones silently write a black frame, which reads
as a broken camera rather than a race.

Usage:
    # numeric agreement only (fast):
    python mounted_camera_check.py
    # also render the robot's-eye view at four headings:
    python mounted_camera_check.py --capture-dir /tmp/mountcam
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from tritium_lib.geo.camera_mount import CameraMount
from tritium_lib.geo.isaac_frame import IsaacFrame, quat_to_yaw_deg

DEFAULT_HOST = "rtx4090"
DEFAULT_BRIDGE = "http://localhost:8211"
DEFAULT_BODY = "/World/Go2"

# A plausible nose mount on a Go2: forward of the chest, slightly to port,
# above the shoulder line, looking 10 deg down so the dog sees the ground in
# front of its own feet rather than the horizon.
DEFAULT_MOUNT = CameraMount(
    forward_m=0.30, left_m=0.10, up_m=0.12,
    pan_deg=0.0, tilt_deg=-10.0,
    hfov_deg=90.0, vfov_deg=60.0, range_m=30.0,
)

# Poses spanning all four quadrants plus two off-cardinal headings -- the
# off-cardinals are the ones a sign error survives.
DEFAULT_POSES = [
    (28.0, -22.0, 0.55, 0.0),
    (12.0, -5.0, 0.55, 45.0),
    (-8.0, 20.0, 0.55, 90.0),
    (30.0, 30.0, 0.55, 180.0),
    (-25.0, -15.0, 0.55, 270.0),
    (5.0, 5.0, 0.55, 315.0),
]

_SETUP = '''
from pxr import Usd, UsdGeom, Gf
import omni.usd
stage = omni.usd.get_context().get_stage()

# Remove any camera left by a previous run before recreating it.  Reusing the
# prim inherits whatever xform-op PRECISION that run authored, and USD raises
# when a later Add*Op asks for a different one -- so a rerun would fail purely
# because of its own history.
if stage.GetPrimAtPath(CAM_PATH):
    stage.RemovePrim(CAM_PATH)

camx = UsdGeom.Camera.Define(stage, CAM_PATH)
cam = camx.GetPrim()
camx.CreateFocalLengthAttr(18.0)
camx.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

xf = UsdGeom.Xformable(cam)
xf.ClearXformOpOrder()
xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*MOUNT_XYZ))
# USD cameras look down their own -Z.  Rotate the lens onto the body's +X
# (forward) axis, then apply the mount's pan and tilt.
xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Vec3d(90.0 + TILT_DEG, 0.0, -90.0 - PAN_DEG)
)
result = {"camera": str(cam.GetPath())}
'''

_PROBE = '''
from pxr import Usd, UsdGeom, Gf
import omni.usd
import omni.kit.viewport.utility as vpu
stage = omni.usd.get_context().get_stage()

body = stage.GetPrimAtPath(BODY_PATH)
bxf = UsdGeom.Xformable(body)
bxf.ClearXformOpOrder()
bxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*BODY_XYZ))
# Both halves of this matter.  The prim may already carry a quatd orient op,
# so requesting the default float precision RAISES; and setting a quatd op
# with a Gf.Quatf is silently DROPPED, which looks exactly like a frozen
# orientation rather than an error.
q = Gf.Rotation(Gf.Vec3d(0, 0, 1), BODY_YAW).GetQuat()
bxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q))

if SET_VIEWPORT:
    vpu.get_active_viewport().camera_path = CAM_PATH

def world_of(path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    t = m.ExtractTranslation()
    r = m.ExtractRotationQuat()
    im = r.GetImaginary()
    return {"t": [t[0], t[1], t[2]], "q": [r.GetReal(), im[0], im[1], im[2]]}

result = {"body": world_of(BODY_PATH), "cam": world_of(CAM_PATH)}
'''


class BridgeError(RuntimeError):
    """The live sim refused or failed a request."""


def post(host: str, bridge: str, endpoint: str, payload: dict) -> dict:
    proc = subprocess.run(
        ["ssh", host,
         f"curl -s -m 300 -X POST {bridge}{endpoint} "
         "-H 'Content-Type: application/json' --data-binary @-"],
        input=json.dumps(payload), capture_output=True, text=True, timeout=360,
    )
    if proc.returncode != 0:
        raise BridgeError(f"ssh to {host} failed: {proc.stderr.strip()}")
    try:
        reply = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"non-JSON reply: {proc.stdout[:200]!r}") from exc
    if reply.get("status") != "success":
        raise BridgeError(reply.get("error") or reply.get("traceback"))
    return reply["result"]


def _literals(**kwargs) -> str:
    """Inject parameters as literal assignments rather than string-formatting
    them into the snippet body -- keeps quoting out of the injection path."""
    return "".join(f"{k} = {v!r}\n" for k, v in kwargs.items())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST, help="ssh host running Isaac")
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE)
    ap.add_argument("--body", default=DEFAULT_BODY, help="prim path of the robot")
    ap.add_argument("--capture-dir", default=None,
                    help="also render the robot's-eye view into this directory")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args(argv)

    mount = DEFAULT_MOUNT
    cam_path = f"{args.body}/MountCam"
    frame = IsaacFrame(meters_per_unit=1.0, up_axis="Z")

    setup = _literals(
        CAM_PATH=cam_path,
        MOUNT_XYZ=mount.stage_offset(up_axis="Z"),
        TILT_DEG=mount.tilt_deg,
        PAN_DEG=mount.pan_deg,
    ) + _SETUP
    print(f"mount camera: {post(args.host, args.bridge, '/execute', {'code': setup})['return_value']}")

    capture_dir = Path(args.capture_dir) if args.capture_dir else None
    if capture_dir:
        capture_dir.mkdir(parents=True, exist_ok=True)

    max_pos = max_head = 0.0
    for x, y, z, yaw in DEFAULT_POSES:
        code = _literals(
            BODY_PATH=args.body, CAM_PATH=cam_path,
            BODY_XYZ=(x, y, z), BODY_YAW=yaw,
            SET_VIEWPORT=bool(capture_dir),
        ) + _PROBE
        res = post(args.host, args.bridge, "/execute", {"code": code})["return_value"]

        body_local = frame.pose_to_local(res["body"]["t"], res["body"]["q"])
        predicted = mount.world_pose(body_local)
        usd_cam = frame.stage_to_local(res["cam"]["t"])

        d_pos = math.dist(
            (predicted.east_m, predicted.north_m, predicted.up_m), usd_cam
        )
        # The prim carries the lens-orientation rotation baked in, so undo that
        # 90 deg before comparing against the boresight bearing the map draws.
        actual_head = frame.yaw_to_heading(quat_to_yaw_deg(res["cam"]["q"]) + 90.0)
        d_head = abs((predicted.heading_deg - actual_head + 180.0) % 360.0 - 180.0)

        max_pos, max_head = max(max_pos, d_pos), max(max_head, d_head)
        print(
            f"  yaw {yaw:6.1f} | body ({body_local.east_m:7.2f},"
            f" {body_local.north_m:7.2f}) hdg {body_local.heading_deg:6.2f}"
            f" | lens predicted ({predicted.east_m:8.3f}, {predicted.north_m:8.3f})"
            f" usd ({usd_cam[0]:8.3f}, {usd_cam[1]:8.3f})"
            f" | dpos {d_pos:.6f} m dhdg {d_head:.6f} deg"
        )

        if capture_dir:
            # Let the kit main loop redraw before capturing -- it only runs
            # BETWEEN bridge requests, so this sleep is doing real work.
            time.sleep(5)
            shot = post(args.host, args.bridge, "/sim/capture", {})
            raw = base64.b64decode(shot["image_base64"])
            out = capture_dir / f"mountcam_yaw{int(yaw)}.png"
            out.write_bytes(raw)
            print(f"           captured {len(raw)} bytes -> {out}")

    print(f"\nMAX position error: {max_pos:.6f} m")
    print(f"MAX heading  error: {max_head:.6f} deg")
    print(f"poses: {len(DEFAULT_POSES)}")
    ok = max_pos < args.tolerance and max_head < args.tolerance
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
