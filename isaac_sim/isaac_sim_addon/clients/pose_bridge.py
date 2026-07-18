#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Isaac body pose -> Tritium tactical map (capability 2: map agreement).

``camera_server.py`` answers "what does the robot SEE".  This answers the other
half of the operator's question: **"where IS it, and which way is it facing?"**
Without this the operator watches a video feed from a body that has no icon on
the map; with it, the icon and the viewport agree, and every downstream
consumer (fusion, dossiers, geofences, dispatch) treats the simulated body
exactly like a real MQTT-reporting robot.

Both North Star halves.  FUN — the sim body shows up on the tactical map as a
first-class tracked target you can click, follow, and dispatch, so an Isaac
scene becomes a playable mission rather than a detached 3D window.
PRODUCTION — this is the pose->map path a real robot uses; proving it against
a simulator where GROUND TRUTH IS KNOWN EXACTLY is the only way to measure map
error honestly.  On real hardware you never know the true pose, so you can
never tell a 2 m map error from a 2 m GPS error.  Here you can.

Dependency hygiene (the isaac-bridge rule)
------------------------------------------
This lives in ``clients/``, not ``connectors/``, and the split is load-bearing:
connectors run inside Isaac's python and may never import tritium; clients run
anywhere else, never import ``isaacsim``, and may import tritium freely.  This
module reads pose over the Isaac MCP bridge's HTTP API, so it runs from a
laptop against a remote RTX host and the only heavy thing it touches is the
network.  The frame maths lives in ``tritium_lib.geo.isaac_frame`` -- pure,
tested, and the single source of truth.

Why the maths is in lib and not here
------------------------------------
Isaac yaw is CCW from +X; Tritium heading is CW from north.  ``heading = 90 -
yaw`` is a REFLECTION.  A rover, an aerial body, and a ROS2 ``/odom`` relay all
need the identical conversion, and only this file runs anywhere near a GPU --
so the conversion is library code and this is just plumbing.

Usage
-----
Print one pose and exit (the bring-up check)::

    python3 pose_bridge.py --bridge http://rtx4090:8211 --prim /World/Go2 --once

Stream converted TrackedTarget payloads (see THE MISSING SEAM below -- SC has
no pose-ingest route yet, so this prints rather than posts)::

    python3 pose_bridge.py --bridge http://rtx4090:8211 --prim /World/Go2 \
        --target-id isaac_go2_01 --hz 4 --emit

No Isaac, no GPU, no network -- exercise the whole conversion + emit path::

    python3 pose_bridge.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Frame maths comes from tritium-lib -- the ONE tested implementation.
#
# This is a ``clients/`` module, not a ``connectors/`` one: it never runs
# inside Isaac's python, so it is free to depend on tritium (see
# ``clients/__init__.py`` for why that distinction is load-bearing).  An
# earlier draft carried a vendored copy of the maths as an import fallback;
# that was deleted deliberately.  A second copy of a sign convention is not
# resilience, it is a silent-divergence bug waiting for the day the two
# disagree and the operator's map is wrong in a way no test catches.
# ---------------------------------------------------------------------------
from tritium_lib.geo.isaac_frame import IsaacFrame, LocalPose, quat_to_yaw_deg

FRAME_SOURCE = "tritium_lib.geo.isaac_frame"


__all__ = [
    "IsaacPoseBridge",
    "StagePose",
    "pose_to_target",
    "POSE_QUERY_CODE",
]

DEFAULT_BRIDGE = "http://localhost:8211"
DEFAULT_PRIM = "/World/Go2"
DEFAULT_TARGET_ID = "isaac_body_01"

# Python evaluated INSIDE the live Isaac process by the MCP bridge's /execute.
# Kept as one module-level constant so the tests can assert on the contract and
# so there is exactly one place where stage introspection is written.
#
# ComputeLocalToWorldTransform is deliberate: a robot is usually parented under
# an Xform (or spawned under /World/<name>), so the prim's LOCAL translation is
# not its world position.  Reading the local one is the bug this avoids.
POSE_QUERY_CODE = """
import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath(PRIM_PATH)
if not prim or not prim.IsValid():
    result = {"ok": False, "error": "prim not found: %s" % PRIM_PATH}
else:
    # Time 0: these scenes are simulated, not animated -- the physics-updated
    # transform is what is authored at the default time sample.
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    result = {
        "ok": True,
        "prim": PRIM_PATH,
        "type": str(prim.GetTypeName()),
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "quat_wxyz": [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])],
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
    }
"""


@dataclass(frozen=True)
class StagePose:
    """A raw pose as read out of the USD stage, before any frame conversion."""

    prim: str
    translation: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    up_axis: str
    meters_per_unit: float
    stamp: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], stamp: float) -> "StagePose":
        if not payload.get("ok"):
            raise LookupError(payload.get("error", "pose query failed"))
        return cls(
            prim=str(payload["prim"]),
            translation=tuple(float(v) for v in payload["translation"]),  # type: ignore[arg-type]
            quat_wxyz=tuple(float(v) for v in payload["quat_wxyz"]),  # type: ignore[arg-type]
            up_axis=str(payload.get("up_axis", "Z")),
            meters_per_unit=float(payload.get("meters_per_unit", 1.0)),
            stamp=stamp,
        )

    def frame(self) -> IsaacFrame:
        """The frame implied by the stage's own metadata -- no guessing."""
        return IsaacFrame.from_stage_metadata(
            {"up_axis": self.up_axis, "meters_per_unit": self.meters_per_unit}
        )

    def to_local(self) -> LocalPose:
        return self.frame().pose_to_local(self.translation, self.quat_wxyz)


def pose_to_target(
    pose: LocalPose,
    *,
    target_id: str = DEFAULT_TARGET_ID,
    stamp: float | None = None,
    source: str = "isaac_sim",
    classification: str = "robot",
    alliance: str = "friendly",
) -> dict[str, Any]:
    """Shape a converted pose as a Tritium tracked-target update.

    Field names follow the ``TrackedTarget`` conventions in ../CLAUDE.md so the
    payload drops into the same ingest path a real robot's MQTT telemetry uses.
    ``source`` is honest about provenance: a fused picture must be able to tell
    a simulated contact from a radio one.
    """
    return {
        "target_id": target_id,
        "source": source,
        "classification": classification,
        "alliance": alliance,
        "x": round(pose.east_m, 4),
        "y": round(pose.north_m, 4),
        "z": round(pose.up_m, 4),
        "heading": round(pose.heading_deg, 3),
        "timestamp": stamp if stamp is not None else time.time(),
    }


class IsaacPoseBridge:
    """Reads a prim's world pose from a live Isaac over the MCP bridge.

    Args:
        bridge_url: base URL of the Isaac MCP bridge (``/health``, ``/execute``).
        prim_path: the body's prim path, e.g. ``/World/Go2``.
        timeout: per-request timeout in seconds.
        transport: injection seam for tests -- ``(path, payload) -> dict``.
            Defaults to real HTTP.  Keeping this a parameter is what lets the
            whole read/convert/emit path be tested with no Isaac and no network.
    """

    def __init__(
        self,
        bridge_url: str = DEFAULT_BRIDGE,
        prim_path: str = DEFAULT_PRIM,
        *,
        timeout: float = 10.0,
        transport: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.bridge_url = bridge_url.rstrip("/")
        self.prim_path = prim_path
        self.timeout = timeout
        self._transport = transport or self._http_post
        self.reads = 0
        self.errors = 0
        self.last_error: str | None = None

    # -- transport --------------------------------------------------------

    def _http_post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        req = urllib.request.Request(
            f"{self.bridge_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- reads ------------------------------------------------------------

    def health(self) -> Mapping[str, Any]:
        """The bridge's stage metadata -- also the liveness check."""
        reply = self._transport("/health", {})
        if reply.get("status") != "success":
            raise ConnectionError(f"bridge unhealthy: {reply}")
        return reply.get("result", {})

    def read_stage_pose(self) -> StagePose:
        """One world-pose read of ``prim_path`` from the live stage."""
        # PRIM_PATH is injected as a literal assignment rather than formatted
        # into the body, so a path containing quotes cannot break the snippet.
        code = f"PRIM_PATH = {self.prim_path!r}\n" + POSE_QUERY_CODE
        stamp = time.time()
        reply = self._transport("/execute", {"code": code})
        if reply.get("status") != "success":
            self.errors += 1
            self.last_error = str(reply.get("error", reply))
            raise ConnectionError(f"/execute failed: {self.last_error}")
        payload = reply.get("result", {}).get("return_value")
        if not isinstance(payload, Mapping):
            self.errors += 1
            self.last_error = f"no return_value in reply: {reply.get('result')}"
            raise ConnectionError(self.last_error)
        pose = StagePose.from_payload(payload, stamp)
        self.reads += 1
        return pose

    def read_target(self, target_id: str = DEFAULT_TARGET_ID) -> dict[str, Any]:
        """Read + convert + shape, in one call: the streaming loop's body."""
        stage_pose = self.read_stage_pose()
        return pose_to_target(
            stage_pose.to_local(), target_id=target_id, stamp=stage_pose.stamp
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bridge_url": self.bridge_url,
            "prim_path": self.prim_path,
            "reads": self.reads,
            "errors": self.errors,
            "last_error": self.last_error,
            "frame_source": FRAME_SOURCE,
        }


# ---------------------------------------------------------------------------
# SC ingest
# ---------------------------------------------------------------------------

# THE MISSING SEAM (measured 2026-07-18, not speculated).
#
# There is currently NO Command Center route that accepts an externally-driven
# body pose.  This was checked against the running SC source, not assumed:
#
#   * ``GET  /api/targets``            read-only; no matching POST/PUT.
#   * ``POST /api/sighting``           dispatches on ``source`` and supports
#                                      only ble / yolo / wifi / mesh.  Any
#                                      other source -- including ``isaac_sim``
#                                      -- returns **501 Not Implemented** by
#                                      design (targets_unified.py), so this is
#                                      a deliberate closed door, not an
#                                      oversight we could sneak through.
#   * ``POST /api/robots``             spawns VIRTUAL robots that integrate
#                                      their own motion via dispatch/patrol.
#                                      There is no "set pose from external
#                                      ground truth" entry point.
#
# So the last mile of capability 2 is blocked on an SC-side ingest seam, NOT on
# anything Isaac-side.  Everything above this line is verified working against
# a live Isaac; deliberately nothing below it pretends to complete the loop.
#
# Rather than ship a call to a route that 404s (which would look like progress
# and silently do nothing), this bridge stops at a correct, converted,
# TrackedTarget-shaped payload.  ``--emit`` prints exactly what an ingest route
# should receive, so the SC seam can be written against a real sample.
#
# NEXT BUILD: a source-agnostic pose ingest -- most naturally by teaching
# ``POST /api/sighting`` a ``robot_pose`` source that routes into
# ``TargetTracker`` the way ble/yolo/wifi/mesh already do.


# ---------------------------------------------------------------------------
# Selftest -- no Isaac, no GPU, no network
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Drive the full read->convert->emit path against a fake stage."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # A fake Isaac: body at stage (10, 20, 0.4) facing +X (east), Z-up, 1 m/unit.
    def fake(path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if path == "/health":
            return {"status": "success", "result": {"up_axis": "Z", "meters_per_unit": 1.0}}
        return {
            "status": "success",
            "result": {
                "return_value": {
                    "ok": True,
                    "prim": "/World/Go2",
                    "type": "Xform",
                    "translation": [10.0, 20.0, 0.4],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],  # yaw 0 => facing east
                    "up_axis": "Z",
                    "meters_per_unit": 1.0,
                }
            },
        }

    bridge = IsaacPoseBridge(transport=fake)
    check("health", bridge.health()["up_axis"] == "Z")

    pose = bridge.read_stage_pose().to_local()
    check("east", abs(pose.east_m - 10.0) < 1e-9, f"{pose.east_m}")
    check("north", abs(pose.north_m - 20.0) < 1e-9, f"{pose.north_m}")
    check("up", abs(pose.up_m - 0.4) < 1e-9, f"{pose.up_m}")
    # Facing +X in Isaac is EAST, which is heading 90 -- not 0.
    check("heading_east_is_90", abs(pose.heading_deg - 90.0) < 1e-9, f"{pose.heading_deg}")

    target = bridge.read_target(target_id="isaac_go2_01")
    check("target_id", target["target_id"] == "isaac_go2_01")
    check("target_xy", (target["x"], target["y"]) == (10.0, 20.0), str((target["x"], target["y"])))
    check("target_source", target["source"] == "isaac_sim")
    check("reads_counted", bridge.reads == 2, str(bridge.reads))

    # A missing prim must raise, not report the origin.
    def missing(path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"status": "success", "result": {"return_value": {"ok": False, "error": "prim not found"}}}

    try:
        IsaacPoseBridge(transport=missing).read_stage_pose()
        check("missing_prim_raises", False, "no exception")
    except LookupError:
        check("missing_prim_raises", True)

    print(f"[POSE BRIDGE SELFTEST] frame_source={FRAME_SOURCE}")
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -> ' + detail) if detail else ''}")
        failed += 0 if ok else 1
    print(f"[POSE BRIDGE SELFTEST] {len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE, help="Isaac MCP bridge base URL")
    ap.add_argument("--prim", default=DEFAULT_PRIM, help="body prim path in the stage")
    ap.add_argument("--emit", action="store_true",
                    help="print the full TrackedTarget JSON per tick (the payload an "
                         "SC ingest seam should accept -- see THE MISSING SEAM above)")
    ap.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    ap.add_argument("--hz", type=float, default=4.0, help="pose poll rate")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to stream (0 = forever)")
    ap.add_argument("--once", action="store_true", help="print one pose and exit")
    ap.add_argument("--selftest", action="store_true", help="run offline selftest and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    bridge = IsaacPoseBridge(args.bridge, args.prim)

    if args.once:
        stage_pose = bridge.read_stage_pose()
        local = stage_pose.to_local()
        print(json.dumps({
            "stage": {
                "prim": stage_pose.prim,
                "translation": list(stage_pose.translation),
                "quat_wxyz": list(stage_pose.quat_wxyz),
                "up_axis": stage_pose.up_axis,
                "meters_per_unit": stage_pose.meters_per_unit,
            },
            "local": {
                "east_m": local.east_m, "north_m": local.north_m,
                "up_m": local.up_m, "heading_deg": local.heading_deg,
            },
            "target": pose_to_target(local, target_id=args.target_id),
            "frame_source": FRAME_SOURCE,
        }, indent=2))
        return 0

    period = 1.0 / max(args.hz, 0.1)
    started = time.time()
    while True:
        try:
            target = bridge.read_target(args.target_id)
            if args.emit:
                print(json.dumps(target), flush=True)
            else:
                print(f"[{target['timestamp']:.1f}] x={target['x']:8.2f} "
                      f"y={target['y']:8.2f} z={target['z']:6.2f} "
                      f"hdg={target['heading']:6.1f}", flush=True)
        except (ConnectionError, LookupError, OSError) as exc:
            print(f"[pose] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.duration and (time.time() - started) >= args.duration:
            return 0
        time.sleep(period)


if __name__ == "__main__":
    raise SystemExit(main())
