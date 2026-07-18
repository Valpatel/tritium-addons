# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fire a hitscan round from a LIVE Isaac body and grade it against the stage.

Capability 6 has sat at CODE_ONLY and then WIRED_UNTESTED since this lane
started: the quadruped server can compute a shot, but no round has ever been
fired inside a running Isaac.  This client closes that, and the way it is
built is the whole point.

**Nothing here decides whether the shot hit.**  The geometry is
:mod:`tritium_lib.geo.hitscan` -- the same call a real turret on a real robot
makes -- and this module only moves numbers between it and the live stage.  A
connector that re-derives the ray gets to disagree with production about where
the round went, and the disagreement shows up as a demo that works and a robot
that misses.

**The stage is the source of truth, not the request.**  Target spheres are
authored and then read BACK off the stage, and the shot is graded against the
poses Isaac reports rather than the ones we asked for.  Tick 17 of this lane
lost two trials to exactly the inverse: a stale obstacle prim outlived its run,
so the planner and the solver held different geometry and both verdicts graded
a box that was not there.  Reading back costs one round trip and makes that
class of lie impossible.

**A hit alone proves nothing.**  Any implementation returns "hit" if you aim
it at a target and never test the other half, so every run here fires a
matched MISS arm -- the same body, the same targets, the boresight slewed off
axis -- and the run is only meaningful if the two arms disagree.

**The round stops at the ground.**  The stage's terrain (the ``GroundSlab``
box whose top face is the floor the body stands on) is read back as a
:class:`~tritium_lib.geo.hitscan.BoxTarget` and resolved in the SAME
``resolve_shot`` call as the spheres, nearest hit wins.  Before this, terrain
was simply not in the target list, and the live control tracer drew to max
range visibly BELOW the slab — a round that passes through the ground is a
solver that will also let a real turret "hit" a target behind a berm.  A
round that ends in terrain grades as a MISS of whatever it was aimed at
(:func:`is_terrain` tells the two apart), but its tracer now terminates at
the impact point on the dirt.

The stage snapshot and the shot are one round trip each, so a walking body
does not move between reading its pose and firing from it.

**The operator sees the round.**  Every resolved shot is reported to SC's
``POST /api/engagement/shot`` (see :class:`ShotPoster`) so the tactical map
draws the tracer and impact, the kill feed logs it, and Amy narrates it.
Reporting is fire-and-forget: a run with no SC loses nothing but the
audience.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tritium_lib.geo.camera_mount import CameraMount
from tritium_lib.geo.hitscan import (
    BoxTarget,
    ShotResult,
    SphereTarget,
    muzzle_from_body,
    resolve_shot,
)
from tritium_lib.geo.isaac_frame import LocalPose

from .nav_bridge import ssh_transport, unwrap_result

DEFAULT_PORT = 8212  # the Newton kit; 8211 is the PhysX one
DEFAULT_BODY_PRIM = "/World/Tritium/go2/base"
DEFAULT_TARGET_ROOT = "/World/FireTargets"

# Prim paths read back as TERRAIN the round terminates against.  Same names
# nav_bridge refuses to treat as obstacles: the slab lidar_server authors and
# the plain ground some scenes carry.  Terrain is never aimed at, and a round
# that ends in it grades as a MISS of its intended target.
DEFAULT_TERRAIN_PRIMS = ("/World/GroundSlab", "/World/Ground")

# target_id prefix marking terrain in a ShotResult, so grading and drawing
# can tell "stopped in the dirt" apart from "hit the thing it was aimed at".
TERRAIN_ID_PREFIX = "terrain:"


def is_terrain(target_id: str | None) -> bool:
    """Does this ``ShotResult.target_id`` name terrain rather than a target?"""
    return bool(target_id) and str(target_id).startswith(TERRAIN_ID_PREFIX)

# Matches TURRET_MOUNT_OFFSET / BARREL_LENGTH_M in isaac_quadruped_server so
# the two firing paths speak about the same weapon.
DEFAULT_MOUNT = CameraMount(forward_m=0.25, left_m=0.0, up_m=0.55)
DEFAULT_BARREL_M = 0.20
DEFAULT_MAX_RANGE_M = 60.0

# How far off the boresight the control arm aims.  Large enough that no target
# of plausible radius at these ranges can be clipped by accident, small enough
# that the round still travels across the same scene.
CONTROL_OFFSET_DEG = 35.0


@dataclass(frozen=True)
class TargetSpec:
    """A sphere to author onto the stage, in metres, world frame."""

    name: str
    east_m: float
    north_m: float
    up_m: float
    radius_m: float = 0.5


# --- stage source (returned, not executed, so tests can assert it) --------


def build_targets_src(
    specs: Sequence[TargetSpec], root: str = DEFAULT_TARGET_ROOT
) -> str:
    """Source that (re-)authors the target spheres and sweeps orphans.

    Every prim is authored fresh each run and anything else under ``root`` is
    removed.  Reusing a prim path that already exists leaves the PREVIOUS
    run's geometry in place with the new run's name on it, which is how tick
    17 graded a shot against a box that was not there.
    """
    payload = [
        {
            "name": s.name,
            "pos": [s.east_m, s.north_m, s.up_m],
            "radius": s.radius_m,
        }
        for s in specs
    ]
    return f'''
import omni.usd
from pxr import Usd, UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
root = "{root}"
specs = {json.dumps(payload)}
wanted = set(root + "/" + s["name"] for s in specs)

# Orphan sweep first: anything under root that this run did not ask for is
# last run's geometry, and it would occlude or absorb this run's rounds.
existing = stage.GetPrimAtPath(root)
if existing and existing.IsValid():
    for child in list(existing.GetChildren()):
        if str(child.GetPath()) not in wanted:
            stage.RemovePrim(child.GetPath())
else:
    UsdGeom.Xform.Define(stage, root)

authored = []
for s in specs:
    path = root + "/" + s["name"]
    stage.RemovePrim(path)
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(s["radius"]))
    sphere.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in s["pos"]]))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.16, 0.43)])
    authored.append(path)

result = {{"authored": authored}}
'''


def read_scene_src(
    body_prim: str = DEFAULT_BODY_PRIM,
    root: str = DEFAULT_TARGET_ROOT,
    terrain_prims: Sequence[str] = DEFAULT_TERRAIN_PRIMS,
) -> str:
    """Source that reads the body pose and every target's ACTUAL world pose.

    Both in one snippet on purpose: a body walking under Newton moves between
    two round trips, and a shot graded from a pose read a moment before the
    targets were read is graded from a scene that never existed.

    The radius is read through the world transform's scale rather than off the
    radius attribute alone, because a scaled Xform ancestor changes how big
    the sphere actually is without touching that attribute.

    Terrain comes back as the world-space AABBs of ``terrain_prims``
    (BBoxCache with the default purpose — the same bound Isaac's own
    selection outline draws), so the fire ray terminates at the ground the
    stage actually has rather than at a z = 0 someone assumed.  Absent prims
    are skipped: a stage with no slab simply has no terrain to stop a round.
    """
    return f'''
import omni.usd, math
from pxr import Usd, UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

def world_of(prim):
    m = xform_cache.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    return m, (float(t[0]), float(t[1]), float(t[2]))

body = stage.GetPrimAtPath("{body_prim}")
if not body or not body.IsValid():
    raise RuntimeError("body prim not found: {body_prim}")

bm, bpos = world_of(body)
rot = bm.ExtractRotationMatrix()
# Body +X is the nose in this asset; heading is the compass bearing of that
# axis projected onto the ground plane.
fwd = Gf.Vec3d(rot[0][0], rot[0][1], rot[0][2])
heading = math.degrees(math.atan2(float(fwd[0]), float(fwd[1]))) % 360.0

targets = []
troot = stage.GetPrimAtPath("{root}")
if troot and troot.IsValid():
    for child in troot.GetChildren():
        sph = UsdGeom.Sphere(child)
        if not sph:
            continue
        m, pos = world_of(child)
        r = float(sph.GetRadiusAttr().Get() or 0.0)
        # Largest axis scale: an ellipsoid's bounding sphere, so a stretched
        # target is never reported SMALLER than it draws.
        scale = max(Gf.Vec3d(m[i][0], m[i][1], m[i][2]).GetLength() for i in range(3))
        targets.append({{"id": child.GetName(), "pos": list(pos), "radius": r * float(scale)}})

# Terrain: world AABB of each named prim, in the SAME snapshot as the body
# and targets, so the ground the round stops at is the ground of this instant.
terrain = []
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for tp in {json.dumps(list(terrain_prims))}:
    prim = stage.GetPrimAtPath(tp)
    if not prim or not prim.IsValid():
        continue
    try:
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception:
        continue
    if rng.IsEmpty():
        continue
    mn, mx = rng.GetMin(), rng.GetMax()
    terrain.append({{"path": tp,
                     "min": [float(mn[i]) for i in range(3)],
                     "max": [float(mx[i]) for i in range(3)]}})

result = {{
    "body": {{"pos": list(bpos), "heading_deg": heading}},
    "targets": targets,
    "terrain": terrain,
}}
'''


def draw_shot_src(
    origin: tuple[float, float, float],
    end: tuple[float, float, float],
    hit: bool,
) -> str:
    """Source that draws the tracer and marks where the round stopped.

    Uses the transient debug-draw interface rather than authoring prims: a
    tracer is a picture, not scene geometry, and a USD prim for it would land
    in the next run's obstacle read and in the LiDAR sweep as a phantom
    return.  Note ``clear_lines`` wipes ALL debug lines, so a route drawn by
    ``nav_bridge`` does not survive a shot.
    """
    colour = "(0.02, 1.0, 0.63, 1.0)" if hit else "(1.0, 0.93, 0.04, 1.0)"
    return f'''
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
draw.clear_lines()
draw.clear_points()
starts = [{list(origin)}]
ends = [{list(end)}]
draw.draw_lines(starts, ends, [{colour}], [8.0])
draw.draw_points([{list(end)}], [{colour}], [22.0])
result = "tracer %s" % ("HIT" if {hit!r} else "MISS")
'''


# --- reporting a round to the Command Center ------------------------------

DEFAULT_SC_URL = "http://localhost:8000"
SC_URL_ENV = "TRITIUM_SC_URL"
ENGAGEMENT_PATH = "/api/engagement/shot"


def default_shooter_id(body_prim: str) -> str:
    """A stable operator-facing id derived from the body prim.

    ``/World/Tritium/go2/base`` -> ``isaac_go2``: the scaffolding segments
    name the scene, not the machine, so they are dropped and the remaining
    stem is what a track on the tactical map gets called.
    """
    skip = {"world", "tritium", "base"}
    parts = [p for p in body_prim.split("/") if p and p.lower() not in skip]
    return f"isaac_{parts[-1] if parts else 'body'}"


class ShotPoster:
    """Reports each resolved round to SC's ``POST /api/engagement/shot``.

    Before this class, a round fired and graded in the live Newton sim
    stopped existing at the connector: the operator's map showed nothing.
    The payload is ``ShotResult.to_dict()`` plus ``shooter_id`` /
    ``shooter_type`` / ``timestamp`` -- exactly the wire shape the SC router
    hands to ``ShotEvent.from_payload``, so the map's tracer, the impact dot,
    the kill feed, and the announcer all light up from the same record this
    client graded.

    **Fire-and-forget, by contract.**  A live sim run must never fail, hang,
    or slow down because SC is absent, so every failure -- connection
    refused, timeout, non-2xx -- is COUNTED, logged once, and swallowed.
    After :data:`MAX_CONSECUTIVE_FAILURES` consecutive failures the poster
    opens its circuit and stops attempting for the rest of the run: against
    the default localhost URL an absent SC refuses instantly, but a
    black-holed remote URL would otherwise tax every shot with a full
    timeout.

    ``opener`` is the injection seam for tests: ``(url, body_bytes) ->
    http status``.  The default is stdlib ``urllib.request`` -- this addon
    never imports tritium.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        base_url: str,
        shooter_id: str,
        *,
        shooter_type: str | None = None,
        timeout_s: float = 1.0,
        opener: Callable[[str, bytes], int] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + ENGAGEMENT_PATH
        self.shooter_id = shooter_id
        self.shooter_type = shooter_type
        self.timeout_s = timeout_s
        self._opener = opener if opener is not None else self._http_post
        self.posted = 0
        self.failures = 0
        self._consecutive = 0
        self._warned = False

    def build_payload(self, shot: ShotResult) -> dict:
        """The exact body ``/api/engagement/shot`` accepts.

        ``ShotResult.to_dict()`` untouched -- re-shaping it here is how the
        connector and the Command Center come to disagree about a round --
        plus the three fields only this side knows: who fired, what kind of
        machine it is (picks the SC magazine loadout), and when.
        """
        payload = dict(shot.to_dict())
        payload["shooter_id"] = self.shooter_id
        if self.shooter_type:
            payload["shooter_type"] = self.shooter_type
        payload["timestamp"] = time.time()
        return payload

    def _http_post(self, url: str, body: bytes) -> int:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return int(resp.status)

    def post(self, shot: ShotResult) -> bool:
        """Report one round.  True on a 2xx; never raises, never blocks long.

        A 409 from SC is a dry trigger pull (the shooter's magazine ran
        empty), which lands in the same failure count -- the trial's own
        verdict is already graded locally and does not depend on it.
        """
        if self._consecutive >= self.MAX_CONSECUTIVE_FAILURES:
            return False
        body = json.dumps(self.build_payload(shot)).encode("utf-8")
        try:
            status = self._opener(self.url, body)
        except Exception as exc:  # noqa: BLE001 -- graceful is the contract
            return self._fail(f"{type(exc).__name__}: {exc}")
        if not 200 <= int(status) < 300:
            return self._fail(f"HTTP {status}")
        self.posted += 1
        self._consecutive = 0
        return True

    def stats(self) -> dict:
        """For the run report: where the rounds went, and how many arrived."""
        return {"url": self.url, "posted": self.posted, "failures": self.failures}

    def _fail(self, why: str) -> bool:
        self.failures += 1
        self._consecutive += 1
        if not self._warned:
            self._warned = True
            print(
                f"[fire_bridge] SC engagement POST failed ({why}) -- "
                "continuing without the operator map",
                file=sys.stderr,
            )
        return False


def poster_from_args(args: argparse.Namespace) -> ShotPoster | None:
    """The CLI's poster, or None when reporting is switched off.

    Posting is ON by default: against the default localhost URL an absent SC
    is an instant connection-refused, so the common no-SC run pays one log
    line, while a run WITH SC lights the operator map with zero extra flags.
    ``--no-sc`` (or an empty ``--sc-url``) is the explicit off switch.
    """
    if getattr(args, "no_sc", False) or not getattr(args, "sc_url", ""):
        return None
    return ShotPoster(
        args.sc_url,
        args.shooter_id or default_shooter_id(args.body_prim),
        shooter_type=getattr(args, "shooter_type", None) or None,
    )


# --- the client ----------------------------------------------------------


def parse_scene(
    payload: Mapping[str, Any],
) -> tuple[LocalPose, list[SphereTarget | BoxTarget]]:
    """Stage snapshot -> the body's pose and the targets, in lib terms.

    Terrain AABBs become :class:`BoxTarget` entries whose ids carry
    :data:`TERRAIN_ID_PREFIX`, in the SAME list as the spheres —
    ``resolve_shot`` takes the mixed set and the nearest hit wins, which is
    exactly how a round comes to stop at the ground instead of reaching a
    target buried behind it.
    """
    body = payload["body"]
    east, north, up = (float(v) for v in body["pos"])
    pose = LocalPose(
        east_m=east,
        north_m=north,
        up_m=up,
        heading_deg=float(body["heading_deg"]) % 360.0,
    )
    targets: list[SphereTarget | BoxTarget] = [
        SphereTarget(
            target_id=str(t["id"]),
            east_m=float(t["pos"][0]),
            north_m=float(t["pos"][1]),
            up_m=float(t["pos"][2]),
            radius_m=float(t["radius"]),
        )
        for t in payload.get("targets", [])
    ]
    for t in payload.get("terrain", []):
        lo, hi = t["min"], t["max"]
        targets.append(
            BoxTarget(
                target_id=f"{TERRAIN_ID_PREFIX}{t.get('path', 'ground')}",
                min_east_m=float(lo[0]),
                min_north_m=float(lo[1]),
                min_up_m=float(lo[2]),
                max_east_m=float(hi[0]),
                max_north_m=float(hi[1]),
                max_up_m=float(hi[2]),
            )
        )
    return pose, targets


def aim_at(
    body: LocalPose, target: SphereTarget, mount: CameraMount = DEFAULT_MOUNT
) -> CameraMount:
    """The mount slewed so the boresight passes through ``target``.

    Solved from the PIVOT rather than the body origin, because the mount sits
    forward and above it: aiming from the body centre and firing from the
    muzzle leaves a parallax error that grows as the target gets closer --
    worst exactly where a short-range weapon is used.

    The barrel length is deliberately not corrected for; it runs ALONG the
    boresight, so it moves the origin without changing the direction.
    """
    pivot = mount.world_pose(body)
    de = target.east_m - pivot.east_m
    dn = target.north_m - pivot.north_m
    du = target.up_m - pivot.up_m

    import math

    bearing = math.degrees(math.atan2(de, dn)) % 360.0
    ground = math.hypot(de, dn)
    tilt = math.degrees(math.atan2(du, ground))
    # CameraMount.world_pose applies pan as heading - pan (positive to the
    # LEFT), so the pan that lands the boresight on `bearing` is the negated
    # difference.  Doing this arithmetic anywhere but next to that method is
    # how the sign gets lost.
    pan = (body.heading_deg - bearing) % 360.0
    if pan > 180.0:
        pan -= 360.0
    return CameraMount(
        forward_m=mount.forward_m,
        left_m=mount.left_m,
        up_m=mount.up_m,
        pan_deg=pan,
        tilt_deg=max(-89.9, min(89.9, tilt)),
    )


class FireBridge:
    """Fires rounds from a body on a live Isaac stage and grades them.

    Args:
        host: ssh host running the kit, or None for a local bridge.
        port: MCP bridge port (8212 Newton, 8211 PhysX).
        body_prim: the body the weapon is mounted on.
        target_root: scope owned by this client; swept every run.
        terrain_prims: prims whose world AABBs terminate the round (ground).
        transport: injection seam for tests -- ``(path, payload) -> dict``.
        poster: optional :class:`ShotPoster`; every resolved round is
            reported through it so the operator's map sees what the sim
            fired.  None (the default here) posts nothing.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = DEFAULT_PORT,
        *,
        body_prim: str = DEFAULT_BODY_PRIM,
        target_root: str = DEFAULT_TARGET_ROOT,
        terrain_prims: Sequence[str] = DEFAULT_TERRAIN_PRIMS,
        mount: CameraMount = DEFAULT_MOUNT,
        barrel_m: float = DEFAULT_BARREL_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        transport: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        poster: ShotPoster | None = None,
    ) -> None:
        self.body_prim = body_prim
        self.target_root = target_root
        self.terrain_prims = tuple(terrain_prims)
        self.mount = mount
        self.barrel_m = barrel_m
        self.max_range_m = max_range_m
        self.poster = poster
        if transport is not None:
            self._transport = transport
        elif host:
            self._transport = ssh_transport(host, port=port)
        else:
            raise ValueError("give either a transport or an ssh host")
        self.errors = 0

    def _execute(self, code: str) -> Any:
        reply = self._transport("/execute", {"code": code})
        if reply.get("status") != "success":
            self.errors += 1
            raise ConnectionError(f"bridge /execute failed: {reply}")
        return unwrap_result(reply)

    def author_targets(self, specs: Sequence[TargetSpec]) -> list[str]:
        return list(self._execute(build_targets_src(specs, self.target_root))["authored"])

    def read_scene(self) -> tuple[LocalPose, list[SphereTarget | BoxTarget]]:
        return parse_scene(
            self._execute(
                read_scene_src(self.body_prim, self.target_root, self.terrain_prims)
            )
        )

    def fire(self, mount: CameraMount, draw: bool = True) -> ShotResult:
        """Read the stage, fire one round through it, draw the tracer.

        The tracer ends where the ROUND ended: at the target it hit, at the
        terrain that stopped it, or at max range in clear air.  A terrain
        stop draws in the MISS colour — the round did not hit what it was
        aimed at — but it terminates AT the ground impact, which is the fix
        for the live run whose control tracer continued below the slab.
        """
        body, targets = self.read_scene()
        muzzle = muzzle_from_body(body, mount, self.barrel_m)
        shot = resolve_shot(muzzle, list(targets), self.max_range_m)

        # The round is resolved: tell the Command Center BEFORE drawing, so
        # the operator's record does not depend on the tracer round trip.
        # post() never raises and never blocks past its short timeout.
        if self.poster is not None:
            self.poster.post(shot)

        if draw:
            origin = muzzle.origin()
            end = shot.impact()
            if end is None:
                aim = muzzle.direction()
                end = tuple(origin[i] + aim[i] * self.max_range_m for i in range(3))
            self._execute(
                draw_shot_src(origin, end, shot.hit and not is_terrain(shot.target_id))
            )
        return shot

    def look_from(
        self, eye: tuple[float, float, float], at: tuple[float, float, float]
    ) -> Any:
        """Park the viewport camera so the tracer is side-on, not end-on.

        A shot photographed down its own boresight is a dot, which is exactly
        the picture that proves nothing.  The bridge's ``/camera/look_at``
        wants a prim, so the viewport camera is posed directly instead.
        """
        return self._execute(f'''
import omni.usd
from pxr import UsdGeom, Gf, Sdf
stage = omni.usd.get_context().get_stage()
cam_path = "/World/FireCam"
cam = UsdGeom.Camera.Define(stage, cam_path)
prim = stage.GetPrimAtPath(cam_path)
prim.GetAttribute("focalLength").Set(24.0) if prim.GetAttribute("focalLength") else None
xf = UsdGeom.Xformable(cam)
xf.ClearXformOpOrder()
eye = Gf.Vec3d({list(eye)})
at = Gf.Vec3d({list(at)})
m = Gf.Matrix4d().SetLookAt(eye, at, Gf.Vec3d(0, 0, 1)).GetInverse()
xf.AddTransformOp().Set(m)
try:
    import omni.kit.viewport.utility as vu
    vu.get_active_viewport().camera_path = cam_path
except Exception as exc:
    pass
result = cam_path
''')

    def capture(self, local_path: str, width: int = 1280, height: int = 720) -> str:
        """Render the viewport and write the PNG to THIS machine.

        Decoding here rather than leaving the image on the GPU box is what
        makes the frame reviewable in the same place the run is graded.
        """
        import base64
        import pathlib

        reply = self._transport("/sim/capture", {"width": width, "height": height})
        if reply.get("status") != "success":
            raise ConnectionError(f"capture failed: {str(reply)[:200]}")
        payload = reply.get("result", {})
        b64 = payload.get("image_base64") if isinstance(payload, Mapping) else None
        if not b64:
            raise ConnectionError("capture returned no image_base64")
        out = pathlib.Path(local_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(b64))
        return str(out)


def run_trial(
    bridge: FireBridge,
    specs: Sequence[TargetSpec],
    on_shot: Callable[[str, ShotResult], None] | None = None,
) -> dict:
    """One matched pair: aim AT a target, then aim off it. Both are needed.

    The aimed arm alone cannot distinguish working fire control from a
    function that returns True, which is why the verdict below requires the
    two arms to disagree rather than requiring the first to hit.
    """
    bridge.author_targets(specs)
    body, targets = bridge.read_scene()
    # Terrain rides in the same list for the RESOLVER, but only spheres are
    # things to aim at -- a trial that "aims" at the ground proves nothing.
    spheres = [t for t in targets if isinstance(t, SphereTarget)]
    if not spheres:
        raise RuntimeError("no targets on the stage after authoring")

    primary = min(
        spheres,
        key=lambda t: (t.east_m - body.east_m) ** 2 + (t.north_m - body.north_m) ** 2,
    )
    aimed_mount = aim_at(body, primary, bridge.mount)
    aimed = bridge.fire(aimed_mount)
    if on_shot is not None:
        on_shot("aimed", aimed)

    off_mount = CameraMount(
        forward_m=aimed_mount.forward_m,
        left_m=aimed_mount.left_m,
        up_m=aimed_mount.up_m,
        pan_deg=aimed_mount.pan_deg + CONTROL_OFFSET_DEG,
        tilt_deg=aimed_mount.tilt_deg,
    )
    # The control arm is drawn only when someone is photographing it: its
    # tracer would otherwise clear the aimed one, and the frame that matters
    # by default is the hit.
    control = bridge.fire(off_mount, draw=on_shot is not None)
    if on_shot is not None:
        on_shot("control", control)

    return {
        "body": {
            "east_m": body.east_m,
            "north_m": body.north_m,
            "up_m": body.up_m,
            "heading_deg": body.heading_deg,
        },
        "targets": [
            {"id": t.target_id, "pos": [t.east_m, t.north_m, t.up_m], "radius": t.radius_m}
            for t in spheres
        ],
        "terrain": [
            {
                "id": t.target_id,
                "min": [t.min_east_m, t.min_north_m, t.min_up_m],
                "max": [t.max_east_m, t.max_north_m, t.max_up_m],
            }
            for t in targets
            if isinstance(t, BoxTarget)
        ],
        "aimed": aimed.to_dict(),
        "control": control.to_dict(),
        "verdict": grade_trial(aimed, control, primary.target_id),
    }


def grade_trial(aimed: ShotResult, control: ShotResult, expected_id: str) -> dict:
    """Did the pair actually demonstrate fire control?

    Three conditions, all required.  The aimed round must hit, it must hit the
    target it was aimed at (hitting SOMETHING is not aiming), and the control
    round must miss -- otherwise the ray is not discriminating and the hit
    carries no information.

    A round that stopped in TERRAIN hit the ground, not a target: for grading
    it counts as a miss (the expected fate of most control rounds now that the
    ground is solid), and the verdict records the terrain stop separately so
    the tracer geometry stays auditable.
    """
    hit_expected = bool(aimed.hit and aimed.target_id == expected_id)
    aimed_hit_target = bool(aimed.hit and not is_terrain(aimed.target_id))
    control_hit_target = bool(control.hit and not is_terrain(control.target_id))
    discriminates = bool(aimed_hit_target and not control_hit_target)
    return {
        "hit_intended_target": hit_expected,
        "control_missed": not control_hit_target,
        "control_stopped_by_terrain": bool(
            control.hit and is_terrain(control.target_id)
        ),
        "discriminates": discriminates,
        "pass": hit_expected and discriminates,
        "aimed_range_m": aimed.range_m,
        "control_miss_distance_m": control.miss_distance_m,
    }


DEFAULT_SPECS = (
    TargetSpec("dummy_near", east_m=0.0, north_m=4.0, up_m=0.5, radius_m=0.4),
    TargetSpec("dummy_left", east_m=-3.0, north_m=6.0, up_m=0.5, radius_m=0.4),
    TargetSpec("dummy_high", east_m=2.5, north_m=7.0, up_m=2.0, radius_m=0.4),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", help="ssh host running the Isaac kit")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--body-prim", default=DEFAULT_BODY_PRIM)
    parser.add_argument("--capture", help="local path to write the viewport PNG")
    parser.add_argument(
        "--eye",
        default="7,-4,4.5",
        help="viewport camera position, 'e,n,u' -- side-on to the tracer",
    )
    parser.add_argument(
        "--print-src",
        action="store_true",
        help="print the stage snippets and exit (no GPU, no bridge)",
    )
    parser.add_argument(
        "--sc-url",
        default=os.environ.get(SC_URL_ENV, DEFAULT_SC_URL),
        help="Tritium SC base URL for live shot reporting "
        f"(env {SC_URL_ENV}; default {DEFAULT_SC_URL})",
    )
    parser.add_argument(
        "--no-sc",
        action="store_true",
        help="do not report shots to the SC engagement endpoint",
    )
    parser.add_argument(
        "--shooter-id",
        default="",
        help="track id SC shows for this shooter (default: derived from the body prim)",
    )
    parser.add_argument(
        "--shooter-type",
        default="robot_dog",
        help="asset type SC uses to pick the shooter's magazine loadout",
    )
    args = parser.parse_args(argv)

    if args.print_src:
        print(build_targets_src(DEFAULT_SPECS))
        print(read_scene_src(args.body_prim))
        return 0

    if not args.host:
        parser.error("--host is required unless --print-src")

    poster = poster_from_args(args)
    bridge = FireBridge(
        host=args.host, port=args.port, body_prim=args.body_prim, poster=poster
    )
    if args.capture:
        # Pose the camera BEFORE firing: the tracer is transient debug-draw,
        # and moving the viewport afterwards can outlive the line it was
        # meant to photograph.
        eye = tuple(float(v) for v in args.eye.split(","))
        bridge.look_from(eye, (0.0, 4.0, 0.6))

    frames: dict[str, str] = {}

    def shoot_and_shoot(arm: str, shot: ShotResult) -> None:
        # One frame per arm, from an identical camera: the PAIR is the proof,
        # since a single hit photograph is what a function returning True
        # would also produce.
        stem = args.capture.rsplit(".png", 1)[0]
        frames[arm] = bridge.capture(f"{stem}-{arm}.png")

    report = run_trial(
        bridge, DEFAULT_SPECS, on_shot=shoot_and_shoot if args.capture else None
    )
    if args.capture:
        report["frames"] = frames
    if poster is not None:
        # How many of this run's rounds actually reached the operator: a
        # trial can pass with SC absent, and the report must say which.
        report["sc"] = poster.stats()

    json.dump(report, sys.stdout, indent=2)
    print()
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
