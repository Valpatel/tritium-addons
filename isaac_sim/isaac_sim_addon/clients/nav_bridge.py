# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Plan a route around the obstacles in a LIVE Isaac stage.

This is the consumer side of :mod:`tritium_lib.planning.scene_costmap`.  It
reads world-space bounding boxes off a running Isaac Sim over the MCP bridge's
``/execute`` endpoint, projects them onto a 2-D costmap through the body band,
and runs lib's A* to produce a route the body can actually walk.

The split is deliberate and matches ``pose_bridge.py``:

- **This file** knows about the bridge, USD prim paths, and ``BBoxCache``.
- **tritium-lib** knows about costmaps and A*, and imports no simulator.

Nothing here is imported by lib, so the "lib must import on a bare Jetson"
invariant holds: swap this client for a LiDAR clusterer and the same planner
runs on a real robot.

Usage::

    # Read the live stage, plan, print the route (no drawing):
    python3 nav_bridge.py --bridge http://localhost:8211 \
        --start 0,0 --goal 12,4 --robot-prim /World/Go2

    # Plan and draw the route in the viewport:
    python3 nav_bridge.py --goal 12,4 --draw

    # No Isaac, no GPU, no network -- exercises read->convert->plan:
    python3 nav_bridge.py --selftest
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tritium_lib.planning import plan_route
from tritium_lib.planning.scene_costmap import (
    DEFAULT_BODY_BAND,
    SceneObstacle,
    costmap_from_scene,
)

DEFAULT_BRIDGE = "http://localhost:8211"
DEFAULT_ROBOT_PRIM = "/World/Go2"

#: Prim paths never treated as obstacles regardless of geometry.  The ground
#: slab is excluded by the body band already, but naming it is cheaper than
#: hulling a 100 m box, and lights/cameras carry meaningless bounds.
DEFAULT_IGNORE = ("/World/GroundSlab", "/World/Ground", "/OmniverseKit_Persp")


# ---------------------------------------------------------------------------
# Stage reading
# ---------------------------------------------------------------------------

# Injected into the live Isaac as-is.  Emits one JSON line describing every
# imageable prim's world-space AABB in METERS.  BBoxCache with the default
# purpose is what Isaac's own selection outline uses, so what this sees is
# what the operator sees in the viewport.
_READ_OBSTACLES_SRC = '''
import json
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

# PointInstancer PROTOTYPES all bbox at the origin regardless of where their
# instances actually stand, so traversing into one invents a pile of fake
# obstacles right on top of the world origin.  Skip those subtrees; real
# instance placement would have to come from GetPositionsAttr().
instancer_roots = [
    str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.PointInstancer)
]

out = []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Imageable):
        continue
    if not prim.IsActive():
        continue
    path_str = str(prim.GetPath())
    if any(path_str == r or path_str.startswith(r + "/") for r in instancer_roots):
        continue
    try:
        bound = cache.ComputeWorldBound(prim)
        rng = bound.ComputeAlignedRange()
    except Exception:
        continue
    if rng.IsEmpty():
        continue
    mn, mx = rng.GetMin(), rng.GetMax()
    center = [((mn[i] + mx[i]) * 0.5) * mpu for i in range(3)]
    half = [(abs(mx[i] - mn[i]) * 0.5) * mpu for i in range(3)]
    if max(half) <= 0.0:
        continue
    out.append({"prim_path": str(prim.GetPath()), "center": center,
                "half_extents": half})

result = json.dumps({"meters_per_unit": mpu, "obstacles": out})
'''


def unwrap_result(reply: Mapping[str, Any]) -> dict:
    """Dig the executed snippet's ``result`` value out of a bridge reply.

    The bridge wraps it as ``{"result": {"stdout", "stderr", "return_value"}}``
    but older builds returned the value directly under ``result``, and the
    value itself may be a JSON string.  Accept all three shapes.
    """
    result = reply.get("result", {})
    if isinstance(result, Mapping) and "return_value" in result:
        result = result["return_value"]
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return {}
    return dict(result) if isinstance(result, Mapping) else {}


def parse_obstacle_payload(
    payload: Mapping[str, Any],
    ignore_prims: Sequence[str] = DEFAULT_IGNORE,
) -> list[SceneObstacle]:
    """Convert the bridge's JSON reply into :class:`SceneObstacle` boxes.

    Prims whose bounds are degenerate or unparseable are skipped rather than
    raising: one bad prim in a large stage must not lose the whole read.
    """
    obstacles: list[SceneObstacle] = []
    for entry in payload.get("obstacles", []):
        path = entry.get("prim_path")
        center = entry.get("center")
        half = entry.get("half_extents")
        if not path or not center or not half:
            continue
        if any(path == ig or path.startswith(ig.rstrip("/") + "/") for ig in ignore_prims):
            continue
        try:
            obstacles.append(
                SceneObstacle(
                    prim_path=str(path),
                    center=(float(center[0]), float(center[1]), float(center[2])),
                    half_extents=(float(half[0]), float(half[1]), float(half[2])),
                )
            )
        except (ValueError, IndexError, TypeError):
            continue
    return obstacles


def crop_to_roi(
    obstacles: Sequence[SceneObstacle],
    start: tuple[float, float],
    goal: tuple[float, float],
    margin_m: float,
) -> list[SceneObstacle]:
    """Keep only obstacles near the start->goal corridor.

    A real city stage returns thousands of prims spread over kilometers.
    Bounding all of them sizes the costmap to the whole map — a 12 m walk
    across a 2 km stage allocated a 7807x5103 grid (40M cells) before this
    existed.  Cropping to the travel corridor plus ``margin_m`` keeps the grid
    proportional to the journey, which is what makes the resolution
    affordable enough to represent doorway-width gaps.
    """
    lo_x = min(start[0], goal[0]) - margin_m
    hi_x = max(start[0], goal[0]) + margin_m
    lo_y = min(start[1], goal[1]) - margin_m
    hi_y = max(start[1], goal[1]) + margin_m

    kept = []
    for obs in obstacles:
        cx, cy, _ = obs.center
        hx, hy, _ = obs.half_extents
        # Keep if the obstacle's footprint overlaps the ROI rectangle at all.
        if cx + hx < lo_x or cx - hx > hi_x:
            continue
        if cy + hy < lo_y or cy - hy > hi_y:
            continue
        kept.append(obs)
    return kept


def descendant_paths(obstacles: Sequence[SceneObstacle], root: str) -> list[str]:
    """Every obstacle path at or under ``root``.  Used to exclude the robot."""
    root_clean = root.rstrip("/")
    return [
        o.prim_path
        for o in obstacles
        if o.prim_path == root_clean or o.prim_path.startswith(root_clean + "/")
    ]


def ssh_transport(
    host: str, port: int = 8211, timeout: float = 300.0
) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Transport that reaches a LOOPBACK-ONLY bridge on ``host`` over ssh.

    The Isaac MCP bridge binds 127.0.0.1 on the GPU box, so it is not
    reachable directly and ``ssh -L`` forwarding is not always available.
    This pipes the JSON payload to a remote ``curl`` over stdin, which avoids
    shell-quoting the (large, multi-line) python source entirely.
    """
    if shutil.which("ssh") is None:  # pragma: no cover - environment guard
        raise RuntimeError("ssh not found on PATH")

    def transport(path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        remote = (
            f'curl -s -m {int(timeout)} -X POST localhost:{port}{path} '
            f'-H "Content-Type: application/json" --data-binary @-'
        )
        proc = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", host, remote],
            input=json.dumps(dict(payload)).encode("utf-8"),
            capture_output=True,
            timeout=timeout + 30,
        )
        if proc.returncode != 0:
            raise ConnectionError(
                f"ssh transport failed ({proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
            )
        try:
            return json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ConnectionError(
                f"bridge returned non-JSON: {proc.stdout[:400]!r}"
            ) from exc

    return transport


class NavBridge:
    """Reads obstacles from a live Isaac and plans routes around them.

    Args:
        bridge_url: base URL of the Isaac MCP bridge.
        robot_prim: the body's prim path — excluded from its own costmap.
        body_band: vertical span the body sweeps, meters.
        resolution: costmap cell size, meters.
        clearance_m: standoff kept from lethal cells.
        roi_margin_m: slack around the start->goal corridor; obstacles outside
            it are dropped so the grid stays proportional to the journey.
        timeout: per-request timeout, seconds.
        transport: injection seam for tests — ``(path, payload) -> dict``.
    """

    def __init__(
        self,
        bridge_url: str = DEFAULT_BRIDGE,
        robot_prim: str = DEFAULT_ROBOT_PRIM,
        *,
        body_band: tuple[float, float] = DEFAULT_BODY_BAND,
        resolution: float = 0.25,
        clearance_m: float = 0.25,
        roi_margin_m: float = 15.0,
        timeout: float = 30.0,
        transport: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.bridge_url = bridge_url.rstrip("/")
        self.robot_prim = robot_prim
        self.body_band = body_band
        self.resolution = resolution
        self.clearance_m = clearance_m
        self.roi_margin_m = roi_margin_m
        self.timeout = timeout
        self._transport = transport or self._http_post
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
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def _execute(self, code: str) -> Mapping[str, Any]:
        reply = self._transport("/execute", {"code": code})
        if reply.get("status") != "success":
            self.errors += 1
            self.last_error = str(reply)
            raise ConnectionError(f"bridge /execute failed: {reply}")
        return reply

    # -- reads ------------------------------------------------------------

    def read_obstacles(self) -> list[SceneObstacle]:
        """Every world-space AABB on the live stage, in meters."""
        reply = self._execute(_READ_OBSTACLES_SRC)
        return parse_obstacle_payload(unwrap_result(reply))

    # -- planning ---------------------------------------------------------

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        obstacles: Sequence[SceneObstacle] | None = None,
    ):
        """Plan ``start`` -> ``goal`` around the live stage's obstacles.

        Reads the stage when ``obstacles`` is not supplied.  The robot's own
        prims are excluded, and start/goal are forced inside the costmap
        bounds so a goal past the outermost obstacle still plans.
        """
        if obstacles is None:
            obstacles = self.read_obstacles()
        obstacles = crop_to_roi(list(obstacles), start, goal, self.roi_margin_m)

        costmap = costmap_from_scene(
            obstacles,
            resolution=self.resolution,
            body_band=self.body_band,
            ignore_prims=descendant_paths(obstacles, self.robot_prim),
            include=[start, goal],
            padding_m=5.0,
        )
        route = plan_route(costmap, start, goal, clearance_m=self.clearance_m)
        return costmap, route


# ---------------------------------------------------------------------------
# Viewport drawing
# ---------------------------------------------------------------------------

def draw_route_src(
    path: Sequence[tuple[float, float]],
    z: float = 0.05,
    meters_per_unit: float = 1.0,
) -> str:
    """Python source that draws ``path`` as a debug polyline in the viewport.

    Returned as source rather than executed so the exact payload can be
    asserted in tests without a live Isaac.
    """
    scale = 1.0 / (meters_per_unit or 1.0)
    pts = [[x * scale, y * scale, z * scale] for x, y in path]
    return f'''
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
draw.clear_lines()
pts = {json.dumps(pts)}
starts = pts[:-1]
ends = pts[1:]
colors = [(0.02, 1.0, 0.63, 1.0)] * len(starts)
widths = [6.0] * len(starts)
draw.draw_lines(starts, ends, colors, widths)
result = "drew %d segments" % len(starts)
'''


# ---------------------------------------------------------------------------
# Selftest -- no Isaac, no GPU, no network
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Drive read->convert->plan against a fake stage reply."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # A wall across the direct line of travel, plus a ground slab and the
    # robot's own body -- the two things that must NOT become obstacles.
    fake = {
        "meters_per_unit": 1.0,
        "obstacles": [
            {"prim_path": "/World/GroundSlab", "center": [0, 0, -0.5],
             "half_extents": [50, 50, 0.5]},
            {"prim_path": "/World/Go2/base", "center": [0, 0, 0.3],
             "half_extents": [0.4, 0.2, 0.15]},
            {"prim_path": "/World/Wall", "center": [6.0, 0.0, 1.0],
             "half_extents": [0.5, 4.0, 1.0]},
        ],
    }

    def transport(path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        assert path == "/execute", path
        return {"status": "success", "result": json.dumps(fake)}

    bridge = NavBridge(transport=transport, resolution=0.25, clearance_m=0.2)

    obstacles = bridge.read_obstacles()
    paths = {o.prim_path for o in obstacles}
    check("ground slab filtered by name", "/World/GroundSlab" not in paths)
    check("wall survives the read", "/World/Wall" in paths, str(sorted(paths)))

    check(
        "robot prims identified for exclusion",
        descendant_paths(obstacles, "/World/Go2") == ["/World/Go2/base"],
    )

    costmap, route = bridge.plan((0.0, 0.0), (12.0, 0.0))
    check("route found", route.success, route.reason)
    check("route has waypoints", len(route.path) >= 2, str(len(route.path)))

    inside_wall = [
        (x, y) for x, y in route.path
        if abs(x - 6.0) <= 0.5 and abs(y) <= 4.0
    ]
    check("route avoids the wall", not inside_wall, str(inside_wall))

    length = sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(route.path, route.path[1:])
    )
    check("route detours (longer than 12 m straight)", length > 12.0, f"{length:.2f} m")

    # The robot must not block itself: a start at the robot's own position
    # must still plan.
    _, self_route = bridge.plan((0.0, 0.0), (3.0, 0.0))
    check("robot is not its own obstacle", self_route.success, self_route.reason)

    # ROI: a far-off building must not size the grid for a short walk.
    far = SceneObstacle("/World/Horizon/tower", (900.0, 900.0, 10.0), (5.0, 5.0, 10.0))
    near_only = crop_to_roi([*obstacles, far], (0.0, 0.0), (12.0, 0.0), 15.0)
    check("ROI drops the distant tower", far not in near_only)
    check("ROI keeps the wall on the route", any(
        o.prim_path == "/World/Wall" for o in near_only))

    wide_cm, _ = bridge.plan((0.0, 0.0), (12.0, 0.0), [*obstacles, far])
    check(
        "grid stays proportional to the journey",
        wide_cm.width * wide_cm.height < 100_000,
        f"{wide_cm.width}x{wide_cm.height}",
    )

    src = draw_route_src([(0.0, 0.0), (1.0, 2.0)])
    check("draw source carries the points", "[1.0, 2.0, 0.05]" in src, src[-200:])
    check("draw source clears stale lines", "clear_lines()" in src)

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE)
    ap.add_argument("--ssh", default=None, metavar="HOST",
                    help="reach a loopback-only bridge over ssh, e.g. --ssh rtx4090")
    ap.add_argument("--port", type=int, default=8211)
    ap.add_argument("--robot-prim", default=DEFAULT_ROBOT_PRIM)
    ap.add_argument("--start", default="0,0", help="world x,y in meters")
    ap.add_argument("--goal", default="10,0", help="world x,y in meters")
    ap.add_argument("--resolution", type=float, default=0.25)
    ap.add_argument("--clearance", type=float, default=0.25)
    ap.add_argument("--roi", type=float, default=15.0,
                    help="margin (m) around the start->goal corridor")
    ap.add_argument("--band", default="0.10,0.55", help="body band z_min,z_max")
    ap.add_argument("--draw", action="store_true", help="draw the route in the viewport")
    ap.add_argument("--json", action="store_true", help="emit the route as JSON")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    def xy(s: str) -> tuple[float, float]:
        a, b = s.split(",")
        return (float(a), float(b))

    band_lo, band_hi = xy(args.band)
    bridge = NavBridge(
        args.bridge,
        args.robot_prim,
        body_band=(band_lo, band_hi),
        resolution=args.resolution,
        clearance_m=args.clearance,
        roi_margin_m=args.roi,
        transport=ssh_transport(args.ssh, args.port) if args.ssh else None,
    )

    obstacles = bridge.read_obstacles()
    start, goal = xy(args.start), xy(args.goal)
    costmap, route = bridge.plan(start, goal, obstacles)

    if args.json:
        print(json.dumps({
            "obstacles": len(obstacles),
            "success": route.success,
            "reason": route.reason,
            "cost": route.cost,
            "expansions": route.expansions,
            "strategy": route.strategy,
            "clearance_relaxed": route.clearance_relaxed,
            "path": [list(p) for p in route.path],
        }, indent=2))
    else:
        print(f"obstacles read : {len(obstacles)}")
        print(f"costmap        : {costmap.width}x{costmap.height} @ {costmap.resolution} m")
        print(f"route          : {route.reason} cost={route.cost:.2f} "
              f"expansions={route.expansions} strategy={route.strategy}")
        for i, (x, y) in enumerate(route.path):
            print(f"  [{i:2d}] {x:8.2f} {y:8.2f}")

    if not route.success:
        return 2

    if args.draw:
        reply = bridge._execute(draw_route_src(route.path))
        print(f"draw           : {reply.get('result')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
