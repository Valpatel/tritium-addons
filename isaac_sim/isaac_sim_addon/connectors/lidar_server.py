#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Isaac Sim LiDAR as a JSON range server — a fake LiDAR endpoint for testing.

The LiDAR twin of ``camera_server.py`` (capability 7).  Where the camera server
exposes a camera SENSOR behind the SAME transport a real IP camera uses (MJPEG),
this exposes a LiDAR behind the SAME transport a generic range server uses:
a LaserScan-style JSON document over plain HTTP.  tritium-edge's
``SensorBridgeNode`` already ingests exactly this — its ``JsonHttpScanSource``
GETs a JSON scan document per tick and ``parse_scan_json`` accepts
``{"ranges": [...]}`` plus optional ``angle_min/angle_max/range_min/range_max``
overrides, republished as ``sensor_msgs/LaserScan`` — so an Isaac LiDAR plugs
into the robot brain like any bench LiDAR and NOTHING downstream needs to know
it is Isaac.  Same separation as the camera: Isaac ray-traces on the RTX 4090;
the sweep crosses the LAN as ordinary JSON; the consumer never imports isaacsim.

Both North Star halves: FUN — the sim robot gets a live 360-degree "radar ring"
the operator can watch sweep around obstacles in the scene.  PRODUCTION —
validates the real LiDAR track (scan -> bridge -> LaserScan -> nav/avoidance)
against a controllable, repeatable environment BEFORE any physical LiDAR spins.

Dependency hygiene (the isaac-bridge rule): Isaac's python imports isaacsim +
stdlib/numpy only; it NEVER imports paho / pydantic / tritium.  Consumers speak
HTTP JSON and never import isaacsim.  No dependency bleed in either direction.

Scan sources
------------
  * ``--source isaac``      real RTX Lidar prim attached under ``/World/Tritium``
                            (needs Isaac's python and a free GPU); reads the
                            range/azimuth scan buffer each step (Newton-safe:
                            only prim transforms + sensor reads, no
                            physics-backend assumptions).
  * ``--source synthetic``  DEFAULT — a no-GPU stand-in: analytic ray casts from
                            the sensor origin against a rectangular room, two
                            static pillars, and one orbiting obstacle.  Pure
                            numpy, deterministic per tick, so the whole
                            transport + the downstream bridge can be proven
                            with no Isaac and no GPU.  ``--selftest`` uses this.

Routes
------
  * ``/scan``    the latest sweep as JSON (also served at ``/``)::

        {"lidar_id": "isaac-lidar-01", "seq": 42, "stamp": 1789600000.0,
         "angle_min": -3.1416, "angle_max": 3.1241, "angle_increment": 0.01745,
         "range_min": 0.1, "range_max": 30.0, "ranges": [4.02, 4.01, ...]}

  * ``/status``  JSON metadata (source, beams, geometry, scan count).

Run
---
    # Real Isaac RTX Lidar (Isaac's bundled python, GPU free — see README)
    ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
        isaac_sim_addon/connectors/lidar_server.py \
        --source isaac --port 8110 --lidar-id isaac-lidar-01 \
        --mount-prim /World/Tritium/Robot/lidar

    # No-GPU stand-in (system python3) — same JSON a real range server serves
    python3 isaac_sim_addon/connectors/lidar_server.py --source synthetic --port 8110

    # No-GPU self-test: generate N sweeps, assert geometry/bounds/no-NaN
    python3 isaac_sim_addon/connectors/lidar_server.py --selftest

Then point the tritium-edge sensor bridge at it (no Isaac code on the robot)::

    ros2 run tritium_perception sensor_bridge --ros-args \
        -p scan_url:=http://<render-host>:8110/scan
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

log = logging.getLogger("isaac-lidar")


# --------------------------------------------------------------------------- #
# Scan geometry helpers (pure — shared by synthetic + Isaac resampling).
# --------------------------------------------------------------------------- #

def beam_angles(num_beams: int, angle_min: float = -math.pi) -> np.ndarray:
    """The ``num_beams`` beam azimuths of a full 360-degree sweep starting at
    ``angle_min``: evenly spaced, increment ``2*pi/num_beams``, last beam one
    increment short of wrapping onto the first (LaserScan convention)."""
    inc = 2.0 * math.pi / max(1, num_beams)
    return angle_min + inc * np.arange(num_beams, dtype=np.float64)


def resample_to_beams(ranges, azimuths, num_beams: int,
                      angle_min: float, range_min: float,
                      range_max: float) -> np.ndarray:
    """Bin an unordered (range, azimuth) point cloud — the shape an RTX Lidar
    scan buffer hands back — into an ordered ``num_beams`` sweep.

    Each return falls into the nearest beam bin; multiple returns in one bin
    keep the CLOSEST (the obstacle that matters).  Empty bins read
    ``range_max`` (no return), never NaN.  Out-of-band ranges are clamped so
    the output always satisfies ``range_min <= r <= range_max``.  Pure numpy,
    GPU-free, unit-tested — this is the seam between Isaac's cloud and the
    LaserScan contract."""
    inc = 2.0 * math.pi / max(1, num_beams)
    out = np.full(num_beams, range_max, dtype=np.float64)
    r = np.asarray(ranges, dtype=np.float64).ravel()
    az = np.asarray(azimuths, dtype=np.float64).ravel()
    if r.size == 0 or az.size != r.size:
        return out
    good = np.isfinite(r) & np.isfinite(az)
    r, az = r[good], az[good]
    if r.size == 0:
        return out
    idx = np.round((az - angle_min) / inc).astype(np.int64) % num_beams
    # Closest return per bin: sort descending by range so the final (closest)
    # write into each bin wins.
    order = np.argsort(-r)
    out[idx[order]] = r[order]
    return np.clip(out, range_min, range_max)


# --------------------------------------------------------------------------- #
# Scan sources.
# --------------------------------------------------------------------------- #

class ScanSource:
    """A LiDAR sweep producer returning ordered 1-D range arrays (metres).

    ``get_scan()`` advances the source's clock and returns the next sweep:
    ``num_beams`` floats, beam ``i`` at azimuth ``angle_min + i * increment``,
    every value within ``[range_min, range_max]``, never NaN/inf."""

    name = "abstract"

    #: True when the source may ONLY be stepped from the process's main thread.
    #: Omniverse Kit is such a source: ``world.step()`` never returns when it is
    #: called from a worker, and it does so SILENTLY — no exception, no error
    #: count, just a scan loop parked forever while /status keeps answering
    #: ``scans: 0, errors: 0``.  Same contract as ``camera_server``'s
    #: ``FrameSource.requires_main_thread``.
    requires_main_thread = False

    def __init__(self, num_beams: int = 360, range_min: float = 0.1,
                 range_max: float = 30.0, angle_min: float = -math.pi):
        self.num_beams = int(num_beams)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.angle_min = float(angle_min)

    @property
    def angle_increment(self) -> float:
        return 2.0 * math.pi / max(1, self.num_beams)

    @property
    def angle_max(self) -> float:
        return self.angle_min + self.angle_increment * (self.num_beams - 1)

    def get_scan(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SyntheticScanSource(ScanSource):
    """No-GPU stand-in: analytic ray casts in a room with obstacles.

    The scene (sensor at the origin, X forward, angles CCW):
      * a rectangular ROOM ``room_w x room_h`` metres, sensor at the centre —
        every beam hits a wall, so the sweep is always fully in-range;
      * two static PILLARS (circles) for fixed structure;
      * one ORBITING obstacle circling the sensor, advancing a fixed step per
        tick — deterministic motion the downstream tracker can follow.

    Deliberately simple (mirrors the camera server's walking subject): it only
    has to be coherent geometry a bridge/avoidance stack can consume.  The REAL
    ray-traced sweep comes from IsaacScanSource."""

    name = "synthetic"

    _PILLARS = ((2.5, 1.5, 0.30), (-3.0, -2.0, 0.40))   # (x, y, radius) m
    _ORBIT_R = 2.5          # orbit radius of the moving obstacle (m)
    _ORBIT_RAD = 0.35       # its body radius (m)
    _ORBIT_STEP = 0.05      # orbit advance per tick (rad) — deterministic

    def __init__(self, num_beams: int = 360, range_min: float = 0.1,
                 range_max: float = 30.0, angle_min: float = -math.pi,
                 room_w: float = 10.0, room_h: float = 8.0):
        super().__init__(num_beams, range_min, range_max, angle_min)
        self.room_w = float(room_w)
        self.room_h = float(room_h)
        self._tick = 0

    # -- analytic intersections (vectorized over all beams) ------------------
    def _wall_distance(self, cos_a: np.ndarray, sin_a: np.ndarray) -> np.ndarray:
        """Distance to the axis-aligned room walls from the centre."""
        hx, hy = self.room_w / 2.0, self.room_h / 2.0
        with np.errstate(divide="ignore"):
            tx = np.where(np.abs(cos_a) > 1e-12, hx / np.abs(cos_a), np.inf)
            ty = np.where(np.abs(sin_a) > 1e-12, hy / np.abs(sin_a), np.inf)
        return np.minimum(tx, ty)

    @staticmethod
    def _circle_distance(cos_a, sin_a, cx: float, cy: float,
                         radius: float) -> np.ndarray:
        """Per-beam distance to a circle (inf where the beam misses it)."""
        b = cos_a * cx + sin_a * cy                    # ray-dot-centre
        disc = b * b - (cx * cx + cy * cy - radius * radius)
        hit = (disc >= 0.0) & (b > 0.0)
        t = np.where(hit, b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
        return np.where(t > 0.0, t, np.inf)

    def scan_at(self, tick: int) -> np.ndarray:
        """The sweep at animation ``tick`` — pure function of the tick, so the
        synthetic LiDAR is reproducible in tests and replays."""
        ang = beam_angles(self.num_beams, self.angle_min)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        dist = self._wall_distance(cos_a, sin_a)
        for cx, cy, radius in self._PILLARS:
            dist = np.minimum(dist, self._circle_distance(cos_a, sin_a, cx, cy, radius))
        # Orbiting obstacle: circles the sensor, one fixed step per tick.
        theta = tick * self._ORBIT_STEP
        ox = self._ORBIT_R * math.cos(theta)
        oy = self._ORBIT_R * math.sin(theta)
        dist = np.minimum(dist, self._circle_distance(cos_a, sin_a, ox, oy,
                                                      self._ORBIT_RAD))
        return np.clip(dist, self.range_min, self.range_max)

    def get_scan(self) -> np.ndarray:
        scan = self.scan_at(self._tick)
        self._tick += 1
        return scan


def flat_scan_to_ranges(frame: dict, num_beams: int, angle_min: float,
                        range_min: float, range_max: float) -> tuple:
    """``IsaacComputeRTXLidarFlatScan`` output -> ``(ranges, has_returns)``.

    Pure (numpy only) so both LiDAR shapes — the own-kit ``IsaacScanSource``
    and the attached mode riding an already-running kit — bin the sensor's
    frame through ONE tested path instead of two drifting copies.

    ``has_returns`` is False when the frame carried no positive depth at all;
    the caller owns warm-up bookkeeping, because "no returns yet" means
    different things to a booting sensor and a running one.

    -1 is the flat scan's NO-RETURN SENTINEL, not a range.  Feeding it through
    unmasked makes every empty bearing the nearest contact in the sweep — an
    obstacle detector would brake for open air.  Dropped here; the resampler
    leaves those beams at ``range_max``, which is what "nothing out there"
    means in a LaserScan.
    """
    depth = np.asarray(frame.get("linearDepthData", []),
                       dtype=np.float64).ravel()
    valid = depth > 0.0
    if depth.size == 0 or not valid.any():
        return np.full(num_beams, range_max, dtype=np.float64), False
    # The flat scan reports its sweep as an azimuth RANGE in degrees plus a
    # dense row of depths, not a per-point azimuth array — bearings are
    # implied by index, so they are reconstructed here.  endpoint=False
    # because the last bearing wraps onto the first (a 360 deg sweep must
    # not sample -180 and +180 as two distinct beams).
    az_lo, az_hi = np.deg2rad(np.asarray(
        frame.get("azimuthRange", (-180.0, 180.0)), dtype=np.float64))
    bearings = np.linspace(az_lo, az_hi, depth.size, endpoint=False)
    return resample_to_beams(depth[valid], bearings[valid], num_beams,
                             angle_min, range_min, range_max), True


class IsaacScanSource(ScanSource):
    """Real Isaac Sim RTX Lidar -> ordered sweeps.

    Runs ONLY inside Isaac's python (imports isaacsim).  Boots a headless
    SimulationApp, attaches an RTX Lidar prim at ``mount_prim`` (under
    ``/World/Tritium`` — the robot's frame), builds a minimal ground + obstacle
    scene when no USD is given, and per get_scan() steps the world and bins the
    lidar's (range, azimuth) scan buffer into the ordered LaserScan sweep via
    :func:`resample_to_beams`.

    Kept intentionally thin — this class is the seam, not a physics
    playground.  Newton-safe: it touches only prim creation, transforms, and
    the RTX sensor annotator; nothing here assumes a PhysX backend."""

    name = "isaac"
    # Kit's update loop is main-thread-only — see ScanSource.requires_main_thread.
    requires_main_thread = True

    def __init__(self, num_beams: int = 360, range_min: float = 0.1,
                 range_max: float = 30.0, angle_min: float = -math.pi,
                 mount_prim: str = "/World/Tritium/Robot/lidar",
                 scene_usd: str | None = None, lidar_config: str = "Example_Rotary_2D",
                 physics_hz: int = 30):
        super().__init__(num_beams, range_min, range_max, angle_min)
        self.mount_prim = mount_prim
        self._sim = None
        self._world = None
        self._lidar = None
        # Warm-up state: `_warmed` flips the first time the sensor returns a
        # single point, and never flips back.  See get_scan().
        self._warmed = False
        self._empty_frames = 0
        self._boot(scene_usd, lidar_config, physics_hz)

    @property
    def never_returned(self) -> bool:
        """True while the sensor has not produced one point since boot.

        A consumer MUST NOT read an all-range_max sweep as an empty room
        until this is False."""
        return not self._warmed

    def _boot(self, scene_usd, lidar_config, physics_hz):
        # Imports are local so plain python3 (selftest/synthetic/tests) never
        # touches isaacsim.  Any import/boot failure is raised for main() to
        # report honestly — no fabricated sweeps from a half-booted sim.
        from isaacsim.simulation_app import SimulationApp  # type: ignore

        self._sim = SimulationApp({"headless": True})

        from pxr import UsdGeom  # type: ignore
        import isaacsim.core.utils.stage as stage_utils  # type: ignore
        from isaacsim.core.api import World  # type: ignore
        from isaacsim.sensors.rtx import LidarRtx  # type: ignore

        if scene_usd:
            stage_utils.open_stage(scene_usd)
        stage = stage_utils.get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self._world = World(stage_units_in_meters=1.0)
        # GROUND: a box slab, never a flat quad.  `add_default_ground_plane()`
        # authors a zero-thickness quad, which qhull cannot hull; under Newton
        # the resulting exception latches the physics extension's
        # `_initializing` flag and silently disables integration for the whole
        # process while the clock keeps advancing (the four-tick blocker fixed
        # in `examples/newton_ground_fix_proof.py`).  The lidar only needs
        # geometry to ray-trace, but the scene must stay Newton-safe for the
        # moment a body is dropped into it — and a slab also gives the sweep an
        # honest floor return instead of an infinitely thin one.
        # Mirrors the lib's `geo.collider_shape.ground_slab`, authored inline
        # because connectors stay dependency-clean (isaacsim + numpy only).
        from pxr import UsdPhysics  # type: ignore
        slab = UsdGeom.Cube.Define(stage, "/World/GroundSlab")
        slab.CreateSizeAttr(2.0)  # unit cube; the scale op below sets extents
        slab_xf = UsdGeom.Xformable(slab.GetPrim())
        slab_xf.AddTranslateOp().Set((0.0, 0.0, -0.5))  # top face at z = 0
        slab_xf.AddScaleOp().Set((25.0, 25.0, 0.5))     # 50 x 50 x 1 m
        UsdPhysics.CollisionAPI.Apply(slab.GetPrim())
        if not scene_usd:
            # Minimal scene: a few obstacle boxes ringing the sensor so the
            # sweep has structure (the synthetic room's Isaac counterpart).
            from isaacsim.core.api.objects import FixedCuboid  # type: ignore
            for i, (x, y) in enumerate(((4.0, 0.0), (0.0, 3.5), (-4.0, -1.0))):
                self._world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/Tritium/Obstacle_{i}",
                        name=f"obstacle_{i}",
                        position=np.array([x, y, 0.75]),
                        scale=np.array([0.8, 0.8, 1.5]),
                    )
                )
        # The RTX Lidar prim on the robot mount.
        #
        # CONFIG MUST BE A 2D LIDAR.  `IsaacComputeRTXLidarFlatScan` refuses to
        # execute against a multi-beam sensor -- with `Example_Rotary` it logs
        # "elevationDeg contains nonzero value -15.0 ... not a 2D Lidar, and
        # node will not execute" and emits nothing.  A LaserScan is a 2D
        # contract, so the sensor has to be 2D at the source.
        self._lidar = LidarRtx(
            prim_path=self.mount_prim,
            name="tritium_lidar",
            position=np.array([0.0, 0.0, 0.6]),
            config_file_name=lidar_config,
        )
        self._world.reset()
        # DATA PATH: attach the replicator annotator DIRECTLY.  The obvious
        # `add_range_data_to_frame()` / `add_azimuth_data_to_frame()` pair is
        # DEPRECATED AS OF ISAAC SIM 5.0 AND ATTACHES NOTHING -- it logs a
        # deprecation warning and leaves `_annotators` empty, so `range` and
        # `azimuth` never entered the frame and EVERY sweep fell into
        # get_scan()'s all-range_max branch.  That is what made the sensor look
        # healthy while producing an empty room for two ticks; it was never the
        # GPU.  `IsaacComputeRTXLidarFlatScan` is Isaac's own ROS LaserScan
        # producer -- the standard wheel, not a reinvention.
        import omni.replicator.core as rep  # type: ignore
        render_product = rep.create.render_product(self.mount_prim, [1, 1],
                                                   name="tritium_lidar_rp")
        self._flat = rep.AnnotatorRegistry.get_annotator(
            "IsaacComputeRTXLidarFlatScan")
        self._flat.attach([render_product])
        self._physics_dt = 1.0 / max(1, physics_hz)

    def get_scan(self) -> np.ndarray:
        self._world.step(render=True)
        frame = self._flat.get_data() or {}
        ranges, has_returns = flat_scan_to_ranges(
            frame, self.num_beams, self.angle_min,
            self.range_min, self.range_max)
        if not has_returns:
            # A sweep of all-range_max is what an empty room looks like AND
            # what a sensor that never rendered looks like.  That ambiguity
            # cost a tick: with the Newton kit app holding 18 of 24 GB, the
            # RTX lidar's render product failed to allocate
            # (VK_ERROR_OUT_OF_DEVICE_MEMORY) and every sweep came back a
            # clean 360x30.0 m — /status showed scans climbing and errors
            # zero, so the sensor "looked healthy and produced nothing".
            # `never_returned` is the flag that tells the two apart; it rides
            # out on /scan and /status so a consumer is never asked to guess.
            self._empty_frames += 1
            if not self._warmed and self._empty_frames in (30, 300, 3000):
                log.warning(
                    "LIDAR HAS NEVER RETURNED A POINT after %d frames — the "
                    "sensor is not ray-tracing (check GPU memory: the RTX "
                    "lidar needs a render product, and an out-of-VRAM "
                    "allocation fails SILENTLY into this branch)",
                    self._empty_frames,
                )
            return ranges
        self._warmed = True
        return ranges

    def close(self) -> None:
        try:
            if self._sim is not None:
                self._sim.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Payload + HTTP server (stdlib only).
# --------------------------------------------------------------------------- #

def build_payload(source: ScanSource, scan: np.ndarray, lidar_id: str,
                  seq: int, stamp: float | None = None) -> dict:
    """The /scan JSON document — a superset of tritium-edge's
    ``parse_scan_json`` contract (``ranges`` + the four geometry overrides);
    the extra keys (``lidar_id``/``seq``/``stamp``/``angle_increment``) are
    metadata consumers may ignore."""
    return {
        "lidar_id": lidar_id,
        "seq": int(seq),
        "stamp": float(stamp if stamp is not None else time.time()),
        "angle_min": source.angle_min,
        "angle_max": source.angle_max,
        "angle_increment": source.angle_increment,
        "range_min": source.range_min,
        "range_max": source.range_max,
        "ranges": [round(float(r), 4) for r in scan],
        # Provenance, not decoration: an all-range_max sweep is ambiguous
        # between "empty room" and "sensor never rendered", and a consumer
        # cannot tell from the ranges alone.  Sources that cannot fail this
        # way (synthetic) report False.
        "never_returned": bool(getattr(source, "never_returned", False)),
    }


class LidarState:
    """Shared latest-scan holder for the HTTP handlers (CameraState's twin).

    A background thread polls the source at ``hz`` and caches the encoded JSON
    payload; a flaky source never kills the server — errors are counted and the
    loop keeps ticking (the /scan route answers 503 until a sweep exists)."""

    def __init__(self, source: ScanSource, lidar_id: str, hz: int):
        self.source = source
        self.lidar_id = lidar_id
        self.hz = max(1, hz)
        self.scans = 0
        self.errors = 0
        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        # Refuse rather than deadlock: a Kit-backed source stepped off the main
        # thread hangs silently forever, which is indistinguishable from "slow"
        # at the /status route.  Fail loudly and name the fix.
        if getattr(self.source, "requires_main_thread", False):
            raise RuntimeError(
                f"source {self.source.name!r} must be scanned on the main thread "
                "(Omniverse Kit deadlocks in a worker); use run_main_thread() and "
                "serve HTTP from a background thread instead"
            )
        threading.Thread(target=self._loop, daemon=True, name="isaac-lidar-scan").start()

    def run_main_thread(self):
        """Drive scanning on the CALLING (main) thread until stopped. Blocks —
        the HTTP server runs in a background thread under this arrangement."""
        self._loop()

    def tick_once(self) -> bool:
        """Poll the source once; cache the payload.  Returns success.  A raising
        source counts an error and leaves the last good sweep in place."""
        try:
            scan = self.source.get_scan()
            payload = build_payload(self.source, scan, self.lidar_id, self.scans)
            body = json.dumps(payload).encode()
        except Exception as exc:
            self.errors += 1
            log.warning("scan failed: %s", exc)
            return False
        with self._lock:
            self._latest = body
            self.scans += 1
        return True

    def _loop(self):
        interval = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.time()
            if not self.tick_once():
                time.sleep(0.5)
            dt = time.time() - t0
            if dt < interval:
                self._stop.wait(interval - dt)

    def latest(self) -> bytes | None:
        with self._lock:
            return self._latest

    def publish(self, body: bytes) -> None:
        """Publish an ALREADY-ENCODED /scan payload to the holder.

        CameraState.publish's twin: the seam that lets the attached mode
        (``attached_sensor_server``) — which bins sweeps inside a running
        Kit's update callback — serve through the same holder, counter and
        HTTP handler as the own-kit poll loop."""
        with self._lock:
            self._latest = body
            self.scans += 1

    def stop(self):
        self._stop.set()
        self.source.close()


def _make_handler(state: LidarState):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send_json(self, body: bytes, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/status"):
                self._send_json(json.dumps({
                    "lidar_id": state.lidar_id,
                    "source": state.source.name,
                    "status": "online",
                    "num_beams": state.source.num_beams,
                    "angle_min": state.source.angle_min,
                    "angle_max": state.source.angle_max,
                    "angle_increment": state.source.angle_increment,
                    "range_min": state.source.range_min,
                    "range_max": state.source.range_max,
                    "hz": state.hz,
                    "scans": state.scans,
                    "errors": state.errors,
                }).encode())
            elif self.path.startswith("/scan") or self.path == "/":
                body = state.latest()
                if body is None:
                    self._send_json(b'{"error": "no scan yet"}', code=503)
                else:
                    self._send_json(body)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def _build_source(args) -> ScanSource:
    if getattr(args, "source", "synthetic") == "isaac":
        return IsaacScanSource(
            num_beams=args.beams, range_min=args.range_min,
            range_max=args.range_max, mount_prim=args.mount_prim,
            scene_usd=args.scene or None, lidar_config=args.lidar_config,
            physics_hz=args.physics_hz,
        )
    return SyntheticScanSource(
        num_beams=args.beams, range_min=args.range_min, range_max=args.range_max,
        room_w=args.room_w, room_h=args.room_h,
    )


def selftest(args) -> int:
    """No-GPU: generate N synthetic sweeps and assert the LaserScan contract —
    beam count, angle bounds, every range in band, no NaN/inf, deterministic
    per tick, and the orbiting obstacle actually moves between sweeps."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # `--selftest` exercises the SYNTHETIC source only — it is the no-GPU
    # contract check.  Silently ignoring `--source isaac` printed a green
    # "SELFTEST OK" for a sensor that had never booted Isaac, which is worse
    # than no check at all: it was read as evidence the live lidar worked.
    if getattr(args, "source", "synthetic") == "isaac":
        print("SELFTEST REFUSED: --selftest is the no-GPU synthetic contract "
              "check and cannot validate --source isaac.  Run the server for "
              "real and read /scan (watch the never_returned flag).",
              file=sys.stderr)
        return 2
    src = SyntheticScanSource(num_beams=args.beams, range_min=args.range_min,
                              range_max=args.range_max)
    scans = []
    for _ in range(args.selftest_scans):
        scan = src.get_scan()
        assert scan.shape == (args.beams,), f"bad sweep shape {scan.shape}"
        assert np.all(np.isfinite(scan)), "sweep contains NaN/inf"
        assert np.all(scan >= src.range_min) and np.all(scan <= src.range_max), \
            "range out of [range_min, range_max]"
        scans.append(scan)
    assert any(not np.array_equal(scans[0], s) for s in scans[1:]), \
        "orbiting obstacle should move between sweeps"
    assert np.array_equal(src.scan_at(0), src.scan_at(0)), "sweep not deterministic"
    payload = build_payload(src, scans[-1], "selftest", len(scans))
    assert len(payload["ranges"]) == args.beams
    print(f"SELFTEST OK ranges={args.beams} "
          f"angle=[{src.angle_min:.3f}..{src.angle_max:.3f}] "
          f"inc={src.angle_increment:.5f} "
          f"range=[{src.range_min:.1f}..{src.range_max:.1f}] "
          f"scans={len(scans)} no_nan deterministic")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Isaac Sim LiDAR JSON range server")
    ap.add_argument("--source", choices=["isaac", "synthetic"], default="synthetic")
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--lidar-id", default="isaac-lidar-01")
    ap.add_argument("--beams", type=int, default=360)
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--range-min", type=float, default=0.1)
    ap.add_argument("--range-max", type=float, default=30.0)
    # Synthetic scene.
    ap.add_argument("--room-w", type=float, default=10.0)
    ap.add_argument("--room-h", type=float, default=8.0)
    # Isaac scene / mount.
    ap.add_argument("--scene", default="")
    ap.add_argument("--mount-prim", default="/World/Tritium/Robot/lidar")
    ap.add_argument("--lidar-config", default="Example_Rotary_2D")
    ap.add_argument("--physics-hz", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-scans", type=int, default=12)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.selftest:
        return selftest(args)

    source = _build_source(args)
    state = LidarState(source, args.lidar_id, args.hz)

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))

    # Kit-backed sources must scan on the main thread, so the two roles swap:
    # HTTP goes to the background and the scan loop keeps main.  Non-Kit sources
    # keep the ordinary arrangement (scan in a worker, HTTP blocking on main).
    main_thread_scan = getattr(source, "requires_main_thread", False)
    if main_thread_scan:
        threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="isaac-lidar-http").start()
    else:
        state.start()

    def _shutdown(*_a):
        log.info("shutting down")
        state.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "ISAAC LIDAR READY source=%s http://%s:%d/scan id=%s beams=%d "
        "angle=[%.3f..%.3f] range=[%.1f..%.1f] hz=%d",
        source.name, args.host, args.port, args.lidar_id, source.num_beams,
        source.angle_min, source.angle_max, source.range_min, source.range_max,
        args.hz,
    )
    try:
        if main_thread_scan:
            state.run_main_thread()   # blocks; HTTP is already serving
        else:
            httpd.serve_forever()
    finally:
        state.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
