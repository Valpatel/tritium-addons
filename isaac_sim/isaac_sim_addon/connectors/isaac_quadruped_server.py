# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Isaac Sim quadruped BODY SERVER — the physics half of the brain/body seam.

SIM-TO-REAL STORY: the robot-template brain (navigator, turret, telemetry,
MQTT — ``examples/robot-template/robot.py``) is the process that will one day
run on a Jetson Orin bolted to a real dog. Its ``hardware/isaac.py`` backend
speaks a tiny JSON-lines TCP protocol to "whatever owns the legs". This file
is that other end: a body server that owns an NVIDIA Isaac Sim stage, steps
real physics at a fixed rate, moves a quadruped body kinematically from the
same twist/gait contract the kinematic ``hardware/quadruped.py`` backend
proved, and serves state read back FROM the simulated body prim. Swap this
process for a Unitree Go2 SDK bridge and the brain never notices — same wire,
heavier body.

GAIT CONTRACT: the integrator below mirrors ``hardware/quadruped.py._step``
(which itself mirrors ``tritium_lib.models.quadruped``): walk 0.7 m/s @1.6 Hz,
trot 1.6 @2.6, bound 3.0 @3.2, turn 120 deg/s, 2.5 m/s^2 accel limit, stand
below |forward| < 0.05, 155 Wh pack, idle 25 W, gait power 65/120/250 W, and
the exact ``_footfalls`` stance rules SC's gait diagram renders.

FRAME MAPPING: Tritium is x=east, y=north, heading 0 = north (x=sin(h),
y=cos(h)). The Isaac stage is standard Z-up: X=east, Y=north, Z=up, yaw
counter-clockwise from +X. So Isaac yaw_deg = 90 - heading_deg, and position
maps 1:1 (x_east -> X, y_north -> Y).

The brain-side TCP client is ``tritium-sc/examples/robot-template/hardware/
isaac.py`` (a lightweight, isaacsim-free example that stays in the operator
repo); this heavy Isaac-side body server lives in the addon per the copper-roof
placement rule. Its on-robot twin is ``tritium-edge/ros2/tritium_quadruped``.

RUN (under Isaac's bundled python — never the system interpreter), from the
tritium-addons repo root:
    ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
        isaac_sim/isaac_sim_addon/connectors/isaac_quadruped_server.py --port 18973

SELF-TEST (system python3, no Isaac install needed — the integrator, protocol,
and TCP server are pure stdlib and live above the Isaac boot):
    python3 isaac_sim/isaac_sim_addon/connectors/isaac_quadruped_server.py --selftest

Isaac API usage is based on the LOCAL build's own standalone examples
(v6.0.0-rc.22): ``standalone_examples/api/isaacsim.simulation_app/hello_world.py``
(boot pattern), ``api/isaacsim.core.api/add_cubes.py`` (World + cuboids +
get_world_pose), ``api/isaacsim.core.api/control_robot.py`` (boot-first import
order, add_reference_to_stage, get_assets_root_path),
``benchmarks/benchmark_core_world.py`` (headless World, add_default_ground_plane,
step(render=False)), ``api/isaacsim.robot.policy.examples/spot_standalone.py``
(assets-root failure handling).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time


# =========================================================================
# Pure-python section — import-safe WITHOUT Isaac Sim.
# Everything above run_isaac() runs under the system interpreter; only
# run_isaac() touches the isaacsim packages (and boots SimulationApp first,
# per the local standalone examples).
# =========================================================================

# ----- Gait contract (mirror of hardware/quadruped.py / tritium_lib) -----

GAIT_DEFAULTS: dict[str, dict[str, float]] = {
    "walk": {
        "speed_mps": 0.7,
        "stride_hz": 1.6,
        "roll_amp_deg": 1.5,
        "pitch_amp_deg": 1.0,
        "bob_amp_m": 0.01,
        "power_w": 65.0,
    },
    "trot": {
        "speed_mps": 1.6,
        "stride_hz": 2.6,
        "roll_amp_deg": 2.5,
        "pitch_amp_deg": 1.8,
        "bob_amp_m": 0.02,
        "power_w": 120.0,
    },
    "bound": {
        "speed_mps": 3.0,
        "stride_hz": 3.2,
        "roll_amp_deg": 4.0,
        "pitch_amp_deg": 6.0,
        "bob_amp_m": 0.04,
        "power_w": 250.0,
    },
}

PROFILE_DEFAULTS: dict[str, float] = {
    "body_height_m": 0.40,
    "turn_rate_dps": 120.0,
    "battery_wh": 155.0,
    "idle_power_w": 25.0,
}

_GAIT_ORDER = ("walk", "trot", "bound")
_LEGS = ("FL", "FR", "RL", "RR")
_WALK_SWING_SEQUENCE = ("FL", "RR", "FR", "RL")
_STAND_THRESHOLD = 0.05
_ACCEL_LIMIT = 2.5


def footfalls(gait: str, phase: float) -> list[str]:
    """Legs in STANCE for the given gait at the given stride phase.

    Exact mirror of hardware/quadruped.py._footfalls — the contract SC's
    gait diagram renders.
    """
    if gait == "trot":
        return ["FL", "RR"] if phase < 0.5 else ["FR", "RL"]
    if gait == "bound":
        return ["FL", "FR"] if phase < 0.5 else ["RL", "RR"]
    if gait == "walk":
        swing = _WALK_SWING_SEQUENCE[min(3, int(phase * 4))]
        return [leg for leg in _LEGS if leg != swing]
    return list(_LEGS)


class GaitIntegrator:
    """Twist -> gait/pose integrator, mirroring hardware/quadruped.py._step.

    This produces the KINEMATIC TARGET pose each physics step. The Isaac main
    loop writes it to the kinematic body prim, steps real physics, then reads
    the pose BACK from the stage — the state served over TCP comes from the
    sim body, not from these variables.
    """

    def __init__(self) -> None:
        self.x = 0.0            # east, m
        self.y = 0.0            # north, m
        self.heading = 0.0      # deg, 0 = north
        self.speed = 0.0        # m/s
        self.accel = 0.0        # m/s^2 (last step)
        self.gait = "stand"
        self.phase = 0.0        # stride phase 0..1
        self.battery = 1.0      # 0..1
        self.odometer = 0.0     # m

    def step(self, dt: float, left: float, right: float) -> None:
        """One integration step from the twist intent (same math as the dog)."""
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        forward = (left + right) / 2.0
        turn = left - right

        max_speed = GAIT_DEFAULTS["bound"]["speed_mps"]
        requested = abs(forward) * max_speed

        if abs(forward) < _STAND_THRESHOLD:
            self.gait = "stand"
            target_speed = 0.0
        else:
            chosen = _GAIT_ORDER[-1]
            for name in _GAIT_ORDER:
                if GAIT_DEFAULTS[name]["speed_mps"] >= requested:
                    chosen = name
                    break
            self.gait = chosen
            target_speed = min(requested, GAIT_DEFAULTS[chosen]["speed_mps"])

        delta = target_speed - self.speed
        max_delta = _ACCEL_LIMIT * dt
        delta = max(-max_delta, min(max_delta, delta))
        self.accel = delta / dt if dt > 0 else 0.0
        self.speed += delta

        if self.gait != "stand":
            self.phase = (self.phase + GAIT_DEFAULTS[self.gait]["stride_hz"] * dt) % 1.0

        self.heading = (self.heading + turn * PROFILE_DEFAULTS["turn_rate_dps"] * dt) % 360

        direction = 1.0 if forward >= 0 else -1.0
        rad = math.radians(self.heading)
        dx = math.sin(rad) * self.speed * direction * dt
        dy = math.cos(rad) * self.speed * direction * dt
        self.x += dx
        self.y += dy
        self.odometer += math.hypot(dx, dy)

        if self.gait == "stand":
            power_w = PROFILE_DEFAULTS["idle_power_w"]
        else:
            power_w = GAIT_DEFAULTS[self.gait]["power_w"]
        drain_per_s = power_w / (PROFILE_DEFAULTS["battery_wh"] * 3600.0)
        self.battery = max(0.0, self.battery - drain_per_s * dt)

    def oscillation(self) -> tuple[float, float, float, float]:
        """(roll_deg, pitch_deg, bob_m, accel_z) gait oscillation at the
        current phase — same shape as hardware/quadruped.py.get_imu()."""
        if self.gait == "stand":
            return (0.0, 0.0, 0.0, 9.81)
        g = GAIT_DEFAULTS[self.gait]
        roll = g["roll_amp_deg"] * math.sin(2 * math.pi * self.phase)
        pitch = (g["pitch_amp_deg"] * math.sin(4 * math.pi * self.phase)
                 - self.accel * 0.5)
        bob = g["bob_amp_m"] * math.sin(4 * math.pi * self.phase)
        accel_z = 9.81 + g["bob_amp_m"] * 100.0 * math.sin(4 * math.pi * self.phase)
        return (roll, pitch, bob, accel_z)

    def stride_hz(self) -> float:
        return 0.0 if self.gait == "stand" else GAIT_DEFAULTS[self.gait]["stride_hz"]


# ----- Fire contract (pure math — no Isaac, unit-tested in tests/) -------
#
# Turret frames: pan is degrees CLOCKWISE from the body's forward direction
# (compass-style, matching heading: body facing north + pan 90 = aim east),
# tilt is degrees above the horizon. The muzzle sits on a turret pivot
# mounted forward/up on the torso; the barrel extends along the aim vector.

TURRET_MOUNT_OFFSET = (0.25, 0.0, 0.55)  # (forward, left, up) m in body frame
BARREL_LENGTH_M = 0.20                   # pivot -> muzzle tip along aim
MAGAZINE_SIZE = 30                       # shots per boot (no reload cmd yet)
FIRE_MAX_RANGE_M = 60.0                  # ray is clipped past this

# The ground the body stands on, as an axis-aligned TERRAIN box the fire ray
# terminates against.  Both stage flavours ground the world with its walkable
# surface at z = 0 (add_default_ground_plane here, the /World/GroundSlab box in
# lidar_server), and until this existed a round aimed below the horizon flew
# to max range UNDERNEATH the world — the live control tracer was visibly
# occluded below the slab.  Kept in SharedBody.terrain, SEPARATE to
# SharedBody.targets, because {cmd: "targets"} replaces that list and the
# Isaac loop republishes it every step: terrain registered there would be
# silently deleted within one tick.  Extents are generous rather than exact
# (the fire ray is clipped at FIRE_MAX_RANGE_M anyway); what matters is the
# top face at z = 0.0.
GROUND_TARGET: dict = {
    "id": "terrain_ground",
    "min": [-1000.0, -1000.0, -100.0],
    "max": [1000.0, 1000.0, 0.0],
}

# Below this, a direction vector is numerically meaningless.  Same threshold
# as ``tritium_lib.geo.hitscan._MIN_DIRECTION_NORM`` — these helpers mirror
# that module's ray_sphere/ray_aabb/resolve_shot op for op (connectors stay
# dependency-clean, so the code is duplicated, not imported; the contract
# tests in tests/test_fire.py hold the two copies together).
_MIN_DIRECTION_NORM = 1e-9


def _unit(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalise ``direction``; reject a degenerate vector rather than aim it.

    Mirror of ``tritium_lib.geo.hitscan._unit``: same threshold, same error,
    so a shot the lib refuses to fire is refused here too.
    """
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < _MIN_DIRECTION_NORM:
        raise ValueError(f"direction must be non-degenerate, got {direction!r}")
    return (dx / norm, dy / norm, dz / norm)


def ray_sphere_t(origin: tuple[float, float, float],
                 direction: tuple[float, float, float],
                 center: tuple[float, float, float],
                 radius_m: float) -> float | None:
    """Distance to the sphere's SURFACE, or None for a miss.

    Mirror of ``tritium_lib.geo.hitscan.ray_sphere``, including the two edges
    the old inline test got wrong: a muzzle INSIDE the sphere is a point-blank
    hit at range 0 (``tca < 0`` alone would refuse it), and the inner sqrt is
    clamped so float noise near a graze can never go negative.
    """
    if radius_m < 0.0:
        raise ValueError(f"radius_m must be non-negative, got {radius_m!r}")
    ax, ay, az = _unit(direction)
    ox, oy, oz = origin
    cx, cy, cz = center

    vx, vy, vz = cx - ox, cy - oy, cz - oz
    tca = vx * ax + vy * ay + vz * az
    dist2 = vx * vx + vy * vy + vz * vz
    r2 = radius_m * radius_m

    if tca < 0.0 and dist2 > r2:
        return None  # centre behind the muzzle, muzzle outside the sphere

    perp2 = dist2 - tca * tca
    if perp2 > r2:
        return None  # ray passes wide

    return max(0.0, tca - math.sqrt(max(0.0, r2 - perp2)))


def ray_box(origin: tuple[float, float, float],
            direction: tuple[float, float, float],
            box_min: tuple[float, float, float],
            box_max: tuple[float, float, float]) -> float | None:
    """Distance to the box's near face, or None for a miss.

    Mirror of ``tritium_lib.geo.hitscan.ray_aabb`` — the slab method with the
    parallel case branched out EXPLICITLY.  The naive ``(lo - o) / d`` with a
    direction component of exactly zero yields NaN (and 0/0 when the origin
    sits exactly ON the plane — a real bug mutation testing found in the lib),
    every comparison against NaN is false, and a miss silently reports as a
    hit.  Parallel-to-a-slab is instead decided by whether the origin lies
    between that slab's planes.
    """
    ax, ay, az = _unit(direction)
    t_near = 0.0  # never report geometry behind the muzzle
    t_far = math.inf

    for o, d, lo, hi in zip(origin, (ax, ay, az), box_min, box_max):
        if lo > hi:
            raise ValueError(f"box bounds inverted on one axis: {lo} > {hi}")
        if abs(d) < _MIN_DIRECTION_NORM:
            if o < lo or o > hi:
                return None  # parallel to this slab and outside it
            continue  # parallel but within: this axis cannot clip the ray
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return None

    return t_near


def muzzle_pose(body_x: float, body_y: float, body_heading: float,
                turret_pan: float, turret_tilt: float,
                mount_offset: tuple[float, float, float] = TURRET_MOUNT_OFFSET,
                barrel_len: float = BARREL_LENGTH_M,
                ) -> tuple[tuple[float, float, float],
                           tuple[float, float, float]]:
    """World-frame muzzle origin + unit aim vector from body pose + turret.

    Tritium frame: x=east, y=north, z=up, heading 0 = north, clockwise.
    The mount offset rotates with the BODY heading (it is bolted to the
    torso); the barrel rotates with body heading + pan and pitches by tilt.
    """
    h = math.radians(body_heading % 360.0)
    fx, fy = math.sin(h), math.cos(h)        # body forward in world
    lx, ly = -math.cos(h), math.sin(h)       # body left in world
    fwd, left, up = mount_offset
    pivot = (body_x + fwd * fx + left * lx,
             body_y + fwd * fy + left * ly,
             up)
    azim = math.radians((body_heading + turret_pan) % 360.0)
    tilt = math.radians(turret_tilt)
    ct = math.cos(tilt)
    aim = (math.sin(azim) * ct, math.cos(azim) * ct, math.sin(tilt))
    origin = (pivot[0] + aim[0] * barrel_len,
              pivot[1] + aim[1] * barrel_len,
              pivot[2] + aim[2] * barrel_len)
    return origin, aim


def ray_hit(origin: tuple[float, float, float],
            aim: tuple[float, float, float],
            targets: list[dict],
            max_range: float = FIRE_MAX_RANGE_M) -> dict | None:
    """Closest target the ray from ``origin`` along ``aim`` hits — the FIRST
    thing the round enters across a MIXED set of spheres and boxes.

    Two target shapes, told apart by their keys:
      sphere: ``{"id": str, "x": m, "y": m, "z": m, "radius": m}`` (z
              defaults 0, radius defaults 0.5)
      box:    ``{"id": str, "min": [x, y, z], "max": [x, y, z]}`` — walls,
              vehicles, and the TERRAIN the round must stop at.

    Mirror of ``tritium_lib.geo.hitscan.resolve_shot``: nearest hit wins (you
    cannot shoot through the front target — nor through the ground), and the
    range gate is applied to the winning hit rather than used to pre-filter.
    The two gates are provably equivalent here — the winner is the global
    minimum, so no farther candidate can pass a gate the minimum fails — but
    matching the lib's shape keeps the copies line-comparable.  Malformed
    targets are skipped, never sink the shot (the lib raises instead; a
    protocol server must not die on a bad JSON entry).

    Returns ``{"target_id", "range", "hit_pos": [x, y, z]}`` or ``None``.
    Pure and deterministic — the same helper serves the no-Isaac path and
    the live-stage path (which feeds it poses read back from the sim).
    """
    ox, oy, oz = origin
    best_id: str | None = None
    best_t: float | None = None
    for tgt in targets:
        try:
            if "min" in tgt and "max" in tgt:
                lo = tuple(float(v) for v in tgt["min"])
                hi = tuple(float(v) for v in tgt["max"])
                if len(lo) != 3 or len(hi) != 3:
                    continue
                t_hit = ray_box(origin, aim, lo, hi)
            else:
                center = (float(tgt["x"]), float(tgt["y"]),
                          float(tgt.get("z", 0.0)))
                t_hit = ray_sphere_t(origin, aim, center,
                                     float(tgt.get("radius", 0.5)))
        except (KeyError, TypeError, ValueError):
            continue  # malformed target: skip, never sink the shot
        if t_hit is None:
            continue
        if best_t is None or t_hit < best_t:
            best_t, best_id = t_hit, tgt.get("id")
    if best_t is None or best_t > max_range:
        return None
    return {
        "target_id": best_id,
        "range": best_t,
        "hit_pos": [ox + aim[0] * best_t,
                    oy + aim[1] * best_t,
                    oz + aim[2] * best_t],
    }


class SharedBody:
    """Lock-protected seam between the TCP server threads and the sim thread.

    Server threads ONLY touch this object (never the stage); the sim main
    thread reads the intent and republishes the served state after each
    world.step().
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Twist intent (written by server, read by sim thread)
        self.left = 0.0
        self.right = 0.0
        # Turret intent (stored + logged; a Go2 asset has no turret to drive)
        self.pan = 0.0
        self.tilt = 0.0
        # Fire mechanic: sphere/box targets + remaining ammo. In Isaac mode
        # the sim thread republishes target poses read from the stage each
        # step; without Isaac the {cmd: "targets"} request seeds them directly.
        self.targets: list[dict] = []
        # Terrain the round terminates against, registered at BOOT and never
        # touched by {cmd: "targets"} or the per-step Isaac republish — both
        # rewrite self.targets wholesale, and the ground must survive that.
        self.terrain: list[dict] = [{
            "id": GROUND_TARGET["id"],
            "min": list(GROUND_TARGET["min"]),
            "max": list(GROUND_TARGET["max"]),
        }]
        self.ammo = MAGAZINE_SIZE
        # Served state (written by sim thread after each step)
        self.sim_time = 0.0
        self.state: dict = {
            "x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0,
            "gait": "stand", "stride_hz": 0.0, "phase": 0.0,
            "footfalls": list(_LEGS), "battery": 1.0, "odometer": 0.0,
            "roll": 0.0, "pitch": 0.0, "accel_z": 9.81,
        }


def handle_request(obj: object, shared: SharedBody) -> dict:
    """One protocol request -> one reply dict. Pure function over SharedBody.

    Protocol (matches examples/robot-template/hardware/isaac.py exactly):
        ping   -> {"ok": true, "physics": "isaac", "sim_time": <s>}
        state  -> {"ok": true, x, y, heading, speed, gait, stride_hz, phase,
                   footfalls, battery, odometer, roll, pitch, accel_z,
                   sim_time, physics}
        twist  -> {"ok": true}   (left/right clamped to [-1, 1], stored)
        turret -> {"ok": true}   (pan/tilt stored + logged)
        targets-> {"ok": true, "count": n}  (target list replaced; entries
                  are spheres {x, y, z, radius} or boxes {min, max}; in
                  Isaac mode the sim thread overwrites the list every step
                  with poses read back from the stage.  The ground terrain
                  is NOT part of this list and survives every replace.)
        fire   -> {"ok": true, "fired": bool, "hit": bool, "terrain": bool,
                   "hit_pos": [x, y, z]|null, "target_id": str|null,
                   "range": m|null, "origin": [x, y, z], "aim": [x, y, z],
                   "ammo": n}  (+"reason": "out_of_ammo" when dry)
                  Ray from the turret muzzle (muzzle_pose) against the
                  current targets PLUS the registered terrain (ray_hit);
                  the nearest hit wins, so a round aimed below the horizon
                  stops at the ground ("terrain": true) instead of passing
                  through it to reach something buried beyond.  Superset of
                  the original {"ok": true} — old brains that only check
                  "ok" keep working.
    """
    if not isinstance(obj, dict):
        return {"ok": False, "error": "request is not a JSON object"}
    cmd = obj.get("cmd")
    if cmd == "ping":
        with shared.lock:
            return {"ok": True, "physics": "isaac",
                    "sim_time": round(shared.sim_time, 4)}
    if cmd == "state":
        with shared.lock:
            reply = {"ok": True, "physics": "isaac",
                     "sim_time": round(shared.sim_time, 4)}
            reply.update(shared.state)
            return reply
    if cmd == "twist":
        try:
            left = float(obj.get("left", 0.0))
            right = float(obj.get("right", 0.0))
        except (TypeError, ValueError):
            return {"ok": False, "error": "twist left/right must be numbers"}
        with shared.lock:
            shared.left = max(-1.0, min(1.0, left))
            shared.right = max(-1.0, min(1.0, right))
        return {"ok": True}
    if cmd == "turret":
        try:
            pan = float(obj.get("pan", 0.0))
            tilt = float(obj.get("tilt", 0.0))
        except (TypeError, ValueError):
            return {"ok": False, "error": "turret pan/tilt must be numbers"}
        with shared.lock:
            shared.pan = pan
            shared.tilt = tilt
        print(f"[ISAAC DOG] turret pan={pan:.1f} tilt={tilt:.1f}")
        return {"ok": True}
    if cmd == "targets":
        raw = obj.get("targets")
        if not isinstance(raw, list):
            return {"ok": False, "error": "targets must be a list"}
        cleaned: list[dict] = []
        for entry in raw:
            if not isinstance(entry, dict):
                return {"ok": False, "error": "each target must be an object"}
            if "min" in entry or "max" in entry:
                # Box target (a wall, a vehicle, an extra terrain slab).
                try:
                    lo = [float(v) for v in entry["min"]]
                    hi = [float(v) for v in entry["max"]]
                except (KeyError, TypeError, ValueError):
                    return {"ok": False,
                            "error": "box target min/max must be "
                                     "3-number lists"}
                if len(lo) != 3 or len(hi) != 3 or any(
                        a > b for a, b in zip(lo, hi)):
                    return {"ok": False,
                            "error": "box target min/max must be 3-number "
                                     "lists with min <= max on every axis"}
                cleaned.append({
                    "id": str(entry.get("id", f"tgt_{len(cleaned)}")),
                    "min": lo,
                    "max": hi,
                })
                continue
            try:
                cleaned.append({
                    "id": str(entry.get("id", f"tgt_{len(cleaned)}")),
                    "x": float(entry["x"]),
                    "y": float(entry["y"]),
                    "z": float(entry.get("z", 0.0)),
                    "radius": float(entry.get("radius", 0.5)),
                })
            except (KeyError, TypeError, ValueError):
                return {"ok": False,
                        "error": "target x/y (and optional z/radius) "
                                 "must be numbers"}
        with shared.lock:
            shared.targets = cleaned
        return {"ok": True, "count": len(cleaned)}
    if cmd == "fire":
        with shared.lock:
            if shared.ammo <= 0:
                print("[ISAAC DOG] fire: out of ammo")
                return {"ok": True, "fired": False, "hit": False,
                        "terrain": False, "hit_pos": None, "target_id": None,
                        "range": None, "ammo": 0, "reason": "out_of_ammo"}
            shared.ammo -= 1
            ammo = shared.ammo
            state = shared.state
            body_x = float(state.get("x", 0.0))
            body_y = float(state.get("y", 0.0))
            heading = float(state.get("heading", 0.0))
            pan, tilt = shared.pan, shared.tilt
            targets = list(shared.targets)
            terrain = [dict(t) for t in shared.terrain]
        origin, aim = muzzle_pose(body_x, body_y, heading, pan, tilt)
        # One resolver over the MIXED set: the round stops in the first
        # thing it enters, and the ground is one of those things.
        hit = ray_hit(origin, aim, targets + terrain)
        terrain_ids = {t.get("id") for t in terrain}
        terrain_hit = hit is not None and hit["target_id"] in terrain_ids
        if terrain_hit:
            print(f"[ISAAC DOG] fire: TERRAIN {hit['target_id']} "
                  f"at {hit['range']:.2f} m (ammo {ammo})")
        elif hit is not None:
            print(f"[ISAAC DOG] fire: HIT {hit['target_id']} "
                  f"at {hit['range']:.2f} m (ammo {ammo})")
        else:
            print(f"[ISAAC DOG] fire: miss (ammo {ammo})")
        return {
            "ok": True,
            "fired": True,
            "hit": hit is not None,
            "terrain": terrain_hit,
            "hit_pos": ([round(v, 4) for v in hit["hit_pos"]]
                        if hit else None),
            "target_id": hit["target_id"] if hit else None,
            "range": round(hit["range"], 3) if hit else None,
            "origin": [round(v, 4) for v in origin],
            "aim": [round(v, 6) for v in aim],
            "ammo": ammo,
        }
    return {"ok": False, "error": f"unknown cmd: {cmd!r}"}


class BodyServer:
    """Stdlib JSON-lines TCP server: one request line -> one response line.

    Accept loop runs in a daemon thread; each client gets its own handler
    thread, so the brain client can reconnect at will (and multiple
    sequential connections just work). Handler threads touch ONLY the
    lock-protected SharedBody — never the Isaac stage.
    """

    def __init__(self, host: str, port: int, shared: SharedBody) -> None:
        self._host = host
        self._port = port
        self._shared = shared
        self._sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Live client sockets, so close() can force-drop blocked readers.
        self._clients_lock = threading.Lock()
        self._clients: set[socket.socket] = set()

    def start(self) -> int:
        """Bind + listen + spawn the accept loop. Returns the bound port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(4)
        sock.settimeout(0.5)  # so the accept loop can notice shutdown
        self._sock = sock
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="body-server-accept", daemon=True)
        self._accept_thread.start()
        return sock.getsockname()[1]

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        # Force-drop live clients: conn.close() alone does NOT close the fd
        # while a makefile() reader holds an io-ref — shutdown(SHUT_RDWR)
        # first so blocked readline()s (ours and the client's) observe EOF.
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
            self._accept_thread = None

    def _accept_loop(self) -> None:
        while not self._stop.is_set() and self._sock is not None:
            try:
                client, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during shutdown
            threading.Thread(
                target=self._serve_client, args=(client, addr),
                name=f"body-server-client-{addr[1]}", daemon=True).start()

    def _serve_client(self, client: socket.socket, addr: tuple) -> None:
        print(f"[ISAAC DOG] client connected {addr[0]}:{addr[1]}")
        with self._clients_lock:
            self._clients.add(client)
        try:
            client.settimeout(60.0)
            rfile = client.makefile("r", encoding="utf-8")
            while not self._stop.is_set():
                try:
                    line = rfile.readline()
                except (socket.timeout, OSError, ValueError):
                    break
                if not line:
                    break  # client closed
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    reply = handle_request(obj, self._shared)
                except json.JSONDecodeError:
                    reply = {"ok": False, "error": "invalid json"}
                except Exception as e:  # protocol layer must never die
                    reply = {"ok": False, "error": f"internal: {e}"}
                try:
                    client.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                except OSError:
                    break
        finally:
            with self._clients_lock:
                self._clients.discard(client)
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
            print(f"[ISAAC DOG] client disconnected {addr[0]}:{addr[1]}")


# =========================================================================
# Isaac Sim section — only touched by run_isaac(); everything above runs
# without an Isaac install.
# =========================================================================

# Default folder for the optional omniverse-mcp Kit extension (--enable-mcp).
_MCP_EXT_FOLDER = "~/Code/omniverse-mcp/extension"
_MCP_EXT_NAME = "isaacsim.mcp.bridge"

# Candidate Go2 USD paths under the cloud assets root, most likely first.
_GO2_CANDIDATES = (
    "/Isaac/Robots/Unitree/Go2/go2.usd",
    "/Isaac/Robots/Unitree/go2.usd",
)

# How long the cloud assets-root lookup may take before we give up and go
# procedural. Asset hunting must never sink the run.
_ASSET_TIMEOUT_S = 20.0

# Procedural dog dimensions (Go2-class silhouette).
_TORSO_SIZE = (0.60, 0.25, 0.15)   # m, X (length) x Y (width) x Z (height)
_LEG_LEN = 0.28                    # m
_HIP_SWING_DEG = 20.0              # visual trot swing amplitude


def _resolve_assets_root_timeboxed(timeout_s: float) -> str | None:
    """get_assets_root_path() in a worker thread with a hard join timeout.

    The nucleus lookup can stall on a bad network; the daemon worker is
    abandoned if it overruns — falling through to the procedural body.
    """
    result: list[str | None] = [None]

    def _lookup() -> None:
        try:
            from isaacsim.storage.native import get_assets_root_path
            result[0] = get_assets_root_path()
        except Exception as e:
            print(f"[ISAAC DOG] assets root lookup failed: {e}")

    worker = threading.Thread(target=_lookup, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        print(f"[ISAAC DOG] assets root lookup timed out after {timeout_s:.0f}s")
        return None
    return result[0]


def _try_load_go2(simulation_app) -> bool:
    """Best-effort Unitree Go2 reference under /World/Dog. False on ANY miss.

    Note: an unactuated articulation will sag under gravity — the Go2 path is
    a visual upgrade only. The procedural body is the deliberate, always-works
    default; this never blocks it.
    """
    try:
        assets_root = _resolve_assets_root_timeboxed(_ASSET_TIMEOUT_S)
        if not assets_root:
            return False
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import is_file
        for rel in _GO2_CANDIDATES:
            usd_path = assets_root + rel
            try:
                if not is_file(usd_path):
                    continue
            except Exception:
                continue  # stat failed: treat as missing, try next candidate
            try:
                prim = add_reference_to_stage(usd_path=usd_path,
                                              prim_path="/World/Dog")
                simulation_app.update()
                if prim is not None and prim.IsValid() and prim.GetChildren():
                    print(f"[ISAAC DOG] loaded Go2 asset: {usd_path}")
                    return True
            except Exception as e:
                print(f"[ISAAC DOG] Go2 reference failed ({usd_path}): {e}")
        return False
    except Exception as e:
        print(f"[ISAAC DOG] Go2 asset path unavailable: {e}")
        return False


def _build_procedural_dog() -> dict:
    """Small kinematic dog from prims: torso box + 4 leg boxes on hip pivots.

    Layout (all under the /World/Dog root Xform that the main loop moves):
        /World/Dog                torso-frame root (kinematic base motion)
        /World/Dog/torso          DynamicCuboid, KINEMATIC rigid body (cyan)
        /World/Dog/hip_FL..RR     Xform pivots at the torso corners
        /World/Dog/hip_*/shank    VisualCuboid legs (animated hip pitch)

    Returns {"hips": {leg: SingleXFormPrim}, "hip_offsets": {leg: (x,y,z)}}.
    """
    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.prims import define_prim, get_prim_at_path

    body_h = PROFILE_DEFAULTS["body_height_m"]
    tl, tw, th = _TORSO_SIZE

    define_prim("/World/Dog", "Xform")

    # Torso: kinematic rigid body — physics tracks it, the transform drives it.
    DynamicCuboid(
        prim_path="/World/Dog/torso",
        name="dog_torso",
        translation=np.array([0.0, 0.0, body_h]),
        scale=np.array([tl, tw, th]),
        size=1.0,
        color=np.array([0.0, 0.94, 1.0]),  # tritium cyan — unmistakably ours
    )
    try:
        from pxr import UsdPhysics
        rb = UsdPhysics.RigidBodyAPI(get_prim_at_path("/World/Dog/torso"))
        rb.CreateKinematicEnabledAttr(True)
    except Exception as e:
        print(f"[ISAAC DOG] warning: could not mark torso kinematic: {e}")

    # Legs: hip pivot Xform at each torso corner, leg box hanging below it.
    # Hips sit just under the torso belly; legs reach near the ground.
    hip_z = body_h - th / 2.0 - 0.01
    hip_x = tl / 2.0 - 0.06
    hip_y = tw / 2.0 - 0.02
    hip_offsets = {
        "FL": (hip_x, hip_y, hip_z),
        "FR": (hip_x, -hip_y, hip_z),
        "RL": (-hip_x, hip_y, hip_z),
        "RR": (-hip_x, -hip_y, hip_z),
    }
    hips: dict[str, object] = {}
    for leg, offset in hip_offsets.items():
        hip_path = f"/World/Dog/hip_{leg}"
        define_prim(hip_path, "Xform")
        hips[leg] = SingleXFormPrim(hip_path, name=f"dog_hip_{leg}",
                                    translation=np.array(offset))
        VisualCuboid(
            prim_path=f"{hip_path}/shank",
            name=f"dog_leg_{leg}",
            translation=np.array([0.0, 0.0, -_LEG_LEN / 2.0]),
            scale=np.array([0.06, 0.05, _LEG_LEN]),
            size=1.0,
            color=np.array([0.15, 0.15, 0.18]),  # dark leg vs cyan torso
        )
    # A small "head" so heading is visually obvious (forward = +X local).
    VisualCuboid(
        prim_path="/World/Dog/head",
        name="dog_head",
        translation=np.array([tl / 2.0 + 0.05, 0.0, body_h + 0.04]),
        scale=np.array([0.12, 0.14, 0.10]),
        size=1.0,
        color=np.array([1.0, 0.16, 0.43]),  # magenta muzzle
    )
    return {"hips": hips, "hip_offsets": hip_offsets}


# Practice-range targets spawned under /World/Targets in Isaac mode. Real
# rigid spheres — they sit on the ground under actual physics (works under
# Newton or PhysX; no engine-specific raycast API is assumed). The fire ray
# is evaluated by the SAME pure ray_hit() the no-Isaac path uses, fed the
# poses read BACK from the stage each step — so a target that physics moved
# is hit where it actually is.
_RANGE_TARGET_SPECS = (
    {"id": "tgt_north", "pos": (0.0, 8.0, 0.4), "radius": 0.4},
    {"id": "tgt_east", "pos": (8.0, 0.0, 0.4), "radius": 0.4},
    {"id": "tgt_west", "pos": (-8.0, 3.0, 0.4), "radius": 0.4},
)


def _build_range_targets() -> dict[str, object]:
    """GUARDED (isaacsim only): spawn the practice-range target spheres.

    Returns {target_id: prim} for the main loop to read poses back from.
    Dynamic (real rigid bodies) preferred; visual spheres as fallback so a
    physics-API mismatch never sinks the server.
    """
    import numpy as np
    from isaacsim.core.utils.prims import define_prim

    define_prim("/World/Targets", "Xform")
    prims: dict[str, object] = {}
    for spec in _RANGE_TARGET_SPECS:
        path = f"/World/Targets/{spec['id']}"
        try:
            from isaacsim.core.api.objects import DynamicSphere
            prims[spec["id"]] = DynamicSphere(
                prim_path=path,
                name=spec["id"],
                position=np.array(spec["pos"]),
                radius=spec["radius"],
                color=np.array([0.99, 0.93, 0.04]),  # tritium yellow
            )
        except Exception as e:
            print(f"[ISAAC DOG] dynamic target failed ({spec['id']}): {e}; "
                  f"using visual sphere")
            from isaacsim.core.api.objects import VisualSphere
            prims[spec["id"]] = VisualSphere(
                prim_path=path,
                name=spec["id"],
                position=np.array(spec["pos"]),
                radius=spec["radius"],
                color=np.array([0.99, 0.93, 0.04]),
            )
    print(f"[ISAAC DOG] spawned {len(prims)} range targets under /World/Targets")
    return prims


def _sync_stage_targets(range_targets: dict[str, object],
                        shared: SharedBody) -> None:
    """GUARDED: republish live stage target poses for the fire ray.

    Runs on the sim thread each step; overwrites shared.targets with the
    poses physics actually produced. Radius comes from the spawn spec.
    """
    radius_by_id = {s["id"]: s["radius"] for s in _RANGE_TARGET_SPECS}
    published: list[dict] = []
    for tid, prim in range_targets.items():
        try:
            pos, _ = prim.get_world_pose()
        except Exception:
            continue  # prim mid-teardown: skip this tick
        published.append({
            "id": tid,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "radius": radius_by_id.get(tid, 0.4),
        })
    with shared.lock:
        shared.targets = published


def run_isaac(args: argparse.Namespace) -> int:
    """Boot Isaac headless, build the scene, and serve the body until stopped.

    Order matters (per the local standalone examples): SimulationApp FIRST,
    all other isaacsim imports after.
    """
    launch_config: dict = {"headless": True}
    if args.enable_mcp:
        import os
        mcp_folder = os.path.expanduser(_MCP_EXT_FOLDER)
        if os.path.isdir(mcp_folder):
            # SimulationApp passes extra_args straight to the kit process;
            # --ext-folder is the same flag the build's own warmup.sh uses.
            launch_config["extra_args"] = ["--ext-folder", mcp_folder]
        else:
            print(f"[ISAAC DOG] --enable-mcp: folder not found: {mcp_folder}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config)

    # --- everything below may import isaacsim.* (app is up) ---
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
    from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

    if args.enable_mcp and "extra_args" in launch_config:
        try:
            from isaacsim.core.utils.extensions import enable_extension
            if enable_extension(_MCP_EXT_NAME):
                print(f"[ISAAC DOG] MCP bridge extension enabled ({_MCP_EXT_NAME})")
            else:
                print("[ISAAC DOG] MCP bridge extension not loadable; continuing")
        except Exception as e:
            print(f"[ISAAC DOG] MCP bridge enable failed (non-fatal): {e}")

    physics_dt = 1.0 / args.physics_hz
    world = World(physics_dt=physics_dt, rendering_dt=physics_dt,
                  stage_units_in_meters=1.0)
    # Walkable surface at z = 0 — the plane GROUND_TARGET mirrors, so the
    # fire ray stops at the same ground the physics stands on.
    world.scene.add_default_ground_plane()

    # --- body: go2 (best-effort) or procedural (deliberate default) ---
    asset_used = "procedural"
    dog_parts: dict = {}
    if args.asset in ("auto", "go2"):
        if _try_load_go2(simulation_app):
            asset_used = "go2"
        elif args.asset == "go2":
            print("[ISAAC DOG] --asset go2 requested but unavailable; "
                  "falling back to procedural")
    if asset_used == "procedural":
        dog_parts = _build_procedural_dog()

    try:
        range_targets = _build_range_targets()
    except Exception as e:
        print(f"[ISAAC DOG] range targets unavailable (non-fatal): {e}")
        range_targets = {}

    world.reset()
    for _ in range(5):  # confirm the stage steps before claiming READY
        world.step(render=False)

    # Root Xform we drive; body prim we read back from. For the procedural
    # dog the read-back prim is the KINEMATIC RIGID torso (the physics view);
    # for a Go2 reference it is the asset root.
    root = SingleXFormPrim("/World/Dog", name="dog_root")
    if asset_used == "procedural":
        try:
            body_read = SingleRigidPrim("/World/Dog/torso", name="dog_torso_rb")
        except Exception as e:
            print(f"[ISAAC DOG] rigid view unavailable ({e}); "
                  f"reading torso via stage xform")
            body_read = SingleXFormPrim("/World/Dog/torso", name="dog_torso_xf")
    else:
        body_read = root
    body_z_offset = PROFILE_DEFAULTS["body_height_m"] if asset_used == "procedural" else 0.0

    shared = SharedBody()
    server = BodyServer(args.host, args.port, shared)
    port = server.start()

    stop_event = threading.Event()

    def _on_signal(signum, frame) -> None:
        print(f"[ISAAC DOG] signal {signum}; shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # The grep line the orchestrator waits for — world is stepping, socket up.
    print(f"ISAAC BODY SERVER READY port={port} asset={asset_used} physics=isaac",
          flush=True)

    integ = GaitIntegrator()
    hips = dog_parts.get("hips", {})
    hip_offsets = dog_parts.get("hip_offsets", {})
    # Trot-style visual swing: diagonal pairs in antiphase.
    swing_phase_off = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}

    next_tick = time.monotonic()
    try:
        while not stop_event.is_set():
            if args.duration > 0 and world.current_time >= args.duration:
                print(f"[ISAAC DOG] duration {args.duration:.0f}s reached")
                break

            # 1. Twist intent -> kinematic target pose.
            with shared.lock:
                left, right = shared.left, shared.right
            integ.step(physics_dt, left, right)
            roll_deg, pitch_deg, bob_m, accel_z = integ.oscillation()
            yaw_deg = 90.0 - integ.heading  # tritium heading -> isaac yaw
            quat = euler_angles_to_quat(
                np.array([roll_deg, pitch_deg, yaw_deg]), degrees=True)
            root.set_world_pose(
                position=np.array([integ.x, integ.y, bob_m]),
                orientation=quat)

            # 2. Visual gait: hip pitch swing on the procedural legs.
            if hips and integ.gait != "stand":
                for leg, hip in hips.items():
                    ph = (integ.phase + swing_phase_off[leg]) % 1.0
                    swing = _HIP_SWING_DEG * math.sin(2 * math.pi * ph)
                    hip.set_local_pose(
                        translation=np.array(hip_offsets[leg]),
                        orientation=euler_angles_to_quat(
                            np.array([0.0, swing, 0.0]), degrees=True))

            # 3. REAL physics step — this is the point of the exercise.
            world.step(render=False)

            # 4. Read the pose BACK from the sim body prim; serve THAT.
            pos, quat_rb = body_read.get_world_pose()
            e = quat_to_euler_angles(np.asarray(quat_rb), degrees=True)
            state = {
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "heading": round((90.0 - float(e[2])) % 360.0, 2),
                "speed": round(integ.speed, 3),
                "gait": integ.gait,
                "stride_hz": integ.stride_hz(),
                "phase": round(integ.phase, 3),
                "footfalls": footfalls(integ.gait, integ.phase),
                "battery": round(integ.battery, 5),
                "odometer": round(integ.odometer, 2),
                "roll": round(float(e[0]), 2),
                "pitch": round(float(e[1]), 2),
                "accel_z": round(accel_z, 2),
            }
            with shared.lock:
                shared.state = state
                shared.sim_time = float(world.current_time)

            # 4b. Republish live target poses so {cmd: "fire"} rays hit the
            #     spheres where physics actually put them.
            if range_targets:
                _sync_stage_targets(range_targets, shared)

            # 5. Real-time pacing: hold wall clock at physics_hz so a live
            #    brain client sees honest dynamics.
            next_tick += physics_dt
            now = time.monotonic()
            if next_tick > now:
                time.sleep(next_tick - now)
            elif next_tick < now - 1.0:
                next_tick = now  # fell far behind (asset load hitch): resync
    finally:
        server.close()
        simulation_app.close()
        print("[ISAAC DOG] clean shutdown complete")
    return 0


# =========================================================================
# Self-test (system python3, no Isaac) + CLI
# =========================================================================

def _selftest() -> int:
    """Integrator + protocol + loopback-socket checks WITHOUT isaacsim."""
    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1

    # --- integrator: straight trot north ---
    integ = GaitIntegrator()
    for _ in range(100):
        integ.step(0.02, 0.5, 0.5)          # forward=0.5 -> 1.5 m/s -> trot
    check("gait selects trot at forward=0.5", integ.gait == "trot")
    check("speed ramped to 1.5 m/s", abs(integ.speed - 1.5) < 1e-6)
    check("moved north (y>1m after 2s)", integ.y > 1.0)
    check("no east drift", abs(integ.x) < 1e-9)
    check("battery drained", 0.0 < integ.battery < 1.0)
    check("odometer accumulated", abs(integ.odometer - integ.y) < 1e-6)

    # --- integrator: pure turn = stand + heading change ---
    integ2 = GaitIntegrator()
    for _ in range(50):
        integ2.step(0.02, 0.5, -0.5)        # forward=0, turn=1.0
    check("pure turn stands", integ2.gait == "stand")
    check("heading 120deg after 1s at 120dps",
          abs(integ2.heading - 120.0) < 0.01)

    # --- footfall rules (the SC gait-diagram contract) ---
    check("trot first half FL+RR", footfalls("trot", 0.25) == ["FL", "RR"])
    check("trot second half FR+RL", footfalls("trot", 0.75) == ["FR", "RL"])
    check("bound front pair", footfalls("bound", 0.1) == ["FL", "FR"])
    check("walk 4-beat drops FL first",
          footfalls("walk", 0.1) == ["FR", "RL", "RR"])
    check("stand all planted", footfalls("stand", 0.0) == list(_LEGS))

    # --- fire mechanic: pure muzzle math + ray-sphere hit (no Isaac) ---
    _, aim = muzzle_pose(0.0, 0.0, 0.0, 90.0, 0.0)
    check("facing north, pan 90 aims east",
          abs(aim[0] - 1.0) < 1e-9 and abs(aim[1]) < 1e-9 and abs(aim[2]) < 1e-9)
    hit = ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                  [{"id": "far", "x": 0.0, "y": 20.0, "z": 1.0, "radius": 1.0},
                   {"id": "near", "x": 0.0, "y": 10.0, "z": 1.0, "radius": 1.0}])
    check("ray picks the NEAREST target in path",
          hit is not None and hit["target_id"] == "near"
          and abs(hit["range"] - 9.0) < 1e-9)
    check("ray misses targets behind the muzzle",
          ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                  [{"id": "b", "x": 0.0, "y": -5.0, "z": 1.0, "radius": 1.0}])
          is None)

    # --- terrain: the round stops at the ground, not 60 m below it ---
    ground = {"id": GROUND_TARGET["id"],
              "min": list(GROUND_TARGET["min"]),
              "max": list(GROUND_TARGET["max"])}
    down45 = (0.0, math.sqrt(0.5), -math.sqrt(0.5))
    thit = ray_hit((0.0, 0.0, 1.0), down45,
                   [{"id": "buried", "x": 0.0, "y": 2.0, "z": -1.0,
                     "radius": 0.5},
                    ground])
    check("round terminates at TERRAIN, not the target behind it",
          thit is not None and thit["target_id"] == GROUND_TARGET["id"]
          and abs(thit["hit_pos"][2]) < 1e-9)
    check("level fire above the slab still flies free",
          ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [ground]) is None)
    check("nearest hit wins across mixed sphere+box",
          ray_hit((0.0, 0.0, 1.0), down45,
                  [{"id": "near_sphere", "x": 0.0, "y": 0.5, "z": 0.5,
                    "radius": 0.3}, ground])["target_id"] == "near_sphere")

    # --- TCP loopback: protocol + sequential reconnect ---
    shared = SharedBody()
    with shared.lock:
        shared.sim_time = 1.23
        shared.state = dict(shared.state, x=2.5, y=7.5, gait="trot")
    server = BodyServer("127.0.0.1", 0, shared)
    port = server.start()

    def ask(sock_file, sock, obj) -> dict:
        sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        return json.loads(sock_file.readline())

    # Connection 1
    s1 = socket.create_connection(("127.0.0.1", port), timeout=2)
    f1 = s1.makefile("r", encoding="utf-8")
    r = ask(f1, s1, {"cmd": "ping"})
    check("ping ok + physics=isaac",
          r.get("ok") is True and r.get("physics") == "isaac"
          and abs(r.get("sim_time", 0) - 1.23) < 1e-6)
    r = ask(f1, s1, {"cmd": "twist", "left": 0.3, "right": 5.0})
    with shared.lock:
        stored = (shared.left, shared.right)
    check("twist stored + clamped", r.get("ok") is True and stored == (0.3, 1.0))
    r = ask(f1, s1, {"cmd": "state"})
    check("state serves sim-published dict",
          r.get("ok") is True and r.get("x") == 2.5 and r.get("y") == 7.5
          and r.get("gait") == "trot" and r.get("physics") == "isaac")
    r = ask(f1, s1, {"cmd": "warp_drive"})
    check("unknown cmd rejected", r.get("ok") is False and "error" in r)
    s1.sendall(b"this is not json\n")
    r = json.loads(f1.readline())
    check("bad json rejected, link survives", r.get("ok") is False)
    r = ask(f1, s1, {"cmd": "ping"})
    check("link still alive after bad json", r.get("ok") is True)
    f1.close()
    s1.close()

    # Connection 2 (the brain reconnects after a drop)
    s2 = socket.create_connection(("127.0.0.1", port), timeout=2)
    f2 = s2.makefile("r", encoding="utf-8")
    r = ask(f2, s2, {"cmd": "turret", "pan": 45.0, "tilt": -10.0})
    check("turret accepted on second connection", r.get("ok") is True)
    with shared.lock:
        check("turret stored", shared.pan == 45.0 and shared.tilt == -10.0)
    # Fire: aim straight ahead (body at 2.5/7.5 heading 0 = north) at a
    # plate 20 m downrange at muzzle height -> structured hit result.
    ask(f2, s2, {"cmd": "turret", "pan": 0.0, "tilt": 0.0})
    r = ask(f2, s2, {"cmd": "targets", "targets": [
        {"id": "plate", "x": 2.5, "y": 27.5, "z": 0.55, "radius": 0.5}]})
    check("targets accepted", r.get("ok") is True and r.get("count") == 1)
    r = ask(f2, s2, {"cmd": "fire"})
    check("fire hits the downrange plate",
          r.get("ok") is True and r.get("fired") is True
          and r.get("hit") is True and r.get("target_id") == "plate"
          and r.get("hit_pos") is not None
          and r.get("ammo") == MAGAZINE_SIZE - 1)
    ask(f2, s2, {"cmd": "turret", "pan": 90.0, "tilt": 0.0})
    r = ask(f2, s2, {"cmd": "fire"})
    check("fire aimed away misses (fired, no hit)",
          r.get("ok") is True and r.get("fired") is True
          and r.get("hit") is False and r.get("hit_pos") is None
          and r.get("target_id") is None
          and r.get("ammo") == MAGAZINE_SIZE - 2)
    # Aim below the horizon: the round must STOP at the ground plane
    # (z = 0), reported as a terrain stop — not fly under the world.
    ask(f2, s2, {"cmd": "turret", "pan": 0.0, "tilt": -45.0})
    r = ask(f2, s2, {"cmd": "fire"})
    check("fire below the horizon terminates at terrain",
          r.get("ok") is True and r.get("hit") is True
          and r.get("terrain") is True
          and r.get("target_id") == GROUND_TARGET["id"]
          and r.get("hit_pos") is not None
          and abs(r["hit_pos"][2]) < 1e-3
          and r.get("ammo") == MAGAZINE_SIZE - 3)
    # A replace of the target list must NOT delete the ground.
    ask(f2, s2, {"cmd": "targets", "targets": []})
    r = ask(f2, s2, {"cmd": "fire"})
    check("terrain survives a targets replace",
          r.get("hit") is True and r.get("terrain") is True)
    f2.close()
    s2.close()

    # Connection 3: still open when the server closes — the shutdown path
    # must force-drop it (SHUT_RDWR) so the client's readline sees EOF.
    s3 = socket.create_connection(("127.0.0.1", port), timeout=2)
    s3.settimeout(2.0)
    f3 = s3.makefile("r", encoding="utf-8")
    r = ask(f3, s3, {"cmd": "ping"})
    check("third connection alive pre-shutdown", r.get("ok") is True)
    server.close()
    try:
        eof = f3.readline()
        check("server close() drops live client (EOF)", eof == "")
    except (socket.timeout, OSError):
        check("server close() drops live client (EOF)", False)
    f3.close()
    s3.close()

    if failures:
        print(f"SELFTEST FAILED ({failures} checks)")
        return 1
    print("SELFTEST OK (integrator + footfalls + TCP loopback, no isaac import)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isaac Sim quadruped body server (JSON lines over TCP) "
                    "for the Tritium robot-template brain.")
    parser.add_argument("--port", type=int, default=18973,
                        help="TCP port to serve on (default 18973)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="bind address (default 127.0.0.1)")
    parser.add_argument("--asset", choices=("auto", "go2", "procedural"),
                        default="auto",
                        help="quadruped body: auto tries the cloud Go2 asset "
                             "then falls back to the procedural dog")
    parser.add_argument("--physics-hz", type=float, default=60.0,
                        help="physics step rate (default 60)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="sim seconds to run; 0 = until SIGINT/SIGTERM")
    parser.add_argument("--enable-mcp", action="store_true",
                        help="also load the omniverse-mcp Kit extension "
                             "(viewport tooling; failure is non-fatal)")
    parser.add_argument("--selftest", action="store_true",
                        help="run the no-GPU integrator/protocol self-test "
                             "under plain python3 and exit")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()
    return run_isaac(args)


if __name__ == "__main__":
    sys.exit(main())
