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
        fire   -> {"ok": true}   (logged)
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
    if cmd == "fire":
        print("[ISAAC DOG] fire")
        return {"ok": True}
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
    r = ask(f2, s2, {"cmd": "fire"})
    check("fire accepted", r.get("ok") is True)
    with shared.lock:
        check("turret stored", shared.pan == 45.0 and shared.tilt == -10.0)
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
