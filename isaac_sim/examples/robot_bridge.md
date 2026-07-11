# Isaac Sim Quadruped Bridge

An NVIDIA Isaac Sim physics body behind the SAME MQTT contract the kinematic
robot dog already proved. The robot-template brain (navigator, turret,
telemetry, MQTT) runs completely unchanged — only the body under it gets
heavier: real physics stepping in Isaac Sim instead of the pure-python gait
engine in `tritium-sc/examples/robot-template/hardware/quadruped.py` (whose
canonical core now lives in `tritium-edge/ros2/tritium_quadruped/`; the
example keeps an embedded fallback copy in `hardware/gait_embedded.py`).

**Placement:** the body server (`isaac_quadruped_server.py`) `import`s
`isaacsim` and lives in **this addon** (`isaac_sim/isaac_sim_addon/connectors/`)
per the copper-roof rule — heavy external runtimes never sit in `tritium-sc`.
The brain-side TCP client (`hardware/isaac.py`) is a lightweight, isaacsim-free
example that stays in `tritium-sc/examples/robot-template`. The two ends meet
only over the neutral JSON-lines TCP seam.

Both North Star halves: FUN — a physics dog you dispatch and fire from the
tactical map (UX Loop 3, Add Robot). PRODUCTION — validates the brain/body
seam a real Unitree-class dog will use: the brain process is the Jetson side,
the body server is the locomotion controller, and the wire between them is
the part that must survive contact with reality.

## Architecture

```
 SC tactical map / combat HUD
        ^ REST / WebSocket
        |
 +------+--------+     MQTT (telemetry out, fire-control commands in)
 | tritium-sc    |<------------------+
 +---------------+                   |
                                     v
                  +------------------+-------------------+
                  | tritium-sc/examples/robot-template/  |  "the brain"
                  |   robot.py + config-isaac.yaml       |  (Jetson side)
                  |   navigator / turret / telemetry     |
                  |   hardware/isaac.py  (TCP client)    |  (isaacsim-free)
                  +------------------+-------------------+
                                     | JSON lines over TCP :18973
                                     | ping / state / twist / turret / fire
                  +------------------v-------------------+
                  | tritium-addons/isaac_sim/            |  "the body"
                  |   connectors/isaac_quadruped_server  |  (imports isaacsim)
                  |   SimulationApp (headless)           |
                  |   World.step(render=False) @ 60 Hz   |
                  |   kinematic torso + visual trot legs |
                  |   state read BACK from the body prim |
                  +--------------------------------------+
                        NVIDIA Isaac Sim v6 (local build)
```

The gait contract (walk 0.7 m/s @ 1.6 Hz, trot 1.6 @ 2.6, bound 3.0 @ 3.2,
120 deg/s turn, 2.5 m/s^2 accel limit, 155 Wh pack, footfall stance rules) is
the same wire contract as `tritium_lib.models.quadruped` and
`hardware/quadruped.py` — SC's gait diagram renders both bodies identically.

## Run recipe

Four terminals (or the real-broker harness), in order:

```bash
# 1. MQTT broker (e.g. the harness container from scripts/real_broker_harness.py)
mosquitto -p 1883

# 2. SC server with MQTT ingest on
cd tritium-sc && TRITIUM_MQTT_ENABLED=true ./start.sh

# 3. The Isaac body server — Isaac's bundled python, NOT the system python
#    (run from the tritium-addons repo root)
~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
    isaac_sim/isaac_sim_addon/connectors/isaac_quadruped_server.py --port 18973
# wait for:  ISAAC BODY SERVER READY port=18973 asset=... physics=isaac

# 4. The brain — same robot.py as every other body (in tritium-sc)
cd tritium-sc/examples/robot-template && python robot.py --config config-isaac.yaml

# 5. Prove it over the real wire (REST-only assertions, no shortcuts)
python tritium-sc/scripts/robot_dog_wire_proof.py
```

First-run validation order (no GPU contention): run `smoke_boot.py` first —
it boots headless, steps 60 physics steps, prints `SMOKE OK steps=60
sim_time=...` and exits 0. Then start the full body server.

No-GPU self-test (system python3; exercises the integrator, footfall rules,
protocol handler, and TCP loopback without importing isaacsim) — from the
tritium-addons repo root:

```bash
python3 isaac_sim/isaac_sim_addon/connectors/isaac_quadruped_server.py --selftest
```

## Protocol (JSON lines over TCP, one request line -> one response line)

| Request                                     | Response                                                                 |
|---------------------------------------------|--------------------------------------------------------------------------|
| `{"cmd":"ping"}`                            | `{"ok":true,"physics":"isaac","sim_time":<s>}`                           |
| `{"cmd":"state"}`                           | `{"ok":true,"x","y","heading","speed","gait","stride_hz","phase","footfalls":[...],"battery","odometer","roll","pitch","accel_z","sim_time","physics":"isaac"}` |
| `{"cmd":"twist","left":-1..1,"right":-1..1}`| `{"ok":true}` (clamped, stored; forward=(l+r)/2, turn=l-r)               |
| `{"cmd":"turret","pan":deg,"tilt":deg}`     | `{"ok":true}` (stored + logged)                                          |
| `{"cmd":"fire"}`                            | `{"ok":true}` (logged)                                                   |
| anything else / bad JSON                    | `{"ok":false,"error":"..."}`                                             |

Server flags: `--port 18973 --host 127.0.0.1 --asset {auto,go2,procedural}
--physics-hz 60 --duration 0 --enable-mcp --selftest`. `--asset auto` tries
the cloud Unitree Go2 asset (timeboxed) and always falls back to the
procedural prim dog; `--duration 0` runs until SIGINT/SIGTERM with a clean
`simulation_app.close()`. `--enable-mcp` additionally loads the local
omniverse-mcp Kit extension (viewport tooling for the orchestrator);
its failure is non-fatal.

State authority: the base pose is driven kinematically from the twist
integrator, but the served `x/y/heading/roll/pitch` are READ BACK from the
sim body prim's world transform after each `world.step()` — position comes
from the stage/physics view, not from integrator variables.

## Placement rationale

The body server lives in `isaac_sim/isaac_sim_addon/connectors/` (this addon)
because it `import`s `isaacsim` — an 8 GB, x86, Python-3.12-pinned runtime.
Per the parent copper-roof rule (`tritium-addons/CLAUDE.md` → parent
`docs/ARCHITECTURE.md`), heavy external runtimes are quarantined in an addon
so `tritium-sc` stays a web app and `tritium-lib` stays framework-free:

- The **body** (`isaac_quadruped_server.py`, imports isaacsim) → **addon**.
- The **brain-side TCP client** (`tritium-sc/examples/robot-template/hardware/
  isaac.py`, isaacsim-free) → stays in the operator repo beside the other
  robot-template bodies. The neutral JSON-lines TCP seam is the only coupling.
- Its **on-robot twin** is `tritium-edge/ros2/tritium_quadruped` (the body
  server becomes a ROS2 locomotion node on the Jetson).

Earlier this file lived in `tritium-sc/examples/isaac-bridge/`; it moved here
once the `isaac_sim` addon existed as the correct home for all Isaac-side code
(see `tritium-sc/examples/ISAAC-MOVED.md`).

## Jetson mapping (sim-to-real)

| Piece                            | Simulation (this example)                | Real dog                                        |
|----------------------------------|------------------------------------------|-------------------------------------------------|
| Brain process                    | `robot.py --config config-isaac.yaml`    | Same code on the Jetson Orin                    |
| Body server                      | `isaac_quadruped_server.py` (Isaac)      | Go2 SDK bridge / Isaac ROS locomotion stack     |
| Wire                             | TCP JSON lines on localhost              | Same protocol on the robot's internal network   |
| Physics                          | Isaac `World.step()`                     | The planet                                      |

Graphling boundary note: everything here is a stand-in body + stand-in
driver. The brain's embodiment slot stays clean — a Graphling checks into
the same seam without either side changing.
