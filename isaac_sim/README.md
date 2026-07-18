# Isaac Sim Connector

The Tritium ↔ NVIDIA Isaac Sim bridge. Renders real Tritium map areas as 3D
digital twins, drives simulated cameras, LiDARs, and physics robot bodies
inside them, and returns frames/scans/telemetry over the LAN — so the
perception, fusion, and autonomy stack is validated against render-quality
imagery and physics before any hardware ships. Rounds fired in the live sim
land on the operator's tactical map.

**This is an optional, heavy-GPU integration.** Tritium runs fully without it
(the sim engine, the classical detectors, the synthetic camera feeds all work
with zero Isaac). That is exactly why it is an **addon** and not part of
`tritium-lib` or `tritium-sc`.

## Why this is an addon (placement)

Everything here `import`s `isaacsim` / `pxr` — an 8 GB, x86, Python-3.12-pinned
runtime. Per the parent copper-roof rule
([`../../CLAUDE.md`](../../CLAUDE.md) → `docs/ARCHITECTURE.md`), heavy external
runtimes are quarantined in an addon so the invariants hold:

- **`tritium-lib`** stays framework-free and imports on the robot's Jetson brain
  — it owns the reusable geometry (`tritium_lib.geo.scene3d` → `Scene3D`) and
  perception primitives (`tritium_lib.perception`).
- **`tritium-sc`** stays a web app — it serves the neutral scene at
  `GET /api/gis/scene3d` and ingests any camera generically (`camera_feeds`).
- **`tritium-edge/ros2`** owns the on-robot ROS2 stack.
- **This addon** owns everything Isaac-side, plus the thin consumer clients
  that speak to it. The **`Scene3D` JSON, the camera MJPEG stream, the LiDAR
  `/scan` JSON, and the robot TCP seam are the neutral contracts** across the
  boundary; nothing here is imported by the SC process. The two tiers have
  opposite import rules: `connectors/` run inside Isaac's python and **never
  import tritium** (a test enforces it); `clients/` run anywhere else, may
  import `tritium_lib`, and never import `isaacsim`.

## Layout

```
isaac_sim/
  tritium_addon.toml                 # manifest (category: simulation, gpu: true)
  isaac_sim_addon/
    connectors/                      # Isaac-SIDE runtime — run on the render host
      usd_scene_builder.py           # Scene3D JSON -> USD stage (per-kind materials)
      render_city.py                 # headless render of a USD twin -> PNG
      camera_server.py               # Isaac camera as an MJPEG IP camera (:8100; depth/stereo channels)
      lidar_server.py                # Isaac RTX Lidar as a JSON /scan range server (:8110)
      isaac_camera_bridge.py         # robot onboard camera -> camera_feeds MQTT (JPEG + optional detections)
      isaac_quadruped_server.py      # robot-dog physics body behind the TCP seam (:18973; turret + fire)
      newton_gait_driver.py          # gait trajectory -> USD joint-drive targets (Go2, Newton)
    clients/                         # consumer side — run anywhere, may import tritium_lib
      pose_bridge.py                 # live body pose -> POST /api/sighting (track on the map)
      nav_bridge.py                  # stage obstacles -> costmap -> A* route
      fire_bridge.py                 # live hitscan trial; shots -> POST /api/engagement/shot
      teleop_send.py                 # scripted / gamepad twist -> UDP live driving
  examples/                          # run recipes + live-kit findings
      README.md                      # per-example recipes
      robot_bridge.md                # robot-dog brain/body run recipe + protocol
      NEWTON-GAIT-FINDINGS.md        # dated findings from the live Newton kit
      go2_newton_gait.py             # lib gait -> live Newton Go2 (stabilizer arms, routes, teleop)
      go2_newton_stand.usd           # Newton-physics Go2 stand scene (loads under physics)
      spot_policy_walk.py            # PhysX velocity-command RL walk runner (SIL twist seam)
      smoke_boot.py                  # first-run Isaac launch validator (60 steps)
      smoke_detect.py                # no-GPU camera->detector->track proof
      ...                            # walk-to-map, sensor rig, probes — see examples/README.md
  tests/                             # no-GPU gates (15 files; 332 passed / 1 skipped)
```

The robot-body connector (`isaac_quadruped_server.py`) is the physics half of
the robot-dog brain/body seam: the `tritium-sc/examples/robot-template` brain
runs unchanged over a JSON-lines TCP wire to this Isaac body. It moved here
from `tritium-sc/examples/isaac-bridge` (see that repo's `examples/
ISAAC-MOVED.md`); its on-robot twin lives in `tritium-edge/ros2/
tritium_quadruped`. Run recipe + protocol: [`examples/robot_bridge.md`](examples/robot_bridge.md).

## The pipeline (map → 3D twin → perception)

```
tritium-lib                    tritium-sc                 this addon (render host, Isaac Python)
 geo.scene3d.build_scene3d  →  GET /api/gis/scene3d   →   connectors/usd_scene_builder.py  → dublin.usd
 (DEM+buildings+roads+water)   (serves Scene3D JSON)      connectors/render_city.py        → dublin.png
                                                          connectors/camera_server.py --scene dublin.usd
                                    ▲  MJPEG over LAN            │
 tritium-sc camera_feeds  ◄────────────────────────────────────┘
   FrameDetectionManager → det_* tracks on the tactical map
```

## Beyond the camera

**LiDAR.** `connectors/lidar_server.py` serves an Isaac RTX Lidar (or a
deterministic synthetic room, the default) as a LaserScan-style JSON document
at `GET /scan` on port 8110 — the same shape `tritium-edge`'s
`SensorBridgeNode` already ingests via `scan_url`, so an Isaac LiDAR plugs
into the robot brain like any bench LiDAR. `--selftest` proves the geometry
and transport with no GPU.

**Newton locomotion.** `connectors/newton_gait_driver.py` is the seam between
tritium-lib's gait trajectory (12 named joint angles in radians per control
tick) and the Go2's USD angular drives (degrees): scheduling, rad→deg,
actuator-envelope clamping, and — only inside Isaac's python — writing
`targetPosition` on the RevoluteJoint prims. Both the trajectory and the
attitude stabilizer arrive **injected** so the module stays tritium-free. The
injection is not optional in practice: on the live kit the open-loop gait
table stayed upright in 17/24 trials (71%) while the closed attitude loop
went 34/34 (100%) — a consumer that skips `stabilize_fn` inherits the 71%
number. Those trials ran on the live RTX 4090 Newton kit and are recorded in
[`examples/NEWTON-GAIT-FINDINGS.md`](examples/NEWTON-GAIT-FINDINGS.md) (tick
19); they are not re-runnable in CI — the no-GPU tests cover the
scheduling/clamp/trim contract, not the physics outcome.

**Live fire reaches the operator.** `clients/fire_bridge.py` fires a hitscan
round from a live Isaac body and grades it against target poses read **back**
off the stage — the geometry is `tritium_lib.geo.hitscan`, the same call a
real turret makes, and the stage terrain is resolved in the same call so a
round stops at the ground instead of passing under it. Every resolved shot is
reported to SC's `POST /api/engagement/shot` (tracer, impact, kill feed,
announcer). Reporting is **on by default** (`--sc-url`, env `TRITIUM_SC_URL`,
default `http://localhost:8000`); `--no-sc` switches it off. It is
fire-and-forget by contract: every failure is counted, logged once, and
swallowed, and after 3 consecutive failures the poster opens its circuit and
stops attempting for the rest of the run — a run with no SC loses nothing but
the audience. The quadruped server's own TCP `fire` path applies the same
terrain rule: its registered ground terrain is resolved alongside whatever
`{cmd: "targets"}` supplies, and survives a targets replace.

## Run (on the render host — GPUs run Isaac; ollama stays on GB10)

```bash
# One-time: Isaac Sim 6.0 needs Python 3.12
python3.12 -m venv ~/isaac_venv && ~/isaac_venv/bin/pip install \
    "isaacsim[all]==6.0.1.0" --extra-index-url https://pypi.nvidia.com

# Build a USD twin straight from the live Tritium map API (over the LAN)
OMNI_KIT_ACCEPT_EULA=YES ~/isaac_venv/bin/python \
    isaac_sim_addon/connectors/usd_scene_builder.py \
    --scene-url "http://<sc-host>:8000/api/gis/scene3d?bbox=-121.912,37.704,-121.880,37.728&ao=dublin&roads=1&water=1" \
    --out dublin.usd

# Render the twin headless -> PNG
OMNI_KIT_ACCEPT_EULA=YES ~/isaac_venv/bin/python \
    isaac_sim_addon/connectors/render_city.py --usd dublin.usd --out dublin.png

# Or serve a camera inside the twin; SC ingests it as an ordinary mjpeg source
OMNI_KIT_ACCEPT_EULA=YES ~/isaac_venv/bin/python \
    isaac_sim_addon/connectors/camera_server.py --source isaac --scene dublin.usd --port 8100

# Or a LiDAR (no-GPU synthetic default; --source isaac for the RTX Lidar)
python3 isaac_sim_addon/connectors/lidar_server.py --port 8110
```

`OMNI_KIT_ACCEPT_EULA=YES` accepts the Omniverse EULA non-interactively (first
run compiles RTX shaders — allow a few minutes).

## No-GPU gates

The `--validate` / `--obj` / `--preview` / `--source synthetic` / `--selftest`
paths all run under plain `python3` with **no Isaac and no GPU**, so the whole
transport and geometry chain is testable in CI. The suite is 15 files, **332
passed / 1 skipped** (the skip is an MQTT broker round-trip that only runs
when a broker is reachable), including the hygiene gate that asserts no
connector imports tritium. See [`tests/`](tests/).
