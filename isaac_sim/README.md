# Isaac Sim Connector

The Tritium ↔ NVIDIA Isaac Sim bridge. Renders real Tritium map areas as 3D
digital twins, drives simulated cameras and robot bodies inside them, and
returns frames/telemetry over the LAN — so the perception, fusion, and
autonomy stack is validated against render-quality imagery and physics before
any hardware ships.

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
- **This addon** owns everything Isaac-side. The **`Scene3D` JSON, the camera
  MJPEG stream, and the robot TCP seam are the neutral contracts** across the
  boundary; nothing here is imported by the SC process, and nothing here imports
  tritium internals.

## Layout

```
isaac_sim/
  tritium_addon.toml                 # manifest (category: simulation, gpu: true)
  isaac_sim_addon/
    connectors/                      # Isaac-SIDE runtime — run on the render host
      usd_scene_builder.py           # Scene3D JSON -> USD stage (per-kind materials)
      render_city.py                 # headless render of a USD twin -> PNG
      camera_server.py               # Isaac camera as an MJPEG IP camera
      isaac_camera_bridge.py         # robot onboard camera -> camera_feeds MQTT (JPEG + optional detections)
      isaac_quadruped_server.py      # robot-dog physics body behind the TCP seam
  examples/                          # run recipes
      README.md                      # scene/camera recipes
      robot_bridge.md                # robot-dog brain/body run recipe + protocol
      go2_newton_stand.usd           # Newton-physics Go2 stand scene (loads under physics)
      spot_policy_walk.py            # PhysX velocity-command RL walk runner (SIL twist seam)
      smoke_boot.py                  # first-run Isaac launch validator (60 steps)
      smoke_detect.py                # no-GPU camera->detector->track proof
  tests/                             # no-GPU gates (validate / OBJ / import guards)
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
```

`OMNI_KIT_ACCEPT_EULA=YES` accepts the Omniverse EULA non-interactively (first
run compiles RTX shaders — allow a few minutes).

## No-GPU gates

The `--validate` / `--obj` / `--preview` / `--source synthetic` / `--selftest`
paths all run under plain `python3` with **no Isaac and no GPU**, so the whole
transport and geometry chain is testable in CI. See `tests/`.
