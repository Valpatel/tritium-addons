# connectors — Isaac-side runtime

These modules run on the **render host**, inside Isaac Sim's own Python 3.12.
They are the only place `isaacsim` / `pxr` is touched. Each keeps its Isaac
imports **lazy** (inside the function that boots the sim) so the module itself
imports under plain `python3` — that is what lets the [no-GPU tests](../../tests/)
exercise the transport and geometry chain in CI with no GPU.

**Dependency hygiene (enforced by a test):** nothing here imports `tritium`, and
nothing in `tritium-sc` imports anything here. The `Scene3D` JSON, the camera
MJPEG/MQTT frames, the LiDAR `/scan` JSON, and the robot TCP wire are the only
seams. (`isaac_camera_bridge`'s optional local detector resolves `tritium_lib`
by name at runtime, opt-in, only where it happens to be installed — there is no
static tritium import anywhere in this package.) The consumer mirror image —
code that runs *outside* Isaac and may import `tritium_lib` — is
[`../clients/`](../clients/).

## The connectors

| File | Purpose | No-GPU path |
|------|---------|-------------|
| `usd_scene_builder.py` | Neutral `Scene3D` JSON → a USD stage (terrain heightfield + extruded buildings + roads/water, per-kind `UsdPreviewSurface` materials). The Isaac-side writer of the map → 3D-twin pipeline. | `--validate` (assert well-formed), `--obj out.obj`, `--preview out.png` (matplotlib) |
| `render_city.py` | Headless Isaac render of a USD twin → PNG. The real-GPU proof the map → 3D-scene pipeline renders. | — (needs Isaac + GPU) |
| `camera_server.py` | Serves an Isaac camera as an **MJPEG IP camera** (`/mjpeg`, `/snapshot`, `/status`) — SC registers it like any security camera; nothing in SC knows it is Isaac. | `--source synthetic` (numpy/cv2 moving subject), `--selftest` |
| `isaac_camera_bridge.py` | Publishes a unit's onboard camera frames as JPEG on the `camera_feeds` MQTT topic `tritium/{site}/cameras/{cam_id}/frame`, keyed to the unit id; optional local `build_frame_detector` publishes detections on `.../detections`. Isaac-free — only `numpy` + JPEG + MQTT. | Imports & unit-tests headless (no Isaac); `isaac_camera_rgb()` lazy-imports Isaac only inside a live sim |
| `isaac_quadruped_server.py` | The physics **robot BODY** behind the brain/body TCP seam: owns an Isaac stage, steps physics, moves a quadruped from the same twist/gait contract as `robot-template`, serves body state back. Also speaks `turret` / `{cmd: "targets"}` / `fire` — hitscan against the registered targets **plus** the ground terrain, so a round below the horizon stops at the dirt. | `--selftest` (gait integrator + footfalls + TCP loopback + terrain fire, no Isaac) |
| `lidar_server.py` | Serves an Isaac RTX Lidar as a **JSON range server** (`GET /scan`, `/status`, port 8110) — the LaserScan-style document `tritium-edge`'s `SensorBridgeNode` ingests via `scan_url`. | `--source synthetic` (DEFAULT — analytic room + orbiting obstacle), `--selftest` |
| `newton_gait_driver.py` | Gait trajectory → per-joint USD drive targets for the Go2 under Newton: fixed-step scheduling, rad→deg, actuator clamping, `apply_to_stage`. Trajectory **and** attitude stabilizer arrive injected (`targets_fn` / `stabilize_fn`) — a consumer that skips `stabilize_fn` runs open-loop, the measured 71% arm rather than the 34/34 closed-loop one (see `../../examples/NEWTON-GAIT-FINDINGS.md`). A third hook, `reflex_fn`, exists for a stepping reflex — lib live-measured `StepReflex` **unfit to gate a walking gait** (baseline 6/6 upright vs reflex 0/5, Fisher p = 0.0022; the authoritative verdict is in `tritium_lib.control.step_reflex`) — a standing body or a future contact-triggered reflex are the only supported bindings. | `--selftest` (mock trajectory; no Isaac, no tritium) |
| `__init__.py` | Empty package marker. | — |

### Two ways an Isaac camera reaches Tritium

- **`camera_server.py`** — pull model: an HTTP MJPEG endpoint SC registers as an
  ordinary `mjpeg` camera source (posed on the map with lat/lng/heading/FOV).
- **`isaac_camera_bridge.py`** — push model: publishes frames on the
  `camera_feeds` MQTT topic keyed to a robot's unit id, so a robot's *onboard*
  view shows up as its feed (UI MJPEG + frame detection → TargetTracker).

## Run (render host — Isaac's Python, `OMNI_KIT_ACCEPT_EULA=YES`)

```bash
# Map area -> USD twin (fetches Scene3D straight from the live SC API over the LAN)
python isaac_sim_addon/connectors/usd_scene_builder.py \
    --scene-url "http://<sc-host>:8000/api/gis/scene3d?ao=dublin&roads=1&water=1" \
    --out dublin.usd

# Render it headless -> PNG
python isaac_sim_addon/connectors/render_city.py --usd dublin.usd --out dublin.png

# Serve a camera inside the twin (SC ingests as an mjpeg source)
python isaac_sim_addon/connectors/camera_server.py --source isaac --scene dublin.usd --port 8100

# Robot-dog body behind the TCP seam (brain = tritium-sc/examples/robot-template)
python isaac_sim_addon/connectors/isaac_quadruped_server.py --port 18973

# LiDAR as a JSON /scan server (synthetic default runs under plain python3)
python isaac_sim_addon/connectors/lidar_server.py --source isaac --port 8110
```

## Related

- [Isaac Sim addon overview](../../README.md) — full pipeline diagram + run recipes
- [Clients](../clients/) — the consumer tier (pose/nav/fire/teleop bridges)
- [Examples](../../examples/) (`go2_newton_gait.py`, `spot_policy_walk.py`, `robot_bridge.md`, `NEWTON-GAIT-FINDINGS.md`)
- [No-GPU tests](../../tests/) · [Package README](../README.md)
