# connectors — Isaac-side runtime

These modules run on the **render host**, inside Isaac Sim's own Python 3.12.
They are the only place `isaacsim` / `pxr` is touched. Each keeps its Isaac
imports **lazy** (inside the function that boots the sim) so the module itself
imports under plain `python3` — that is what lets the [no-GPU tests](../../tests/)
exercise the transport and geometry chain in CI with no GPU.

**Dependency hygiene (enforced by a test):** nothing here imports `tritium`, and
nothing in `tritium-sc` imports anything here. The `Scene3D` JSON, the camera
MJPEG/MQTT frames, and the robot TCP wire are the only seams.

## The connectors

| File | Purpose | No-GPU path |
|------|---------|-------------|
| `usd_scene_builder.py` | Neutral `Scene3D` JSON → a USD stage (terrain heightfield + extruded buildings + roads/water, per-kind `UsdPreviewSurface` materials). The Isaac-side writer of the map → 3D-twin pipeline. | `--validate` (assert well-formed), `--obj out.obj`, `--preview out.png` (matplotlib) |
| `render_city.py` | Headless Isaac render of a USD twin → PNG. The real-GPU proof the map → 3D-scene pipeline renders. | — (needs Isaac + GPU) |
| `camera_server.py` | Serves an Isaac camera as an **MJPEG IP camera** (`/mjpeg`, `/snapshot`, `/status`) — SC registers it like any security camera; nothing in SC knows it is Isaac. | `--source synthetic` (numpy/cv2 moving subject), `--selftest` |
| `isaac_camera_bridge.py` | Publishes a unit's onboard camera frames as JPEG on the `camera_feeds` MQTT topic `tritium/{site}/cameras/{cam_id}/frame`, keyed to the unit id; optional local `build_frame_detector` publishes detections on `.../detections`. Isaac-free — only `numpy` + JPEG + MQTT. | Imports & unit-tests headless (no Isaac); `isaac_camera_rgb()` lazy-imports Isaac only inside a live sim |
| `isaac_quadruped_server.py` | The physics **robot BODY** behind the brain/body TCP seam: owns an Isaac stage, steps physics, moves a quadruped from the same twist/gait contract as `robot-template`, serves body state back. | `--selftest` (gait integrator + footfalls + TCP loopback, no Isaac) |
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
```

## Related

- [Isaac Sim addon overview](../../README.md) — full pipeline diagram + run recipes
- [Examples](../../examples/) (`spot_policy_walk.py`, `smoke_boot.py`, `robot_bridge.md`)
- [No-GPU tests](../../tests/) · [Package README](../README.md)
