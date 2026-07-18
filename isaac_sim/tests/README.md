# tests — Isaac Sim connector no-GPU gates

pytest coverage for the [Isaac Sim connector](../README.md): **15 files, 332
passed / 1 skipped** (2026-07-18; the skip is an MQTT broker round-trip that
only runs when a broker is reachable). Every test runs under plain `python3`
with **no Isaac, no `pxr`, no GPU** — proving the connectors stay inspectable,
the neutral contracts hold in CI, and the dependency-hygiene invariant
(connectors never import `tritium`; the guard walks every connector source).

**Deps:** `pytest`, `numpy`, `cv2`, and `tritium_lib` (the `clients/` tests —
fire bridge, pose bridge, shot poster — exercise the lib maths those clients
are allowed to import). No `isaacsim`.

## Run

```bash
cd tritium-addons
python3 -m pytest isaac_sim/tests/ -q
```

## What is covered

- `test_no_gpu.py` — the original gate: `usd_scene_builder` validate/OBJ,
  `camera_server` synthetic frames + depth/depth16/stereo channels,
  `isaac_quadruped_server` gait integrator + footfalls + `--selftest`, and the
  "connectors never import tritium" guard.
- `test_fire.py` / `test_fire_bridge.py` / `test_shot_poster.py` — the fire
  path: TCP `fire`/`targets` with terrain, the live-trial client's stage
  snippets and verdicts, and the SC `POST /api/engagement/shot` poster
  (payload shape, graceful failure, circuit breaker).
- `test_newton_gait_driver.py` — the gait scheduler contract: rad→deg,
  clamping, stabilizer injection, foot-height trim.
- `test_newton_gait.py` / `test_gait_*.py` / `test_newton_stand_and_walk.py` —
  the `go2_newton_gait.py` driver's pure logic and generated-source plumbing
  (kick gating, live teleop, route following) plus `build_walk_plan`.
- `test_isaac_camera_bridge.py` / `test_isaac_sensor_rig.py` /
  `test_mounted_camera_check.py` / `test_pose_bridge.py` / `test_teleop_send.py`
  — the camera_feeds MQTT contract (broker test skips without a broker), the
  rig launcher, camera-mount geometry, sighting POST shape, and teleop framing.

What these gates deliberately do **not** cover: live-Isaac outcomes (Newton
walking, RTX Lidar returns, viewport captures). Those are recorded, dated, in
[`../examples/NEWTON-GAIT-FINDINGS.md`](../examples/NEWTON-GAIT-FINDINGS.md).

## Related

- [Isaac Sim addon overview](../README.md) · [Connectors](../isaac_sim_addon/connectors/) · [Clients](../isaac_sim_addon/clients/)
