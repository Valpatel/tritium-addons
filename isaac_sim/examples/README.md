# Isaac Sim connector — examples

Run recipes live in the addon [`../README.md`](../README.md). In short, on the
render host (Isaac's Python 3.12 venv, `OMNI_KIT_ACCEPT_EULA=YES`):

1. `usd_scene_builder.py --scene-url http://<sc>:8000/api/gis/scene3d?bbox=...&ao=dublin --out dublin.usd`
2. `render_city.py --usd dublin.usd --out dublin.png`
3. `camera_server.py --source isaac --scene dublin.usd --port 8100` → register in SC as an `mjpeg` camera.

No-GPU equivalents for CI (plain python3): `usd_scene_builder.py --validate --obj`,
`camera_server.py --selftest`. See [`../tests/test_no_gpu.py`](../tests/test_no_gpu.py).

## Robot-dog body bridge

`connectors/isaac_quadruped_server.py` is an Isaac physics body behind the same
TCP seam the `tritium-sc/examples/robot-template` brain speaks — dispatch and
fire a physics dog from the tactical map. First run `smoke_boot.py` (boots Isaac
headless, steps 60 physics steps, prints `SMOKE OK`); no-GPU self-test is
`isaac_quadruped_server.py --selftest` (integrator + footfalls + TCP loopback,
no isaacsim). Full run recipe, JSON protocol, and Jetson mapping:
[`robot_bridge.md`](robot_bridge.md).

## Locomotion (SIL)

- `go2_newton_gait.py` — the Newton lane's main driver: emits a tritium-lib
  gait table (`--emit-gait`, no GPU) and drives it into a **live Newton kit**
  over the MCP bridge (`--gait-file`, default port 8212), scoring each run
  with non-gameable `score_trace` metrics. Grew the whole lane's control
  surface: `--stabilize on/off/both` (closed attitude loop vs the open-loop
  control arm — **on** by default), `--trials N` for reliability counts,
  `--live-port` for UDP teleop (see `../isaac_sim_addon/clients/teleop_send.py`),
  `--route` / `--plan-to` / `--obstacle` for waypoint and A*-planned walks.
  The measured outcomes (34/34 closed-loop vs 17/24 open-loop, and every
  dead end on the way) live in
  [`NEWTON-GAIT-FINDINGS.md`](NEWTON-GAIT-FINDINGS.md) — dated live-kit
  observations, not re-runnable without the RTX host.
- `go2_newton_stand.usd` — a Newton-physics Go2 stand scene; a real actuated
  quadruped stands under physics (load it in Isaac to verify the render host).
- `newton_stand_and_walk.py` — the committed **stand + attempted trot**
  scaffold for the Newton lane: spawns the Go2 under `/World/Tritium/go2`,
  registers it with the solver (spawn while stopped + `World.reset()`), stands
  it via USD drive `targetPosition` (hip 0 / thigh +50° / calf −100°), then
  applies a low-speed tritium-lib trot through the addon `GaitScheduler`
  every control step, scoring the run with the same non-gameable
  `score_trace` metrics as `go2_newton_gait.py`. The pure core
  `build_walk_plan(duration_s, dt, gait, speed, targets_fn)` is unit-tested
  headless (`../tests/test_newton_stand_and_walk.py`). No-GPU:
  `newton_stand_and_walk.py --selftest` / `--emit-plan plan.json [--mock]`.
  Live (own kit — stop the bridged 8212 kit first):

  ```bash
  ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
      tritium-addons/isaac_sim/examples/newton_stand_and_walk.py \
      --live --headless --seconds 8 --speed 0.35 \
      --capture /tmp/go2_walk.png --record /tmp/go2_walk.json
  ```
- `spot_policy_walk.py` — a **velocity-commanded** RL walk driven by
  `[vx, vy, yaw_rate]`, which *is* the brain/body twist seam: the same command
  the navigator/autonomy stack emits to a real machine drives this SIL body.
  Runs on the PhysX backend (the shipped pretrained policy is PhysX-validated).
  Headless on an RTX host:

  ```bash
  ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
      tritium-addons/isaac_sim/examples/spot_policy_walk.py --headless
  ```

## Map integration

- `go2_walk_to_map.py` — the first whole-chain run: a Go2 *actually
  locomoting* under Newton streams its pose through
  `POST /api/sighting` (`clients/pose_bridge.py` shape) so its track moves on
  a running Command Center's tactical map while it walks.

## Sensors on a body

- `isaac_sensor_rig.py` — one command brings up a robot's whole sensor set
  (camera :8100, LiDAR :8110, body :18973) as child processes. The launcher
  itself is glue only — plain python3, stdlib, no isaacsim, no tritium.
- `mounted_camera_check.py` — proves a **body-mounted** camera against a live
  sim: the mount is rigid in the body frame, so the lens swings when the body
  turns (`tritium_lib.geo.camera_mount.CameraMount` is the function under
  test).
- `lidar_probe.py` — what a live RTX Lidar *actually* hands back: written
  because `lidar_server.py --source isaac` can boot clean and still return
  360 beams of exactly `range_max` (an empty room and a sensor that never
  ray-traced are byte-identical).

## Probes (how the Newton lane got unstuck)

- `newton_ground_fix_proof.py` — a released cube finally falls (`verdict
  STEPPED`): the rank-2 flat ground plane was silently disabling Newton
  physics for the whole process.
- `newton_world_probe.py` — is the articulation actually stepped by the
  solver? Found the real cause one layer up.

Both are kept runnable because the failure they diagnose — sim time advancing
while nothing integrates — looks healthy from every casual observable. Full
story: [`NEWTON-GAIT-FINDINGS.md`](NEWTON-GAIT-FINDINGS.md).
