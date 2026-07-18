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
