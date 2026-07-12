# Tritium SIL example: velocity-commanded quadruped walking in Isaac Sim.
#
# Adapted from NVIDIA's `spot_standalone.py`
# (isaacsim.robot.policy.examples, Apache-2.0) with the Tritium framing + the
# exact setup that makes the pretrained RL policy walk **stably** — each item
# below was a real blocker discovered while bringing this up live (2026-07-12,
# tick BN). See `../../../docs/ISAAC-SIM-STATUS.md`.
#
# WHY THIS MATTERS FOR TRITIUM: the policy is driven by a velocity command
# `[vx, vy, yaw_rate]` — which IS the brain/body twist seam. The same command the
# navigator/autonomy stack emits to drive a real machine drives this SIL body.
# Swap the Isaac body for real hardware and the identical command walks the robot.
#
# THE RECIPE (do not "simplify" these away — each one, removed, makes it fall):
#   1. GROUND: use a real ground plane (the Isaac Grid env). A scaled-unit-cube
#      collider gives poor DYNAMIC foot contact — the robot stands statically but
#      the policy's rapid foot strikes destabilise it and it collapses.
#   2. HOOK: drive `policy.forward(step, cmd)` from
#      `SimulationManager.register_callback(fn, IsaacEvents.POST_PHYSICS_STEP)`
#      — the correct per-physics-step Isaac hook. The app-update stream (render
#      rate) and `omni.physx` step events have the wrong timing.
#   3. RATE: `SimulationManager.set_physics_dt(1/200)` — the policy's trained rate.
#   4. BACKEND: PhysX. The shipped policy is PhysX-validated; under the Newton
#      backend it destabilises (Newton is great for looks/soft-body, but this
#      pretrained locomotion policy wants PhysX until a Newton-native/retrained
#      policy exists).
#
# Run (headless, on an RTX host):
#   ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
#       tritium-addons/isaac_sim/examples/spot_policy_walk.py --headless
#
# Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC (Tritium glue);
# derived from NVIDIA Apache-2.0 example code (the policy + setup calls).

import argparse

parser = argparse.ArgumentParser(description="Tritium SIL: Spot velocity-commanded walk.")
parser.add_argument("--headless", action="store_true", help="Run without a window (RTX host).")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Physics device.")
parser.add_argument("--seconds", type=float, default=20.0, help="Sim seconds to run.")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import omni.timeline  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.experimental.utils.stage import define_prim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Isaac assets root not found")

# (1) GROUND — a real ground plane with proper contact/friction.
ground = define_prim("/World/Ground", "Xform")
ground.GetReferences().AddReference(assets_root + "/Isaac/Environments/Grid/default_environment.usd")
define_prim("/World/PhysicsScene", "PhysicsScene")

# (3) RATE + (4) BACKEND
SimulationManager.set_physics_sim_device(args.device)
SimulationManager.set_physics_dt(1.0 / 200.0)

# The pretrained RL locomotion policy (loads .pt + env.yaml from the asset server).
spot = SpotFlatTerrainPolicy(prim_path="/World/Spot", position=[0.0, 0.0, 0.8])

# THE TWIST SEAM: [vx (m/s), vy (m/s), yaw_rate (rad/s)] in the body frame.
twist = torch.zeros(3, device=args.device)

_state = {"first": True}


# (2) HOOK — the correct per-physics-step callback.
def on_physics_step(step_size, context=None):
    if _state["first"]:
        spot.initialize()          # must init once, on a live physics step
        _state["first"] = False
    else:
        spot.forward(step_size, twist)


SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

omni.timeline.get_timeline_interface().play()
simulation_app.update()

# Simple command script: stand, walk forward, turn, walk forward. In production
# `twist` is fed by the autonomy stack / navigator instead of this schedule.
schedule = [
    (2.0, [0.0, 0.0, 0.0]),   # settle / stand
    (6.0, [1.3, 0.0, 0.0]),   # forward
    (4.0, [1.0, 0.0, 1.0]),   # forward + turn
    (6.0, [1.3, 0.0, 0.0]),   # forward
]
elapsed, phase_i, phase_t = 0.0, 0, 0.0
dt = 1.0 / 200.0
while simulation_app.is_running() and elapsed < args.seconds:
    simulation_app.update()
    if SimulationManager.is_simulating():
        if phase_i < len(schedule):
            dur, cmd = schedule[phase_i]
            twist[:] = torch.tensor(cmd, device=args.device)
            phase_t += dt
            if phase_t >= dur:
                phase_i += 1
                phase_t = 0.0
        else:
            twist[:] = 0.0
        elapsed += dt

print("final base pose:", spot.robot.get_world_poses()[0])
simulation_app.close()
