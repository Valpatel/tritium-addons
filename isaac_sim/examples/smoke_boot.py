# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Minimal Isaac Sim launch validator — the first thing to run when VRAM frees.

Boots SimulationApp headless, builds the smallest honest physics scene
(ground plane + one dynamic cube), steps 60 real physics steps, and prints
one grep-able line. If this passes, the local Isaac build can boot, create a
World, and step physics — which is everything isaac_quadruped_server.py
needs beyond its own (separately self-tested) TCP/gait layer.

RUN (under Isaac's bundled python), from the tritium-addons repo root:
    ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
        isaac_sim/examples/smoke_boot.py

Expected output line:  SMOKE OK steps=60 sim_time=1.000

API pattern ground truth (local build v6.0.0-rc.22):
standalone_examples/api/isaacsim.simulation_app/hello_world.py (boot),
standalone_examples/api/isaacsim.core.api/add_cubes.py (World + DynamicCuboid),
standalone_examples/benchmarks/benchmark_core_world.py (headless +
add_default_ground_plane + step(render=False)).
"""

import sys

from isaacsim import SimulationApp

# Boot FIRST — before any other isaacsim import (standard pattern in the
# local standalone examples).
simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

STEPS = 60
PHYSICS_DT = 1.0 / 60.0


def main() -> int:
    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT,
                  stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/smoke_cube",
            name="smoke_cube",
            position=np.array([0.0, 0.0, 1.0]),
            size=0.2,
            color=np.array([0.0, 0.94, 1.0]),
        )
    )
    world.reset()
    for _ in range(STEPS):
        world.step(render=False)
    print(f"SMOKE OK steps={STEPS} sim_time={world.current_time:.3f}",
          flush=True)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)
