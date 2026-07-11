#!/usr/bin/env python3
# Created by Matthew Valancy — Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""Headless Isaac Sim render of a Tritium AO USD twin -> PNG (rtx3080 host).

Opens a USD built from /api/gis/scene3d (terrain + buildings + roads + water,
Z-up, AO-local metres), lights it, aims a camera over the city centre, and
captures an RGB frame. This is the real-GPU proof that the map->3D-scene
pipeline renders in Isaac Sim.
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", default="city.png")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--cam", type=float, nargs=3, default=[1400.0, -1400.0, 750.0],
                help="camera position east,north,up (m)")
ap.add_argument("--look", type=float, nargs=3, default=[0.0, 0.0, 150.0],
                help="look-at point east,north,up (m)")
args = ap.parse_args()

from isaacsim.simulation_app import SimulationApp  # noqa: E402
sim = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import UsdGeom, UsdLux, Gf  # noqa: E402

# Open the city stage.
omni.usd.get_context().open_stage(args.usd)
for _ in range(10):
    sim.update()
stage = omni.usd.get_context().get_stage()
try:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
except Exception:
    pass

# Sun + sky.
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500.0)
sun.CreateAngleAttr(1.0)
xf = UsdGeom.Xformable(sun.GetPrim())
xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))
sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
sky.CreateIntensityAttr(800.0)

# Camera over the city centre.
cam = rep.create.camera(position=tuple(args.cam), look_at=tuple(args.look))
rp = rep.create.render_product(cam, (args.width, args.height))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])

# Warm the RTX pipeline (shader/warp compile) then accumulate a clean frame.
for _ in range(90):
    sim.update()
for _ in range(4):
    rep.orchestrator.step(rt_subframes=16, pause_timeline=False)
    sim.update()

data = annot.get_data()
arr = np.asarray(data, dtype=np.uint8)
if arr.ndim == 3 and arr.shape[2] == 4:
    arr = arr[:, :, :3]

try:
    from PIL import Image
    Image.fromarray(arr).save(args.out)
except Exception:
    # Fallback: raw npy if PIL missing.
    np.save(args.out + ".npy", arr)
print(f"RENDER OK {args.out} shape={arr.shape} "
      f"mean={float(arr.mean()):.1f} nonblack={(arr.sum(axis=2) > 20).mean()*100:.1f}%")
sim.close()
sys.exit(0)
