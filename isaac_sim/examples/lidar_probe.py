#!/usr/bin/env python3
"""Probe: what does a live Isaac RTX Lidar actually hand back?

Why this exists
---------------
`lidar_server.py --source isaac` boots clean, reports `scans` climbing and
`errors: 0`, and returns 360 beams of exactly `range_max` — an empty room and a
sensor that never ray-traced are byte-identical.  Tick 21 blamed VRAM
(`VK_ERROR_OUT_OF_DEVICE_MEMORY` with the Newton kit holding 18 of 24 GB) and
added the `never_returned` flag so the two could be told apart.

With 21 GB free the Vulkan error is GONE and the sweep is STILL empty, so the
failure lives in a different layer: `LidarRtx.get_current_frame()` is not
producing `range`/`azimuth`.  This probe does not guess which layer — it dumps
what the live sensor returns, step by step, so the next change is driven by an
observation instead of an API assumption.

Probes, not fixes.  Run it, read it, then change the connector.

    ~/miniconda3/envs/isaaclab/bin/python3.11 examples/lidar_probe.py

Copyright (c) Matthew Valancy / Valpatel Software LLC.  AGPL-3.0.
"""

import sys

import numpy as np


def _dump(tag, frame):
    """Print a frame's shape without assuming any particular key exists."""
    if not isinstance(frame, dict):
        print(f"  [{tag}] frame is {type(frame).__name__}: {frame!r}")
        return
    if not frame:
        print(f"  [{tag}] frame is an EMPTY dict")
        return
    for key, val in frame.items():
        arr = np.asarray(val) if not isinstance(val, (str, bytes)) else None
        if arr is not None and arr.dtype != object and arr.size:
            print(f"  [{tag}] {key}: shape={arr.shape} dtype={arr.dtype} "
                  f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g}")
        elif arr is not None and arr.dtype != object:
            print(f"  [{tag}] {key}: shape={arr.shape} EMPTY")
        else:
            print(f"  [{tag}] {key}: {type(val).__name__} {val!r:.120}")


def main() -> int:
    from isaacsim.simulation_app import SimulationApp
    sim = SimulationApp({"headless": True})

    from pxr import UsdGeom, UsdPhysics
    import isaacsim.core.utils.stage as stage_utils
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.sensors.rtx import LidarRtx

    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = World(stage_units_in_meters=1.0)

    # Box slab, never a flat quad — the zero-thickness ground that qhull cannot
    # hull silently disables Newton integration process-wide (tick 9/10).
    slab = UsdGeom.Cube.Define(stage, "/World/GroundSlab")
    slab.CreateSizeAttr(2.0)
    slab_xf = UsdGeom.Xformable(slab.GetPrim())
    slab_xf.AddTranslateOp().Set((0.0, 0.0, -0.5))
    slab_xf.AddScaleOp().Set((25.0, 25.0, 0.5))
    UsdPhysics.CollisionAPI.Apply(slab.GetPrim())

    # The same three boxes the connector authors: their bearings are the honest
    # metric — 0 deg / 90 deg / ~194 deg at ~3.6 / 3.1 / 3.7 m.
    for i, (x, y) in enumerate(((4.0, 0.0), (0.0, 3.5), (-4.0, -1.0))):
        world.scene.add(FixedCuboid(
            prim_path=f"/World/Tritium/Obstacle_{i}", name=f"obstacle_{i}",
            position=np.array([x, y, 0.75]), scale=np.array([0.8, 0.8, 1.5])))

    lidar = LidarRtx(prim_path="/World/Tritium/Robot/lidar", name="probe_lidar",
                     position=np.array([0.0, 0.0, 0.6]),
                     config_file_name="Example_Rotary_2D")
    world.reset()
    lidar.add_range_data_to_frame()
    lidar.add_azimuth_data_to_frame()

    print("=" * 70)
    print("PROBE A — LidarRtx.get_current_frame() over 40 steps")
    print("=" * 70)
    print(f"  type(lidar)      = {type(lidar)}")
    print(f"  has initialize   = {hasattr(lidar, 'initialize')}")
    print(f"  render_product   = {getattr(lidar, '_render_product_path', None)}")
    print(f"  annotators       = {list(getattr(lidar, '_annotators', {}) or {})}")
    for step in range(40):
        world.step(render=True)
        if step in (0, 1, 5, 20, 39):
            _dump(f"step {step}", lidar.get_current_frame())

    # PROBE B — the raw replicator annotator, which is what LidarRtx wraps.
    # Probe A showed `annotators = []`: add_range_data_to_frame() attached
    # NOTHING, so `range`/`azimuth` never entered the frame and every sweep fell
    # into get_scan()'s empty branch.  That is an API mismatch, not the GPU.
    # These are the names this build actually registers (read off the
    # AnnotatorRegistryError above).  IsaacComputeRTXLidarFlatScan is the ROS
    # LaserScan producer — the standard wheel for a 2D sweep.
    print("=" * 70)
    print("PROBE B — annotators this build ACTUALLY registers")
    print("=" * 70)
    flat = None
    try:
        import omni.replicator.core as rep
        rp = rep.create.render_product("/World/Tritium/Robot/lidar", [1, 1],
                                       name="probe_rp")
        for ann_name in ("IsaacComputeRTXLidarFlatScan",
                         "IsaacCreateRTXLidarScanBuffer",
                         "GenericModelOutput"):
            try:
                ann = rep.AnnotatorRegistry.get_annotator(ann_name)
                ann.attach([rp])
            except Exception as exc:
                print(f"  [{ann_name}] attach FAILED: {type(exc).__name__}: {exc}")
                continue
            for _ in range(30):
                world.step(render=True)
            try:
                data = ann.get_data()
                _dump(ann_name, data)
                if ann_name == "IsaacComputeRTXLidarFlatScan":
                    flat = data
            except Exception as exc:
                print(f"  [{ann_name}] get_data FAILED: {type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"  replicator probe unavailable: {type(exc).__name__}: {exc}")

    # PROBE C — the honest metric.  A constant-filled array cannot fake
    # BEARINGS: the three boxes sit at (4, 0), (0, 3.5) and (-4, -1), so the
    # sweep's range minima must land near 0 deg / 90 deg / 194 deg at roughly
    # 3.6 / 3.1 / 3.7 m.  "The server returned 360 floats" would wave through
    # exactly the empty sweep that cost this lane two ticks.
    print("=" * 70)
    print("PROBE C — do the range minima land on the boxes' true bearings?")
    print("=" * 70)
    if not flat:
        print("  NO FLAT SCAN DATA — cannot score bearings.")
    else:
        depth = np.asarray(flat.get("linearDepthData", []), dtype=np.float64).ravel()
        if depth.size == 0:
            print("  flat scan carried no linearDepthData")
        else:
            info = flat.get("info", {}) or {}
            az_range = info.get("azimuthRange", (-np.pi, np.pi))
            print(f"  beams={depth.size} range=[{depth.min():.3f}..{depth.max():.3f}] "
                  f"azimuthRange={az_range}")
            hits = depth[(depth > 0.05) & (depth < 29.0)]
            print(f"  returns under 29 m: {hits.size} of {depth.size}")
            bearings = np.linspace(az_range[0], az_range[1], depth.size,
                                   endpoint=False)
            for name, want_deg, want_m in (("box@(4,0)", 0.0, 3.6),
                                           ("box@(0,3.5)", 90.0, 3.1),
                                           ("box@(-4,-1)", 194.0, 3.7)):
                lo, hi = np.deg2rad(want_deg - 20), np.deg2rad(want_deg + 20)
                bb = (bearings + 2 * np.pi) % (2 * np.pi)
                sel = ((bb >= (lo + 2 * np.pi) % (2 * np.pi)) &
                       (bb <= (hi + 2 * np.pi) % (2 * np.pi)))
                if not sel.any():
                    print(f"  {name}: no beams in window")
                    continue
                idx = np.argmin(np.where(sel, depth, np.inf))
                got_deg = np.rad2deg(bearings[idx]) % 360
                print(f"  {name}: min {depth[idx]:.3f} m at {got_deg:.1f} deg "
                      f"(expected ~{want_m} m at {want_deg} deg)")

    print("=" * 70)
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
