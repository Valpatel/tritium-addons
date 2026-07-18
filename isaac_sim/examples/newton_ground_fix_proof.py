#!/usr/bin/env python3
"""The Newton lane's blocker, found and cleared: a body finally falls.

Four ticks of this lane ended at the same wall.  A cube with RigidBodyAPI,
released 3 m above a ground plane, dropped 0.0 m while sim time advanced 4.08 s
-- under default gravity, under authored gravity, and when everything was
authored before the process's first play.  The Go2 with all 12 drives zeroed
behaved identically.  The conclusion recorded last tick was correct as far as
it went: "the Newton solver advances its clock but integrates nothing."

**The cause is the ground plane itself.**  Newton's MuJoCo solver reduces every
collision shape to a convex hull.  The ground was authored the obvious way, as
a 100 x 100 quad with all four corners at z = 0, and qhull cannot seed a
simplex from a rank-2 point set:

    QH6154 Qhull precision error: Initial simplex is flat
    - p3(v4):   -50    50     0
    - p2(v3):    50    50     0
    - p1(v2):    50   -50     0
    - p0(v1):   -50   -50     0
      2:         0         0  difference=    0     <-- no thickness

**Why it was invisible for four ticks** is the part worth remembering.  Look at
`isaacsim/physics/newton/impl/newton_stage.py`:

    def initialize_newton(self, device):
        if getattr(self, "_initializing", False):
            return                       # <-- latched
        ...
        self._initializing = True        # <-- set BEFORE the work
        ...                              # <-- qhull raises in here
                                         # <-- never cleared on the error path

    def step_sim(self, dt):
        if not self.initialized:
            self.initialize_newton(self.device)   # returns instantly, forever
        self.sim_time += dt                       # clock advances anyway
        self.simulation_step_count += 1           # counter advances anyway
        if self.playing:
            self.simulate(dt=dt)                  # on a half-built model

So one exception, once, disables physics for the entire life of the process --
and every observable a person would check to see whether the sim is healthy
keeps moving.  The live kit confirmed the latch directly: `_initializing=True`
and `initialized=False` after 329,580 steps and 329 s of sim time.

**The fix is to give the ground thickness.**  A 50 x 50 x 1 m box slab has the
same footprint and is rank 3, so qhull seeds from it happily.  With that one
change, on the same kit, same solver, same everything else:

    z 2.965 m  ->  0.498 m   over 4.08 s of sim
    initialized: True    _initializing: False

0.498 is not merely "smaller".  It is half the height of a 1 m cube whose
bottom face is resting on the top of a slab at z = 0 -- the body fell AND it
came to rest in the geometrically correct place, which a broken integrator does
not do by accident.  Viewport captures of the airborne and landed states are in
`docs/images/isaac/newton-cube-falls-2026-07-18.png`.

**The guard now lives in the library.**  `tritium_lib.geo.collider_shape`
performs qhull's own rank test on a vertex list with no GPU, no USD and no
Isaac, so a scene builder can refuse to author a shape that would kill physics
silently.  Note that the check is a rank test rather than a per-axis extent
test on purpose: tilt the flat quad and all three of its AABB extents become
non-zero while the point set stays rank 2 and qhull still fails.

Usage (run anywhere; talks to the kit's bridge):
    python3 newton_ground_fix_proof.py [--port 8212] [--updates 240]

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_PORT = 8212

# The scene, authored in the kit.  Ground is a BOX slab, never a flat quad.
SCENE = """
import json, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import omni.usd, omni.timeline, omni.kit.app
    from pxr import UsdGeom, UsdPhysics, Gf, UsdLux
    ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    sc = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
    UsdLux.DistantLight.Define(stage, "/World/Key").CreateIntensityAttr(3000)

    # GROUND: a 50 x 50 x 1 m slab.  A zero-thickness quad here is what broke
    # the whole lane -- see this file's docstring.
    g = UsdGeom.Cube.Define(stage, "/World/Ground"); g.CreateSizeAttr(2.0)
    gx = UsdGeom.Xformable(g.GetPrim())
    gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    gx.AddScaleOp().Set(Gf.Vec3f(25, 25, 0.5))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    c = UsdGeom.Cube.Define(stage, "/World/DropCube"); c.CreateSizeAttr(1.0)
    c.CreateDisplayColorAttr().Set([(0.0, 0.94, 0.63)])
    cx = UsdGeom.Xformable(c.GetPrim())
    cx.AddTranslateOp().Set(Gf.Vec3d(0, 0, DROP_Z))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(1.0)

    cam = UsdGeom.Camera.Define(stage, "/World/DropCam")
    camx = UsdGeom.Xformable(cam.GetPrim())
    camx.AddTranslateOp().Set(Gf.Vec3d(9.0, -9.0, 2.2))
    camx.AddRotateXYZOp().Set(Gf.Vec3f(84.0, 0.0, 45.0))

    app = omni.kit.app.get_app()
    for _ in range(5): app.update()
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/DropCam"
    except Exception:
        pass
    for _ in range(5): app.update()
    omni.timeline.get_timeline_interface().play()
    for _ in range(3): app.update()
    import isaacsim.physics.newton as ipn
    st = ipn.acquire_stage()
    z = float(st.state_0.body_q.numpy()[0][2])
result = json.dumps({
    "z": z, "sim_time": st.sim_time,
    "initialized": st.initialized,
    "_initializing": getattr(st, "_initializing", None),
})
"""

# A headless kit does not pump its own update loop -- see NEWTON-GAIT-FINDINGS.
PUMP = """
import json, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import omni.kit.app, isaacsim.physics.newton as ipn
    app = omni.kit.app.get_app()
    for _ in range(N_UPDATES): app.update()
    st = ipn.acquire_stage()
    z = float(st.state_0.body_q.numpy()[0][2])
result = json.dumps({
    "z": z, "sim_time": st.sim_time, "initialized": st.initialized,
})
"""


class Bridge:
    """Minimal client for the Omniverse MCP bridge's /execute endpoint."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout: float = 600.0) -> None:
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def execute(self, code: str) -> dict:
        payload = json.dumps({"code": code}).encode()
        req = urllib.request.Request(
            f"{self.base}/execute", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.load(resp)
        # the bridge wraps twice
        return json.loads(body["result"]["return_value"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--updates", type=int, default=240)
    ap.add_argument("--drop", type=float, default=3.0)
    args = ap.parse_args()

    bridge = Bridge(args.host, args.port)

    print(f"authoring drop scene (cube at z={args.drop} m, ground = box slab)")
    start = bridge.execute(SCENE.replace("DROP_Z", repr(args.drop)))
    print(f"  z_start        {start['z']:.4f} m")
    print(f"  initialized    {start['initialized']}")
    print(f"  _initializing  {start['_initializing']}")

    if not start["initialized"]:
        print("\nFAIL: Newton did not initialize. If _initializing is True the "
              "latch described in this file's docstring has stuck -- the kit "
              "must be restarted (newton_kit.sh restart) before any scene will "
              "simulate.", file=sys.stderr)
        return 2

    print(f"\npumping {args.updates} app updates")
    end = bridge.execute(PUMP.replace("N_UPDATES", str(args.updates)))
    print(f"  z_end          {end['z']:.4f} m")
    print(f"  sim_time       {end['sim_time']:.3f} s")

    dropped = start["z"] - end["z"]
    resting = 0.5  # half a 1 m cube, sitting on a slab whose top is z=0
    print(f"\n  dropped        {dropped:.4f} m")
    print(f"  resting height {end['z']:.4f} m (expected {resting})")

    # Two independent conditions.  Falling alone could be a body sinking through
    # a broken collider; resting at the right height is what proves contact.
    fell = dropped > 0.5
    landed = abs(end["z"] - resting) < 0.05
    verdict = "STEPPED" if (fell and landed) else "NOT_STEPPED"
    print(f"\n  verdict        {verdict}")
    if not fell:
        print("  -> the body did not move; the solver is not integrating")
    elif not landed:
        print("  -> the body moved but did not rest on the slab; check the "
              "ground collider")
    return 0 if verdict == "STEPPED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
