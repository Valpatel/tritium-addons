# Newton gait lane — live-sim findings

Notes from driving Isaac Sim 6.0 with **Newton physics** on the RTX 4090 via
the Omniverse MCP bridge. Everything here was observed against a live kit, not
inferred from docs. Dated so a later tick can tell stale from current.

## 2026-07-18 (tick 9) — SOLVED: the ground plane was killing physics

**A body falls.** After four ticks stuck at "the solver advances its clock but
integrates nothing", a 1 kg cube released 3 m above the ground **fell 2.489 m
and came to rest at z = 0.498 m** — exactly half a 1 m cube sitting on a slab
whose top face is z = 0. Verified in the live kit and looked at:
`docs/images/isaac/newton-cube-falls-2026-07-18.png`. Reproduce with
`newton_ground_fix_proof.py` (prints `verdict STEPPED`).

**The cause was the ground plane.** Newton's MuJoCo solver reduces every
collision shape to a convex hull. Our ground was authored the obvious way — a
100 × 100 quad, four corners, all at z = 0 — and qhull cannot seed a simplex
from a rank-2 point set:

```
QH6154 Qhull precision error: Initial simplex is flat
- p3(v4):   -50    50     0
- p2(v3):    50    50     0
- p1(v2):    50   -50     0
- p0(v1):   -50   -50     0
  2:         0         0  difference=    0      <-- no thickness
```

Bitterly, the *workaround* recorded in the previous entry — "`world.step()` is
unavailable, use a plain USD ground plane" — is what planted the bug. The plain
ground plane was the poison.

**Why it stayed invisible for four ticks** is the part worth carrying forward.
In `isaacsim/physics/newton/impl/newton_stage.py`:

```python
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
```

One exception, once, disables physics for the **entire life of the process** —
while every observable a person would check to see whether the sim is healthy
keeps moving. The wedged kit confirmed the latch directly: `_initializing=True`
and `initialized=False` after **329,580 steps and 329 s of sim time**. This is
why "sim time is advancing" was such a convincing false signal, and why the
diagnosis kept landing one layer too high.

**Diagnostic shortcut for next time.** `isaacsim.physics.newton.acquire_stage()`
returns the `NewtonStage`, and it answers the health question in one call:

```python
import isaacsim.physics.newton as ipn
st = ipn.acquire_stage()
st.initialized, getattr(st, "_initializing", None)   # (True, False) = healthy
                                                     # (False, True) = wedged
st.model.body_count, st.model.shape_count, st.model.gravity
```

`initialized=False` with `_initializing=True` means the latch has stuck and
**no scene will ever simulate in that process** — restart the kit
(`newton_kit.sh restart 8212`) before doing anything else.

**The fix, and the guard.** Give the ground thickness: a 50 × 50 × 1 m box slab
has the same footprint and is rank 3. The general guard now lives in the
library as **`tritium_lib.geo.collider_shape`**, which runs qhull's own rank
test on a vertex list with no GPU, no USD and no Isaac, so a scene builder can
refuse to author a shape that would kill physics silently. It is a *rank* test,
not a per-axis extent test, and deliberately so: tilt the flat quad and all
three of its AABB extents become non-zero while the point set stays rank 2 and
qhull still fails.

**Premises from the previous entry that are now retired:** the solver is not
broken, the kit build is fine, the Newton model was never empty (it reported
`body_count 1`, correct gravity `(0,0,-9.81)`), and NVIDIA's shipped sample did
not need to be run. What was true and still is: a headless kit does not pump
its own update loop (`omni.kit.app.get_app().update()` is the stepping
primitive), physics writes to Fabric rather than USD, and `SingleArticulation`
remains retracted.

**Next:** the gait itself. The blocker that justified four ticks of deferral is
gone, so the Go2's 12 drives can now be commanded against a solver that
actually integrates.

## 2026-07-17 — `SingleArticulation` is unusable under Newton on this build

**Symptom.** Constructing `isaacsim.core.prims.SingleArticulation` against a
Newton-stepped stage raises:

```
CUDA error: an illegal memory access was encountered
  File isaacsim/core/prims/impl/single_articulation.py:109, in __init__
  File isaacsim/core/prims/impl/articulation.py:175,  in __init__
      Articulation._on_physics_ready(self, None)
  File isaacsim/core/prims/impl/articulation.py:5096, in _on_physics_ready
      default_positions, default_orientations = self.get_world_poses()
  File isaacsim/core/prims/impl/articulation.py:1888, in get_world_poses
      rot = self._backend_utils.xyzw2wxyz(pose[indices, 3:7])
  File isaacsim/core/utils/torch/rotations.py:394
```

**The fault is STICKY and process-fatal.** Once it fires, every later CUDA
call in that kit process dies the same way — including a bare
`torch.zeros(4, device="cuda").sum()`. The sim keeps *reporting*
`state: playing` with `sim_time` advancing and `fps: 60`, so the bridge looks
healthy while the GPU side is dead. Only a full kit restart recovers it
(`./newton_kit.sh restart 8212`).

> Consequence for anyone iterating here: **you get one attempt per process
> life.** Introspection (`dir()`, `inspect.signature`, module imports, reading
> `joint_names`) is CUDA-free and safe to batch; anything touching the physics
> tensors is a one-shot. Budget ~40 s per restart and plan the experiment
> before spending it.

**What it is NOT.** Three hypotheses were tested against a freshly restarted
kit and all three were wrong:

1. *A long-running process degrading over time.* No — reproduced on a kit that
   had been up under a minute.
2. *A stale physics view because the robot was spawned after `play`.* No —
   reproduced with the correct order (build scene while `stop`ped → `play` →
   wait for warm start → construct).
3. *A torch-vs-numpy backend misconfiguration.*
   `SimulationManager.get_backend()` already reports `numpy`, yet the
   traceback goes through `isaacsim/core/utils/**torch**/rotations.py`. The
   wrapper picks the torch backend regardless of the reported backend.

The most likely root cause is that the `isaacsim.core.prims` wrapper stack is
written against the PhysX tensor API, and its assumptions don't hold for the
Newton tensor view — it reads out of bounds and CUDA kills the context. This
contradicts the "verified working path" note in
`docs/ISAAC-SIM-STATUS.md` (2026-07-12); either the earlier session ran a
different build, or the claim was never reproduced. Treat that note as stale.

**The fix — use the Newton-native tensor API directly.** The layer *underneath*
the broken wrapper is fine:

```python
from isaacsim.core.simulation_manager import SimulationManager as SM
view = SM.get_physics_sim_view().create_articulation_view("/World/Tritium/go2/base")
view.set_dof_position_targets(targets, None)   # targets: (count, dof) float32
view.set_dof_stiffnesses(k, None)              # PD gains
view.set_dof_dampings(d, None)
root = view.get_root_transforms()               # (count, 7) xyz + quat
```

`SM.get_physics_sim_view()` returns a
`isaacsim.physics.newton.tensors.tensor_api.NewtonSimulationView`. Note the
trailing `None` on the setters — that's the indices argument, and omitting it
is a TypeError. `view.joint_names` gives the solver's own DOF ordering; map
the gait's joint names onto it rather than assuming they match.

## 2026-07-17 — the remaining blocker: the articulation is not being stepped

The driver now runs end-to-end against the live kit — scene built, 12 USD
drives configured, articulation view created, joint names resolved in solver
order, and gait targets written **471 times over 7.8 s** — and the dog still
does not move. Scored honestly: `verdict: STATIONARY`, `displacement_m: 0.0`.
Two captured viewport frames taken minutes apart are pixel-identical in pose.

A targeted diagnostic (`get_dof_positions` + `get_root_transforms` sampled 5 s
apart) shows why:

| Probe | t=0 | t=+5 s |
|---|---|---|
| `SM.get_simulation_time()` | 59.485 | 64.585 |
| `SM.is_simulating()` / `is_paused()` | `True` / `False` | — |
| `get_dof_positions()[0][:6]` | all `0.0` | all `0.0` |
| `get_root_transforms()[0][:3]` | `[0, 0, 0.45]` | `[0, 0, 0.45]` |

Sim time advances, so the timeline is genuinely running — but **every joint
reads exactly 0.0 and the root never moves**. The giveaway is gravity: a body
whose base sits at z=0.45 with joints at zero would either fall or settle. It
does neither, and it holds its authored pose to 4 decimal places. So the
articulation is *present in the tensor view* but is **not part of what Newton
is stepping** — the view is reading an unpopulated buffer, and writing
position targets into it is a no-op.

This is NOT the gait's fault, and not the trajectory's: `tritium-lib`'s
generator is unit-tested (31 cases), and the driver demonstrably applies its
output at ~60 Hz. The gap is one layer lower — getting the Go2 admitted into
the Newton solver's model.

**Next tick starts here.** Leads, roughly in order of promise:

1. **Build the scene through `isaacsim.core.api.World`** rather than raw USD +
   MCP calls, then `world.reset()` before play. The Spot walk that *did* work
   (see `docs/ISAAC-SIM-STATUS.md`) went through `World`, and `World` is what
   calls `initialize_physics()`/`setup_simulation()` and pumps
   `SimulationManager`'s default callbacks. That the POST_PHYSICS_STEP
   callback also never fired here (physics_steps stayed 0 until the driver
   was moved onto the app update stream) is the same smell: no `World`, so
   none of the core physics plumbing is initialized.
2. Check `SM.get_physics_scenes()` / `get_default_physics_scene()` — the
   Go2 may need to live under an explicit physics scene the solver owns.
3. Confirm the articulation root is where Newton expects it. The
   `ArticulationRootAPI` sits on `/World/Tritium/go2/base` while the
   reference is added at `/World/Tritium/go2`; try creating the view against
   the parent, or with a `*` pattern, and compare `view.count`/`max_dofs`.
4. Verify the Newton model rebuild actually happens on `stop → play` after a
   variant switch — the payload arrives late, and the solver may have already
   built its model from the variant-`None` (body-less) state.

Do **not** let the temptation to "just switch to the PhysX kit" close this
out. Newton is the point of the lane; PhysX isn't even available in this kit
(`get_available_physics_engines()` → `[('newton', True), ('physx', False)]`).

## Environment facts (verified live, 2026-07-17)

| Fact | Value |
|---|---|
| Kit app | `apps/isaacsim.exp.full.newton.kit` (`--no-window`) |
| Bridge port | **8212** — Newton. (8211 is a *separate* PhysX `isaacsim.exp.full` kit from a conda env) |
| `get_active_physics_engine()` | `newton` |
| `get_available_physics_engines()` | `[('newton', True), ('physx', False)]` — PhysX is **not available**, so an accidental fallback is impossible here |
| `newton.__version__` | `1.0.0` |
| `get_physics_dt()` | `0.002` |
| `get_backend()` / `get_device()` | `numpy` / `cuda:0` |
| Solver type | `TGS`, GPU dynamics + fabric enabled |

## Go2 asset notes

- Spawning the Go2 gives **`joints: []`** until you select the asset's
  `Physics` variant. The variant set is named `Physics` with options
  `["None", "physx"]`, and `None` is the default.
- **`"physx"` here is the VARIANT name, not the physics engine.** It selects
  the rigid-body/joint payload. The kit chooses the engine; selecting this
  variant is *not* a fallback to PhysX and does not violate the Newton-only
  rule.
- With the variant selected: **17 rigid bodies**, articulation root at
  `/World/Tritium/go2/base`, and 12 revolute joints named
  `{FL,FR,RL,RR}_{hip,thigh,calf}_joint` — a clean 1:1 map onto the joint
  names `tritium_lib.models.gait_trajectory` emits (strip the `_joint` suffix).
- A ground plane with a real collider matters for dynamic foot contact; a
  scaled unit cube is not good enough (carried over from the Spot walk lesson
  in `docs/ISAAC-SIM-STATUS.md`).

## Reproducing

```bash
# on the RTX host
./newton_kit.sh restart 8212

# anywhere (no GPU, no Isaac) — generate the trajectory from tritium-lib
python go2_newton_gait.py --emit-gait trot --speed 0.6 -o gait_trot.json

# on the RTX host — drive it and score the motion
python go2_newton_gait.py --gait-file gait_trot.json --seconds 6 \
    --capture go2_gait.png
```

The driver prints non-gameable motion metrics (`displacement_m`,
`height_retained`, `collapsed`, `verdict`) computed from the recorded root
transform trace — a dog that vibrates in place reads `STATIONARY`, and one
that falls over reads `COLLAPSED`. Don't report a gait as working off a
screenshot alone.

## 2026-07-18 — the articulation was never the problem: the solver integrates nothing

The previous entry concluded the Go2 was "present in the tensor view but not
part of what Newton is stepping". That was too narrow, and one of its premises
was wrong. Three findings, in the order they fell out, driven by
`newton_world_probe.py` against a freshly restarted kit.

### 1. A headless kit does not pump its own update loop

With the timeline reporting `is_playing() == True`, the timeline clock advanced
**0.033 s across 6 s of wall clock** — about one frame per bridge call. Physics
time advanced only when something poked the app.

This retroactively explains the old measurement. The gait driver's "471 writes
over 7.8 s at ~60 Hz" were real writes into a world that barely stepped between
them, and the `sim_time` it watched advancing was an artifact of *its own
polling*. Sleeping on the client and expecting the kit to simulate does not
work.

The stepping primitive the lane was missing:

```python
app = omni.kit.app.get_app()
for _ in range(240):
    app.update()      # 240 updates -> +4.08 s of sim time, reproducibly
```

`world.step()` is **not** available here: `SimulationContext` only builds its
`PhysicsContext` when `ISAAC_LAUNCHED_FROM_TERMINAL is False`
(`simulation_context.py:150`), which is never true in a kit app. So
`world.step()` dies on `'NoneType' object has no attribute '_step'`. Lead 1 of
the previous entry — "build the scene through `World`" — is therefore a **dead
end** for this deployment: `World()` in a kit app skips physics-context setup
entirely.

`world.reset()` does work, but only after `SimulationManager.set_backend("torch")`
and only if nothing is registered in `world.scene` — an `XformPrim.post_reset()`
feeds numpy arrays into `isaacsim.core.utils.**torch**.transformations` and
raises `'numpy.ndarray' object has no attribute 'detach'`. Same
wrapper-assumes-torch bug family as the `SingleArticulation` CUDA fault above.
Use a plain USD ground plane, not `world.scene.add_default_ground_plane()`.

### 2. Read poses from the physics view, never off USD

Physics writes to Fabric and does not write back to USD attributes, so
`UsdGeom.Xformable.GetLocalTransformation()` returns the *authored* value
forever. A healthy falling body looks frozen. Every pose in this entry is read
through `SM.get_physics_sim_view()` views.

### 3. The scene's gravity is `(nan, nan, nan)`

`SM.get_physics_scenes()[0].get_gravity()` returned `(nan, nan, nan)` on a kit
whose solver is `newton`, integrator `euler`, dt `0.002`. Authoring
`gravityDirection`/`gravityMagnitude` on the USD `/PhysicsScene` while stopped
fixes the reading — it then reports `(0, 0, -9.81)`.

**But fixing gravity did not make anything fall.** Which leads to the finding
that supersedes the previous entry's conclusion:

### The real blocker: nothing is simulated, articulation or not

A control experiment removes the Go2 from the question entirely. A plain
`UsdGeom.Cube` with `RigidBodyAPI` + `CollisionAPI` + a 1 kg `MassAPI`, sitting
3 m above a collider ground plane, read through `NewtonRigidBodyView`:

| Condition | sim time advanced | drop |
|---|---|---|
| default (gravity NaN) | 4.08 s | **0.0 m** |
| gravity authored `(0,0,-9.81)` | 4.08 s | **0.0 m** |
| everything authored before the process's FIRST play | 4.08 s | **0.0 m** |

The Go2 under the same pumping, with all 12 drives zeroed and its base at
z=0.8: `max_dof_delta 0.0 rad`, `root_drop 0.0 m`, verdict `NOT_STEPPED`.

So the Newton solver in this kit build **advances its clock but integrates
nothing** — for a free rigid body as much as for a 12-DOF articulation. The
gait lane is not blocked on our trajectory, our joint mapping, or the
articulation's presence in the model. It is blocked one layer below all of
that.

> **Honesty note.** This entry is a *diagnosis*, not a fix — nothing here made
> a body move. The viewport capture that would corroborate it visually was not
> obtained this tick (the camera-posing snippet errored before `/sim/capture`),
> so the evidence is numeric only: three independent bodies, two different
> tensor-view types, consistent zeros while sim time advanced 4.08 s.

**Start the next tick here**, cheapest first:

1. **Is the Newton model empty?** Inspect the solver's own model — body count,
   shape count, joint count — via the `newton` package directly rather than
   through the Isaac wrappers. A model with zero bodies would explain every
   zero above and would be visible immediately.
2. **Does the shipped Newton sample fall?** Run whatever
   `newton`/`isaacsim.physics.newton` example the build ships, unmodified. If
   NVIDIA's own sample is also frozen, this is a broken kit/build, not our
   scene — and the fix is a rebuild or a different Isaac release, not more
   scene-authoring.
3. **Does `/PhysicsScene` need to be Newton-owned?** The scene reporting NaN
   gravity by default smells like a scene the Newton backend adopted rather
   than created. Try letting the Newton extension create its own scene.
4. Only after a **cube visibly falls** is it worth returning to the gait.

---

## 2026-07-18, tick 10 — the body walks, and one third of the time it does not

The previous entry's four-tick diagnosis was right about the symptom and the
tick-9 entry found the cause: a zero-thickness ground quad that qhull cannot
hull, whose exception latched the physics extension's `_initializing` flag and
disabled integration process-wide while the clock kept advancing.

**This driver was still authoring that exact ground.** `SCENE_TEMPLATE` opened
with `GroundPlane(prim_path="/World/GroundPlane", z_position=0.0)`, so every
gait run this lane ever made was commanding joints into a world that never
integrated. Replacing it with a box slab from
`tritium_lib.geo.collider_shape.ground_slab` (rank 3, surface at z=0) is the
whole fix. Nothing about the trajectory or the joint mapping changed.

### The body walks

A trot at `--speed 0.6` over 6 s of sim, driven through the Newton articulation
view, 353 physics steps:

| run | forward dx | max tilt | verdict |
|---|---|---|---|
| trot | **+1.34 m** | 28.0° | WALKED |

**The control is what makes that mean something.** The same scene, same drives,
same 353 physics steps, with every frame of the table frozen at the stand pose:

| run | displacement | verdict |
|---|---|---|
| trot gait | **1.49 m** | MOVED |
| frozen stand pose | **0.05 m** | STATIONARY |

A 30× ratio under identical conditions. The motion comes from the leg
trajectory, not from solver drift, scene settling, or the body sliding down
something. Confirmed visually against a provably static camera
(`/World/GaitCam` at `(1, -7, 2.6)`, a non-physics prim): the Go2 is a third of
the way across the frame at t=1.5 s and most of the way across at t=6 s.
Evidence: `docs/images/isaac/newton-gait-walk-and-tumble-2026-07-18.png`.

### The metric that certified a robot lying on its back

The t=4.5 s frame of that contact sheet shows the Go2 **upside down with its
legs in the air**. Its score card for that run read `displacement_m: 1.27`,
`height_retained: 0.89`, `collapsed: false`, `verdict: MOVED`. Every number is
correct and the conclusion is completely wrong.

`collapsed` watched height, and an inverted quadruped occupies almost exactly
the height of a standing one — the body is a similar distance off the floor
either way. **Height cannot see rotation.** A robot that flips and skitters
along on its shoulders covers ground and passes every gate this scorer had.

The fix is `tritium_lib.geo.body_attitude`: the angle between the body's own up
axis and world up — 0° standing, 90° on its side, 180° on its back. Sliding and
bouncing cannot move it, and yaw deliberately does not register, because a
walking body changes heading constantly and that is not a fall. `score_trace`
now reports `max_tilt_deg` and ranks `TUMBLED` **above** distance, since a
tumbling body is precisely the thing distance would otherwise reward.

### Honest stability: 4 of 6

Six identical 6 s trials:

| trial | forward dx | max tilt | verdict |
|---|---|---|---|
| 1 | +1.457 m | 28.0° | WALKED |
| 2 | +1.455 m | 25.9° | WALKED |
| 3 | +1.373 m | 23.7° | WALKED |
| 4 | −0.249 m | 180.0° | TUMBLED |
| 5 | +1.566 m | 25.8° | WALKED |
| 6 | +1.269 m | 179.8° | TUMBLED |

**67% success.** A clean walk covers 1.37–1.57 m in 6 s (≈0.25 m/s) with peak
tilt 24–28°, comfortably inside the 45° gate — that band is stride lean, not
instability. The failures are total: 180°, fully inverted.

Note trial 6 especially. It travelled **1.27 m and ended on its back** — a
result the old scorer would have called a successful walk, and the single best
argument for keeping the attitude gate.

**So: the gait is real and it is open-loop.** The trajectory is a fixed
kinematic table with no feedback, so nothing corrects an accumulating roll;
whether a given run survives depends on how the initial contact transient
happens to settle. That is the next piece of work, and it is a controls
problem rather than a physics one:

1. **Close the loop on attitude.** Feed body roll/pitch back into hip targets.
   A stabilizer that only ever fights tilt should take 67% toward the high 90s
   without touching the gait table.
2. **Settle before walking.** Every failure develops out of the first stride.
   Hold the stand pose until the body's tilt and height are quiet, *then* start
   the cycle.
3. **Report the rate, never a single run.** One trial of a 67%-stable gait is a
   coin flip that reads as proof either way. `--trials N` belongs in this
   script.

---

## 2026-07-18, tick 13 — two silent no-ops between "the code ran" and "the body moved"

Building the push-recovery experiment surfaced two failures that are dangerous
for the same reason: in both, everything reports success and nothing happens.

### 1. `set_root_velocities()` is silently discarded on a floating-base articulation

The natural way to deliver an impulse is to read the root velocity, add `J/m`,
and write it back. `NewtonArticulationView.set_root_velocities()` accepts the
call, raises nothing, and the solver **discards it entirely** — the velocity
reads back bit-identical to what it was before the write.

This is worse than an exception. The schedule fired, the kick counter
incremented, and the first live run cheerfully reported **"recovered 1/1"** for
an experiment in which the robot was never pushed. Only a read-back of the
velocity immediately after the write exposed it:

```
dv_measured=[0.0, 0.0, 0.0]      # <-- the entire experiment, silently vacuous
```

**Deliver impulses as a force held over a window instead** (`J = F·T`, via
`view.apply_forces(fd, idx, True)` with `fd` shaped `(count, links, 3)` and the
force on link 0). Forces go through the solver's own accumulation path and
actually move the body:

```
dv_measured=[-0.291, 0.5443, 0.3022]   # real, and ~half of J/m because the
                                        # feet are in contact and friction
                                        # absorbs the rest
```

**The general rule this earns:** "the actuation call returned" is never
evidence the body was actuated. Read the state back and compare. Every
disturbance run now records `vel_before`, `vel_after` and `measured_dv_mps`,
and a trial whose kick did not land is reported `NOT_APPLIED` rather than
folded into a recovery rate.

### 2. The kit caches tritium-lib, including the *directory listing*

The kit is a long-lived process, so a `tritium-lib` edit is invisible to it.
Purging `sys.modules` of `tritium_lib*` is the obvious half of the fix and is
**not sufficient**: `importlib`'s `FileFinder` caches each package directory's
contents, so a module file *added* to a package after the kit last scanned it
raises `ModuleNotFoundError` for a file that is plainly on disk. Both halves
are now in the driver preamble:

```python
for _m in [m for m in _sys.modules if m == "tritium_lib" or m.startswith("tritium_lib.")]:
    del _sys.modules[_m]
import importlib as _importlib
_importlib.invalidate_caches()
```

Without this, every lib edit costs a kit restart.

### Calibrating the push

Impulse magnitude was swept rather than guessed, since a disturbance that
always tumbles both arms measures nothing and one that tumbles neither
measures nothing either:

| J (N·s) | open-loop | closed-loop |
|---|---|---|
| 2 | — | WALKED, peak 7.1° |
| 3 | **TUMBLED, 179.9°** | **WALKED, peak 8.1°** |
| 5 | TUMBLED, 179.9° | TUMBLED, 162.6° |
| 8 | TUMBLED, 170.2° | — |
| 15 | — | TUMBLED, 150.7° (thrown 8 m) |

**J = 3 N·s lateral is the discriminating level** — the only one where the two
arms disagree. Above 5 N·s the disturbance overwhelms the controller; below 2
it does not challenge it. Note the run is 8 s with the kick at t=3 s, so the
first ~2 s of start-up transient (peak tilt 22–25°) is excluded from the
post-kick score by construction — `score_recovery` measures peak *after* the
disturbance and reports the pre-kick worst separately as `baseline_deg`.
