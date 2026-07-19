# Attached sensors — live findings (2026-07-18, Newton kit :8212)

The attached mode (`connectors/attached_sensor_server.py`) exists because of a
process boundary no amount of configuration removes: `camera_server` and
`lidar_server` each boot a PRIVATE `SimulationApp`, and a USD stage is
per-process — so a standalone sensor kit can never image the body the
dedicated Newton kit is stepping.  Attached mode sends the sensors INTO the
running kit (MCP `/execute`), authors prims under `/World/Tritium/Sensors`,
encodes on the kit's update loop, and serves the standalone servers' exact
wire contracts (`/mjpeg` `/depth16` `/mjpeg_right` `/snapshot` `/intrinsics`
`/status` on one port; `/scan` `/status` on another) through the REUSED
`CameraState`/`LidarState` machinery via their `publish()` seams.

## What was demonstrated live (graded 50/50 by `attached_sensors_grade.py`)

One run: `attached_sensors_live_proof.py` against the dedicated Newton kit,
scene = box-slab ground + blue wall at exactly 9.9 m camera depth + red 0.6 m
dynamic cube authored at z=6 + yellow bouncy ball at z=14 + cyan kinematic
cube aloft.  Evidence: `scratch_tritium/attached_proof_02/` on the render
host; grading requires only the evidence dir + `tritium-lib/src`.

* **Newton stepped the scene** (0 → 2637 steps) and the red body fell
  z 5.99 → 3.13 → 0.29 (a clean gravity curve in the pose log) and settled at
  **0.297 m ≈ half-height** — the tick-9 integration proof, re-run.
* **RGB**: 21 frames of the fall are ON CAMERA (aloft → mid-air → landed);
  the red centroid sits within ~1 px of the projected Fabric pose.
* **depth16 is DEPTH, not a picture of depth**: decoded by the canonical
  `tritium_lib.perception.depth_codec` — wall band median **9.900 m**
  (authored 9.9), body band median **5.700 m** (front face at 5.70 from the
  Fabric pose), sky 100% no-return, 654 distinct mm values (an 8-bit hop caps
  at 256), and the blob round-trips bit-exactly through the lib codec.
* **Stereo right is a real second eye**: measured body disparity **10.4 px**
  vs 10.3 px predicted by `fx*B*(1/Z − 1/D)` for the converged pair.
* **LiDAR**: the beam at the body corner's own bearing reads **2.40 m @ −45°**
  (computed corner: 2.40 m); with the body hidden that bearing opens to
  30 m and the global minimum becomes the ball at 3.23 m @ −83° — per-bearing
  agreement no stale sweep can fake.
* **Negative control**: `MakeInvisible` on the body removes it from RGB,
  depth (band reverts to the far scene), stereo and LiDAR within a second;
  `MakeVisible` restores all four.  The frames are re-rendered from the live
  stage, not cached.

## Hard-won build facts (verified by probing, not inferred)

1. **A dynamic rigid body's `xformOp:scale` is DROPPED the moment physics
   owns its transform.**  A 0.6 m cube authored `size=2` + `scale=0.3`
   simulated *and rendered* as its unscaled 2 m self (the lidar measured its
   corner at exactly √2 m — the unscaled geometry's signature).  Author
   dynamic dimensions in geometry attrs (`Cube.size`, `Sphere.radius`);
   statics keep working with scale.
2. **Newton does not register prims authored mid-play.**  A dynamic cube
   added while the timeline ran never appeared in Fabric and never fell.
   Everything physical must exist before `play()`.
3. **A pre-registered kinematic body cannot be released mid-run** on this
   build: flipping `physics:kinematicEnabled` False was a silent no-op (the
   cyan cube hung in the air for the rest of the run).  Spawn-held-release
   choreography is NOT available; plan scenes accordingly.
4. **Poses must be read through `RigidPrim.get_world_poses()` (Fabric).**
   USD (`ComputeLocalToWorldTransform`) and usdrt world-position both stay
   frozen at authored values while physics runs.  The view returns **CUDA
   torch tensors** — convert with `float(t)`, never `.numpy()`.
5. **The first `isaacsim.core.prims` import inside the kit costs ~10 s** and
   blocks the update loop (physics froze until it finished).  Warm the pose
   view before any timing-sensitive phase.
6. **Annotators produce nothing until the timeline plays** in this kit —
   there is no pre-play "still frame" phase; warm-up happens with physics
   already running.
7. **`RigidPrim.apply_forces` did not move a plain rigid body** (applied
   without error, zero displacement read back).  The articulation-view
   `apply_forces` (tick 13) remains the only proven force path; this harness
   uses gravity and visibility toggles instead.

## Honest scope

* The cameras and lidar are world-fixed; nothing here exercises a
  body-MOUNTED attached sensor (`--mount-prim` equivalent) or per-sweep pose.
* One scene, one kit, one GPU tenant configuration; the ball's restitution
  bounce decayed before the moving-pair captures (recorded as informational).
* `uninstall()` is code-reviewed and tested for refusals but was not
  exercised in the graded run (the rig was left serving).
