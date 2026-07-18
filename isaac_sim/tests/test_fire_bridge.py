# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""fire_bridge without a GPU: the payloads it sends and the verdicts it gives."""

import json

import pytest

from tritium_lib.geo.camera_mount import CameraMount
from tritium_lib.geo.hitscan import BoxTarget, SphereTarget
from tritium_lib.geo.isaac_frame import LocalPose

from isaac_sim_addon.clients.fire_bridge import (
    DEFAULT_SPECS,
    DEFAULT_TERRAIN_PRIMS,
    FireBridge,
    TargetSpec,
    aim_at,
    build_targets_src,
    draw_shot_src,
    grade_trial,
    is_terrain,
    main,
    parse_scene,
    read_scene_src,
    run_trial,
)


class FakeStage:
    """A stage that remembers what it was told and answers what it holds."""

    def __init__(self, body=(0.0, 0.0, 0.4), heading=0.0, targets=None,
                 terrain=None):
        self.body = body
        self.heading = heading
        self.targets = targets if targets is not None else []
        self.terrain = terrain if terrain is not None else []
        self.calls = []

    def __call__(self, path, payload):
        code = payload.get("code", "")
        self.calls.append((path, code))
        if path != "/execute":
            return {"status": "success", "result": {"captured": path}}
        if "authored" in code:
            names = [t["id"] for t in self.targets]
            return _ok({"authored": names})
        if '"body"' in code or "heading" in code:
            return _ok(
                {
                    "body": {"pos": list(self.body), "heading_deg": self.heading},
                    "targets": self.targets,
                    "terrain": self.terrain,
                }
            )
        return _ok("tracer")


#: The slab lidar_server authors: 50 x 50 x 1 m, top face at z = 0.
GROUND = {"path": "/World/GroundSlab",
          "min": [-25.0, -25.0, -1.0], "max": [25.0, 25.0, 0.0]}


def _ok(value):
    return {"status": "success", "result": {"stdout": "", "stderr": "", "return_value": value}}


def _target(tid, e, n, u, r=0.4):
    return {"id": tid, "pos": [e, n, u], "radius": r}


# --- the source it sends --------------------------------------------------


def test_targets_source_sweeps_orphans_from_a_previous_run():
    """The stale-prim defect that invalidated two trials of tick 17."""
    src = build_targets_src([TargetSpec("a", 0.0, 4.0, 0.5)])
    assert "RemovePrim" in src
    assert "wanted" in src


def test_targets_source_reauthors_rather_than_reusing_a_path():
    src = build_targets_src([TargetSpec("a", 0.0, 4.0, 0.5)])
    # The prim is removed before it is defined, so a leftover of the same name
    # cannot survive with the new run's label on it.
    assert src.index("stage.RemovePrim(path)") < src.index("UsdGeom.Sphere.Define")


def test_targets_source_carries_the_requested_geometry():
    src = build_targets_src([TargetSpec("dummy", 1.5, -2.0, 0.75, radius_m=0.3)])
    assert "1.5" in src and "-2.0" in src and "0.75" in src and "0.3" in src


def test_scene_source_reads_body_and_targets_in_one_snippet():
    """Two round trips let a walking body move between pose and targets."""
    src = read_scene_src()
    assert src.count("result = ") == 1
    assert "targets" in src and "heading" in src


def test_scene_source_scales_the_radius_by_the_world_transform():
    assert "scale" in read_scene_src()


def test_scene_source_raises_when_the_body_prim_is_absent():
    """A missing body must fail loudly, not silently fire from the origin."""
    assert "raise RuntimeError" in read_scene_src()


def test_tracer_colours_differ_between_hit_and_miss():
    hit = draw_shot_src((0, 0, 0), (0, 5, 0), True)
    miss = draw_shot_src((0, 0, 0), (0, 5, 0), False)
    assert hit != miss
    assert "HIT" in hit


def test_tracer_is_transient_not_authored_geometry():
    """A tracer prim would show up as an obstacle and a LiDAR return."""
    src = draw_shot_src((0, 0, 0), (0, 5, 0), True)
    assert "debug_draw" in src
    assert "Define" not in src


def test_every_generated_snippet_compiles():
    """A generated source string can be broken by its own quoting."""
    for src in (
        build_targets_src(DEFAULT_SPECS),
        read_scene_src(),
        draw_shot_src((0, 0, 0), (1, 2, 3), True),
    ):
        compile(src, "<generated>", "exec")


# --- reading the stage back ----------------------------------------------


def test_parse_scene_builds_lib_types():
    pose, targets = parse_scene(
        {"body": {"pos": [1.0, 2.0, 0.4], "heading_deg": 90.0},
         "targets": [_target("a", 5.0, 2.0, 0.5)]}
    )
    assert isinstance(pose, LocalPose)
    assert pose.east_m == 1.0 and pose.heading_deg == 90.0
    assert targets[0] == SphereTarget("a", 5.0, 2.0, 0.5, 0.4)


def test_parse_scene_normalises_a_wrapped_heading():
    pose, _ = parse_scene({"body": {"pos": [0, 0, 0], "heading_deg": 450.0}, "targets": []})
    assert pose.heading_deg == pytest.approx(90.0)


# --- aiming ---------------------------------------------------------------


def test_aim_at_puts_the_round_on_the_target():
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.4, heading_deg=0.0)
    target = SphereTarget("t", east_m=3.0, north_m=5.0, up_m=1.2, radius_m=0.4)

    bridge = FireBridge(transport=FakeStage(
        body=(0.0, 0.0, 0.4), heading=0.0,
        targets=[_target("t", 3.0, 5.0, 1.2)],
    ))
    shot = bridge.fire(aim_at(body, target), draw=False)

    assert shot.hit is True
    assert shot.target_id == "t"


def test_aim_at_works_when_the_body_is_not_facing_north():
    """The heading-rotation bug that a north-facing-only suite never sees."""
    body = LocalPose(east_m=2.0, north_m=-1.0, up_m=0.4, heading_deg=215.0)
    target = SphereTarget("t", east_m=-4.0, north_m=6.0, up_m=0.9, radius_m=0.4)

    bridge = FireBridge(transport=FakeStage(
        body=(2.0, -1.0, 0.4), heading=215.0,
        targets=[_target("t", -4.0, 6.0, 0.9)],
    ))
    assert bridge.fire(aim_at(body, target), draw=False).hit is True


def test_aim_at_solves_from_the_pivot_not_the_body_origin():
    """Parallax: the mount sits forward and up, and it matters up close."""
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.0, heading_deg=0.0)
    # A tight target very close by, offset laterally: solving from the body
    # origin instead of the pivot misses it.
    target = SphereTarget("t", east_m=1.0, north_m=1.0, up_m=0.55, radius_m=0.06)
    mount = aim_at(body, target)

    bridge = FireBridge(transport=FakeStage(
        body=(0.0, 0.0, 0.0), heading=0.0,
        targets=[_target("t", 1.0, 1.0, 0.55, r=0.06)],
    ))
    assert bridge.fire(mount, draw=False).hit is True


def test_aim_at_pan_stays_within_half_a_turn():
    """A pan of 350 deg is a 10 deg slew the wrong way round a real turret."""
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.0, heading_deg=10.0)
    target = SphereTarget("t", east_m=1.0, north_m=0.0, up_m=0.0)
    assert -180.0 <= aim_at(body, target).pan_deg <= 180.0


# --- grading --------------------------------------------------------------


def _shot(hit, target_id=None, rng=None, miss=None):
    from tritium_lib.geo.hitscan import Muzzle, ShotResult

    return ShotResult(
        hit=hit,
        muzzle=Muzzle(0.0, 0.0, 0.0, 0.0, 0.0),
        max_range_m=60.0,
        target_id=target_id,
        range_m=rng,
        miss_distance_m=miss,
    )


def test_a_hit_with_a_control_miss_passes():
    v = grade_trial(_shot(True, "t", 5.0), _shot(False, miss=8.0), "t")
    assert v["pass"] is True


def test_hitting_something_other_than_the_intended_target_fails():
    """Hitting SOMETHING is not aiming."""
    v = grade_trial(_shot(True, "other", 5.0), _shot(False, miss=8.0), "t")
    assert v["hit_intended_target"] is False
    assert v["pass"] is False


def test_a_control_that_also_hits_fails_the_trial():
    """If the off-axis round hits too, the ray is not discriminating and the
    aimed hit carries no information at all."""
    v = grade_trial(_shot(True, "t", 5.0), _shot(True, "t", 5.0), "t")
    assert v["discriminates"] is False
    assert v["pass"] is False


def test_a_missed_aimed_shot_fails():
    v = grade_trial(_shot(False, miss=1.0), _shot(False, miss=9.0), "t")
    assert v["pass"] is False


# --- the whole trial ------------------------------------------------------


def test_run_trial_fires_both_arms_and_passes_on_a_clean_pair():
    stage = FakeStage(
        body=(0.0, 0.0, 0.4), heading=0.0,
        targets=[_target("dummy_near", 0.0, 4.0, 0.5), _target("dummy_far", 0.0, 20.0, 0.5)],
    )
    report = run_trial(FireBridge(transport=stage), DEFAULT_SPECS)

    assert report["verdict"]["pass"] is True
    assert report["aimed"]["target_id"] == "dummy_near"
    assert report["control"]["hit"] is False
    assert report["control"]["miss_distance_m"] > 0.0


def test_run_trial_aims_at_the_nearest_target():
    stage = FakeStage(
        body=(0.0, 0.0, 0.4), heading=0.0,
        targets=[_target("far", 0.0, 20.0, 0.5), _target("near", 0.0, 4.0, 0.5)],
    )
    report = run_trial(FireBridge(transport=stage), DEFAULT_SPECS)
    assert report["aimed"]["target_id"] == "near"


def test_run_trial_grades_against_the_stage_not_the_request():
    """Ask for a target at 4 m, have the stage report it somewhere else.

    The verdict must follow the stage.  This is the stale-prim failure mode
    made explicit: if the client graded against DEFAULT_SPECS, it would report
    a hit on geometry that is not there.
    """
    stage = FakeStage(
        body=(0.0, 0.0, 0.4), heading=0.0,
        targets=[_target("dummy_near", 12.0, -9.0, 0.5)],
    )
    report = run_trial(FireBridge(transport=stage), [TargetSpec("dummy_near", 0.0, 4.0, 0.5)])

    assert report["targets"][0]["pos"] == [12.0, -9.0, 0.5]
    assert report["aimed"]["hit"] is True
    # And the round travelled the STAGE distance, not the requested 4 m.
    assert report["aimed"]["range_m"] > 10.0


def test_run_trial_refuses_a_stage_with_no_targets():
    stage = FakeStage(targets=[])
    with pytest.raises(RuntimeError, match="no targets"):
        run_trial(FireBridge(transport=stage), DEFAULT_SPECS)


def test_a_trace_can_be_regraded_offline():
    """The record carries the muzzle and the aim, not just a boolean."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])
    report = run_trial(FireBridge(transport=stage), DEFAULT_SPECS)

    record = json.loads(json.dumps(report))  # must survive a JSON round trip
    assert record["aimed"]["muzzle"]["heading_deg"] is not None
    assert len(record["aimed"]["aim"]) == 3


def test_bridge_requires_a_transport_or_a_host():
    with pytest.raises(ValueError):
        FireBridge()


def test_bridge_raises_on_a_failed_execute():
    def broken(path, payload):
        return {"status": "error", "message": "boom"}

    with pytest.raises(ConnectionError):
        FireBridge(transport=broken).read_scene()


def test_print_src_needs_no_bridge_and_no_gpu():
    assert main(["--print-src"]) == 0


# --- terrain: the round stops at the ground -------------------------------


def test_scene_source_reads_the_terrain_prims():
    """The snippet must read the slab's world AABB in the SAME snapshot as
    the body and targets, and still carry exactly one result assignment."""
    src = read_scene_src()
    for prim in DEFAULT_TERRAIN_PRIMS:
        assert prim in src
    assert "BBoxCache" in src
    assert src.count("result = ") == 1
    compile(src, "<generated>", "exec")


def test_parse_scene_builds_terrain_boxes():
    _, targets = parse_scene(
        {"body": {"pos": [0.0, 0.0, 0.4], "heading_deg": 0.0},
         "targets": [], "terrain": [GROUND]}
    )
    assert len(targets) == 1
    box = targets[0]
    assert isinstance(box, BoxTarget)
    assert box.target_id == "terrain:/World/GroundSlab"
    assert is_terrain(box.target_id)
    assert box.max_up_m == 0.0 and box.min_up_m == -1.0


def test_parse_scene_without_terrain_still_works():
    """A stage with no slab (or an old snippet) parses exactly as before."""
    _, targets = parse_scene(
        {"body": {"pos": [0.0, 0.0, 0.4], "heading_deg": 0.0},
         "targets": [_target("a", 0.0, 4.0, 0.5)]}
    )
    assert targets == [SphereTarget("a", 0.0, 4.0, 0.5, 0.4)]


def test_round_terminates_at_terrain_not_below_it():
    """THE named live defect: the control tracer drew to max range visibly
    BELOW the ground slab, because terrain was not in the target list.  A
    round aimed below the horizon must now stop AT the slab's top face, and
    the tracer must end at that impact — in the MISS colour, because dirt is
    not a target."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[], terrain=[GROUND])
    bridge = FireBridge(transport=stage)
    mount = CameraMount(forward_m=0.25, left_m=0.0, up_m=0.55,
                        pan_deg=0.0, tilt_deg=-45.0)
    shot = bridge.fire(mount)  # draw=True: the tracer is the point

    assert shot.hit is True
    assert is_terrain(shot.target_id)
    impact = shot.impact()
    assert abs(impact[2]) < 1e-9         # AT the top face, not 60 m under it
    assert shot.range_m < 2.0            # ~1.14 m of travel, not max range
    draw_code = stage.calls[-1][1]
    assert str(list(impact)) in draw_code               # tracer ends at impact
    assert "(1.0, 0.93, 0.04, 1.0)" in draw_code        # MISS colour, not HIT


def test_target_behind_terrain_is_occluded():
    """A shot aimed at a target buried beyond the ground hits the TERRAIN."""
    buried = _target("buried", 0.0, 6.0, -2.0)
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.4, heading_deg=0.0)
    tgt = SphereTarget("buried", 0.0, 6.0, -2.0, 0.4)

    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[buried], terrain=[GROUND])
    shot = FireBridge(transport=stage).fire(aim_at(body, tgt), draw=False)
    assert shot.hit is True
    assert is_terrain(shot.target_id)
    assert shot.target_id != "buried"

    # The same aim on a stage WITHOUT the slab reaches it — the terrain is
    # what stopped the round, not a bad solution.
    open_stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0, targets=[buried])
    shot2 = FireBridge(transport=open_stage).fire(aim_at(body, tgt), draw=False)
    assert shot2.hit is True and shot2.target_id == "buried"


def test_clear_line_of_sight_still_hits_over_solid_ground():
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("t", 0.0, 8.0, 1.2)], terrain=[GROUND])
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.4, heading_deg=0.0)
    tgt = SphereTarget("t", 0.0, 8.0, 1.2, 0.4)
    shot = FireBridge(transport=stage).fire(aim_at(body, tgt), draw=False)
    assert shot.hit is True and shot.target_id == "t"


def test_a_control_stopped_by_terrain_still_passes():
    """The whole reason terrain grading exists: with a solid floor most
    control rounds end in the dirt.  That is a MISS of the target — the
    trial must pass, and the verdict must say where the round went."""
    v = grade_trial(_shot(True, "t", 5.0),
                    _shot(True, "terrain:/World/GroundSlab", 3.0), "t")
    assert v["pass"] is True
    assert v["control_missed"] is True
    assert v["control_stopped_by_terrain"] is True
    assert v["discriminates"] is True


def test_an_aimed_round_that_hits_terrain_fails():
    """Shooting the ground is not fire control, whatever the ray says."""
    v = grade_trial(_shot(True, "terrain:/World/GroundSlab", 2.0),
                    _shot(False, miss=5.0), "t")
    assert v["hit_intended_target"] is False
    assert v["discriminates"] is False
    assert v["pass"] is False


def test_run_trial_with_solid_ground_aims_at_spheres_and_passes():
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)],
                      terrain=[GROUND])
    report = run_trial(FireBridge(transport=stage), DEFAULT_SPECS)

    assert report["aimed"]["target_id"] == "dummy_near"
    assert report["verdict"]["pass"] is True
    assert report["terrain"] == [{"id": "terrain:/World/GroundSlab",
                                  "min": [-25.0, -25.0, -1.0],
                                  "max": [25.0, 25.0, 0.0]}]
    # Terrain never appears in the aimable target list.
    assert all(not is_terrain(t["id"]) for t in report["targets"])


def test_run_trial_refuses_a_stage_with_only_terrain():
    """Terrain is not something to aim at: a stage holding nothing but the
    slab has no targets, and the trial must say so."""
    stage = FakeStage(targets=[], terrain=[GROUND])
    with pytest.raises(RuntimeError, match="no targets"):
        run_trial(FireBridge(transport=stage), DEFAULT_SPECS)
