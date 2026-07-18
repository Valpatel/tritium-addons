"""Headless tests for the route-following + obstacle plumbing in the gait driver.

No Isaac, no GPU, no network.  Two classes of failure are covered, and they
fail in very different ways:

1. **A missed placeholder.**  The scene and driver snippets are Python source
   rendered by string substitution and shipped to a live kit.  A surviving
   ``__TOKEN__`` is a NameError (or a SyntaxError) discovered only after a
   GPU run has been set up, played, and thrown away.  So every rendered
   variant is checked for leftover placeholders AND compiled here, for free.

2. **Two obstacle lists that disagree.**  The planner routes around boxes and
   the solver collides with boxes; the moment those are different numbers,
   the demo either detours around nothing or walks through a wall while the
   log reports a clean arrival.  The round-trip test pins them together.

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: Anything of the form ``__NAME__`` left in rendered source is an unfilled
#: slot.  The dotted form catches the nastier variant where an inner token was
#: substituted and left the surrounding underscores behind (``__1000.0__``).
PLACEHOLDER_RE = re.compile(r"__[A-Za-z0-9_.]+__")


def _load_driver():
    """Import the example by path -- examples/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "go2_newton_gait", EXAMPLES / "go2_newton_gait.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


driver = _load_driver()

GAIT = {"phases": [0.0, 0.5], "frames": [[0.0] * 12, [0.1] * 12],
        "stride_hz": 2.0}
ROUTE = {
    "waypoints": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
    "lookahead": 0.8, "cruise": 0.6, "max_angular": 0.8,
    "goal_tolerance": 0.35, "slow_radius": 0.0,
    "track": 0.26, "nominal": 0.6,
}


class _Args:
    """Just the fields the route/plan helpers read off argparse."""

    speed = 0.6
    lookahead = 0.8
    max_angular = 0.8
    goal_tolerance = 0.35
    slow_radius = 0.0
    steer_track = 0.26
    plan_to = "6,0"
    plan_resolution = 0.15
    # Two DIFFERENT radii on purpose, and the gap between them is the point:
    # plan_clearance is the planner's inflation preference, body_radius is the
    # hull that actually collides.  Setting them equal here would let a bug
    # that swaps the two pass every test in this file.
    plan_clearance = 0.30
    body_radius = 0.16
    # Default OFF: the open-loop arm must stay the default path so the A/B
    # keeps a control.
    closed_loop_yaw = False
    yaw_kp = 1.0
    yaw_ki = 6.0
    yaw_max = 4.0


# --------------------------------------------------------------- parse_points
def test_parse_points_reads_a_polyline():
    assert driver.parse_points("0,0;1.5,-2") == [(0.0, 0.0), (1.5, -2.0)]


def test_parse_points_tolerates_whitespace_and_a_trailing_semicolon():
    assert driver.parse_points(" 1 , 2 ; 3,4 ; ") == [(1.0, 2.0), (3.0, 4.0)]


def test_parse_points_of_nothing_is_no_waypoints():
    assert driver.parse_points("") == []
    assert driver.parse_points("  ") == []


@pytest.mark.parametrize("bad", ["1", "1,2,3", "x,2"])
def test_parse_points_rejects_malformed_pairs(bad):
    with pytest.raises((SystemExit, ValueError)):
        driver.parse_points(bad)


# ------------------------------------------------------------ parse_obstacles
def test_parse_obstacles_seats_the_box_on_the_slab():
    """center z == HZ, so the box's underside is exactly the slab top (z=0).

    A box seated anywhere else is either buried (the body walks over a stub)
    or floating (the body walks under a wall), and in both cases the planner's
    detour is around something the solver does not have.
    """
    (obs,) = driver.parse_obstacles(["3,0,0.6,1.5,0.5"])
    assert obs.center == (3.0, 0.0, 0.5)
    assert obs.half_extents == (0.6, 1.5, 0.5)
    assert obs.z_min == pytest.approx(0.0)
    assert obs.z_max == pytest.approx(1.0)


def test_parse_obstacles_seated_box_blocks_the_body_band():
    """The seating rule is only worth anything if the planner then sees it."""
    from tritium_lib.planning.scene_costmap import DEFAULT_BODY_BAND

    (obs,) = driver.parse_obstacles(["3,0,0.6,1.5,0.5"])
    assert obs.intersects_band(*DEFAULT_BODY_BAND)


def test_parse_obstacles_numbers_prims_the_way_the_scene_does():
    """The scene names its boxes /World/obstacle_%d by enumeration index."""
    obstacles = driver.parse_obstacles(["1,0,1,1,0.5", "2,0,1,1,0.5"])
    assert [o.prim_path for o in obstacles] == [
        "/World/obstacle_0", "/World/obstacle_1"]


def test_parse_obstacles_of_none_is_empty():
    assert driver.parse_obstacles(None) == []
    assert driver.parse_obstacles([]) == []


@pytest.mark.parametrize("bad", ["1,2,3", "1,2,3,4,5,6", "1,2,3,4,x"])
def test_parse_obstacles_rejects_malformed_specs(bad):
    with pytest.raises((SystemExit, ValueError)):
        driver.parse_obstacles([bad])


def test_parse_obstacles_rejects_a_negative_half_extent_as_a_cli_error():
    """A typo in --obstacle should read as a CLI error, not a library traceback."""
    with pytest.raises(SystemExit):
        driver.parse_obstacles(["1,1,-0.5,1,0.5"])


# ------------------------------------------------------------- obstacle_specs
def test_obstacle_specs_round_trips_the_geometry():
    """The box the scene authors IS the box the planner routes around.

    Rebuilding a SceneObstacle from the emitted spec must land on the same
    numbers the planner was handed.  This is the single invariant that keeps
    one obstacle list from becoming two.
    """
    from tritium_lib.planning.scene_costmap import SceneObstacle

    obstacles = driver.parse_obstacles(["3,-1,0.6,1.5,0.4", "-2,4,1.0,0.25,0.8"])
    specs = driver.obstacle_specs(obstacles)
    assert len(specs) == len(obstacles)
    for obs, spec in zip(obstacles, specs):
        assert set(spec) == {"center", "half"}
        rebuilt = SceneObstacle(
            prim_path=obs.prim_path,
            center=tuple(spec["center"]),
            half_extents=tuple(spec["half"]),
        )
        assert rebuilt.center == obs.center
        assert rebuilt.half_extents == obs.half_extents
        assert (rebuilt.z_min, rebuilt.z_max) == (obs.z_min, obs.z_max)


def test_obstacle_specs_emits_plain_json_able_types():
    """The specs go through repr() into remote source, so no numpy, no dataclass."""
    import json

    specs = driver.obstacle_specs(driver.parse_obstacles(["3,0,0.6,1.5,0.5"]))
    assert json.loads(json.dumps(specs)) == [
        {"center": [3.0, 0.0, 0.5], "half": [0.6, 1.5, 0.5]}]


def test_obstacle_specs_refuses_a_yawed_box():
    """The scene authors axis-aligned boxes; a dropped yaw is a silent divergence."""
    from tritium_lib.planning.scene_costmap import SceneObstacle

    yawed = SceneObstacle("/World/obstacle_0", (1.0, 0.0, 0.5),
                          (1.0, 0.5, 0.5), yaw_deg=30.0)
    with pytest.raises(ValueError, match="yaw"):
        driver.obstacle_specs([yawed])


def test_obstacle_specs_of_nothing_is_an_empty_list():
    assert driver.obstacle_specs([]) == []


# ----------------------------------------------------------------- route_spec
def test_route_spec_is_none_without_waypoints():
    """The unrouted arm must run the ORIGINAL code path, or it is not a control."""
    assert driver.route_spec(_Args(), []) is None
    assert driver.route_spec(_Args(), None) is None


def test_route_spec_carries_every_follower_gain():
    spec = driver.route_spec(_Args(), [(0.0, 0.0), (2.0, 1.0)])
    assert spec["waypoints"] == [[0.0, 0.0], [2.0, 1.0]]
    assert spec["lookahead"] == 0.8
    assert spec["cruise"] == 0.6
    assert spec["max_angular"] == 0.8
    assert spec["goal_tolerance"] == 0.35
    assert spec["slow_radius"] == 0.0
    assert spec["track"] == 0.26
    assert spec["nominal"] == 0.6


def test_route_spec_waypoints_survive_repr_into_remote_source():
    """Tuples would still repr fine, but the driver indexes p[0]/p[1] -- pin it."""
    spec = driver.route_spec(_Args(), [(1.0, 2.0)])
    assert all(isinstance(p, list) and len(p) == 2 for p in spec["waypoints"])


# --------------------------------------------------- rendered code: scene side
@pytest.mark.parametrize("obstacles", [
    pytest.param(None, id="no_obstacles"),
    pytest.param(driver.parse_obstacles(["3,0,0.6,1.5,0.5"]), id="one_obstacle"),
    pytest.param(driver.parse_obstacles(["3,0,0.6,1.5,0.5", "-1,2,1,1,0.3"]),
                 id="two_obstacles"),
])
def test_scene_code_leaves_no_placeholder_and_compiles(obstacles):
    """A surviving __TOKEN__ costs a whole GPU run to discover.  Not here."""
    code = driver.build_scene_code(1000.0, 50.0, obstacles=obstacles)
    assert PLACEHOLDER_RE.findall(code) == []
    compile(code, "<scene>", "exec")


def test_scene_code_substitutes_the_obstacles_json():
    obstacles = driver.parse_obstacles(["3,0,0.6,1.5,0.5"])
    code = driver.build_scene_code(1000.0, 50.0, obstacles=obstacles)
    assert "__OBSTACLES_JSON__" not in code
    assert repr(driver.obstacle_specs(obstacles)) in code


def test_scene_code_without_obstacles_authors_an_empty_list():
    code = driver.build_scene_code(1000.0, 50.0)
    assert "_obs_specs = []" in code                  # loop ships, body inert
    assert "/World/obstacle_%d" in code               # authoring block still there


def test_scene_code_re_authors_obstacles_instead_of_skipping_existing_ones():
    """Regression: a stale obstacle prim silently outlived the run that made it.

    The scene builder used to ``continue`` past any /World/obstacle_N that
    already existed, on the reasonable-sounding theory that a prim at the
    right path is the right prim.  On a live Isaac that outlives any single
    trial it is not: a box authored by an earlier run kept its OWN centre and
    half-extents, so the planner routed around the geometry it was handed
    while the solver held something else entirely.

    Measured on the live Newton stage before the fix: asked for centre
    (1.5, 0.4) half (0.5, 0.5, 0.35) -- i.e. AABB x 1.0..2.0, y -0.1..0.9 --
    and got x 1.65..2.35, y -1.0..1.0.  Every clearance the scorer produced
    that run graded a box that was not there, which is worse than no metric,
    because it reads as a measurement.
    """
    code = driver.build_scene_code(
        1000.0, 50.0, obstacles=driver.parse_obstacles(["3,0,0.6,1.5,0.5"]))
    assert "stage.RemovePrim(_path)" in code
    # The skip is what caused the bug; its absence is the fix.  Matched as the
    # exact guard-then-define sequence rather than a bare "continue", which
    # appears legitimately elsewhere in the scene code.
    assert "continue\n    _box = UsdGeom.Cube.Define" not in code


def test_scene_code_sweeps_obstacles_left_by_a_longer_previous_run():
    """Two boxes then one must leave ONE box on the stage, not two.

    Re-authoring only the boxes this run asks for still leaves obstacle_1
    standing when the previous run had two and this one has one -- a phantom
    wall the planner has never heard of, in the one direction the body was
    told is clear.
    """
    code = driver.build_scene_code(
        1000.0, 50.0, obstacles=driver.parse_obstacles(["3,0,0.6,1.5,0.5"]))
    assert "len(_obs_specs) + _stale" in code


# -------------------------------------------------- rendered code: driver side
@pytest.mark.parametrize("route", [
    pytest.param(None, id="no_route"),
    pytest.param(ROUTE, id="with_route"),
])
def test_driver_code_leaves_no_placeholder_and_compiles(route):
    code = driver.build_driver_code(
        GAIT, 6.0, 1000.0, 50.0, 5, route=route)
    assert PLACEHOLDER_RE.findall(code) == []
    compile(code, "<driver>", "exec")


def test_driver_code_substitutes_the_route_json():
    code = driver.build_driver_code(GAIT, 6.0, 1000.0, 50.0, 5, route=ROUTE)
    assert "__ROUTE_JSON__" not in code
    assert repr(ROUTE) in code
    assert "PurePursuitFollower" in code


def test_driver_code_without_a_route_binds_none():
    """The control arm must see _ROUTE = None, not an empty dict."""
    code = driver.build_driver_code(GAIT, 6.0, 1000.0, 50.0, 5, route=None)
    assert "_ROUTE = None" in code


def test_driver_code_route_and_disturbance_coexist():
    """Both features substitute into the same template; neither may eat the other."""
    code = driver.build_driver_code(
        GAIT, 6.0, 1000.0, 50.0, 5, route=ROUTE,
        disturb=[{"at_time": 2.0, "linear": [3.0, 0.0, 0.0], "label": "cli"}],
        twist={"linear": 0.6, "angular": 0.2, "track": 0.26, "nominal": 0.6})
    assert PLACEHOLDER_RE.findall(code) == []
    compile(code, "<driver>", "exec")


# ---------------------------------------------------------------- score_route
def _trace(points):
    """[(x, y)] -> the driver's rows [t, x, y, z, qx, qy, qz, qw]."""
    return [[i * 0.1, x, y, 0.4, 0.0, 0.0, 0.0, 1.0]
            for i, (x, y) in enumerate(points)]


def test_score_route_prefixes_every_field_and_reports_the_verdict():
    collected = {"trace": _trace([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
                 "follow": [], "arrived_at": 4.2}
    route = dict(ROUTE, waypoints=[[0.0, 0.0], [2.0, 0.0]])
    out = driver.score_route(collected, route, [])

    for key in ("route_verdict", "route_reached_goal", "route_final_gap_m",
                "route_max_cross_track_m", "route_rms_cross_track_m",
                "route_min_clearance_m", "route_collided",
                "route_progress_ratio", "route_samples"):
        assert key in out, key
    assert out["route_verdict"] == "REACHED"
    assert out["route_reached_goal"] is True
    assert out["route_final_gap_m"] == pytest.approx(0.0)
    assert out["route_progress_ratio"] == pytest.approx(1.0)
    assert out["route_samples"] == 3
    assert out["route_arrived_at_s"] == 4.2


def test_score_route_verdict_matches_the_dataclass():
    from tritium_lib.control.route_trace import score_route_trace

    positions = [(0.0, 0.0), (1.0, 0.0), (1.2, 0.0)]
    route = dict(ROUTE, waypoints=[[0.0, 0.0], [4.0, 0.0]])
    out = driver.score_route({"trace": _trace(positions)}, route, [])
    expected = score_route_trace(
        positions, [(0.0, 0.0), (4.0, 0.0)], [],
        goal_tolerance_m=route["goal_tolerance"], max_footprint_m=None)
    assert out["route_verdict"] == expected.verdict == "SHORT"
    assert out["route_final_gap_m"] == pytest.approx(expected.final_gap_m)


def test_score_route_grades_the_ground_truth_not_the_followers_self_report():
    """A follower with a broken pose reads zero error while walking off-route.

    The trace says the body went sideways; the follower's own samples claim a
    perfect run.  score_route must side with the trace.
    """
    collected = {
        "trace": _trace([(0.0, 0.0), (1.0, 3.0), (2.0, 3.0)]),
        "follow": [[1.0, 0.0], [2.0, 0.0]],   # self-reported zero cross-track
    }
    route = dict(ROUTE, waypoints=[[0.0, 0.0], [2.0, 0.0]])
    out = driver.score_route(collected, route, [])
    assert out["route_max_cross_track_m"] > 2.0
    assert out["route_verdict"] != "REACHED"
    # the follower's number is still reported -- clearly labelled as its own
    assert out["follower_samples"] == 2
    assert out["follower_final_cross_track_m"] == 0.0


def test_score_route_sees_a_wall_the_body_walked_through():
    """The whole point of handing obstacles to the scorer."""
    obstacles = driver.parse_obstacles(["1.0,0,0.4,2.0,0.5"])
    collected = {"trace": _trace([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])}
    route = dict(ROUTE, waypoints=[[0.0, 0.0], [2.0, 0.0]])
    out = driver.score_route(collected, route, obstacles)
    assert out["route_collided"] is True
    assert out["route_verdict"] == "COLLIDED"
    assert out["route_min_clearance_m"] < 0.0


def test_score_route_does_not_discard_a_wall_as_terrain():
    """A long --obstacle wall is CURATED, not scenery scraped off a stage.

    tritium_lib's default max_footprint_m drops boxes over 100 m as terrain.
    Applied here it would score a body jammed against a 400 m wall as a clean
    arrival, because the hand-typed wall was silently reclassified as world.
    """
    obstacles = driver.parse_obstacles(["3,0,0.5,200,0.5"])
    collected = {"trace": _trace([(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)])}
    route = dict(ROUTE, waypoints=[[0.0, 0.0], [6.0, 0.0]])
    out = driver.score_route(collected, route, obstacles)
    assert out["route_collided"] is True
    assert out["route_verdict"] == "COLLIDED"


def test_score_route_with_no_trace_is_no_trace_not_a_failure():
    out = driver.score_route({"trace": []}, dict(ROUTE), [])
    assert out["route_verdict"] == "NO_TRACE"
    assert out["route_collided"] is False


# ----------------------------------------------------------------- plan_route
def test_plan_route_clears_a_box_squarely_on_the_straight_line():
    """The planned path is graded by the SAME scorer, against the SAME boxes.

    Planning around a box is only meaningful if the resulting polyline
    actually clears it -- so the route is fed back through score_route_trace
    rather than eyeballed.
    """
    from tritium_lib.control.route_trace import score_route_trace

    obstacles = driver.parse_obstacles(["3,0,0.6,1.5,0.5"])
    path = driver.plan_route(_Args(), obstacles)

    assert len(path) >= 2
    assert path[0] == (0.0, 0.0)
    assert path[-1] == (6.0, 0.0)
    # A straight shot would be exactly two points; a detour cannot be.
    assert len(path) > 2

    scored = score_route_trace(path, path, obstacles, max_footprint_m=None)
    assert scored.collided is False
    assert scored.min_clearance_m > 0.0
    assert scored.verdict == "REACHED"


def test_plan_route_detours_around_a_long_wall():
    """A wall that spans most of the corridor still gets routed around."""
    from tritium_lib.control.route_trace import score_route_trace

    obstacles = driver.parse_obstacles(["3,0,0.5,2.5,0.5"])
    path = driver.plan_route(_Args(), obstacles)
    assert len(path) > 2
    assert max(abs(y) for _, y in path) > 2.5   # it went AROUND the end
    scored = score_route_trace(path, path, obstacles, max_footprint_m=None)
    assert scored.collided is False


def test_plan_route_never_returns_a_straight_line_through_a_huge_wall():
    """The silent-terrain trap, on the planning side.

    tritium_lib's default max_footprint_m drops boxes over 100 m across as
    terrain.  Applied to a hand-typed --obstacle, a 400 m wall vanished from
    the costmap and the planner cheerfully returned [(0,0), (6,0)] -- straight
    through a box the scene authors as a solid collider.  Refusing out loud is
    the only acceptable answer: the wall genuinely cannot be routed around
    inside the corridor bounds.
    """
    obstacles = driver.parse_obstacles(["3,0,0.5,200,0.5"])
    with pytest.raises(SystemExit, match="no route"):
        driver.plan_route(_Args(), obstacles)


def test_plan_route_leaves_an_unobstructed_goal_alone():
    path = driver.plan_route(_Args(), [])
    assert path[0] == (0.0, 0.0)
    assert path[-1] == (6.0, 0.0)


def test_plan_route_exits_when_the_body_is_boxed_in():
    walls = driver.parse_obstacles([
        "2,0,0.3,6,0.5", "-2,0,0.3,6,0.5", "0,2,6,0.3,0.5", "0,-2,6,0.3,0.5"])
    with pytest.raises(SystemExit, match="no route"):
        driver.plan_route(_Args(), walls)


def test_plan_route_exits_when_the_goal_is_inside_a_box():
    obstacles = driver.parse_obstacles(["6,0,1.0,1.0,0.5"])
    with pytest.raises(SystemExit, match="no route"):
        driver.plan_route(_Args(), obstacles)


def test_plan_route_exits_on_a_goal_that_parsed_to_nothing():
    args = _Args()
    args.plan_to = " "
    with pytest.raises(SystemExit):
        driver.plan_route(args, [])


def test_planned_route_feeds_route_spec_end_to_end():
    """plan -> spec -> rendered driver source, with no placeholder left."""
    obstacles = driver.parse_obstacles(["3,0,0.6,1.5,0.5"])
    spec = driver.route_spec(_Args(), driver.plan_route(_Args(), obstacles))
    assert spec is not None
    code = driver.build_driver_code(GAIT, 6.0, 1000.0, 50.0, 5, route=spec)
    assert PLACEHOLDER_RE.findall(code) == []
    compile(code, "<driver>", "exec")


# --------------------------------------------------------------- _route_bounds
def test_route_bounds_covers_start_and_goal_with_margin():
    assert driver._route_bounds((0.0, 0.0), (6.0, -2.0), margin_m=4.0) == (
        -4.0, -6.0, 10.0, 4.0)


def test_route_bounds_is_ordered_min_then_max_for_any_goal_quadrant():
    for goal in ((6.0, 2.0), (-6.0, 2.0), (-6.0, -2.0), (6.0, -2.0)):
        min_x, min_y, max_x, max_y = driver._route_bounds(
            (0.0, 0.0), goal, margin_m=4.0)
        assert min_x < max_x and min_y < max_y


def test_route_bounds_stays_small_enough_to_plan_on():
    """The 40M-cell costmap came from bounding the whole stage instead."""
    min_x, min_y, max_x, max_y = driver._route_bounds(
        (0.0, 0.0), (6.0, 0.0), margin_m=4.0)
    cells = ((max_x - min_x) / 0.15) * ((max_y - min_y) / 0.15)
    assert cells < 100_000


# ------------------------------------------- inflation vs footprint radius
# These two pin the distinction that broke the referee on the first live run:
# the planner's clearance is a PREFERENCE satisfied on a discrete grid, while
# the body radius is the hull that actually collides.  Grading with the
# planner's number failed the planner's own optimal path.
def test_route_spec_grades_with_the_body_radius_not_the_plan_clearance():
    spec = driver.route_spec(_Args(), [(0.0, 0.0), (4.0, 0.0)])
    assert spec["clearance"] == _Args.body_radius
    assert spec["clearance"] != _Args.plan_clearance


def test_the_referee_accepts_the_planners_own_optimal_path():
    """A gate the plan itself cannot pass is a broken gate, not a strict one.

    Measured live: at clearance 0.30 on a 0.15 m grid the planner's own route
    scored COLLIDED with min_clearance 0.177, so no run could ever have
    reached the goal no matter how well the body walked.
    """
    obstacles = driver.parse_obstacles(["2.0,0,0.35,1.0,0.4"])
    args = _Args()
    args.plan_to = "4.0,0"
    path = driver.plan_route(args, obstacles)
    scored = driver.score_route(
        {"trace": [[0.0, x, y, 0.0, 0.0, 0.0, 0.0, 1.0] for x, y in path]},
        driver.route_spec(args, path), obstacles)
    assert scored["route_verdict"] == "REACHED", scored


def test_a_straight_line_to_the_same_goal_is_scored_collided():
    """The control arm: the metric must be able to FAIL the obvious cheat."""
    obstacles = driver.parse_obstacles(["2.0,0,0.35,1.0,0.4"])
    args = _Args()
    args.plan_to = "4.0,0"
    path = driver.plan_route(args, obstacles)
    straight = [(i * 4.0 / 20.0, 0.0) for i in range(21)]
    scored = driver.score_route(
        {"trace": [[0.0, x, y, 0.0, 0.0, 0.0, 0.0, 1.0] for x, y in straight]},
        driver.route_spec(args, path), obstacles)
    assert scored["route_verdict"] == "COLLIDED"
    assert scored["route_min_clearance_m"] < 0.0


# ------------------------------------------------------- inner yaw-rate loop
def test_route_spec_defaults_the_yaw_loop_off():
    """The open-loop arm is the control, so it must be what you get by
    default.  A loop that switches itself on silently destroys the only
    baseline the live A/B can be measured against."""
    spec = driver.route_spec(_Args(), [(0.0, 0.0), (3.0, 1.5)])
    assert spec["yaw_closed_loop"] is False


def test_route_spec_carries_the_yaw_gains_when_enabled():
    args = _Args()
    args.closed_loop_yaw = True
    args.yaw_kp, args.yaw_ki, args.yaw_max = 2.0, 9.0, 5.0
    spec = driver.route_spec(args, [(0.0, 0.0), (3.0, 1.5)])
    assert spec["yaw_closed_loop"] is True
    assert spec["yaw_kp"] == 2.0
    assert spec["yaw_ki"] == 9.0
    assert spec["yaw_max"] == 5.0


def test_yaw_settings_survive_repr_into_the_remote_source():
    """The spec is injected into the in-sim script by repr(), so a value that
    does not round-trip arrives as a syntax error inside Isaac rather than a
    failure here."""
    args = _Args()
    args.closed_loop_yaw = True
    spec = driver.route_spec(args, [(0.0, 0.0), (3.0, 1.5)])
    revived = eval(repr(spec))  # noqa: S307 — exactly what the driver does
    assert revived["yaw_closed_loop"] is True
    assert revived["yaw_ki"] == spec["yaw_ki"]


def test_yaw_rate_helper_is_imported_even_when_the_loop_is_off():
    """The step callback measures achieved yaw rate in BOTH arms.

    Regression: the helper was first imported inside the
    ``if _ROUTE.get("yaw_closed_loop")`` guard while the callback referenced
    it unconditionally, so running the OPEN-loop control arm raised
    ``NameError`` on every physics step.  The driver reported ``NO_TRACE``
    rather than a bogus success, but the whole A/B was unrunnable.  A
    ``compile()`` check cannot see this — the name only fails at call time.
    """
    import ast

    code = driver.build_driver_code(GAIT, 6.0, 1000.0, 50.0, 5, route=ROUTE)
    tree = ast.parse(code)

    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "yaw_closed_loop" not in ast.get_source_segment(code, node.test):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom):
                guarded.extend(a.name for a in inner.names)

    assert "yaw_rate_from_headings" not in guarded, (
        "yaw_rate_from_headings is imported under the yaw_closed_loop guard, "
        "but the step callback uses it in both arms"
    )
    assert "yaw_rate_from_headings" in code
