# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fire mechanic (capability 6) — pure math + protocol, NO Isaac, NO GPU.

The turret muzzle pose, aim vector, and ray-sphere/ray-box hit tests are
pure functions in ``isaac_quadruped_server.py``; the {cmd: "fire"} handler
runs entirely over SharedBody. Everything here executes under plain python3 —
the guarded Isaac path (spawning /World/Targets spheres and republishing
their stage poses) feeds the SAME ray_hit tested here.

TERRAIN: the ground is a registered box target, so a round aimed below the
horizon terminates at z = 0 instead of flying to max range underneath the
world (the live control tracer was visibly occluded below the slab before
this).  The helpers mirror ``tritium_lib.geo.hitscan`` op for op — the
contract tests at the bottom hold the two copies together bit-exactly, and
skip only where tritium_lib is absent (which is precisely the Isaac-python
case the duplication exists for).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_CONN = Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "connectors"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"conn_{name}", _CONN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _srv():
    return _load("isaac_quadruped_server")


def _close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# -- muzzle_pose: body pose + turret angles -> world origin + unit aim -----

def test_muzzle_facing_north_pan_zero_aims_north():
    mod = _srv()
    origin, aim = mod.muzzle_pose(0.0, 0.0, 0.0, 0.0, 0.0)
    assert _close(aim, (0.0, 1.0, 0.0))
    fwd, left, up = mod.TURRET_MOUNT_OFFSET
    # Mount forward of body center + barrel, all along +Y (north).
    assert _close(origin, (left, fwd + mod.BARREL_LENGTH_M, up))


def test_muzzle_facing_north_pan_90_aims_east():
    mod = _srv()
    _, aim = mod.muzzle_pose(0.0, 0.0, 0.0, 90.0, 0.0)
    assert _close(aim, (1.0, 0.0, 0.0))


def test_muzzle_facing_east_pan_zero_aims_east():
    mod = _srv()
    origin, aim = mod.muzzle_pose(5.0, 3.0, 90.0, 0.0, 0.0)
    assert _close(aim, (1.0, 0.0, 0.0))
    fwd, left, up = mod.TURRET_MOUNT_OFFSET
    # Mount offset rotates WITH the body: forward is now +X (east).
    assert _close(origin, (5.0 + fwd + mod.BARREL_LENGTH_M, 3.0, up))


def test_muzzle_facing_east_pan_minus_90_aims_north():
    mod = _srv()
    _, aim = mod.muzzle_pose(0.0, 0.0, 90.0, -90.0, 0.0)
    assert _close(aim, (0.0, 1.0, 0.0))


def test_muzzle_tilt_pitches_aim_up():
    mod = _srv()
    _, aim = mod.muzzle_pose(0.0, 0.0, 0.0, 0.0, 30.0)
    assert abs(aim[2] - 0.5) < 1e-9                       # sin(30)
    assert abs(aim[1] - math.cos(math.radians(30.0))) < 1e-9
    assert abs(aim[0]) < 1e-9
    _, straight_up = mod.muzzle_pose(0.0, 0.0, 0.0, 0.0, 90.0)
    assert _close(straight_up, (0.0, 0.0, 1.0))


def test_muzzle_aim_is_always_unit_length():
    mod = _srv()
    for heading in (0.0, 33.3, 90.0, 181.5, 270.0, 359.9):
        for pan in (-120.0, 0.0, 45.0, 200.0):
            for tilt in (-45.0, 0.0, 60.0):
                _, aim = mod.muzzle_pose(1.0, -2.0, heading, pan, tilt)
                norm = math.sqrt(sum(c * c for c in aim))
                assert abs(norm - 1.0) < 1e-9, (heading, pan, tilt)


def test_muzzle_offset_left_component():
    mod = _srv()
    # Facing north with a pure LEFT mount offset -> muzzle west of body.
    origin, _ = mod.muzzle_pose(0.0, 0.0, 0.0, 0.0, 0.0,
                                mount_offset=(0.0, 1.0, 0.5), barrel_len=0.0)
    assert _close(origin, (-1.0, 0.0, 0.5))
    # Facing east the same left offset points north.
    origin, _ = mod.muzzle_pose(0.0, 0.0, 90.0, 0.0, 0.0,
                                mount_offset=(0.0, 1.0, 0.5), barrel_len=0.0)
    assert _close(origin, (0.0, 1.0, 0.5))


# -- ray_hit: nearest sphere in path, misses, behind, range clip -----------

def test_ray_hit_nearest_target_wins():
    mod = _srv()
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "far", "x": 0.0, "y": 20.0, "z": 1.0, "radius": 1.0},
        {"id": "near", "x": 0.0, "y": 10.0, "z": 1.0, "radius": 1.0},
    ])
    assert hit is not None and hit["target_id"] == "near"
    assert abs(hit["range"] - 9.0) < 1e-9        # sphere surface, not center
    assert _close(hit["hit_pos"], (0.0, 9.0, 1.0))


def test_ray_hit_misses_off_axis_target():
    mod = _srv()
    assert mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "wide", "x": 5.0, "y": 10.0, "z": 1.0, "radius": 1.0},
    ]) is None


def test_ray_hit_ignores_target_behind_muzzle():
    mod = _srv()
    assert mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "behind", "x": 0.0, "y": -5.0, "z": 1.0, "radius": 1.0},
    ]) is None


def test_ray_hit_grazing_edge_hits_center_line_offset():
    mod = _srv()
    # Center offset 0.5 from the ray line, radius 1.0 -> still a hit.
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "graze", "x": 0.5, "y": 10.0, "z": 1.0, "radius": 1.0},
    ])
    assert hit is not None and hit["target_id"] == "graze"


def test_ray_hit_respects_max_range():
    mod = _srv()
    targets = [{"id": "distant", "x": 0.0, "y": 100.0, "z": 1.0, "radius": 1.0}]
    assert mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), targets,
                       max_range=50.0) is None
    assert mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), targets,
                       max_range=150.0) is not None


def test_ray_hit_empty_and_malformed_targets():
    mod = _srv()
    assert mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), []) is None
    # Malformed entries are skipped, well-formed ones still hit.
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "junk"},                                   # no coordinates
        {"id": "ok", "x": 0.0, "y": 5.0, "z": 1.0, "radius": 0.5},
    ])
    assert hit is not None and hit["target_id"] == "ok"


# -- {cmd: "fire"} handler: structured result over SharedBody, no Isaac ----

def _armed_shared(mod):
    """SharedBody posed at origin facing north with one plate 20 m out."""
    shared = mod.SharedBody()
    with shared.lock:
        shared.state = dict(shared.state, x=0.0, y=0.0, heading=0.0)
    r = mod.handle_request({"cmd": "targets", "targets": [
        {"id": "plate", "x": 0.0, "y": 20.0,
         "z": mod.TURRET_MOUNT_OFFSET[2], "radius": 0.5}]}, shared)
    assert r == {"ok": True, "count": 1}
    return shared


def test_fire_returns_structured_hit():
    mod = _srv()
    shared = _armed_shared(mod)
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["ok"] is True and r["fired"] is True
    assert r["hit"] is True and r["target_id"] == "plate"
    assert r["ammo"] == mod.MAGAZINE_SIZE - 1
    assert r["hit_pos"] is not None and abs(r["hit_pos"][1] - 19.5) < 1e-3
    assert r["range"] is not None and r["range"] > 0
    assert len(r["origin"]) == 3 and len(r["aim"]) == 3
    assert abs(r["aim"][1] - 1.0) < 1e-6                  # aimed north


def test_fire_miss_when_aimed_away():
    mod = _srv()
    shared = _armed_shared(mod)
    assert mod.handle_request(
        {"cmd": "turret", "pan": 90.0, "tilt": 0.0}, shared)["ok"] is True
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["ok"] is True and r["fired"] is True
    assert r["hit"] is False and r["target_id"] is None
    assert r["hit_pos"] is None and r["range"] is None
    assert r["ammo"] == mod.MAGAZINE_SIZE - 1             # a miss still spends


def test_fire_uses_turret_pan_relative_to_body():
    mod = _srv()
    shared = mod.SharedBody()
    with shared.lock:
        # Body facing EAST; plate due EAST -> pan 0 hits, pan 90 (south) misses.
        shared.state = dict(shared.state, x=0.0, y=0.0, heading=90.0)
    mod.handle_request({"cmd": "targets", "targets": [
        {"id": "east_plate", "x": 15.0, "y": 0.0,
         "z": mod.TURRET_MOUNT_OFFSET[2], "radius": 0.5}]}, shared)
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["hit"] is True and r["target_id"] == "east_plate"
    mod.handle_request({"cmd": "turret", "pan": 90.0, "tilt": 0.0}, shared)
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["hit"] is False


def test_fire_out_of_ammo():
    mod = _srv()
    shared = _armed_shared(mod)
    with shared.lock:
        shared.ammo = 1
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["fired"] is True and r["ammo"] == 0
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["ok"] is True and r["fired"] is False
    assert r["hit"] is False and r["hit_pos"] is None and r["target_id"] is None
    assert r["ammo"] == 0 and r["reason"] == "out_of_ammo"


def test_targets_cmd_validation():
    mod = _srv()
    shared = mod.SharedBody()
    r = mod.handle_request({"cmd": "targets", "targets": "nope"}, shared)
    assert r["ok"] is False
    r = mod.handle_request({"cmd": "targets", "targets": [{"x": "bad"}]}, shared)
    assert r["ok"] is False
    r = mod.handle_request(
        {"cmd": "targets", "targets": [{"id": "a", "x": 1, "y": 2}]}, shared)
    assert r == {"ok": True, "count": 1}
    with shared.lock:
        tgt = shared.targets[0]
    assert tgt["z"] == 0.0 and tgt["radius"] == 0.5       # defaults filled


def test_fire_path_never_imports_isaac(monkeypatch):
    """The whole no-Isaac fire path runs with isaacsim imports poisoned."""
    import builtins
    real_import = builtins.__import__

    def _poisoned(name, *args, **kwargs):
        if name.startswith(("isaacsim", "pxr", "omni")):
            raise AssertionError(f"fire path imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _poisoned)
    mod = _srv()
    shared = _armed_shared(mod)
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["hit"] is True


# -- ray_box: the slab method, with its notorious edges --------------------

_GROUND_LO = (-1000.0, -1000.0, -100.0)
_GROUND_HI = (1000.0, 1000.0, 0.0)


def test_ray_box_hits_the_near_face():
    mod = _srv()
    t = mod.ray_box((0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                    (-1.0, 5.0, 0.0), (1.0, 7.0, 2.0))
    assert t is not None and abs(t - 5.0) < 1e-12  # near face, not the centre


def test_ray_box_misses_a_box_behind_the_muzzle():
    mod = _srv()
    assert mod.ray_box((0.0, 10.0, 1.0), (0.0, 1.0, 0.0),
                       (-1.0, 5.0, 0.0), (1.0, 7.0, 2.0)) is None


def test_ray_box_down_into_the_ground_stops_at_the_top_face():
    mod = _srv()
    down45 = (0.0, math.sqrt(0.5), -math.sqrt(0.5))
    t = mod.ray_box((0.0, 0.0, 1.0), down45, _GROUND_LO, _GROUND_HI)
    assert t is not None and abs(t - math.sqrt(2.0)) < 1e-9


def test_ray_box_parallel_above_the_slab_misses():
    """Level fire above the ground: aim z is EXACTLY 0.0 (sin(0)), so the
    naive slab method divides by zero here. The explicit parallel branch
    decides by position: origin above the top face -> miss."""
    mod = _srv()
    assert mod.ray_box((0.0, 0.0, 0.55), (0.0, 1.0, 0.0),
                       _GROUND_LO, _GROUND_HI) is None


def test_ray_box_origin_exactly_on_the_plane_is_not_nan():
    """The exact-boundary 0/0 case mutation testing found: origin ON the top
    face, ray parallel to it. Naive: (0 - 0) / 0 = NaN, every comparison
    false, verdict garbage. The branch decides by position: on the plane is
    within the slab, so the other axes decide -- a grazing contact at 0."""
    mod = _srv()
    t = mod.ray_box((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), _GROUND_LO, _GROUND_HI)
    assert t == 0.0
    assert not math.isnan(t)


def test_ray_box_origin_inside_reports_point_blank_zero():
    mod = _srv()
    t = mod.ray_box((0.0, 0.0, -0.5), (0.0, 1.0, 0.0), _GROUND_LO, _GROUND_HI)
    assert t == 0.0  # inside the box: contact, never a negative range


def test_ray_box_parallel_below_the_slab_misses():
    mod = _srv()
    assert mod.ray_box((0.0, 0.0, -200.0), (0.0, 1.0, 0.0),
                       _GROUND_LO, _GROUND_HI) is None


def test_ray_box_rejects_inverted_bounds_and_degenerate_aim():
    mod = _srv()
    with pytest.raises(ValueError):
        mod.ray_box((0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                    (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
    with pytest.raises(ValueError):
        mod.ray_box((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                    (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))


def test_ray_sphere_point_blank_inside_is_a_contact_hit():
    """Muzzle INSIDE the target: contact shot at range 0 (lib semantics).
    The old inline gate (tca < 0 -> skip) wrongly refused it when the centre
    sat behind the muzzle."""
    mod = _srv()
    assert mod.ray_sphere_t((0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                            (0.0, -0.2, 1.0), 1.0) == 0.0


# -- mixed sphere + box resolution: nearest hit wins -----------------------

def test_ray_hit_nearest_across_mixed_sphere_and_box():
    mod = _srv()
    targets = [
        {"id": "sphere", "x": 0.0, "y": 10.0, "z": 1.0, "radius": 1.0},
        {"id": "box", "min": [-1.0, 4.0, 0.0], "max": [1.0, 5.0, 2.0]},
    ]
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), targets)
    assert hit is not None and hit["target_id"] == "box"
    assert abs(hit["range"] - 4.0) < 1e-9
    # List order must not matter: nearest wins, not first-listed.
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), list(reversed(targets)))
    assert hit is not None and hit["target_id"] == "box"


def test_ray_hit_sphere_wins_when_it_is_the_nearer_one():
    mod = _srv()
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "s", "x": 0.0, "y": 2.0, "z": 1.0, "radius": 0.5},
        {"id": "b", "min": [-1.0, 4.0, 0.0], "max": [1.0, 5.0, 2.0]},
    ])
    assert hit is not None and hit["target_id"] == "s"
    assert abs(hit["range"] - 1.5) < 1e-9


def test_ray_hit_malformed_box_is_skipped_not_fatal():
    mod = _srv()
    hit = mod.ray_hit((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), [
        {"id": "junk_box", "min": [0.0, 5.0], "max": [4.0, 6.0, 2.0]},
        {"id": "inverted", "min": [1.0, 9.0, 2.0], "max": [-1.0, 8.0, 0.0]},
        {"id": "ok", "x": 0.0, "y": 5.0, "z": 1.0, "radius": 0.5},
    ])
    assert hit is not None and hit["target_id"] == "ok"


# -- terrain: the round stops at the ground --------------------------------

def test_fire_stops_at_terrain_before_a_target_behind_it():
    """A shot at a target buried BEYOND the ground plane hits the TERRAIN."""
    mod = _srv()
    shared = mod.SharedBody()
    with shared.lock:
        shared.state = dict(shared.state, x=0.0, y=0.0, heading=0.0)
    mod.handle_request({"cmd": "targets", "targets": [
        {"id": "buried", "x": 0.0, "y": 3.0, "z": -2.0, "radius": 0.6}]},
        shared)
    mod.handle_request({"cmd": "turret", "pan": 0.0, "tilt": -45.0}, shared)
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["hit"] is True and r["terrain"] is True
    assert r["target_id"] == mod.GROUND_TARGET["id"]
    assert abs(r["hit_pos"][2]) < 1e-3          # stopped AT the ground plane
    # The same ray WITHOUT the ground reaches the buried plate — terrain is
    # what stopped the round, not a bad aim.
    with shared.lock:
        spheres_only = list(shared.targets)
    direct = mod.ray_hit(tuple(r["origin"]), tuple(r["aim"]), spheres_only)
    assert direct is not None and direct["target_id"] == "buried"


def test_fire_with_clear_line_of_sight_still_hits_over_the_ground():
    """The ground being solid must not eat a legitimate shot above it."""
    mod = _srv()
    shared = _armed_shared(mod)                # plate 20 m north, level fire
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["hit"] is True and r["target_id"] == "plate"
    assert r["terrain"] is False


def test_targets_replace_never_deletes_the_ground():
    """{cmd: "targets"} rewrites the target list wholesale — the terrain
    must survive it (in Isaac mode the per-step republish does the same
    rewrite 60 times a second)."""
    mod = _srv()
    shared = mod.SharedBody()
    mod.handle_request({"cmd": "targets", "targets": []}, shared)
    mod.handle_request({"cmd": "turret", "pan": 0.0, "tilt": -30.0}, shared)
    r = mod.handle_request({"cmd": "fire"}, shared)
    assert r["hit"] is True and r["terrain"] is True


def test_targets_cmd_accepts_box_entries():
    mod = _srv()
    shared = mod.SharedBody()
    r = mod.handle_request({"cmd": "targets", "targets": [
        {"id": "wall", "min": [0.0, 5.0, 0.0], "max": [4.0, 6.0, 2.0]},
        {"id": "s", "x": 1.0, "y": 2.0}]}, shared)
    assert r == {"ok": True, "count": 2}
    with shared.lock:
        assert shared.targets[0] == {"id": "wall", "min": [0.0, 5.0, 0.0],
                                     "max": [4.0, 6.0, 2.0]}
    r = mod.handle_request({"cmd": "targets", "targets": [
        {"id": "bad", "min": [0.0, 5.0], "max": [4.0, 6.0, 2.0]}]}, shared)
    assert r["ok"] is False
    r = mod.handle_request({"cmd": "targets", "targets": [
        {"id": "bad", "min": [5.0, 5.0, 5.0], "max": [0.0, 0.0, 0.0]}]},
        shared)
    assert r["ok"] is False


# -- CONTRACT: the connector's math must equal tritium_lib.geo.hitscan -----
#
# The helpers are duplicated on purpose — connectors run in Isaac's python
# and stay tritium-free (see test_no_gpu.test_connectors_do_not_import_tritium)
# — so these are the tests that stop the two copies drifting. They compare
# EXACTLY (==, not approx): the implementations mirror each other op for op,
# so any inequality is a semantic divergence, not float noise. Skipped where
# tritium_lib is absent, which is precisely the Isaac-python case.

def test_ray_box_matches_lib_ray_aabb_exactly():
    lib = pytest.importorskip("tritium_lib.geo.hitscan")
    mod = _srv()
    cases = [
        # square hit on a near face
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 5.0, -1.0), (1.0, 9.0, 1.0)),
        # down into the ground
        ((0.0, 0.0, 1.0), (0.0, math.sqrt(0.5), -math.sqrt(0.5)),
         _GROUND_LO, _GROUND_HI),
        # level above the slab: parallel outside
        ((0.0, 0.0, 0.55), (0.0, 1.0, 0.0), _GROUND_LO, _GROUND_HI),
        # origin exactly ON the top plane: the 0/0 boundary
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), _GROUND_LO, _GROUND_HI),
        # inside the slab
        ((0.0, 0.0, -0.5), (0.0, 1.0, 0.0), _GROUND_LO, _GROUND_HI),
        # box behind the muzzle
        ((0.0, 10.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 5.0, 0.0), (1.0, 7.0, 2.0)),
        # non-unit direction (both must normalise identically)
        ((0.0, 0.0, 1.0), (0.3, 2.0, -0.4), _GROUND_LO, _GROUND_HI),
        # oblique diagonal
        ((5.0, 5.0, 5.0), (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        # grazing along an edge
        ((-1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (-1.0, 5.0, 0.0), (1.0, 7.0, 2.0)),
    ]
    for origin, direction, lo, hi in cases:
        assert mod.ray_box(origin, direction, lo, hi) == \
            lib.ray_aabb(origin, direction, lo, hi), (origin, direction, lo, hi)


def test_ray_box_error_semantics_match_lib():
    lib = pytest.importorskip("tritium_lib.geo.hitscan")
    mod = _srv()
    for fn in (mod.ray_box, lib.ray_aabb):
        with pytest.raises(ValueError):
            fn((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
               (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))     # degenerate direction
        with pytest.raises(ValueError):
            fn((0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
               (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))     # inverted bounds


def test_ray_sphere_matches_lib_exactly():
    lib = pytest.importorskip("tritium_lib.geo.hitscan")
    mod = _srv()
    cases = [
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 10.0, 1.0), 1.0),   # hit
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (5.0, 10.0, 1.0), 1.0),   # wide
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, -5.0, 1.0), 1.0),   # behind
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.2, 1.0), 1.0),    # inside
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, -0.2, 1.0), 1.0),   # inside,
        # centre behind -- the case the old tca<0 gate got wrong
        ((0.0, 0.0, 1.0), (0.5, 2.0, 0.1), (1.0, 8.0, 1.4), 0.9),    # oblique
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.5, 10.0, 1.0), 0.5),   # graze
    ]
    for origin, d, c, r in cases:
        assert mod.ray_sphere_t(origin, d, c, r) == \
            lib.ray_sphere(origin, d, c, r), (origin, d, c, r)


def test_ray_hit_matches_lib_resolve_shot_across_mixed_targets():
    """The full resolver: same winner, same range, same impact — including
    the range gate, which the two apply differently in code but provably
    identically in outcome (the winner is the global minimum, so no farther
    candidate can pass a gate the minimum fails)."""
    lib = pytest.importorskip("tritium_lib.geo.hitscan")
    mod = _srv()

    spheres = [
        ("s_near", (1.4, 3.0, 0.4), 0.5),
        ("s_far", (2.6, 8.5, 0.3), 0.6),
        ("s_wide", (-8.0, 2.0, 0.5), 0.4),
    ]
    boxes = [
        ("wall", (-0.5, 5.0, -1.0), (4.0, 5.4, 2.0)),
        ("terrain_ground", _GROUND_LO, _GROUND_HI),
    ]
    dict_targets = (
        [{"id": i, "x": c[0], "y": c[1], "z": c[2], "radius": r}
         for i, c, r in spheres]
        + [{"id": i, "min": list(lo), "max": list(hi)} for i, lo, hi in boxes]
    )
    lib_targets = (
        [lib.SphereTarget(i, c[0], c[1], c[2], r) for i, c, r in spheres]
        + [lib.BoxTarget(i, *lo, *hi) for i, lo, hi in boxes]
    )

    muzzles = [
        # sphere winner, wall winner, terrain winner, clear-air miss, upward
        lib.Muzzle(0.45, -1.2, 0.55, 14.0, -9.0),
        lib.Muzzle(0.0, 0.0, 0.55, 20.0, 0.0),
        lib.Muzzle(0.0, 0.0, 0.55, 180.0, -30.0),
        lib.Muzzle(0.0, 0.0, 0.55, 270.0, 10.0),
        lib.Muzzle(0.0, 0.0, 0.55, 0.0, 45.0),
    ]
    for muzzle in muzzles:
        origin, aim = muzzle.origin(), muzzle.direction()
        got = mod.ray_hit(origin, aim, dict_targets, max_range=60.0)
        want = lib.resolve_shot(muzzle, lib_targets, 60.0)
        assert (got is not None) == want.hit, muzzle
        if want.hit:
            assert got["target_id"] == want.target_id, muzzle
            assert got["range"] == want.range_m, muzzle
            assert tuple(got["hit_pos"]) == want.impact(), muzzle


def test_ray_hit_range_gate_matches_lib_at_the_boundary():
    """A hit at EXACTLY max_range counts, in both implementations."""
    lib = pytest.importorskip("tritium_lib.geo.hitscan")
    mod = _srv()
    muzzle = lib.Muzzle(0.0, 0.0, 1.0, 0.0, 0.0)
    tgt = {"id": "edge", "x": 0.0, "y": 51.0, "z": 1.0, "radius": 1.0}
    lib_tgt = lib.SphereTarget("edge", 0.0, 51.0, 1.0, 1.0)
    origin, aim = muzzle.origin(), muzzle.direction()
    # Surface at exactly 50 m.
    assert mod.ray_hit(origin, aim, [tgt], max_range=50.0) is not None
    assert lib.resolve_shot(muzzle, [lib_tgt], 50.0).hit is True
    # One metre short of reaching it: both refuse.
    assert mod.ray_hit(origin, aim, [tgt], max_range=49.0) is None
    assert lib.resolve_shot(muzzle, [lib_tgt], 49.0).hit is False
