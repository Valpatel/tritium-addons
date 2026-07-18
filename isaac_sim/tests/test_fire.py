# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fire mechanic (capability 6) — pure math + protocol, NO Isaac, NO GPU.

The turret muzzle pose, aim vector, and ray-sphere hit test are pure
functions in ``isaac_quadruped_server.py``; the {cmd: "fire"} handler runs
entirely over SharedBody. Everything here executes under plain python3 —
the guarded Isaac path (spawning /World/Targets spheres and republishing
their stage poses) feeds the SAME ray_hit tested here.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

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
