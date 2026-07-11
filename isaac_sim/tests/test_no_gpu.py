# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""No-GPU gates for the Isaac Sim connector addon.

Proves the Isaac-side connectors are inspectable and the neutral contracts hold
WITHOUT Isaac, pxr, or a GPU — so the addon stays testable in CI. Also enforces
the dependency-hygiene invariant: connectors import neither ``tritium`` nor a
heavy runtime at module load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CONN = Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "connectors"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"conn_{name}", _CONN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mini_scene() -> dict:
    """A tiny Z-up Scene3D dict (one triangle terrain + one 2-tri building)."""
    return {
        "ao": "test",
        "origin_lat": 37.7159,
        "origin_lng": -121.896,
        "up_axis": "Z",
        "meshes": [
            {"name": "terr", "kind": "terrain",
             "vertices": [[0, 0, 0], [10, 0, 0], [0, 10, 0]],
             "faces": [[0, 1, 2]]},
            {"name": "b0", "kind": "building",
             "vertices": [[2, 2, 0], [4, 2, 0], [4, 4, 5], [2, 4, 5]],
             "faces": [[0, 1, 2], [0, 2, 3]], "height_m": 5.0},
        ],
        "metadata": {},
    }


# -- usd_scene_builder: validate + OBJ work with NO pxr --------------------

def test_usd_builder_imports_without_pxr():
    """Module import must not require pxr/isaacsim (they are lazy in write_usd)."""
    mod = _load("usd_scene_builder")
    assert hasattr(mod, "validate") and hasattr(mod, "emit_obj")


def test_validate_passes_on_wellformed_scene():
    mod = _load("usd_scene_builder")
    assert mod.validate(_mini_scene()) == 0


def test_validate_rejects_out_of_range_face():
    mod = _load("usd_scene_builder")
    bad = _mini_scene()
    bad["meshes"][0]["faces"] = [[0, 1, 99]]  # index out of range
    try:
        mod.validate(bad)
        assert False, "validate should have raised on a bad face index"
    except AssertionError as exc:
        assert "range" in str(exc).lower()


def test_emit_obj_roundtrips_counts(tmp_path):
    mod = _load("usd_scene_builder")
    out = tmp_path / "s.obj"
    mod.emit_obj(_mini_scene(), str(out))
    text = out.read_text()
    assert sum(1 for l in text.splitlines() if l.startswith("v ")) == 7
    assert sum(1 for l in text.splitlines() if l.startswith("f ")) == 3


# -- camera_server: synthetic (no-GPU) source produces frames --------------

def test_camera_server_synthetic_source_frames():
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=64, height=48)
    f1 = src.get_frame()
    f2 = src.get_frame()
    assert f1.shape == (48, 64, 3)
    import numpy as np
    assert not np.array_equal(f1, f2), "subject should move between frames"


# -- isaac_quadruped_server: integrator + protocol WITHOUT isaacsim --------

def test_quadruped_server_imports_without_isaacsim():
    """Module import must not require isaacsim (it is lazy inside run_isaac)."""
    mod = _load("isaac_quadruped_server")
    assert hasattr(mod, "GaitIntegrator") and hasattr(mod, "footfalls")
    assert hasattr(mod, "BodyServer") and hasattr(mod, "_selftest")


def test_quadruped_gait_integrator_trots_north():
    """The gait contract (walk/trot thresholds + north-is-y) holds in pure python."""
    mod = _load("isaac_quadruped_server")
    integ = mod.GaitIntegrator()
    for _ in range(100):
        integ.step(0.02, 0.5, 0.5)          # forward=0.5 -> 1.5 m/s -> trot
    assert integ.gait == "trot"
    assert abs(integ.speed - 1.5) < 1e-6
    assert integ.y > 1.0 and abs(integ.x) < 1e-9   # moved north, no east drift


def test_quadruped_footfall_stance_rules():
    """Footfall stance rules match the SC gait-diagram contract."""
    mod = _load("isaac_quadruped_server")
    assert mod.footfalls("trot", 0.25) == ["FL", "RR"]
    assert mod.footfalls("trot", 0.75) == ["FR", "RL"]
    assert mod.footfalls("bound", 0.1) == ["FL", "FR"]


def test_quadruped_selftest_passes():
    """The server's own pure-python self-test (integrator + protocol + TCP
    loopback) returns 0 with no Isaac import and no GPU."""
    mod = _load("isaac_quadruped_server")
    assert mod._selftest() == 0


# -- dependency hygiene: connectors never import tritium -------------------

def test_connectors_do_not_import_tritium():
    for f in _CONN.glob("*.py"):
        src = f.read_text()
        assert "import tritium" not in src and "from tritium" not in src, (
            f"{f.name} imports tritium — connectors must stay tritium-free "
            "(the Scene3D JSON / MJPEG / TCP contracts are the only seam)"
        )
