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


def test_camera_server_extra_channels_off_by_default():
    """A plain synthetic source exposes no depth/stereo (capability 8 opt-in)."""
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=64, height=48)
    src.get_frame()
    assert src.get_depth() is None
    assert src.get_right_frame() is None


def test_camera_server_depth_channel_is_a_ramp():
    """--depth synthesizes a colorized depth image (near != far) with NO GPU."""
    import numpy as np
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=64, height=48, with_depth=True)
    src.get_frame()
    depth = src.get_depth()
    assert depth is not None and depth.shape == (48, 64, 3)
    assert depth.dtype == np.uint8
    # Depth ramp: the near ground (bottom rows) must read differently from the
    # far ground near the horizon — not a flat image.
    assert depth.std() > 0.0, "depth image is uniform — no ramp"
    ground = int(48 * 0.55)
    near = depth[46, :, :].mean()
    far = depth[ground + 1, :, :].mean()
    assert near != far, "near vs far ground should differ in the depth ramp"


def test_camera_server_stereo_right_is_parallax_shifted():
    """--stereo yields a right eye that differs from the left, same shape."""
    import numpy as np
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=96, height=64, with_stereo=True)
    left = src.get_frame()
    right = src.get_right_frame()
    assert right is not None and right.shape == left.shape
    assert not np.array_equal(left, right), "right eye should be parallax-shifted"


def test_camera_server_channels_track_flags():
    """The served channel set follows the --depth/--stereo flags (order fixed)."""
    mod = _load("camera_server")

    class _A:
        depth = False
        stereo = False

    assert mod._channels_for(_A()) == ("main",)
    _A.depth = True
    assert mod._channels_for(_A()) == ("main", "depth")
    _A.stereo = True
    assert mod._channels_for(_A()) == ("main", "depth", "right")


def test_camera_server_state_renders_all_channels():
    """CameraState._render_channels produces main+depth+right in one instant."""
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=64, height=48, with_depth=True, with_stereo=True)
    state = mod.CameraState(src, meta={}, fps=10, encoder=lambda rgb: b"",
                            channels=("main", "depth", "right"))
    rendered = state._render_channels()
    assert set(rendered) == {"main", "depth", "right"}
    for name, frame in rendered.items():
        assert frame.shape == (48, 64, 3), f"{name} bad shape {frame.shape}"


def test_camera_server_colorize_depth_near_bright_far_dark():
    """colorize_depth maps near->bright, far->dark and clamps inf/nan (sky)."""
    import numpy as np
    mod = _load("camera_server")
    d = np.array([[0.5, 60.0], [np.inf, np.nan]], dtype=np.float32)
    col = mod.colorize_depth(d, near=0.5, far=60.0)
    assert col.shape == (2, 2, 3) and col.dtype == np.uint8
    # near pixel brighter than the far pixel; inf/nan clamp to the far value.
    assert col[0, 0].mean() > col[0, 1].mean()
    assert np.array_equal(col[1, 0], col[1, 1]), "inf and nan both clamp to far"


def test_camera_server_selftest_covers_depth_and_stereo():
    """The server self-test (synthetic, no GPU) exercises RGB+depth+stereo -> 0."""
    mod = _load("camera_server")

    class _Args:
        width = 96
        height = 64
        selftest_frames = 4

    assert mod.selftest(_Args()) == 0


# -- lidar_server: synthetic (no-GPU) source produces valid sweeps ---------

def test_lidar_server_imports_without_isaacsim():
    """Module import must not require isaacsim (it is lazy inside IsaacScanSource)."""
    mod = _load("lidar_server")
    assert hasattr(mod, "SyntheticScanSource") and hasattr(mod, "IsaacScanSource")
    assert hasattr(mod, "LidarState") and hasattr(mod, "build_payload")


def test_lidar_synthetic_sweep_is_valid():
    """A synthetic sweep honors the LaserScan contract: N beams, every range
    within [range_min, range_max], no NaN/inf, correct angle bounds."""
    import math
    import numpy as np
    mod = _load("lidar_server")
    src = mod.SyntheticScanSource(num_beams=360, range_min=0.1, range_max=30.0)
    scan = src.get_scan()
    assert scan.shape == (360,)
    assert np.all(np.isfinite(scan)), "sweep contains NaN/inf"
    assert np.all(scan >= 0.1) and np.all(scan <= 30.0)
    assert src.angle_min == -math.pi
    assert abs(src.angle_increment - 2 * math.pi / 360) < 1e-12
    assert abs(src.angle_max - (-math.pi + 359 * src.angle_increment)) < 1e-12


def test_lidar_synthetic_is_deterministic_and_animated():
    """Same tick -> identical sweep (replayable); successive ticks differ
    (the orbiting obstacle moves)."""
    import numpy as np
    mod = _load("lidar_server")
    a = mod.SyntheticScanSource(num_beams=180)
    b = mod.SyntheticScanSource(num_beams=180)
    assert np.array_equal(a.scan_at(7), b.scan_at(7)), "sweep not deterministic"
    s1 = a.get_scan()
    s2 = a.get_scan()
    for _ in range(20):
        s2 = a.get_scan()
    assert not np.array_equal(s1, s2), "orbiting obstacle should move"


def test_lidar_synthetic_sees_the_room_and_obstacles():
    """The sweep reads the scene, not a constant: the near wall (hy=4 m at
    +90 deg) is closer than the far corner, and a pillar shadows its bearing."""
    import math
    import numpy as np
    mod = _load("lidar_server")
    src = mod.SyntheticScanSource(num_beams=360, room_w=10.0, room_h=8.0)
    scan = src.scan_at(0)
    ang = mod.beam_angles(360)
    up = int(np.argmin(np.abs(ang - math.pi / 2)))       # +Y wall: 4.0 m
    assert abs(scan[up] - 4.0) < 0.05
    # Static pillar at (2.5, 1.5) r=0.3 -> bearing atan2(1.5,2.5), dist ~2.62.
    pb = int(np.argmin(np.abs(ang - math.atan2(1.5, 2.5))))
    assert scan[pb] < 2.8, "pillar should shadow its bearing"


def test_lidar_payload_matches_edge_scan_contract():
    """/scan JSON carries ranges + the geometry keys tritium-edge's
    parse_scan_json consumes, JSON-serializable, no NaN."""
    import json as _json
    import math
    mod = _load("lidar_server")
    src = mod.SyntheticScanSource(num_beams=90, range_min=0.2, range_max=25.0)
    payload = mod.build_payload(src, src.get_scan(), "test-lidar", 3, stamp=123.5)
    body = _json.dumps(payload)          # must serialize (json rejects NaN/inf)
    doc = _json.loads(body)
    assert doc["lidar_id"] == "test-lidar" and doc["seq"] == 3
    assert doc["stamp"] == 123.5
    assert len(doc["ranges"]) == 90
    assert all(0.2 <= r <= 25.0 for r in doc["ranges"])
    for key in ("angle_min", "angle_max", "angle_increment",
                "range_min", "range_max"):
        assert isinstance(doc[key], float)
    assert abs(doc["angle_min"] + math.pi) < 1e-9


def test_lidar_resample_bins_cloud_to_ordered_sweep():
    """resample_to_beams (the Isaac scan-buffer seam) bins an unordered
    range/azimuth cloud: closest return wins a bin, empty bins read range_max,
    output clamped in band — pure numpy, no GPU."""
    import math
    import numpy as np
    mod = _load("lidar_server")
    out = mod.resample_to_beams(
        ranges=[5.0, 2.0, 7.0, np.nan, 100.0],
        azimuths=[0.0, 0.0, math.pi / 2, 1.0, math.pi],
        num_beams=4, angle_min=-math.pi, range_min=0.1, range_max=30.0)
    assert out.shape == (4,)
    assert out[2] == 2.0            # bin at azimuth 0: closest of (5.0, 2.0)
    assert out[3] == 7.0            # bin at +pi/2
    assert out[0] == 30.0           # pi wraps onto angle_min's bin, clamped
    assert out[1] == 30.0           # empty bin -> range_max, never NaN
    assert np.all(np.isfinite(out))


def test_lidar_state_survives_a_failing_source():
    """Graceful on missing/broken sensor: a raising source counts errors,
    leaves no fabricated sweep, and the state keeps ticking."""
    mod = _load("lidar_server")

    class _Broken(mod.ScanSource):
        name = "broken"

        def get_scan(self):
            raise RuntimeError("sensor missing")

    state = mod.LidarState(_Broken(num_beams=8), "broken-lidar", hz=10)
    assert state.tick_once() is False
    assert state.errors == 1 and state.scans == 0
    assert state.latest() is None    # -> the /scan route answers 503


def test_lidar_state_caches_latest_payload():
    """A healthy source yields a cached JSON body the handlers can serve."""
    import json as _json
    mod = _load("lidar_server")
    state = mod.LidarState(mod.SyntheticScanSource(num_beams=45), "ok-lidar", hz=10)
    assert state.tick_once() is True
    doc = _json.loads(state.latest())
    assert doc["lidar_id"] == "ok-lidar" and len(doc["ranges"]) == 45
    assert state.scans == 1 and state.errors == 0


def test_lidar_server_selftest_passes():
    """The server's own no-GPU self-test returns 0."""
    mod = _load("lidar_server")

    class _Args:
        beams = 120
        range_min = 0.1
        range_max = 30.0
        selftest_scans = 6

    assert mod.selftest(_Args()) == 0


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
