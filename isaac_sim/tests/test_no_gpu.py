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

import pytest

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


# -- the MAIN-THREAD contract (regression: live Isaac rendered 0 frames) ----
#
# Observed on the RTX 4090 2026-07-18: `--source isaac` came up healthy
# (/status 200, source=isaac, 3 channels) and served ZERO bytes on every MJPEG
# route, forever, with no error in the log.  Cause: CameraState.start() drives
# world.step()/rep.orchestrator.step() from a background thread, but Omniverse
# Kit must be pumped from the MAIN thread — the worker blocks there silently.
# The identical source stepped from the main thread rendered 12/12 frames.
# The synthetic source is pure numpy and thread-safe, which is why every
# previous headless test passed while the real path was dead.


def test_frame_sources_declare_their_threading_requirement():
    """Isaac needs the main thread; synthetic does not. The server must be able
    to tell them apart WITHOUT importing isaacsim."""
    mod = _load("camera_server")
    assert mod.SyntheticFrameSource.requires_main_thread is False
    assert mod.IsaacFrameSource.requires_main_thread is True


def test_camera_state_start_refuses_background_thread_for_main_thread_source():
    """start() must NOT spawn a render thread for a main-thread-only source —
    that is exactly the deadlock that served 0 frames from live Isaac."""
    mod = _load("camera_server")

    class _MainThreadOnly(mod.SyntheticFrameSource):
        requires_main_thread = True

    src = _MainThreadOnly(width=64, height=48)
    state = mod.CameraState(src, meta={}, fps=10, encoder=lambda rgb: b"x",
                            channels=("main",))
    with pytest.raises(RuntimeError, match="main thread"):
        state.start()


def test_camera_state_render_once_advances_frames_on_the_calling_thread():
    """render_once() is the main-thread pump: one call, one frame, in-thread."""
    import threading
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=64, height=48)
    state = mod.CameraState(src, meta={}, fps=10, encoder=lambda rgb: b"jpeg",
                            channels=("main",))
    caller = threading.get_ident()
    seen = []
    orig = src.get_frame

    def _spy():
        seen.append(threading.get_ident())
        return orig()

    src.get_frame = _spy
    assert state.frames == 0
    state.render_once()
    assert state.frames == 1, "render_once must produce a frame"
    assert state.latest("main") == b"jpeg"
    assert seen == [caller], "rendering must happen on the CALLING thread"


def test_mjpeg_depth_route_is_not_silently_rgb():
    """`/mjpeg_depth` must resolve to the DEPTH channel.

    It previously fell through the `/mjpeg` prefix and served RGB with a 200 —
    so an operator registering the natural sibling of `/mjpeg_right` got colour
    frames labelled as depth, with nothing anywhere reporting a problem.
    Caught on the wire against live Isaac, not by any unit test."""
    mod = _load("camera_server")
    for path, expected in [
        ("/mjpeg", "main"),
        ("/", "main"),
        ("/mjpeg_right", "right"),
        ("/depth", "depth"),
        ("/mjpeg_depth", "depth"),
    ]:
        assert mod.resolve_channel(path) == expected, f"{path} -> wrong channel"
    assert mod.resolve_channel("/nope") is None


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


# -- metric depth (depth16): the channel that carries RANGE, not a picture ----

def test_depth16_route_precedes_the_depth_prefix():
    """`/depth16` must resolve to the METRIC channel, not the colorized one.

    `/depth` is a prefix of `/depth16`, so a naive longest-prefix-last table
    would hand a perception consumer a TURBO colormap while it believed it was
    reading millimetres — the same silent-wrong-channel failure that once made
    `/mjpeg_depth` serve RGB, but far worse: colormapped pixels parse as
    plausible distances instead of looking obviously broken."""
    mod = _load("camera_server")
    assert mod.resolve_channel("/depth16") == "depth16"
    assert mod.resolve_channel("/depth") == "depth"
    assert mod.channel_mime("depth16") == "image/png"
    assert mod.channel_mime("depth") == "image/jpeg"
    assert mod.channel_mime("main") == "image/jpeg"


def test_depth16_encoder_preserves_metres():
    """The whole point: metres in, the same metres out. A colorized JPEG
    cannot do this, which is why the metric channel exists at all."""
    import numpy as np
    mod = _load("camera_server")
    depth = np.array([[1.0, 12.5], [40.0, 0.25]], dtype=np.float32)
    blob = mod.encode_depth16(depth)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n", "depth16 must be a lossless PNG"
    # Decode with the CANONICAL lib codec — see the contract test below.
    import numpy as np
    units = _decode_png16_bytes(blob)
    back = units.astype(np.float32) / mod.DEPTH16_SCALE
    assert np.allclose(back, depth, atol=0.001)


def test_depth16_holes_and_saturation_do_not_lie():
    """Sky/no-return folds to the 0 sentinel; beyond-range CLAMPS.

    Wrapping instead of clamping would turn a 70 m sky pixel into a ~4 m
    contact standing in front of the robot."""
    import numpy as np
    mod = _load("camera_server")
    units = _decode_png16_bytes(mod.encode_depth16(
        np.array([[np.nan, np.inf, 0.0, -2.0, 70.0, 1000.0]], dtype=np.float32)))
    assert list(units[0, :4]) == [0, 0, 0, 0], "invalid must be the 0 sentinel"
    assert units[0, 4] == 65535 and units[0, 5] == 65535, "must clamp, not wrap"


def test_depth_channels_derive_from_one_metric_read():
    """The viewable ramp and the metric PNG must describe the SAME instant.

    They come from a single get_depth_metres() call per render, so a moving
    scene can never produce a colormap of frame N beside millimetres of N+1."""
    import numpy as np
    mod = _load("camera_server")
    src = mod.SyntheticFrameSource(width=32, height=24, with_depth=True)
    calls = []
    orig = src.get_depth_metres

    def _spy():
        calls.append(1)
        return orig()

    src.get_depth_metres = _spy
    state = mod.CameraState(src, meta={"width": 32, "height": 24}, fps=10,
                            encoder=lambda rgb: b"jpeg",
                            channels=("main", "depth", "depth16"))
    state.render_once()
    assert len(calls) == 1, "depth must be read ONCE and both channels derived"
    assert state.latest("depth") == b"jpeg"
    assert state.latest("depth16")[:8] == b"\x89PNG\r\n\x1a\n"


def test_intrinsics_match_the_pinhole_model():
    """A consumer cannot unproject depth without fx/fy/cx/cy. Isaac renders
    through an ideal distortion-free lens, so the pinhole relation is exact."""
    import math
    mod = _load("camera_server")
    state = mod.CameraState(mod.SyntheticFrameSource(width=640, height=480),
                            meta={"width": 640, "height": 480}, fps=10,
                            encoder=lambda rgb: b"", channels=("main",),
                            hfov_deg=90.0)
    k = state.intrinsics()
    assert k["cx"] == 320.0 and k["cy"] == 240.0
    # hfov 90 deg -> fx = (w/2)/tan(45) = w/2
    assert k["fx"] == pytest.approx(320.0) and k["fy"] == pytest.approx(k["fx"])
    assert k["depth_scale"] == 1000.0 and k["depth_encoding"] == "16UC1_png"


def test_depth16_contract_matches_the_lib_codec():
    """CONTRACT: the connector's inlined encoder and tritium_lib's canonical
    decoder must agree exactly.

    The encoder is duplicated on purpose — connectors run in Isaac's python and
    must stay tritium-free (see test_connectors_do_not_import_tritium) — so this
    is the test that stops the two copies drifting. Skipped where tritium_lib is
    absent, which is precisely the Isaac-python case."""
    import numpy as np
    lib = pytest.importorskip("tritium_lib.perception.depth_codec")
    mod = _load("camera_server")
    depth = np.linspace(0.5, 50.0, 64, dtype=np.float32).reshape(8, 8)
    from_connector = lib.decode_depth16_png(mod.encode_depth16(depth))
    assert np.allclose(from_connector, depth, atol=0.001)
    assert mod.DEPTH16_SCALE == lib.DEPTH_SCALE_MM, "scale must not drift"
    # Holes must survive as NaN through the canonical decoder, not as 0 m.
    holed = lib.decode_depth16_png(
        mod.encode_depth16(np.array([[np.nan, 5.0]], dtype=np.float32)))
    assert np.isnan(holed[0, 0]) and holed[0, 1] == pytest.approx(5.0, abs=0.001)


def _decode_png16_bytes(blob: bytes):
    """Read a 16-bit PNG back with whatever codec is present — test-local so the
    depth16 tests do not depend on tritium_lib being installed."""
    import io

    import numpy as np
    try:
        import cv2

        return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    except ImportError:
        from PIL import Image

        return np.array(Image.open(io.BytesIO(blob)))


# -- capability 8b: a camera MOUNTED on a body ------------------------------
#
# Until now every camera_server render came from a wall: rep.create.camera at a
# fixed world pose.  A camera bolted to a robot is a different object — its lens
# pose is a FUNCTION of body pose, so the prim must be parented under the body
# and carry a body-frame offset plus a boresight orientation.  These gates pin
# the orientation math, which is the part that is easy to get wrong and
# impossible to eyeball from a single north-facing frame.

def _mount_axes(mod, **kw):
    """Boresight and up vector implied by the connector's mount orientation."""
    import numpy as np
    q = mod.mount_camera_quat(**kw)          # (w, x, y, z)
    w, x, y, z = q
    # Quaternion -> rotation matrix (columns are the rotated basis vectors).
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    # USD cameras look down -Z_cam with +Y_cam up.
    return R @ np.array([0.0, 0.0, -1.0]), R @ np.array([0.0, 1.0, 0.0])


def test_mounted_camera_looks_out_the_nose_of_the_body():
    """Zero tilt: boresight is body +X (forward), up is body +Z. On a Z-up
    stage that is the whole point of a nose camera."""
    import numpy as np
    mod = _load("camera_server")
    fwd, up = _mount_axes(mod, tilt_deg=0.0)
    assert np.allclose(fwd, [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9)


def test_mounted_camera_tilt_is_positive_up():
    """A +30 deg tilt raises the boresight; -30 lowers it. Sign errors here
    point a ground robot's camera at the sky, which still renders and so
    survives every check except this one."""
    import math

    import numpy as np
    mod = _load("camera_server")
    up30, _ = _mount_axes(mod, tilt_deg=30.0)
    down30, _ = _mount_axes(mod, tilt_deg=-30.0)
    assert np.allclose(up30, [math.cos(math.radians(30)), 0.0, math.sin(math.radians(30))], atol=1e-9)
    assert down30[2] == pytest.approx(-math.sin(math.radians(30)), abs=1e-9)
    # Tilt is a rotation, not a translation — the boresight stays a unit vector.
    assert np.linalg.norm(up30) == pytest.approx(1.0)


def test_mounted_camera_never_rolls():
    """The horizon must stay level at every tilt: the camera's up vector keeps
    zero component along the body's left axis (+Y on a Z-up stage)."""
    import numpy as np
    mod = _load("camera_server")
    for tilt in (-89.0, -45.0, 0.0, 15.0, 89.0):
        _, up = _mount_axes(mod, tilt_deg=tilt)
        assert abs(float(up[1])) < 1e-9, f"camera rolled at tilt {tilt}"
        assert float(up[2]) > 0.0, f"camera upside down at tilt {tilt}"


def test_mount_offset_contract_matches_the_lib_camera_mount():
    """CONTRACT: the offset the connector hands USD and the offset
    tritium_lib's CameraMount uses to project the FOV cone must be the SAME
    numbers, or the rendered image and the drawn cone silently disagree.

    Duplicated for the same reason as the depth codec: connectors run in
    Isaac's python and must stay tritium-free."""
    mount = pytest.importorskip("tritium_lib.geo.camera_mount")
    mod = _load("camera_server")
    kw = dict(forward_m=0.30, left_m=-0.05, up_m=0.12)
    lib_xyz = mount.CameraMount(tilt_deg=-10.0, **kw).stage_offset(up_axis="Z")
    assert mod.mount_stage_offset(**kw) == pytest.approx(lib_xyz)


def test_mount_flags_reach_the_isaac_source_and_the_served_meta():
    """The CLI plumbing, which the geometry gates above cannot see.

    A --mount-prim that parses but never reaches IsaacFrameSource yields a
    server that looks configured and still renders from the wall."""
    mod = _load("camera_server")
    ap = _camera_server_parser(mod)
    args = ap.parse_args([
        "--source", "isaac", "--mount-prim", "/World/Go2",
        "--mount-forward", "0.31", "--mount-up", "0.27",
        "--mount-tilt", "-12.5", "--attach-to", "sim_go2_01",
    ])
    assert args.mount_prim == "/World/Go2"
    assert args.attach_to == "sim_go2_01"
    assert args.mount_tilt == pytest.approx(-12.5)
    # A synthetic run must NOT advertise a mount it does not have.
    plain = ap.parse_args(["--source", "synthetic"])
    assert plain.mount_prim == "" and plain.attach_to == ""

    # ...and the flags must survive _build_source into the Isaac source. Stub
    # the class out so this stays a no-GPU gate.
    seen = {}
    mod.IsaacFrameSource = lambda *a, **kw: seen.update(kw) or object()
    mod._build_source(args)
    assert seen["mount_prim"] == "/World/Go2"
    assert seen["mount_forward"] == pytest.approx(0.31)
    assert seen["mount_up"] == pytest.approx(0.27)
    assert seen["mount_tilt"] == pytest.approx(-12.5)


def _camera_server_parser(mod):
    """main()'s parser, without running the server: parse a no-op argv and
    grab it. Keeps the test honest about the REAL flag set rather than a
    re-declared copy that can drift."""
    import argparse
    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **kw):
        captured.setdefault("ap", self)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        try:
            mod.main([])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    assert "ap" in captured, "main() built no parser"
    return captured["ap"]
