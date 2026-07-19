# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""No-GPU gates for the ATTACHED sensor mode (attached_sensor_server).

The attached mode's whole reason to exist is the one thing the standalone
servers cannot do — image the body that the dedicated Newton kit is stepping,
because stages are per-process.  These tests prove everything about it that
does not need a kit: the pure camera geometry, the publish() seams it feeds,
the reused sweep binning, and the loud refusals at every boundary a silent
failure used to live behind.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

_CONN = Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "connectors"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"conn_{name}", _CONN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def att():
    return _load("attached_sensor_server")


# -- pure geometry ---------------------------------------------------------- #

def test_lookat_matrix_camera_looking_plus_x(att):
    """Z-up world, camera at origin aimed +X: USD's -Z-forward/+Y-up camera
    must get right=-Y, up=+Z — the exact rows a Gf.Matrix4d expects."""
    m = np.array(att.usd_lookat_matrix((0, 0, 0), (10, 0, 0)))
    assert np.allclose(m[0, :3], [0, -1, 0])   # camera X (right)
    assert np.allclose(m[1, :3], [0, 0, 1])    # camera Y (up)
    assert np.allclose(m[2, :3], [-1, 0, 0])   # camera Z (backwards)
    assert np.allclose(m[3], [0, 0, 0, 1])


def test_lookat_matrix_rows_are_orthonormal_and_translated(att):
    eye, target = (-4.0, 2.5, 1.2), (6.0, -1.0, 0.4)
    m = np.array(att.usd_lookat_matrix(eye, target))
    r = m[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)          # right-handed, no flip
    assert np.allclose(m[3, :3], eye)
    # -Z really points at the target.
    fwd = -m[2, :3]
    want = np.array(target) - np.array(eye)
    assert np.allclose(fwd, want / np.linalg.norm(want), atol=1e-9)


def test_lookat_matrix_refuses_a_directionless_camera(att):
    with pytest.raises(ValueError):
        att.usd_lookat_matrix((1, 2, 3), (1, 2, 3))


def test_hfov_aperture_inverts_the_pinhole_identity(att):
    ap = att.hfov_to_aperture_mm(70.0, 18.0)
    assert math.isclose(2 * math.degrees(math.atan(ap / (2 * 18.0))), 70.0)
    with pytest.raises(ValueError):
        att.hfov_to_aperture_mm(0.0)
    with pytest.raises(ValueError):
        att.hfov_to_aperture_mm(180.0)


def test_predicted_disparity_is_fx_b_over_z(att):
    assert math.isclose(att.predicted_disparity_px(457.0, 0.3, 5.7),
                        457.0 * 0.3 / 5.7)
    with pytest.raises(ValueError):
        att.predicted_disparity_px(457.0, 0.3, 0.0)


# -- stubs + refusals ------------------------------------------------------- #

def test_frame_stub_refuses_to_be_pulled(att):
    stub = att.AttachedFrameStub()
    assert stub.requires_main_thread is True
    with pytest.raises(RuntimeError, match="publish"):
        stub.get_frame()


def test_camera_state_refuses_background_start_for_the_stub(att):
    cam = _load("camera_server")
    state = cam.CameraState(att.AttachedFrameStub(), {}, 10, lambda f: b"x")
    with pytest.raises(RuntimeError, match="main thread"):
        state.start()


def test_scan_stub_geometry_and_refusal(att):
    lid = _load("lidar_server")
    g = att.AttachedScanGeometry(num_beams=180, range_min=0.2, range_max=25.0)
    assert g.never_returned is True
    g.warmed = True
    assert g.never_returned is False
    with pytest.raises(RuntimeError, match="publish"):
        g.get_scan()
    payload = lid.build_payload(g, np.full(180, 25.0), "att", 0)
    assert payload["never_returned"] is False
    assert len(payload["ranges"]) == 180


def test_install_refuses_outside_a_kit(att):
    with pytest.raises(RuntimeError, match="[Kk]it"):
        att.install()


def test_install_refuses_a_double_install(att):
    att._STATE["installed"] = True
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            att.install()
    finally:
        att._STATE["installed"] = False


def test_provenance_names_this_directory(att):
    prov = att.provenance()
    for key in ("attached_sensor_server", "camera_server", "lidar_server"):
        assert Path(prov[key]).parent == _CONN, prov


# -- the publish() seams the attached pump feeds ---------------------------- #

def test_camera_state_publish_serves_and_counts(att):
    cam = _load("camera_server")
    state = cam.CameraState(att.AttachedFrameStub(), {}, 10, lambda f: b"x",
                            channels=("main", "depth16", "right"))
    assert state.frames == 0
    assert state.publish({"main": b"JJ", "depth16": b"PP", "right": b"RR"})
    assert state.frames == 1
    assert state.latest("main") == b"JJ"
    assert state.latest("depth16") == b"PP"
    assert state.latest("right") == b"RR"


def test_camera_state_publish_without_main_does_not_count_as_a_frame(att):
    """A depth-only update refreshes its channel but must not tick the
    liveness counter — 'frames climbing' is the one signal a watcher has."""
    cam = _load("camera_server")
    state = cam.CameraState(att.AttachedFrameStub(), {}, 10, lambda f: b"x",
                            channels=("main", "depth16"))
    assert state.publish({"depth16": b"PP"}) is False
    assert state.frames == 0
    assert state.latest("depth16") == b"PP"


def test_lidar_state_publish_serves_and_counts(att):
    lid = _load("lidar_server")
    g = att.AttachedScanGeometry()
    state = lid.LidarState(g, "att", hz=10)
    assert state.latest() is None
    body = json.dumps({"ranges": [1.0]}).encode()
    state.publish(body)
    assert state.latest() == body
    assert state.scans == 1


# -- the extracted sweep binning (reused, not mirrored) --------------------- #

def test_flat_scan_to_ranges_empty_frame_is_range_max_and_unwarmed():
    lid = _load("lidar_server")
    ranges, has = lid.flat_scan_to_ranges({}, 8, -math.pi, 0.1, 30.0)
    assert has is False
    assert np.array_equal(ranges, np.full(8, 30.0))


def test_flat_scan_to_ranges_drops_the_no_return_sentinel():
    lid = _load("lidar_server")
    frame = {"linearDepthData": [-1.0, 5.0, -1.0, -1.0],
             "azimuthRange": (-180.0, 180.0)}
    ranges, has = lid.flat_scan_to_ranges(frame, 4, -math.pi, 0.1, 30.0)
    assert has is True
    # Only the one valid return lands; sentinel beams stay at range_max.
    assert np.count_nonzero(ranges < 30.0) == 1
    assert ranges.min() == pytest.approx(5.0)


def test_flat_scan_to_ranges_bins_bearings_from_the_azimuth_range():
    lid = _load("lidar_server")
    n = 36
    # A single 2.0 m return dead ahead (azimuth 0 in a -180..180 sweep).
    depth = [-1.0] * n
    depth[n // 2] = 2.0
    frame = {"linearDepthData": depth, "azimuthRange": (-180.0, 180.0)}
    ranges, has = lid.flat_scan_to_ranges(frame, n, -math.pi, 0.1, 30.0)
    assert has
    idx = int(np.argmin(ranges))
    bearing = -math.pi + idx * (2 * math.pi / n)
    assert abs(bearing) < 2 * math.pi / n  # dead ahead, within one beam
    assert ranges[idx] == pytest.approx(2.0)


def test_isaac_scan_source_still_uses_the_shared_binning():
    """The own-kit source must call through flat_scan_to_ranges — the whole
    point of the extraction is ONE path, not a refactor that forked it."""
    import ast
    src = (_CONN / "lidar_server.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "IsaacScanSource":
            calls = {n.func.id for n in ast.walk(node)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)}
            assert "flat_scan_to_ranges" in calls
            return
    raise AssertionError("IsaacScanSource not found")


# -- harness/grader wiring -------------------------------------------------- #

def test_live_proof_harness_is_stdlib_only():
    """The harness runs on the render host's bare python3 — one third-party
    import and the proof silently becomes a proof nobody can run."""
    import ast
    path = _CONN.parents[1] / "examples" / "attached_sensors_live_proof.py"
    tree = ast.parse(path.read_text())
    stdlib_ok = {"argparse", "http", "json", "math", "os", "sys", "time",
                 "urllib", "__future__"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [(node.module or "").split(".")[0]]
        for n in names:
            assert n in stdlib_ok, f"non-stdlib import in harness: {n}"


def test_live_proof_harness_refuses_the_foreign_kit():
    path = _CONN.parents[1] / "examples" / "attached_sensors_live_proof.py"
    src = path.read_text()
    assert "8211" in src and "REFUSED" in src
    # And the refusal is in the client's constructor, not a comment.
    ns: dict = {}
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Bridge":
            body_src = ast.get_source_segment(src, node)
            assert "8211" in body_src
            return
    raise AssertionError("Bridge client not found in harness")
