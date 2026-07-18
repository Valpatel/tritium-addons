# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the Isaac pose bridge — capability 2's read/convert/emit path.

Everything here runs with NO Isaac, NO GPU and NO network: the bridge takes a
``transport`` seam precisely so the whole path can be pinned offline.  The live
behaviour these fakes stand in for was measured against a real Isaac Sim 6.0 on
the RTX 4090 (7 ground-truth poses, max error 0.000 m / 0.000 deg) — see
``docs/ISAAC-SIM-STATUS.md``.

The failure that matters most is the LAST one: a dropped pose must raise rather
than quietly reporting the origin facing north, because a body silently pinned
at (0, 0) on the operator's map is worse than a body missing from it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "clients")
)

import pose_bridge  # noqa: E402
from pose_bridge import (  # noqa: E402
    IsaacPoseBridge,
    StagePose,
    pose_to_target,
)


def _stage_reply(translation, quat_wxyz, *, up_axis="Z", mpu=1.0):
    return {
        "status": "success",
        "result": {
            "return_value": {
                "ok": True,
                "prim": "/World/Go2",
                "type": "Xform",
                "translation": list(translation),
                "quat_wxyz": list(quat_wxyz),
                "up_axis": up_axis,
                "meters_per_unit": mpu,
            }
        },
    }


def _yaw_quat(yaw_deg: float):
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _bridge(reply, *, record=None):
    def transport(path, payload):
        if record is not None:
            record.append((path, payload))
        if path == "/health":
            return {"status": "success", "result": {"up_axis": "Z", "meters_per_unit": 1.0}}
        return reply

    return IsaacPoseBridge(transport=transport)


# --------------------------------------------------------------------------
# The conversion, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaw,heading",
    [(0.0, 90.0), (90.0, 0.0), (180.0, 270.0), (270.0, 180.0), (30.0, 60.0)],
)
def test_live_pose_converts_to_expected_heading(yaw, heading):
    """Mirrors the live sweep: isaac yaw -> tritium compass heading."""
    bridge = _bridge(_stage_reply((12.0, -7.0, 0.6), _yaw_quat(yaw)))
    pose = bridge.read_stage_pose().to_local()
    assert pose.east_m == pytest.approx(12.0)
    assert pose.north_m == pytest.approx(-7.0)
    assert pose.heading_deg == pytest.approx(heading, abs=1e-6)


def test_stage_metadata_drives_the_frame_not_a_hardcoded_default():
    """A centimetre Y-up stage must still report correct metres/heading."""
    bridge = _bridge(
        _stage_reply((300.0, 500.0, -400.0), _yaw_quat(0.0), up_axis="Y", mpu=0.01)
    )
    pose = bridge.read_stage_pose().to_local()
    # Y-up: east=X, north=-Z, up=Y; then scaled by 0.01.
    assert (pose.east_m, pose.north_m, pose.up_m) == pytest.approx((3.0, 4.0, 5.0))


def test_read_target_shapes_a_tracked_target():
    bridge = _bridge(_stage_reply((1.0, 2.0, 0.3), _yaw_quat(90.0)))
    t = bridge.read_target("isaac_go2_01")
    assert t["target_id"] == "isaac_go2_01"
    assert t["source"] == "isaac_sim"       # provenance stays honest
    assert t["classification"] == "robot"
    assert (t["x"], t["y"], t["z"]) == pytest.approx((1.0, 2.0, 0.3))
    assert t["heading"] == pytest.approx(0.0)
    assert isinstance(t["timestamp"], float)


def test_pose_to_target_rounds_but_does_not_distort():
    from pose_bridge import LocalPose

    t = pose_to_target(LocalPose(1.23456789, -2.3456789, 0.5, 123.456789))
    assert t["x"] == pytest.approx(1.23456789, abs=1e-4)
    assert t["heading"] == pytest.approx(123.456789, abs=1e-3)


# --------------------------------------------------------------------------
# The query contract
# --------------------------------------------------------------------------

def test_prim_path_is_injected_as_a_literal_not_interpolated():
    """A quoted prim path must not be able to break the executed snippet."""
    record: list = []
    bridge = _bridge(_stage_reply((0, 0, 0), _yaw_quat(0)), record=record)
    bridge.prim_path = "/World/it's_a_\"dog\""
    bridge.read_stage_pose()
    code = record[-1][1]["code"]
    assert code.startswith("PRIM_PATH = ")
    # The snippet must still be valid python despite the hostile path.
    compile(code, "<snippet>", "exec")


def test_uses_world_transform_not_local():
    """Reading the LOCAL translation of a parented robot is the classic bug."""
    record: list = []
    bridge = _bridge(_stage_reply((0, 0, 0), _yaw_quat(0)), record=record)
    bridge.read_stage_pose()
    assert "ComputeLocalToWorldTransform" in record[-1][1]["code"]


# --------------------------------------------------------------------------
# Failures must be loud
# --------------------------------------------------------------------------

def test_missing_prim_raises_rather_than_reporting_origin():
    def transport(path, payload):
        return {"status": "success",
                "result": {"return_value": {"ok": False, "error": "prim not found"}}}

    with pytest.raises(LookupError, match="prim not found"):
        IsaacPoseBridge(transport=transport).read_stage_pose()


def test_bridge_error_is_raised_and_counted():
    def transport(path, payload):
        return {"status": "error", "error": "stage is closed"}

    bridge = IsaacPoseBridge(transport=transport)
    with pytest.raises(ConnectionError, match="stage is closed"):
        bridge.read_stage_pose()
    assert bridge.errors == 1
    assert "stage is closed" in (bridge.last_error or "")


def test_malformed_reply_is_raised_not_silently_defaulted():
    def transport(path, payload):
        return {"status": "success", "result": {}}

    with pytest.raises(ConnectionError):
        IsaacPoseBridge(transport=transport).read_stage_pose()


def test_degenerate_quaternion_does_not_masquerade_as_north():
    """A zeroed pose must raise -- reporting heading 0 would be a lie."""
    bridge = _bridge(_stage_reply((5.0, 5.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="quaternion"):
        bridge.read_stage_pose().to_local()


def test_reads_are_counted_for_observability():
    bridge = _bridge(_stage_reply((0, 0, 0), _yaw_quat(0)))
    for _ in range(3):
        bridge.read_stage_pose()
    assert bridge.to_dict()["reads"] == 3
    assert bridge.to_dict()["errors"] == 0


def test_frame_maths_has_exactly_one_source_of_truth():
    """The vendored fallback copy was deleted on purpose -- keep it deleted.

    Two copies of a sign convention diverge silently and put the operator's
    icon in the wrong place with every test still green.
    """
    bridge = _bridge(_stage_reply((0, 0, 0), _yaw_quat(0)))
    assert bridge.to_dict()["frame_source"] == "tritium_lib.geo.isaac_frame"


def test_no_isaacsim_import_anywhere_in_the_module():
    """The isaac-bridge rule: consumers never import isaacsim."""
    src = (
        Path(__file__).resolve().parents[1]
        / "isaac_sim_addon" / "clients" / "pose_bridge.py"
    ).read_text()
    assert "import isaacsim" not in src
    assert "from isaacsim" not in src


def test_does_not_post_to_a_nonexistent_sc_route():
    """Guards the honesty fix: SC has no pose-ingest seam yet (501/404).

    If someone re-adds a POST helper here, it must be because the route now
    exists -- this test failing is the prompt to check that, not to delete it.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "isaac_sim_addon" / "clients" / "pose_bridge.py"
    ).read_text()
    assert "/api/targets/update" not in src


# ---------------------------------------------------------------------------
# SC ingest — the seam that was missing until 2026-07-18
# ---------------------------------------------------------------------------

def test_target_to_sighting_shapes_the_robot_pose_ingest():
    """Dispatch source is the MODALITY; provenance rides along in ``origin``."""
    bridge = _bridge(_stage_reply((10.0, 20.0, 0.4), _yaw_quat(0)))
    sighting = pose_bridge.target_to_sighting(bridge.read_target("go2_01"))

    assert sighting["source"] == "robot_pose"      # what SC dispatches on
    assert sighting["origin"] == "isaac_sim"       # what actually produced it
    assert sighting["target_id"] == "go2_01"
    assert sighting["position"] == {"x": 10.0, "y": 20.0}
    assert sighting["heading"] == 90.0             # Isaac yaw 0 == east
    assert sighting["ground_truth"] is True        # a stage pose IS exact
    assert sighting["asset_type"] == "quadruped"


def test_post_sighting_hits_the_sighting_route_with_json():
    seen = {}

    def opener(url, data, timeout):
        seen["url"] = url
        seen["body"] = json.loads(data.decode("utf-8"))
        return b'{"status": "accepted", "source": "robot_pose"}'

    reply = pose_bridge.post_sighting(
        {"source": "robot_pose", "target_id": "go2_01"},
        sc_url="http://sc.example:8000/",
        opener=opener,
    )

    assert seen["url"] == "http://sc.example:8000/api/sighting"
    assert seen["body"]["source"] == "robot_pose"
    assert reply["status"] == "accepted"


def test_full_stage_to_sc_path_runs_offline():
    """Read a fake stage, convert, shape, post — no Isaac, no GPU, no network."""
    posted = []

    def opener(url, data, timeout):
        posted.append(json.loads(data.decode("utf-8")))
        return b'{"status": "accepted"}'

    bridge = _bridge(_stage_reply((-3.5, 21.0, 0.45), _yaw_quat(180.0)))
    pose_bridge.post_sighting(
        pose_bridge.target_to_sighting(bridge.read_target("go2_01")),
        opener=opener,
    )

    assert len(posted) == 1
    assert posted[0]["position"] == {"x": -3.5, "y": 21.0}
    assert posted[0]["heading"] == 270.0  # Isaac yaw 180 (west) == heading 270
