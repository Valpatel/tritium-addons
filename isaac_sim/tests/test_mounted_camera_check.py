# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""No-GPU tests for examples/mounted_camera_check.py.

The live agreement number this script reports can only be produced against a
running Isaac.  What CAN be pinned on any box is the part that decides what
gets sent there: the parameter-injection helper and the mount geometry handed
to USD.  Both have bitten before -- a mis-injected parameter and a silently
dropped precision each looked exactly like a broken camera.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "mounted_camera_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("mounted_camera_check", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("tritium_lib.geo.camera_mount")
mcc = _load()


def test_literals_emit_one_assignment_per_parameter():
    src = mcc._literals(A=1, B="x")
    assert src == "A = 1\nB = 'x'\n"


def test_literals_repr_quotes_hostile_prim_paths():
    """Parameters are injected as reprs, not interpolated, so a path
    containing quotes cannot terminate the string and inject code."""
    src = mcc._literals(CAM_PATH="/World/a'; import os; x='")
    scope: dict = {}
    exec(src, scope)
    assert scope["CAM_PATH"] == "/World/a'; import os; x='"


def test_default_mount_offset_is_forward_port_and_up():
    """The stage offset handed to USD is the mount in body axes on a Z-up
    stage -- forward on +X, port on +Y, up on +Z."""
    assert mcc.DEFAULT_MOUNT.stage_offset(up_axis="Z") == pytest.approx(
        (0.30, 0.10, 0.12)
    )


def test_default_mount_looks_downward():
    """A nose camera aimed at the horizon sees no ground in front of the dog,
    which is the whole point of mounting it there."""
    assert mcc.DEFAULT_MOUNT.tilt_deg < 0.0


def test_pose_sweep_covers_all_four_quadrants_and_off_cardinals():
    """A world-frame-offset bug agrees at heading 0 and diverges elsewhere, so
    the sweep is only meaningful if it leaves the cardinals."""
    yaws = {yaw for _, _, _, yaw in mcc.DEFAULT_POSES}
    assert {0.0, 90.0, 180.0, 270.0} <= yaws
    assert any(y % 90.0 != 0.0 for y in yaws)


def test_example_does_not_import_isaacsim():
    """It runs on the operator's box and talks to Isaac over the bridge; a
    real isaacsim import would make it un-runnable where it is meant to run."""
    src = _EXAMPLE.read_text()
    assert "import isaacsim" not in src
    assert "from isaacsim" not in src
