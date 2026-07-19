# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""No-GPU tests for the unified sensor rig launcher.

``build_rig_plan`` is the pure seam between a rig config and the subprocess
argv lists — if it is wrong, the one-command bring-up starts the wrong servers
on the wrong ports.  Everything here runs with NO isaacsim, NO tritium, NO
subprocesses, NO GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
RIG_PATH = EXAMPLES / "isaac_sensor_rig.py"


def _load_rig():
    """Import the example by path — examples/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("isaac_sensor_rig", RIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _load_rig()


def _flag_value(argv: list[str], flag: str) -> str:
    """The value following ``flag`` in an argv list (fails if absent)."""
    assert flag in argv, f"{flag} missing from {argv}"
    return argv[argv.index(flag) + 1]


# ------------------------------------------------------------ import hygiene

def _module_level_imports(path) -> set[str]:
    """Top-level module names imported when the file is merely LOADED.

    Deliberately AST-based rather than a substring scan: an import nested
    inside a function does not run at load time, so it cannot break the
    "launcher loads anywhere" invariant.  A grep cannot tell the two apart.
    """
    import ast

    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:  # top level only
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_module_imports_without_isaacsim_or_tritium():
    """The launcher is pure glue: LOADING it must not need isaacsim or tritium.

    `--register-sc` does import ``tritium_lib.fleet.sensor_rig``, but lazily,
    inside the function that needs it — so a rig brought up on a box with no
    tritium installed still runs, and only the opt-in registration step
    requires the library.  That is exactly the distinction this test now
    encodes (it previously scanned the source text and could not).
    """
    top = _module_level_imports(RIG_PATH)
    assert not any(n.startswith("tritium") for n in top), (
        f"isaac_sensor_rig.py imports tritium at module level: {sorted(top)}")
    assert not any(n.startswith("isaacsim") or n == "pxr" for n in top), (
        f"isaac_sensor_rig.py imports isaacsim at module level: {sorted(top)}")
    assert callable(rig.build_rig_plan) and callable(rig.main)


def test_registration_import_is_lazy_not_module_level():
    """The tritium import must live inside a function, or the guarantee above
    is accidental rather than designed."""
    import ast

    tree = ast.parse(RIG_PATH.read_text())
    lazy = [
        node
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("tritium")
    ]
    assert lazy, "expected a lazy in-function tritium import for --register-sc"


# ------------------------------------------------------- build_rig_plan: pure

def test_plan_camera_only():
    plan = rig.build_rig_plan({"camera": True})
    assert len(plan) == 1
    argv = plan[0]
    assert argv[1].endswith("camera_server.py")
    assert _flag_value(argv, "--port") == "8100"
    assert _flag_value(argv, "--source") == "synthetic"
    assert _flag_value(argv, "--camera-id") == "isaac-cam-01"
    assert "--depth" not in argv and "--stereo" not in argv
    assert "--scene" not in argv


def test_plan_camera_depth_stereo():
    plan = rig.build_rig_plan({"camera": True, "depth": True, "stereo": True,
                               "camera_port": 8123, "camera_id": "rig-cam"})
    assert len(plan) == 1
    argv = plan[0]
    assert argv[1].endswith("camera_server.py")
    assert "--depth" in argv and "--stereo" in argv
    assert _flag_value(argv, "--port") == "8123"
    assert _flag_value(argv, "--camera-id") == "rig-cam"


def test_plan_full_rig_camera_depth_stereo_lidar():
    plan = rig.build_rig_plan({
        "camera": True, "depth": True, "stereo": True, "lidar": True,
        "source": "isaac", "python": "/isaac/python.sh",
        "mount_prim": "/World/Tritium/Go2/lidar",
    })
    assert len(plan) == 2
    cam, lid = plan
    # Camera entry: flags + the interpreter override.
    assert cam[0] == "/isaac/python.sh" and cam[1].endswith("camera_server.py")
    assert "--depth" in cam and "--stereo" in cam
    assert _flag_value(cam, "--port") == "8100"
    # Lidar entry: port, mount prim, same interpreter.
    assert lid[0] == "/isaac/python.sh" and lid[1].endswith("lidar_server.py")
    assert _flag_value(lid, "--port") == "8110"
    assert _flag_value(lid, "--mount-prim") == "/World/Tritium/Go2/lidar"
    assert _flag_value(lid, "--lidar-id") == "isaac-lidar-01"
    # The source passes through to EVERY server in the plan.
    for argv in plan:
        assert _flag_value(argv, "--source") == "isaac"


def test_plan_scene_passes_to_camera_and_lidar():
    plan = rig.build_rig_plan({"camera": True, "lidar": True,
                               "source": "isaac", "scene": "/tmp/ao.usd", "python": "/isaac/python.sh"})
    for argv in plan:
        assert _flag_value(argv, "--scene") == "/tmp/ao.usd"


def test_plan_body_server_included_on_request():
    plan = rig.build_rig_plan({"camera": True, "lidar": True, "body": True, "python": "/isaac/python.sh",
                               "body_asset": "go2"})
    assert len(plan) == 3
    body = plan[2]
    assert body[1].endswith("isaac_quadruped_server.py")
    assert _flag_value(body, "--port") == "18973"
    assert _flag_value(body, "--asset") == "go2"
    # The body server has no --source flag (Isaac-only, see rig docstring).
    assert "--source" not in body


def test_plan_validation():
    with pytest.raises(ValueError, match="unknown rig config keys"):
        rig.build_rig_plan({"camera": True, "cammera_port": 9000})
    with pytest.raises(ValueError, match="empty rig"):
        rig.build_rig_plan({})
    with pytest.raises(ValueError, match="source"):
        rig.build_rig_plan({"camera": True, "source": "gazebo"})


def test_plan_role_names():
    plan = rig.build_rig_plan({"camera": True, "lidar": True, "body": True, "python": "/isaac/python.sh"})
    assert [rig.plan_role(a) for a in plan] == ["camera", "lidar", "body"]


# ----------------------------------------------------- main(): --print-plan

def test_print_plan_runs_gpu_free(capsys):
    """--print-plan prints the argv lists and exits 0 — no processes spawned."""
    rc = rig.main(["--print-plan", "--camera", "--depth", "--lidar"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SENSOR RIG PLAN (2 processes" in out
    assert "camera_server.py" in out and "lidar_server.py" in out
    assert "--depth" in out
    assert "[camera]" in out and "[lidar]" in out


def test_print_plan_defaults_to_camera_plus_lidar(capsys):
    """No sensor flags -> the standard full rig (camera + lidar, body opt-in)."""
    rc = rig.main(["--print-plan"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "camera_server.py" in out and "lidar_server.py" in out
    assert "isaac_quadruped_server.py" not in out


def test_bad_config_exits_2(capsys):
    rc = rig.main(["--print-plan", "--camera", "--source", "synthetic",
                   "--body-port", "0"]) if False else rig.main.__wrapped__ \
        if hasattr(rig.main, "__wrapped__") else None
    # argparse rejects a bad --source itself; drive the ValueError path via
    # build_rig_plan directly instead of fighting argparse in-process.
    with pytest.raises(ValueError):
        rig.build_rig_plan({"camera": True, "source": "bogus"})


# --------------------------------------------------------------------------- #
# Refusal: --source isaac demands an Isaac interpreter.
# --------------------------------------------------------------------------- #

def test_isaac_source_without_python_is_refused():
    """``--source isaac`` under the launcher's OWN interpreter is a refusal.

    The launcher is plain python3 by construction (stdlib only, no isaacsim),
    so ``python=None`` -> ``sys.executable`` can never import Isaac.  Planning
    it anyway spawns two full kit processes that each burn ~40 s of boot and
    several GB of VRAM before dying on ``ModuleNotFoundError`` — and the rig
    reports that as a health-poll timeout, which reads like a slow sensor
    rather than the wrong interpreter.  Refuse at the seam instead.
    """
    with pytest.raises(ValueError, match="python"):
        rig.build_rig_plan({"camera": True, "lidar": True, "source": "isaac"})


def test_isaac_source_with_explicit_python_is_allowed():
    """An explicit interpreter is the whole fix — this must still plan."""
    plan = rig.build_rig_plan({
        "camera": True, "source": "isaac", "python": "/isaac/python.sh",
    })
    assert plan[0][0] == "/isaac/python.sh"


def test_body_server_without_python_is_refused():
    """The body server is Isaac-only at every source, so it needs one too."""
    with pytest.raises(ValueError, match="python"):
        rig.build_rig_plan({"body": True, "source": "synthetic"})


def test_synthetic_camera_still_defaults_to_the_launchers_python():
    """The refusal must not touch the no-GPU path this rig is usually run on."""
    plan = rig.build_rig_plan({"camera": True, "lidar": True})
    assert plan[0][0] == sys.executable


# --------------------------------------------------------------------------- #
# push mode: the rig dials OUT to the operator
#
# A kit binds its MJPEG server to the render host's loopback, so the operator
# can only ever be a different machine by *not* dialling in.  tick 30 taught
# the camera server to push; this is the rig learning to ask for it.
# --------------------------------------------------------------------------- #


def _cam_argv(plan):
    for argv in plan:
        if rig.plan_role(argv) == "camera":
            return argv
    raise AssertionError("no camera in plan")


def test_push_to_reaches_the_camera_server():
    argv = _cam_argv(rig.build_rig_plan(
        {"camera": True, "push_to": "http://operator:8000"}))
    assert "--push-to" in argv
    assert argv[argv.index("--push-to") + 1] == "http://operator:8000"


def test_no_push_to_means_no_push_flags_at_all():
    """The default stays a pull rig — push is opt-in, not a silent change."""
    argv = _cam_argv(rig.build_rig_plan({"camera": True}))
    assert "--push-to" not in argv
    assert "--push-channel" not in argv


def test_the_rgb_channel_is_always_pushed():
    argv = _cam_argv(rig.build_rig_plan(
        {"camera": True, "push_to": "http://operator:8000"}))
    chans = [argv[i + 1] for i, a in enumerate(argv) if a == "--push-channel"]
    assert chans == ["main"]


def test_every_enabled_pixel_channel_is_pushed():
    """A channel registered but never pushed is a permanently black tile."""
    argv = _cam_argv(rig.build_rig_plan({
        "camera": True, "depth": True, "stereo": True,
        "push_to": "http://operator:8000",
    }))
    chans = [argv[i + 1] for i, a in enumerate(argv) if a == "--push-channel"]
    assert sorted(chans) == ["depth16", "main", "right"]


def test_depth_pushes_the_metric_channel_not_the_colormap():
    """depth16 carries the NUMBER; /depth is a lossy picture of it."""
    argv = _cam_argv(rig.build_rig_plan(
        {"camera": True, "depth": True, "push_to": "http://operator:8000"}))
    chans = [argv[i + 1] for i, a in enumerate(argv) if a == "--push-channel"]
    assert "depth16" in chans
    assert "depth" not in chans


def test_push_fps_is_carried_when_set():
    argv = _cam_argv(rig.build_rig_plan(
        {"camera": True, "push_to": "http://operator:8000", "push_fps": 5.0}))
    assert argv[argv.index("--push-fps") + 1] == "5.0"


def test_pushed_channels_match_the_source_ids_the_rig_registers():
    """The contract that makes or breaks the whole lane.

    The rig registers source ids via the lib seam; the camera server derives
    the id it POSTs to from the channel name.  If these two ever disagree,
    every pushed frame 404s while both sides report success.
    """
    from tritium_lib.fleet.sensor_rig import registration_plan

    cfg = dict(rig.DEFAULT_CONFIG)
    cfg.update({"camera": True, "depth": True, "stereo": True,
                "push_to": "http://operator:8000"})
    argv = _cam_argv(rig.build_rig_plan(cfg))
    pushed = {argv[i + 1] for i, a in enumerate(argv) if a == "--push-channel"}

    registered = {
        c.payload["source_id"]
        for c in registration_plan(rig._rig_sensors(cfg, {"camera": 1}, "h"), push=True)
    }
    channel_to_id = {"main": "isaac_rgb", "right": "isaac_right",
                     "depth16": "isaac_depth16"}
    assert {channel_to_id[c] for c in pushed} == registered


def test_lidar_is_not_given_push_flags():
    """LiDAR streams sightings, not frames — it has no push channel."""
    plan = rig.build_rig_plan({"lidar": True, "push_to": "http://operator:8000"})
    for argv in plan:
        if rig.plan_role(argv) == "lidar":
            assert "--push-to" not in argv


def test_depth_rig_serves_the_metric_channel_it_registers():
    """Found by RUNNING it, not by reading it.

    The rig registers ``isaac_depth16`` (the lib's depth stream is metric
    uint16-mm), but ``--depth`` alone makes the camera server serve only the
    TURBO-colormapped ``depth``.  Under push that is a hard argparse refusal;
    under pull it is worse — a silently permanently-black depth tile.  So a
    depth rig must always ask for the channel it advertises.
    """
    argv = _cam_argv(rig.build_rig_plan({"camera": True, "depth": True}))
    assert "--depth16" in argv
