#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Unified Isaac SENSOR RIG launcher — one command, all of a robot's sensors.

The connector servers (``camera_server.py`` MJPEG :8100, ``lidar_server.py``
JSON :8110, ``isaac_quadruped_server.py`` TCP :18973) each stand alone; this
example composes them into a RIG: pick the sensors a robot carries, and one
command brings the whole set up for software-in-the-loop testing.  The rig
launcher itself is glue only — plain python3, stdlib only, NO isaacsim, NO
tritium — the child processes are the ones that touch Isaac (and only under
``--source isaac`` inside Isaac's bundled python).

Both North Star halves: FUN — one command turns a sim robot into a fully
instrumented unit (camera + depth + stereo + a sweeping LiDAR ring) the
operator can register on the tactical map and watch light up.  PRODUCTION —
validates the multi-sensor bring-up story itself: real robots boot N sensor
services at once, and the fleet stack must see them all come healthy; this rig
proves the transport + health-check path (spawn -> poll /status -> ready
summary -> clean teardown) with zero hardware and zero GPU.

Design
------
  * ``build_rig_plan(config) -> list[list[str]]`` is PURE: given a config dict
    it returns the subprocess argv lists — no subprocess, no Isaac, no network.
    That seam is what the no-GPU tests exercise.
  * ``main()`` is the thin runnable shell: spawn the plan, poll each server's
    health endpoint (camera ``/status``, lidar ``/status``, body TCP accept),
    print a readiness summary, tear everything down on SIGINT/SIGTERM.
  * ``--print-plan`` prints the argv lists and exits — GPU-free, no processes;
    the copy-pasteable truth of what the rig would run.

Honest limits: the BODY server (``isaac_quadruped_server.py``) has no
synthetic serving mode — it always boots Isaac (its ``--asset procedural``
body is still an Isaac scene) — so ``--body`` only makes sense with an Isaac
python as ``--python``.  Camera + lidar run fully synthetic under plain
python3.

Run
---
    # Full synthetic rig, no GPU (system python3): camera + depth + stereo + lidar
    python3 examples/isaac_sensor_rig.py --camera --depth --stereo --lidar

    # Show the plan only (no processes, no GPU)
    python3 examples/isaac_sensor_rig.py --print-plan --camera --depth --lidar

    # Real Isaac render rig on the RTX box (Isaac's bundled python drives the
    # children; the launcher itself stays plain python3)
    python3 examples/isaac_sensor_rig.py --camera --depth --lidar \
        --source isaac \
        --python ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh

Then register the sensors in tritium-sc exactly as the individual server
docstrings describe (MJPEG camera source, edge sensor-bridge scan_url) — the
rig changes nothing about the wire contracts, it only brings them up together.
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# The connector servers this rig composes (repo-relative, resolved from here).
CONNECTORS_DIR = Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "connectors"

CAMERA_SERVER = "camera_server.py"
LIDAR_SERVER = "lidar_server.py"
BODY_SERVER = "isaac_quadruped_server.py"

# --------------------------------------------------------------------------- #
# The PURE core: config -> subprocess argv lists.  No subprocess, no Isaac.
# --------------------------------------------------------------------------- #

# Every key build_rig_plan understands, with its default.  A config dict may
# override any subset; unknown keys are an error (typo protection).
DEFAULT_CONFIG: dict = {
    # Which sensors the robot carries.
    "camera": False,          # MJPEG camera server (:8100)
    "depth": False,           # + colorized depth channel at /depth
    "stereo": False,          # + right-eye stereo channel at /mjpeg_right
    "lidar": False,           # JSON LiDAR range server (:8110)
    "body": False,            # quadruped BODY server (:18973) — Isaac-only
    # Frame/scan source for camera + lidar: "synthetic" (no GPU) or "isaac".
    "source": "synthetic",
    # Ports + bind host.
    "camera_port": 8100,
    "lidar_port": 8110,
    "body_port": 18973,
    "host": "0.0.0.0",
    # Identity + placement.
    "camera_id": "isaac-cam-01",
    "lidar_id": "isaac-lidar-01",
    "mount_prim": "/World/Tritium/Robot/lidar",   # lidar mount on the robot
    "scene": "",              # optional scene USD for camera + lidar (isaac)
    "body_asset": "auto",     # body server --asset: auto | go2 | procedural
    # Interpreter for the child servers (Isaac's python.sh for --source isaac).
    "python": None,           # None -> sys.executable
    # Where the connector scripts live (overridable for tests).
    "connectors_dir": None,   # None -> CONNECTORS_DIR
    # Command Center to register the rig's camera channels with once every
    # server is healthy ("" -> bring the rig up but leave the operator's map
    # alone).  Consumed by the runnable shell, not by build_rig_plan.
    "register_sc": "",
    "attach_to": "",          # tracked target the camera rides on ("" -> ask /status)
    # Dial OUT instead of being dialled: the camera server POSTs its frames to
    # this Command Center ("" -> pull, the historical default).  Needed
    # whenever the rig and the operator are different machines, because a kit
    # binds its stream to the RENDER HOST's loopback — an address the operator
    # cannot reach no matter what the registration advertises.
    "push_to": "",
    "push_fps": 0.0,          # 0 -> let the camera server pick its default
}

#: Pixel channels the camera server can push, keyed by the rig config flag
#: that enables each.  ``main`` (RGB) is unconditional — a camera rig always
#: has an RGB channel.  Depth pushes ``depth16`` (metric uint16-mm), never
#: ``depth``: the latter is a TURBO-colormapped JPEG, a picture of depth with
#: the range already destroyed, which perception cannot measure from.
_PUSH_CHANNELS: tuple[tuple[str, str], ...] = (
    ("", "main"),
    ("depth", "depth16"),
    ("stereo", "right"),
)


def build_rig_plan(config: dict | None = None) -> list[list[str]]:
    """The rig plan: which processes to start, as subprocess argv lists.

    PURE — validates the config, applies :data:`DEFAULT_CONFIG`, and returns
    one argv list per enabled server, in bring-up order (camera, lidar, body).
    Never spawns anything, never imports Isaac; ``main()`` and the tests are
    both consumers of this single seam.
    """
    cfg = dict(DEFAULT_CONFIG)
    overrides = dict(config or {})
    unknown = sorted(set(overrides) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown rig config keys: {', '.join(unknown)}")
    cfg.update(overrides)
    if cfg["source"] not in ("synthetic", "isaac"):
        raise ValueError(f"source must be 'synthetic' or 'isaac', got {cfg['source']!r}")
    if not (cfg["camera"] or cfg["lidar"] or cfg["body"]):
        raise ValueError("empty rig: enable at least one of camera/lidar/body")
    # An Isaac child needs an Isaac interpreter, and this launcher's own is by
    # construction NOT one (stdlib only, no isaacsim — see the module
    # docstring), so `python=None` under --source isaac is always wrong.
    # Without this the rig spawns full kit processes that each burn ~40 s of
    # boot and several GB of VRAM before dying on ModuleNotFoundError, which
    # surfaces as a health-poll timeout — indistinguishable from a slow sensor.
    needs_isaac = "--source isaac" if cfg["source"] == "isaac" else (
        "the body server (Isaac-only at every source)" if cfg["body"] else "")
    if needs_isaac and not cfg["python"]:
        raise ValueError(
            f"{needs_isaac} needs an Isaac interpreter: pass --python "
            "<isaac-sim>/python.sh (the rig launcher's own python cannot "
            "import isaacsim)")

    python = cfg["python"] or sys.executable
    conn = Path(cfg["connectors_dir"]) if cfg["connectors_dir"] else CONNECTORS_DIR

    plan: list[list[str]] = []
    if cfg["camera"]:
        argv = [
            python, str(conn / CAMERA_SERVER),
            "--source", cfg["source"],
            "--host", cfg["host"],
            "--port", str(cfg["camera_port"]),
            "--camera-id", cfg["camera_id"],
        ]
        if cfg["depth"]:
            # Both: --depth is the human-readable colormapped preview,
            # --depth16 the METRIC uint16-mm frame perception measures from.
            # The registration advertises depth16, so serving only --depth
            # would publish a channel that does not exist.
            argv += ["--depth", "--depth16"]
        if cfg["stereo"]:
            argv.append("--stereo")
        if cfg["scene"]:
            argv += ["--scene", cfg["scene"]]
        if cfg["push_to"]:
            argv += ["--push-to", cfg["push_to"]]
            if cfg["push_fps"]:
                argv += ["--push-fps", str(cfg["push_fps"])]
            # Every channel the rig REGISTERS must also be pushed, or the
            # operator gets a tile that can never render.
            for flag, channel in _PUSH_CHANNELS:
                if not flag or cfg[flag]:
                    argv += ["--push-channel", channel]
        plan.append(argv)
    if cfg["lidar"]:
        argv = [
            python, str(conn / LIDAR_SERVER),
            "--source", cfg["source"],
            "--host", cfg["host"],
            "--port", str(cfg["lidar_port"]),
            "--lidar-id", cfg["lidar_id"],
            "--mount-prim", cfg["mount_prim"],
        ]
        if cfg["scene"]:
            argv += ["--scene", cfg["scene"]]
        plan.append(argv)
    if cfg["body"]:
        # No --source here: the body server is Isaac-only (see docstring).
        plan.append([
            python, str(conn / BODY_SERVER),
            "--host", cfg["host"],
            "--port", str(cfg["body_port"]),
            "--asset", cfg["body_asset"],
        ])
    return plan


def plan_role(argv: list[str]) -> str:
    """Human name for a plan entry (from the connector script it runs)."""
    script = Path(argv[1]).name if len(argv) > 1 else "?"
    return {
        CAMERA_SERVER: "camera",
        LIDAR_SERVER: "lidar",
        BODY_SERVER: "body",
    }.get(script, script)


# --------------------------------------------------------------------------- #
# Health polling (stdlib only) — used by main(), skipped by --print-plan.
# --------------------------------------------------------------------------- #

def _poll_host(bind_host: str) -> str:
    """The address to poll a server bound on ``bind_host``."""
    return "127.0.0.1" if bind_host in ("0.0.0.0", "::") else bind_host


def _http_status(url: str, timeout: float = 2.0) -> dict | None:
    """GET a /status endpoint -> parsed JSON, or None while unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return None


def _tcp_accepts(host: str, port: int, timeout: float = 2.0) -> bool:
    """True when a TCP server (the body server) is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _check_ready(role: str, host: str, cfg: dict) -> tuple[bool, str]:
    """One readiness probe for a rig role -> (ready, human detail)."""
    if role == "camera":
        url = f"http://{host}:{cfg['camera_port']}/status"
        doc = _http_status(url)
        if doc:
            return True, (f"{url} online source={doc.get('source')} "
                          f"channels={','.join(doc.get('channels', []))}")
        return False, f"{url} unreachable"
    if role == "lidar":
        url = f"http://{host}:{cfg['lidar_port']}/status"
        doc = _http_status(url)
        if doc:
            return True, (f"{url} online source={doc.get('source')} "
                          f"beams={doc.get('num_beams')}")
        return False, f"{url} unreachable"
    if role == "body":
        if _tcp_accepts(host, cfg["body_port"]):
            return True, f"tcp://{host}:{cfg['body_port']} accepting"
        return False, f"tcp://{host}:{cfg['body_port']} unreachable"
    return False, "unknown role"


# --------------------------------------------------------------------------- #
# Runnable shell: spawn the plan, poll health, summarize, tear down cleanly.
# --------------------------------------------------------------------------- #

def _rig_sensors(cfg: dict, ready_roles: dict, host: str):
    """Ready rig roles -> the ``RigSensor`` list the lib seam plans against.

    ONE camera process serves up to three pixel channels, so a single ready
    ``camera`` role fans out into rgb / depth16 / right depending on what the
    rig was asked to bring up.  LiDAR and body are included as sensors even
    though they carry no pixels — the seam is what decides they get no feed,
    not this function, so the decision stays tested in one place.
    """
    from tritium_lib.fleet.sensor_rig import RigSensor

    sensors = []
    if "camera" in ready_roles:
        roles = ["camera"]
        if cfg.get("depth"):
            roles.append("depth")
        if cfg.get("stereo"):
            roles.append("stereo_right")
        for role in roles:
            sensors.append(RigSensor(
                role=role, host=host, port=cfg["camera_port"], ready=True,
                attach_to=cfg.get("attach_to", "") or "",
            ))
    if "lidar" in ready_roles:
        sensors.append(RigSensor(role="lidar", host=host,
                                 port=cfg["lidar_port"], ready=True))
    if "body" in ready_roles:
        sensors.append(RigSensor(role="body", host=host,
                                 port=cfg["body_port"], ready=True))
    return sensors


def _register_with_sc(cfg: dict, ready_roles: dict, host: str, sc_url: str) -> bool:
    """POST the rig's sensors to the Command Center; report honestly.

    Returns True only when the rig genuinely reached the operator.  The
    outcome scoring is :func:`tritium_lib.fleet.sensor_rig.summarize_bringup`,
    which refuses to call an empty or partly-failed bring-up healthy.
    """
    try:
        from tritium_lib.fleet.sensor_rig import (
            registration_plan, summarize_bringup,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        print(f"--register-sc needs tritium_lib importable: {exc}", file=sys.stderr)
        return False

    push = bool(cfg.get("push_to"))
    calls = registration_plan(_rig_sensors(cfg, ready_roles, host), push=push)
    if not calls:
        print("REGISTER: no pixel sensors in this rig — nothing to register")
        return False

    outcomes: list[tuple[str, str]] = []
    for call in calls:
        source_id = call.payload["source_id"]
        url = sc_url.rstrip("/") + call.path
        body = json.dumps(call.payload).encode()
        req = urllib.request.Request(
            url, data=body, method=call.method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                doc = json.loads(resp.read().decode() or "{}")
                already = bool(doc.get("already_registered"))
                outcomes.append(
                    (source_id, "already_registered" if already else "registered"))
                print(f"  {'~' if already else '+'} {source_id:<16} "
                      f"{call.role:<13} attached={doc.get('attached_to')} "
                      f"mount_discovered={doc.get('mount_discovered')}")
        except Exception as exc:
            outcomes.append((source_id, "failed"))
            print(f"  ! {source_id:<16} {call.role:<13} {exc}")

    report = summarize_bringup(outcomes)
    print(f"\n{report}")
    return report.ok


def _run_rig(plan: list[list[str]], cfg: dict, timeout_s: float) -> int:
    host = _poll_host(cfg["host"])
    procs: list[tuple[str, subprocess.Popen]] = []
    stop = {"requested": False}

    def _teardown():
        for role, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5.0
        for role, proc in procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"  stopped {role} (rc={proc.returncode})")

    def _on_signal(signum, _frame):
        print(f"\nsignal {signum} — tearing down the rig")
        stop["requested"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    for argv in plan:
        role = plan_role(argv)
        print(f"starting {role}: {shlex.join(argv)}")
        procs.append((role, subprocess.Popen(argv)))

    # Poll every server until ready, a child dies, or the timeout expires.
    ready: dict[str, str] = {}
    deadline = time.time() + timeout_s
    while len(ready) < len(procs) and time.time() < deadline and not stop["requested"]:
        for role, proc in procs:
            if role in ready:
                continue
            if proc.poll() is not None:
                print(f"RIG FAILED: {role} exited early (rc={proc.returncode})")
                _teardown()
                return 1
            ok, detail = _check_ready(role, host, cfg)
            if ok:
                ready[role] = detail
        if len(ready) < len(procs):
            time.sleep(0.5)

    print(f"\nSENSOR RIG {'READY' if len(ready) == len(procs) else 'PARTIAL'} "
          f"{len(ready)}/{len(procs)}")
    for role, proc in procs:
        detail = ready.get(role) or _check_ready(role, host, cfg)[1]
        mark = "ok " if role in ready else "!! "
        print(f"  {mark}{role:<7} {detail}")
    if len(ready) < len(procs) and not stop["requested"]:
        print("RIG FAILED: not all servers became ready — tearing down")
        _teardown()
        return 1

    if cfg.get("register_sc"):
        print(f"\nregistering the rig with the Command Center at "
              f"{cfg['register_sc']}")
        if not _register_with_sc(cfg, ready, host, cfg["register_sc"]):
            # A rig the operator cannot see is not a rig that came up.  Failing
            # here rather than idling is the point: the alternative is a green
            # console next to a tactical map with nothing on it.
            print("RIG FAILED: sensors did not reach the operator — tearing down")
            _teardown()
            return 1

    print("\nrig up — Ctrl-C to stop")
    try:
        while not stop["requested"]:
            for role, proc in procs:
                if proc.poll() is not None:
                    print(f"{role} exited (rc={proc.returncode}) — tearing down")
                    _teardown()
                    return 1
            time.sleep(1.0)
    finally:
        _teardown()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Unified Isaac sensor rig launcher (camera + lidar + body)")
    ap.add_argument("--camera", action="store_true", help="MJPEG camera server")
    ap.add_argument("--depth", action="store_true", help="camera depth channel")
    ap.add_argument("--stereo", action="store_true", help="camera stereo channel")
    ap.add_argument("--lidar", action="store_true", help="JSON LiDAR server")
    ap.add_argument("--body", action="store_true",
                    help="quadruped body server (Isaac-only — see docstring)")
    ap.add_argument("--source", choices=["synthetic", "isaac"], default="synthetic")
    ap.add_argument("--camera-port", type=int, default=DEFAULT_CONFIG["camera_port"])
    ap.add_argument("--lidar-port", type=int, default=DEFAULT_CONFIG["lidar_port"])
    ap.add_argument("--body-port", type=int, default=DEFAULT_CONFIG["body_port"])
    ap.add_argument("--host", default=DEFAULT_CONFIG["host"])
    ap.add_argument("--camera-id", default=DEFAULT_CONFIG["camera_id"])
    ap.add_argument("--lidar-id", default=DEFAULT_CONFIG["lidar_id"])
    ap.add_argument("--mount-prim", default=DEFAULT_CONFIG["mount_prim"],
                    help="robot prim the lidar mounts under")
    ap.add_argument("--scene", default="", help="scene USD for camera + lidar (isaac)")
    ap.add_argument("--body-asset", choices=["auto", "go2", "procedural"],
                    default=DEFAULT_CONFIG["body_asset"])
    ap.add_argument("--python", default=None,
                    help="interpreter for the child servers "
                         "(Isaac's python.sh for --source isaac)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="seconds to wait for every server to become ready")
    ap.add_argument("--register-sc", default="", metavar="URL",
                    help="after the rig is ready, register its camera "
                         "channels with this Command Center "
                         "(e.g. http://localhost:8000) so the operator sees "
                         "the robot's own eyes with nothing typed")
    ap.add_argument("--attach-to", default="",
                    help="tracked target id the camera rides on; omit to let "
                         "the camera server's /status advertisement decide")
    ap.add_argument("--push-to", default="", metavar="URL",
                    help="camera dials OUT: the camera server POSTs its "
                         "frames to this Command Center instead of waiting "
                         "to be dialled. Use whenever the rig and the "
                         "operator are different machines -- a kit binds its "
                         "stream to the RENDER HOST's loopback, so a pull "
                         "registration can never reach it. Implies push-mode "
                         "registration for --register-sc")
    ap.add_argument("--push-fps", type=float, default=0.0,
                    help="frames per second to push (0 -> the camera "
                         "server's own default)")
    ap.add_argument("--print-plan", action="store_true",
                    help="print the subprocess argv lists and exit (no processes)")
    args = ap.parse_args(argv)

    # A rig is normally launched with its output redirected to a log.  Python
    # block-buffers a non-tty stdout, so the readiness summary and the
    # registration result -- the two things an operator actually reads -- would
    # sit unflushed in the buffer for the entire life of a long-running rig.
    # Found by running it: the child servers' stderr appeared in the log and
    # the launcher's own stdout did not.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - exotic streams
        pass

    cfg = {
        "camera": args.camera, "depth": args.depth, "stereo": args.stereo,
        "lidar": args.lidar, "body": args.body,
        "source": args.source,
        "camera_port": args.camera_port, "lidar_port": args.lidar_port,
        "body_port": args.body_port, "host": args.host,
        "camera_id": args.camera_id, "lidar_id": args.lidar_id,
        "mount_prim": args.mount_prim, "scene": args.scene,
        "body_asset": args.body_asset, "python": args.python,
        "register_sc": args.register_sc, "attach_to": args.attach_to,
        "push_to": args.push_to, "push_fps": args.push_fps,
    }
    if not (cfg["camera"] or cfg["lidar"] or cfg["body"]):
        # No sensors named -> the standard full sensor rig (body stays opt-in:
        # it needs Isaac's python even for a procedural asset).
        cfg["camera"] = cfg["lidar"] = True

    try:
        plan = build_rig_plan(cfg)
    except ValueError as exc:
        print(f"bad rig config: {exc}", file=sys.stderr)
        return 2

    if args.print_plan:
        print(f"SENSOR RIG PLAN ({len(plan)} process"
              f"{'es' if len(plan) != 1 else ''}, source={cfg['source']}):")
        for entry in plan:
            print(f"  [{plan_role(entry)}] {shlex.join(entry)}")
        return 0

    return _run_rig(plan, cfg, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
