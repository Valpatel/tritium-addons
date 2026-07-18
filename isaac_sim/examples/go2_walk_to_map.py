#!/usr/bin/env python3
"""Walk a Newton Go2 and watch its track appear on the Tritium tactical map.

This is the first script in the lane that runs the *whole* chain at once.
Every piece of it has been demonstrated live on its own -- the Go2 walks
under Newton (tick 10), a pose lands on the map through
``POST /api/sighting`` (tick 5), stage coordinates convert to Tritium local
ENU (``tritium_lib.geo.isaac_frame``) -- but nothing has ever closed the
loop: a body that is *actually locomoting* streaming its pose to a running
Command Center, with the resulting operator-visible track compared against
what the simulator says really happened.

That comparison is the whole point, and it is the one thing a stack of
individually-green components cannot give you.  A pose ingest can wedge and
keep repainting its first fix; a frame conversion can transpose two axes; a
feed can run a second behind.  In every one of those cases the track exists,
the endpoint returns 200, and the map looks alive.  Only
``tritium_lib.geo.path_fidelity.compare_paths`` -- time-aligned, scored on
the worst sample rather than the average -- tells them apart.

How the timing works, because it is the part that is easy to get wrong.  The
loop below runs at a fixed rate and does three things per tick: read the
newest true pose out of the running sim, POST it as a ``robot_pose``
sighting, then read ``GET /api/targets`` back and record what the map is
showing.  Both timelines are stamped on the *client's* wall clock, seconds
since the stream opened, so the error this reports is a real end-to-end
number -- it includes ingest and serving latency, and at one tick of lag a
0.25 m/s walk should sit a few centimetres out.  Only the newest sample is
sent per tick rather than every buffered row: that is what fixed-rate robot
telemetry actually looks like, and batching several rows under one wall
timestamp would corrupt the reference timeline.

Topology: this runs **on the RTX host** beside the Newton kit, because the
MCP bridge listens on loopback and resets forwarded connections.  Reach a
Command Center running on your workstation with a reverse forward:

    ssh -R 18000:127.0.0.1:8000 rtx4090      # then --sc http://127.0.0.1:18000

Usage:
    python go2_newton_gait.py --emit-gait trot --speed 0.6 -o gait_trot.json
    python go2_walk_to_map.py --gait-file gait_trot.json --seconds 8 \
        --sc http://127.0.0.1:18000 --capture walk_to_map.png

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

from go2_newton_gait import (
    Bridge,
    COLLECT_CODE,
    CLEANUP_CODE,
    DEFAULT_PORT,
    ROBOT_PATH,
    _ok,
    build_driver_code,
    build_scene_code,
    score_trace,
)

DEFAULT_SC_URL = "http://127.0.0.1:8000"
DEFAULT_TARGET_ID = "isaac_go2_walk"

# Read only the tail of the trace each tick.  The full trace grows for the
# length of the run and shipping all of it back over the bridge every 100 ms
# would make the poll cost grow with time -- which shows up as streaming lag,
# i.e. as fake error in the very metric this script exists to measure.
POLL_CODE = """
import builtins
g = getattr(builtins, "_tritium_gait", None)
if g is None:
    result = {"row": None, "n": 0, "err": "driver not installed"}
else:
    st = g["state"]
    tr = st["trace"]
    result = {"row": tr[-1] if tr else None, "n": len(tr), "err": st["err"]}
"""


def _http_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    """POST (with payload) or GET (without) one JSON document."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}


def _stage_metadata(br: Bridge) -> dict:
    """Ask the live stage for its own units instead of assuming them.

    Hardcoding Z-up/1.0 would work on this kit and silently produce a track
    100x off on a centimetre-authored stage -- exactly the class of quiet
    frame bug the fidelity metric is here to catch, so it would be perverse
    to bake one in on the way.
    """
    code = """
import omni.usd
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()
result = {"meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
          "up_axis": str(UsdGeom.GetStageUpAxis(stage))}
"""
    return _ok("stage_meta", br.execute(code))


def stream_walk(br: Bridge, frame, sc_url: str, target_id: str,
                duration: float, hz: float, verbose: bool = True,
                freeze: bool = False, mid_capture: str | None = None):
    """Drive the stream loop; return (reference, reported) sample lists.

    ``reference`` is ground truth as the client learned it; ``reported`` is
    what ``GET /api/targets`` was serving at the same moment.

    ``freeze`` is the negative control.  It keeps reading and recording true
    poses but posts the *first* one forever, simulating the single most
    likely real failure of this pipeline: an ingest that accepts, stores and
    serves without ever advancing.  A passing run and a frozen run must not
    produce the same verdict -- if they do, the metric is measuring nothing
    and the green result above it is worthless.
    """
    from tritium_lib.geo.path_fidelity import PathSample

    sighting_url = sc_url.rstrip("/") + "/api/sighting"
    targets_url = sc_url.rstrip("/") + "/api/targets"

    reference: list[PathSample] = []
    reported: list[PathSample] = []
    posted = 0
    post_errors = 0
    poll_errors = 0
    last_n = -1
    frozen_pose = None
    captured = False

    t0 = time.time()
    period = 1.0 / hz
    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed > duration:
            break

        try:
            tick = _ok("poll", br.execute(POLL_CODE))
        except Exception as exc:
            poll_errors += 1
            if verbose:
                print(f"[walk] sim poll failed: {exc}", file=sys.stderr)
            time.sleep(period)
            continue

        row = tick.get("row")
        # A row is [t, x, y, z, qx, qy, qz, qw]; Isaac's root transform uses
        # an xyzw quaternion while tritium_lib.geo speaks wxyz.
        if row and len(row) >= 8 and tick.get("n") != last_n:
            last_n = tick.get("n")
            qx, qy, qz, qw = row[4], row[5], row[6], row[7]
            pose = frame.pose_to_local((row[1], row[2], row[3]), (qw, qx, qy, qz))
            stamp = time.time() - t0
            reference.append(PathSample(t=stamp, x=pose.east_m, y=pose.north_m,
                                        heading_deg=pose.heading_deg))
            sent = frozen_pose if frozen_pose is not None else pose
            if freeze and frozen_pose is None:
                frozen_pose = pose
                sent = pose
            try:
                _http_json(sighting_url, {
                    "source": "robot_pose",
                    "origin": "isaac_newton",
                    "target_id": target_id,
                    "name": "Go2 (Newton)",
                    "asset_type": "quadruped",
                    "alliance": "friendly",
                    "position": {"x": sent.east_m, "y": sent.north_m},
                    "heading": sent.heading_deg,
                    "ground_truth": True,
                })
                posted += 1
            except (urllib.error.URLError, OSError) as exc:
                post_errors += 1
                if verbose and post_errors <= 3:
                    print(f"[walk] sighting POST failed: {exc}", file=sys.stderr)

        # Read the map back.  This is deliberately a separate round trip
        # through the public read path -- trusting the POST's own reply would
        # only prove the request was well-formed, not that the operator's map
        # actually serves the body's position.
        try:
            payload = _http_json(targets_url)
            for tgt in payload.get("targets", []):
                if tgt.get("target_id") != target_id:
                    continue
                pos = tgt.get("position") or {}
                reported.append(PathSample(
                    t=time.time() - t0,
                    x=float(pos.get("x", 0.0)),
                    y=float(pos.get("y", 0.0)),
                    heading_deg=(float(tgt["heading"])
                                 if tgt.get("heading") is not None else None),
                ))
                break
        except (urllib.error.URLError, OSError, ValueError) as exc:
            poll_errors += 1
            if verbose and poll_errors <= 3:
                print(f"[walk] targets GET failed: {exc}", file=sys.stderr)

        # Capture INSIDE the scored window.  The end-of-run capture is taken
        # after the driver stops commanding, and a Go2 left holding its last
        # joint targets topples within a second or two -- so an end frame can
        # show a robot on its back for a window that was genuinely upright,
        # and (worse) looks like proof of failure for a run that passed.  The
        # picture has to be of the thing that was measured.
        if mid_capture and not captured and elapsed >= duration * 0.5:
            captured = True
            try:
                shot = br.capture()
                data = shot.get("result", {})
                b64 = (data.get("image_base64") or data.get("image")
                       or data.get("data"))
                if b64:
                    with open(mid_capture, "wb") as fh:
                        fh.write(base64.b64decode(b64))
                    if verbose:
                        print(f"[walk] mid-run frame at t={elapsed:.1f}s "
                              f"-> {mid_capture}")
            except Exception as exc:
                if verbose:
                    print(f"[walk] mid capture failed: {exc}", file=sys.stderr)

        slack = period - (time.time() - now)
        if slack > 0:
            time.sleep(slack)

    stats = {"posted": posted, "post_errors": post_errors,
             "poll_errors": poll_errors}
    return reference, reported, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gait-file", required=True,
                    help="gait table from `go2_newton_gait.py --emit-gait`")
    ap.add_argument("--host", default="127.0.0.1", help="Newton MCP bridge host")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--sc", default=DEFAULT_SC_URL, help="Command Center base URL")
    ap.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--hz", type=float, default=10.0, help="pose stream rate")
    ap.add_argument("--stiffness", type=float, default=60.0)
    ap.add_argument("--damping", type=float, default=4.0)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=0.50,
                    help="max position error still called agreement (m)")
    ap.add_argument("--control", action="store_true",
                    help="negative control: freeze the posted pose at the "
                         "first sample. The body still walks, so a working "
                         "metric MUST report DIVERGED. Exits 0 only if it does.")
    ap.add_argument("--capture", help="write a viewport PNG here when done")
    ap.add_argument("--capture-mid",
                    help="write a viewport PNG from the MIDDLE of the scored "
                         "window -- the frame that actually corresponds to "
                         "the metrics, unlike the end-of-run one")
    ap.add_argument("--report", help="write the full JSON report here")
    args = ap.parse_args()

    from tritium_lib.geo.isaac_frame import IsaacFrame
    from tritium_lib.geo.path_fidelity import compare_paths

    with open(args.gait_file) as fh:
        gait = json.load(fh)

    br = Bridge(args.host, args.port)
    state = br.sim_state().get("result", {}).get("state")
    print(f"[walk] bridge {args.host}:{args.port} -> {state}")

    # Fail fast and loudly if the Command Center is not actually up: every
    # sample would otherwise "stream" into a connection error and the run
    # would end with an empty reported path, which reads like an ingest bug.
    try:
        seen = _http_json(args.sc.rstrip("/") + "/api/targets")
        print(f"[walk] SC {args.sc} -> {seen.get('count', '?')} targets already tracked")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[walk] Command Center unreachable at {args.sc}: {exc}",
              file=sys.stderr)
        return 2

    br.sim_control("stop")
    time.sleep(1.0)
    print(f"[walk] scene: {_ok('scene', br.execute(build_scene_code(args.stiffness, args.damping)))}")
    br.sim_control("play")
    time.sleep(2.0)

    meta = _stage_metadata(br)
    frame = IsaacFrame.from_stage_metadata(meta)
    print(f"[walk] stage frame: {meta}")

    info = _ok("driver", br.execute(build_driver_code(
        gait, args.seconds, args.stiffness, args.damping, args.sample_every)))
    print(f"[walk] driving {info['dof']} DOF, streaming to {args.sc} at {args.hz} Hz")

    if args.capture_mid:
        # Aim once, before the walk, and leave the camera fixed.  A camera
        # that re-frames the robot every capture makes a moving body and a
        # stationary one look identical; a fixed one lets the frame itself
        # show displacement.
        br.camera_look_at(ROBOT_PATH, distance=4.0)
        time.sleep(0.3)

    reference, reported, stats = stream_walk(
        br, frame, args.sc, args.target_id, args.seconds, args.hz,
        freeze=args.control, mid_capture=args.capture_mid)

    collected = _ok("collect", br.execute(COLLECT_CODE))
    if collected.get("err"):
        print(f"[walk] callback error: {collected['err']}", file=sys.stderr)
    _ok("cleanup", br.execute(CLEANUP_CODE))

    # The gait verdict and the map verdict are independent and both matter.
    # A tumbling robot whose tumble is faithfully mirrored on the map is a
    # working pipeline and a broken gait; reporting one number would hide
    # whichever half failed.
    gait_score = score_trace(collected.get("trace", []))
    gait_score["physics_steps"] = collected.get("steps")
    fidelity = compare_paths(reference, reported, tolerance_m=args.tolerance)

    if args.control:
        # In control mode the body must still have walked -- a frozen report
        # of a body that never moved is trivially "correct" and proves
        # nothing about the metric.
        verdict = (
            "CONTROL_DETECTED"
            if gait_score.get("verdict") == "WALKED"
            and fidelity.verdict == "DIVERGED"
            else f"CONTROL_INCONCLUSIVE gait={gait_score.get('verdict')} "
                 f"map={fidelity.verdict}"
        )
    else:
        verdict = (
            "MAP_TRACKS_BODY"
            if gait_score.get("verdict") == "WALKED"
            and fidelity.verdict == "AGREES"
            else f"gait={gait_score.get('verdict')} map={fidelity.verdict}"
        )

    report = {
        "mode": "control" if args.control else "live",
        "stream": stats | {"reference_samples": len(reference),
                           "reported_samples": len(reported),
                           "hz": args.hz, "seconds": args.seconds},
        "gait": gait_score,
        "map_fidelity": fidelity.as_dict(),
        "verdict": verdict,
    }
    print(json.dumps(report, indent=1))
    print(f"[walk] {fidelity.summary()}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[walk] wrote {args.report}")

    if args.capture:
        br.camera_look_at(ROBOT_PATH, distance=4.0)
        time.sleep(0.5)
        shot = br.capture()
        data = shot.get("result", {})
        b64 = data.get("image_base64") or data.get("image") or data.get("data")
        if b64:
            with open(args.capture, "wb") as fh:
                fh.write(base64.b64decode(b64))
            print(f"[walk] wrote {args.capture}")
        else:
            print(f"[walk] no image in capture response: {list(data)[:8]}",
                  file=sys.stderr)

    # Both halves must pass.  A body that walked but whose track never
    # reached the map is the exact failure this script was written to make
    # impossible to miss, so it must exit nonzero.
    return 0 if report["verdict"] in ("MAP_TRACKS_BODY", "CONTROL_DETECTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
