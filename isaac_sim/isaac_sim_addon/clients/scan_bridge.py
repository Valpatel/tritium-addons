#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LiDAR ``/scan`` -> Command Center ``/api/sighting`` — the missing last mile.

``lidar_server.py`` serves real sweeps (Isaac RTX Lidar or synthetic) and
``tritium-sc`` accepts them at ``POST /api/sighting {"source": "lidar"}``.
Nothing joined the two, so the sensor rig's LiDAR came up healthy and its
returns reached the operator's tactical map never — the gap
``docs/ISAAC-SIM-STATUS.md`` records against capability 9.

This bridge is the join, and it is deliberately thin: every decision worth
testing (is this sweep stale, does it carry returns, has the Command Center
stopped answering) lives in :class:`tritium_lib.fleet.scan_pump.ScanPump`, so
the sim path and a real rover's scanner share one implementation rather than
getting two chances to disagree about what a dead LiDAR looks like.

What is left here is transport: poll ``/scan``, hand the sweep to the pump,
POST what it approves, tell it what happened.

Usage::

    # Against a running lidar_server (:8110) and Command Center (:8000)
    python3 scan_bridge.py --lidar http://localhost:8110 --sc http://localhost:8000

    # One sweep, printed not posted -- inspect before wiring
    python3 scan_bridge.py --once --dry-run

    # Offline selftest, no server and no GPU
    python3 scan_bridge.py --selftest

Both North Star halves: FUN — a body walking a scene leaves obstacle contacts
behind it on the tactical map instead of crossing an empty grid.  PRODUCTION —
this is the same polar-sweep ingest a ground robot's 2D scanner uses, and the
staleness refusal is the field's most common LiDAR failure made visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

DEFAULT_LIDAR_URL = "http://localhost:8110"
DEFAULT_SC_URL = "http://localhost:8000"
DEFAULT_LIDAR_ID = "isaac-lidar-01"
SIGHTING_PATH = "/api/sighting"

#: Type of the injected transport, so every test below runs with no socket.
Fetch = Callable[[str], Mapping[str, Any]]
Post = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


# --------------------------------------------------------------------------- #
# Transport (the only part that touches a socket)
# --------------------------------------------------------------------------- #


def http_get_json(url: str, timeout: float = 5.0) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def http_post_json(url: str, payload: Mapping[str, Any],
                   timeout: float = 10.0) -> Mapping[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


# --------------------------------------------------------------------------- #
# The bridge
# --------------------------------------------------------------------------- #


class ScanBridge:
    """Polls a LiDAR server and forwards approved sweeps to the Command Center.

    Holds no policy of its own — :class:`ScanPump` decides, this moves bytes.
    That split is why the interesting behaviour is testable without a socket
    and why a rover reusing the pump inherits the same refusals for free.
    """

    def __init__(self, lidar_url: str = DEFAULT_LIDAR_URL,
                 sc_url: str = DEFAULT_SC_URL,
                 lidar_id: str = DEFAULT_LIDAR_ID,
                 sensor_x: float = 0.0, sensor_y: float = 0.0,
                 sensor_yaw_deg: float = 0.0,
                 max_failures: int = 3,
                 fetch: Fetch | None = None,
                 post: Post | None = None) -> None:
        from tritium_lib.fleet.scan_pump import ScanPump

        self.lidar_url = lidar_url.rstrip("/")
        self.sc_url = sc_url.rstrip("/")
        self.pump = ScanPump(
            lidar_id=lidar_id, sensor_x=sensor_x, sensor_y=sensor_y,
            sensor_yaw_deg=sensor_yaw_deg, max_failures=max_failures,
        )
        self._fetch = fetch or http_get_json
        self._post = post or http_post_json
        #: Every target id the Command Center has minted from our sweeps.  The
        #: rig's proof that pixels-equivalent data reached the map: a count of
        #: POSTs proves only that SC answered, not that anything landed.
        self.target_ids: set[str] = set()

    # -- one cycle ------------------------------------------------------- #

    def poll_once(self) -> dict[str, Any]:
        """Fetch one sweep, forward it if the pump approves, report what happened.

        A fetch failure is reported, never raised: a LiDAR that drops off mid
        run must not take the bridge down with it, because the rig's other
        sensors are still healthy and the operator still needs them.
        """
        try:
            scan = self._fetch(f"{self.lidar_url}/scan")
        except Exception as exc:
            return {"forwarded": False, "reason": "fetch_failed",
                    "detail": str(exc)}

        decision = self.pump.offer(dict(scan) if scan else None)
        if not decision.forward:
            return {"forwarded": False, "reason": decision.reason}

        try:
            reply = self._post(f"{self.sc_url}{SIGHTING_PATH}",
                               decision.payload or {})
        except Exception as exc:
            self.pump.record_result(False)
            return {"forwarded": False, "reason": "post_failed",
                    "detail": str(exc)}

        ids = list(reply.get("target_ids") or [])
        self.target_ids.update(str(i) for i in ids)
        self.pump.record_result(True)
        return {"forwarded": True, "reason": "forward", "target_ids": ids}

    # -- the loop -------------------------------------------------------- #

    def run(self, hz: float = 2.0, duration: float = 0.0,
            verbose: bool = True,
            sleep: Callable[[float], None] = time.sleep,
            now: Callable[[], float] = time.monotonic) -> dict[str, Any]:
        """Poll until ``duration`` elapses (0 = forever) or the pump trips.

        Stopping on a tripped breaker is deliberate: continuing to poll a
        LiDAR whose sweeps can no longer be delivered burns the sensor's
        bandwidth to produce nothing an operator will ever see.
        """
        period = 1.0 / hz if hz > 0 else 0.0
        deadline = now() + duration if duration > 0 else None

        while True:
            result = self.poll_once()
            if verbose:
                self._print(result)
            if self.pump.tripped:
                if verbose:
                    print("SCAN BRIDGE TRIPPED: Command Center stopped "
                          "accepting sweeps", file=sys.stderr)
                break
            if deadline is not None and now() >= deadline:
                break
            if period:
                sleep(period)

        return self.summary()

    def summary(self) -> dict[str, Any]:
        """What actually reached the map, stated so it cannot be overread."""
        stats = self.pump.stats()
        stats["target_ids"] = sorted(self.target_ids)
        stats["reached_map"] = bool(self.target_ids)
        return stats

    @staticmethod
    def _print(result: Mapping[str, Any]) -> None:
        if result.get("forwarded"):
            ids = result.get("target_ids") or []
            print(f"  + sweep forwarded -> {len(ids)} target(s): "
                  f"{', '.join(map(str, ids)) or '(none clustered)'}")
        else:
            detail = result.get("detail")
            print(f"  . refused: {result['reason']}"
                  + (f" ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# Offline selftest — no lidar server, no Command Center, no GPU
# --------------------------------------------------------------------------- #


def _selftest() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    def sweep(hit_at=(90, 91, 92)):
        r = [30.0] * 360
        for i in hit_at:
            r[i] = 4.0
        return {"angle_min": -3.1416, "angle_increment": 0.01745,
                "range_min": 0.1, "range_max": 30.0, "ranges": r}

    # A real sweep reaches SC and the minted ids are captured.
    posted: list[Mapping[str, Any]] = []

    def post_ok(url, payload):
        posted.append(payload)
        return {"status": "accepted", "source": "lidar",
                "target_ids": ["lidar_isaac-lidar-01_0"]}

    b = ScanBridge(fetch=lambda url: sweep(), post=post_ok)
    res = b.poll_once()
    check("a sweep with returns is forwarded", res["forwarded"] is True)
    check("the posted payload is a lidar sighting",
          posted and posted[0].get("source") == "lidar")
    check("minted target ids are captured", b.target_ids == {"lidar_isaac-lidar-01_0"})
    check("summary reports the map was reached", b.summary()["reached_map"] is True)

    # The same sweep again is refused as stale, and never reaches SC.
    before = len(posted)
    res = b.poll_once()
    check("an unchanged sweep is refused as stale", res["reason"] == "stale")
    check("a stale sweep is never POSTed", len(posted) == before)

    # A dead LiDAR does not take the bridge down.
    def fetch_dead(url):
        raise OSError("connection refused")

    b2 = ScanBridge(fetch=fetch_dead, post=post_ok)
    res = b2.poll_once()
    check("a dead lidar is reported, not raised", res["reason"] == "fetch_failed")

    # A wedged Command Center trips the breaker and stops the loop.
    def post_dead(url, payload):
        raise OSError("connection refused")

    calls = {"n": 0}

    def fetch_moving(url):
        calls["n"] += 1
        i = calls["n"] * 3
        return sweep(hit_at=(i, i + 1, i + 2))

    b3 = ScanBridge(fetch=fetch_moving, post=post_dead, max_failures=3)
    summary = b3.run(hz=0, duration=0, verbose=False)
    check("a wedged Command Center trips the breaker", summary["tripped"] is True)
    check("a tripped run does not claim the map was reached",
          summary["reached_map"] is False)
    check("the breaker stops the loop rather than polling forever",
          calls["n"] <= 4, f"polled {calls['n']} times")

    print(f"\n{len(failures)} failure(s)" if failures else "\nselftest OK")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lidar", default=DEFAULT_LIDAR_URL,
                    help="lidar_server base URL (serves /scan)")
    ap.add_argument("--sc", default=DEFAULT_SC_URL,
                    help="Command Center base URL")
    ap.add_argument("--lidar-id", default=DEFAULT_LIDAR_ID,
                    help="namespaces the derived track ids per sensor")
    ap.add_argument("--sensor-x", type=float, default=0.0)
    ap.add_argument("--sensor-y", type=float, default=0.0)
    ap.add_argument("--sensor-yaw-deg", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=2.0, help="sweep poll rate")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to stream (0 = forever)")
    ap.add_argument("--once", action="store_true", help="one sweep, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sighting payload instead of POSTing it")
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline selftest and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    post: Post | None = None
    if args.dry_run:
        def post(url, payload):
            shown = dict(payload)
            ranges = shown.pop("ranges", [])
            print(f"  DRY RUN -> POST {url}\n    {json.dumps(shown)}"
                  f"\n    ranges: {len(ranges)} beams, "
                  f"min={min(ranges) if ranges else 'n/a'}")
            return {"status": "dry-run", "target_ids": []}

    bridge = ScanBridge(
        lidar_url=args.lidar, sc_url=args.sc, lidar_id=args.lidar_id,
        sensor_x=args.sensor_x, sensor_y=args.sensor_y,
        sensor_yaw_deg=args.sensor_yaw_deg,
        max_failures=args.max_failures, post=post,
    )

    if args.once:
        bridge._print(bridge.poll_once())
        summary = bridge.summary()
    else:
        summary = bridge.run(hz=args.hz, duration=args.duration)

    print(f"\nSCAN BRIDGE {'ONLINE' if summary['reached_map'] else 'DEGRADED'}: "
          f"{summary['forwarded']} forwarded, {summary['accepted']} accepted, "
          f"{summary['refused']} refused {summary['refusals']}, "
          f"{len(summary['target_ids'])} target(s) on the map")
    return 0 if summary["reached_map"] or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
