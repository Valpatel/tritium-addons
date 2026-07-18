# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""The last mile of the last mile: a resolved shot reaches the operator.

fire_bridge grades REAL rounds in the live Newton sim; SC's
``POST /api/engagement/shot`` turns such a record into a tracer, an impact
dot, a kill-feed line, and narration.  These tests pin the wire between them
from the addon side:

* the payload is EXACTLY what the SC router accepts -- proven by feeding it
  through ``ShotEvent.from_payload``, the very call the router makes, and by
  pinning the field names so schema drift breaks loudly here;
* an absent SC degrades gracefully -- a real connection-refused, a raising
  opener, and an unexpected non-2xx all count a failure, log once, and never
  escape;
* a **409 is an answer, not a failure** -- SC refusing a round (empty
  magazine, mid-reload) is counted separately, never trips the circuit
  breaker, suppresses the tracer in favour of a dry-trigger marker, keeps
  the refused arm out of trial grading, and a ``reloading`` countdown holds
  the next trigger pull for a BOUNDED wait;
* the disabled path posts nothing at all;
* the module stays importable with no isaacsim and no tritium.
"""

import http.server
import json
import socket
import sys
import threading

import pytest

from tritium_lib.geo.hitscan import Muzzle, ShotResult
from tritium_lib.tracking.engagement import ShotEvent

from isaac_sim_addon.clients.fire_bridge import (
    DEFAULT_SPECS,
    ENGAGEMENT_PATH,
    FireBridge,
    PostResult,
    ShotPoster,
    default_shooter_id,
    poster_from_args,
    run_trial,
)
from .test_fire_bridge import FakeStage, _target


def _hit_shot() -> ShotResult:
    return ShotResult(
        hit=True,
        muzzle=Muzzle(east_m=1.0, north_m=2.0, up_m=0.95,
                      heading_deg=30.0, elevation_deg=-2.0),
        max_range_m=60.0,
        target_id="dummy_near",
        range_m=4.2,
        impact_east_m=3.1,
        impact_north_m=5.6,
        impact_up_m=0.8,
    )


def _miss_shot() -> ShotResult:
    return ShotResult(
        hit=False,
        muzzle=Muzzle(east_m=0.0, north_m=0.0, up_m=0.75,
                      heading_deg=90.0, elevation_deg=0.0),
        max_range_m=60.0,
        miss_distance_m=2.7,
    )


def _poster(**kw) -> ShotPoster:
    kw.setdefault("shooter_type", "robot_dog")
    return ShotPoster("http://sc.test:8000", "isaac_go2", **kw)


# --- the payload IS the SC schema ----------------------------------------


def test_payload_field_names_are_pinned_to_the_sc_router_schema():
    """The exact shape ``/api/engagement/shot`` accepts: ShotResult.to_dict()
    plus shooter_id / shooter_type / timestamp.  A renamed or dropped field
    on either side must break HERE, not silently on the operator's map."""
    payload = _poster(opener=lambda u, b: 200).build_payload(_hit_shot())

    assert set(payload) == {
        "hit", "target_id", "range_m", "impact", "miss_distance_m",
        "max_range_m", "muzzle", "aim",
        "shooter_id", "shooter_type", "timestamp",
    }
    assert set(payload["muzzle"]) == {
        "east_m", "north_m", "up_m", "heading_deg", "elevation_deg",
    }
    assert payload["shooter_id"] == "isaac_go2"
    assert payload["shooter_type"] == "robot_dog"
    assert payload["hit"] is True
    assert payload["impact"] == [3.1, 5.6, 0.8]
    assert len(payload["aim"]) == 3
    assert isinstance(payload["timestamp"], float)


def test_payload_survives_the_ingest_the_sc_router_actually_runs():
    """SC's route body handler is ``ShotEvent.from_payload(body)`` -- so the
    strongest drift pin available to the addon is to run the SAME call on a
    JSON round trip of what we send."""
    payload = json.loads(json.dumps(
        _poster(opener=lambda u, b: 200).build_payload(_hit_shot())))

    event = ShotEvent.from_payload(payload)
    assert event.shooter_id == "isaac_go2"
    assert event.hit is True
    assert event.target_id == "dummy_near"
    assert event.origin == pytest.approx((1.0, 2.0, 0.95), abs=0.05)
    assert event.terminus == pytest.approx((3.1, 5.6, 0.8))
    assert event.range_m == 4.2
    assert event.timestamp == payload["timestamp"]


def test_a_miss_payload_is_accepted_and_draws_to_the_range_gate():
    """A miss is still a drawable record: no impact in the payload, and the
    ingest resolves the terminus at the range gate along the aim."""
    payload = json.loads(json.dumps(
        _poster(opener=lambda u, b: 200).build_payload(_miss_shot())))
    assert payload["impact"] is None

    event = ShotEvent.from_payload(payload)
    assert event.hit is False
    assert event.miss_distance_m == 2.7
    # Heading 90 deg, level: 60 m due east of the muzzle.
    assert event.terminus == pytest.approx((60.0, 0.0, 0.75), abs=1e-6)


def test_poster_url_is_the_engagement_route():
    poster = _poster(opener=lambda u, b: 200)
    assert poster.url == "http://sc.test:8000" + ENGAGEMENT_PATH
    # A trailing slash on the base must not double up.
    assert ShotPoster("http://x/", "s").url == "http://x" + ENGAGEMENT_PATH


# --- graceful when SC is absent ------------------------------------------


def _closed_port() -> int:
    """A port nothing is listening on: bind, read it back, close it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_connection_refused_is_swallowed_and_counted():
    """The REAL urllib path against a port with no listener: post() returns
    a falsy non-refusal, counts one failure, and lets no exception escape.
    This is the common case -- a live sim run with no SC anywhere."""
    poster = ShotPoster(f"http://127.0.0.1:{_closed_port()}", "isaac_go2",
                        timeout_s=0.5)
    outcome = poster.post(_hit_shot())
    assert not outcome
    assert outcome.refused is False       # unreachable, NOT a refusal
    assert poster.failures == 1 and poster.posted == 0 and poster.refused == 0


def test_failure_is_logged_once_not_per_shot(capsys):
    calls = []

    def refuse(url, body):
        calls.append(url)
        raise ConnectionRefusedError("no SC today")

    poster = _poster(opener=refuse)
    poster.post(_hit_shot())
    poster.post(_miss_shot())

    err = capsys.readouterr().err
    assert err.count("SC engagement POST failed") == 1
    assert poster.failures == 2


def test_circuit_opens_after_consecutive_failures():
    """A black-holed remote URL must not tax every shot with a timeout: after
    MAX_CONSECUTIVE_FAILURES the poster stops attempting for the run."""
    calls = []

    def refuse(url, body):
        calls.append(1)
        raise TimeoutError("black hole")

    poster = _poster(opener=refuse)
    for _ in range(10):
        poster.post(_hit_shot())

    assert len(calls) == ShotPoster.MAX_CONSECUTIVE_FAILURES
    assert poster.failures == ShotPoster.MAX_CONSECUTIVE_FAILURES


def test_unexpected_non_2xx_counts_as_failure():
    """A 500 is a wire that answered wrongly: it counts a failure and feeds
    the breaker, exactly as before the 409 carve-out."""
    poster = _poster(opener=lambda u, b: 500)
    assert not poster.post(_hit_shot())
    assert poster.failures == 1 and poster.refused == 0
    assert poster._consecutive == 1


def test_a_success_resets_the_consecutive_counter():
    answers = iter([500, 200])
    poster = _poster(opener=lambda u, b: next(answers))
    assert not poster.post(_hit_shot())
    assert poster.post(_hit_shot())
    assert poster.posted == 1 and poster.failures == 1
    assert poster._consecutive == 0


def test_a_failing_poster_never_fails_the_trial():
    """The whole graceful contract, end to end: a full run_trial with every
    POST refused still completes and still grades."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])

    def refuse(url, body):
        raise ConnectionRefusedError("no SC")

    poster = _poster(opener=refuse)
    report = run_trial(FireBridge(transport=stage, poster=poster), DEFAULT_SPECS)

    assert report["verdict"]["pass"] is True
    assert poster.failures == 2 and poster.posted == 0


# --- SC refuses the round: a 409 is an answer, not a failure ---------------


def _refusal_body(reason: str = "out_of_ammo", remaining: float = 0.0) -> bytes:
    """The EXACT body SC's engagement router returns with its 409."""
    return json.dumps({
        "status": "rejected",
        "reason": reason,
        "ammo": {
            "tracked": True,
            "source": "engagement",
            "weapon_status": {
                "device_id": "isaac_go2", "weapon_id": "blaster",
                "ammo": 0, "max_ammo": 12,
                "reloading": reason == "reloading",
                "reload_remaining_s": remaining,
                "pan_deg": 0.0, "tilt_deg": 0.0,
            },
        },
    }).encode("utf-8")


def _fake_time():
    """(clock, sleeper, sleeps): a clock the sleeper advances, no real wait."""
    state = {"now": 1000.0}
    sleeps: list[float] = []

    def clock() -> float:
        return state["now"]

    def sleeper(s: float) -> None:
        sleeps.append(s)
        state["now"] += s

    return clock, sleeper, sleeps


def test_409_is_a_refusal_counted_apart_from_failures():
    """SC said NO, and that is a different fact from SC being unreachable:
    it lands in ``refused``, not ``failures``, with SC's own reason."""
    poster = _poster(opener=lambda u, b: (409, _refusal_body("out_of_ammo")))
    outcome = poster.post(_hit_shot())

    assert isinstance(outcome, PostResult)
    assert not outcome                       # the operator did NOT get a round
    assert outcome.refused is True
    assert outcome.reason == "out_of_ammo"
    assert poster.refused == 1
    assert poster.failures == 0 and poster.posted == 0


def test_refusals_never_open_the_circuit_breaker():
    """The breaker exists for endpoints that cannot be REACHED.  An endpoint
    that answers 409 every time is healthy and must keep being asked --
    the magazine may refill."""
    calls = []

    def dry(url, body):
        calls.append(1)
        return (409, _refusal_body("out_of_ammo"))

    poster = _poster(opener=dry)
    for _ in range(10):
        poster.post(_hit_shot())

    assert len(calls) == 10                  # every pull attempted
    assert poster.refused == 10 and poster.failures == 0


def test_a_refusal_resets_the_transport_failure_streak():
    """Two timeouts, then a 409, then two more timeouts: the 409 proved the
    wire works, so the streak restarts and the breaker (threshold 3) stays
    closed -- the final 200 must still be attempted and accepted."""
    answers = iter([
        TimeoutError("slow"), TimeoutError("slow"),
        (409, _refusal_body()),
        TimeoutError("slow"), TimeoutError("slow"),
        (200, b""),
    ])

    def opener(url, body):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    poster = _poster(opener=opener)
    for _ in range(6):
        poster.post(_hit_shot())

    assert poster.posted == 1
    assert poster.refused == 1 and poster.failures == 4


def test_legacy_bare_status_opener_still_refuses_on_409():
    """An opener that returns only the status (no body) keeps working: the
    refusal is certain from the code alone, the reason degrades to
    ``unknown``."""
    poster = _poster(opener=lambda u, b: 409)
    outcome = poster.post(_hit_shot())
    assert outcome.refused is True and outcome.reason == "unknown"
    assert poster.refused == 1 and poster.failures == 0


def test_malformed_409_body_degrades_gracefully():
    """A 409 whose body will not parse is still a refusal -- reason unknown,
    no countdown, no exception, breaker untouched."""
    for garbage in (b"", b"not json", b"[]", b'{"ammo": 7}'):
        poster = _poster(opener=lambda u, b, g=garbage: (409, g))
        outcome = poster.post(_hit_shot())
        assert outcome.refused is True
        assert outcome.reason in ("unknown",) or isinstance(outcome.reason, str)
        assert outcome.reload_remaining_s == 0.0
        assert poster.refused == 1 and poster.failures == 0
        assert poster.reload_wait_s() == 0.0


def test_an_opener_returning_a_str_body_still_parses_the_refusal():
    """The seam is tolerant: (409, str) is as good as (409, bytes)."""
    poster = _poster(
        opener=lambda u, b: (409, _refusal_body("reloading", 1.5).decode()))
    outcome = poster.post(_hit_shot())
    assert outcome.refused is True and outcome.reason == "reloading"
    assert outcome.reload_remaining_s == pytest.approx(1.5)


def test_first_refusal_is_logged_once(capsys):
    poster = _poster(opener=lambda u, b: (409, _refusal_body()))
    poster.post(_hit_shot())
    poster.post(_hit_shot())
    err = capsys.readouterr().err
    assert err.count("SC refused the round") == 1
    assert "out_of_ammo" in err


def test_real_http_409_is_a_refusal_not_a_transport_failure():
    """THE original defect lived here: urllib raises HTTPError on a 409, and
    the old poster's blanket except classified it as a transport failure and
    fed the breaker.  Against a REAL server answering 409, the real opener
    must now hand back the status and body as an answer."""

    class Refuse(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = _refusal_body("reloading", remaining=2.0)
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep pytest output clean
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Refuse)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        clock, sleeper, _ = _fake_time()
        poster = ShotPoster(
            f"http://127.0.0.1:{server.server_address[1]}", "isaac_go2",
            timeout_s=2.0, clock=clock, sleeper=sleeper,
        )
        outcome = poster.post(_hit_shot())
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert outcome.refused is True
    assert outcome.reason == "reloading"
    assert outcome.reload_remaining_s == pytest.approx(2.0)
    assert poster.refused == 1 and poster.failures == 0
    assert poster.reload_wait_s() == pytest.approx(2.0)


def test_stats_report_refused_and_failed_separately():
    answers = iter([(200, b""), (409, _refusal_body()),
                    ConnectionRefusedError("gone")])

    def opener(url, body):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    poster = _poster(opener=opener)
    for _ in range(3):
        poster.post(_hit_shot())
    stats = poster.stats()
    assert stats["posted"] == 1
    assert stats["refused"] == 1
    assert stats["failures"] == 1


# --- reload backoff: bounded, honest, never a hang -------------------------


def test_reloading_409_arms_a_wait_for_the_reported_countdown():
    clock, sleeper, sleeps = _fake_time()
    poster = _poster(opener=lambda u, b: (409, _refusal_body("reloading", 2.5)),
                     clock=clock, sleeper=sleeper)
    outcome = poster.post(_hit_shot())

    assert outcome.reason == "reloading"
    assert outcome.reload_remaining_s == pytest.approx(2.5)
    assert poster.reload_wait_s() == pytest.approx(2.5)

    waited = poster.wait_for_reload()
    assert waited == pytest.approx(2.5)
    assert sleeps == [pytest.approx(2.5)]
    # The window elapsed with the wait: a second call costs nothing.
    assert poster.wait_for_reload() == 0.0
    assert len(sleeps) == 1


def test_reload_wait_is_capped_never_unbounded():
    """A confused server reporting a 999 s reload can cost one bounded pause
    and no more."""
    clock, sleeper, sleeps = _fake_time()
    poster = _poster(opener=lambda u, b: (409, _refusal_body("reloading", 999.0)),
                     clock=clock, sleeper=sleeper)
    poster.post(_hit_shot())

    waited = poster.wait_for_reload()
    assert waited == ShotPoster.MAX_RELOAD_WAIT_S
    assert sleeps == [ShotPoster.MAX_RELOAD_WAIT_S]


def test_out_of_ammo_sets_no_wait():
    """No countdown, no backoff: an empty magazine with no reload underway
    is refused instantly on each pull, and each pull is one cheap answer."""
    clock, sleeper, sleeps = _fake_time()
    poster = _poster(opener=lambda u, b: (409, _refusal_body("out_of_ammo")),
                     clock=clock, sleeper=sleeper)
    poster.post(_hit_shot())
    assert poster.reload_wait_s() == 0.0
    assert poster.wait_for_reload() == 0.0
    assert sleeps == []


def test_fire_holds_the_trigger_through_the_reload_window():
    """Burst across a reload: shot 1 draws a 409/reloading, shot 2 waits the
    reported window BEFORE reading the stage, then fires and is accepted."""
    clock, sleeper, sleeps = _fake_time()
    answers = iter([(409, _refusal_body("reloading", 3.0)), (200, b"")])
    poster = _poster(opener=lambda u, b: next(answers),
                     clock=clock, sleeper=sleeper)
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("t", 0.0, 4.0, 0.5)])
    bridge = FireBridge(transport=stage, poster=poster)

    from tritium_lib.geo.camera_mount import CameraMount
    mount = CameraMount(forward_m=0.25, left_m=0.0, up_m=0.55)

    bridge.fire(mount, draw=False)
    assert bridge.last_post.refused is True
    assert sleeps == []                       # no window armed before shot 1

    bridge.fire(mount, draw=False)
    assert sleeps == [pytest.approx(3.0)]     # held for the reload, bounded
    assert bridge.last_post.accepted is True
    assert poster.posted == 1 and poster.refused == 1 and poster.failures == 0


# --- a refused round is not drawn and not graded ---------------------------


def test_a_refused_round_draws_a_dry_trigger_marker_not_a_tracer():
    """SC refused, so no round may appear to travel: the draw snippet is the
    magenta muzzle marker -- a point, no line downrange, and visually
    distinct from both the HIT green and the MISS yellow."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("t", 0.0, 4.0, 0.5)])
    poster = _poster(opener=lambda u, b: (409, _refusal_body("out_of_ammo")))
    bridge = FireBridge(transport=stage, poster=poster)

    from tritium_lib.geo.hitscan import SphereTarget
    from tritium_lib.geo.isaac_frame import LocalPose
    from isaac_sim_addon.clients.fire_bridge import aim_at
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.4, heading_deg=0.0)
    shot = bridge.fire(aim_at(body, SphereTarget("t", 0.0, 4.0, 0.5, 0.4)))

    assert bridge.last_post.refused is True
    draw_code = stage.calls[-1][1]
    assert "dry trigger" in draw_code
    assert "draw_lines" not in draw_code                  # nothing downrange
    assert "(1.0, 0.16, 0.43, 1.0)" in draw_code          # magenta, not
    assert "(0.02, 1.0, 0.63, 1.0)" not in draw_code      # ...the HIT green
    assert "(1.0, 0.93, 0.04, 1.0)" not in draw_code      # ...or MISS yellow
    # The local resolution still returns for the audit trail.
    assert shot.hit is True and shot.target_id == "t"


def test_an_accepted_round_draws_the_tracer_exactly_as_before():
    """The 200 path is byte-identical: same tracer snippet, same colours."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("t", 0.0, 4.0, 0.5)])
    poster = _poster(opener=lambda u, b: 200)
    bridge = FireBridge(transport=stage, poster=poster)

    from tritium_lib.geo.camera_mount import CameraMount
    bridge.fire(CameraMount(forward_m=0.25, left_m=0.0, up_m=0.55))

    draw_code = stage.calls[-1][1]
    assert "draw_lines" in draw_code
    assert "dry trigger" not in draw_code
    assert bridge.last_post.accepted is True


def test_unreachable_sc_still_draws_and_fires_on_local_authority():
    """Refused and unreachable are DIFFERENT: with SC gone the round fires,
    draws its normal tracer, and only the audience is lost."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("t", 0.0, 4.0, 0.5)])

    def gone(url, body):
        raise ConnectionRefusedError("no SC")

    bridge = FireBridge(transport=stage, poster=_poster(opener=gone))

    from tritium_lib.geo.hitscan import SphereTarget
    from tritium_lib.geo.isaac_frame import LocalPose
    from isaac_sim_addon.clients.fire_bridge import aim_at
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.4, heading_deg=0.0)
    shot = bridge.fire(aim_at(body, SphereTarget("t", 0.0, 4.0, 0.5, 0.4)))

    assert shot.hit is True
    assert bridge.last_post.refused is False
    draw_code = stage.calls[-1][1]
    assert "draw_lines" in draw_code and "dry trigger" not in draw_code


def test_run_trial_with_a_dry_magazine_records_refusals_and_cannot_pass():
    """Both arms refused: the report names them with SC's reason, keeps the
    local resolutions as audit records, counts them apart from failures,
    and the verdict is an explicit fail -- no round flew, nothing was
    demonstrated."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])
    poster = _poster(opener=lambda u, b: (409, _refusal_body("out_of_ammo")))
    report = run_trial(FireBridge(transport=stage, poster=poster), DEFAULT_SPECS)

    assert report["refused"] == {"aimed": "out_of_ammo",
                                 "control": "out_of_ammo"}
    assert report["verdict"] == {
        "pass": False,
        "reason": "sc_refused_round",
        "rounds_refused": ["aimed", "control"],
    }
    # Audit trail survives; counters keep the two states apart.
    assert report["aimed"]["target_id"] == "dummy_near"
    assert poster.refused == 2 and poster.failures == 0 and poster.posted == 0


def test_run_trial_with_only_the_control_arm_refused_still_fails():
    """The magazine ran dry BETWEEN the arms: the aimed round is real, but
    half a trial demonstrates nothing -- fail, and say which arm never
    fired."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])
    answers = iter([(200, b""), (409, _refusal_body("out_of_ammo"))])
    poster = _poster(opener=lambda u, b: next(answers))
    report = run_trial(FireBridge(transport=stage, poster=poster), DEFAULT_SPECS)

    assert report["refused"] == {"control": "out_of_ammo"}
    assert report["verdict"]["pass"] is False
    assert report["verdict"]["rounds_refused"] == ["control"]
    assert poster.posted == 1 and poster.refused == 1


def test_a_transport_failing_trial_still_grades_with_no_refused_section():
    """Pin the boundary from the other side: unreachable SC does not put a
    ``refused`` section in the report, and grading runs exactly as before."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])

    def gone(url, body):
        raise ConnectionRefusedError("no SC")

    report = run_trial(FireBridge(transport=stage, poster=_poster(opener=gone)),
                       DEFAULT_SPECS)
    assert "refused" not in report
    assert report["verdict"]["pass"] is True


# --- every resolved round is reported ------------------------------------


def test_fire_posts_each_resolved_round_with_the_shooter_id():
    """One trial = two rounds (aimed + control), BOTH reported: the operator
    sees the hit and the honest near-miss, same as the lib draws them."""
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])
    sent = []

    def record(url, body):
        sent.append(json.loads(body))
        return 200

    poster = _poster(opener=record)
    report = run_trial(FireBridge(transport=stage, poster=poster), DEFAULT_SPECS)

    assert report["verdict"]["pass"] is True
    assert poster.posted == 2 and poster.failures == 0
    hits = [p["hit"] for p in sent]
    assert hits == [True, False]      # aimed arm, then the control arm
    assert all(p["shooter_id"] == "isaac_go2" for p in sent)
    # What we told SC is what we graded: the payloads carry the same verdict
    # fields the local report holds.
    assert sent[0]["target_id"] == report["aimed"]["target_id"]
    assert sent[0]["range_m"] == report["aimed"]["range_m"]


def test_disabled_path_posts_nothing(monkeypatch):
    """poster=None must never touch the network -- urlopen is booby-trapped
    to prove no code path reaches it."""
    import urllib.request as _ur

    def boom(*a, **kw):  # pragma: no cover - failing is the test
        raise AssertionError("disabled path performed an HTTP request")

    monkeypatch.setattr(_ur, "urlopen", boom)
    stage = FakeStage(body=(0.0, 0.0, 0.4), heading=0.0,
                      targets=[_target("dummy_near", 0.0, 4.0, 0.5)])
    report = run_trial(FireBridge(transport=stage), DEFAULT_SPECS)
    assert report["verdict"]["pass"] is True
    assert "sc" not in report


# --- the CLI seam ---------------------------------------------------------


def _args(**over):
    import argparse

    ns = argparse.Namespace(
        sc_url="http://localhost:8000", no_sc=False, shooter_id="",
        shooter_type="robot_dog", body_prim="/World/Tritium/go2/base",
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_posting_is_on_by_default_with_a_derived_shooter_id():
    poster = poster_from_args(_args())
    assert poster is not None
    assert poster.url == "http://localhost:8000" + ENGAGEMENT_PATH
    assert poster.shooter_id == "isaac_go2"
    assert poster.shooter_type == "robot_dog"


def test_no_sc_flag_and_empty_url_both_disable_posting():
    assert poster_from_args(_args(no_sc=True)) is None
    assert poster_from_args(_args(sc_url="")) is None


def test_explicit_shooter_id_wins_over_the_derived_one():
    assert poster_from_args(_args(shooter_id="go2_alpha")).shooter_id == "go2_alpha"


def test_default_shooter_id_drops_the_scene_scaffolding():
    assert default_shooter_id("/World/Tritium/go2/base") == "isaac_go2"
    assert default_shooter_id("/World/spot/base") == "isaac_spot"
    assert default_shooter_id("") == "isaac_body"


# --- hygiene: no isaacsim, no tritium, stdlib HTTP only -------------------


def test_module_imports_with_no_isaacsim_and_no_tritium():
    """fire_bridge (and its poster) must load where CI loads it: a box with
    no GPU, no isaacsim, and no tritium package."""
    assert "isaac_sim_addon.clients.fire_bridge" in sys.modules
    assert "isaacsim" not in sys.modules
    assert "tritium" not in sys.modules       # tritium_lib is the ONLY seam


def test_post_path_is_stdlib_urllib_not_a_client_dependency():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "clients"
           / "fire_bridge.py").read_text()
    assert "urllib.request" in src
    for forbidden in ("import httpx", "import requests", "import aiohttp"):
        assert forbidden not in src, f"poster must stay stdlib: {forbidden}"
