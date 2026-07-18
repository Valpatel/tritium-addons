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
  opener, and a non-2xx all count a failure, log once, and never escape;
* the disabled path posts nothing at all;
* the module stays importable with no isaacsim and no tritium.
"""

import json
import socket
import sys

import pytest

from tritium_lib.geo.hitscan import Muzzle, ShotResult
from tritium_lib.tracking.engagement import ShotEvent

from isaac_sim_addon.clients.fire_bridge import (
    DEFAULT_SPECS,
    ENGAGEMENT_PATH,
    FireBridge,
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
    False, counts one failure, and lets no exception escape.  This is the
    common case -- a live sim run with no SC anywhere."""
    poster = ShotPoster(f"http://127.0.0.1:{_closed_port()}", "isaac_go2",
                        timeout_s=0.5)
    assert poster.post(_hit_shot()) is False
    assert poster.failures == 1 and poster.posted == 0


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


def test_non_2xx_counts_as_failure():
    """A 409 (dry trigger: SC magazine empty) and a 500 both count -- the
    trial's own verdict is graded locally and must not depend on them."""
    poster = _poster(opener=lambda u, b: 409)
    assert poster.post(_hit_shot()) is False
    assert poster.failures == 1


def test_a_success_resets_the_consecutive_counter():
    answers = iter([500, 200])
    poster = _poster(opener=lambda u, b: next(answers))
    assert poster.post(_hit_shot()) is False
    assert poster.post(_hit_shot()) is True
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
