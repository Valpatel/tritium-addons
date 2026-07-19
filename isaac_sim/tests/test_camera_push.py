# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""The camera server DIALS OUT: `--push-to` proves frames reach an operator
that cannot reach the robot.

Every test here runs with NO Isaac, NO GPU and NO tritium-sc — the far end is a
throwaway stdlib HTTP server on loopback, and the frame source is a stub.  That
is the point: the failure this feature exists to fix (the renderer binds to
localhost, the operator has no inbound route) is a TRANSPORT failure, and a
transport is testable without a simulator.

Three behaviours are load-bearing and each has a test that fails without it:
frames actually land; a refusing far end produces BACKOFF rather than a hot
loop; and an outage does not kill the pusher thread — it resumes when the
operator comes back.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_CONN = Path(__file__).resolve().parents[1] / "isaac_sim_addon" / "connectors"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"push_{name}", _CONN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cam = _load("camera_server")


# --------------------------------------------------------------------------- #
# A throwaway far end that can be told to refuse.
# --------------------------------------------------------------------------- #

class FarEnd:
    """A loopback HTTP server standing in for the Command Center.

    ``status`` is mutable mid-flight so a single test can take the operator
    down and bring it back — the outage/recovery case that a mock-only test
    cannot express honestly.
    """

    def __init__(self, status: int = 200):
        self.status = status
        self.posts: list[tuple[str, str, bytes]] = []
        self._lock = threading.Lock()
        far = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                with far._lock:
                    far.posts.append(
                        (self.path, self.headers.get("Content-Type", ""), body)
                    )
                    code = far.status
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def count(self) -> int:
        with self._lock:
            return len(self.posts)

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


class StubFrames:
    """Latest-frame holder with the one method the pusher uses.

    Hands out a DISTINCT blob every call so "the pusher sent 5 frames" cannot
    be satisfied by re-posting the same stale bytes.
    """

    def __init__(self, prefix: bytes = b"\xff\xd8jpeg"):
        self.prefix = prefix
        self.n = 0

    def latest(self, channel: str = "main") -> bytes:
        self.n += 1
        return self.prefix + str(self.n).encode()


def _wait(pred, timeout=5.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def far():
    f = FarEnd()
    yield f
    f.close()


# --------------------------------------------------------------------------- #
# The contract: the connector's mirrored policy IS the lib policy.
# --------------------------------------------------------------------------- #

def test_push_path_matches_the_lib_seam():
    """The route and its percent-encoding must match ``tritium_lib.fleet``.

    Connectors run inside Isaac's python and may never import tritium (see
    test_connectors_do_not_import_tritium), so this path is mirrored — and a
    mirror that drifts posts frames at a 404 while reporting "refused"."""
    lib = pytest.importorskip("tritium_lib.fleet.frame_push")
    for sid in ("isaac_rgb", "isaac_right", "isaac_depth16", "a/b", "x y"):
        assert cam.frame_push_path(sid) == lib.frame_push_path(sid)
    with pytest.raises(ValueError):
        cam.frame_push_path("")


def test_policy_decisions_match_the_lib_policy_frame_for_frame():
    """CONTRACT: drive both policies through the identical script and demand
    identical decisions, reasons and stats.  This is what stops the mirror
    silently becoming a different rate limiter than the tested one."""
    lib = pytest.importorskip("tritium_lib.fleet.frame_push")
    a = cam.FramePushPolicy(target_fps=10.0, base_backoff_s=0.5, max_backoff_s=4.0)
    b = lib.FramePushPolicy(target_fps=10.0, base_backoff_s=0.5, max_backoff_s=4.0)
    # (now, outcome) — outcome applied only when the decision says send.
    script = [
        (0.00, "ok"), (0.01, "ok"), (0.20, "fail"), (0.25, "ok"),
        (0.80, "fail"), (0.85, "ok"), (1.90, "fail"), (3.00, "ok"),
        (3.10, "ok"), (9.00, "ok"),
    ]
    for now, outcome in script:
        da, db = a.offer(now), b.offer(now)
        assert (da.send, da.reason) == (db.send, db.reason), f"diverged at t={now}"
        if da.send:
            if outcome == "ok":
                a.sent(now)
                b.sent(now)
            else:
                a.failed(now)
                b.failed(now)
        assert a.backoff_remaining(now) == pytest.approx(b.backoff_remaining(now))
    fields = ("sent", "failed", "dropped_rate_limited", "dropped_in_flight",
              "dropped_backoff", "consecutive_failures")
    assert {f: getattr(a.stats, f) for f in fields} == \
           {f: getattr(b.stats, f) for f in fields}
    assert a.stats.sent > 0 and a.stats.failed > 0, "script exercised nothing"
    assert a.healthy == b.healthy


def test_default_source_ids_match_the_rig():
    """A pushed channel must land on the SAME source id the rig registered,
    or the operator gets a live feed nobody is looking at."""
    assert cam.DEFAULT_PUSH_SOURCE_IDS["main"] == "isaac_rgb"
    assert cam.DEFAULT_PUSH_SOURCE_IDS["right"] == "isaac_right"
    assert cam.DEFAULT_PUSH_SOURCE_IDS["depth16"] == "isaac_depth16"


def test_a_pusher_that_never_delivered_is_not_healthy():
    """Healthy must mean "frames are landing", not "nothing has gone wrong yet".

    A rig that has never delivered a byte — no far end configured, no frames
    rendered, nobody listening — reports green under the optimistic reading,
    which is precisely how a dark feed goes unnoticed."""
    pol = cam.FramePushPolicy(target_fps=10.0)
    assert not pol.healthy, "never-sent must not read as healthy"
    assert pol.stats.sent == 0 and pol.stats.failed == 0
    pol.offer(0.0)
    pol.sent(0.0)
    assert pol.healthy
    pol.offer(1.0)
    pol.failed(1.0)
    assert not pol.healthy


def test_only_a_2xx_counts_as_delivered(monkeypatch):
    """A response is not a delivery.  urllib raises on 4xx/5xx, so the status
    check exists for the responses it does NOT raise on — a redirect landing on
    some other 3xx page must not be counted as a frame the operator received."""
    import urllib.request as ur

    class _Resp:
        status = 302

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ur, "urlopen", lambda *a, **k: _Resp())
    p = cam.FramePusher(StubFrames(), "main", "http://op:8000", "isaac_rgb")
    assert p._post(b"\xff\xd8x") is False
    _Resp.status = 204
    assert p._post(b"\xff\xd8x") is True


# --------------------------------------------------------------------------- #
# Frames actually land.
# --------------------------------------------------------------------------- #

def test_pusher_posts_jpeg_bytes_to_the_operator(far):
    frames = StubFrames()
    p = cam.FramePusher(frames, "main", far.url, "isaac_rgb", target_fps=50.0)
    p.start()
    try:
        assert _wait(lambda: far.count() >= 3), "no frames reached the far end"
    finally:
        p.stop()
    path, ctype, body = far.posts[0]
    assert path == "/api/camera-feeds/sources/isaac_rgb/frame"
    assert ctype == "image/jpeg"
    assert body.startswith(b"\xff\xd8jpeg")
    # Bodies must be DISTINCT frames, not one blob re-posted.
    bodies = {b for _, _, b in far.posts}
    assert len(bodies) >= 3
    assert p.policy.stats.sent >= 3 and p.policy.healthy


def test_pusher_honours_the_target_fps(far):
    """The decimation is the policy's, and it must actually bite: a 5 fps
    pusher against a source producing frames as fast as it is asked must not
    put dozens of frames on the link in half a second."""
    p = cam.FramePusher(StubFrames(), "main", far.url, "isaac_rgb", target_fps=5.0)
    p.start()
    try:
        time.sleep(0.6)
    finally:
        p.stop()
    assert far.count() <= 5, f"5 fps sent {far.count()} frames in 0.6 s"
    assert far.count() >= 1


def test_pusher_does_not_repost_a_stale_frame(far):
    """A source that has not rendered anything new must not be re-sent — a
    duplicate frame is bandwidth spent making the feed look live."""
    class Frozen:
        blob = b"\xff\xd8frozen"

        def latest(self, channel="main"):
            return self.blob

    p = cam.FramePusher(Frozen(), "main", far.url, "isaac_rgb", target_fps=50.0)
    p.start()
    try:
        assert _wait(lambda: far.count() >= 1)
        time.sleep(0.4)
    finally:
        p.stop()
    assert far.count() == 1, "the same frame was pushed more than once"


# --------------------------------------------------------------------------- #
# A refusing far end backs off instead of hot-looping.
# --------------------------------------------------------------------------- #

def test_far_end_500_drives_backoff_not_a_hot_loop(far):
    far.status = 500
    p = cam.FramePusher(StubFrames(), "main", far.url, "isaac_rgb",
                        target_fps=100.0, base_backoff_s=0.2, max_backoff_s=1.0)
    p.start()
    try:
        assert _wait(lambda: far.count() >= 1)
        time.sleep(0.7)
    finally:
        p.stop()
    # 100 fps unthrottled would be ~70 attempts; 0.2 s doubling gives ~3.
    assert far.count() <= 6, f"hot loop: {far.count()} attempts in 0.7 s"
    st = p.policy.stats
    assert st.failed >= 2 and st.sent == 0
    assert st.consecutive_failures >= 2
    assert not p.policy.healthy, "a pusher refused on every frame is not healthy"


def test_unreachable_operator_is_a_failure_not_a_crash():
    """A dead port must be counted, not raised — and the thread must live."""
    p = cam.FramePusher(StubFrames(), "main", "http://127.0.0.1:1", "isaac_rgb",
                        target_fps=100.0, base_backoff_s=0.05, max_backoff_s=0.2)
    p.start()
    try:
        assert _wait(lambda: p.policy.stats.failed >= 2)
        assert p.is_alive(), "pusher thread died on an unreachable operator"
    finally:
        p.stop()


# --------------------------------------------------------------------------- #
# Outage -> recovery.
# --------------------------------------------------------------------------- #

def test_pusher_survives_an_outage_and_resumes(far):
    far.status = 503
    p = cam.FramePusher(StubFrames(), "main", far.url, "isaac_rgb",
                        target_fps=50.0, base_backoff_s=0.05, max_backoff_s=0.2)
    p.start()
    try:
        assert _wait(lambda: p.policy.stats.failed >= 2), "never tried during outage"
        assert p.policy.stats.sent == 0
        far.status = 200                      # operator comes back
        assert _wait(lambda: p.policy.stats.sent >= 2), "did not resume after outage"
        assert p.is_alive() and p.policy.healthy
        assert p.policy.stats.consecutive_failures == 0
    finally:
        p.stop()


def test_stop_joins_the_thread():
    p = cam.FramePusher(StubFrames(), "main", "http://127.0.0.1:1", "isaac_rgb",
                        target_fps=20.0, base_backoff_s=0.01, max_backoff_s=0.05)
    p.start()
    assert p.daemon, "pusher must be a daemon so the server can exit"
    p.stop(timeout=3.0)
    assert not p.is_alive(), "stop() left the pusher thread running"


# --------------------------------------------------------------------------- #
# The operator can SEE that it is pushing (and being refused).
# --------------------------------------------------------------------------- #

def test_status_reports_push_state_including_refusal(far):
    far.status = 500
    state = cam.CameraState(cam.SyntheticFrameSource(width=32, height=24),
                            meta={"width": 32, "height": 24}, fps=10,
                            encoder=lambda rgb: b"\xff\xd8x", channels=("main",))
    p = cam.FramePusher(state, "main", far.url, "isaac_rgb", target_fps=50.0,
                        base_backoff_s=0.05, max_backoff_s=0.2)
    state.attach_pusher(p)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cam._make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.render_once()
    p.start()
    try:
        assert _wait(lambda: p.policy.stats.failed >= 1)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{httpd.server_address[1]}/status", timeout=3
        ) as r:
            body = json.loads(r.read())
    finally:
        p.stop()
        httpd.shutdown()
        httpd.server_close()
    push = body["push"]
    assert push["url"] == far.url
    assert push["fps"] == pytest.approx(50.0)
    ch = push["channels"]["main"]
    assert ch["source_id"] == "isaac_rgb"
    assert ch["target"] == far.url + "/api/camera-feeds/sources/isaac_rgb/frame"
    assert ch["failed"] >= 1 and ch["sent"] == 0
    assert ch["consecutive_failures"] >= 1
    assert ch["healthy"] is False, "being refused must not read as healthy"
    assert "dropped" in ch


def test_status_has_no_push_block_when_not_pushing():
    """Absence is the honest signal for a pull-only server — an empty push
    block would read as "configured but idle"."""
    state = cam.CameraState(cam.SyntheticFrameSource(width=32, height=24),
                            meta={"width": 32, "height": 24}, fps=10,
                            encoder=lambda rgb: b"\xff\xd8x", channels=("main",))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cam._make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{httpd.server_address[1]}/status", timeout=3
        ) as r:
            body = json.loads(r.read())
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert "push" not in body


# --------------------------------------------------------------------------- #
# CLI wiring.
# --------------------------------------------------------------------------- #

def test_cli_builds_pushers_for_the_requested_channels():
    ns = cam._parse_args([
        "--push-to", "http://operator:8000/", "--push-fps", "4",
        "--push-channel", "main", "--push-channel", "depth16", "--depth16",
    ])
    assert ns.push_to == "http://operator:8000/"
    plans = cam.push_plans(ns, channels=("main", "depth16"))
    assert [(pl.channel, pl.source_id, pl.target_fps) for pl in plans] == [
        ("main", "isaac_rgb", 4.0), ("depth16", "isaac_depth16", 4.0)]
    # Trailing slash must not produce a doubled slash in the route.
    assert plans[0].url() == "http://operator:8000/api/camera-feeds/sources/isaac_rgb/frame"


def test_cli_rejects_pushing_a_channel_that_is_not_served():
    """Better to fail at startup than to run a pusher that will never find a
    frame and report itself merely 'not yet healthy'."""
    ns = cam._parse_args(["--push-to", "http://operator:8000",
                          "--push-channel", "right"])
    with pytest.raises(ValueError, match="right"):
        cam.push_plans(ns, channels=("main",))


def test_cli_source_id_override():
    ns = cam._parse_args(["--push-to", "http://op:8000", "--push-channel", "main",
                          "--push-source-id", "main=rooftop_cam"])
    assert cam.push_plans(ns, channels=("main",))[0].source_id == "rooftop_cam"


def test_push_defaults_are_off():
    ns = cam._parse_args([])
    assert ns.push_to == "" and cam.push_plans(ns, channels=("main",)) == []
    assert ns.push_fps == 10.0
