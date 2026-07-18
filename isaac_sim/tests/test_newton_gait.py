"""Headless tests for the Newton gait driver's pure logic.

No Isaac, no GPU, no network.  The parts under test are the ones that decide
whether a run gets reported as a working gait -- if `score_trace` is wrong,
every claim built on it is wrong, so it gets the most attention here.

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load_driver():
    """Import the example by path -- examples/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "go2_newton_gait", EXAMPLES / "go2_newton_gait.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


driver = _load_driver()


# ---------------------------------------------------------------- score_trace
def _trace(points):
    """[(t, x, y, z)] -> the driver's row format [t, x, y, z, qx, qy, qz]."""
    return [[t, x, y, z, 0.0, 0.0, 0.0] for (t, x, y, z) in points]


def test_walking_forward_reads_moved():
    trace = _trace([(0.0, 0.0, 0.0, 0.40), (2.0, 0.8, 0.0, 0.40)])
    score = driver.score_trace(trace)
    assert score["verdict"] == "MOVED"
    assert score["displacement_m"] == pytest.approx(0.8)
    assert score["forward_dx_m"] == pytest.approx(0.8)
    assert score["mean_speed_mps"] == pytest.approx(0.4)
    assert score["collapsed"] is False


def test_vibrating_in_place_reads_stationary_not_moved():
    """The failure mode a screenshot cannot distinguish from walking."""
    trace = _trace([
        (0.0, 0.0, 0.0, 0.40),
        (1.0, 0.02, -0.01, 0.40),
        (2.0, -0.01, 0.01, 0.40),
    ])
    score = driver.score_trace(trace)
    assert score["verdict"] == "STATIONARY"
    assert score["displacement_m"] < 0.10


def test_collapsed_dog_reads_collapsed_even_if_it_slid_far():
    """A body that fell over must never be scored as a successful gait,
    however much ground it covered while toppling."""
    trace = _trace([(0.0, 0.0, 0.0, 0.40), (2.0, 1.5, 0.0, 0.10)])
    score = driver.score_trace(trace)
    assert score["collapsed"] is True
    assert score["verdict"] == "COLLAPSED"
    assert score["displacement_m"] > 0.10  # it did move -- still not a pass


def test_height_retention_boundary():
    # 0.6 * 0.40 = 0.24 exactly -> not collapsed (strict <)
    assert driver.score_trace(_trace([
        (0.0, 0.0, 0.0, 0.40), (1.0, 0.5, 0.0, 0.24)]))["collapsed"] is False
    # just below the threshold -> collapsed
    assert driver.score_trace(_trace([
        (0.0, 0.0, 0.0, 0.40), (1.0, 0.5, 0.0, 0.239)]))["collapsed"] is True


def test_min_z_uses_the_whole_trace_not_just_the_endpoints():
    """A dog that dips to the floor mid-run and pops back up has not walked."""
    trace = _trace([
        (0.0, 0.0, 0.0, 0.40),
        (1.0, 0.4, 0.0, 0.05),  # face-plant
        (2.0, 0.8, 0.0, 0.40),  # recovered by the final sample
    ])
    score = driver.score_trace(trace)
    assert score["min_z_m"] == pytest.approx(0.05)
    assert score["collapsed"] is True


def test_lateral_drift_counts_toward_displacement():
    trace = _trace([(0.0, 0.0, 0.0, 0.40), (1.0, 3.0, 4.0, 0.40)])
    score = driver.score_trace(trace)
    assert score["displacement_m"] == pytest.approx(5.0)
    assert score["lateral_dy_m"] == pytest.approx(4.0)


def test_empty_and_single_sample_traces_are_not_success():
    for trace in ([], _trace([(0.0, 0.0, 0.0, 0.4)])):
        score = driver.score_trace(trace)
        assert score["verdict"] == "NO_TRACE"


def test_zero_duration_trace_does_not_divide_by_zero():
    score = driver.score_trace(_trace([(1.0, 0.0, 0.0, 0.4), (1.0, 0.5, 0.0, 0.4)]))
    assert score["mean_speed_mps"] > 0  # clamped denominator, no ZeroDivisionError


# ---------------------------------------------------- driver code generation
def test_build_driver_code_substitutes_every_placeholder():
    gait = {"stride_hz": 1.0, "stand": {"FL_hip": 0.0}, "table": [
        {"phase": 0.0, "angles": {"FL_hip": 0.0}}]}
    code = driver.build_driver_code(gait, duration=5.0, stiffness=60.0,
                                    damping=4.0, sample_every=10)
    for placeholder in ("GAIT_JSON", "ARTICULATION_PATH", "STIFFNESS",
                        "DAMPING", "DURATION", "SAMPLE_EVERY", "CALLBACK_EVENT"):
        assert placeholder not in code, f"{placeholder} left unsubstituted"
    assert driver.ARTICULATION_PATH in code
    assert compile(code, "<driver>", "exec")  # it must at least be valid Python


def test_build_driver_code_embeds_the_gait_recoverably():
    gait = {"stride_hz": 0.975, "stand": {"FL_thigh": 0.87},
            "table": [{"phase": 0.5, "angles": {"FL_thigh": 1.0}}]}
    code = driver.build_driver_code(gait, 5.0, 60.0, 4.0, 10)
    # the embedded literal must round-trip back to the same gait
    start = code.index("gait = json.loads(") + len("gait = json.loads(")
    end = code.index(")\n", start)
    assert json.loads(eval(code[start:end])) == gait


def test_driver_never_imports_isaac_at_module_scope():
    """The example must stay importable on a box with no Isaac, so the
    gait-generation half (--emit-gait) works anywhere."""
    for banned in ("isaacsim", "omni", "pxr"):
        assert banned not in sys.modules or banned in ("omni",)
    src = (EXAMPLES / "go2_newton_gait.py").read_text()
    header = src.split("def emit_gait")[0]
    for banned in ("\nimport isaacsim", "\nimport omni", "\nfrom pxr",
                   "\nimport torch"):
        assert banned not in header, f"module-scope {banned.strip()} in driver"


def test_bridge_builds_a_wellformed_http_request(monkeypatch):
    """Guard the hand-rolled protocol: a wrong Content-Length hangs the kit."""
    sent = {}

    class FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, _):
            pass

        def sendall(self, data):
            sent["raw"] = data

        def recv(self, _n):
            # The real bridge is keep-alive: after the response it just goes
            # quiet rather than sending EOF.  Reading to EOF therefore HANGS,
            # which is the bug this test exists to catch -- so model the
            # silence as a timeout instead of returning b"".
            if sent.get("done"):
                raise TimeoutError("bridge is keep-alive; no EOF is coming")
            sent["done"] = True
            body = b'{"status":"success","result":{"return_value":{"ok":1}}}'
            return (b"HTTP/1.1 200 OK\r\nContent-Length: "
                    + str(len(body)).encode() + b"\r\n\r\n" + body)

    monkeypatch.setattr(driver.socket, "create_connection",
                        lambda *a, **k: FakeSock())
    out = driver.Bridge(port=9999).execute("result = 1")
    assert out["status"] == "success"

    raw = sent["raw"].decode()
    head, _, body = raw.partition("\r\n\r\n")
    assert head.startswith("POST /execute HTTP/1.1")
    declared = int([ln.split(":")[1] for ln in head.split("\r\n")
                    if ln.lower().startswith("content-length")][0])
    assert declared == len(body.encode()), "Content-Length must match the body"
    assert json.loads(body)["code"] == "result = 1"
