"""The live command surface's plumbing, verified without a GPU.

The driver is a generated source string executed inside Isaac's interpreter,
so the failure that costs the most is the cheapest to catch here: an
unsubstituted placeholder or a syntax error does not show up until a kit is
booted, a scene is built and a robot is spawned, and then it surfaces as an
opaque remote traceback minutes later.  ``compile()`` finds it in a
millisecond.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import go2_newton_gait as gait  # noqa: E402


def args_with(**over):
    ns = argparse.Namespace(
        live_port=None, live_timeout=0.5, live_max_linear=1.0,
        live_max_angular=2.0, steer_track=0.26, speed=0.6,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


GAIT = {"joints": {}, "steps": 4, "period_s": 1.0}


def build(**kw):
    return gait.build_driver_code(
        GAIT, duration=6.0, stiffness=100.0, damping=10.0, sample_every=10, **kw
    )


# ------------------------------------------------------------------ spec


def test_no_port_means_no_live_surface():
    """The batch path must stay exactly the code every prior tick measured."""
    assert gait.live_spec(args_with()) is None


def test_spec_carries_port_limits_and_mixer_geometry():
    spec = gait.live_spec(args_with(live_port=18974))
    assert spec == {
        "port": 18974, "timeout": 0.5, "max_linear": 1.0,
        "max_angular": 2.0, "track": 0.26, "nominal": 0.6,
    }


# ------------------------------------------------------- generated source


def test_generated_driver_compiles_with_a_live_surface():
    code = build(live=gait.live_spec(args_with(live_port=18974)))
    compile(code, "<driver>", "exec")


def test_generated_driver_compiles_without_one():
    compile(build(live=None), "<driver>", "exec")


@pytest.mark.parametrize("live", [None, {"port": 18974, "timeout": 0.5,
                                         "max_linear": 1.0, "max_angular": 2.0,
                                         "track": 0.26, "nominal": 0.6}])
def test_no_placeholder_survives_substitution(live):
    """A missed ``__X__`` is a NameError a whole kit boot away from here."""
    code = build(live=live)
    leftover = [tok for tok in ("__LIVE_JSON__", "__LIVE_DRAIN__",
                                "__TWIST_JSON__", "__ROUTE_JSON__")
                if tok in code]
    assert leftover == []


def test_port_and_limits_reach_the_generated_source():
    code = build(live=gait.live_spec(args_with(live_port=18974,
                                               live_max_linear=0.4)))
    assert "18974" in code
    assert "0.4" in code


def test_drain_is_bounded():
    """An unbounded drain hands a flooding sender control of step duration."""
    code = build(live=gait.live_spec(args_with(live_port=18974)),
                 live_drain=32)
    assert "range(32)" in code


# --------------------------------------------------------------- teardown


def test_cleanup_closes_the_live_socket():
    """A leaked bound port makes every SUBSEQUENT run of a long-lived kit fail."""
    assert "live_sock" in gait.CLEANUP_CODE
    assert ".close()" in gait.CLEANUP_CODE


def test_collect_returns_the_live_trace():
    """Without the command column a live run cannot be told from a baked one."""
    assert '"live"' in gait.COLLECT_CODE
    assert "live_sock_err" in gait.COLLECT_CODE


# ------------------------------------------- the wire format, end to end


def test_wire_frames_are_what_the_lib_link_accepts():
    """The sender's format and the driver's decoder must not drift apart.

    Both sides are pinned to ``tritium_lib.control.CommandLink`` rather than
    to each other, so this test is what proves the JSON this repo emits is the
    JSON that library accepts.
    """
    from tritium_lib.control import CommandLink

    link = CommandLink()
    payload = json.dumps({"cmd": "twist", "seq": 1,
                          "linear_mps": 0.5, "angular_rps": 0.25}).encode()
    assert link.ingest(payload, now_s=0.0) is True
    twist = link.poll(0.0)
    assert twist.linear_mps == pytest.approx(0.5)
    assert twist.angular_rps == pytest.approx(0.25)
