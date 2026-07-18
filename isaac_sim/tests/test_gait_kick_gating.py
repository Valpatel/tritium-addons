"""A kick RECORD is not a kick: the driver must admit only landed pushes.

`score_disturbance` decides whether a trial counts as evidence at all, so a
bug here silently rewrites every recovery rate this lane publishes.  The
specific failure it guards is the one the first live 3 N-s A/B produced: a
push recorded, the solver's force accepted, and the body's velocity change
landing almost entirely on an axis nobody pushed on -- because the body was
falling.  That trial's 178-degree tumble was charged to the controller.

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
GO2_MASS_KG = 15.016999691724777  # read off the live stage, not assumed


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "go2_newton_gait", EXAMPLES / "go2_newton_gait.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """Just the fields score_disturbance reads."""

    disturb_at = 2.0
    settle_below = 10.0
    settle_hold = 1.0


def _collected(measured_dv, *, tilts):
    """A trace plus one kick record, shaped as the step callback returns it."""
    trace = []
    for t, tilt in tilts:
        # Quaternion for a roll of `tilt` degrees: (w, x, y, z) laid out in the
        # trace's (.., qx, qy, qz, qw) column order.
        import math
        h = math.radians(tilt) / 2.0
        trace.append([t, 0.0, 0.0, 0.5, math.sin(h), 0.0, 0.0, math.cos(h)])
    return {
        "body_mass": GO2_MASS_KG,
        "trace": trace,
        "kicks": [{
            "at_time": 2.0, "fired_at": 2.0,
            "impulse_ns": [0.0, 3.0, 0.0],
            "measured_dv_mps": list(measured_dv),
        }],
    }


def test_push_on_the_commanded_axis_is_admitted():
    d = _load_driver()
    # Trial 2 of the live A/B: dv_y = 0.348 against an expected 0.1998.
    out = d.score_disturbance(
        _collected((0.0011, 0.3481, -0.0233),
                   tilts=[(t / 10, 2.0) for t in range(60)]),
        _Args())
    assert out["disturb_ok"] is True
    assert out["disturbance"]["verdict"] in ("RECOVERED", "NOT_RECOVERED")


def test_falling_body_is_rejected_even_though_dv_magnitude_looks_right():
    d = _load_driver()
    # Trial 1 of the live A/B.  |dv| = 0.212 vs expected 0.1998 -- a magnitude
    # check passes it.  Y gained 0.0121, about 6% of what was commanded.
    out = d.score_disturbance(
        _collected((-0.089, 0.0121, 0.1921),
                   tilts=[(t / 10, 2.0) for t in range(20)]
                         + [(2.0 + t / 10, 178.0) for t in range(40)]),
        _Args())
    assert out["disturb_ok"] is False
    assert out["disturbance"]["verdict"] == "NOT_APPLIED"
    assert "commanded axis" in out["disturbance"]["reason"]


def test_rejection_cuts_both_ways_and_drops_a_favourable_trial():
    d = _load_driver()
    # Trial 9: dv_y = -0.087, i.e. the body moved BACKWARDS along the push
    # axis -- yet it stayed upright and scored RECOVERED.  It must still be
    # excluded, or the filter is a thumb on the scale.
    out = d.score_disturbance(
        _collected((-0.041, -0.0874, 0.0422),
                   tilts=[(t / 10, 2.0) for t in range(60)]),
        _Args())
    assert out["disturb_ok"] is False
    assert out["disturbance"]["verdict"] == "NOT_APPLIED"


def test_no_kick_recorded_is_still_not_applied():
    d = _load_driver()
    out = d.score_disturbance({"body_mass": GO2_MASS_KG, "trace": [], "kicks": []},
                              _Args())
    assert out["disturb_ok"] is False
    assert out["disturbance"]["verdict"] == "NOT_APPLIED"


def test_no_disturbance_configured_scores_nothing():
    d = _load_driver()

    class NoDisturb(_Args):
        disturb_at = None

    out = d.score_disturbance(_collected((0.0, 0.3, 0.0), tilts=[(0.0, 2.0)]),
                              NoDisturb())
    assert out["disturbance"] is None
