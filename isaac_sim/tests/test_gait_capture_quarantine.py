"""A capture can frame a run; it can never score one.

The mid-run ``/sim/capture`` render runs synchronously on the kit main
thread -- the same thread that pumps the app-update stream driving the gait
control callback -- so a captured trial's controller freezes mid-render while
Newton keeps integrating underneath.  Measured on the live kit: same command
and push, **0/8 upright capture-on vs 8/8 capture-free**.  That confound
minted a false "~5 N*s inverts the body" ceiling that reached three
documents before anyone caught it, guarded only by a WARNING in a help
string nobody is obliged to read.

These tests pin the quarantine that makes a recurrence structural rather
than procedural:

* a scored trial NEVER captures -- ``main()`` takes the frame from a
  dedicated evidence run and hands every scored trial ``capture=None``, and
  the scored path's bridge traffic is independent of every capture setting;
* a captured run can never contribute to a rate -- ``run_trial`` stamps it
  ``capture_perturbed`` and wraps every verdict ``UNSCORED[...]``, and
  ``arm_summary``/``split_scoreable`` bar it from every aggregate, visibly.

No Isaac, no GPU, no network: the bridge is faked at the request layer.

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import types
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

FAKE_PNG = b"not-a-real-png"


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "go2_newton_gait", EXAMPLES / "go2_newton_gait.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


driver = _load_driver()

GAIT = {"stride_hz": 1.0, "stand": {"FL_thigh": 0.8},
        "table": [{"phase": 0.0, "angles": {"FL_thigh": 0.8}}]}

# A level, walking trace: two samples 1.2 m apart at standing height.
LEVEL_TRACE = [[0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 1.0],
               [6.0, 1.2, 0.0, 0.40, 0.0, 0.0, 0.0, 1.0]]


def _collected(**overrides) -> dict:
    """What COLLECT_CODE returns for a healthy, kick-free run."""
    out = {
        "steps": 360, "err": None, "trace": list(LEVEL_TRACE),
        "joint_names": [f"J{i}" for i in range(12)], "kicks": [],
        "body_mass": 15.0, "stabilized": 350, "max_cmd_seen": 0.05,
        "max_cb_gap_s": 0.017, "follow": [], "arrived_at": None,
        "live": [], "live_sock_err": None,
    }
    out.update(overrides)
    return out


class FakeBridge(driver.Bridge):
    """Answers the bridge protocol from canned payloads, recording traffic."""

    def __init__(self, collected: dict | None = None) -> None:
        super().__init__(port=0)
        self.collected = collected or _collected()
        self.paths: list[str] = []

    def request(self, path: str, body: dict | None = None) -> dict:
        self.paths.append(path)
        if path == "/sim/capture":
            return {"status": "success",
                    "result": {"image_base64":
                               base64.b64encode(FAKE_PNG).decode()}}
        if path != "/execute":
            return {"status": "success", "result": {}}
        code = (body or {}).get("code", "")
        if "add_reference_to_stage" in code:          # scene build
            payload = {"variant": "physx", "robot_valid": True,
                       "drives_configured": 12}
        elif "create_articulation_view" in code:      # driver install
            payload = {"joint_names": [f"J{i}" for i in range(12)],
                       "count": 1, "dof": 12, "stabilize": True,
                       "legs_wired": ["FL", "FR", "RL", "RR"]}
        elif "removed" in code:                       # cleanup
            payload = {"removed": True}
        else:                                         # collect
            payload = self.collected
        return {"status": "success", "result": {"return_value": payload}}


def _args(**overrides):
    """Just the fields run_trial and its helpers read."""
    ns = types.SimpleNamespace(
        stiffness=60.0, damping=4.0, seconds=6.0, sample_every=10,
        kp=None, kd=None, speed=0.6, steer=0.0, steer_track=0.26,
        live_port=None, live_timeout=0.5, live_max_linear=1.0,
        live_max_angular=2.0, disturb_at=None, disturb_impulse="0,15,0",
        disturb_window=0.1, settle_below=10.0, settle_hold=1.0,
        capture_distance=3.0, capture_elevation=18.0, capture_at=None,
    )
    for key, val in overrides.items():
        setattr(ns, key, val)
    return ns


def _no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(driver.time, "sleep", slept.append)
    return slept


# ------------------------------------------------- scored trials never touch it
def test_scored_trial_never_touches_the_capture_rpc(monkeypatch):
    _no_sleep(monkeypatch)
    br = FakeBridge()
    score = driver.run_trial(br, GAIT, _args(), stabilize=True, capture=None)
    assert "/sim/capture" not in br.paths
    assert "/camera/look_at" not in br.paths
    assert score["capture_perturbed"] is False
    assert score["verdict"] == "WALKED"
    assert score["max_cb_gap_s"] == 0.017  # the stall tripwire rides along


def test_scored_bridge_traffic_is_independent_of_capture_settings(monkeypatch):
    """The whole guarantee in one assertion: with capture=None the run makes
    the same bridge calls and the same waits no matter what any --capture-*
    flag says, so a scored outcome cannot depend on capture configuration."""
    runs = []
    for overrides in ({}, {"capture_at": 1.0, "capture_distance": 9.0,
                           "capture_elevation": 80.0}):
        slept = _no_sleep(monkeypatch)
        br = FakeBridge()
        score = driver.run_trial(br, GAIT, _args(**overrides),
                                 stabilize=True, capture=None)
        runs.append((br.paths, list(slept), score["verdict"]))
    assert runs[0] == runs[1]


# --------------------------------------------------- captured runs are UNSCORED
def test_captured_run_is_stamped_wrapped_and_still_writes_the_frame(
        monkeypatch, tmp_path):
    _no_sleep(monkeypatch)
    br = FakeBridge()
    shot = tmp_path / "evidence.png"
    score = driver.run_trial(br, GAIT, _args(), stabilize=True,
                             capture=str(shot))
    # The evidence survives -- the frames are cited by the findings doc...
    assert br.paths.count("/sim/capture") == 1
    assert shot.read_bytes() == FAKE_PNG
    # ...but the numbers can never pass as clean.
    assert score["capture_perturbed"] is True
    assert score["verdict"] == "UNSCORED[WALKED]"


def test_captured_run_cannot_claim_disturbance_evidence(monkeypatch, tmp_path):
    """A perturbed trial must not count as a recovery either way.  The same
    collected payload scores disturb_ok=True capture-free -- proving the
    downgrade comes from the capture, not from the data."""
    args = _args(disturb_at=2.0, disturb_impulse="0,3,0")
    # A kick that really landed (dv_y ~ J/m = 3/15 = 0.2) on a level body.
    tilt_trace = [[t / 10, 0.0, t / 60, 0.40, 0.0, 0.0, 0.0, 1.0]
                  for t in range(61)]
    kicks = [{"at_time": 2.0, "fired_at": 2.0, "impulse_ns": [0.0, 3.0, 0.0],
              "window_s": 0.1, "measured_dv_mps": [0.0, 0.21, 0.0]}]

    _no_sleep(monkeypatch)
    clean = driver.run_trial(FakeBridge(_collected(trace=tilt_trace,
                                                   kicks=kicks)),
                             GAIT, args, stabilize=True, capture=None)
    assert clean["disturb_ok"] is True
    assert clean["disturbance"]["verdict"] == "RECOVERED"

    _no_sleep(monkeypatch)
    perturbed = driver.run_trial(FakeBridge(_collected(trace=tilt_trace,
                                                       kicks=kicks)),
                                 GAIT, args, stabilize=True,
                                 capture=str(tmp_path / "kick.png"))
    assert perturbed["disturb_ok"] is False
    assert perturbed["disturbance"]["verdict"] == "UNSCORED[RECOVERED]"


# ------------------------------------------------------- rate-level enforcement
def _clean(verdict="WALKED", tumbled=False, tilt=5.0, dist=1.2):
    return {"verdict": verdict, "tumbled": tumbled, "max_tilt_deg": tilt,
            "displacement_m": dist}


def _perturbed(verdict="UNSCORED[TUMBLED]", tumbled=True):
    return {"verdict": verdict, "tumbled": tumbled, "max_tilt_deg": 178.0,
            "displacement_m": 0.4, "capture_perturbed": True}


def test_split_scoreable_partitions_on_the_stamp():
    clean, perturbed = _clean(), _perturbed()
    scoreable, dropped = driver.split_scoreable([clean, perturbed, _clean()])
    assert perturbed not in scoreable
    assert dropped == [perturbed]
    assert len(scoreable) == 2


def test_arm_summary_excludes_perturbed_and_says_so():
    """The 0/8-vs-8/8 scenario: eight clean upright trials plus one captured
    tumble must read 8/8, with the exclusion visible, not 8/9."""
    runs = [_clean() for _ in range(8)] + [_perturbed()]
    s = driver.arm_summary(runs)
    assert s["trials"] == 8
    assert s["upright_rate"] == 1.0
    assert s["walked_rate"] == 1.0
    assert s["capture_excluded"] == 1
    assert s["capture_excluded_verdicts"] == ["UNSCORED[TUMBLED]"]
    assert "UNSCORED[TUMBLED]" not in s["verdicts"]


def test_a_perturbed_walk_cannot_inflate_the_rate():
    """The exclusion is not a thumb on the scale: a captured trial that
    happened to walk is dropped exactly like one that tumbled."""
    lucky = _perturbed(verdict="UNSCORED[WALKED]", tumbled=False)
    s = driver.arm_summary([_clean("TUMBLED", tumbled=True, tilt=178.0), lucky])
    assert s["trials"] == 1
    assert s["walked"] == 0
    assert s["upright"] == 0


def test_arm_summary_refuses_a_rate_from_only_perturbed_trials():
    s = driver.arm_summary([_perturbed(), _perturbed()])
    assert s["trials"] == 0
    assert s["walked_rate"] is None
    assert s["upright_rate"] is None
    assert s["capture_excluded"] == 2


# -------------------------------------------------------- main() level policy
def test_main_captures_only_on_the_dedicated_unscored_evidence_run(
        monkeypatch, tmp_path, capsys):
    gait_file = tmp_path / "gait.json"
    gait_file.write_text(json.dumps(GAIT))

    calls: list[tuple[bool, str | None]] = []

    def stub_run_trial(br, gait, args, stabilize, capture=None,
                       route=None, obstacles=None):
        calls.append((stabilize, capture))
        if capture is not None:
            return dict(_perturbed(), max_cb_gap_s=0.412)
        return dict(_clean(), max_cb_gap_s=0.017)

    class StubBridge:
        def __init__(self, host, port):
            pass

        def sim_state(self):
            return {"result": {"state": "stopped"}}

    monkeypatch.setattr(driver, "run_trial", stub_run_trial)
    monkeypatch.setattr(driver, "Bridge", StubBridge)
    monkeypatch.setattr(sys, "argv", [
        "go2_newton_gait.py", "--gait-file", str(gait_file),
        "--capture", str(tmp_path / "shot.png"),
        "--trials", "3", "--stabilize", "both"])

    rc = driver.main()
    assert rc == 0  # 3/3 scored WALKED; the evidence tumble moved nothing

    closed = [c for c in calls if c[0] is True]
    opened = [c for c in calls if c[0] is False]
    for arm, suffix in ((closed, "-closed.png"), (opened, "-open.png")):
        assert len(arm) == 4  # one evidence run + three scored trials
        assert arm[0][1] is not None and arm[0][1].endswith(suffix)
        assert all(capture is None for _, capture in arm[1:])

    out = capsys.readouterr().out
    assert "excluded from every rate" in out       # the evidence line says so
    assert '"trials": 3' in out                    # rate is over scored trials
    assert '"capture_excluded"' not in out         # nothing leaked into runs


def test_main_without_capture_runs_no_evidence_trial(monkeypatch, tmp_path):
    gait_file = tmp_path / "gait.json"
    gait_file.write_text(json.dumps(GAIT))
    calls: list[tuple[bool, str | None]] = []

    def stub_run_trial(br, gait, args, stabilize, capture=None,
                       route=None, obstacles=None):
        calls.append((stabilize, capture))
        return dict(_clean(), max_cb_gap_s=0.017)

    class StubBridge:
        def __init__(self, host, port):
            pass

        def sim_state(self):
            return {"result": {"state": "stopped"}}

    monkeypatch.setattr(driver, "run_trial", stub_run_trial)
    monkeypatch.setattr(driver, "Bridge", StubBridge)
    monkeypatch.setattr(sys, "argv", [
        "go2_newton_gait.py", "--gait-file", str(gait_file), "--trials", "2"])
    assert driver.main() == 0
    assert calls == [(True, None), (True, None)]


# ------------------------------------------------------------ stall tripwire
def test_stall_tripwire_is_wired_from_callback_to_collect():
    """The gap between control callbacks is measured in the kit and carried
    out with the trace, so a stalled run is a number, not an inference from
    an 0/8-vs-8/8 anomaly three documents later."""
    assert "max_cb_gap_s" in driver.DRIVER_CODE
    assert "max_cb_gap_s" in driver.COLLECT_CODE
    built = driver.build_driver_code(GAIT, 5.0, 60.0, 4.0, 10)
    assert "max_cb_gap_s" in built
    assert compile(built, "<driver>", "exec")
    assert compile(driver.COLLECT_CODE, "<collect>", "exec")
