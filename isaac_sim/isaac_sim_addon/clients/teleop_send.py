"""Send live twist commands to a Newton-driven body over UDP.

The counterpart to ``go2_newton_gait.py --live-port``.  Two modes:

* ``--program`` plays a *scripted* timeline of commands — ``"t:linear,angular"``
  segments, plus ``t:silent`` to stop transmitting entirely.
* ``--stick`` reads a real gamepad through ``tritium_lib.control.teleop``.

The scripted mode exists because a live proof needs to be *checkable*.  A hand
on a stick produces a run nobody can reproduce and no test can grade: to claim
the body followed an external command you must know exactly what was sent and
when, and be able to send it again.  ``t:silent`` is in the grammar for the
same reason — it is how the watchdog gets tested, and a watchdog that has
never actually been starved is a comment, not a safety mechanism.

Nothing here decides anything about the body.  Shaping, clamping and the
staleness rules all live in ``tritium_lib.control``; this is a socket and a
clock.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

DEFAULT_PORT = 18974
DEFAULT_RATE_HZ = 20.0


class Segment:
    """One leg of a scripted command program."""

    def __init__(self, start_s: float, linear: float | None,
                 angular: float | None) -> None:
        self.start_s = start_s
        self.linear = linear
        self.angular = angular

    @property
    def silent(self) -> bool:
        """A silent segment sends nothing at all — it does not send a zero.

        The distinction is the entire point: a zero twist is the sender saying
        "stop", while silence is the sender *being gone*.  Only the second one
        exercises the watchdog, and a body that cannot tell them apart is a
        body that keeps walking when its operator's process dies.
        """
        return self.linear is None

    def __repr__(self) -> str:
        what = "silent" if self.silent else f"{self.linear},{self.angular}"
        return f"Segment({self.start_s}:{what})"


def parse_program(text: str) -> list[Segment]:
    """Parse ``"0:0.6,0; 3:silent; 5:0,0.8"`` into ordered segments."""
    segments: list[Segment] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"segment wants T:LINEAR,ANGULAR or T:silent, got {chunk!r}")
        when, _, what = chunk.partition(":")
        start = float(when)
        what = what.strip().lower()
        if what == "silent":
            segments.append(Segment(start, None, None))
            continue
        parts = what.split(",")
        if len(parts) != 2:
            raise ValueError(f"segment wants LINEAR,ANGULAR, got {what!r}")
        segments.append(Segment(start, float(parts[0]), float(parts[1])))
    if not segments:
        raise ValueError("empty program")
    segments.sort(key=lambda s: s.start_s)
    return segments


def segment_at(segments: list[Segment], elapsed: float) -> Segment | None:
    """The segment in force at ``elapsed``, or None before the first starts."""
    current = None
    for seg in segments:
        if seg.start_s <= elapsed:
            current = seg
        else:
            break
    return current


def frame(seq: int, linear: float, angular: float) -> bytes:
    return json.dumps({
        "cmd": "twist",
        "seq": seq,
        "linear_mps": linear,
        "angular_rps": angular,
    }).encode()


def run_program(host: str, port: int, segments: list[Segment],
                duration_s: float, rate_hz: float,
                out=sys.stdout) -> dict:
    """Play the program, returning a log of what was actually sent.

    The returned log — not the program — is the evidence.  A scheduler that
    fell behind, or a segment boundary that landed a tick late, is visible
    here and invisible in the argument that was passed in.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / rate_hz
    sent: list[list[float]] = []
    seq = 0
    t0 = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= duration_s:
                break
            seg = segment_at(segments, elapsed)
            if seg is not None and not seg.silent:
                seq += 1
                sock.sendto(frame(seq, seg.linear, seg.angular), (host, port))
                sent.append([round(elapsed, 3), seg.linear, seg.angular, seq])
            time.sleep(period)
    finally:
        sock.close()
    log = {
        "sent": sent,
        "frames": len(sent),
        "duration_s": round(time.monotonic() - t0, 3),
        "silent_s": round(duration_s - len(sent) * period, 3),
    }
    print(json.dumps(log, indent=2), file=out)
    return log


def run_stick(host: str, port: int, duration_s: float, rate_hz: float,
              max_linear: float, max_angular: float, out=sys.stdout) -> dict:
    """Read a real pad and send it. Requires pygame and a connected device."""
    import pygame  # imported late: the scripted path must not need it

    from tritium_lib.control import GamepadState, TeleopProfile, twist_from_stick

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("no gamepad detected")
    pad = pygame.joystick.Joystick(0)
    pad.init()
    profile = TeleopProfile(max_linear_mps=max_linear, max_angular_rps=max_angular)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / rate_hz
    seq = 0
    sent = []
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            pygame.event.pump()
            state = GamepadState(
                left_x=pad.get_axis(0), left_y=pad.get_axis(1),
                right_x=pad.get_axis(2) if pad.get_numaxes() > 2 else 0.0,
                right_y=pad.get_axis(3) if pad.get_numaxes() > 3 else 0.0,
            )
            twist = twist_from_stick(state, profile)
            seq += 1
            sock.sendto(frame(seq, twist.linear_mps, twist.angular_rps),
                        (host, port))
            sent.append([round(time.monotonic() - t0, 3),
                         round(twist.linear_mps, 4),
                         round(twist.angular_rps, 4), seq])
            time.sleep(period)
    finally:
        sock.close()
        pygame.quit()
    log = {"sent": sent, "frames": len(sent), "device": pad.get_name()}
    print(json.dumps(log, indent=2), file=out)
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--program", metavar="T:LIN,ANG;T:silent;...",
                    help="scripted command timeline in seconds")
    ap.add_argument("--stick", action="store_true",
                    help="read a real gamepad instead of a script")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ,
                    help=f"send rate in Hz (default {DEFAULT_RATE_HZ})")
    ap.add_argument("--max-linear", type=float, default=0.8)
    ap.add_argument("--max-angular", type=float, default=1.5)
    args = ap.parse_args(argv)

    if args.stick:
        run_stick(args.host, args.port, args.seconds, args.rate,
                  args.max_linear, args.max_angular)
        return 0
    if not args.program:
        ap.error("one of --program or --stick is required")
    run_program(args.host, args.port, parse_program(args.program),
                args.seconds, args.rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
