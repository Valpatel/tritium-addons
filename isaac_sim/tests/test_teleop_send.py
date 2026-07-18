"""The scripted command sender, including a real-socket loopback.

The loopback test at the bottom is the one that matters.  Both halves of this
path are pinned to ``tritium_lib.control.CommandLink`` rather than to each
other, and unit tests on either side would keep passing if the wire formats
drifted apart.  Sending real datagrams through a real socket into the real
decoder is what proves they did not — and it needs no GPU, no Isaac and no
robot.
"""

import io
import socket

import pytest

from isaac_sim_addon.clients.teleop_send import (
    Segment,
    parse_program,
    run_program,
    segment_at,
)


# ------------------------------------------------------------- parsing


def test_parses_a_multi_segment_program():
    segs = parse_program("0:0.6,0; 3:0,0.8; 6:0,0")
    assert [s.start_s for s in segs] == [0.0, 3.0, 6.0]
    assert segs[0].linear == pytest.approx(0.6)
    assert segs[1].angular == pytest.approx(0.8)


def test_segments_are_sorted_by_time():
    """Out-of-order input is a typo, not a different program."""
    segs = parse_program("5:0,0; 0:0.6,0")
    assert [s.start_s for s in segs] == [0.0, 5.0]


def test_silent_segment_is_distinct_from_a_zero_twist():
    """A zero says 'stop'; silence says the sender is GONE. Only one of those
    exercises the watchdog."""
    segs = parse_program("0:0.6,0; 3:silent; 5:0,0")
    assert segs[1].silent is True
    assert segs[2].silent is False
    assert segs[2].linear == 0.0


def test_whitespace_and_trailing_separators_are_tolerated():
    assert len(parse_program(" 0:0.5,0 ;  2:silent ; ")) == 2


@pytest.mark.parametrize("bad", ["", "nonsense", "0:", "0:1", "0:1,2,3", "x:1,2"])
def test_malformed_programs_are_refused(bad):
    with pytest.raises(ValueError):
        parse_program(bad)


# ------------------------------------------------------------ selection


def test_nothing_is_in_force_before_the_first_segment():
    segs = parse_program("2:0.6,0")
    assert segment_at(segs, 0.0) is None
    assert segment_at(segs, 1.99) is None


def test_the_latest_started_segment_is_in_force():
    segs = parse_program("0:0.6,0; 3:0,0.8; 6:0,0")
    assert segment_at(segs, 0.0).linear == pytest.approx(0.6)
    assert segment_at(segs, 2.9).linear == pytest.approx(0.6)
    assert segment_at(segs, 3.0).angular == pytest.approx(0.8)
    assert segment_at(segs, 99.0).angular == 0.0


# ------------------------------------------------- real-socket loopback


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.5)
    yield sock
    sock.close()


def drain(sock):
    packets = []
    sock.setblocking(False)
    while True:
        try:
            packets.append(sock.recv(2048))
        except BlockingIOError:
            return packets


def test_sent_frames_are_accepted_by_the_drivers_own_decoder(receiver):
    """End to end over a real socket: sender -> UDP -> CommandLink."""
    from tritium_lib.control import CommandLink

    port = receiver.getsockname()[1]
    log = run_program("127.0.0.1", port, parse_program("0:0.5,0.25"),
                      duration_s=0.3, rate_hz=40.0, out=io.StringIO())
    assert log["frames"] > 0

    link = CommandLink()
    packets = drain(receiver)
    assert len(packets) > 0
    for i, pkt in enumerate(packets):
        assert link.ingest(pkt, now_s=i * 0.025) is True
    assert link.rejected == 0
    twist = link.poll((len(packets) - 1) * 0.025)
    assert twist.linear_mps == pytest.approx(0.5)
    assert twist.angular_rps == pytest.approx(0.25)


def test_a_silent_segment_actually_stops_the_datagrams(receiver):
    """The watchdog can only be trusted if silence is really silence."""
    port = receiver.getsockname()[1]
    log = run_program("127.0.0.1", port, parse_program("0:0.5,0; 0.15:silent"),
                      duration_s=0.4, rate_hz=40.0, out=io.StringIO())
    # Everything sent landed before the silent segment began.
    assert log["frames"] > 0
    assert max(row[0] for row in log["sent"]) < 0.15 + 0.05


def test_sequence_numbers_are_strictly_increasing(receiver):
    """The receiver's ordering gate drops anything that is not."""
    port = receiver.getsockname()[1]
    log = run_program("127.0.0.1", port, parse_program("0:0.4,0"),
                      duration_s=0.3, rate_hz=40.0, out=io.StringIO())
    seqs = [row[3] for row in log["sent"]]
    assert seqs == sorted(set(seqs))
    assert seqs[0] == 1


def test_the_log_records_what_was_sent_not_what_was_asked_for(receiver):
    """A scheduler that fell behind is visible in the log and nowhere else."""
    port = receiver.getsockname()[1]
    log = run_program("127.0.0.1", port, parse_program("0:0.6,-0.3"),
                      duration_s=0.2, rate_hz=40.0, out=io.StringIO())
    assert all(row[1] == pytest.approx(0.6) for row in log["sent"])
    assert all(row[2] == pytest.approx(-0.3) for row in log["sent"])
    assert log["duration_s"] >= 0.2


def test_segment_repr_is_readable():
    assert "silent" in repr(Segment(1.0, None, None))
