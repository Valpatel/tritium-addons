#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""No-GPU end-to-end proof: Isaac camera server -> MJPEG -> detector -> track.

Starts ``connectors/camera_server.py`` in its no-GPU ``synthetic`` mode (the same
MJPEG a real Isaac render or a real IP camera serves), opens the stream with
the SAME classical detector the tritium-sc FrameDetectionManager uses, and
asserts a posed camera detection projects to a world position — the whole
camera perception chain, no Isaac and no GPU required.

When Isaac is launched with a free GPU, ``--source isaac`` serves render-quality
frames through the identical HTTP path and this proof holds verbatim; only the
pixels get better.

Run:  python3 examples/smoke_detect.py
Needs tritium_lib on PYTHONPATH (the perception pipeline lives there).
"""

from __future__ import annotations

import sys
import time
import urllib.request

import numpy as np

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


def _read_mjpeg_frames(url: str, want: int, timeout: float = 15.0):
    """Yield up to `want` BGR frames decoded from a multipart MJPEG stream."""
    import cv2

    stream = urllib.request.urlopen(url, timeout=5)
    buf = b""
    got = 0
    deadline = time.time() + timeout
    while got < want and time.time() < deadline:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a + 2)
            if a == -1 or b == -1:
                break
            jpg = buf[a:b + 2]
            buf = buf[b + 2:]
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                got += 1
                yield frame
                if got >= want:
                    return


def main() -> int:
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve().parent
    # The MJPEG camera server moved to the connectors package (was a sibling
    # ``isaac_camera_server.py`` next to this example).
    server = here.parent / "isaac_sim_addon" / "connectors" / "camera_server.py"
    assert server.exists(), f"camera server not found at {server}"

    # 1. Start the Isaac camera server (no-GPU synthetic frames).
    proc = subprocess.Popen(
        [sys.executable, str(server), "--source", "synthetic",
         "--port", str(PORT), "--fps", "12", "--width", "320", "--height", "240"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # 2. Wait for /status online.
        import json
        online = False
        for _ in range(40):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{BASE}/status", timeout=2) as r:
                    st = json.loads(r.read())
                    if st.get("status") == "online" and st.get("frames", 0) > 0:
                        online = True
                        break
            except Exception:
                continue
        assert online, "isaac camera server never came online"
        print("camera online:", st.get("source"), "frames:", st.get("frames"))

        # 3. Detect on the live MJPEG stream with the tritium-lib pipeline.
        from tritium_lib.perception import (
            build_frame_detector, GroundCameraModel, FrameDetectionPipeline,
        )
        from tritium_lib.tracking.target_tracker import TargetTracker

        detector = build_frame_detector(prefer="auto")
        print("detector backend:", detector.backend_name)
        tracker = TargetTracker()
        model = GroundCameraModel(
            lat=st["lat"], lng=st["lng"], heading_deg=st["heading"],
            fov_deg=st["fov_angle"], range_m=st["fov_range"],
            image_w=st["width"], image_h=st["height"],
        )
        frames = list(_read_mjpeg_frames(f"{BASE}/mjpeg", want=40))
        assert frames, "no frames decoded from MJPEG"
        print(f"decoded {len(frames)} MJPEG frames {frames[0].shape}")

        it = iter(frames)
        pipe = FrameDetectionPipeline(
            detector=detector,
            frame_provider=lambda: next(it, None),
            detection_sink=tracker.update_from_detection,
            model_provider=lambda: model,
            source_id=st["camera_id"],
        )
        emitted = 0
        for _ in range(len(frames)):
            emitted += pipe.tick()

        det_tracks = [t for t in tracker.get_all()
                      if str(t.target_id).startswith("det_")]
        assert emitted > 0, "detector emitted nothing on the Isaac camera stream"
        assert det_tracks, "no det_* track reached the tracker"
        t = det_tracks[0]
        print(f"PROOF OK — {emitted} detections, {len(det_tracks)} det_* track(s); "
              f"{t.target_id} source={t.source} class={t.classification} "
              f"pos=({t.position[0]:.1f},{t.position[1]:.1f}) "
              f"cam={(t.kinematics or {}).get('camera_id')}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
