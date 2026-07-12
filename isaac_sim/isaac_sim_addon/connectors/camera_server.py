#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Isaac Sim camera as an MJPEG IP camera — a fake camera stream for testing.

The camera twin of ``examples/isaac-bridge/isaac_quadruped_server.py``.  Where
the quadruped server exposes a robot BODY behind a TCP seam, this exposes a
camera SENSOR behind the SAME transport a real IP security camera uses: MJPEG
over HTTP (plus a single-frame ``/snapshot`` and a ``/status`` JSON).  Tritium
already ingests exactly this — ``plugins/camera_feeds`` ``MJPEGSource`` +
``FrameDetectionManager`` run YOLO/motion detection on any posed camera-feeds
source — so an Isaac camera registers like any other camera and NOTHING in
tritium-sc needs to know it is Isaac.  That is the whole point of the
separation: Isaac renders on the RTX 4090; the frames cross the 100 GbE LAN as
ordinary MJPEG; the detector (this box or the RTX 3080) never imports isaacsim.

Both North Star halves: FUN — a photorealistic Isaac scene (a person or the
robot dog walking) becomes a camera the operator drops on the tactical map and
watches light up with detections.  PRODUCTION — validates the real RTSP/MJPEG
security-camera track (frame -> detector -> tracker -> fusion -> map) against
render-quality imagery, at the "zero hardware ... or hundreds of nodes" scale,
BEFORE any physical camera is on site.

Dependency hygiene (the isaac-bridge rule): Isaac's python imports isaacsim +
stdlib only; it NEVER imports paho / pydantic / tritium.  tritium-sc consumes
the MJPEG and never imports isaacsim.  No dependency bleed in either direction.

Frame sources
-------------
  * ``--source isaac``    real Isaac Sim render product (needs Isaac's python
                          and a free GPU; the RTX 4090 render path).
  * ``--source synthetic``  a no-GPU stand-in that renders a moving subject with
                          numpy + cv2/PIL — runs under plain python3 so the whole
                          transport + the downstream detector can be proven with
                          no Isaac and no GPU (the isaac-bridge ``procedural``
                          fallback pattern).  ``--selftest`` uses this.

Run
---
    # Real Isaac render (Isaac's bundled python, GPU free — see README VRAM note)
    ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
        examples/isaac-camera/isaac_camera_server.py \
        --source isaac --port 8100 --camera-id isaac-cam-01 \
        --lat 37.7159 --lng -121.896 --heading 90 --fov 70 --range 80

    # No-GPU stand-in (system python3) — same MJPEG a real camera serves
    python3 examples/isaac-camera/isaac_camera_server.py --source synthetic --port 8100

    # No-GPU self-test: render N frames, assert they encode, exit 0
    python3 examples/isaac-camera/isaac_camera_server.py --selftest

Then register it in tritium-sc as an ordinary mjpeg camera (no Isaac code in SC):
    POST /api/camera-feeds/sources
    { "source_id":"isaac-cam-01", "source_type":"mjpeg",
      "uri":"http://<render-host>:8100/mjpeg",
      "lat":37.7159, "lng":-121.896, "heading":90, "fov_angle":70,
      "fov_range":80, "detect":true }
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

log = logging.getLogger("isaac-camera")


# --------------------------------------------------------------------------- #
# JPEG encoding — cv2 if present (Isaac bundles neither reliably), else PIL.
# --------------------------------------------------------------------------- #

def _make_jpeg_encoder():
    try:
        import cv2

        def _enc(rgb: np.ndarray, quality: int = 80) -> bytes:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes() if ok else b""

        return _enc, "cv2"
    except Exception:
        pass
    try:
        from PIL import Image

        def _enc(rgb: np.ndarray, quality: int = 80) -> bytes:
            im = Image.fromarray(rgb, "RGB")
            bio = io.BytesIO()
            im.save(bio, format="JPEG", quality=quality)
            return bio.getvalue()

        return _enc, "PIL"
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "no JPEG encoder available (need cv2 or Pillow in the render env)"
        ) from exc


# --------------------------------------------------------------------------- #
# Frame sources.
# --------------------------------------------------------------------------- #

class FrameSource:
    """A camera frame producer returning HxWx3 RGB uint8 arrays."""

    name = "abstract"

    def get_frame(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SyntheticFrameSource(FrameSource):
    """No-GPU stand-in: a subject walking across a static scene.

    This is the deliberate fallback (mirrors isaac-bridge's procedural dog) so
    the MJPEG transport + the downstream detector can be validated with no
    Isaac and no GPU.  It is intentionally simple — the REAL imagery comes from
    IsaacFrameSource; this only has to be motion a background-subtraction
    detector can find, i.e. a coherent moving foreground on a static scene.
    """

    name = "synthetic"

    def __init__(self, width: int = 640, height: int = 480, with_vehicle: bool = False):
        self.width = width
        self.height = height
        self.with_vehicle = with_vehicle
        self._tick = 0
        try:
            import cv2  # noqa: F401
            self._cv2 = cv2
        except Exception:
            self._cv2 = None

    def get_frame(self) -> np.ndarray:
        w, h = self.width, self.height
        frame = np.full((h, w, 3), 60, dtype=np.uint8)
        ground_y = int(h * 0.55)
        frame[ground_y:] = (86, 82, 78)  # RGB ground
        cv2 = self._cv2
        # Static wall block for scene texture.
        if cv2 is not None:
            cv2.rectangle(frame, (int(w * 0.08), int(h * 0.2)),
                          (int(w * 0.22), ground_y), (74, 70, 70), -1)
        # Subject walks left->right across the lower third, looping.
        span = int(w * 0.9)
        px = int(w * 0.05) + (self._tick * 6) % span
        py = int(h * 0.82)
        ph = int(h * 0.28)
        pw = ph // 3
        if cv2 is not None:
            cv2.rectangle(frame, (px - pw // 2, py - ph), (px + pw // 2, py),
                          (40, 34, 32), -1)
            cv2.circle(frame, (px, py - ph - pw // 2), pw // 2, (38, 32, 30), -1)
            if self.with_vehicle:
                cx = int(w * 0.95) - (self._tick * 9) % span
                cy = int(h * 0.62)
                cw, ch = int(w * 0.14), int(h * 0.09)
                cv2.rectangle(frame, (cx - cw, cy - ch), (cx + cw, cy + ch),
                              (52, 46, 44), -1)
        else:
            frame[py - ph:py, px - pw // 2:px + pw // 2] = (40, 34, 32)
        self._tick += 1
        return frame


class IsaacFrameSource(FrameSource):
    """Real Isaac Sim render product -> RGB frames.

    Runs ONLY inside Isaac's python (imports isaacsim).  Builds a headless
    SimulationApp, drops a camera prim at the requested pose, attaches an RGB
    render product / annotator, and returns the rendered frame each step.  If a
    scene USD is given it is loaded; otherwise a minimal ground + a moving
    subject prim is created so the camera has something to see.

    Kept intentionally thin — the heavy, GPU-coordinated launch is documented in
    the README VRAM note; this class is the seam, not a physics playground.
    """

    name = "isaac"

    def __init__(self, width, height, cam_pos, cam_target, scene_usd=None,
                 physics_hz=30):
        self.width = width
        self.height = height
        self._sim = None
        self._annot = None
        self._world = None
        self._subject = None
        self._t0 = time.time()
        self._boot(cam_pos, cam_target, scene_usd, physics_hz)

    def _boot(self, cam_pos, cam_target, scene_usd, physics_hz):
        # Imports are local so plain python3 (selftest/synthetic) never touches
        # isaacsim.  Any import/boot failure is raised for main() to report.
        from isaacsim.simulation_app import SimulationApp  # type: ignore

        self._sim = SimulationApp({"headless": True})

        import omni.replicator.core as rep  # type: ignore
        from pxr import UsdGeom  # type: ignore
        import isaacsim.core.utils.stage as stage_utils  # type: ignore
        from isaacsim.core.api import World  # type: ignore

        if scene_usd:
            stage_utils.open_stage(scene_usd)
        # Minimal scene: ground plane + a movable subject the camera can see.
        stage = stage_utils.get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()
        # Subject: a person-sized box that walks across the camera's view.
        from isaacsim.core.api.objects import DynamicCuboid  # type: ignore
        self._subject = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Subject",
                name="subject",
                position=np.array([2.0, -4.0, 0.9]),
                scale=np.array([0.5, 0.5, 1.8]),
                color=np.array([0.15, 0.15, 0.15]),
            )
        )
        # Camera prim + render product + rgb annotator.
        cam = rep.create.camera(position=tuple(cam_pos), look_at=tuple(cam_target))
        rp = rep.create.render_product(cam, (self.width, self.height))
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annot.attach([rp])
        self._world.reset()
        self._physics_dt = 1.0 / max(1, physics_hz)

    def get_frame(self) -> np.ndarray:
        import omni.replicator.core as rep  # type: ignore
        # Walk the subject across the view (deterministic sine sweep).
        if self._subject is not None:
            t = time.time() - self._t0
            x = 2.0 + 3.0 * math.sin(t * 0.5)
            try:
                self._subject.set_world_pose(position=np.array([x, -4.0, 0.9]))
            except Exception:
                pass
        self._world.step(render=True)
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        data = self._annot.get_data()
        rgba = np.asarray(data, dtype=np.uint8)
        if rgba.ndim == 3 and rgba.shape[2] == 4:
            return np.ascontiguousarray(rgba[:, :, :3])
        return rgba

    def close(self) -> None:
        try:
            if self._sim is not None:
                self._sim.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# MJPEG HTTP server (stdlib only).
# --------------------------------------------------------------------------- #

class CameraState:
    """Shared latest-frame holder + camera metadata for the HTTP handlers."""

    def __init__(self, source: FrameSource, meta: dict, fps: int, encoder):
        self.source = source
        self.meta = meta
        self.fps = max(1, fps)
        self.encode = encoder
        self._latest_jpeg: bytes = b""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.frames = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="isaac-cam-render").start()

    def _loop(self):
        interval = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                rgb = self.source.get_frame()
                jpeg = self.encode(rgb)
                if jpeg:
                    with self._lock:
                        self._latest_jpeg = jpeg
                        self.frames += 1
            except Exception as exc:
                log.warning("frame render failed: %s", exc)
                time.sleep(0.5)
            dt = time.time() - t0
            if dt < interval:
                self._stop.wait(interval - dt)

    def latest(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def stop(self):
        self._stop.set()
        self.source.close()


def _make_handler(state: CameraState):
    import json

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/status"):
                body = json.dumps({
                    **state.meta, "frames": state.frames,
                    "source": state.source.name, "status": "online",
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/snapshot"):
                jpeg = state.latest()
                if not jpeg:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path.startswith("/mjpeg") or self.path == "/":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                try:
                    interval = 1.0 / state.fps
                    while True:
                        jpeg = state.latest()
                        if jpeg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def _build_source(args) -> FrameSource:
    if args.source == "isaac":
        cam_pos = [args.cam_x, args.cam_y, args.cam_z]
        cam_target = [args.target_x, args.target_y, args.target_z]
        return IsaacFrameSource(
            args.width, args.height, cam_pos, cam_target,
            scene_usd=args.scene or None, physics_hz=args.physics_hz,
        )
    return SyntheticFrameSource(args.width, args.height, with_vehicle=args.with_vehicle)


def selftest(args) -> int:
    """No-GPU: render N synthetic frames, assert they encode, report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    encode, enc_name = _make_jpeg_encoder()
    src = SyntheticFrameSource(args.width, args.height, with_vehicle=True)
    sizes = []
    for _ in range(args.selftest_frames):
        rgb = src.get_frame()
        assert rgb.shape == (args.height, args.width, 3), f"bad frame shape {rgb.shape}"
        jpeg = encode(rgb)
        assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9", "not a JPEG"
        sizes.append(len(jpeg))
    print(f"SELFTEST OK frames={len(sizes)} encoder={enc_name} "
          f"jpeg_bytes~{int(np.mean(sizes))} res={args.width}x{args.height}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Isaac Sim MJPEG camera server")
    ap.add_argument("--source", choices=["isaac", "synthetic"], default="synthetic")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--camera-id", default="isaac-cam-01")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10)
    # Camera geo pose (for tritium-sc registration / detection geo-placement).
    ap.add_argument("--lat", type=float, default=37.7159)
    ap.add_argument("--lng", type=float, default=-121.896)
    ap.add_argument("--heading", type=float, default=90.0)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--range", dest="range_m", type=float, default=80.0)
    # Isaac scene / camera world placement.
    ap.add_argument("--scene", default="")
    ap.add_argument("--cam-x", type=float, default=0.0)
    ap.add_argument("--cam-y", type=float, default=0.0)
    ap.add_argument("--cam-z", type=float, default=2.0)
    ap.add_argument("--target-x", type=float, default=3.0)
    ap.add_argument("--target-y", type=float, default=-4.0)
    ap.add_argument("--target-z", type=float, default=0.9)
    ap.add_argument("--physics-hz", type=int, default=30)
    ap.add_argument("--with-vehicle", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-frames", type=int, default=12)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.selftest:
        return selftest(args)

    encode, enc_name = _make_jpeg_encoder()
    source = _build_source(args)
    meta = {
        "camera_id": args.camera_id,
        "lat": args.lat, "lng": args.lng, "heading": args.heading,
        "fov_angle": args.fov, "fov_range": args.range_m,
        "width": args.width, "height": args.height,
    }
    state = CameraState(source, meta, args.fps, encode)
    state.start()

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))

    def _shutdown(*_a):
        log.info("shutting down")
        state.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "ISAAC CAMERA READY source=%s encoder=%s http://%s:%d/mjpeg id=%s "
        "pose=(%.4f,%.4f h=%.0f fov=%.0f r=%.0f)",
        source.name, enc_name, args.host, args.port, args.camera_id,
        args.lat, args.lng, args.heading, args.fov, args.range_m,
    )
    try:
        httpd.serve_forever()
    finally:
        state.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
