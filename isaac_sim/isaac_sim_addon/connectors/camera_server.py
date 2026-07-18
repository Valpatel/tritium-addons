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

Channels / routes
-----------------
  * ``/mjpeg`` (+ ``/``)   the RGB stream (``main`` channel; always on).
  * ``/depth``             ``--depth``: a colorized DEPTH stream (capability 8).
                           Isaac side = ``distance_to_image_plane`` annotator;
                           synthetic side = a plausible depth ramp, so depth is
                           testable headless.
  * ``/mjpeg_right``       ``--stereo``: the right eye of a stereo pair. Isaac
                           side = a second posed camera offset by
                           ``--stereo-baseline`` metres; synthetic side = a
                           parallax-shifted frame. ``/mjpeg`` is the left eye.
  * ``/snapshot``          one JPEG of ``main`` (or ``?channel=depth|right``).
  * ``/status``            JSON metadata incl. the enabled ``channels`` list.

Run
---
    # Real Isaac render (Isaac's bundled python, GPU free — see README VRAM note)
    ~/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release/python.sh \
        isaac_sim_addon/connectors/camera_server.py \
        --source isaac --port 8100 --camera-id isaac-cam-01 --depth --stereo \
        --lat 37.7159 --lng -121.896 --heading 90 --fov 70 --range 80

    # No-GPU stand-in (system python3) — same MJPEG a real camera serves
    python3 isaac_sim_addon/connectors/camera_server.py --source synthetic \
        --port 8100 --depth --stereo

    # No-GPU self-test: render N frames (RGB + depth + stereo), assert they encode
    python3 isaac_sim_addon/connectors/camera_server.py --selftest

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
# Depth colorizing (shared by synthetic + Isaac depth channels).
# --------------------------------------------------------------------------- #

def colorize_depth(depth: np.ndarray, near: float = 0.5, far: float = 60.0) -> np.ndarray:
    """Map a HxW float depth (metres; ``inf``/``nan`` = sky/no-return) to an
    HxWx3 RGB uint8 image — bright/warm = near, dark = far — so the depth channel
    is a viewable MJPEG stream like the RGB one. cv2's TURBO map when available,
    else a plain grayscale ramp (keeps this no-GPU + cv2-optional)."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    d = np.nan_to_num(d, nan=far, posinf=far, neginf=far)
    d = np.clip(d, near, far)
    inv = 1.0 - (d - near) / (far - near)          # 1.0 near .. 0.0 far
    gray = (inv * 255.0).astype(np.uint8)
    try:
        import cv2

        col = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)   # BGR
        return np.ascontiguousarray(col[:, :, ::-1])        # -> RGB
    except Exception:
        return np.repeat(gray[:, :, None], 3, axis=2)


# --------------------------------------------------------------------------- #
# Frame sources.
# --------------------------------------------------------------------------- #

class FrameSource:
    """A camera frame producer returning HxWx3 RGB uint8 arrays.

    Optional extra channels (return ``None`` when the source does not provide
    them): :meth:`get_depth` (a colorized depth image, capability 8) and
    :meth:`get_right_frame` (the right eye of a stereo pair). Both, when present,
    correspond to the SAME instant as the most recent :meth:`get_frame`.
    """

    name = "abstract"

    #: True when the source may ONLY be stepped from the process's main thread.
    #: Omniverse Kit is such a source: ``world.step()`` /
    #: ``rep.orchestrator.step()`` block forever — silently, with no exception —
    #: when driven from a worker thread, which is why a healthy-looking
    #: ``--source isaac`` server once served zero frames.  Pure-numpy sources
    #: (the synthetic stand-in) are thread-safe and leave this False.
    requires_main_thread = False

    def get_frame(self) -> np.ndarray:
        raise NotImplementedError

    def get_depth(self) -> np.ndarray | None:
        return None

    def get_right_frame(self) -> np.ndarray | None:
        return None

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

    def __init__(self, width: int = 640, height: int = 480, with_vehicle: bool = False,
                 with_depth: bool = False, with_stereo: bool = False,
                 baseline_px: int | None = None):
        self.width = width
        self.height = height
        self.with_vehicle = with_vehicle
        self.with_depth = with_depth
        self.with_stereo = with_stereo
        # Right-eye foreground disparity, scaled to the frame width (nearer =
        # this much; the far background stays ~0-disparity). Deterministic so the
        # stereo pair is reproducible headless.
        self.baseline_px = int(baseline_px) if baseline_px is not None else max(4, width // 45)
        self._tick = 0          # advances once per get_frame (animation clock)
        self._render_tick = 0   # the tick of the last get_frame — depth/right reuse it
        try:
            import cv2  # noqa: F401
            self._cv2 = cv2
        except Exception:
            self._cv2 = None

    # -- geometry shared by both eyes + the depth channel --------------------
    def _subject_geom(self, tick: int):
        """(cx, foot_y, height, width) of the walking subject at ``tick``."""
        w, h = self.width, self.height
        span = int(w * 0.9)
        px = int(w * 0.05) + (tick * 6) % span
        py = int(h * 0.82)
        ph = int(h * 0.28)
        pw = ph // 3
        return px, py, ph, pw

    def _render_rgb(self, tick: int, eye_shift: int = 0) -> np.ndarray:
        """Render one RGB frame. ``eye_shift`` > 0 shifts near foreground LEFT to
        synthesize the right eye of a stereo pair (near objects have the largest
        parallax; the far background/ground carry ~0 disparity)."""
        w, h = self.width, self.height
        frame = np.full((h, w, 3), 60, dtype=np.uint8)
        ground_y = int(h * 0.55)
        frame[ground_y:] = (86, 82, 78)  # RGB ground
        cv2 = self._cv2
        span = int(w * 0.9)
        # Static wall block for scene texture (background depth -> ~0 disparity).
        if cv2 is not None:
            cv2.rectangle(frame, (int(w * 0.08), int(h * 0.2)),
                          (int(w * 0.22), ground_y), (74, 70, 70), -1)
        # Subject walks left->right across the lower third, looping (near -> full
        # disparity in the right eye).
        px, py, ph, pw = self._subject_geom(tick)
        px = px - eye_shift
        if cv2 is not None:
            cv2.rectangle(frame, (px - pw // 2, py - ph), (px + pw // 2, py),
                          (40, 34, 32), -1)
            cv2.circle(frame, (px, py - ph - pw // 2), pw // 2, (38, 32, 30), -1)
            if self.with_vehicle:
                # Vehicle sits at mid-depth -> half the foreground disparity.
                cx = int(w * 0.95) - (tick * 9) % span - eye_shift // 2
                cy = int(h * 0.62)
                cw, ch = int(w * 0.14), int(h * 0.09)
                cv2.rectangle(frame, (cx - cw, cy - ch), (cx + cw, cy + ch),
                              (52, 46, 44), -1)
        else:
            x0 = max(0, px - pw // 2)
            x1 = min(w, px + pw // 2)
            frame[py - ph:py, x0:x1] = (40, 34, 32)
        return frame

    def _depth_metres(self, tick: int) -> np.ndarray:
        """A plausible per-pixel depth (metres) for the synthetic scene: the
        ground recedes to the horizon, a mid-depth wall, and a near subject."""
        w, h = self.width, self.height
        ground_y = int(h * 0.55)
        depth = np.full((h, w), 55.0, dtype=np.float32)   # sky / background: far
        rows = np.arange(ground_y, h)
        frac = (rows - ground_y) / max(1, (h - 1 - ground_y))  # 0 horizon .. 1 near
        depth[ground_y:h, :] = (40.0 - frac * 38.0)[:, None]   # 40 m -> 2 m
        # Static wall at mid depth.
        depth[int(h * 0.2):ground_y, int(w * 0.08):int(w * 0.22)] = 15.0
        # Near subject.
        px, py, ph, pw = self._subject_geom(tick)
        x0, x1 = max(0, px - pw // 2), min(w, px + pw // 2)
        y0, y1 = max(0, py - ph), min(h, py)
        depth[y0:y1, x0:x1] = 3.0
        return depth

    def get_frame(self) -> np.ndarray:
        self._render_tick = self._tick
        frame = self._render_rgb(self._tick, eye_shift=0)
        self._tick += 1
        return frame

    def get_depth(self) -> np.ndarray | None:
        if not self.with_depth:
            return None
        return colorize_depth(self._depth_metres(self._render_tick))

    def get_right_frame(self) -> np.ndarray | None:
        if not self.with_stereo:
            return None
        return self._render_rgb(self._render_tick, eye_shift=self.baseline_px)


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
    # Kit's update loop is main-thread-only — see FrameSource.requires_main_thread.
    requires_main_thread = True

    def __init__(self, width, height, cam_pos, cam_target, scene_usd=None,
                 physics_hz=30, with_depth=False, with_stereo=False,
                 stereo_baseline=0.12):
        self.width = width
        self.height = height
        self.with_depth = with_depth
        self.with_stereo = with_stereo
        self.stereo_baseline = stereo_baseline
        self._sim = None
        self._annot = None
        self._depth_annot = None
        self._annot_right = None
        self._world = None
        self._subject = None
        self._t0 = time.time()
        self._boot(cam_pos, cam_target, scene_usd, physics_hz)

    @staticmethod
    def _right_eye_position(cam_pos, cam_target, baseline_m):
        """Left-cam position offset by ``baseline_m`` along the camera's right
        axis (Z-up scene), giving a rectified-ish stereo pair aimed at the same
        target. Physics-engine agnostic — this only touches camera transforms,
        so it holds under Newton or any other backend (no PhysX assumptions)."""
        p = np.asarray(cam_pos, dtype=np.float64)
        t = np.asarray(cam_target, dtype=np.float64)
        fwd = t - p
        n = float(np.linalg.norm(fwd))
        if n < 1e-9:
            return list(p)
        fwd /= n
        up = np.array([0.0, 0.0, 1.0])   # Z-up (matches the scene up-axis below)
        right = np.cross(fwd, up)
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
        return list(p + right * float(baseline_m))

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
        # (capability 8) Optional DEPTH: distance-to-image-plane annotator on the
        # SAME render product — colorized to RGB in get_depth().
        if self.with_depth:
            self._depth_annot = rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane"
            )
            self._depth_annot.attach([rp])
        # (capability 8) Optional STEREO: a second posed camera to the right of
        # the first, aimed at the same target -> a left/right pair.
        if self.with_stereo:
            right_pos = self._right_eye_position(cam_pos, cam_target, self.stereo_baseline)
            cam_r = rep.create.camera(position=tuple(right_pos), look_at=tuple(cam_target))
            rp_r = rep.create.render_product(cam_r, (self.width, self.height))
            self._annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
            self._annot_right.attach([rp_r])
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
        return self._rgba_to_rgb(self._annot.get_data())

    @staticmethod
    def _rgba_to_rgb(data) -> np.ndarray:
        rgba = np.asarray(data, dtype=np.uint8)
        if rgba.ndim == 3 and rgba.shape[2] == 4:
            return np.ascontiguousarray(rgba[:, :, :3])
        return rgba

    def get_depth(self) -> np.ndarray | None:
        # The depth annotator was filled by the orchestrator.step in the
        # preceding get_frame — read + colorize it (same instant as the RGB).
        if self._depth_annot is None:
            return None
        data = self._depth_annot.get_data()
        if data is None:
            return None
        return colorize_depth(np.asarray(data, dtype=np.float32))

    def get_right_frame(self) -> np.ndarray | None:
        # The right camera renders in the same orchestrator.step as the left.
        if self._annot_right is None:
            return None
        data = self._annot_right.get_data()
        if data is None:
            return None
        return self._rgba_to_rgb(data)

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
    """Shared latest-frame holder + camera metadata for the HTTP handlers.

    Serves one or more channels — always ``"main"`` (RGB), plus optional
    ``"depth"`` and ``"right"`` (stereo) when the source provides them. Each
    channel keeps its own latest JPEG so the extra MJPEG routes are independent.
    """

    def __init__(self, source: FrameSource, meta: dict, fps: int, encoder,
                 channels=("main",)):
        self.source = source
        self.meta = meta
        self.fps = max(1, fps)
        self.encode = encoder
        self.channels = tuple(channels)
        self._latest: dict[str, bytes] = {c: b"" for c in self.channels}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.frames = 0

    def start(self):
        """Render in a BACKGROUND thread. Only legal for thread-safe sources —
        an Isaac/Kit source stepped off the main thread deadlocks silently, so
        we refuse loudly rather than serve an eternally empty stream."""
        if getattr(self.source, "requires_main_thread", False):
            raise RuntimeError(
                f"source {self.source.name!r} must be rendered on the main thread "
                "(Omniverse Kit deadlocks in a worker); use run_main_thread() and "
                "serve HTTP from the background thread instead"
            )
        threading.Thread(target=self._loop, daemon=True, name="isaac-cam-render").start()

    def render_once(self) -> bool:
        """Render + encode ONE instant across all channels on the CALLING
        thread, publishing to the latest-frame holder. Returns True on success.

        This is the main-thread pump used for Isaac; :meth:`_loop` is the same
        step plus pacing, for background-thread sources."""
        try:
            encoded = {}
            for name, rgb in self._render_channels().items():
                jpeg = self.encode(rgb)
                if jpeg:
                    encoded[name] = jpeg
            if encoded.get("main"):
                with self._lock:
                    self._latest.update(encoded)
                    self.frames += 1
                return True
            return False
        except Exception as exc:
            log.warning("frame render failed: %s", exc)
            return False

    def run_main_thread(self):
        """Drive rendering on the CALLING (main) thread until stopped. Blocks —
        the HTTP server runs in the background thread under this arrangement."""
        interval = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            if not self.render_once():
                self._stop.wait(0.5)
            dt = time.time() - t0
            if dt < interval:
                self._stop.wait(interval - dt)

    def _render_channels(self) -> dict:
        """Render every enabled channel for the current instant. ``main`` first
        (it advances the source's animation clock); depth/right reuse it."""
        out = {"main": self.source.get_frame()}
        if "depth" in self.channels:
            d = self.source.get_depth()
            if d is not None:
                out["depth"] = d
        if "right" in self.channels:
            r = self.source.get_right_frame()
            if r is not None:
                out["right"] = r
        return out

    def _loop(self):
        # Background-thread variant — identical step, same pacing, so the two
        # render paths cannot drift apart.
        self.run_main_thread()

    def latest(self, channel: str = "main") -> bytes:
        with self._lock:
            return self._latest.get(channel, b"")

    def stop(self):
        self._stop.set()
        self.source.close()


#: Streaming routes -> channel, longest-prefix first so `/mjpeg_depth` and
#: `/mjpeg_right` are never swallowed by the bare `/mjpeg` prefix.
_STREAM_ROUTES = (
    ("/mjpeg_right", "right"),
    ("/mjpeg_depth", "depth"),
    ("/depth", "depth"),
    ("/mjpeg", "main"),
)


def resolve_channel(path: str) -> str | None:
    """Map a request path to a channel name, or None when it is not a stream
    route. Pure — the routing table is unit-testable without a socket."""
    if path == "/":
        return "main"
    for prefix, channel in _STREAM_ROUTES:
        if path.startswith(prefix):
            return channel
    return None


def _make_handler(state: CameraState):
    import json

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _channel_from_query(self, default: str = "main") -> str:
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            ch = (q.get("channel") or [default])[0]
            return ch if ch in state.channels else default

        def _stream_channel(self, channel: str):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()
            try:
                interval = 1.0 / state.fps
                while True:
                    jpeg = state.latest(channel)
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

        def do_GET(self):
            if self.path.startswith("/status"):
                body = json.dumps({
                    **state.meta, "frames": state.frames,
                    "source": state.source.name, "status": "online",
                    "channels": list(state.channels),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/snapshot"):
                jpeg = state.latest(self._channel_from_query())
                if not jpeg:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif (channel := resolve_channel(self.path)) is not None:
                self._stream_channel(channel)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def _channels_for(args) -> tuple:
    """The enabled channels (in serve order) implied by the CLI flags."""
    channels = ["main"]
    if getattr(args, "depth", False):
        channels.append("depth")
    if getattr(args, "stereo", False):
        channels.append("right")
    return tuple(channels)


def _build_source(args) -> FrameSource:
    if args.source == "isaac":
        cam_pos = [args.cam_x, args.cam_y, args.cam_z]
        cam_target = [args.target_x, args.target_y, args.target_z]
        return IsaacFrameSource(
            args.width, args.height, cam_pos, cam_target,
            scene_usd=args.scene or None, physics_hz=args.physics_hz,
            with_depth=args.depth, with_stereo=args.stereo,
            stereo_baseline=args.stereo_baseline,
        )
    return SyntheticFrameSource(
        args.width, args.height, with_vehicle=args.with_vehicle,
        with_depth=args.depth, with_stereo=args.stereo,
    )


def selftest(args) -> int:
    """No-GPU: render N synthetic frames (RGB + depth + stereo-right), assert
    they encode, and prove the stereo right eye differs from the left."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    encode, enc_name = _make_jpeg_encoder()
    src = SyntheticFrameSource(args.width, args.height, with_vehicle=True,
                               with_depth=True, with_stereo=True)
    sizes, depth_sizes, right_sizes = [], [], []
    shape = (args.height, args.width, 3)
    for _ in range(args.selftest_frames):
        rgb = src.get_frame()
        assert rgb.shape == shape, f"bad frame shape {rgb.shape}"
        jpeg = encode(rgb)
        assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9", "not a JPEG"
        sizes.append(len(jpeg))
        # DEPTH (capability 8): colorized depth ramp, encodable like any frame.
        depth = src.get_depth()
        assert depth is not None and depth.shape == shape, "bad depth frame"
        dj = encode(depth)
        assert dj[:2] == b"\xff\xd8" and dj[-2:] == b"\xff\xd9", "depth not a JPEG"
        depth_sizes.append(len(dj))
        # STEREO (capability 8): right eye, parallax-shifted from the left.
        right = src.get_right_frame()
        assert right is not None and right.shape == shape, "bad right frame"
        assert not np.array_equal(right, rgb), "stereo right must differ from left"
        rj = encode(right)
        assert rj[:2] == b"\xff\xd8" and rj[-2:] == b"\xff\xd9", "right not a JPEG"
        right_sizes.append(len(rj))
    print(f"SELFTEST OK frames={len(sizes)} encoder={enc_name} "
          f"jpeg_bytes~{int(np.mean(sizes))} depth~{int(np.mean(depth_sizes))} "
          f"right~{int(np.mean(right_sizes))} res={args.width}x{args.height} "
          f"channels=main,depth,right")
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
    # (capability 8) Depth + stereo extra channels — synthetic-safe (no GPU);
    # under --source isaac they wire the real annotators / a second camera.
    ap.add_argument("--depth", action="store_true",
                    help="serve a colorized DEPTH channel at /depth")
    ap.add_argument("--stereo", action="store_true",
                    help="serve a right-eye STEREO channel at /mjpeg_right")
    ap.add_argument("--stereo-baseline", type=float, default=0.12,
                    help="isaac stereo right-eye offset in metres (default 0.12)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-frames", type=int, default=12)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.selftest:
        return selftest(args)

    encode, enc_name = _make_jpeg_encoder()
    source = _build_source(args)
    channels = _channels_for(args)
    meta = {
        "camera_id": args.camera_id,
        "lat": args.lat, "lng": args.lng, "heading": args.heading,
        "fov_angle": args.fov, "fov_range": args.range_m,
        "width": args.width, "height": args.height,
    }
    state = CameraState(source, meta, args.fps, encode, channels=channels)
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))

    # Which thread owns rendering vs HTTP depends on the source. Isaac/Kit must
    # be pumped from the MAIN thread, so there the HTTP server is the one that
    # moves to the background; thread-safe sources keep the classic split.
    main_thread_render = getattr(source, "requires_main_thread", False)
    if main_thread_render:
        threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="isaac-cam-http").start()
    else:
        state.start()

    def _shutdown(*_a):
        log.info("shutting down")
        state.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "ISAAC CAMERA READY source=%s encoder=%s http://%s:%d/mjpeg id=%s "
        "channels=%s render=%s pose=(%.4f,%.4f h=%.0f fov=%.0f r=%.0f)",
        source.name, enc_name, args.host, args.port, args.camera_id,
        ",".join(channels), "main-thread" if main_thread_render else "background",
        args.lat, args.lng, args.heading, args.fov, args.range_m,
    )
    try:
        if main_thread_render:
            state.run_main_thread()   # blocks; HTTP is already serving
        else:
            httpd.serve_forever()
    finally:
        state.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
