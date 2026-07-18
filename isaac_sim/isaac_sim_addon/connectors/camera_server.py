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
  * ``/depth16``           ``--depth16``: the METRIC depth stream — lossless
                           uint16-millimetre PNG (ROS ``16UC1``, 0 = no return),
                           the same encoding RealSense/ZED/``depth_image_proc``
                           speak.  This is the channel a perception consumer
                           needs; ``/depth`` is only viewable (a colormap cannot
                           be turned back into metres).
  * ``/intrinsics``        JSON pinhole model (fx/fy/cx/cy from ``--hfov``) —
                           what a consumer needs to unproject ``/depth16``.
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

#: ``depth16`` wire contract — uint16 PNG, millimetres, 0 = no return. This is
#: ROS ``16UC1``, the encoding RealSense / ZED / Kinect / ``depth_image_proc``
#: already share, so an Isaac depth frame and a real ZED frame reach perception
#: in the same units with the same holes.
DEPTH16_SCALE = 1000.0          # units per metre (millimetres)
_DEPTH16_MAX = 65535            # ~65.5 m at mm scale


def encode_depth16(depth_m: np.ndarray) -> bytes:
    """Encode a HxW float depth (METRES) as a lossless uint16-millimetre PNG.

    Deliberately written out in full rather than imported from
    ``tritium_lib.perception.depth_codec``, which is the CANONICAL definition of
    this format: connectors run inside Isaac's python and must stay tritium-free
    (the dependency-hygiene gate), exactly as the gait contract is mirrored from
    ``tritium_lib.models.gait_trajectory``.  The two sides are held together by a
    contract test that round-trips this encoder through the lib decoder — if
    they ever drift, that test fails rather than an operator silently reading
    wrong ranges.

    ``NaN``/``inf``/non-positive become the 0 no-return sentinel; values past the
    ceiling CLAMP (a wrapped 70 m sky pixel reading as 4 m would put a phantom
    contact in the operator's lap).
    """
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be HxW metres, got shape {d.shape}")

    invalid = ~np.isfinite(d) | (d <= 0.0)
    d = np.where(invalid, 0.0, d)
    units = np.clip(np.rint(d * DEPTH16_SCALE), 0, _DEPTH16_MAX)
    # A valid-but-sub-millimetre reading must not round down into the hole
    # sentinel and masquerade as "no return".
    units[(units == 0) & ~invalid] = 1
    units[invalid] = 0
    return _encode_png16(units.astype(np.uint16))


def _encode_png16(units: np.ndarray) -> bytes:
    """16-bit single-channel PNG via cv2, else Pillow — same optional-codec
    pattern as the JPEG encoder above, so this works in Isaac's python."""
    try:
        import cv2

        ok, buf = cv2.imencode(".png", units)
        if not ok:
            raise RuntimeError("cv2 failed to encode the depth PNG")
        return buf.tobytes()
    except ImportError:
        pass
    from PIL import Image

    bio = io.BytesIO()
    Image.fromarray(units, mode="I;16").save(bio, format="PNG")
    return bio.getvalue()


def depth16_available() -> bool:
    """True when a 16-bit PNG encoder is present, i.e. the metric channel can
    actually serve bytes. Checked at startup so a missing codec fails loudly
    instead of advertising a channel that never produces a frame."""
    try:
        _encode_png16(np.zeros((2, 2), dtype=np.uint16))
    except Exception:
        return False
    return True


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
# Mounting a camera ON a body (capability 8b).
# --------------------------------------------------------------------------- #
#
# A wall camera's pose is a constant.  A robot's camera is a function of body
# pose, so the prim is PARENTED under the body and carries only a body-frame
# offset plus a boresight orientation; USD then composes the world pose for
# free, every frame, with no pose plumbing in this process.
#
# Both helpers below are duplicated from ``tritium_lib.geo.camera_mount`` on
# purpose — connectors run inside Isaac's python, which has no tritium on its
# path (see test_connectors_do_not_import_tritium).  The drift between the two
# copies is pinned by test_mount_offset_contract_matches_the_lib_camera_mount:
# if the render and the map's FOV cone ever disagree, that test goes red first.

def mount_stage_offset(forward_m: float = 0.0, left_m: float = 0.0,
                       up_m: float = 0.0) -> tuple:
    """The mount offset as a Z-up USD stage translation (body axes -> XYZ)."""
    return (float(forward_m), float(left_m), float(up_m))


def mount_camera_quat(tilt_deg: float = 0.0) -> tuple:
    """Orientation quaternion ``(w, x, y, z)`` for a mounted camera prim.

    A USD camera looks down its own -Z with +Y up, but a nose camera must look
    down the body's +X with +Z up — so the prim needs a fixed rotation even at
    zero tilt.  ``tilt_deg`` is elevation, positive UP, applied about the body's
    left axis, matching ``CameraMount.tilt_deg``.

    Returned as a quaternion rather than an Euler triple deliberately:
    ``rotateXYZ`` ordering is a coin-flip that renders plausibly either way,
    whereas a basis built from the boresight is checkable one axis at a time.
    """
    t = math.radians(float(tilt_deg))
    # Body frame, Z-up: +X forward, +Y left, +Z up.
    fwd = (math.cos(t), 0.0, math.sin(t))          # boresight, tilted up
    left = (0.0, 1.0, 0.0)
    up = _cross(fwd, left)                          # level horizon at any tilt
    right = _cross(fwd, up)                         # == -left at zero tilt
    # Camera basis columns: X_cam = right, Y_cam = up, Z_cam = -boresight.
    m = (
        (right[0], up[0], -fwd[0]),
        (right[1], up[1], -fwd[1]),
        (right[2], up[2], -fwd[2]),
    )
    return _matrix_to_quat(m)


def _cross(a: tuple, b: tuple) -> tuple:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _matrix_to_quat(m) -> tuple:
    """Row-major 3x3 rotation matrix -> ``(w, x, y, z)``, Shepperd's method.

    Branching on the largest diagonal term avoids the near-zero divide that the
    naive trace formula hits at 180 deg — reachable here with a rear-facing
    mount."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return ((m[2][1] - m[1][2]) / s, 0.25 * s,
                (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s,
                0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s, 0.25 * s)


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

    def get_depth_metres(self) -> np.ndarray | None:
        """The METRIC depth for the current instant — HxW float32 metres, with
        ``inf``/``nan`` for no-return.  This, not :meth:`get_depth`, is the
        source of truth: a colorized depth image is many-to-one and JPEG is
        lossy, so range cannot be recovered from it.  Sources implement THIS."""
        return None

    def get_depth(self) -> np.ndarray | None:
        """The viewable (colorized) depth image, derived from the metric one so
        the two channels can never disagree about what they are showing."""
        metres = self.get_depth_metres()
        if metres is None:
            return None
        return colorize_depth(metres)

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

    def get_depth_metres(self) -> np.ndarray | None:
        if not self.with_depth:
            return None
        return self._depth_metres(self._render_tick)

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
                 stereo_baseline=0.12, mount_prim=None, mount_forward=0.0,
                 mount_left=0.0, mount_up=0.0, mount_tilt=0.0, body_spin_dps=0.0):
        self.width = width
        self.height = height
        self.with_depth = with_depth
        self.with_stereo = with_stereo
        self.stereo_baseline = stereo_baseline
        self.mount_prim = mount_prim
        self.mount_forward = mount_forward
        self.mount_left = mount_left
        self.mount_up = mount_up
        self.mount_tilt = mount_tilt
        self.body_spin_dps = body_spin_dps
        self._sim = None
        self._annot = None
        self._depth_annot = None
        self._annot_right = None
        self._world = None
        self._subject = None
        self._body_xform = None
        self._body_orient = None
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
        # Camera prim + render product + rgb annotator.  Two shapes: bolted to
        # the world at a fixed pose (the wall camera this server has always
        # served), or parented UNDER a body prim so USD composes the lens pose
        # from the body's every frame (capability 8b).
        if self.mount_prim:
            cam = self._mount_camera_prim(stage)
        else:
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

    def _mount_camera_prim(self, stage) -> str:
        """Define a camera prim parented under ``self.mount_prim`` and return
        its path.

        If the mount prim does not exist — the common case when this server
        boots its OWN SimulationApp rather than attaching to an already-populated
        stage — a stand-in body is created there: an Xform carrying a chassis
        box, so there is something for the lens to be rigidly attached TO.

        Note the chassis does NOT appear in frame at the default mount (0.30 m
        forward, 0.25 m up clears it).  The evidence that the lens really rides
        the body is therefore the sweep, not an occlusion: yaw the body and the
        rendered scene rotates with it, which a world-posed camera cannot do.
        """
        from pxr import Gf, UsdGeom  # type: ignore

        body_path = str(self.mount_prim)
        body = stage.GetPrimAtPath(body_path)
        if not body or not body.IsValid():
            body_xform = UsdGeom.Xform.Define(stage, body_path)
            body_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
            # quatd, not quatf: a Quatf handed to a quatd op is silently
            # DROPPED by USD (env defect recorded 2026-07-18).
            self._body_orient = body_xform.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble
            )
            self._body_orient.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            self._body_xform = body_xform
            chassis = UsdGeom.Cube.Define(stage, body_path + "/chassis")
            chassis.CreateSizeAttr(1.0)
            chassis.AddScaleOp().Set(Gf.Vec3f(0.35, 0.15, 0.08))
            chassis.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.15, 0.35)])

        cam_path = body_path + "/tritium_cam"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        tx, ty, tz = mount_stage_offset(self.mount_forward, self.mount_left,
                                        self.mount_up)
        cam.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
        w, x, y, z = mount_camera_quat(self.mount_tilt)
        cam.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(w, Gf.Vec3d(x, y, z))
        )
        return cam_path

    def _spin_body(self) -> None:
        """Yaw the stand-in body so the mounted view actually sweeps.

        A mounted camera that never moves is indistinguishable from a wall
        camera in the resulting stream — the motion IS the demonstration."""
        if self._body_xform is None or self.body_spin_dps == 0.0:
            return
        from pxr import Gf  # type: ignore

        yaw = math.radians((time.time() - self._t0) * self.body_spin_dps)
        half = yaw * 0.5
        self._body_orient.Set(
            Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        )

    def get_frame(self) -> np.ndarray:
        import omni.replicator.core as rep  # type: ignore
        self._spin_body()
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

    def get_depth_metres(self) -> np.ndarray | None:
        # The depth annotator was filled by the orchestrator.step in the
        # preceding get_frame — read it at the same instant as the RGB.  Isaac's
        # distance_to_image_plane annotator is already in METRES; hand it over
        # unconverted and let the codec quantize once, at the wire.
        if self._depth_annot is None:
            return None
        data = self._depth_annot.get_data()
        if data is None:
            return None
        depth = np.asarray(data, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth

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
                 channels=("main",), hfov_deg: float = 60.0):
        self.source = source
        self.meta = meta
        self.fps = max(1, fps)
        self.encode = encoder
        self.channels = tuple(channels)
        self.hfov_deg = float(hfov_deg)
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
            for name, frame in self._render_channels().items():
                # depth16 is metric float metres and must go out LOSSLESS —
                # JPEG-ing it would destroy the very numbers it exists to carry.
                blob = encode_depth16(frame) if name == "depth16" else self.encode(frame)
                if blob:
                    encoded[name] = blob
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
        if "depth" in self.channels or "depth16" in self.channels:
            # Read the metric depth ONCE and derive both channels from it, so
            # the viewable ramp and the metric PNG always describe the same
            # instant and the same distances.
            metres = self.source.get_depth_metres()
            if metres is not None:
                if "depth" in self.channels:
                    out["depth"] = colorize_depth(metres)
                if "depth16" in self.channels:
                    out["depth16"] = metres
        if "right" in self.channels:
            r = self.source.get_right_frame()
            if r is not None:
                out["right"] = r
        return out

    def _loop(self):
        # Background-thread variant — identical step, same pacing, so the two
        # render paths cannot drift apart.
        self.run_main_thread()

    def intrinsics(self) -> dict:
        """The pinhole intrinsics implied by the frame size + horizontal FOV.

        A square-pixel pinhole is the right model here: Isaac renders through an
        ideal lens with no distortion, so ``fx = (w/2) / tan(hfov/2)`` is exact
        rather than an approximation, and ``fy == fx``.  Shape matches
        ``tritium_lib.perception.CameraIntrinsics`` so a consumer can build one
        straight from this JSON.
        """
        w = int(self.meta.get("width", 0))
        h = int(self.meta.get("height", 0))
        fx = (w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0) if w else 0.0
        return {
            "width": w, "height": h,
            "fx": fx, "fy": fx,
            "cx": w / 2.0, "cy": h / 2.0,
            "hfov_deg": self.hfov_deg,
            "depth_scale": 1000.0,   # depth16 units per metre (millimetres)
            "depth_encoding": "16UC1_png",
        }

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
    # depth16 MUST precede /depth — otherwise the bare prefix swallows it and
    # an operator asking for metric depth silently receives a colormap.
    ("/depth16", "depth16"),
    ("/depth", "depth"),
    ("/mjpeg", "main"),
)

#: Per-channel wire MIME type. Everything is JPEG except the metric depth
#: channel, which is a lossless 16-bit PNG.
_CHANNEL_MIME = {"depth16": "image/png"}


def channel_mime(channel: str) -> str:
    return _CHANNEL_MIME.get(channel, "image/jpeg")


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
            if channel not in state.channels:
                self.send_response(404)
                self.end_headers()
                return
            mime = channel_mime(channel).encode()
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()
            try:
                interval = 1.0 / state.fps
                while True:
                    blob = state.latest(channel)
                    if blob:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: " + mime + b"\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(blob)}\r\n\r\n".encode()
                        )
                        self.wfile.write(blob)
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
            elif self.path.startswith("/intrinsics"):
                # The pinhole model a depth consumer needs to unproject
                # depth16 into camera-frame XYZ. Without this the metric
                # channel is just numbers with no geometry attached.
                body = json.dumps(state.intrinsics()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/snapshot"):
                channel = self._channel_from_query()
                blob = state.latest(channel)
                if not blob:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", channel_mime(channel))
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
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
    # --depth16 implies depth capture; it is the metric sibling of the same
    # annotator, so asking for it alone is legal (viewable ramp not required).
    if getattr(args, "depth16", False):
        channels.append("depth16")
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
            with_depth=args.depth or args.depth16, with_stereo=args.stereo,
            stereo_baseline=args.stereo_baseline,
            mount_prim=args.mount_prim or None,
            mount_forward=args.mount_forward, mount_left=args.mount_left,
            mount_up=args.mount_up, mount_tilt=args.mount_tilt,
            body_spin_dps=args.body_spin_dps,
        )
    return SyntheticFrameSource(
        args.width, args.height, with_vehicle=args.with_vehicle,
        with_depth=args.depth or args.depth16, with_stereo=args.stereo,
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
    # Capability 8b — mount the camera ON a body instead of on a wall. The
    # offsets are body-frame metres and MUST match the CameraMount the operator
    # side uses for the FOV cone, or image and cone drift apart.
    ap.add_argument("--mount-prim", default="",
                    help="parent the render camera under this prim (e.g. "
                         "/World/Go2); created as a stand-in body if absent")
    ap.add_argument("--mount-forward", type=float, default=0.30,
                    help="mount offset out the body's nose, metres")
    ap.add_argument("--mount-left", type=float, default=0.0)
    ap.add_argument("--mount-up", type=float, default=0.25)
    ap.add_argument("--mount-tilt", type=float, default=-10.0,
                    help="boresight elevation, positive UP")
    ap.add_argument("--body-spin-dps", type=float, default=0.0,
                    help="yaw the stand-in body at this rate so the mounted "
                         "view sweeps; 0 leaves the body still")
    ap.add_argument("--attach-to", default="",
                    help="tracked target id this camera rides; served in "
                         "/meta so SC can register the feed with attach_to")
    ap.add_argument("--with-vehicle", action="store_true")
    # (capability 8) Depth + stereo extra channels — synthetic-safe (no GPU);
    # under --source isaac they wire the real annotators / a second camera.
    ap.add_argument("--depth", action="store_true",
                    help="serve a colorized DEPTH channel at /depth")
    ap.add_argument("--depth16", action="store_true",
                    help="serve METRIC depth at /depth16 — lossless uint16-mm "
                         "PNG (ROS 16UC1). This is the channel a perception "
                         "consumer needs; /depth is only viewable")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="horizontal field of view in degrees, used to publish "
                         "pinhole intrinsics at /intrinsics (default 60)")
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
    # A body-mounted feed advertises the mount so SC can bind it with attach_to
    # and derive the cone from the SAME geometry the render used. Absent these
    # keys the feed is a wall camera and the operator's pose stays a constant.
    if args.attach_to or args.mount_prim:
        meta["mount"] = {
            "attach_to": args.attach_to or None,
            "prim": args.mount_prim or None,
            "forward_m": args.mount_forward, "left_m": args.mount_left,
            "up_m": args.mount_up, "tilt_deg": args.mount_tilt,
        }
    if "depth16" in channels and not depth16_available():
        # Fail at startup rather than advertise a metric channel that will
        # never produce a byte — the exact silent-healthy failure mode that
        # cost a tick when the Isaac render path served zero frames.
        ap.error("--depth16 needs a 16-bit PNG encoder (cv2 or Pillow) in the "
                 "render env — install one or drop --depth16")
    state = CameraState(source, meta, args.fps, encode, channels=channels,
                        hfov_deg=args.hfov)
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
