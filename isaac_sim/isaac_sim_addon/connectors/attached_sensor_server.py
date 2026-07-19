#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Attached-mode sensors — cameras + LiDAR riding an ALREADY-RUNNING kit.

``camera_server`` and ``lidar_server`` each boot a PRIVATE ``SimulationApp``,
which means their pixels can — by construction — never show the body that the
dedicated Newton kit is stepping: USD stages are per-process, and a second kit
cannot attach to the first one's stage.  Every "camera watches the Newton
body" ambition dies on that boundary.

This module is the third shape.  It is IMPORTED INSIDE a running kit (sent in
via the MCP bridge's ``/execute``), authors its sensor prims under
``/World/Tritium/Sensors`` on the kit's OWN stage, encodes frames from the
kit's update loop (main thread — Kit is main-thread-only, and violating that
hangs silently; recorded gotcha), and serves the EXACT same wire contracts the
standalone servers serve:

  * camera port —  ``/mjpeg`` ``/depth`` ``/depth16`` ``/mjpeg_right``
                   ``/snapshot?channel=…`` ``/intrinsics`` ``/status``
  * lidar  port —  ``/scan`` ``/status``

so a consumer cannot tell (and must not care) whether it is talking to a
standalone sensor kit or to sensors riding the body's own kit.  The serving
machinery is REUSED, not mirrored: ``CameraState``/``LidarState`` + their
handlers come straight from the sibling modules, fed through their
``publish()`` seams; the sweep binning goes through
``lidar_server.flat_scan_to_ranges``.  One tested path, not two drifting
copies.

Good-neighbor rules (the kit may share a GPU with another tenant):
  * every prim lands under ``root`` (default ``/World/Tritium/Sensors``);
  * :func:`install` may *play* the timeline (physics + RTX sensors need it)
    but NOTHING here ever stops it — :func:`uninstall` removes prims and
    servers and leaves the clock alone.

Dependency hygiene: stdlib + numpy at import (unit-testable on any box);
``isaacsim``/``omni``/``pxr`` only inside :func:`install`; NEVER ``tritium``.

Run (from the driving box, against the dedicated Newton kit)::

    IsaacSimClient(port=8212).execute('''
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "tritium_attached_sensor_server",
            "<clone>/tritium-addons/isaac_sim/isaac_sim_addon/connectors/attached_sensor_server.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        print(m.install(cam_port=8130, lidar_port=8131))
    ''')

Loading by FILE PATH is deliberate: this build's kit caches stale modules by
bare name (recorded gotcha), and :func:`install`'s return carries
``provenance`` — the ``__file__`` of every module involved — precisely so a
harness can assert it is talking to the code it just synced.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np


def _load_sibling(name: str):
    """Import a sibling connector by FILE PATH, never by bare name.

    Two consumers, one constraint each: the kit caches stale modules by bare
    name (a stale ``camera_server`` cost a live session; recorded gotcha),
    and the no-GPU tests load connectors outside any package context.  A
    pinned path serves both — you always get THIS directory's module, and
    the path is assertable provenance."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    key = f"tritium_conn_{name}"
    mod = sys.modules.get(key)
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


_cam = _load_sibling("camera_server")
_lid = _load_sibling("lidar_server")


# --------------------------------------------------------------------------- #
# Pure geometry helpers (unit-tested without isaacsim).
# --------------------------------------------------------------------------- #

def usd_lookat_matrix(eye, target, up=(0.0, 0.0, 1.0)) -> list:
    """Row-major 4x4 camera-to-world matrix posing a USD camera at ``eye``
    looking at ``target`` in a Z-up world.

    USD cameras look down **-Z** with **+Y** up, and ``Gf.Matrix4d`` uses the
    row-vector convention — rows 0..2 are the camera's basis axes expressed
    in world coordinates, row 3 is the translation.  Feed the 16 values
    straight into ``Gf.Matrix4d(*flat)`` and set it as a ``transform`` op.
    """
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    fwd = t - e
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        raise ValueError("eye and target coincide — a camera needs a direction")
    fwd /= n
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        # Looking straight along ``up``: any horizontal right-axis serves.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn
    cam_z = -fwd                     # camera looks down its own -Z
    cam_y = np.cross(cam_z, right)   # orthonormal by construction
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = cam_y
    m[2, :3] = cam_z
    m[3, :3] = e
    return [[float(v) for v in row] for row in m]


def hfov_to_aperture_mm(hfov_deg: float, focal_mm: float = 18.0) -> float:
    """The horizontal aperture that gives ``hfov_deg`` at ``focal_mm`` —
    the pinhole identity ``hfov = 2*atan(aperture / (2*focal))`` inverted.
    Keeps the served ``/intrinsics`` (fx from hfov) and the rendered image in
    agreement, so a stereo consumer's ``fx*B/Z`` prediction lands on pixels."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov must be in (0, 180) deg, got {hfov_deg}")
    return 2.0 * focal_mm * math.tan(math.radians(hfov_deg) / 2.0)


def predicted_disparity_px(fx: float, baseline_m: float, depth_m: float) -> float:
    """Rectified-stereo disparity (px) for a point at ``depth_m`` — the number
    a live stereo pair must reproduce for its baseline to be believed."""
    if depth_m <= 0.0:
        raise ValueError(f"depth must be positive, got {depth_m}")
    return fx * baseline_m / depth_m


# --------------------------------------------------------------------------- #
# Stub sources — the holders' metadata/refusal halves, with no render loop.
# --------------------------------------------------------------------------- #

class AttachedFrameStub(_cam.FrameSource):
    """CameraState requires a FrameSource; in attached mode nothing ever
    PULLS frames through it — the kit's update callback pushes encoded blobs
    via ``CameraState.publish``.  ``get_frame`` therefore refuses loudly:
    reaching it means someone started the pull loop against a source that
    has no frames to pull, which would serve an eternally empty stream."""

    name = "isaac-attached"
    requires_main_thread = True

    def get_frame(self):  # pragma: no cover - defensive
        raise RuntimeError(
            "attached mode publishes frames from the kit's update loop; "
            "nothing may pull this source (use CameraState.publish)")

    def close(self) -> None:
        pass


class AttachedScanGeometry(_lid.ScanSource):
    """The LaserScan geometry the /scan payload and /status advertise, plus
    the warm-up flag — with the same no-pull refusal as the camera stub."""

    name = "isaac-attached"
    requires_main_thread = True

    def __init__(self, num_beams: int = 360, range_min: float = 0.1,
                 range_max: float = 30.0, angle_min: float = -math.pi):
        super().__init__(num_beams, range_min, range_max, angle_min)
        self.warmed = False

    @property
    def never_returned(self) -> bool:
        return not self.warmed

    def get_scan(self):  # pragma: no cover - defensive
        raise RuntimeError(
            "attached mode publishes sweeps from the kit's update loop; "
            "nothing may pull this source (use LidarState.publish)")


# --------------------------------------------------------------------------- #
# Module state — /execute gets a FRESH namespace per call, so continuity
# between install / status / uninstall lives here, in sys.modules.
# --------------------------------------------------------------------------- #

_STATE: dict = {"installed": False}


def provenance() -> dict:
    """The file paths actually loaded — assert these against the synced clone
    before believing anything else this module reports (stale-module gotcha)."""
    return {
        "attached_sensor_server": os.path.abspath(__file__),
        "camera_server": os.path.abspath(_cam.__file__),
        "lidar_server": os.path.abspath(_lid.__file__),
    }


def status() -> dict:
    """Counters a harness polls INSIDE the kit (the HTTP /status routes are
    the outside view; this is the producer's own ledger)."""
    out = {
        "installed": _STATE.get("installed", False),
        "pumps": _STATE.get("pumps", 0),
        "empty_reads": _STATE.get("empty_reads", 0),
        "errors": _STATE.get("errors", 0),
        "last_error": _STATE.get("last_error", ""),
        "lidar_error": _STATE.get("lidar_error", ""),
    }
    cam_state = _STATE.get("cam_state")
    if cam_state is not None:
        out["frames"] = cam_state.frames
    lid_state = _STATE.get("lid_state")
    if lid_state is not None:
        out["scans"] = lid_state.scans
    return out


def _require_kit():
    try:
        import omni.usd  # noqa: F401
        import omni.kit.app  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "attached_sensor_server.install() runs INSIDE a running Isaac "
            "Kit — send it in via the MCP bridge's /execute.  From plain "
            "python only the pure helpers are usable.") from exc


def _ensure_extension(ext_id: str) -> None:
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    if not mgr.is_extension_enabled(ext_id):
        # Immediate, not queued: the very next line will import from it.
        mgr.set_extension_enabled_immediate(ext_id, True)


def install(*, cam_port: int = 8130, lidar_port: int = 8131,
            host: str = "0.0.0.0", width: int = 640, height: int = 480,
            encode_fps: float = 10.0, hfov_deg: float = 70.0,
            cam_pos=(-4.0, 0.0, 1.0), cam_target=(6.0, 0.0, 1.0),
            with_depth: bool = True, with_stereo: bool = True,
            stereo_baseline: float = 0.30,
            with_lidar: bool = True, lidar_pos=(0.0, 0.0, 0.5),
            num_beams: int = 360, lidar_range_min: float = 0.1,
            lidar_range_max: float = 30.0,
            lidar_config: str = "Example_Rotary_2D",
            camera_id: str = "isaac-attached-cam",
            lidar_id: str = "isaac-attached-lidar",
            root: str = "/World/Tritium/Sensors",
            play_timeline: bool = False) -> dict:
    """Author sensors on the RUNNING kit's stage and start serving.

    Returns a summary dict (ports, channels, prim paths, provenance).  The
    LiDAR half degrades gracefully: any failure to stand the RTX sensor up is
    reported in ``lidar_error`` while the cameras keep serving — a camera
    wall with no sweep beats no evidence at all, and the absence is *stated*,
    never papered over.
    """
    if _STATE.get("installed"):
        raise RuntimeError("already installed — call uninstall() first")
    _require_kit()

    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("the running kit has no stage open")

    # ---- cameras --------------------------------------------------------- #
    def _author_cam(name: str, eye, target) -> str:
        path = f"{root}/{name}"
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(hfov_to_aperture_mm(hfov_deg, 18.0))
        # Square pixels: vertical aperture follows the frame's aspect ratio.
        cam.CreateVerticalApertureAttr(
            hfov_to_aperture_mm(hfov_deg, 18.0) * height / width)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
        flat = [v for row in usd_lookat_matrix(eye, target) for v in row]
        UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*flat))
        return path

    left_path = _author_cam("cam_left", cam_pos, cam_target)
    rp_left = rep.create.render_product(left_path, (width, height))
    ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    ann_rgb.attach([rp_left])

    ann_depth = None
    if with_depth:
        ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        ann_depth.attach([rp_left])

    ann_right = None
    right_path = ""
    if with_stereo:
        right_pos = _cam.IsaacFrameSource._right_eye_position(
            list(cam_pos), list(cam_target), stereo_baseline)
        right_path = _author_cam("cam_right", right_pos, cam_target)
        rp_right = rep.create.render_product(right_path, (width, height))
        ann_right = rep.AnnotatorRegistry.get_annotator("rgb")
        ann_right.attach([rp_right])

    # ---- lidar (graceful: cameras must survive a lidar failure) ---------- #
    ann_scan = None
    lidar_error = ""
    lidar_path = f"{root}/lidar"
    if with_lidar:
        try:
            _ensure_extension("isaacsim.sensors.rtx")
            import omni.kit.commands
            omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar", path=lidar_path, parent=None,
                config=lidar_config,
                translation=Gf.Vec3d(*[float(v) for v in lidar_pos]),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0))
            rp_lidar = rep.create.render_product(
                lidar_path, [1, 1], name="tritium_attached_lidar_rp")
            ann_scan = rep.AnnotatorRegistry.get_annotator(
                "IsaacComputeRTXLidarFlatScan")
            ann_scan.attach([rp_lidar])
        except Exception as exc:  # noqa: BLE001 - degrade, report, keep cameras
            ann_scan = None
            lidar_error = f"{type(exc).__name__}: {exc}"

    # ---- serving states (REUSED machinery, fed via publish()) ------------ #
    encoder, encoder_name = _cam._make_jpeg_encoder()
    channels = ["main"]
    if with_depth:
        channels += ["depth", "depth16"]
    if with_stereo:
        channels.append("right")
    meta = {
        "camera_id": camera_id, "width": width, "height": height,
        "fps": encode_fps, "mode": "attached", "encoder": encoder_name,
        "cam_prim": left_path, "right_prim": right_path,
    }
    cam_state = _cam.CameraState(
        AttachedFrameStub(), meta, int(encode_fps), encoder,
        channels=tuple(channels), hfov_deg=hfov_deg)
    scan_geom = AttachedScanGeometry(
        num_beams=num_beams, range_min=lidar_range_min,
        range_max=lidar_range_max)
    lid_state = _lid.LidarState(scan_geom, lidar_id, hz=int(encode_fps))

    # ---- timeline -------------------------------------------------------- #
    # Physics AND the RTX sensor only advance while the timeline plays.
    # Playing is additive and allowed; STOPPING is not ours to do, ever
    # (good-neighbor rule) — uninstall() leaves the clock untouched.
    timeline = omni.timeline.get_timeline_interface()
    timeline_started = False
    if play_timeline and not timeline.is_playing():
        timeline.play()
        timeline_started = True

    # ---- the pump: encode on the kit's update loop (main thread) --------- #
    interval = 1.0 / max(0.1, float(encode_fps))
    pace = {"next": 0.0}

    def _pump() -> None:
        rgba = np.asarray(ann_rgb.get_data())
        if rgba.size == 0:
            _STATE["empty_reads"] = _STATE.get("empty_reads", 0) + 1
            return
        encoded = {"main": encoder(_cam.IsaacFrameSource._rgba_to_rgb(rgba))}
        if ann_depth is not None:
            d = np.asarray(ann_depth.get_data(), dtype=np.float32)
            if d.size:
                if d.ndim == 3:
                    d = d[..., 0]
                # One metric read feeds both depth channels — the viewable
                # ramp and the metric PNG must describe the same instant.
                encoded["depth"] = encoder(_cam.colorize_depth(d))
                encoded["depth16"] = _cam.encode_depth16(d)
        if ann_right is not None:
            r = np.asarray(ann_right.get_data())
            if r.size:
                encoded["right"] = encoder(_cam.IsaacFrameSource._rgba_to_rgb(r))
        cam_state.publish(encoded)
        if ann_scan is not None:
            frame = ann_scan.get_data() or {}
            ranges, has_returns = _lid.flat_scan_to_ranges(
                frame, scan_geom.num_beams, scan_geom.angle_min,
                scan_geom.range_min, scan_geom.range_max)
            if has_returns:
                scan_geom.warmed = True
            if scan_geom.warmed:
                body = json.dumps(_lid.build_payload(
                    scan_geom, ranges, lidar_id, lid_state.scans)).encode()
                lid_state.publish(body)
        _STATE["pumps"] = _STATE.get("pumps", 0) + 1

    def _on_update(_event) -> None:
        now = time.time()
        if now < pace["next"]:
            return
        pace["next"] = now + interval
        try:
            _pump()
        except Exception as exc:  # noqa: BLE001 - a raising update callback
            # would take the whole app loop with it; count + expose instead.
            _STATE["errors"] = _STATE.get("errors", 0) + 1
            _STATE["last_error"] = f"{type(exc).__name__}: {exc}"

    subscription = (omni.kit.app.get_app().get_update_event_stream()
                    .create_subscription_to_pop(
                        _on_update, name="tritium.attached_sensor_server"))

    # ---- HTTP (background threads serving CACHED bytes only) ------------- #
    cam_httpd = ThreadingHTTPServer((host, cam_port), _cam._make_handler(cam_state))
    threading.Thread(target=cam_httpd.serve_forever, daemon=True,
                     name="attached-cam-http").start()
    lid_httpd = None
    if with_lidar:
        # Served even when the sensor failed to stand up: /scan answers 503
        # and /status shows scans=0 — an honest absence a poller can see,
        # instead of a connection refused it must guess about.
        lid_httpd = ThreadingHTTPServer((host, lidar_port),
                                        _lid._make_handler(lid_state))
        threading.Thread(target=lid_httpd.serve_forever, daemon=True,
                         name="attached-lidar-http").start()

    _STATE.update({
        "installed": True, "pumps": 0, "empty_reads": 0, "errors": 0,
        "last_error": "", "lidar_error": lidar_error,
        "cam_state": cam_state, "lid_state": lid_state,
        "cam_httpd": cam_httpd, "lid_httpd": lid_httpd,
        "subscription": subscription, "root": root,
        "annotators": [a for a in (ann_rgb, ann_depth, ann_right, ann_scan)
                       if a is not None],
        "timeline_started": timeline_started,
    })
    return {
        "ok": True, "cam_port": cam_port,
        "lidar_port": lidar_port if with_lidar else None,
        "channels": channels, "camera_prims": [left_path, right_path],
        "lidar_prim": lidar_path if with_lidar else "",
        "lidar_error": lidar_error, "encoder": encoder_name,
        "timeline_started": timeline_started,
        "provenance": provenance(),
    }


def uninstall() -> dict:
    """Stop serving, drop the update subscription, remove OUR prims.

    Deliberately never touches the timeline: whether it was playing before
    install() or because of it, stopping the clock in a shared kit is the
    one act a good neighbor cannot take back."""
    if not _STATE.get("installed"):
        return {"ok": True, "was_installed": False}
    for key in ("cam_httpd", "lid_httpd"):
        httpd = _STATE.get(key)
        if httpd is not None:
            threading.Thread(target=httpd.shutdown, daemon=True).start()
    _STATE["subscription"] = None          # dropping the ref unsubscribes
    for ann in _STATE.get("annotators", []):
        try:
            ann.detach()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            stage.RemovePrim(_STATE.get("root", "/World/Tritium/Sensors"))
    except Exception:  # noqa: BLE001
        pass
    pumps = _STATE.get("pumps", 0)
    _STATE.clear()
    _STATE["installed"] = False
    return {"ok": True, "was_installed": True, "pumps": pumps}
