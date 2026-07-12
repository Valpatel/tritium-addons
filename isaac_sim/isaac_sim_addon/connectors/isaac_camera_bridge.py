# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Bridge an Isaac robot's onboard camera into Tritium's camera_feeds.

An Isaac-simulated (or real) robot renders an RGB frame; this bridge JPEG-encodes
it and publishes it on the exact MQTT topic the Command Center's ``camera_feeds``
plugin consumes — ``tritium/{site}/cameras/{cam_id}/frame`` — so the robot's view
shows up as a live feed in Tritium (UI MJPEG + frame detection), keyed to the
unit's id. Optionally runs :func:`tritium_lib.perception.build_frame_detector`
locally and publishes detections on ``.../detections`` in the shape
``camera_feeds`` expects (``label``/``confidence``/``center_x``/``center_y``),
which the ``MQTTSource`` forwards to the ``TargetTracker`` (targets on the map).

Deliberately importable with **no** ``isaacsim`` dependency — it only handles
``numpy`` frames, JPEG, and MQTT — so it unit-tests headless. Grabbing the frame
from a live Isaac render product is a separate concern (:meth:`isaac_camera_rgb`,
which lazily imports Isaac only when called).

Wire (matches ``tritium-sc/plugins/camera_feeds``):
    Isaac camera --RGB--> IsaacCameraBridge --JPEG--> MQTT
        tritium/{site}/cameras/{cam_id}/frame --> MQTTSource.on_frame
        --> camera_feeds (UI MJPEG) [+ detections --> TargetTracker]
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:  # optional at import time; required only to encode/decode
    import cv2
except Exception:  # pragma: no cover - cv2 always present in the tritium env
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


def frame_topic(site: str, cam_id: str) -> str:
    """The camera_feeds JPEG-frame topic for a unit's camera."""
    return f"tritium/{site}/cameras/{cam_id}/frame"


def detection_topic(site: str, cam_id: str) -> str:
    """The camera_feeds detection-event topic for a unit's camera."""
    return f"tritium/{site}/cameras/{cam_id}/detections"


def encode_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes | None:
    """Encode an HxWx3 **RGB** ndarray to JPEG bytes (BGR on the wire, as OpenCV /
    ``camera_feeds`` decode with ``cv2.imdecode(..., IMREAD_COLOR)``)."""
    if cv2 is None:
        raise RuntimeError("cv2 required to encode frames")
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        bgr = rgb[:, :, 2::-1]  # RGB(A) -> BGR
    else:
        bgr = rgb
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


class IsaacCameraBridge:
    """Publishes a unit's camera frames (and optional detections) to Tritium.

    Args:
        cam_id: The unit / camera id — the ``cam_id`` the ``camera_feeds``
            ``MQTTSource`` is configured with. Keep it == the robot's unit id so
            the operator can tie the feed to the unit in the map/unit-list.
        site: MQTT site segment (default ``"isaac"``).
        mqtt_client: an already-connected paho client, or ``None`` to build one
            in :meth:`connect`.
        host/port: broker to connect to when ``mqtt_client`` is ``None``.
        jpeg_quality: 1..100.
        run_perception: build a local frame detector and publish detections.
        detector_prefer: forwarded to ``build_frame_detector`` (``auto``/``yolo``/
            ``onnx``/``motion``).
    """

    def __init__(
        self,
        cam_id: str,
        site: str = "isaac",
        *,
        mqtt_client: Any = None,
        host: str = "127.0.0.1",
        port: int = 1883,
        jpeg_quality: int = 80,
        run_perception: bool = False,
        detector_prefer: str = "auto",
    ) -> None:
        self.cam_id = cam_id
        self.site = site
        self.frame_topic = frame_topic(site, cam_id)
        self.detection_topic = detection_topic(site, cam_id)
        self.jpeg_quality = jpeg_quality
        self._client = mqtt_client
        self._own_client = mqtt_client is None
        self._host = host
        self._port = port
        self.frames_published = 0
        self.detections_published = 0
        self._detector = None
        if run_perception:
            from tritium_lib.perception.detector import build_frame_detector

            self._detector = build_frame_detector(prefer=detector_prefer)
            logger.info("perception backend: %s", getattr(self._detector, "backend_name", "?"))

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> "IsaacCameraBridge":
        if self._client is None:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client()
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()
        return self

    def close(self) -> None:
        if self._client is not None and self._own_client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass

    # -- publish -------------------------------------------------------------
    def publish_frame(self, rgb: np.ndarray) -> dict:
        """JPEG-encode ``rgb`` and publish it on the camera_feeds frame topic.

        Returns a small summary dict (topic, byte count, detections if enabled).
        """
        jpeg = encode_jpeg(rgb, self.jpeg_quality)
        if jpeg is None:
            return {"ok": False, "reason": "encode_failed"}
        if self._client is not None:
            self._client.publish(self.frame_topic, jpeg)
        self.frames_published += 1
        out = {"ok": True, "topic": self.frame_topic, "bytes": len(jpeg)}
        if self._detector is not None:
            out["detections"] = self.publish_detections(rgb)
        return out

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Run the local detector and return detections in camera_feeds shape:
        ``{label, confidence, center_x, center_y}`` (centres normalised 0..1)."""
        if self._detector is None:
            return []
        h, w = rgb.shape[:2]
        bgr = rgb[:, :, 2::-1] if (rgb.ndim == 3 and rgb.shape[2] >= 3) else rgb
        dets = self._detector.detect(bgr, source_id=self.cam_id)
        out: list[dict] = []
        for d in dets:
            bb = getattr(d, "bbox", None)
            cx = (bb.x + bb.w / 2.0) / w if bb and w else 0.5
            cy = (bb.y + bb.h / 2.0) / h if bb and h else 0.5
            out.append({
                "label": getattr(d, "label", getattr(d, "class_name", "object")),
                "confidence": float(getattr(d, "confidence", 0.5)),
                "center_x": float(cx),
                "center_y": float(cy),
            })
        return out

    def publish_detections(self, rgb: np.ndarray) -> list[dict]:
        import json

        dets = self.detect(rgb)
        if dets and self._client is not None:
            self._client.publish(self.detection_topic, json.dumps(dets))
            self.detections_published += len(dets)
        return dets

    # -- Isaac frame grab (lazy Isaac import; used only inside a live sim) ----
    @staticmethod
    def isaac_camera_rgb(render_product_path_or_annotator: Any) -> np.ndarray | None:
        """Grab an RGB frame from an Isaac render product / rgb annotator.

        Kept separate + lazily imported so the rest of this module has no Isaac
        dependency. Pass either an ``omni.replicator`` rgb annotator (has
        ``get_data``) or a render-product path (an annotator is attached).
        """
        ann = render_product_path_or_annotator
        if hasattr(ann, "get_data"):
            data = ann.get_data()
        else:  # a render-product path -> attach an rgb annotator
            import omni.replicator.core as rep

            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([ann])
            data = annot.get_data()
        if data is None:
            return None
        arr = np.asarray(data)
        return arr[:, :, :3] if arr.ndim == 3 and arr.shape[2] == 4 else arr
