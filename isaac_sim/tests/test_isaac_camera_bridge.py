# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the Isaac-camera -> Tritium camera_feeds bridge.

Pin the ingest contract: the topic string and the JPEG encode/decode round-trip
that the Command Center's ``camera_feeds`` ``MQTTSource.on_frame`` performs, so a
robot's onboard camera frame reaches Tritium byte-faithfully. The broker test is
skipped when no MQTT broker is reachable (kept hermetic in CI).
"""
from __future__ import annotations

import socket

import cv2
import numpy as np
import pytest

from isaac_sim_addon.connectors.isaac_camera_bridge import (
    IsaacCameraBridge,
    detection_topic,
    encode_jpeg,
    frame_topic,
)


def _synthetic_rgb(h=64, w=96):
    rng = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    return rng


def test_topics_match_camera_feeds_contract():
    # tritium/{site}/cameras/{cam_id}/frame — the topic camera_feeds subscribes to
    assert frame_topic("isaac", "spot-01") == "tritium/isaac/cameras/spot-01/frame"
    assert detection_topic("site7", "dog9") == "tritium/site7/cameras/dog9/detections"


def test_encode_jpeg_roundtrips_through_camera_feeds_decode():
    rgb = _synthetic_rgb()
    jpeg = encode_jpeg(rgb, quality=90)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker
    # decode exactly as MQTTSource.on_frame does
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert frame is not None
    assert frame.shape == rgb.shape  # same HxWx3 survives the wire


def test_encode_handles_rgba():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 0] = 200  # red channel
    jpeg = encode_jpeg(rgba, quality=80)
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape == (32, 32, 3)
    # red (RGB) must land in the BGR red channel (index 2), not blue
    assert int(frame[..., 2].mean()) > int(frame[..., 0].mean())


def test_bridge_counts_without_client():
    br = IsaacCameraBridge("unit-1", site="isaac")  # no client -> publish is a no-op encode
    res = br.publish_frame(_synthetic_rgb())
    assert res["ok"] and res["topic"] == "tritium/isaac/cameras/unit-1/frame"
    assert br.frames_published == 1


def _broker(host="127.0.0.1", ports=(18883, 1883)):
    for p in ports:
        try:
            with socket.create_connection((host, p), timeout=0.5):
                return p
        except OSError:
            continue
    return None


@pytest.mark.skipif(_broker() is None, reason="no MQTT broker on 18883/1883")
def test_publish_roundtrip_over_broker():
    import time

    import paho.mqtt.client as mqtt

    port = _broker()
    rgb = _synthetic_rgb(120, 160)
    got = {}

    def on_msg(c, u, m):
        got["frame"] = cv2.imdecode(np.frombuffer(m.payload, np.uint8), cv2.IMREAD_COLOR)

    sub = mqtt.Client()
    sub.on_message = on_msg
    sub.connect("127.0.0.1", port, 60)
    sub.loop_start()
    sub.subscribe(frame_topic("isaac", "rt-cam"))
    time.sleep(0.3)

    br = IsaacCameraBridge("rt-cam", site="isaac", host="127.0.0.1", port=port).connect()
    br.publish_frame(rgb)
    time.sleep(0.5)
    br.close()
    sub.loop_stop()

    assert got.get("frame") is not None
    assert got["frame"].shape == rgb.shape
