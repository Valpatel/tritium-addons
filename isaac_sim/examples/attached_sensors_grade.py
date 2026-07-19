#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Off-box grader for ``attached_sensors_live_proof.py`` evidence.

The live harness is stdlib-only (the render host has no numpy/PIL); every
claim about PIXELS is graded here, on the operator box, with the CANONICAL
decoder — ``tritium_lib.perception.depth_codec`` — against the geometry the
harness wrote into ``geometry.json`` and the FABRIC poses the sim itself
reported.  Nothing in this file trusts a log line: every number is recomputed
from the bytes on disk.

Checks:
  * depth16 is DEPTH, not a picture of depth: lossless PNG magic, decoded
    METRES match the authored scene — wall band ~9.9 m, body band at the
    ACTUAL landed pose, sky no-return, value count beyond any 8-bit hop,
    bit-exact round-trip through the lib codec;
  * RGB carries Newton's verdicts: the red cube's centroid sits where the
    projected FABRIC pose says, vanishes when HIDDEN, returns when SHOWN;
    the yellow ball's centroid moves between the two MOVING captures;
  * stereo right is a real second eye: body disparity matches
    fx*B*(1/Z - 1/D) for the converged pair;
  * LiDAR: the beam AT THE BODY CORNER'S OWN BEARING (computed from the
    fabric pose) reads the corner range when shown, and opens to >20 m when
    hidden — a per-bearing check no stale sweep can pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def load_jpeg(path: str) -> np.ndarray:
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except ImportError:
        import cv2
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable jpeg: {path}")
        return bgr[:, :, ::-1]


def color_centroid(rgb: np.ndarray, kind: str) -> tuple | None:
    """(u, v, count) of a colour-keyed blob, or None below the speckle floor."""
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    if kind == "red":
        mask = (r > 90) & (r > g * 1.6) & (r > b * 1.6)
    elif kind == "yellow":
        mask = (r > 120) & (g > 100) & (b * 2 < r + g)
    elif kind == "cyan":
        mask = (b > 100) & (g > 100) & (r * 2 < g + b)
    else:
        raise ValueError(kind)
    n = int(mask.sum())
    if n < 40:
        return None
    vs, us = np.nonzero(mask)
    return float(us.mean()), float(vs.mean()), n


class Projector:
    """The left camera's pinhole (looking +X, Z-up, +y is screen-left)."""

    def __init__(self, eye, fx, cx, cy):
        self.eye, self.fx, self.cx, self.cy = list(eye), fx, cx, cy

    def project(self, pt) -> tuple:
        z = pt[0] - self.eye[0]
        u = self.cx - self.fx * (pt[1] - self.eye[1]) / z
        v = self.cy - self.fx * (pt[2] - self.eye[2]) / z
        return u, v, z

    def front_band(self, center, half: float, shrink: float = 0.30) -> tuple:
        """Inner (rows, cols) slices of an axis-aligned cube's front face."""
        front_x = center[0] - half
        us, vs = [], []
        for dy in (-half, half):
            for dz in (-half, half):
                u, v, _ = self.project((front_x, center[1] + dy, center[2] + dz))
                us.append(u)
                vs.append(v)
        u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
        mu, mv = (u1 - u0) * shrink, (v1 - v0) * shrink
        return (slice(int(v0 + mv), int(v1 - mv)),
                slice(int(u0 + mu), int(u1 - mu)))


def median_band(depth: np.ndarray, band: tuple) -> float:
    vals = depth[band]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def nearest_box_point_2d(lidar_xy, center_xy, half: float) -> tuple:
    """(distance, bearing_rad) from the lidar to the box's nearest point."""
    px = min(max(lidar_xy[0], center_xy[0] - half), center_xy[0] + half)
    py = min(max(lidar_xy[1], center_xy[1] - half), center_xy[1] + half)
    dx, dy = px - lidar_xy[0], py - lidar_xy[1]
    return math.hypot(dx, dy), math.atan2(dy, dx)


def range_at_bearing(scan: dict, bearing_rad: float, halfwidth_beams: int = 3) -> float:
    """Min range in a small window around a bearing."""
    ranges = scan["ranges"]
    n = len(ranges)
    idx = round((bearing_rad - scan["angle_min"]) / scan["angle_increment"])
    lo = max(0, idx - halfwidth_beams)
    hi = min(n, idx + halfwidth_beams + 1)
    return min(ranges[lo:hi]) if hi > lo else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--lib-src", required=True,
                    help="tritium-lib/src (canonical depth codec)")
    args = ap.parse_args()
    ev = args.evidence

    sys.path.insert(0, os.path.abspath(args.lib_src))
    from tritium_lib.perception.depth_codec import (  # noqa: E402
        decode_depth16_png, encode_depth16_png)

    def jload(name):
        with open(os.path.join(ev, name)) as f:
            return json.load(f)

    geom = jload("geometry.json")
    intr = jload("intrinsics.json")
    live = jload("results_live.json")
    fx, cx, cy = intr["fx"], intr["cx"], intr["cy"]
    proj = Projector(geom["cam_eye"], fx, cx, cy)
    half = geom["body_half"]
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    def body_pose(phase: str):
        key = {"rest": "pose_rest", "moving_1": "pose_moving_1",
               "moving_2": "pose_moving_2", "released": "pose_released",
               "hidden": "pose_hidden", "shown": "pose_shown"}[phase]
        p = (live.get(key) or {}).get("body")
        return p if isinstance(p, list) else None

    phases = [p for p in ("rest", "moving_1", "moving_2", "released",
                          "hidden", "shown")
              if os.path.exists(os.path.join(ev, f"{p}_main.jpg"))]
    print(f"grading phases: {phases}\n")

    # ---- depth16: metric, lossless, alive --------------------------------- #
    print("— depth16 (canonical tritium_lib decode) —")
    wall_d = geom["wall_planar_depth_m"]
    # Wall sample: y in [2.2, 2.9] (clear of body, cyan cube and ball),
    # z in [1.5, 2.5].
    wall_band = (slice(int(cy - fx * 1.5 / wall_d), int(cy - fx * 0.5 / wall_d)),
                 slice(int(cx - fx * 2.9 / wall_d), int(cx - fx * 2.2 / wall_d)))
    sky_band = (slice(5, 40), slice(int(cx + fx * 3.2 / wall_d), None))
    for phase in phases:
        path = os.path.join(ev, f"{phase}_depth16.png")
        if not os.path.exists(path):
            check(f"depth16 {phase} exists", False, "file missing")
            continue
        blob = open(path, "rb").read()
        check(f"depth16 {phase} PNG magic", blob[:8].hex() == "89504e470d0a1a0a",
              blob[:8].hex())
        depth = decode_depth16_png(blob)
        finite = depth[np.isfinite(depth)]
        distinct = len(np.unique(np.round(finite * 1000).astype(np.int64)))
        check(f"depth16 {phase} >256 distinct mm values (not an 8-bit hop)",
              distinct > 256, f"{distinct} distinct")
        rt = decode_depth16_png(encode_depth16_png(np.nan_to_num(depth, nan=0.0)))
        same = np.array_equal(np.isnan(depth), np.isnan(rt)) and np.allclose(
            np.nan_to_num(depth, nan=-1), np.nan_to_num(rt, nan=-1), atol=0.0)
        check(f"depth16 {phase} bit-exact on the mm grid", same,
              "decode->encode->decode identical")
        wm = median_band(depth, wall_band)
        check(f"depth16 {phase} wall band ~{wall_d} m", abs(wm - wall_d) < 0.15,
              f"median {wm:.3f} m")
        sky = depth[sky_band]
        holes = float(np.mean(~np.isfinite(sky) | (sky >= 65.0)))
        check(f"depth16 {phase} sky is no-return/ceiling", holes > 0.9,
              f"{holes * 100:.0f}% holes")
        pose = body_pose(phase)
        if pose and phase != "hidden":
            band = proj.front_band(pose, half)
            bm = median_band(depth, band)
            expect = pose[0] - half - geom["cam_eye"][0]
            check(f"depth16 {phase} body band ~{expect:.2f} m (fabric pose)",
                  abs(bm - expect) < 0.20, f"median {bm:.3f} m")
        if phase == "hidden" and body_pose("rest"):
            band = proj.front_band(body_pose("rest"), half)
            bm = median_band(depth, band)
            check("depth16 hidden: body band opens to the far scene",
                  bm > 8.0, f"median {bm:.3f} m (body was at 5.7)")
        if phase == "released":
            kin = (live.get("pose_released") or {}).get("kin_cube")
            if isinstance(kin, list):
                band = proj.front_band(kin, geom["kin_half"])
                bm = median_band(depth, band)
                expect = kin[0] - geom["kin_half"] - geom["cam_eye"][0]
                check(f"depth16 released cyan band ~{expect:.2f} m",
                      abs(bm - expect) < 0.25, f"median {bm:.3f} m")

    # ---- RGB left: bodies are where Fabric says they are ------------------ #
    print("\n— RGB (left eye) —")
    cents: dict = {}
    for phase in phases:
        rgb = load_jpeg(os.path.join(ev, f"{phase}_main.jpg"))
        cents[phase] = color_centroid(rgb, "red")
        pose = body_pose(phase)
        if phase == "hidden":
            check("rgb hidden: red cube gone", cents[phase] is None,
                  f"centroid={cents[phase]}")
            continue
        if pose and cents[phase]:
            u, v, n = cents[phase]
            pu, pv, _ = proj.project((pose[0] - half, pose[1], pose[2]))
            check(f"rgb {phase} red centroid at projected fabric pose",
                  abs(u - pu) < 25 and abs(v - pv) < 30,
                  f"measured ({u:.0f},{v:.0f}) vs projected ({pu:.0f},{pv:.0f}), {n}px")
        else:
            check(f"rgb {phase} red cube visible", cents[phase] is not None,
                  f"centroid={cents[phase]} pose={pose}")
    if "released" in phases:
        kin = (live.get("pose_released") or {}).get("kin_cube")
        rgb = load_jpeg(os.path.join(ev, "released_main.jpg"))
        c = color_centroid(rgb, "cyan")
        if isinstance(kin, list) and c:
            pu, pv, _ = proj.project((kin[0] - geom["kin_half"], kin[1], kin[2]))
            check("rgb released: cyan centroid at landed pose",
                  abs(c[0] - pu) < 25 and abs(c[1] - pv) < 30,
                  f"measured ({c[0]:.0f},{c[1]:.0f}) vs projected ({pu:.0f},{pv:.0f})")
    # The bouncing ball between the MOVING captures.
    if "moving_1" in phases and "moving_2" in phases:
        b1 = (live.get("pose_moving_1") or {}).get("ball")
        b2 = (live.get("pose_moving_2") or {}).get("ball")
        c1 = color_centroid(load_jpeg(os.path.join(ev, "moving_1_main.jpg")), "yellow")
        c2 = color_centroid(load_jpeg(os.path.join(ev, "moving_2_main.jpg")), "yellow")
        if isinstance(b1, list) and isinstance(b2, list):
            dpose = math.dist(b1, b2)
            if dpose > 0.05:
                moved_px = (math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                            if (c1 and c2) else None)
                check("rgb moving: ball centroid moved with the sim",
                      moved_px is not None and moved_px > 3.0,
                      f"pose moved {dpose:.2f} m; centroid moved "
                      f"{moved_px if moved_px is None else round(moved_px, 1)}px "
                      f"({c1} -> {c2})")
            else:
                check("rgb moving: ball had settled (informational)",
                      True, f"pose delta {dpose:.3f} m — nothing to move")

    # ---- stereo right: a real second eye ---------------------------------- #
    print("\n— stereo right —")
    B, D = geom["baseline_m"], geom["converge_dist_m"]
    for phase in ("rest", "shown"):
        if phase not in phases:
            continue
        lp = os.path.join(ev, f"{phase}_main.jpg")
        rp = os.path.join(ev, f"{phase}_right.jpg")
        if not os.path.exists(rp):
            check(f"stereo {phase} right frame exists", False, "missing")
            continue
        lb, rb = open(lp, "rb").read(), open(rp, "rb").read()
        check(f"stereo {phase} right differs from left (bytes)", lb != rb,
              f"{len(lb)} vs {len(rb)} bytes")
        cl = color_centroid(load_jpeg(lp), "red")
        cr = color_centroid(load_jpeg(rp), "red")
        pose = body_pose(phase)
        if cl and cr and pose:
            z = pose[0] - half - geom["cam_eye"][0]
            pred = fx * B * (1.0 / z - 1.0 / D)
            meas = cl[0] - cr[0]
            check(f"stereo {phase} disparity ~ fx*B*(1/Z-1/D)",
                  abs(meas - pred) < max(5.0, 0.5 * pred),
                  f"measured {meas:.1f}px vs predicted {pred:.1f}px (Z={z:.2f})")
        else:
            check(f"stereo {phase} body found in both eyes",
                  bool(cl and cr), f"left={cl} right={cr}")

    # ---- LiDAR: per-bearing, against the fabric pose ---------------------- #
    print("\n— lidar /scan —")
    lx, ly = geom["lidar_pos"][0], geom["lidar_pos"][1]
    rest_pose = body_pose("rest")
    for phase in phases:
        spath = os.path.join(ev, f"{phase}_scan.json")
        if not os.path.exists(spath):
            check(f"lidar {phase} scan exists", False, "missing")
            continue
        scan = json.load(open(spath))
        if not scan.get("ranges"):
            check(f"lidar {phase} has beams", False, "empty ranges")
            continue
        check(f"lidar {phase} never_returned is False",
              scan.get("never_returned") is False,
              str(scan.get("never_returned")))
        pose = body_pose(phase) or rest_pose
        if pose is None:
            continue
        dist, bearing = nearest_box_point_2d((lx, ly), (pose[0], pose[1]), half)
        got = range_at_bearing(scan, bearing)
        if phase == "hidden":
            check("lidar hidden: body's bearing opens up",
                  got > dist + 1.5,
                  f"{got:.2f} m at {math.degrees(bearing):.0f}deg "
                  f"(body corner would be {dist:.2f} m)")
        else:
            check(f"lidar {phase} corner return at the body's own bearing",
                  abs(got - dist) < 0.25,
                  f"{got:.2f} m at {math.degrees(bearing):.0f}deg vs "
                  f"corner {dist:.2f} m")
        if phase == "released":
            kin = (live.get("pose_released") or {}).get("kin_cube")
            if isinstance(kin, list):
                kd, kb = nearest_box_point_2d((lx, ly), (kin[0], kin[1]),
                                              geom["kin_half"])
                kg = range_at_bearing(scan, kb)
                check("lidar released: cyan cube appears in the sweep",
                      abs(kg - kd) < 0.3,
                      f"{kg:.2f} m at {math.degrees(kb):.0f}deg vs {kd:.2f} m")

    # ---- verdict ---------------------------------------------------------- #
    passed = sum(1 for c in checks if c["pass"])
    out = {"checks": checks, "passed": passed, "total": len(checks)}
    with open(os.path.join(ev, "results_graded.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGRADE: {passed}/{len(checks)} checks pass "
          f"({'ALL GREEN' if passed == len(checks) else 'FAILURES ABOVE'})")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
