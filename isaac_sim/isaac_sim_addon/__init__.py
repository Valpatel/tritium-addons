# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""NVIDIA Isaac Sim connector addon.

Optional, heavy-GPU integration that bridges Tritium to Isaac Sim. All
Isaac-side runtime (imports ``isaacsim`` / ``pxr``) lives under
``connectors/`` and runs on a render host — never in the tritium-lib or
tritium-sc processes. The neutral contracts across the boundary are:

  * ``Scene3D`` geometry (``tritium_lib.geo.scene3d``) → USD via
    ``connectors.usd_scene_builder``.
  * a camera MJPEG stream (``connectors.camera_server``) that tritium-sc
    ingests like any IP camera.
  * the robot-body TCP seam (to migrate from tritium-sc/examples/isaac-bridge).

This package intentionally imports nothing from tritium and nothing heavy at
module import time, so it stays inspectable in CI without a GPU.
"""

ADDON_ID = "isaac-sim"
