# isaac_sim_addon — Isaac Sim connector package (Python)

The Python package for the [NVIDIA Isaac Sim connector addon](../README.md).

Unlike the sensor addons (hackrf, meshtastic), this package is **not an
in-process SC plugin**. Its payload is the Isaac-**side** runtime under
[`connectors/`](connectors/), which runs on the render host inside Isaac's own
Python 3.12. The top-level `__init__.py` imports nothing from Tritium and
nothing heavy, so it — and its `ADDON_ID = "isaac-sim"` — stay inspectable in CI
without a GPU.

The three **neutral contracts** across the boundary (nothing here is imported by
the SC process, nothing here imports Tritium internals) are:

- **`Scene3D` JSON** — `tritium_lib.geo.scene3d`, served at `GET /api/gis/scene3d`,
  materialized to USD by `connectors/usd_scene_builder.py`.
- **camera MJPEG / MQTT frames** — `connectors/camera_server.py` and
  `connectors/isaac_camera_bridge.py`, ingested by `tritium-sc` `camera_feeds`.
- **the robot-body TCP seam** — `connectors/isaac_quadruped_server.py`, the
  physics half of the `robot-template` brain/body protocol.

## Layout

| Path | What |
|------|------|
| `__init__.py` | `ADDON_ID` + package docstring; zero heavy / Tritium imports |
| [`connectors/`](connectors/) | Isaac-side runtime — USD builder, renderer, camera servers, robot body |

**Deps:** `isaacsim[all]==6.0.1.0` on the render host only (Python 3.12,
RTX-class GPU). The no-GPU code paths (`--validate`, `--selftest`,
`--source synthetic`) run under plain `python3`. See [`../tests/`](../tests/).

## Related

- [Isaac Sim addon overview](../README.md) — pipeline, run recipes, placement rationale
- [Connectors](connectors/) · [Examples](../examples/) · [Manifest](../tritium_addon.toml)
- [DEVELOPER-GUIDE.md §10](../../DEVELOPER-GUIDE.md) (parent repo)
