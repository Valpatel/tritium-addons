# tests — Isaac Sim connector no-GPU gates

pytest coverage for the [Isaac Sim connector](../README.md): **15 tests across 2
files**. Every test runs under plain `python3` with **no Isaac, no `pxr`, no
GPU** — proving the connectors stay inspectable and the neutral contracts hold
in CI, and enforcing the dependency-hygiene invariant (connectors import neither
`tritium` nor a heavy runtime at module load).

**Deps:** `pytest`, `numpy`, `cv2` (already in the Tritium env). No `isaacsim`.

## Run

```bash
cd tritium-addons
python3 -m pytest isaac_sim/tests/ -v
```

## Key files

| File | Covers |
|------|--------|
| `test_no_gpu.py` | `usd_scene_builder` validate/OBJ, `camera_server` synthetic source, `isaac_quadruped_server` gait integrator + footfalls + `--selftest`, and the "connectors never import tritium" guard |
| `test_isaac_camera_bridge.py` | The `camera_feeds` ingest contract — topic strings + JPEG encode/decode round-trip; a broker round-trip test that skips when no MQTT broker is reachable |

## Related

- [Isaac Sim addon overview](../README.md) · [Connectors](../isaac_sim_addon/connectors/)
