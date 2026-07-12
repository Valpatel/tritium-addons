# meshtastic_addon — Meshtastic LoRa mesh backend (Python)

The Python backend package for the [Meshtastic LoRa mesh addon](../README.md).
It integrates any **Meshtastic radio** (T-Beam, Heltec, RAK, T-LoRa, etc.) into
Tritium over four transports — USB serial, BLE, TCP/WiFi, or the public MQTT
broker — tracking 250+ mesh nodes with GPS as TrackedTargets, bridging text /
position / telemetry, and managing device configuration and firmware.

**Status:** Functional. The addon auto-detects serial radios by VID:PID and
connects non-blocking on startup; runs cleanly with no radio attached.

**Deps:** `tritium_lib.sdk`, the `meshtastic` Python library, and optionally
`bleak` (BLE). See the [addon README](../README.md#hardware) for transport modes.

## Key modules

| File | Role |
|------|------|
| `__init__.py` | `MeshtasticAddon(SensorAddon)` — entry point, multi-radio, non-blocking auto-connect |
| `connection.py` | Serial / TCP / BLE / MQTT connection manager (auto-detect, retries, DTR drain) |
| `node_manager.py` | Mesh nodes → Tritium targets, BFS hop estimation, position anchors |
| `device_manager.py` | Device config (LoRa params, channels, user), firmware |
| `message_bridge.py` | Bidirectional mesh ↔ Tritium messaging (text/position/telemetry/nodeinfo) |
| `ble_direct.py` | Direct `bleak` BLE path — bypasses meshtastic-lib BLE race conditions |
| `data_store.py` | Persistent SQLite store (nodes, positions, telemetry, messages, stats) |
| `mqtt_bridge.py` | Auto-discovers remote Meshtastic runners over MQTT |
| `router.py` | FastAPI routes + GeoJSON endpoints (`/api/addons/meshtastic`) |
| `runner.py` | `MeshtasticRunner(BaseRunner)` — standalone headless (Raspberry Pi) mode |

## Related

- [Meshtastic addon overview](../README.md) — mermaid pipeline, hardware, quick start
- [Hard-won library notes](../docs/MESHTASTIC-API-NOTES.md) · [Tests](../tests/)
- [tritium-addons top-level README](../../README.md) · [DEVELOPER-GUIDE.md](../../DEVELOPER-GUIDE.md)
