# tests — Meshtastic addon test suite

pytest coverage for the [Meshtastic LoRa mesh addon](../README.md): **524 tests
across 10 files** (plus a shared `conftest.py`). The suite runs **without any
Meshtastic radio** — connection, node tracking, and message bridging are
exercised against fakes and the non-blocking startup path, so CI stays green
with nothing plugged in.

**Deps:** `pytest` plus an editable `tritium-lib` install
(`pip install -e ../../../tritium-lib`).

## Run

```bash
cd tritium-addons
python3 -m pytest meshtastic/tests/ -v
```

## Key files

| File | Covers |
|------|--------|
| `test_meshtastic_addon.py` | `MeshtasticAddon` lifecycle + registration |
| `test_connection.py` | Serial / TCP / BLE / MQTT connection manager |
| `test_nonblocking_startup.py` | Auto-connect must not block server startup |
| `test_device_manager.py` | Device config + firmware management |
| `test_config_roundtrip.py` | Config write → read fidelity |
| `test_multi_radio.py` | Multiple radios at once |
| `test_meshtastic_mqtt_bridge.py` | Remote-runner MQTT discovery |
| `test_geojson.py` | Node map-layer GeoJSON output |
| `test_api_endpoints.py` | FastAPI routes |
| `test_full_sweep.py` | End-to-end packet → node → target sweep |
| `conftest.py` | Shared fixtures (fake radios, sample packets) |

## Related

- [Meshtastic addon overview](../README.md) · [Backend package](../meshtastic_addon/)
