# tests — HackRF addon test suite

pytest coverage for the [HackRF One SDR addon](../README.md): **333 tests across
13 files**. The suite runs **without any HackRF hardware** — device detection,
sweeps, and decoders are exercised against fixtures and the degraded-mode paths,
so CI stays green on a machine with no radio attached.

**Deps:** `pytest` plus an editable `tritium-lib` install
(`pip install -e ../../../tritium-lib`).

## Run

```bash
cd tritium-addons
python3 -m pytest hackrf/tests/ -v
```

## Key files

| File | Covers |
|------|--------|
| `test_device.py` | Control-plane device detection, firmware, clock/antenna |
| `test_sdr_device.py` | `HackRFSDRDevice` — the `tritium_lib.sdr.SDRDevice` ABC adapter |
| `test_spectrum.py` | `hackrf_sweep` wrapper + CSV parsing |
| `test_decoders.py` | ADS-B / TPMS / ISM / FM / rtl_433 decoders |
| `test_signal_db.py` | In-memory ring buffer + peak detection |
| `test_continuous_scan.py` | 24/7 band scanner |
| `test_radio_lock.py` | Single-radio mutual exclusion |
| `test_hackrf_mqtt_bridge.py` | Remote-runner MQTT discovery |
| `test_multi_device.py` | Multiple HackRF units |
| `test_api_endpoints.py` · `test_router_connected_honest.py` | FastAPI routes |
| `test_geojson.py` | Map-layer GeoJSON output |
| `test_edge_cases.py` | Failure / boundary handling |

## Related

- [HackRF addon overview](../README.md) · [Backend package](../hackrf_addon/)
