# hackrf_addon — HackRF One SDR backend (Python)

The Python backend package for the [HackRF One SDR addon](../README.md). It
integrates a **HackRF One** (1 MHz – 6 GHz software-defined radio) into Tritium:
spectrum sweeps, FM demodulation, and ADS-B / TPMS / ISM device decoding, all
surfaced as TrackedTargets and GeoJSON map layers.

**Status:** Functional. Runs in degraded mode (no live data) without hardware;
the full test suite still passes.

**Deps:** `tritium_lib.sdk` (AddonBase / runner / manifest), `tritium_lib.sdr`
(the `SDRDevice` ABC), `numpy`/`scipy`, and the external `hackrf_*` CLI tools
(`hackrf_info`, `hackrf_sweep`, `hackrf_transfer`) plus optional `rtl_433` /
SoapySDR. Install via [`../setup.sh`](../setup.sh).

## Key modules

| File | Role |
|------|------|
| `__init__.py` | `HackRFAddon(SensorAddon)` — plugin entry point, multi-device support, background poll loop |
| `device.py` | **Control plane** — detection (`hackrf_info`), firmware flash, clock/antenna/bias-tee; `detect()` returns a rich dict |
| `sdr_device.py` | **Data plane** — `HackRFSDRDevice(SDRDevice)`, the real-hardware adapter for the `tritium_lib.sdr` ABC |
| `spectrum.py` | Spectrum analyzer wrapping `hackrf_sweep` + CSV parser |
| `receiver.py` | IQ capture via `hackrf_transfer` (configurable gain / sample rate) |
| `fm_player.py` | Real-time FM demodulation + WAV streaming |
| `signal_db.py` | In-memory ring buffer (100K measurements), peak detection |
| `data_store.py` | Persistent SQLite store (signals, aircraft, TPMS, devices) |
| `continuous_scan.py` | 24/7 scanner cycling 9 frequency bands |
| `radio_lock.py` | Mutual exclusion — the radio serves one operation at a time |
| `mqtt_bridge.py` | Auto-discovers remote HackRF runners over MQTT |
| `router.py` | FastAPI routes + GeoJSON endpoints (`/api/addons/hackrf`) |
| `runner.py` | `HackRFRunner(BaseRunner)` — standalone headless (Raspberry Pi) mode |
| [`decoders/`](decoders/) | Signal decoders (FM, ADS-B, TPMS, ISM, rtl_433) |

The control-plane vs data-plane split (why there are two device classes) is
explained in the [addon README](../README.md#two-device-abstractions--control-plane-vs-data-plane).

## Related

- [HackRF addon overview](../README.md) — full file table, mermaid pipeline, hardware, runbook
- [Signal decoders](decoders/) · [Tests](../tests/)
- [tritium-addons top-level README](../../README.md) · [DEVELOPER-GUIDE.md](../../DEVELOPER-GUIDE.md)
