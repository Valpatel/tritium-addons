# decoders — HackRF signal decoders

Signal decoders for the [HackRF backend](../README.md). Each turns raw IQ or a
subprocess feed from the HackRF One into structured detections (aircraft, tire
sensors, ISM devices, radio stations) that become TrackedTargets and map layers.

**Status:** Functional. The pure-Python decoders (ADS-B, TPMS, ISM, FM) run
without extra tools; `rtl433_wrapper.py` additionally needs the external
`rtl_433` binary for its 200+ device protocols.

## Key modules

| File | Decodes | Notes |
|------|---------|-------|
| `__init__.py` | — | Exports all decoders |
| `adsb.py` | Aircraft (1090 MHz) | Preamble detect, CRC-24, CPR position, velocity |
| `tpms.py` | Tire pressure sensors (315 / 433 MHz) | OOK envelope detection, sensor ID extraction |
| `ism_monitor.py` | ISM-band devices (315 / 433 / 868 / 915 MHz) | Device fingerprinting |
| `fm_radio.py` | FM broadcast | Demodulation + US station lookup table |
| `rtl433_wrapper.py` | 200+ ISM protocols | Wraps the external `rtl_433` subprocess |

## Related

- [hackrf_addon backend](../README.md) · [HackRF addon overview](../../README.md)
- [Tests](../../tests/) — `test_decoders.py` covers this package
