# Tritium Addons

Drop-in sensor integrations for the [Tritium](https://github.com/Valpatel/tritium) tactical operating platform.

## Addons

| Addon | Hardware | What It Does |
|-------|----------|--------------|
| **[hackrf](hackrf/)** | HackRF One | Spectrum analysis, FM radio, ADS-B aircraft, TPMS vehicles, ISM band monitoring (rtl_433) |
| **[meshtastic](meshtastic/)** | Any Meshtastic radio | LoRa mesh network — 250+ node tracking, GPS positions, messaging, device config |

Each addon works three ways:
- **Command Center plugin** — panels in the tactical UI, targets on the map
- **Standalone app** — full-screen at `/addon/{id}/` with PWA support
- **Headless runner** — standalone on a Raspberry Pi, publishes to MQTT

## Quick Start

```bash
# Addons are auto-discovered by the Command Center
# Just clone the parent repo with submodules:
git clone --recurse-submodules git@github.com:Valpatel/tritium.git

# Or add this repo as a submodule to an existing Tritium install:
git submodule add -b dev git@github.com:Valpatel/tritium-addons.git
```

## Creating an Addon

See [CLAUDE.md](CLAUDE.md) for the full development guide, manifest format, and conventions.

## License

AGPL-3.0 — See [LICENSE](LICENSE) for details.

Created by Matthew Valancy | Copyright 2026 Valpatel Software LLC
