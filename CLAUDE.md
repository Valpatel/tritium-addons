# Tritium Addons — Plugin Ecosystem

> **North Star:** *Build a fun simulator for the purpose of testing and validating the software stack that drives independent machines.*

Every addon is dual-use: it lights up real-world capability (a sensor, a comms channel, a data feed) AND becomes a playable element in the simulator (a layer, a panel, a mission trigger, a narrated event). If a new addon doesn't have both surfaces, it's incomplete.

**What qualifies as an addon (the bin):** any OPTIONAL connector to an EXTERNAL
system, tool, sensor, or simulator that the core runs fine without and that
brings heavy or specialized dependencies. That includes **simulator / tool
connectors** — e.g. **Isaac Sim lives here as `isaac_sim`** (USD builder,
headless renderer, camera/robot connectors that `import isaacsim`/`pxr`, plus
its own examples). Heavy deps are quarantined inside the addon so `tritium-lib`
stays framework-free and `tritium-sc` stays a web app. If the code runs on the
robot's own compute it is **not** an addon — that is `tritium-edge/ros2`. See
the copper-roof placement rule in [`../CLAUDE.md`](../CLAUDE.md) →
`docs/ARCHITECTURE.md` (parent repo).

Tritium addons (this repo) are **open-source under AGPL-3.0** and define the public extension points. Closed-source intelligence plugins — including the Graphlings cognition stack which drives the *mind* of the machines — load against the same SDK without forking it. See top-level [../CLAUDE.md](../CLAUDE.md) and `project_north_star.md`.

Addons for the Tritium tactical operating platform. Two addons are functional (hackrf, meshtastic) with full backend, frontend, runner, and tests. Ten comms addons (discord, telegram, irc, matrix, signal, slack, email, sms_gateway, satellite, webhooks) are stubs.

> **wifi_csi removed pending real implementation.** The previous `wifi_csi/` directory shipped a manifest declaring routes, frontend tabs, and an edge HAL but only contained a 4-line `__init__.py`. Deleted in W203 along with the matching SC plugin (`tritium-sc/plugins/wifi_csi/`) and frontend tab (`sensing-wifi-csi-tab.js`). Reintroduce only when there is a working edge `hal_wifi_csi`, real CSI capture, and a backend that actually serves `/api/wifi-csi/status`. See `docs/technical-brief-ruview-csi-analysis.md` (W199 RuView research note) for the planned architecture.

**Parent context:** See [../CLAUDE.md](../CLAUDE.md) for full system architecture and conventions.

## Git Conventions

- **No Co-Authored-By lines in commits** — NEVER add these
- Remote: `git@github.com:Valpatel/tritium-addons.git`
- Copyright: Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
- Branch: `dev` (matches all other submodules)

## Architecture

Each addon is a folder at the repo root:

```
tritium-addons/
├── hackrf/                     # HackRF One SDR
│   ├── hackrf_addon/           # Python backend module
│   │   ├── __init__.py         # HackRFAddon(SensorAddon) — SC plugin
│   │   ├── runner.py           # HackRFRunner(BaseRunner) — standalone Pi mode
│   │   ├── mqtt_bridge.py      # Auto-discovers remote runners via MQTT
│   │   ├── router.py           # FastAPI API routes + GeoJSON endpoints
│   │   ├── device.py           # Device detection (multi-device)
│   │   ├── spectrum.py         # Spectrum analyzer (hackrf_sweep wrapper)
│   │   ├── fm_player.py        # FM radio demodulation + playback
│   │   ├── signal_db.py        # In-memory signal ring buffer
│   │   ├── data_store.py       # Persistent SQLite store
│   │   ├── radio_lock.py       # Mutual exclusion for radio operations
│   │   ├── continuous_scan.py   # 24/7 band scanning
│   │   └── decoders/           # Signal decoders (FM, TPMS, ISM, ADS-B, rtl_433)
│   ├── frontend/               # Vanilla JS UI panels
│   │   └── hackrf.js           # 7-tab panel (Radio, Spectrum, Signals, etc.)
│   ├── tests/                  # pytest tests (314 tests)
│   ├── tritium_addon.toml      # Addon manifest
│   ├── setup.sh                # Install dependencies (hackrf tools, rtl_433, etc.)
│   └── docs/                   # Addon-specific docs
├── meshtastic/                 # Meshtastic LoRa Mesh
│   ├── meshtastic_addon/       # Python backend module
│   │   ├── __init__.py         # MeshtasticAddon(SensorAddon) — SC plugin
│   │   ├── runner.py           # MeshtasticRunner(BaseRunner) — standalone Pi mode
│   │   ├── mqtt_bridge.py      # Auto-discovers remote radios via MQTT
│   │   ├── router.py           # FastAPI API routes + GeoJSON endpoints
│   │   ├── connection.py       # Serial/BLE/TCP/MQTT connection manager
│   │   ├── node_manager.py     # Mesh node tracking (250+ nodes)
│   │   ├── device_manager.py   # Device config, firmware, control
│   │   ├── message_bridge.py   # Bidirectional mesh ↔ Tritium messaging
│   │   └── data_store.py       # Persistent SQLite store
│   ├── frontend/               # Vanilla JS UI panels
│   │   └── meshtastic.js       # 7-tab panel (Radio, Nodes, Messages, etc.)
│   ├── tests/                  # pytest tests (522 tests)
│   ├── tritium_addon.toml      # Addon manifest
│   └── docs/                   # Hard-won API notes
└── CLAUDE.md                   # This file
```

## How Addons Work

Each addon has three modes:

1. **SC Plugin** — Loaded by tritium-sc's AddonLoader, registers FastAPI routes, panels appear in WINDOWS menu, targets appear on tactical map
2. **Standalone App** — Full-screen at `/addon/{id}/`, works on tablets, supports PWA "Add to Home Screen"
3. **Runner** — Headless standalone mode for Raspberry Pi, publishes data to MQTT for remote operation

## Dependencies

Addons depend on:
- `tritium-lib` SDK (`tritium_lib.sdk`) — AddonBase, DeviceRegistry, manifest system
- `tritium-sc` at runtime — TargetTracker, EventBus, FastAPI app (injected via register())
- No direct imports from `tritium-sc` source code — all SC access is via the `app` context passed to `register()`

## Coding Conventions

- Python 3.12+, 4-space indentation
- Vanilla JavaScript only (no frameworks)
- Cyberpunk aesthetic: cyan #00f0ff, magenta #ff2a6d, green #05ffa1, yellow #fcee0a
- Background: #0a0a0f, surfaces #0e0e14/#12121a
- Type hints on all public Python functions
- Functional addons must have tests — hackrf (314 tests) and meshtastic (522 tests) are covered; comms stubs have none

## Testing

```bash
# Test a specific addon
cd tritium-addons
python3 -m pytest hackrf/tests/ -v
python3 -m pytest meshtastic/tests/ -v

# Test all addons
python3 -m pytest */tests/ -v
```

Tests require `tritium-lib` installed (`pip install -e ../tritium-lib`).

## Creating a New Addon

1. Create a folder: `my-addon/`
2. Create manifest: `my-addon/tritium_addon.toml`
3. Create Python module: `my-addon/my_addon/__init__.py` with a class extending `SensorAddon` (or other type)
4. Create frontend: `my-addon/frontend/my-addon.js` exporting a PanelDef
5. Add tests: `my-addon/tests/`
6. The SC addon loader will auto-discover it on next restart

## Addon Manifest (tritium_addon.toml)

```toml
[addon]
id = "my-addon"
name = "My Custom Addon"
version = "1.0.0"
description = "What it does"
author = "Your Name"
license = "AGPL-3.0"

[addon.category]
window = "sensors"
tab_order = 10

[dependencies]
requires = []
python_packages = ["some-library>=1.0"]

[hardware]
devices = ["My Device"]
serial_vid_pid = ["1234:5678"]
auto_detect = true

[permissions]
serial = true
network = false
mqtt = true
storage = true

[backend]
module = "my_addon"
router_prefix = "/api/addons/my-addon"

[[frontend.panels]]
id = "my-addon"
title = "MY ADDON"
file = "my-addon.js"
```
