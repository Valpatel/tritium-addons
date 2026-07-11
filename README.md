# tritium-addons — Sensor Integrations

Drop-in sensor addons for the [Tritium](https://github.com/Valpatel/tritium) system. Each addon works three ways:

> **New here?** Start with the parent repo's [`docs/QUICKSTART.md`](https://github.com/Valpatel/tritium/blob/main/docs/QUICKSTART.md). Public SDK for closed-source cognition: [`tritium-sc/docs/EMBODIMENTS.md`](https://github.com/Valpatel/tritium-sc/blob/main/docs/EMBODIMENTS.md). Glossary: [`docs/GLOSSARY.md`](https://github.com/Valpatel/tritium/blob/main/docs/GLOSSARY.md).

```mermaid
flowchart LR
    subgraph Addon
        backend[Python backend]
        frontend[JS frontend]
        runner[Headless runner]
    end

    backend -->|registers routes + panels| sc[Command Center :8000]
    frontend -->|panels in UI menu| browser[Browser]
    runner -->|publishes detections| mqtt[MQTT broker]
    mqtt --> sc

    style Addon fill:#0e1a2b,stroke:#fcee0a,color:#fcee0a
    style sc fill:#0e1a2b,stroke:#ff2a6d,color:#ff2a6d
```

1. **SC plugin** — panels in the Command Center UI, targets on the tactical map
2. **Standalone app** — full-screen at `/addon/{id}/`, PWA support for tablets
3. **Headless runner** — standalone on a Raspberry Pi, publishes to MQTT

## Addon status

| Addon | Status | Hardware | What it does |
|-------|--------|----------|-------------|
| [hackrf/](hackrf/) | **Functional** | HackRF One | Spectrum analysis, FM radio, ADS-B aircraft, TPMS vehicles, ISM bands |
| [meshtastic/](meshtastic/) | **Functional** | Any Meshtastic radio | LoRa mesh — GPS tracking, messaging, device config |
| [isaac_sim/](isaac_sim/) | **Connector** (in-progress) | RTX GPU render host | NVIDIA Isaac Sim digital twins — Scene3D→USD, MJPEG cameras, robot-body TCP seam (see [DEVELOPER-GUIDE.md §10](DEVELOPER-GUIDE.md)) |
| discord/ | Stub | — | Discord bot (scaffolding only) |
| telegram/ | Stub | — | Telegram bot (scaffolding only) |
| irc/ | Stub | — | IRC bridge (scaffolding only) |
| matrix/ | Stub | — | Matrix chat (scaffolding only) |
| signal_bridge/ | Stub | — | Signal messenger (scaffolding only; dir renamed — `signal/` shadowed the Python stdlib `signal` module) |
| slack/ | Stub | — | Slack integration (scaffolding only) |
| email_bridge/ | Stub | — | Email notifications (scaffolding only; dir renamed — `email/` shadowed the Python stdlib `email` package) |
| sms_gateway/ | Stub | — | SMS gateway (scaffolding only) |
| satellite/ | Stub | — | Satellite tracking (scaffolding only) |
| webhooks/ | Stub | — | Generic webhooks (scaffolding only) |

> Previously listed `wifi_csi/` as an empty placeholder; deleted in W203 because it was a lying manifest. See `tritium-sc/docs/technical-brief-ruview-csi-analysis.md` for the planned RuView-based implementation.

The stubs share the same pattern: a plugin class that logs "started (stub)" and a `send_message()` that returns `True` without connecting to anything. They exist as scaffolding for future implementation.

## Verified addon index (public + private catalog)

[`addon-index.json`](addon-index.json) is the catalog of **all known
addons across repos** — public ones here, plus advanced/premium ones in
private repos (e.g. `tritium-addon-priv`). Each entry carries `name`,
short `description`, `license`, `owner`, source `repo`, `status`, and a
`verified` flag. The Command Center reads it to present a searchable
verified-addon list and **grays out** any addon whose source repo isn't
installed — so you can see what's available and where to get it without
the code being present (Blender-style).

| Addon | Repo | License | Owner | Status |
|-------|------|---------|-------|--------|
| nav-pro | tritium-addon-priv (private) | Proprietary | Valpatel Software LLC | functional |
| hackrf | tritium-addons | AGPL-3.0 | Valpatel Software LLC | functional |
| meshtastic | tritium-addons | AGPL-3.0 | Valpatel Software LLC | functional |
| isaac-sim | tritium-addons | AGPL-3.0 | Valpatel Software LLC | in-progress |
| (10 comms stubs) | tritium-addons | AGPL-3.0 | Valpatel Software LLC | stub |

The index is **extensible**: add a `repos[]` entry to advertise a
third-party addon source, then list its addons. A private addon may be
**promoted to public** by moving its directory into this repo, switching
its manifest `license` to `AGPL-3.0`, and updating its index entry's
`repo`/`license` — the addon code already targets only the open SDK, so
no code change is needed.

## Quick start

```bash
# Addons are auto-discovered by the Command Center.
# Clone the parent repo with submodules:
git clone --recurse-submodules git@github.com:Valpatel/tritium.git

# Test a specific addon:
cd tritium-addons
python3 -m pytest hackrf/tests/ -v
python3 -m pytest meshtastic/tests/ -v
```

## Creating a new addon

**Follow the [Addon Developer Guide](DEVELOPER-GUIDE.md)** — the
canonical, code-grounded walkthrough (manifest, entry-point class, the
loader lifecycle, getting targets on the map, headless runner mode,
publishing). The layout it expects:

```
my-addon/
├── my_addon/
│   ├── __init__.py          # MyAddon(SensorAddon) — entry point
│   ├── runner.py            # MyRunner(BaseRunner) — headless mode
│   ├── router.py            # FastAPI routes
│   └── mqtt_bridge.py       # MQTT discovery for remote runners
├── frontend/
│   └── my-addon.js          # UI panel
├── tests/
│   └── test_my_addon.py
└── tritium_addon.toml        # Manifest (metadata, routes, capabilities)
```

The addon SDK lives in `tritium-lib` (`tritium_lib.sdk`). Full walkthrough: [DEVELOPER-GUIDE.md](DEVELOPER-GUIDE.md). Manifest quick-reference and repo conventions: [CLAUDE.md](CLAUDE.md).

## How it grows

Each sensor type or data source becomes its own addon. The addon doesn't need to know about other addons — it just publishes detections to MQTT and/or registers with the Command Center's event bus. The target tracker and fusion engine handle the rest.

This means ADS-B aircraft tracking, TPMS tire pressure monitoring, LoRa mesh mapping, and spectrum analysis all work the same way: detect → publish → track → fuse → display.

---

AGPL-3.0 | Copyright 2026 Valpatel Software LLC
