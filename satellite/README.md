# Satellite Bridge — comms addon

> Parent: [tritium-addons](../README.md) · Archetype: [COMMS-BRIDGES.md](../COMMS-BRIDGES.md)

**What it's for:** Beyond-line-of-sight comms via Iridium, Starlink, or Inmarsat.

**Status:** **Stub.** `SatellitePlugin.send_message()` logs "Would send" and returns `True` without connecting. It is loaded and `start()`ed by the dispatcher, but never actually invoked — the fan-out only calls plugins that define `send_message_sync`, and this one doesn't (see [COMMS-BRIDGES.md → D3](../COMMS-BRIDGES.md#reality-check)). No tests.

## Wire

- **Entry class:** `SatellitePlugin` in [`plugin.py`](plugin.py) — a bare, duck-typed
  comms-bridge plugin (no base class), *not* a `SensorAddon`. See
  [COMMS-BRIDGES.md](../COMMS-BRIDGES.md) for the shared contract.
- **Loaded by:** the `CommsDispatcher`
  (`tritium-sc/src/app/comms_dispatcher.py`) — it calls `configure()` →
  `start()` and fans out `notification:new` events above a severity threshold.
- **Routes:** [`routes.py`](routes.py) declares `/api/comms/satellite`
  (`/config`, `/status`, `/test`, `/send`) — **not currently mounted** by the
  running app (COMMS-BRIDGES.md D2).
- **Frontend:** [`frontend/`](frontend/) — a `comms-container` tab.

## Config keys (`SatellitePlugin._config`)

| Key | Meaning |
|-----|---------|
| `provider` | backend provider (e.g. twilio / modem / iridium) |
| `modem_port` | serial port of the satellite modem |
| `baud_rate` | modem serial baud rate |
| `bridge_alerts` | relay Tritium alerts (on/off or filter) |
| `enabled` | master switch — real I/O only fires when true |

## To make it real

Drive the modem's AT/serial interface or the provider's data API. Then **add a synchronous `send_message_sync(text, payload)`** — without it the dispatcher loads and `start()`s the plugin but never calls it. Follow [COMMS-BRIDGES.md → Making a stub real](../COMMS-BRIDGES.md#making-a-stub-real).

---

AGPL-3.0 | Copyright 2026 Valpatel Software LLC
